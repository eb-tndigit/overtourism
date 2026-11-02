# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    make_url,
    select,
    text,
)
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import SQLAlchemyError

# Keep this copied schema aligned with dt_manager SQLBase and backend AuthBase.
_DATABASE_SCHEMA = MetaData()

Table(
    "problems",
    _DATABASE_SCHEMA,
    Column("territory", Text),
    Column("problem_id", String, primary_key=True),
    Column("version", Integer, nullable=False, default=1),
    Column("name", Text),
    Column("description", Text),
    Column("created", String),
    Column("updated", String),
    Column("extras", JSON, nullable=False, default=dict),
    Index("ix_problems_territory_created", "territory", "created"),
)

Table(
    "sessions",
    _DATABASE_SCHEMA,
    Column("session_id", String, primary_key=True),
    Column("territory", Text, nullable=False),
    Column("owner_id", Text, nullable=True),
    Column("created", String),
    Column("updated", String),
    Column("metadata", JSON, nullable=False, default=dict),
    Column("active_scenario_id", String, nullable=True),
    Index(
        "ix_sessions_territory_owner_created",
        "territory",
        "owner_id",
        "created",
    ),
)

Table(
    "proposals",
    _DATABASE_SCHEMA,
    Column(
        "problem_id",
        String,
        ForeignKey("problems.problem_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("proposal_id", String, primary_key=True),
    Column("version", Integer, nullable=False, default=1),
    Column("name", Text),
    Column("description", Text),
    Column("status", String),
    Column("created", String),
    Column("updated", String),
    Column("extras", JSON, nullable=False, default=dict),
    Index("ix_proposals_problem_id_created", "problem_id", "created"),
)

Table(
    "scenarios",
    _DATABASE_SCHEMA,
    Column("scenario_id", String, primary_key=True),
    Column("territory", Text, nullable=False),
    Column(
        "session_id",
        String,
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=True,
    ),
    Column("version", Integer, nullable=False, default=1),
    Column("name", Text),
    Column("description", Text),
    Column("summary", Text),
    Column("created", String),
    Column("updated", String),
    Column("extras", JSON, nullable=False, default=dict),
    Column("param_overrides", JSON, nullable=False, default=dict),
    Index("ix_scenarios_territory_created", "territory", "created"),
)

Table(
    "proposal_scenario_relationship",
    _DATABASE_SCHEMA,
    Column(
        "proposal_id",
        String,
        ForeignKey("proposals.proposal_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "scenario_id",
        String,
        ForeignKey("scenarios.scenario_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Index("ix_proposal_scenario_relationship_scenario_id", "scenario_id"),
)

Table(
    "evaluations",
    _DATABASE_SCHEMA,
    Column("evaluation_id", String, primary_key=True),
    Column("version", Integer, nullable=False, default=1),
    Column("scenario_id", String, nullable=False),
    Column("session_id", String, nullable=True),
    Column("type", String, nullable=False),
    Column("state", String, nullable=False),
    Column("started", String),
    Column("finished", String),
    Column("result", LargeBinary, nullable=True),
    ForeignKeyConstraint(
        ["scenario_id"],
        ["scenarios.scenario_id"],
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["session_id"],
        ["sessions.session_id"],
        ondelete="CASCADE",
    ),
    Index("ix_evaluations_scenario_id_started", "scenario_id", "started"),
)

_USERS = Table(
    "users",
    _DATABASE_SCHEMA,
    Column("user_id", String, primary_key=True),
    Column("identifier", String, nullable=False, unique=True),
    Column("subject", String, nullable=True, unique=True),
    Column("role", String, nullable=False),
    Column("is_active", Boolean, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

Table(
    "user_territories",
    _DATABASE_SCHEMA,
    Column(
        "user_id",
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("territory", String, primary_key=True),
)


def _database_url() -> str:
    configured_url = os.getenv("OVERTOURISM_DATABASE")
    if configured_url:
        return configured_url

    database_path = (
        Path(__file__).resolve().parents[1]
        / "overtourism-backend"
        / "overtourism"
        / "overtourism"
        / "database"
        / "overtourism.sqlite"
    )
    return f"sqlite:///{database_path}"


def _ensure_sqlite_parent_dirs(database: URL) -> None:
    if database.get_backend_name() != "sqlite":
        return

    database_path = database.database
    if database_path in (None, "", ":memory:"):
        return

    Path(database_path).expanduser().resolve().parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def _lock_user_registry(connection, dialect_name: str) -> None:
    if dialect_name == "sqlite":
        connection.exec_driver_sql("BEGIN IMMEDIATE")
    elif dialect_name == "postgresql":
        connection.execute(text("LOCK TABLE users IN EXCLUSIVE MODE"))
    else:
        raise RuntimeError(f"First-admin bootstrap is unsupported for {dialect_name}")


def bootstrap_db(
    identifier: str | None = None,
    database_url: str | None = None,
) -> int:
    normalized_identifier = None
    if identifier is not None:
        normalized_identifier = identifier.strip().casefold()
        if not normalized_identifier:
            print("Provide a non-empty admin identifier.", file=sys.stderr)
            return 2

    try:
        database = make_url(database_url or _database_url())
        _ensure_sqlite_parent_dirs(database)
        engine = create_engine(database)
    except (OSError, SQLAlchemyError):
        print("The configured database could not be initialized.", file=sys.stderr)
        return 1

    try:
        _DATABASE_SCHEMA.create_all(engine)
        if normalized_identifier is None:
            print("Database schema initialized.")
            return 0

        with engine.connect() as connection:
            _lock_user_registry(connection, engine.dialect.name)
            existing_user = (
                connection.execute(
                    select(_USERS.c.role, _USERS.c.is_active).where(
                        _USERS.c.identifier == normalized_identifier
                    )
                )
                .mappings()
                .first()
            )
            if existing_user is not None:
                connection.rollback()
                if existing_user["role"] == "admin" and existing_user["is_active"]:
                    print(f"Admin already provisioned: {normalized_identifier}")
                    return 0
                print(
                    "The identifier already belongs to a non-admin or inactive user.",
                    file=sys.stderr,
                )
                return 1

            if connection.scalar(select(_USERS.c.user_id).limit(1)) is not None:
                connection.rollback()
                print(
                    "Cannot bootstrap the first admin because the user registry is not empty.",
                    file=sys.stderr,
                )
                return 1

            now = datetime.now(UTC)
            connection.execute(
                insert(_USERS).values(
                    user_id=uuid4().hex,
                    identifier=normalized_identifier,
                    subject=None,
                    role="admin",
                    is_active=True,
                    created_at=now,
                    updated_at=now,
                )
            )
            connection.commit()

        print(
            f"Pending admin created for {normalized_identifier}. "
            "Its first login must use a token with a matching configured identifier claim."
        )
        return 0
    except SQLAlchemyError:
        print("The database schema or user registry is unavailable.", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create the SQL schema and optionally provision the first admin."
    )
    parser.add_argument(
        "identifier",
        nargs="?",
        help="Identifier claim value for the first admin",
    )
    args = parser.parse_args(argv)
    return bootstrap_db(args.identifier)


if __name__ == "__main__":
    raise SystemExit(main())
