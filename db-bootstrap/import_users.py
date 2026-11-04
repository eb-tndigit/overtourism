# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, delete, insert, make_url, select, update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from bootstrap_db import (
    _DATABASE_SCHEMA,
    _USERS,
    _database_url,
    _ensure_sqlite_parent_dirs,
)

_VALID_ROLES = frozenset({"admin", "multieditor", "editor", "viewer"})
_USER_TERRITORIES = _DATABASE_SCHEMA.tables["user_territories"]


@dataclass(frozen=True)
class UserImportEntry:
    email: str
    role: str
    territories: tuple[str, ...] | None


@dataclass(frozen=True)
class ImportSummary:
    inserted: int
    updated: int
    unchanged: int


def _read_csv(csv_path: Path) -> list[UserImportEntry]:
    entries: list[UserImportEntry] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError("The CSV file is empty or has no header.")

        headers = [header.strip().casefold() for header in reader.fieldnames]
        if len(headers) != len(set(headers)):
            raise ValueError("The CSV header contains duplicate column names.")
        if not {"email", "role"}.issubset(headers):
            raise ValueError("The CSV header must contain 'email' and 'role'.")

        email_header = reader.fieldnames[headers.index("email")]
        role_header = reader.fieldnames[headers.index("role")]
        territories_header = (
            reader.fieldnames[headers.index("territories")]
            if "territories" in headers
            else None
        )
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"CSV line {line_number} has too many columns.")
            if all(value is None or not value.strip() for value in row.values()):
                continue

            email = (row.get(email_header) or "").strip().casefold()
            role = (row.get(role_header) or "").strip().casefold()
            if not email:
                raise ValueError(f"CSV line {line_number} has an empty email.")
            if role not in _VALID_ROLES:
                supported_roles = ", ".join(sorted(_VALID_ROLES))
                raise ValueError(
                    f"CSV line {line_number} has an invalid role '{role}'. "
                    f"Choose one of: {supported_roles}."
                )
            territories = (
                None
                if territories_header is None
                else tuple(
                    sorted(
                        {
                            territory.strip()
                            for territory in (row.get(territories_header) or "").split(
                                ","
                            )
                            if territory.strip()
                        }
                    )
                )
            )
            if role == "admin" and territories:
                raise ValueError(
                    f"CSV line {line_number} assigns territories to an admin."
                )
            entries.append(UserImportEntry(email, role, territories))

    return entries


def _upsert_user(connection: Connection, entry: UserImportEntry) -> str:
    user = (
        connection.execute(
            select(_USERS.c.user_id, _USERS.c.role).where(
                _USERS.c.identifier == entry.email
            )
        )
        .mappings()
        .first()
    )
    now = datetime.now(UTC)
    desired_territories = set(entry.territories or ())
    if user is None:
        if entry.role == "multieditor" and not desired_territories:
            raise ValueError(
                f"Cannot create multieditor '{entry.email}' without territories."
            )
        user_id = uuid4().hex
        connection.execute(
            insert(_USERS).values(
                user_id=user_id,
                identifier=entry.email,
                subject=None,
                role=entry.role,
                is_active=True,
                created_at=now,
                updated_at=now,
            )
        )
        if desired_territories:
            connection.execute(
                insert(_USER_TERRITORIES),
                [
                    {"user_id": user_id, "territory": territory}
                    for territory in sorted(desired_territories)
                ],
            )
        return "inserted"

    user_id = user["user_id"]
    existing_territories = set(
        connection.scalars(
            select(_USER_TERRITORIES.c.territory).where(
                _USER_TERRITORIES.c.user_id == user_id
            )
        )
    )
    if entry.territories is None:
        desired_territories = existing_territories
    if entry.role == "admin":
        desired_territories = set()
    if entry.role == "multieditor" and not desired_territories:
        raise ValueError(
            f"Cannot assign multieditor role to '{entry.email}' without territories."
        )

    role_changed = user["role"] != entry.role
    territories_changed = existing_territories != desired_territories
    if not role_changed and not territories_changed:
        return "unchanged"

    if role_changed:
        connection.execute(
            update(_USERS)
            .where(_USERS.c.user_id == user_id)
            .values(role=entry.role, updated_at=now)
        )
    if territories_changed:
        connection.execute(
            delete(_USER_TERRITORIES).where(_USER_TERRITORIES.c.user_id == user_id)
        )
        if desired_territories:
            connection.execute(
                insert(_USER_TERRITORIES),
                [
                    {"user_id": user_id, "territory": territory}
                    for territory in sorted(desired_territories)
                ],
            )
    if territories_changed and not role_changed:
        connection.execute(
            update(_USERS).where(_USERS.c.user_id == user_id).values(updated_at=now)
        )
    return "updated"


def import_users(
    csv_path: str | Path,
    database_url: str | None = None,
) -> ImportSummary:
    entries = _read_csv(Path(csv_path))
    database = make_url(database_url or _database_url())
    _ensure_sqlite_parent_dirs(database)
    engine = create_engine(database)
    try:
        counts = {"inserted": 0, "updated": 0, "unchanged": 0}
        with engine.begin() as connection:
            for entry in entries:
                result = _upsert_user(connection, entry)
                counts[result] += 1
        return ImportSummary(**counts)
    finally:
        engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Import users from a semicolon-delimited email;role CSV, "
            "optionally with comma-separated territories."
        )
    )
    parser.add_argument("csv_file", type=Path, help="Path to the email;role CSV file")
    args = parser.parse_args(argv)

    try:
        summary = import_users(args.csv_file)
    except (OSError, ValueError, SQLAlchemyError, csv.Error) as exc:
        print(f"User import failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Users imported: {summary.inserted} inserted, "
        f"{summary.updated} updated, {summary.unchanged} unchanged."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
