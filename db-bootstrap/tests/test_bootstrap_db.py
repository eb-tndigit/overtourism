# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from overtourism.backend.auth.identity.sql_repository import AuthBase
from overtourism.dt_manager.stores.classes.sql.orm import SQLBase
from sqlalchemy import MetaData, Table, create_engine, insert, inspect, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import UniqueConstraint

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "bootstrap_db.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("bootstrap_db", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
bootstrap_db = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(bootstrap_db)


def _sqlite_url(database_path: Path) -> str:
    return f"sqlite:///{database_path}"


def _insert_user(database_url: str, *, identifier: str, role: str) -> None:
    engine = create_engine(database_url)
    users = Table("users", MetaData(), autoload_with=engine)
    try:
        with engine.begin() as connection:
            connection.execute(
                insert(users).values(
                    user_id=f"{role}-id",
                    identifier=identifier,
                    subject=f"{role}-subject",
                    role=role,
                    is_active=True,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )
    finally:
        engine.dispose()


def test_standalone_schema_matches_backend_metadata() -> None:
    source_tables = SQLBase.metadata.tables | AuthBase.metadata.tables
    copied_tables = bootstrap_db._DATABASE_SCHEMA.tables
    assert set(copied_tables) == set(source_tables)

    engine = create_engine("sqlite:///:memory:")
    try:
        for table_name, source_table in source_tables.items():
            copied_table = copied_tables[table_name]
            assert list(copied_table.columns.keys()) == list(
                source_table.columns.keys()
            )
            for column_name, source_column in source_table.columns.items():
                copied_column = copied_table.columns[column_name]
                assert copied_column.nullable == source_column.nullable
                assert copied_column.primary_key == source_column.primary_key
                assert copied_column.type.compile(dialect=engine.dialect) == (
                    source_column.type.compile(dialect=engine.dialect)
                )

            source_foreign_keys = {
                (
                    foreign_key.parent.name,
                    foreign_key.target_fullname,
                    foreign_key.ondelete,
                )
                for column in source_table.columns
                for foreign_key in column.foreign_keys
            }
            copied_foreign_keys = {
                (
                    foreign_key.parent.name,
                    foreign_key.target_fullname,
                    foreign_key.ondelete,
                )
                for column in copied_table.columns
                for foreign_key in column.foreign_keys
            }
            assert copied_foreign_keys == source_foreign_keys

            source_indexes = {
                (index.name, tuple(column.name for column in index.columns))
                for index in source_table.indexes
            }
            copied_indexes = {
                (index.name, tuple(column.name for column in index.columns))
                for index in copied_table.indexes
            }
            assert copied_indexes == source_indexes

            source_unique_constraints = {
                tuple(column.name for column in constraint.columns)
                for constraint in source_table.constraints
                if isinstance(constraint, UniqueConstraint)
            }
            copied_unique_constraints = {
                tuple(column.name for column in constraint.columns)
                for constraint in copied_table.constraints
                if isinstance(constraint, UniqueConstraint)
            }
            assert copied_unique_constraints == source_unique_constraints
    finally:
        engine.dispose()


def test_bootstrap_db_creates_sqlite_file_and_all_tables(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "overtourism.sqlite"
    database_url = _sqlite_url(database_path)

    assert bootstrap_db.bootstrap_db(database_url=database_url) == 0
    assert database_path.is_file()

    engine = create_engine(database_url)
    try:
        table_names = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert table_names == {
        "evaluations",
        "problems",
        "proposal_scenario_relationship",
        "proposals",
        "scenarios",
        "sessions",
        "user_territories",
        "users",
    }


def test_bootstrap_db_creates_normalized_first_admin(tmp_path: Path) -> None:
    database_url = _sqlite_url(tmp_path / "overtourism.sqlite")

    assert bootstrap_db.bootstrap_db(" Admin@Example.org ", database_url) == 0

    engine = create_engine(database_url)
    users = Table("users", MetaData(), autoload_with=engine)
    try:
        with engine.connect() as connection:
            admin = connection.execute(select(users)).mappings().one()
    finally:
        engine.dispose()

    assert admin["identifier"] == "admin@example.org"
    assert admin["subject"] is None
    assert admin["role"] == "admin"
    assert admin["is_active"] is True


def test_bootstrap_db_accepts_a_non_email_identifier(tmp_path: Path) -> None:
    database_url = _sqlite_url(tmp_path / "overtourism.sqlite")

    assert bootstrap_db.bootstrap_db(" CF123XYZ ", database_url) == 0

    engine = create_engine(database_url)
    users = Table("users", MetaData(), autoload_with=engine)
    try:
        with engine.connect() as connection:
            admin = connection.execute(select(users)).mappings().one()
    finally:
        engine.dispose()

    assert admin["identifier"] == "cf123xyz"


def test_bootstrap_db_is_idempotent_for_existing_admin_and_rejects_another(
    tmp_path: Path,
) -> None:
    database_url = _sqlite_url(tmp_path / "overtourism.sqlite")

    assert bootstrap_db.bootstrap_db("admin@example.org", database_url) == 0
    assert bootstrap_db.bootstrap_db("admin@example.org", database_url) == 0
    assert bootstrap_db.bootstrap_db("another@example.org", database_url) == 1


def test_bootstrap_db_rejects_existing_non_admin(tmp_path: Path) -> None:
    database_url = _sqlite_url(tmp_path / "overtourism.sqlite")
    assert bootstrap_db.bootstrap_db(database_url=database_url) == 0
    _insert_user(database_url, identifier="viewer@example.org", role="viewer")

    assert bootstrap_db.bootstrap_db("viewer@example.org", database_url) == 1


def test_bootstrap_db_rejects_nonempty_user_registry(tmp_path: Path) -> None:
    database_url = _sqlite_url(tmp_path / "overtourism.sqlite")
    assert bootstrap_db.bootstrap_db(database_url=database_url) == 0
    _insert_user(database_url, identifier="viewer@example.org", role="viewer")

    assert bootstrap_db.bootstrap_db("admin@example.org", database_url) == 1


def test_bootstrap_db_rejects_empty_identifier_without_creating_database(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "overtourism.sqlite"

    assert bootstrap_db.bootstrap_db("   ", _sqlite_url(database_path)) == 2
    assert not database_path.exists()


def test_bootstrap_db_uses_environment_database_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "configured" / "overtourism.sqlite"
    monkeypatch.setenv("OVERTOURISM_DATABASE", _sqlite_url(database_path))

    assert bootstrap_db.bootstrap_db() == 0
    assert database_path.is_file()


def test_bootstrap_db_creates_default_database_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_directory = tmp_path / "db-bootstrap"
    script_directory.mkdir()
    monkeypatch.delenv("OVERTOURISM_DATABASE", raising=False)
    monkeypatch.setattr(
        bootstrap_db,
        "__file__",
        str(script_directory / "bootstrap_db.py"),
    )

    assert bootstrap_db.bootstrap_db() == 0
    assert (
        tmp_path
        / "overtourism-backend"
        / "overtourism"
        / "overtourism"
        / "database"
        / "overtourism.sqlite"
    ).is_file()


def test_bootstrap_db_supports_sqlite_memory_database() -> None:
    assert bootstrap_db.bootstrap_db(database_url="sqlite:///:memory:") == 0


def test_bootstrap_db_rejects_invalid_database_url() -> None:
    assert bootstrap_db.bootstrap_db(database_url="invalid://database") == 1


def test_bootstrap_db_reports_parent_directory_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bootstrap_db,
        "_ensure_sqlite_parent_dirs",
        lambda _database: (_ for _ in ()).throw(OSError("permission denied")),
    )

    assert (
        bootstrap_db.bootstrap_db(database_url=_sqlite_url(tmp_path / "db.sqlite")) == 1
    )


def test_bootstrap_db_reports_schema_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_create_all(_engine) -> None:
        raise SQLAlchemyError("schema unavailable")

    monkeypatch.setattr(bootstrap_db._DATABASE_SCHEMA, "create_all", fail_create_all)

    assert (
        bootstrap_db.bootstrap_db(database_url=_sqlite_url(tmp_path / "db.sqlite")) == 1
    )


def test_bootstrap_db_reports_unsupported_admin_dialect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_dialect(_connection, _dialect_name: str) -> None:
        raise RuntimeError("First-admin bootstrap is unsupported for mysql")

    monkeypatch.setattr(bootstrap_db, "_lock_user_registry", reject_dialect)

    assert (
        bootstrap_db.bootstrap_db(
            "admin@example.org",
            _sqlite_url(tmp_path / "db.sqlite"),
        )
        == 1
    )


def test_admin_transaction_uses_postgresql_lock() -> None:
    class RecordingConnection:
        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, statement) -> None:
            self.statements.append(str(statement))

        def exec_driver_sql(self, statement: str) -> None:
            self.statements.append(statement)

    connection = RecordingConnection()
    bootstrap_db._lock_user_registry(connection, "postgresql")

    assert connection.statements == ["LOCK TABLE users IN EXCLUSIVE MODE"]


def test_admin_transaction_rejects_unsupported_dialect() -> None:
    with pytest.raises(RuntimeError, match="unsupported for mysql"):
        bootstrap_db._lock_user_registry(object(), "mysql")


def test_main_accepts_no_email_and_optional_admin_email(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "overtourism.sqlite"
    monkeypatch.setenv("OVERTOURISM_DATABASE", _sqlite_url(database_path))

    assert bootstrap_db.main([]) == 0
    assert bootstrap_db.main(["admin@example.org"]) == 0


def test_script_entry_point_runs_without_backend_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "overtourism.sqlite"
    monkeypatch.setenv("OVERTOURISM_DATABASE", _sqlite_url(database_path))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH)])
    loaded_backend_modules = {
        name
        for name in sys.modules
        if name == "overtourism" or name.startswith("overtourism.")
    }

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(SCRIPT_PATH), run_name="__main__")

    assert exit_info.value.code == 0
    assert database_path.is_file()
    assert {
        name
        for name in sys.modules
        if name == "overtourism" or name.startswith("overtourism.")
    } == loaded_backend_modules
