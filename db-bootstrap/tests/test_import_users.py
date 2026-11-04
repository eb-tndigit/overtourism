# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import MetaData, Table, create_engine, insert, select

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bootstrap_db = _load_module("bootstrap_db", PROJECT_DIR / "bootstrap_db.py")
import_users = _load_module("import_users", PROJECT_DIR / "import_users.py")


def _database_url(path: Path) -> str:
    return f"sqlite:///{path}"


def test_import_users_inserts_normalized_users_and_updates_existing_role(
    tmp_path: Path,
) -> None:
    database_url = _database_url(tmp_path / "users.sqlite")
    assert bootstrap_db.bootstrap_db(database_url=database_url) == 0
    engine = create_engine(database_url)
    users = Table("users", MetaData(), autoload_with=engine)
    try:
        with engine.begin() as connection:
            connection.execute(
                insert(users).values(
                    user_id="existing-user",
                    identifier="existing@example.org",
                    subject="identity-provider-subject",
                    role="viewer",
                    is_active=False,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )
    finally:
        engine.dispose()

    csv_path = tmp_path / "users.csv"
    csv_path.write_text(
        "email;role\n" " New@Example.org ;editor\n" "EXISTING@example.org;admin\n",
        encoding="utf-8",
    )

    summary = import_users.import_users(csv_path, database_url)

    assert summary.inserted == 1
    assert summary.updated == 1
    assert summary.unchanged == 0
    engine = create_engine(database_url)
    users = Table("users", MetaData(), autoload_with=engine)
    try:
        with engine.connect() as connection:
            rows = {
                row["identifier"]: row
                for row in connection.execute(select(users)).mappings()
            }
    finally:
        engine.dispose()

    assert rows["new@example.org"]["role"] == "editor"
    assert rows["new@example.org"]["is_active"] is True
    assert rows["existing@example.org"]["role"] == "admin"
    assert rows["existing@example.org"]["subject"] == "identity-provider-subject"
    assert rows["existing@example.org"]["is_active"] is False


def test_import_users_leaves_same_role_unchanged(tmp_path: Path) -> None:
    database_url = _database_url(tmp_path / "users.sqlite")
    assert bootstrap_db.bootstrap_db(database_url=database_url) == 0
    csv_path = tmp_path / "users.csv"
    csv_path.write_text("email;role\nperson@example.org;viewer\n", encoding="utf-8")

    first = import_users.import_users(csv_path, database_url)
    second = import_users.import_users(csv_path, database_url)

    assert first.inserted == 1
    assert second.unchanged == 1
    assert second.updated == 0


def test_import_users_rejects_invalid_role_without_partial_changes(
    tmp_path: Path,
) -> None:
    database_url = _database_url(tmp_path / "users.sqlite")
    assert bootstrap_db.bootstrap_db(database_url=database_url) == 0
    csv_path = tmp_path / "users.csv"
    csv_path.write_text(
        "email;role\nvalid@example.org;editor\ninvalid@example.org;owner\n",
        encoding="utf-8",
    )

    try:
        import_users.import_users(csv_path, database_url)
    except ValueError as exc:
        assert "invalid role" in str(exc)
    else:
        raise AssertionError("Invalid user role should be rejected.")

    engine = create_engine(database_url)
    users = Table("users", MetaData(), autoload_with=engine)
    try:
        with engine.connect() as connection:
            assert connection.scalar(select(users.c.user_id).limit(1)) is None
    finally:
        engine.dispose()


def test_import_users_rejects_new_multieditors(tmp_path: Path) -> None:
    database_url = _database_url(tmp_path / "users.sqlite")
    assert bootstrap_db.bootstrap_db(database_url=database_url) == 0
    csv_path = tmp_path / "users.csv"
    csv_path.write_text(
        "email;role\nperson@example.org;multieditor\n", encoding="utf-8"
    )

    try:
        import_users.import_users(csv_path, database_url)
    except ValueError as exc:
        assert "without territories" in str(exc)
    else:
        raise AssertionError("A multieditor without territories should be rejected.")


def test_import_users_supports_territories_and_multieditors(tmp_path: Path) -> None:
    database_url = _database_url(tmp_path / "users.sqlite")
    assert bootstrap_db.bootstrap_db(database_url=database_url) == 0
    csv_path = tmp_path / "users.csv"
    csv_path.write_text(
        "email;role;territories\n"
        "editor@example.org;editor; Molveno, Fazzon \n"
        "viewer@example.org;viewer;Fazzon\n"
        "multi@example.org;multieditor;Molveno\n",
        encoding="utf-8",
    )

    summary = import_users.import_users(csv_path, database_url)

    assert summary.inserted == 3
    engine = create_engine(database_url)
    users = Table("users", MetaData(), autoload_with=engine)
    user_territories = Table("user_territories", MetaData(), autoload_with=engine)
    try:
        with engine.connect() as connection:
            ids = {
                row["identifier"]: row["user_id"]
                for row in connection.execute(select(users)).mappings()
            }
            territories: dict[str, set[str]] = {}
            for row in connection.execute(
                select(user_territories.c.user_id, user_territories.c.territory)
            ).mappings():
                territories.setdefault(row["user_id"], set()).add(row["territory"])
    finally:
        engine.dispose()

    assert territories[ids["editor@example.org"]] == {"Fazzon", "Molveno"}
    assert territories[ids["viewer@example.org"]] == {"Fazzon"}
    assert territories[ids["multi@example.org"]] == {"Molveno"}


def test_import_users_clears_blank_territories_and_preserves_when_column_missing(
    tmp_path: Path,
) -> None:
    database_url = _database_url(tmp_path / "users.sqlite")
    assert bootstrap_db.bootstrap_db(database_url=database_url) == 0
    with_territory_column = tmp_path / "with-territories.csv"
    with_territory_column.write_text(
        "email;role;territories\nperson@example.org;editor;Molveno\n",
        encoding="utf-8",
    )
    without_territory_column = tmp_path / "without-territories.csv"
    without_territory_column.write_text(
        "email;role\nperson@example.org;editor\n", encoding="utf-8"
    )
    cleared_csv = tmp_path / "cleared.csv"
    cleared_csv.write_text(
        "email;role;territories\nperson@example.org;editor;\n", encoding="utf-8"
    )

    import_users.import_users(with_territory_column, database_url)
    preserved = import_users.import_users(without_territory_column, database_url)
    cleared = import_users.import_users(cleared_csv, database_url)

    assert preserved.unchanged == 1
    assert cleared.updated == 1
    engine = create_engine(database_url)
    user_territories = Table("user_territories", MetaData(), autoload_with=engine)
    try:
        with engine.connect() as connection:
            assert (
                connection.scalar(select(user_territories.c.user_id).limit(1)) is None
            )
    finally:
        engine.dispose()


def test_import_users_rejects_multieditor_without_territories(tmp_path: Path) -> None:
    database_url = _database_url(tmp_path / "users.sqlite")
    assert bootstrap_db.bootstrap_db(database_url=database_url) == 0
    csv_path = tmp_path / "users.csv"
    csv_path.write_text(
        "email;role;territories\nperson@example.org;multieditor;\n",
        encoding="utf-8",
    )

    try:
        import_users.import_users(csv_path, database_url)
    except ValueError as exc:
        assert "without territories" in str(exc)
    else:
        raise AssertionError("A multieditor without territories should be rejected.")
