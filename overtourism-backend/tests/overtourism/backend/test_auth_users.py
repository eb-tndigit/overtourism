# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from datetime import UTC
from typing import cast

import pytest
from overtourism.backend.auth.identity.sql_repository import SQLUserRepository
from overtourism.backend.auth.identity.user_manager import UserManager
from overtourism.backend.auth.identity.users import UserRole
from overtourism.backend.auth.tokens.context import AuthContext
from overtourism.backend.auth.tokens.dependencies import get_auth_context
from overtourism.backend.auth.tokens.settings import AuthSettings, get_auth_settings
from overtourism.backend.handler import Handler
from overtourism.dt_manager.manager.manager import Manager
from overtourism.dt_manager.stores.classes.sql.store import SQLStore


@pytest.fixture
def user_manager(tmp_path):
    store = SQLStore(f"sqlite:///{tmp_path / 'users.db'}")
    repository = SQLUserRepository(store.engine, store.session_factory)
    yield UserManager(repository), repository
    store.engine.dispose()


def _register_admin(
    client,
    handler: Handler,
    manager: Manager,
) -> tuple[UserManager, SQLUserRepository]:
    store = cast(SQLStore, manager.store)
    repository = SQLUserRepository(store.engine, store.session_factory)
    user_manager = UserManager(repository)
    admin = user_manager.create_user(
        identifier="admin@example.org",
        role=UserRole.ADMIN,
        territories=[],
    )
    repository.save_user(replace(admin, subject="admin-sub"))
    user_manager.reload()
    handler.user_manager = user_manager
    client.app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        authenticated=True,
        subject="admin-sub",
        token="test-token",
        claims={"sub": "admin-sub"},
    )
    return user_manager, repository


def test_user_changes_are_persisted_and_refreshed_in_memory(user_manager) -> None:
    manager, repository = user_manager

    user = manager.create_user(
        identifier="  PERSON@example.org ",
        role=UserRole.EDITOR,
        territories={"molveno"},
    )
    assert user.identifier == "person@example.org"
    assert user.created_at.tzinfo is UTC
    assert manager.list_users() == [user]

    updated_user = manager.update_user(
        user.user_id,
        territories={"fazzon"},
    )
    assert updated_user.territories == frozenset({"fazzon"})
    assert manager.list_users() == [updated_user]

    linked_user = replace(updated_user, subject="subject-1")
    repository.save_user(linked_user)
    manager.reload()
    assert manager.get_active_user_by_subject("subject-1") == linked_user

    manager.deactivate_user(user.user_id)
    assert manager.get_active_user_by_subject("subject-1") is None
    assert manager.list_users()[0].is_active is False


@pytest.mark.parametrize(
    ("role", "territories"),
    [
        (UserRole.MULTIEDITOR, set()),
        (UserRole.ADMIN, {"molveno"}),
    ],
)
def test_user_creation_enforces_role_territory_cardinality(
    user_manager,
    role: UserRole,
    territories: set[str],
) -> None:
    manager, _ = user_manager

    with pytest.raises(ValueError):
        manager.create_user(
            identifier="person@example.org",
            role=role,
            territories=territories,
        )


@pytest.mark.parametrize(
    ("role", "territories"),
    [
        (UserRole.ADMIN, []),
        (UserRole.MULTIEDITOR, ["molveno", "fazzon"]),
        (UserRole.EDITOR, []),
        (UserRole.EDITOR, ["molveno"]),
        (UserRole.EDITOR, ["molveno", "fazzon"]),
        (UserRole.VIEWER, []),
        (UserRole.VIEWER, ["molveno"]),
        (UserRole.VIEWER, ["molveno", "fazzon"]),
    ],
)
def test_user_creation_accepts_zero_or_more_role_territories(
    user_manager,
    role: UserRole,
    territories: list[str],
) -> None:
    manager, _ = user_manager

    user = manager.create_user(
        identifier="person@example.org",
        role=role,
        territories=territories,
    )

    assert user.role is role
    assert user.territories == frozenset(territories)


def test_user_creation_rejects_empty_identifier(user_manager) -> None:
    manager, _ = user_manager

    with pytest.raises(ValueError, match="identifier must not be empty"):
        manager.create_user(
            identifier="  ",
            role=UserRole.ADMIN,
            territories=[],
        )


def test_user_update_to_admin_clears_territory_assignments(user_manager) -> None:
    manager, _ = user_manager
    user = manager.create_user(
        identifier="person@example.org",
        role=UserRole.EDITOR,
        territories=["molveno"],
    )

    updated_user = manager.update_user(user.user_id, role=UserRole.ADMIN)

    assert updated_user.territories == frozenset()


def test_user_update_rejects_unknown_user(user_manager) -> None:
    manager, _ = user_manager

    with pytest.raises(KeyError, match="not found"):
        manager.update_user("missing-user", is_active=False)


def test_inactive_and_unlinked_users_are_not_resolved(user_manager) -> None:
    manager, _ = user_manager
    user = manager.create_user(
        identifier="person@example.org",
        role=UserRole.VIEWER,
        territories={"molveno"},
    )

    assert manager.get_active_user_by_subject("missing-subject") is None
    assert manager.get_active_user_by_subject(user.user_id) is None


def test_failed_reload_clears_cached_users(user_manager, monkeypatch) -> None:
    manager, repository = user_manager
    user = manager.create_user(
        identifier="person@example.org",
        role=UserRole.VIEWER,
        territories={"molveno"},
    )
    repository.save_user(replace(user, subject="subject-1"))
    manager.reload()

    def fail_to_load_users():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(repository, "load_users", fail_to_load_users)
    with pytest.raises(RuntimeError, match="database unavailable"):
        manager.reload()

    assert manager.get_active_user_by_subject("subject-1") is None
    assert manager.list_users() == []


def test_admin_user_api_mutations_refresh_cache(client, handler, manager) -> None:
    user_manager, _ = _register_admin(client, handler, manager)

    created_response = client.post(
        "/api/auth/users",
        json={
            "identifier": " Editor@Example.org ",
            "role": "multieditor",
            "territories": ["molveno", "fazzon"],
        },
    )
    assert created_response.status_code == 201
    created = created_response.json()
    assert created["user_id"]
    assert created["identifier"] == "editor@example.org"
    assert created["territories"] == ["fazzon", "molveno"]
    assert user_manager.get_active_user_by_subject("editor-sub") is None

    linked_user = user_manager.claim_user_by_identifier(
        "editor@example.org",
        "editor-sub",
    )
    assert linked_user is not None

    updated_response = client.patch(
        f"/api/auth/users/{created['user_id']}",
        json={"role": "editor", "territories": ["molveno"]},
    )
    assert updated_response.status_code == 200
    assert updated_response.json()["role"] == "editor"
    assert updated_response.json()["identifier"] == "editor@example.org"
    assert user_manager.get_active_user_by_subject("editor-sub").territories == {
        "molveno"
    }

    deleted_response = client.delete(f"/api/auth/users/{created['user_id']}")
    assert deleted_response.status_code == 200
    assert deleted_response.json()["is_active"] is False
    assert deleted_response.json()["identifier"] == "editor@example.org"
    assert user_manager.get_active_user_by_subject("editor-sub") is None


def test_multieditor_defaults_to_all_model_territories(
    client,
    handler: Handler,
    manager: Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_admin(client, handler, manager)
    monkeypatch.setattr(
        "overtourism.backend.auth.api.router.list_models",
        lambda: [{"key": "fazzon"}, {"key": "molveno"}],
    )

    created_response = client.post(
        "/api/auth/users",
        json={
            "identifier": "editor@example.org",
            "role": "multieditor",
        },
    )

    assert created_response.status_code == 201
    created = created_response.json()
    assert created["territories"] == ["fazzon", "molveno"]

    editor_response = client.patch(
        f"/api/auth/users/{created['user_id']}",
        json={"role": "editor", "territories": ["molveno"]},
    )
    assert editor_response.status_code == 200

    restored_multieditor_response = client.patch(
        f"/api/auth/users/{created['user_id']}",
        json={"role": "multieditor"},
    )

    assert restored_multieditor_response.status_code == 200
    assert restored_multieditor_response.json()["territories"] == [
        "fazzon",
        "molveno",
    ]


def test_current_user_and_role_list_require_no_admin_role(
    client,
    handler: Handler,
    manager: Manager,
) -> None:
    store = cast(SQLStore, manager.store)
    repository = SQLUserRepository(store.engine, store.session_factory)
    user_manager = UserManager(repository)
    user = user_manager.create_user(
        identifier="viewer@example.org",
        role=UserRole.VIEWER,
        territories=["molveno"],
    )
    repository.save_user(replace(user, subject="viewer-sub"))
    user_manager.reload()
    handler.user_manager = user_manager
    client.app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        authenticated=True,
        subject="viewer-sub",
        territory="molveno",
        token="test-token",
        claims={"sub": "viewer-sub"},
    )

    me_response = client.get("/api/auth/me")
    assert me_response.status_code == 200
    assert me_response.json()["user_id"] == user.user_id
    assert me_response.json()["role"] == "viewer"
    assert me_response.json()["territories"] == ["molveno"]

    roles_response = client.get("/api/auth/roles")
    assert roles_response.status_code == 200
    assert {role["role"]: role["description"] for role in roles_response.json()} == {
        "admin": "Amministratore globale con accesso a tutti i territori.",
        "multieditor": "Editor assegnato a tutti i territori per impostazione predefinita.",
        "editor": "Editor con accesso a un insieme di territori, anche vuoto.",
        "viewer": "Utente di sola lettura assegnato a un insieme di territori, anche vuoto.",
    }

    users_response = client.get("/api/auth/users")
    assert users_response.status_code == 403


def _register_pending_admin(
    handler: Handler,
    manager: Manager,
) -> UserManager:
    store = cast(SQLStore, manager.store)
    repository = SQLUserRepository(store.engine, store.session_factory)
    user_manager = UserManager(repository)
    user_manager.create_user(
        identifier="Admin@example.org",
        role=UserRole.ADMIN,
        territories=[],
    )
    handler.user_manager = user_manager
    return user_manager


def test_verified_email_claims_first_admin_and_allows_user_registration(
    client,
    handler: Handler,
    manager: Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_manager = _register_pending_admin(handler, manager)
    client.app.dependency_overrides.pop(get_auth_context)
    client.app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.org/jwks",
    )
    token_claims = {
        "sub": "first-admin-sub",
        "email": "admin@example.org",
        "email_verified": True,
    }
    monkeypatch.setattr(
        "overtourism.backend.auth.tokens.dependencies.decode_jwt",
        lambda token, settings: token_claims,
    )

    response = client.post(
        "/api/auth/users",
        json={
            "identifier": "editor@example.org",
            "role": "editor",
            "territories": ["molveno"],
        },
        headers={"Authorization": "Bearer verified-token"},
    )

    assert response.status_code == 201
    linked_admin = user_manager.get_active_user_by_subject("first-admin-sub")
    assert linked_admin is not None
    assert linked_admin.role is UserRole.ADMIN
    assert linked_admin.identifier == "admin@example.org"

    token_claims.update(
        {
            "sub": "editor-sub",
            "email": "editor@example.org",
            "email_verified": True,
        }
    )
    editor_response = client.get(
        "/api/auth/me",
        headers={"Authorization": "Bearer editor-token"},
    )

    assert editor_response.status_code == 200
    assert editor_response.json()["user_id"] == response.json()["user_id"]
    assert editor_response.json()["role"] == "editor"
    assert user_manager.get_active_user_by_subject("editor-sub") is not None


@pytest.mark.parametrize(
    "claims",
    [
        {"email": "admin@example.org", "email_verified": False},
        {"email": "admin@example.org"},
        {"email": "someone-else@example.org", "email_verified": True},
    ],
)
def test_unverified_or_mismatched_email_cannot_claim_pending_admin(
    client,
    handler: Handler,
    manager: Manager,
    claims: dict[str, object],
) -> None:
    user_manager = _register_pending_admin(handler, manager)
    client.app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        authenticated=True,
        subject="first-admin-sub",
        token="unverified-token",
        claims={"sub": "first-admin-sub", **claims},
    )

    response = client.get("/api/auth/me")

    assert response.status_code == 403
    assert user_manager.get_active_user_by_subject("first-admin-sub") is None
