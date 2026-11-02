# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import pytest
from fastapi.testclient import TestClient
from overtourism.backend.auth.identity.sql_repository import SQLUserRepository
from overtourism.backend.auth.identity.user_manager import UserManager
from overtourism.backend.auth.identity.users import UserRole
from overtourism.backend.auth.tokens.settings import AuthSettings, get_auth_settings
from overtourism.backend.main import create_app
from overtourism.dt_manager.stores.classes.sql.store import SQLStore


@pytest.fixture
def user_manager(handler):
    store = cast(SQLStore, handler.manager.store)
    user_manager = UserManager(SQLUserRepository(store.engine, store.session_factory))
    handler.user_manager = user_manager
    return user_manager


def test_list_territories_returns_model_keys_when_auth_is_disabled(
    handler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(enabled=False)
    monkeypatch.setattr(
        "overtourism.backend.api.routes.territory.list_models",
        lambda: [{"key": "territory-alpha"}, {"key": "territory-beta"}],
    )

    with TestClient(app) as client:
        response = client.get("/api/default/territories")

    assert response.status_code == 200
    assert response.json() == ["territory-alpha", "territory-beta"]


def test_list_territories_returns_all_model_keys_for_multieditor(
    handler,
    user_manager: UserManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_manager.create_user(
        identifier="user@example.org",
        role=UserRole.MULTIEDITOR,
        territories=["territory-alpha", "territory-beta"],
    )
    user_manager.claim_user_by_identifier("user@example.org", "user-1")

    app = create_app(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    monkeypatch.setattr(
        "overtourism.backend.auth.tokens.dependencies.decode_jwt",
        lambda token, settings: {"sub": "user-1"},
    )
    monkeypatch.setattr(
        "overtourism.backend.api.routes.territory.list_models",
        lambda: [
            {"key": "territory-alpha"},
            {"key": "territory-gamma"},
            {"key": "territory-beta"},
        ],
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/default/territories",
            headers={"Authorization": "Bearer signed-token"},
        )

    assert response.status_code == 200
    assert response.json() == [
        "territory-alpha",
        "territory-gamma",
        "territory-beta",
    ]


@pytest.mark.parametrize(
    ("role", "assignments", "expected_territories"),
    [
        (UserRole.EDITOR, [], []),
        (UserRole.VIEWER, ["territory-alpha"], ["territory-alpha"]),
        (
            UserRole.EDITOR,
            ["territory-alpha", "territory-gamma", "territory-missing"],
            ["territory-alpha", "territory-gamma"],
        ),
    ],
)
def test_list_territories_limits_editor_and_viewer_to_assignments(
    handler,
    user_manager: UserManager,
    monkeypatch: pytest.MonkeyPatch,
    role: UserRole,
    assignments: list[str],
    expected_territories: list[str],
) -> None:
    user_manager.create_user(
        identifier="user@example.org",
        role=role,
        territories=assignments,
    )
    user_manager.claim_user_by_identifier("user@example.org", "user-1")

    app = create_app(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    monkeypatch.setattr(
        "overtourism.backend.auth.tokens.dependencies.decode_jwt",
        lambda token, settings: {"sub": "user-1"},
    )
    monkeypatch.setattr(
        "overtourism.backend.api.routes.territory.list_models",
        lambda: [
            {"key": "territory-alpha"},
            {"key": "territory-gamma"},
            {"key": "territory-beta"},
        ],
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/default/territories",
            headers={"Authorization": "Bearer signed-token"},
        )

    assert response.status_code == 200
    assert response.json() == expected_territories


def test_list_territories_returns_all_model_keys_for_global_admin(
    handler,
    user_manager: UserManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_manager.create_first_admin("admin@example.org")
    user_manager.claim_user_by_identifier("admin@example.org", "admin-1")

    app = create_app(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    monkeypatch.setattr(
        "overtourism.backend.auth.tokens.dependencies.decode_jwt",
        lambda token, settings: {"sub": "admin-1"},
    )
    monkeypatch.setattr(
        "overtourism.backend.api.routes.territory.list_models",
        lambda: [
            {"key": "territory-alpha"},
            {"key": "territory-gamma"},
            {"key": "territory-beta"},
        ],
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/default/territories",
            headers={"Authorization": "Bearer signed-token"},
        )

    assert response.status_code == 200
    assert response.json() == [
        "territory-alpha",
        "territory-gamma",
        "territory-beta",
    ]


def test_list_territories_rejects_authenticated_unregistered_users(
    handler,
    user_manager: UserManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    monkeypatch.setattr(
        "overtourism.backend.auth.tokens.dependencies.decode_jwt",
        lambda token, settings: {"sub": "user-1"},
    )
    monkeypatch.setattr(
        "overtourism.backend.api.routes.territory.list_models",
        lambda: [{"key": "territory-alpha"}],
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/default/territories",
            headers={"Authorization": "Bearer signed-token"},
        )

    assert response.status_code == 403
    assert response.json() == {"detail": "User is not registered or active"}
