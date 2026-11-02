# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
import requests
from fastapi import HTTPException
from fastapi.testclient import TestClient
from overtourism.backend.api.routes import executor as executor_api
from overtourism.backend.auth.identity.authorization import require_global_admin
from overtourism.backend.auth.identity.sql_repository import SQLUserRepository
from overtourism.backend.auth.identity.user_manager import UserManager
from overtourism.backend.auth.identity.users import UserRole
from overtourism.backend.auth.tokens import jwt as auth_jwt
from overtourism.backend.auth.tokens.context import AuthContext
from overtourism.backend.auth.tokens.dependencies import get_auth_context
from overtourism.backend.auth.tokens.settings import AuthSettings, get_auth_settings
from overtourism.backend.handler import Handler
from overtourism.backend.main import create_app
from overtourism.dt_manager.manager.manager import Manager
from overtourism.dt_manager.stores.classes.sql.store import SQLStore
from overtourism.dt_manager.stores.config import StoreConfig
from overtourism.dt_manager.stores.enums import StoreType
from overtourism.overtourism.backend_extension.api.routes import indexes as indexes_api
from overtourism.overtourism.backend_extension.api.routes.indexes import indexes_router
from overtourism.overtourism.backend_extension.api.routes.problem import (
    problem_router as overtourism_problem_router,
)
from overtourism.overtourism.backend_extension.api.routes.proposal import (
    proposal_router as overtourism_proposal_router,
)


@pytest.fixture
def handler(tmp_path) -> Handler:
    manager = Manager(
        store_config=StoreConfig(
            store_type=StoreType.SQL.value,
            config={"url": f"sqlite:///{tmp_path / 'store.db'}"},
        ),
    )
    store = cast(SQLStore, manager.store)
    repository = SQLUserRepository(store.engine, store.session_factory)
    user_manager = UserManager(repository)
    for subject in ("101", "user-1"):
        user = user_manager.create_user(
            identifier=f"{subject}@example.org",
            role=UserRole.VIEWER,
            territories=["territory-alpha"],
        )
        repository.save_user(replace(user, subject=subject))
    user_manager.reload()
    return Handler(manager=manager, user_manager=user_manager)


def test_auth_roles_rejects_an_unauthenticated_request(handler: Handler) -> None:
    app = create_app(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(enabled=False)

    with TestClient(app) as client:
        response = client.get("/api/auth/roles")

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication is required"}


def test_auth_me_rejects_a_missing_subject_claim(handler: Handler) -> None:
    app = create_app(handler)
    app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        authenticated=True,
        subject=None,
        token="signed-token",
        claims={},
    )

    with TestClient(app) as client:
        response = client.get("/api/auth/me")

    assert response.status_code == 401
    assert response.json() == {"detail": "Missing user identity claim"}


def test_auth_me_fails_when_user_manager_is_not_configured(handler: Handler) -> None:
    handler.user_manager = None
    app = create_app(handler)
    app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        authenticated=True,
        subject="viewer-sub",
        token="signed-token",
        claims={"sub": "viewer-sub"},
    )

    with TestClient(app) as client:
        response = client.get("/api/auth/me")

    assert response.status_code == 503
    assert response.json() == {"detail": "User manager is not configured"}


def test_auth_user_list_fails_when_user_manager_is_not_configured(
    handler: Handler,
) -> None:
    handler.user_manager = None
    app = create_app(handler)
    app.dependency_overrides[require_global_admin] = lambda: None

    with TestClient(app) as client:
        response = client.get("/api/auth/users")

    assert response.status_code == 503
    assert response.json() == {"detail": "User manager is not configured"}


@pytest.mark.parametrize(
    ("app_factory", "auth_path"),
    [
        (create_app, "/api/auth/me"),
    ],
)
def test_auth_me_returns_unauthenticated_context_when_auth_is_disabled(
    handler,
    app_factory,
    auth_path: str,
) -> None:
    app = app_factory(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(enabled=False)

    with TestClient(app) as client:
        response = client.get(auth_path)

    assert response.status_code == 200
    assert response.json() == {
        "authenticated": False,
        "subject": None,
        "user_id": None,
        "identifier": None,
        "role": None,
        "territories": [],
    }


def test_auth_me_links_user_using_configured_identifier_claim(
    handler: Handler,
) -> None:
    user_manager = handler.user_manager
    assert user_manager is not None
    pending_user = user_manager.create_user(
        identifier="CF123XYZ",
        role=UserRole.VIEWER,
        territories=["territory-alpha"],
    )
    app = create_app(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        identifier_claim="tax_id",
    )
    app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        authenticated=True,
        subject="cf-subject",
        token="signed-token",
        claims={"sub": "cf-subject", "tax_id": "CF123XYZ"},
    )

    with TestClient(app) as client:
        response = client.get("/api/auth/me")

    assert response.status_code == 200
    assert response.json()["user_id"] == pending_user.user_id
    assert user_manager.get_active_user_by_subject("cf-subject") is not None


@pytest.mark.parametrize("claim_value", [None, "", "   ", 123])
def test_auth_me_rejects_missing_or_invalid_configured_identifier_claim(
    handler: Handler,
    claim_value: object,
) -> None:
    user_manager = handler.user_manager
    assert user_manager is not None
    user_manager.create_user(
        identifier="CF123XYZ",
        role=UserRole.VIEWER,
        territories=["territory-alpha"],
    )
    claims = {} if claim_value is None else {"tax_id": claim_value}
    app = create_app(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        identifier_claim="tax_id",
    )
    app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        authenticated=True,
        subject="cf-subject",
        token="signed-token",
        claims={"sub": "cf-subject", **claims},
    )

    with TestClient(app) as client:
        response = client.get("/api/auth/me")

    assert response.status_code == 403
    assert user_manager.get_active_user_by_subject("cf-subject") is None


@pytest.mark.parametrize(
    ("app_factory", "auth_path"),
    [
        (create_app, "/api/auth/me"),
    ],
)
def test_auth_me_requires_bearer_token_when_auth_is_enabled(
    handler,
    app_factory,
    auth_path: str,
) -> None:
    app = app_factory(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.com/.well-known/jwks.json",
    )

    with TestClient(app) as client:
        response = client.get(auth_path)

    assert response.status_code == 401
    assert response.json() == {"detail": "Missing bearer token"}


@pytest.mark.parametrize(
    ("app_factory", "auth_path"),
    [
        (create_app, "/api/auth/me"),
    ],
)
def test_auth_me_returns_authenticated_context_and_database_territory(
    handler,
    monkeypatch: pytest.MonkeyPatch,
    app_factory,
    auth_path: str,
) -> None:
    app = app_factory(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    monkeypatch.setattr(
        "overtourism.backend.auth.tokens.dependencies.decode_jwt",
        lambda token, settings: {"sub": 101},
    )

    with TestClient(app) as client:
        response = client.get(
            auth_path,
            headers={"Authorization": "Bearer signed-token"},
        )

    assert response.status_code == 200
    user_data = response.json()
    assert user_data["authenticated"] is True
    assert "territory" not in user_data
    assert user_data["subject"] == "101"
    assert user_data["user_id"]
    assert user_data["identifier"] == "101@example.org"
    assert user_data["role"] == "viewer"
    assert user_data["territories"] == ["territory-alpha"]


def test_auth_me_returns_all_assigned_territories_without_singular_field(
    handler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = handler.user_manager.create_user(
        identifier="multi-viewer@example.org",
        role=UserRole.VIEWER,
        territories=["territory-beta", "territory-alpha"],
    )
    handler.user_manager.claim_user_by_identifier(user.identifier, "multi-viewer")

    app = create_app(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    monkeypatch.setattr(
        "overtourism.backend.auth.tokens.dependencies.decode_jwt",
        lambda token, settings: {"sub": "multi-viewer"},
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer signed-token"},
        )

    assert response.status_code == 200
    assert "territory" not in response.json()
    assert response.json()["identifier"] == "multi-viewer@example.org"
    assert response.json()["territories"] == ["territory-alpha", "territory-beta"]


@pytest.mark.parametrize(
    ("app_factory", "problems_path"),
    [
        (create_app, "/api/territory-alpha/problems"),
    ],
)
@pytest.mark.parametrize("tenant_claim", [None, "territory-beta"])
def test_territory_scoped_routes_allow_database_assignment_without_territory_claim(
    handler,
    monkeypatch: pytest.MonkeyPatch,
    app_factory,
    problems_path: str,
    tenant_claim: str | None,
) -> None:
    app = app_factory(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    claims = {"sub": "user-1"}
    if tenant_claim is not None:
        claims["tenant_id"] = tenant_claim
    monkeypatch.setattr(
        "overtourism.backend.auth.tokens.dependencies.decode_jwt",
        lambda token, settings: claims,
    )

    with TestClient(app) as client:
        response = client.get(
            problems_path,
            headers={"Authorization": "Bearer signed-token"},
        )

    assert response.status_code == 200


@pytest.mark.parametrize(
    ("app_factory", "route_path"),
    [
        (create_app, "/api/territory-gamma/problems"),
        (create_app, "/api/territory-gamma/proposals"),
        (create_app, "/api/territory-gamma/scenarios"),
        (create_app, "/api/territory-gamma/evaluations"),
        (create_app, "/api/territory-gamma/configuration"),
        (create_app, "/api/territory-gamma/sessions"),
    ],
)
def test_territory_scoped_routes_reject_database_unassigned_territory_despite_jwt_claim(
    handler,
    monkeypatch: pytest.MonkeyPatch,
    app_factory,
    route_path: str,
) -> None:
    app = app_factory(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    monkeypatch.setattr(
        "overtourism.backend.auth.tokens.dependencies.decode_jwt",
        lambda token, settings: {
            "sub": "user-1",
            "tenant_id": "territory-gamma",
        },
    )

    with TestClient(app) as client:
        response = client.get(
            route_path,
            headers={"Authorization": "Bearer signed-token"},
        )

    assert response.status_code == 403
    assert response.json() == {"detail": "User is not assigned to this territory"}


def test_global_admin_can_access_unassigned_application_territories(
    handler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = cast(SQLStore, handler.manager.store)
    repository = SQLUserRepository(store.engine, store.session_factory)
    user_manager = UserManager(repository)
    admin = user_manager.create_user(
        identifier="global-admin@example.org",
        role=UserRole.ADMIN,
        territories=[],
    )
    repository.save_user(replace(admin, subject="global-admin"))
    user_manager.reload()
    handler.user_manager = user_manager

    app = create_app(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    monkeypatch.setattr(
        "overtourism.backend.auth.tokens.dependencies.decode_jwt",
        lambda token, settings: {"sub": "global-admin"},
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/territory-gamma/problems",
            headers={"Authorization": "Bearer signed-token"},
        )

    assert response.status_code == 200


def test_auth_me_returns_all_model_territories_for_admin(
    handler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = cast(SQLStore, handler.manager.store)
    repository = SQLUserRepository(store.engine, store.session_factory)
    user_manager = UserManager(repository)
    admin = user_manager.create_user(
        identifier="global-admin@example.org",
        role=UserRole.ADMIN,
        territories=[],
    )
    repository.save_user(replace(admin, subject="global-admin"))
    user_manager.reload()
    handler.user_manager = user_manager

    app = create_app(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    monkeypatch.setattr(
        "overtourism.backend.auth.tokens.dependencies.decode_jwt",
        lambda token, settings: {"sub": "global-admin"},
    )
    monkeypatch.setattr(
        "overtourism.backend.auth.api.router.list_models",
        lambda: [{"key": "territory-beta"}, {"key": "territory-alpha"}],
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer signed-token"},
        )

    assert response.status_code == 200
    assert response.json()["identifier"] == "global-admin@example.org"
    assert response.json()["territories"] == ["territory-alpha", "territory-beta"]


def test_overtourism_routes_use_db_scope_but_indexes_require_only_a_valid_jwt(
    handler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(
        handler,
        include_problem_router=False,
        include_proposal_router=False,
        extra_routers=[
            overtourism_problem_router,
            overtourism_proposal_router,
            indexes_router,
        ],
    )
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    monkeypatch.setattr(
        "overtourism.backend.auth.tokens.dependencies.decode_jwt",
        lambda token, settings: {"sub": "unregistered-user"},
    )
    monkeypatch.setattr(indexes_api, "_REGISTRY", {})

    with TestClient(app) as client:
        problem_response = client.get(
            "/api/territory-gamma/problems",
            headers={"Authorization": "Bearer signed-token"},
        )
        proposal_response = client.get(
            "/api/territory-gamma/proposals",
            headers={"Authorization": "Bearer signed-token"},
        )
        index_response = client.get(
            "/api/default/indexes/get-index-list",
            headers={"Authorization": "Bearer signed-token"},
        )
        unauthenticated_index_response = client.get(
            "/api/default/indexes/get-index-list"
        )

    assert problem_response.status_code == 403
    assert proposal_response.status_code == 403
    assert index_response.status_code == 200
    assert unauthenticated_index_response.status_code == 401


@pytest.mark.parametrize(
    ("subject", "expected_status"),
    [("user-1", 403), ("global-admin", 200)],
)
def test_force_index_download_requires_global_admin(
    handler: Handler,
    monkeypatch: pytest.MonkeyPatch,
    subject: str,
    expected_status: int,
) -> None:
    store = cast(SQLStore, handler.manager.store)
    repository = SQLUserRepository(store.engine, store.session_factory)
    user_manager = UserManager(repository)
    admin = user_manager.create_user(
        identifier="global-admin@example.org",
        role=UserRole.ADMIN,
        territories=[],
    )
    repository.save_user(replace(admin, subject="global-admin"))
    user_manager.reload()
    handler.user_manager = user_manager

    app = create_app(handler, extra_routers=[indexes_router])
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    monkeypatch.setattr(
        "overtourism.backend.auth.tokens.dependencies.decode_jwt",
        lambda token, settings: {"sub": subject},
    )
    download_calls: list[bool] = []
    monkeypatch.setattr(
        indexes_api,
        "download_index_data",
        lambda *, overwrite: download_calls.append(overwrite),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/default/indexes/download-index-data",
            headers={"Authorization": "Bearer signed-token"},
        )

    assert response.status_code == expected_status
    assert download_calls == ([True] if subject == "global-admin" else [])
    if subject == "global-admin":
        assert response.json() == {"message": "Index data download completed"}


@pytest.mark.parametrize(
    ("subject", "expected_status"),
    [("user-1", 403), ("global-admin", 200)],
)
def test_executor_data_download_requires_global_admin(
    handler: Handler,
    monkeypatch: pytest.MonkeyPatch,
    subject: str,
    expected_status: int,
) -> None:
    store = cast(SQLStore, handler.manager.store)
    repository = SQLUserRepository(store.engine, store.session_factory)
    user_manager = UserManager(repository)
    admin = user_manager.create_user(
        identifier="global-admin@example.org",
        role=UserRole.ADMIN,
        territories=[],
    )
    repository.save_user(replace(admin, subject="global-admin"))
    user_manager.reload()
    handler.user_manager = user_manager

    app = create_app(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    monkeypatch.setattr(
        "overtourism.backend.auth.tokens.dependencies.decode_jwt",
        lambda token, settings: {"sub": subject},
    )
    download_calls: list[str] = []

    def download_model_data(model: str) -> dict[str, str]:
        download_calls.append(model)
        return {"message": f"{model} data downloaded", "data_dir": "/data/model"}

    monkeypatch.setattr(
        executor_api,
        "call_download_data",
        download_model_data,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/territory-alpha/configuration/download-data",
            headers={"Authorization": "Bearer signed-token"},
        )

    assert response.status_code == expected_status
    assert download_calls == (["territory-alpha"] if subject == "global-admin" else [])
    if subject == "global-admin":
        assert response.json() == {
            "message": "territory-alpha data downloaded",
            "data_dir": "/data/model",
        }


@pytest.mark.parametrize(
    ("status_code", "body", "expected_status", "expected_detail"),
    [
        (
            404,
            b'{"detail":"Unknown model: \'molveno\'"}',
            404,
            "Unknown model: 'molveno'",
        ),
        (502, b"upstream error", 502, "Model API data download failed"),
    ],
)
def test_executor_data_download_propagates_upstream_http_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    body: bytes,
    expected_status: int,
    expected_detail: str,
) -> None:
    upstream_response = requests.Response()
    upstream_response.status_code = status_code
    upstream_response._content = body

    def raise_http_error(model: str) -> dict[str, str]:
        raise requests.HTTPError(response=upstream_response)

    monkeypatch.setattr(executor_api, "call_download_data", raise_http_error)

    with pytest.raises(HTTPException) as exc_info:
        executor_api.download_model_data("molveno")

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail


def test_executor_data_download_maps_connection_errors_to_bad_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_connection_error(model: str) -> dict[str, str]:
        raise requests.ConnectionError("executor unavailable")

    monkeypatch.setattr(executor_api, "call_download_data", raise_connection_error)

    with pytest.raises(HTTPException) as exc_info:
        executor_api.download_model_data("molveno")

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Model API data download request failed"


@pytest.mark.parametrize(
    ("use_overtourism_routes", "method", "path", "payload"),
    [
        (
            False,
            "POST",
            "/api/territory-alpha/problems",
            {"name": "New", "description": "Problem"},
        ),
        (False, "PUT", "/api/territory-alpha/problems/missing", {"version": 1}),
        (False, "DELETE", "/api/territory-alpha/problems/missing", None),
        (False, "POST", "/api/territory-alpha/proposals", {"problem_id": "missing"}),
        (False, "PUT", "/api/territory-alpha/proposals/missing", {"version": 1}),
        (False, "DELETE", "/api/territory-alpha/proposals/missing", None),
        (False, "POST", "/api/territory-alpha/scenarios", {}),
        (False, "PUT", "/api/territory-alpha/scenarios/missing", {"version": 1}),
        (False, "DELETE", "/api/territory-alpha/scenarios/missing", None),
        (False, "POST", "/api/territory-alpha/evaluations", {"scenario_id": "missing"}),
        (False, "PUT", "/api/territory-alpha/evaluations/missing", {}),
        (False, "DELETE", "/api/territory-alpha/evaluations/missing", None),
        (
            True,
            "POST",
            "/api/territory-alpha/problems",
            {"name": "New", "description": "Problem"},
        ),
        (True, "PUT", "/api/territory-alpha/problems/missing", {"version": 1}),
        (True, "DELETE", "/api/territory-alpha/problems/missing", None),
        (True, "POST", "/api/territory-alpha/proposals", {"problem_id": "missing"}),
        (True, "PUT", "/api/territory-alpha/proposals/missing", {"version": 1}),
        (True, "DELETE", "/api/territory-alpha/proposals/missing", None),
    ],
)
def test_viewer_cannot_write_persistent_resources(
    handler: Handler,
    monkeypatch: pytest.MonkeyPatch,
    use_overtourism_routes: bool,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
) -> None:
    if use_overtourism_routes:
        app = create_app(
            handler,
            include_problem_router=False,
            include_proposal_router=False,
            extra_routers=[overtourism_problem_router, overtourism_proposal_router],
        )
    else:
        app = create_app(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    monkeypatch.setattr(
        "overtourism.backend.auth.tokens.dependencies.decode_jwt",
        lambda token, settings: {"sub": "user-1"},
    )

    request_kwargs: dict[str, Any] = {
        "headers": {"Authorization": "Bearer signed-token"},
    }
    if payload is not None:
        request_kwargs["json"] = payload

    with TestClient(app) as client:
        response = client.request(method, path, **request_kwargs)

    assert response.status_code == 403


@pytest.mark.parametrize(
    ("app_factory", "auth_path"),
    [
        (create_app, "/api/auth/me"),
    ],
)
def test_auth_me_does_not_require_tenant_claim(
    handler,
    monkeypatch: pytest.MonkeyPatch,
    app_factory,
    auth_path: str,
) -> None:
    app = app_factory(handler)
    app.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        enabled=True,
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    monkeypatch.setattr(
        "overtourism.backend.auth.tokens.dependencies.decode_jwt",
        lambda token, settings: {"sub": "user-1"},
    )

    with TestClient(app) as client:
        response = client.get(
            auth_path,
            headers={"Authorization": "Bearer signed-token"},
        )

    assert response.status_code == 200
    assert "territory" not in response.json()
    assert response.json()["territories"] == ["territory-alpha"]
    assert response.json()["subject"] == "user-1"


def test_auth_settings_from_env_reads_configured_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_ISSUER", "issuer")
    monkeypatch.setenv("AUTH_AUDIENCE", "audience")
    monkeypatch.setenv("AUTH_JWKS_URL", "https://example.com/.well-known/jwks.json")
    monkeypatch.setenv("AUTH_ALGORITHMS", "RS256, ES256")
    monkeypatch.setenv("AUTH_LEEWAY_SECONDS", "45")
    monkeypatch.setenv("AUTH_IDENTIFIER_CLAIM", "tax_id")

    settings = AuthSettings.from_env()

    assert settings == AuthSettings(
        enabled=True,
        issuer="issuer",
        audience="audience",
        jwks_url="https://example.com/.well-known/jwks.json",
        algorithms=("RS256", "ES256"),
        leeway_seconds=45,
        identifier_claim="tax_id",
    )


def test_auth_settings_from_env_defaults_identifier_claim_to_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.delenv("AUTH_IDENTIFIER_CLAIM", raising=False)

    settings = AuthSettings.from_env()

    assert settings.identifier_claim == "email"


def test_create_app_rejects_incomplete_auth_configuration(
    handler: Handler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_JWKS_URL", "https://example.com/.well-known/jwks.json")
    monkeypatch.setenv("AUTH_ISSUER", "issuer")
    monkeypatch.delenv("AUTH_AUDIENCE", raising=False)
    get_auth_settings.cache_clear()

    with pytest.raises(ValueError, match="AUTH_AUDIENCE"):
        create_app(handler)


@pytest.mark.parametrize(
    "missing_setting",
    ["AUTH_JWKS_URL", "AUTH_ISSUER", "AUTH_AUDIENCE"],
)
def test_auth_settings_from_env_requires_validation_settings_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    missing_setting: str,
) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_JWKS_URL", "https://example.com/.well-known/jwks.json")
    monkeypatch.setenv("AUTH_ISSUER", "issuer")
    monkeypatch.setenv("AUTH_AUDIENCE", "audience")
    monkeypatch.delenv(missing_setting)

    with pytest.raises(ValueError, match=missing_setting):
        AuthSettings.from_env()


def test_decode_jwt_requires_jwks_url_when_enabled() -> None:
    with pytest.raises(RuntimeError, match="AUTH_JWKS_URL"):
        auth_jwt.decode_jwt("signed-token", AuthSettings(enabled=True))


@pytest.mark.parametrize(
    ("issuer", "audience", "missing_setting"),
    [
        (None, "audience", "AUTH_ISSUER"),
        ("issuer", None, "AUTH_AUDIENCE"),
    ],
)
def test_decode_jwt_rejects_missing_issuer_or_audience(
    issuer: str | None,
    audience: str | None,
    missing_setting: str,
) -> None:
    with pytest.raises(RuntimeError, match=missing_setting):
        auth_jwt.decode_jwt(
            "signed-token",
            AuthSettings(
                enabled=True,
                issuer=issuer,
                audience=audience,
                jwks_url="https://example.com/.well-known/jwks.json",
            ),
        )


@pytest.mark.parametrize(
    ("audience", "issuer", "verify_aud", "verify_iss"),
    [
        ("audience", "issuer", True, True),
    ],
)
def test_decode_jwt_uses_the_configured_validation_settings(
    monkeypatch: pytest.MonkeyPatch,
    audience: str | None,
    issuer: str | None,
    verify_aud: bool,
    verify_iss: bool,
) -> None:
    captured: dict[str, Any] = {}

    class FakeSigningKey:
        key = "public-key"

    class FakeJwksClient:
        def get_signing_key_from_jwt(self, token: str) -> FakeSigningKey:
            captured["jwks_token"] = token
            return FakeSigningKey()

    def fake_decode(token: str, signing_key: str, **kwargs: Any) -> dict[str, str]:
        captured["token"] = token
        captured["signing_key"] = signing_key
        captured["kwargs"] = kwargs
        return {"sub": "user-1"}

    monkeypatch.setattr(auth_jwt, "_jwks_client", lambda url: FakeJwksClient())
    monkeypatch.setattr(auth_jwt.jwt, "decode", fake_decode)

    claims = auth_jwt.decode_jwt(
        "signed-token",
        AuthSettings(
            enabled=True,
            issuer=issuer,
            audience=audience,
            jwks_url="https://example.com/.well-known/jwks.json",
            algorithms=("RS256", "ES256"),
            leeway_seconds=45,
        ),
    )

    assert claims == {"sub": "user-1"}
    assert captured == {
        "jwks_token": "signed-token",
        "token": "signed-token",
        "signing_key": "public-key",
        "kwargs": {
            "algorithms": ["RS256", "ES256"],
            "leeway": 45,
            "options": {
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": False,
                "verify_aud": verify_aud,
                "verify_iss": verify_iss,
            },
            **({"audience": audience} if audience is not None else {}),
            **({"issuer": issuer} if issuer is not None else {}),
        },
    }
