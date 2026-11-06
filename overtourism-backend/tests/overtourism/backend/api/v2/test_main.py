# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

from fastapi import APIRouter
from fastapi.testclient import TestClient
from overtourism.backend.main import create_app
from overtourism.dt_manager.session.config import SessionCleanupConfig


def test_create_app_includes_extra_routers_and_metadata(handler) -> None:
    extra_router = APIRouter()

    @extra_router.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    app = create_app(
        handler,
        title="Custom API",
        version="9.9.9",
        description="Custom description",
        extra_routers=[extra_router],
    )

    with TestClient(app) as client:
        openapi_response = client.get("/openapi.json")
        ping_response = client.get("/ping")

    assert openapi_response.status_code == 200
    assert openapi_response.json()["info"] == {
        "title": "Custom API",
        "version": "9.9.9",
        "description": "Custom description",
    }
    assert ping_response.status_code == 200
    assert ping_response.json() == {"status": "ok"}


def test_create_app_does_not_register_overtourism_extension_routes_by_default(
    handler,
) -> None:
    app = create_app(handler)

    paths = app.openapi()["paths"]

    assert "/api/{territory}/data/overtourism/indexes/categories" not in paths
    assert "/api/{territory}/widgets" not in paths


def test_create_app_registers_configuration_route_by_default(handler) -> None:
    app = create_app(handler)

    paths = app.openapi()["paths"]

    assert "/api/{territory}/configuration" in paths
    assert not any(path.startswith("/api/v2/") for path in paths)


def test_indexes_router_registers_under_default_api_namespace(handler) -> None:
    from overtourism.overtourism.backend_extension.api.routes.indexes import (
        indexes_router,
    )

    app = create_app(handler, extra_routers=[indexes_router])

    assert "/api/indexes/get-index-list" in app.openapi()["paths"]


def test_create_app_exposes_bearer_auth_in_openapi(handler) -> None:
    app = create_app(handler)

    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    openapi = response.json()

    assert openapi["components"]["securitySchemes"]["BearerAuth"] == {
        "type": "http",
        "scheme": "bearer",
    }
    assert openapi["paths"]["/api/{territory}/problems"]["get"]["security"] == [
        {"BearerAuth": []}
    ]
    assert openapi["paths"]["/api/auth/me"]["get"]["security"] == [{"BearerAuth": []}]
    assert "/api/{territory}/auth/me" not in openapi["paths"]


def test_create_app_groups_routes_by_domain_tags_in_openapi(handler) -> None:
    app = create_app(handler)

    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    openapi = response.json()

    assert openapi["tags"] == [
        {
            "name": "Problems",
            "description": "Create and manage optimization problems.",
        },
        {
            "name": "Proposals",
            "description": "Manage proposals linked to problems.",
        },
        {
            "name": "Sessions",
            "description": "Create and manage sessions.",
        },
        {
            "name": "Scenarios",
            "description": "Inspect and update scenarios within a problem.",
        },
        {
            "name": "Evaluations",
            "description": "Run and inspect scenario evaluations.",
        },
        {
            "name": "Auth",
            "description": "Authentication and current user context.",
        },
        {
            "name": "Territories",
            "description": "List territories available to the current user.",
        },
        {
            "name": "Health",
            "description": "Operational liveness and readiness probes.",
        },
    ]
    assert openapi["paths"]["/api/{territory}/problems"]["get"]["tags"] == ["Problems"]
    assert openapi["paths"]["/api/{territory}/proposals"]["get"]["tags"] == [
        "Proposals"
    ]
    assert openapi["paths"]["/api/{territory}/sessions"]["post"]["tags"] == ["Sessions"]
    assert openapi["paths"]["/api/{territory}/scenarios"]["get"]["tags"] == [
        "Scenarios"
    ]
    assert openapi["paths"]["/api/{territory}/evaluations"]["post"]["tags"] == [
        "Evaluations"
    ]
    assert openapi["paths"]["/api/auth/me"]["get"]["tags"] == ["Auth"]


def test_app_lifespan_runs_and_stops_session_cleanup_worker(
    handler,
    monkeypatch,
) -> None:
    cleanup_config = SessionCleanupConfig(
        session_scenario_ttl_seconds=604800,
        session_cleanup_interval_seconds=0.01,
    )
    handler.manager.session_manager.cleanup_config = cleanup_config
    cleanup_called = threading.Event()
    monkeypatch.setattr(
        handler.manager.session_manager,
        "delete_expired_sessions",
        lambda: cleanup_called.set() or 0,
    )
    app = create_app(handler)

    with TestClient(app):
        assert cleanup_called.wait(timeout=2)

    assert not any(
        thread.name == "session-scenario-cleanup" and thread.is_alive()
        for thread in threading.enumerate()
    )
