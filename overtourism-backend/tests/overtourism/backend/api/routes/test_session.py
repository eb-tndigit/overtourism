# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

from overtourism.backend.auth.identity.sql_repository import SQLUserRepository
from overtourism.backend.auth.identity.user_manager import UserManager
from overtourism.backend.auth.identity.users import UserRole
from overtourism.backend.auth.tokens.context import AuthContext
from overtourism.backend.auth.tokens.dependencies import get_auth_context
from overtourism.backend.handler import Handler
from overtourism.dt_manager.manager.manager import Manager
from overtourism.dt_manager.session import manager as session_manager_module
from overtourism.dt_manager.stores.classes.sql.store import SQLStore


def test_delete_all_sessions_removes_only_owned_sessions(
    client,
    manager: Manager,
    territory: str,
    problem_id: str,
) -> None:
    for _ in range(2):
        response = client.post(
            f"/api/{territory}/sessions",
            params={"problem_id": problem_id},
            json={"metadata": {}},
        )
        assert response.status_code == 200

    foreign_session = manager.create_session(
        territory=territory,
        owner_id="other-owner",
    )
    other_territory_session = manager.create_session(
        territory="territory-beta",
        owner_id=f"anonymous:{territory}",
    )

    delete_response = client.delete(
        f"/api/{territory}/sessions",
        params={"problem_id": problem_id},
    )

    assert delete_response.status_code == 200
    assert delete_response.json() == {"message": "All sessions deleted successfully"}
    assert {
        session.session_id for session in manager.session_manager.list_sessions()
    } == {foreign_session.session_id, other_territory_session.session_id}


def test_session_routes_manage_the_full_session_lifecycle(
    client,
    manager: Manager,
    territory: str,
    problem_id: str,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        session_manager_module,
        "uuid4",
        lambda: SimpleNamespace(hex="session-fixed"),
    )

    create_response = client.post(
        f"/api/{territory}/sessions",
        params={},
        json={"metadata": {"source": "ui"}},
    )

    assert create_response.status_code == 200
    assert create_response.json()["session_id"] == "session-fixed"
    assert create_response.json()["metadata"] == {"source": "ui"}
    assert create_response.json()["draft_ids"] == []

    list_response = client.get(
        f"/api/{territory}/sessions",
        params={},
    )

    assert list_response.status_code == 200
    assert [item["session_id"] for item in list_response.json()] == ["session-fixed"]

    read_response = client.get(
        f"/api/{territory}/sessions/session-fixed",
        params={},
    )

    assert read_response.status_code == 200
    assert read_response.json()["drafts"] == []
    assert read_response.json()["evaluations"] == {}

    delete_response = client.delete(
        f"/api/{territory}/sessions/session-fixed",
        params={},
    )

    assert delete_response.status_code == 200
    assert delete_response.json() == {"message": "Session deleted successfully"}
    assert manager.session_manager.list_sessions() == []


def test_session_owner_uses_internal_user_id_and_hides_it_from_response(
    client,
    handler: Handler,
    manager: Manager,
    territory: str,
) -> None:
    store = cast(SQLStore, manager.store)
    repository = SQLUserRepository(store.engine, store.session_factory)
    user_manager = UserManager(repository)
    handler.user_manager = user_manager

    client.app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        authenticated=True,
        territory=territory,
        subject="user-sub",
        token="signed-token",
        claims={"sub": "user-sub", "email": "user@example.com"},
    )

    unregistered_response = client.post(
        f"/api/{territory}/sessions",
        params={},
        json={"metadata": {}},
    )
    assert unregistered_response.status_code == 403

    invited_user = user_manager.create_user(
        identifier="user@example.com",
        role=UserRole.VIEWER,
        territories=["territory-beta"],
    )
    repository.save_user(replace(invited_user, subject="user-sub"))
    user_manager.reload()

    out_of_scope_response = client.post(
        f"/api/{territory}/sessions",
        params={},
        json={"metadata": {}},
    )
    assert out_of_scope_response.status_code == 403

    user_manager.update_user(invited_user.user_id, territories=[territory])
    response = client.post(
        f"/api/{territory}/sessions",
        params={},
        json={"metadata": {}},
    )

    assert response.status_code == 200
    assert "owner_id" not in response.json()
    session = manager.read_session(response.json()["session_id"])
    assert session.owner_id == invited_user.user_id


def test_viewer_can_edit_a_session_draft_but_cannot_save_it(
    client,
    handler: Handler,
    manager: Manager,
    territory: str,
    scenario_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = cast(SQLStore, manager.store)
    repository = SQLUserRepository(store.engine, store.session_factory)
    user_manager = UserManager(repository)
    viewer = user_manager.create_user(
        identifier="viewer@example.org",
        role=UserRole.VIEWER,
        territories=[territory],
    )
    repository.save_user(replace(viewer, subject="viewer-sub"))
    user_manager.reload()
    handler.user_manager = user_manager
    client.app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        authenticated=True,
        subject="viewer-sub",
        token="signed-token",
        claims={"sub": "viewer-sub"},
    )
    monkeypatch.setattr(
        "overtourism.backend.api.routes.session.call_executor",
        lambda territory, param_overrides: {"values": param_overrides},
    )

    session = manager.create_session(territory=territory, owner_id=viewer.user_id)
    draft_response = client.post(
        f"/api/{territory}/sessions/{session.session_id}/scenarios",
        json={
            "base_scenario_id": scenario_id,
            "summary": "Viewer draft",
        },
    )

    assert draft_response.status_code == 200
    draft = draft_response.json()
    assert draft["session_id"] == session.session_id
    assert draft["summary"] == "Viewer draft"

    save_response = client.post(
        f"/api/{territory}/sessions/{session.session_id}/scenarios/{draft['scenario_id']}",
        json={},
    )

    assert save_response.status_code == 403
    assert (
        manager.read_session_scenario(session.session_id, draft["scenario_id"]).summary
        == "Viewer draft"
    )

    evaluation_response = client.post(
        f"/api/{territory}/sessions/{session.session_id}/evaluations",
        json={"scenario_id": draft["scenario_id"]},
    )

    assert evaluation_response.status_code == 200
    assert evaluation_response.json()["session_id"] == session.session_id


def test_admin_cannot_read_or_delete_another_users_session(
    client,
    handler: Handler,
    manager: Manager,
    territory: str,
) -> None:
    store = cast(SQLStore, manager.store)
    repository = SQLUserRepository(store.engine, store.session_factory)
    user_manager = UserManager(repository)
    admin = user_manager.create_first_admin("admin@example.org")
    user_manager.claim_user_by_identifier(admin.identifier, "admin-sub")
    handler.user_manager = user_manager
    client.app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        authenticated=True,
        subject="admin-sub",
        token="signed-token",
        claims={"sub": "admin-sub"},
    )

    session = manager.create_session(
        territory=territory,
        owner_id="another-user-id",
    )

    read_response = client.get(f"/api/{territory}/sessions/{session.session_id}")
    delete_response = client.delete(f"/api/{territory}/sessions/{session.session_id}")

    assert read_response.status_code == 404
    assert delete_response.status_code == 404
    assert manager.read_session(session.session_id).owner_id == "another-user-id"


def test_expired_session_is_rejected_before_periodic_cleanup(
    client,
    manager: Manager,
    territory: str,
) -> None:
    session = manager.create_session(
        territory=territory,
        owner_id=f"anonymous:{territory}",
    )
    session_data = manager.store.load_session(session.session_id)
    session_data["created"] = "2000-01-01T00:00:00Z"
    manager.store.save_session(session_data)

    response = client.get(f"/api/{territory}/sessions/{session.session_id}")

    assert response.status_code == 404
    assert response.json() == {"detail": f"Session '{session.session_id}' not found."}
    assert manager.store.load_session(session.session_id)["session_id"] == (
        session.session_id
    )


def test_session_detail_embeds_evaluation_metadata_without_result(
    client,
    manager: Manager,
    territory: str,
    problem_id: str,
    scenario_id: str,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "overtourism.backend.api.routes.session.call_executor",
        lambda territory, param_overrides: {"values": param_overrides},
    )
    monkeypatch.setattr(
        session_manager_module,
        "uuid4",
        lambda: SimpleNamespace(hex="session-detail"),
    )

    create_session_response = client.post(
        f"/api/{territory}/sessions",
        params={"problem_id": problem_id},
        json={"metadata": {"source": "ui"}},
    )
    assert create_session_response.status_code == 200

    draft_response = client.post(
        f"/api/{territory}/sessions/session-detail/scenarios",
        params={"problem_id": problem_id},
        json={
            "base_scenario_id": scenario_id,
            "param_overrides": {"visits": 5},
            "name": "Session draft",
        },
    )
    assert draft_response.status_code == 200
    draft_id = draft_response.json()["scenario_id"]

    evaluation_response = client.post(
        f"/api/{territory}/sessions/session-detail/evaluations",
        params={"problem_id": problem_id},
        json={"scenario_id": draft_id, "ensemble_size": 4},
    )
    assert evaluation_response.status_code == 200
    evaluation = evaluation_response.json()
    stored_evaluation = manager.read_evaluation(evaluation["evaluation_id"])
    stored_evaluation.result = None
    manager.save_evaluation(stored_evaluation)

    response = client.get(
        f"/api/{territory}/sessions/session-detail",
        params={"problem_id": problem_id},
    )

    assert response.status_code == 200
    evaluation_data = response.json()["evaluations"][draft_id]
    assert evaluation_data["evaluation_id"] == evaluation["evaluation_id"]
    assert evaluation_data["scenario_id"] == draft_id
    assert evaluation_data["session_id"] == "session-detail"
    assert evaluation_data["type"] == "default"
    assert evaluation_data["version"] == evaluation["version"]
    assert evaluation_data["state"] == "COMPLETED"
    assert evaluation_data["started"] == evaluation["started"]
    assert evaluation_data["finished"] == evaluation["finished"]

    data_response = client.get(
        f"/api/{territory}/sessions/session-detail/evaluations/"
        f"{evaluation['evaluation_id']}/data",
        params={"problem_id": problem_id},
    )

    assert data_response.status_code == 200
    assert data_response.json()["data"] == {}
