# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import requests
from overtourism.backend.api.utils import utils as api_utils
from overtourism.backend.auth.identity.authorization import require_territory_access
from overtourism.backend.auth.tokens.context import AuthContext
from overtourism.backend.auth.tokens.dependencies import get_auth_context
from overtourism.dt_manager.manager.manager import Manager


def test_list_and_read_stored_scenarios(
    client, territory: str, problem_id: str
) -> None:
    base_scenario_id = f"{territory}_base_scenario"
    list_response = client.get(
        f"/api/{territory}/scenarios",
    )

    assert list_response.status_code == 200
    assert [item["scenario_id"] for item in list_response.json()] == [base_scenario_id]

    read_response = client.get(
        f"/api/{territory}/scenarios/{base_scenario_id}",
    )

    assert read_response.status_code == 200
    assert read_response.json()["scenario_id"] == base_scenario_id
    assert read_response.json()["version"] == 1
    assert read_response.json()["summary"] is None


def test_list_scenarios_orders_base_first_then_alphabetically(
    client,
    territory: str,
    manager: Manager,
) -> None:
    manager.scenario_manager.create_scenario(
        "scenario-zulu",
        territory,
        name="Zulu",
    )
    manager.scenario_manager.create_scenario(
        "scenario-bravo",
        territory,
        name="bravo",
    )
    manager.scenario_manager.create_scenario(
        "scenario-alpha",
        territory,
        name="Alpha",
    )

    response = client.get(f"/api/{territory}/scenarios")

    assert response.status_code == 200
    assert [scenario["scenario_id"] for scenario in response.json()] == [
        f"{territory}_base_scenario",
        "scenario-alpha",
        "scenario-bravo",
        "scenario-zulu",
    ]


def test_list_scenarios_can_return_only_the_base_scenario(
    client,
    territory: str,
    problem_id: str,
    manager: Manager,
) -> None:
    base_scenario_id = f"{territory}_base_scenario"
    extra_scenario = manager.scenario_manager.create_scenario(
        "scenario-extra",
        territory,
        param_overrides={"visits": 3},
        name="Extra scenario",
    )

    response = client.get(
        f"/api/{territory}/scenarios",
        params={"problem_id": problem_id, "base_only": True},
    )

    assert response.status_code == 200
    assert response.json()["scenario_id"] == base_scenario_id
    assert response.json()["scenario_id"] != extra_scenario.scenario_id


def test_scenario_routes_expose_index_diffs_in_extras(
    client,
    territory: str,
    problem_id: str,
    monkeypatch,
) -> None:
    base_scenario_id = f"{territory}_base_scenario"
    monkeypatch.setattr(
        api_utils,
        "scenario_index_diffs",
        lambda scenarios: {
            scenario.scenario_id: {"visits": "+3"} for scenario in scenarios
        },
    )

    list_response = client.get(
        f"/api/{territory}/scenarios",
        params={"problem_id": problem_id},
    )
    read_response = client.get(
        f"/api/{territory}/scenarios/{base_scenario_id}",
        params={"problem_id": problem_id},
    )

    assert list_response.status_code == 200
    assert list_response.json()[0]["extras"]["index_diffs"] == {"visits": "+3"}
    assert read_response.status_code == 200
    assert read_response.json()["extras"]["index_diffs"] == {"visits": "+3"}


def test_create_stored_scenario_persists_values_and_metadata(
    client,
    territory: str,
) -> None:
    response = client.post(
        f"/api/{territory}/scenarios",
        json={
            "name": "Created through API",
            "description": "Stored scenario payload",
            "summary": "A concise scenario summary",
            "param_overrides": {"visits": 12},
            "extras": {"channel": "api"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scenario_id"]
    assert payload["name"] == "Created through API"
    assert payload["description"] == "Stored scenario payload"
    assert payload["summary"] == "A concise scenario summary"
    assert "summary" not in payload["extras"]
    assert payload["extras"]["channel"] == "api"
    assert payload["param_overrides"] == {"visits": 12}

    read_response = client.get(f"/api/{territory}/scenarios/{payload['scenario_id']}")

    assert read_response.status_code == 200
    assert read_response.json()["summary"] == "A concise scenario summary"


def test_update_stored_scenario_persists_top_level_summary(
    client,
    territory: str,
    problem_id: str,
) -> None:
    create_response = client.post(
        f"/api/{territory}/scenarios",
        json={"summary": "Initial summary"},
    )
    assert create_response.status_code == 200
    scenario_id = create_response.json()["scenario_id"]

    update_response = client.put(
        f"/api/{territory}/scenarios/{scenario_id}",
        params={"problem_id": problem_id},
        json={
            "version": create_response.json()["version"],
            "summary": "Updated summary",
        },
    )

    assert update_response.status_code == 200
    assert update_response.json()["summary"] == "Updated summary"
    assert "summary" not in update_response.json()["extras"]


def test_update_configuration_summaries_calls_agent_for_each_scenario(
    client,
    handler,
    territory: str,
    monkeypatch,
) -> None:
    second_scenario_id = "scenario-summary-second"
    handler.manager.scenario_manager.create_scenario(
        second_scenario_id,
        territory,
        name="Second scenario",
    )
    client.app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        authenticated=True,
        territory=territory,
        subject="test-subject",
        token="agent-access-token",
    )
    client.app.dependency_overrides[require_territory_access] = lambda: None
    calls = []

    class AgentResponse:
        def __init__(self, scenario_id: str) -> None:
            self.scenario_id = scenario_id

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {"message": f"Summary for {self.scenario_id}"}

    def post(url, *, params, json, headers, timeout):
        calls.append((url, params, json, headers, timeout))
        return AgentResponse(json["current_context"]["scenarios"][0])

    monkeypatch.setattr(
        "overtourism.backend.utils.summaries.requests.post",
        post,
    )

    response = client.post(f"/api/{territory}/configuration/update/summaries")

    assert response.status_code == 200
    assert set(response.json()["updated_scenario_ids"]) == {
        f"{territory}_base_scenario",
        second_scenario_id,
    }
    assert response.json()["skipped"] is False
    assert response.json()["updated_summaries"] == {
        f"{territory}_base_scenario": f"Summary for {territory}_base_scenario",
        second_scenario_id: f"Summary for {second_scenario_id}",
    }
    assert {call[2]["current_context"]["scenarios"][0] for call in calls} == {
        f"{territory}_base_scenario",
        second_scenario_id,
    }
    assert all(
        call[3] == {"Authorization": "Bearer agent-access-token"} for call in calls
    )
    assert (
        handler.manager.scenario_manager.read_scenario(
            f"{territory}_base_scenario"
        ).summary
        == f"Summary for {territory}_base_scenario"
    )
    assert handler.manager.scenario_manager.read_scenario(
        second_scenario_id
    ).summary == (f"Summary for {second_scenario_id}")


def test_update_configuration_summaries_skips_when_agent_backend_is_absent(
    client,
    territory: str,
    monkeypatch,
) -> None:
    client.app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        authenticated=True,
        territory=territory,
        subject="test-subject",
        token="agent-access-token",
    )
    client.app.dependency_overrides[require_territory_access] = lambda: None

    def post(*args, **kwargs):
        raise requests.ConnectionError("agent backend is unavailable")

    monkeypatch.setattr(
        "overtourism.backend.utils.summaries.requests.post",
        post,
    )

    response = client.post(f"/api/{territory}/configuration/update/summaries")

    assert response.status_code == 200
    assert response.json() == {
        "updated_scenario_ids": [],
        "updated_summaries": {},
        "skipped": True,
    }


def test_list_stored_scenarios_can_filter_by_related_proposal(
    client,
    manager: Manager,
    territory: str,
    problem_id: str,
    proposal_id: str,
) -> None:
    related_scenario = manager.scenario_manager.create_scenario(
        "scenario-related",
        territory,
        param_overrides={"visits": 4},
        name="Related scenario",
    )
    manager.link_scenario_to_proposal(proposal_id, related_scenario.scenario_id)

    response = client.get(
        f"/api/{territory}/scenarios",
        params={"problem_id": problem_id, "proposal_id": proposal_id},
    )

    assert response.status_code == 200
    assert {item["scenario_id"] for item in response.json()} == {
        related_scenario.scenario_id,
        f"{territory}_base_scenario",
    }


def test_list_scenarios_excludes_session_scenarios(
    client,
    territory: str,
    problem_id: str,
) -> None:
    base_scenario_id = f"{territory}_base_scenario"
    session_response = client.post(
        f"/api/{territory}/sessions",
        params={"problem_id": problem_id},
        json={"metadata": {}},
    )
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    draft_response = client.post(
        f"/api/{territory}/sessions/{session_id}/scenarios",
        params={"problem_id": problem_id},
        json={"base_scenario_id": base_scenario_id},
    )
    assert draft_response.status_code == 200
    draft_id = draft_response.json()["scenario_id"]

    response = client.get(f"/api/{territory}/scenarios")

    assert response.status_code == 200
    assert draft_id not in {item["scenario_id"] for item in response.json()}
    assert base_scenario_id in {item["scenario_id"] for item in response.json()}


def test_update_stored_scenario_requires_the_current_version(
    client,
    territory: str,
    problem_id: str,
) -> None:
    base_scenario_id = f"{territory}_base_scenario"
    missing_version = client.put(
        f"/api/{territory}/scenarios/{base_scenario_id}",
        params={"problem_id": problem_id},
        json={"name": "Updated default", "param_overrides": {"visits": 11}},
    )

    assert missing_version.status_code == 400
    assert missing_version.json() == {
        "detail": "Base scenario cannot be modified or deleted."
    }


def test_session_scenario_can_be_created_updated_and_saved(
    client,
    manager: Manager,
    territory: str,
    problem_id: str,
    proposal_id: str,
) -> None:
    base_scenario_id = f"{territory}_base_scenario"
    session_response = client.post(
        f"/api/{territory}/sessions",
        params={"problem_id": problem_id},
        json={"metadata": {}},
    )
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]
    create_response = client.post(
        f"/api/{territory}/sessions/{session_id}/scenarios",
        params={"problem_id": problem_id},
        json={
            "base_scenario_id": base_scenario_id,
            "name": "Draft scenario",
            "description": "Scenario under discussion",
            "summary": "Draft summary",
            "param_overrides": {"visits": 7},
            "extras": {"stage": "draft"},
        },
    )

    assert create_response.status_code == 200
    draft_id = create_response.json()["scenario_id"]
    assert draft_id
    assert create_response.json()["version"] == 1
    assert create_response.json()["summary"] == "Draft summary"

    read_response = client.get(
        f"/api/{territory}/sessions/{session_id}/scenarios/{draft_id}",
        params={"problem_id": problem_id},
    )

    assert read_response.status_code == 200
    assert read_response.json()["extras"] == {"index_diffs": {}}

    update_response = client.put(
        f"/api/{territory}/sessions/{session_id}/scenarios/{draft_id}",
        params={"problem_id": problem_id},
        json={
            "version": 1,
            "name": "Draft scenario updated",
            "param_overrides": {"visits": 9},
            "extras": {"stage": "review"},
        },
    )

    assert update_response.status_code == 405

    save_response = client.post(
        f"/api/{territory}/sessions/{session_id}/scenarios/{draft_id}",
        params={"problem_id": problem_id},
        json={
            "version": 2,
            "name": "Saved scenario",
            "description": "Persisted after review",
            "summary": "Published summary",
            "proposal_id": proposal_id,
        },
    )

    assert save_response.status_code == 200
    assert save_response.json()["scenario_id"] == draft_id
    assert save_response.json()["session_id"] is None
    assert save_response.json()["version"] == 1
    assert save_response.json()["name"] == "Saved scenario"
    assert save_response.json()["summary"] == "Published summary"
    public_scenarios_response = client.get(f"/api/{territory}/scenarios")
    assert public_scenarios_response.status_code == 200
    assert draft_id in {
        item["scenario_id"] for item in public_scenarios_response.json()
    }
    assert (
        manager.relationship_manager.get_related_scenario_ids(proposal_id)[-1]
        == draft_id
    )
    assert session_id in {
        session.session_id for session in manager.session_manager.list_sessions()
    }


def test_session_scenario_routes_expose_index_diffs_in_extras(
    client,
    territory: str,
    problem_id: str,
    monkeypatch,
) -> None:
    base_scenario_id = f"{territory}_base_scenario"
    session_response = client.post(
        f"/api/{territory}/sessions",
        params={"problem_id": problem_id},
        json={"metadata": {}},
    )
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]
    monkeypatch.setattr(
        api_utils,
        "scenario_index_diffs",
        lambda scenarios: {
            scenario.scenario_id: {"visits": "+7"} for scenario in scenarios
        },
    )

    create_response = client.post(
        f"/api/{territory}/sessions/{session_id}/scenarios",
        params={"problem_id": problem_id},
        json={
            "base_scenario_id": base_scenario_id,
            "name": "Draft scenario",
            "param_overrides": {"visits": 7},
            "extras": {"stage": "draft"},
        },
    )

    assert create_response.status_code == 200
    draft_id = create_response.json()["scenario_id"]
    assert create_response.json()["extras"] == {
        "index_diffs": {"visits": "+7"},
    }

    read_response = client.get(
        f"/api/{territory}/sessions/{session_id}/scenarios/{draft_id}",
        params={"problem_id": problem_id},
    )

    assert read_response.status_code == 200
    assert read_response.json()["extras"] == {"index_diffs": {"visits": "+7"}}


def test_session_scenario_can_be_deleted_without_affecting_stored_scenarios(
    client,
    territory: str,
    problem_id: str,
    manager: Manager,
) -> None:
    base_scenario_id = f"{territory}_base_scenario"
    session_response = client.post(
        f"/api/{territory}/sessions",
        params={"problem_id": problem_id},
        json={"metadata": {}},
    )
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]
    draft_response = client.post(
        f"/api/{territory}/sessions/{session_id}/scenarios",
        params={"problem_id": problem_id},
        json={
            "base_scenario_id": base_scenario_id,
            "param_overrides": {"visits": 3},
            "name": "Disposable draft",
        },
    )
    assert draft_response.status_code == 200
    draft_id = draft_response.json()["scenario_id"]

    delete_response = client.request(
        "DELETE",
        f"/api/{territory}/sessions/{session_id}/scenarios/{draft_id}",
        params={"problem_id": problem_id},
        json={"version": 1},
    )

    assert delete_response.status_code == 405
    assert [
        scenario.scenario_id for scenario in manager.list_session_scenarios(session_id)
    ] == [draft_id]


def test_delete_stored_scenario_removes_it_from_the_problem(
    client,
    error_client,
    territory: str,
    problem_id: str,
    manager: Manager,
) -> None:
    scenario = manager.scenario_manager.create_scenario(
        "scenario-delete",
        territory,
        param_overrides={"visits": 5},
        name="Scenario delete",
    )
    scenario.version = 2
    manager.store.save_scenario(scenario.to_dict())

    delete_response = error_client.request(
        "DELETE",
        f"/api/{territory}/scenarios/{scenario.scenario_id}",
        params={"problem_id": problem_id},
        json={"version": 2},
    )

    assert delete_response.status_code == 200
    assert scenario.scenario_id not in {
        item.scenario_id for item in manager.list_scenarios(territory)
    }
