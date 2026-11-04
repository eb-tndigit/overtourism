# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from overtourism.dt_manager.scenario.manager import ScenarioManager

logger = logging.getLogger(__name__)

DEFAULT_AGENT_BACKEND_URL = "http://localhost:9000"
AGENT_REQUEST_TIMEOUT_SECONDS = 60


def update_scenario_summaries(
    scenario_manager: ScenarioManager,
    territory: str,
    authorization_token: str,
) -> dict[str, Any]:
    """Generate and persist a summary for each stored scenario in a territory."""
    agent_backend_url = os.getenv("AGENT_BACKEND_URL", DEFAULT_AGENT_BACKEND_URL)
    updated_scenario_ids: list[str] = []
    updated_summaries: dict[str, str] = {}

    scenarios = [
        scenario
        for scenario in scenario_manager.list_scenarios(territory=territory)
        if scenario.session_id is None
    ]
    for scenario in scenarios:
        try:
            response = requests.post(
                f"{agent_backend_url.rstrip('/')}/agent/tool",
                params={"territory": territory},
                json={
                    "key": "summary",
                    "current_context": {"scenarios": [scenario.scenario_id]},
                },
                headers={"Authorization": f"Bearer {authorization_token}"},
                timeout=AGENT_REQUEST_TIMEOUT_SECONDS,
            )
        except requests.ConnectionError:
            logger.info(
                "Agent backend is not present; skipping summary updates for territory %s",
                territory,
            )
            return {
                "updated_scenario_ids": updated_scenario_ids,
                "updated_summaries": updated_summaries,
                "skipped": True,
            }

        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError(
                f"Agent backend returned invalid JSON for scenario {scenario.scenario_id}"
            ) from exc
        summary = (
            payload.get("summary", payload.get("message"))
            if isinstance(payload, dict)
            else None
        )
        if not isinstance(summary, str):
            raise ValueError(
                f"Agent backend response for scenario {scenario.scenario_id} "
                "must contain a string 'message' or 'summary'"
            )

        scenario_manager.update_scenario(
            scenario.scenario_id,
            summary=summary,
        )
        persisted_scenario = scenario_manager.read_scenario(scenario.scenario_id)
        if persisted_scenario.summary != summary:
            raise RuntimeError(
                f"Summary update for scenario {scenario.scenario_id} was not persisted"
            )
        updated_scenario_ids.append(scenario.scenario_id)
        updated_summaries[scenario.scenario_id] = summary

    return {
        "updated_scenario_ids": updated_scenario_ids,
        "updated_summaries": updated_summaries,
        "skipped": False,
    }
