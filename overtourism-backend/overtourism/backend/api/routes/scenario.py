# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from overtourism.backend.api.models.scenario import (
    CreateScenarioData,
    ScenarioData,
    UpdateScenarioData,
)
from overtourism.backend.api.utils.utils import (
    check_version,
    get_scenario_or_404,
    raise_immutable_base_scenario_error,
    scenario_to_api,
    scenarios_to_api,
)
from overtourism.backend.auth.identity.authorization import (
    require_persistent_write_access,
    require_territory_access,
)
from overtourism.backend.handler import Handler, get_handler
from overtourism.backend.utils.config import TERRITORY_ROUTE_PREFIX
from overtourism.dt_manager.manager.config import BootstrapConfig

logger = logging.getLogger(__name__)

scenario_router = APIRouter(
    prefix=f"{TERRITORY_ROUTE_PREFIX}/scenarios",
    tags=["Scenarios"],
    dependencies=[Depends(require_territory_access)],
)


@scenario_router.get(
    "",
    response_model=ScenarioData | list[ScenarioData],
    responses={
        500: {"description": "Scenario manager error"},
        404: {"description": "Problem does not exist"},
        200: {"description": "Scenario list or base scenario"},
    },
)
async def list_scenarios(
    territory: str,
    proposal_id: str | None = None,
    base_only: bool = False,
    *,
    handler: Annotated[Handler, Depends(get_handler)],
) -> dict[str, Any] | list[dict[str, Any]]:
    try:
        if base_only:
            names_cfg = BootstrapConfig(territory)
            scenario = get_scenario_or_404(territory, handler, names_cfg.scenario_id)
            return scenario_to_api(scenario)
        scenarios = handler.manager.list_scenarios(
            territory=territory, proposal_id=proposal_id
        )
        base_scenario_id = BootstrapConfig(territory).scenario_id
        stored_scenarios = [
            scenario for scenario in scenarios if scenario.session_id is None
        ]
        stored_scenarios.sort(
            key=lambda scenario: (
                scenario.scenario_id != base_scenario_id,
                (scenario.name or "").casefold(),
                scenario.scenario_id.casefold(),
            )
        )
        return scenarios_to_api(stored_scenarios)
    except Exception as e:
        logger.error(f"Error listing scenarios: {e}")
        raise


@scenario_router.get(
    "/{scenario_id}",
    response_model=ScenarioData,
    responses={
        500: {"description": "Scenario manager error"},
        404: {"description": "Scenario does not exist"},
        200: {"description": "Scenario details"},
    },
)
async def read_scenario(
    territory: str,
    scenario_id: str,
    handler: Annotated[Handler, Depends(get_handler)],
) -> dict[str, Any]:
    try:
        scenario = get_scenario_or_404(territory, handler, scenario_id)
        return scenario_to_api(scenario)
    except Exception as e:
        logger.error(f"Error reading scenario {scenario_id}: {e}")
        raise


@scenario_router.post(
    "",
    response_model=ScenarioData,
    responses={
        500: {"description": "Scenario manager error"},
        404: {"description": "Problem does not exist"},
        200: {"description": "Scenario created"},
    },
    dependencies=[Depends(require_persistent_write_access)],
)
async def create_scenario(
    territory: str,
    data: CreateScenarioData,
    handler: Annotated[Handler, Depends(get_handler)],
) -> dict[str, Any]:
    try:
        scenario_payload = data.model_dump(exclude_unset=True)
        scenario = handler.manager.create_scenario(
            territory=territory, **scenario_payload
        )
        logger.info(f"Scenario created: {scenario.scenario_id}")
        return scenario_to_api(scenario)
    except Exception as e:
        logger.error(f"Error creating scenario: {e}")
        raise


@scenario_router.put(
    "/{scenario_id}",
    response_model=ScenarioData,
    responses={
        500: {"description": "Scenario manager error"},
        404: {"description": "Scenario does not exist"},
        400: {"description": "Base scenario cannot be updated"},
        200: {"description": "Scenario updated"},
    },
    dependencies=[Depends(require_persistent_write_access)],
)
async def update_scenario(
    territory: str,
    scenario_id: str,
    data: UpdateScenarioData,
    handler: Annotated[Handler, Depends(get_handler)],
) -> dict[str, Any]:
    try:
        raise_immutable_base_scenario_error(handler, territory, scenario_id)
        current_scenario = get_scenario_or_404(territory, handler, scenario_id)
        check_version(current_scenario.version, data.version)
        handler.manager.update_scenario(
            scenario_id,
            **data.model_dump(exclude_none=True, exclude={"version"}),
        )
        scenario = handler.manager.read_scenario(scenario_id)
        logger.info(f"Scenario updated: {scenario_id}")
        return scenario_to_api(scenario)
    except Exception as e:
        logger.error(f"Error updating scenario {scenario_id}: {e}")
        raise


@scenario_router.delete(
    "/{scenario_id}",
    responses={
        500: {"description": "Scenario manager error"},
        404: {"description": "Scenario does not exist"},
        400: {"description": "Base scenario cannot be deleted"},
        200: {"description": "Scenario deleted"},
    },
    dependencies=[Depends(require_persistent_write_access)],
)
async def delete_scenario(
    territory: str,
    scenario_id: str,
    handler: Annotated[Handler, Depends(get_handler)],
) -> dict:
    try:
        raise_immutable_base_scenario_error(handler, territory, scenario_id)
        get_scenario_or_404(territory, handler, scenario_id)
        handler.manager.delete_scenario(scenario_id)
        logger.info(f"Scenario deleted: {scenario_id}")
        return {"message": "Scenario deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting scenario {scenario_id}: {e}")
        raise
