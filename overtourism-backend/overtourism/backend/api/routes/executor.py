# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import requests
from fastapi import APIRouter, Depends, HTTPException

from overtourism.backend.api.models.configuration import ModelSchema
from overtourism.backend.auth.identity.authorization import (
    require_global_admin,
    require_persistent_write_access,
    require_territory_access,
)
from overtourism.backend.auth.tokens.context import AuthContext
from overtourism.backend.auth.tokens.dependencies import get_auth_context
from overtourism.backend.handler import Handler, get_handler
from overtourism.backend.utils.config import TERRITORY_ROUTE_PREFIX
from overtourism.backend.utils.executor_utils import call_download_data, call_schema
from overtourism.backend.utils.summaries import update_scenario_summaries

logger = logging.getLogger(__name__)

configuration_router = APIRouter(
    prefix=TERRITORY_ROUTE_PREFIX,
    dependencies=[Depends(require_territory_access)],
)


@configuration_router.get(
    "/configuration",
    response_model=ModelSchema,
    responses={
        500: {"description": "View manager error"},
        200: {"description": "Configuration api"},
    },
)
async def get_configuration(
    territory: str,
) -> dict[str, Any]:
    """List all available configuration for the given territory."""
    try:
        return call_schema(territory)
    except Exception as exc:
        logger.error(f"Error listing configuration: {exc}")
        raise


@configuration_router.post(
    "/configuration/download-data",
    response_model=dict[str, str],
    dependencies=[Depends(require_global_admin)],
)
def download_model_data(territory: str) -> dict[str, str]:
    """Download model data through the Model API; admins only."""
    try:
        return call_download_data(territory)
    except requests.HTTPError as exc:
        response = exc.response
        status_code = 502 if response is None else response.status_code
        detail = "Model API data download failed"
        if response is not None:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                detail = str(payload.get("detail", detail))
        logger.error("Model API data download failed: %s", detail)
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except requests.RequestException as exc:
        logger.error("Model API data download request failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Model API data download request failed",
        ) from exc


@configuration_router.post(
    "/configuration/update/summaries",
    response_model=dict[str, Any],
    dependencies=[Depends(require_persistent_write_access)],
    description=(
        "Regenerate and persist the summaries for every stored scenario in this "
        "territory. The agent backend is meant to be active; if it is not present, "
        "the update is skipped."
    ),
)
def update_configuration_summaries(
    territory: str,
    handler: Handler = Depends(get_handler),
    context: AuthContext = Depends(get_auth_context),
) -> dict[str, Any]:
    """Update stored scenario summaries using the agent backend."""
    if context.token is None:
        raise HTTPException(
            status_code=401,
            detail="A bearer token is required to call the agent backend",
        )
    return update_scenario_summaries(
        handler.manager.scenario_manager,
        territory,
        context.token,
    )
