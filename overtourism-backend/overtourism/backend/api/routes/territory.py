# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from overtourism.backend.auth.identity.authorization import resolve_current_user
from overtourism.backend.auth.identity.users import UserRole
from overtourism.backend.auth.tokens.context import AuthContext
from overtourism.backend.auth.tokens.dependencies import get_auth_context
from overtourism.backend.auth.tokens.settings import AuthSettings, get_auth_settings
from overtourism.backend.handler import Handler, get_handler
from overtourism.backend.utils.config import BASE_ROUTE
from overtourism.backend.utils.executor_utils import list_models

territory_router = APIRouter(
    prefix=f"{BASE_ROUTE}/default",
    tags=["Territories"],
)


@territory_router.get("/territories", response_model=list[str])
async def list_territories(
    context: Annotated[AuthContext, Depends(get_auth_context)],
    handler: Annotated[Handler, Depends(get_handler)],
    auth_settings: Annotated[AuthSettings, Depends(get_auth_settings)],
) -> list[str]:
    models: list[dict[str, Any]] = list_models()
    model_keys = [str(model["key"]) for model in models]

    if not context.authenticated:
        return model_keys

    user = resolve_current_user(context, handler, auth_settings)
    if user is None:
        return []
    if user.role in (UserRole.ADMIN, UserRole.MULTIEDITOR):
        return model_keys

    return [key for key in model_keys if key in user.territories]
