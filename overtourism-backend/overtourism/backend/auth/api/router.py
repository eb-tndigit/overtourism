# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError

from overtourism.backend.auth.api.models import (
    AuthMeResponse,
    AuthRoleResponse,
    AuthUserResponse,
    CreateAuthUserRequest,
    UpdateAuthUserRequest,
)
from overtourism.backend.auth.identity.authorization import (
    get_current_user,
    get_user_manager,
    require_global_admin,
    resolve_current_user,
)
from overtourism.backend.auth.identity.user_manager import UserManager
from overtourism.backend.auth.identity.users import User, UserRole
from overtourism.backend.auth.tokens.context import AuthContext
from overtourism.backend.auth.tokens.dependencies import get_auth_context
from overtourism.backend.auth.tokens.settings import AuthSettings, get_auth_settings
from overtourism.backend.handler import Handler, get_handler
from overtourism.backend.utils.executor_utils import list_models

auth_router = APIRouter(prefix="/auth")

_ROLE_DESCRIPTIONS = {
    UserRole.ADMIN: "Amministratore globale con accesso a tutti i territori.",
    UserRole.MULTIEDITOR: "Editor assegnato a tutti i territori per impostazione predefinita.",
    UserRole.EDITOR: "Editor con accesso a un insieme di territori, anche vuoto.",
    UserRole.VIEWER: "Utente di sola lettura assegnato a un insieme di territori, anche vuoto.",
}


def _all_model_territories() -> list[str]:
    return sorted({str(model["key"]) for model in list_models()})


def _user_response(user: User) -> AuthUserResponse:
    return AuthUserResponse(
        user_id=user.user_id,
        identifier=user.identifier,
        subject=user.subject,
        role=user.role,
        is_active=user.is_active,
        territories=sorted(user.territories),
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


@auth_router.get("/me", response_model=AuthMeResponse)
async def read_auth_me(
    context: Annotated[AuthContext, Depends(get_auth_context)],
    handler: Annotated[Handler, Depends(get_handler)],
    auth_settings: Annotated[AuthSettings, Depends(get_auth_settings)],
) -> AuthMeResponse:
    user = resolve_current_user(context, handler, auth_settings)
    if user is None:
        territories = []
    elif user.role is UserRole.ADMIN:
        territories = _all_model_territories()
    else:
        territories = sorted(user.territories)
    return AuthMeResponse(
        authenticated=context.authenticated,
        subject=context.subject,
        user_id=None if user is None else user.user_id,
        identifier=None if user is None else user.identifier,
        role=None if user is None else user.role,
        territories=territories,
    )


@auth_router.get(
    "/roles",
    response_model=list[AuthRoleResponse],
    dependencies=[Depends(get_current_user)],
)
async def read_auth_roles() -> list[AuthRoleResponse]:
    return [
        AuthRoleResponse(role=role, description=description)
        for role, description in _ROLE_DESCRIPTIONS.items()
    ]


@auth_router.get(
    "/users",
    response_model=list[AuthUserResponse],
    dependencies=[Depends(require_global_admin)],
)
async def list_auth_users(
    user_manager: Annotated[UserManager, Depends(get_user_manager)],
    territory: Annotated[str | None, Query()] = None,
) -> list[AuthUserResponse]:
    return [
        _user_response(user) for user in user_manager.list_users(territory=territory)
    ]


@auth_router.post(
    "/users",
    status_code=status.HTTP_201_CREATED,
    response_model=AuthUserResponse,
    dependencies=[Depends(require_global_admin)],
)
async def create_auth_user(
    data: CreateAuthUserRequest,
    user_manager: Annotated[UserManager, Depends(get_user_manager)],
) -> AuthUserResponse:
    try:
        territories = data.territories
        if data.role is UserRole.MULTIEDITOR and not territories:
            territories = _all_model_territories()
        user = user_manager.create_user(
            identifier=data.identifier,
            role=data.role,
            territories=territories,
        )
        user_manager.reload()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this identifier already exists",
        ) from exc
    return _user_response(user)


@auth_router.patch(
    "/users/{user_id}",
    response_model=AuthUserResponse,
    dependencies=[Depends(require_global_admin)],
)
async def update_auth_user(
    user_id: str,
    data: UpdateAuthUserRequest,
    user_manager: Annotated[UserManager, Depends(get_user_manager)],
) -> AuthUserResponse:
    if all(value is None for value in (data.role, data.territories, data.is_active)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one user field must be provided",
        )
    try:
        territories = data.territories
        if data.role is UserRole.MULTIEDITOR and not territories:
            territories = _all_model_territories()
        elif territories == [] and data.role is None:
            current_user = next(
                (user for user in user_manager.list_users() if user.user_id == user_id),
                None,
            )
            if current_user is not None and current_user.role is UserRole.MULTIEDITOR:
                territories = _all_model_territories()
        user = user_manager.update_user(
            user_id,
            role=data.role,
            territories=territories,
            is_active=data.is_active,
        )
        user_manager.reload()
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this identifier already exists",
        ) from exc
    return _user_response(user)


@auth_router.delete(
    "/users/{user_id}",
    response_model=AuthUserResponse,
    dependencies=[Depends(require_global_admin)],
)
async def deactivate_auth_user(
    user_id: str,
    user_manager: Annotated[UserManager, Depends(get_user_manager)],
) -> AuthUserResponse:
    try:
        user = user_manager.deactivate_user(user_id)
        user_manager.reload()
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        ) from exc
    return _user_response(user)
