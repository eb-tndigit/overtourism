# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status

from overtourism.backend.auth.identity.user_manager import UserManager
from overtourism.backend.auth.identity.users import User, UserRole
from overtourism.backend.auth.tokens.context import AuthContext
from overtourism.backend.auth.tokens.dependencies import get_auth_context
from overtourism.backend.auth.tokens.enums import AuthClaim
from overtourism.backend.auth.tokens.settings import AuthSettings, get_auth_settings
from overtourism.backend.handler import Handler, get_handler


def resolve_current_user(
    context: AuthContext,
    handler: Handler,
    auth_settings: AuthSettings,
) -> User | None:
    if not context.authenticated:
        return None
    if context.subject is None or not context.subject.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing user identity claim",
        )
    if handler.user_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User manager is not configured",
        )

    user = handler.user_manager.get_active_user_by_subject(context.subject)
    identifier = context.claims.get(auth_settings.identifier_claim)
    if (
        user is None
        and (
            auth_settings.identifier_claim != AuthClaim.EMAIL
            or context.claims.get(AuthClaim.EMAIL_VERIFIED) is True
        )
        and isinstance(identifier, str)
    ):
        user = handler.user_manager.claim_user_by_identifier(
            identifier,
            context.subject,
        )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not registered or active",
        )
    return user


def get_current_user(
    context: Annotated[AuthContext, Depends(get_auth_context)],
    handler: Annotated[Handler, Depends(get_handler)],
    auth_settings: Annotated[AuthSettings, Depends(get_auth_settings)],
) -> User:
    user = resolve_current_user(context, handler, auth_settings)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required",
        )
    return user


def require_global_admin(
    user: Annotated[User, Depends(get_current_user)],
) -> User:
    if user.role is not UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Global admin role is required",
        )
    return user


def require_territory_access(
    territory: str,
    context: Annotated[AuthContext, Depends(get_auth_context)],
    handler: Annotated[Handler, Depends(get_handler)],
    auth_settings: Annotated[AuthSettings, Depends(get_auth_settings)],
) -> User | None:
    user = resolve_current_user(context, handler, auth_settings)
    if (
        user is not None
        and user.role is not UserRole.ADMIN
        and territory not in user.territories
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not assigned to this territory",
        )
    return user


def require_persistent_write_access(
    user: Annotated[User | None, Depends(require_territory_access)],
) -> User | None:
    if user is not None and user.role is UserRole.VIEWER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Viewer role cannot modify persistent data",
        )
    return user


def get_user_manager(
    handler: Annotated[Handler, Depends(get_handler)],
) -> UserManager:
    if handler.user_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User manager is not configured",
        )
    return handler.user_manager
