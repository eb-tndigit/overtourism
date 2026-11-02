# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from overtourism.backend.auth.tokens.enums import (
    AuthEnvironmentVariable,
    JwtAlgorithm,
)

_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_DEFAULT_IDENTIFIER_CLAIM = "email"


@dataclass(frozen=True, slots=True)
class AuthSettings:
    enabled: bool = False
    issuer: str | None = None
    audience: str | None = None
    jwks_url: str | None = None
    algorithms: tuple[str, ...] = (JwtAlgorithm.RS256,)
    leeway_seconds: int = 30
    identifier_claim: str = _DEFAULT_IDENTIFIER_CLAIM

    @classmethod
    def from_env(cls) -> AuthSettings:
        enabled_value = os.getenv(AuthEnvironmentVariable.ENABLED)
        algorithms_value = os.getenv(AuthEnvironmentVariable.ALGORITHMS)
        leeway_value = os.getenv(AuthEnvironmentVariable.LEEWAY_SECONDS)
        settings = cls(
            enabled=enabled_value is not None
            and enabled_value.strip().lower() in _TRUTHY_ENV_VALUES,
            issuer=(os.getenv(AuthEnvironmentVariable.ISSUER) or "").strip() or None,
            audience=(os.getenv(AuthEnvironmentVariable.AUDIENCE) or "").strip()
            or None,
            jwks_url=(os.getenv(AuthEnvironmentVariable.JWKS_URL) or "").strip()
            or None,
            algorithms=(
                tuple(
                    item.strip() for item in algorithms_value.split(",") if item.strip()
                )
                if algorithms_value and algorithms_value.strip()
                else (JwtAlgorithm.RS256,)
            ),
            leeway_seconds=(
                int(leeway_value) if leeway_value and leeway_value.strip() else 30
            ),
            identifier_claim=(
                (os.getenv(AuthEnvironmentVariable.IDENTIFIER_CLAIM) or "").strip()
                or _DEFAULT_IDENTIFIER_CLAIM
            ),
        )
        if settings.enabled:
            missing_settings = [
                name
                for name, value in (
                    (AuthEnvironmentVariable.JWKS_URL, settings.jwks_url),
                    (AuthEnvironmentVariable.ISSUER, settings.issuer),
                    (AuthEnvironmentVariable.AUDIENCE, settings.audience),
                )
                if value is None
            ]
            if missing_settings:
                raise ValueError(
                    f"{', '.join(missing_settings)} must be set when "
                    f"{AuthEnvironmentVariable.ENABLED} is true"
                )
        return settings


@lru_cache(maxsize=1)
def get_auth_settings() -> AuthSettings:
    return AuthSettings.from_env()
