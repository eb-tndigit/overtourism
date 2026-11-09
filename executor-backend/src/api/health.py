# SPDX-License-Identifier: Apache-2.0
"""Liveness and readiness probes, at the root path of the API (no `/health` prefix)."""

from __future__ import annotations

from fastapi import APIRouter, Response

from src.api.registry import BACKEND_REGISTRY, MODEL_TITLES

router = APIRouter(tags=["Health"])


@router.get("/live")
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/ready")
def readiness(response: Response) -> dict[str, str]:
    # The model registry must be internally consistent before serving requests.
    if not BACKEND_REGISTRY or BACKEND_REGISTRY.keys() != MODEL_TITLES.keys():
        response.status_code = 503
        return {"status": "not_ready"}
    return {"status": "ready"}
