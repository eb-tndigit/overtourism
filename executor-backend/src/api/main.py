# SPDX-License-Identifier: Apache-2.0
"""FastAPI entry point — Layer 5 REST API over the Fazzon/Molveno computation backends.

Run with::

    uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8082 \
        --proxy-headers --log-config uvicorn_log_config.json

The models API and the `/live` and `/ready` probes share port 8082.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.health import router as health_router
from src.api.logging_config import configure_logging
from src.api.registry import DATA_STARTUP
from src.api.routes import router

APP_VERSION = "0.1.0"

configure_logging()


def _get_cors_allowed_origins() -> list[str]:
    configured_origins = os.getenv("CORS_ALLOWED_ORIGINS", "*")
    return [
        origin.strip() for origin in configured_origins.split(",") if origin.strip()
    ]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Make sure every model has its data before serving (download it unless in
    standalone mode); any failure stops the startup."""
    for model_key, ensure_data in DATA_STARTUP.items():
        data_dir = ensure_data()
        logging.getLogger(__name__).info("%s data ready in %s", model_key, data_dir)
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="Overtourism Digital Twin API",
        version=APP_VERSION,
        description="REST layer over the CDT computation backends.",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_get_cors_allowed_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    application.include_router(router)
    application.include_router(health_router)
    return application


app = create_app()
