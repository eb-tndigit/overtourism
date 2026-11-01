# SPDX-License-Identifier: Apache-2.0
"""Streamlit dashboard for Fazzon, driven entirely over HTTP by `overtourism.api`.

Unlike `fazzon_dashboard.py`, this process never imports `FazzonBackend` or
any other model/computation code — see `http_adapter.py`. Start the API
first, then run this dashboard::

    uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8001 \
        --proxy-headers --log-config uvicorn_log_config.json
    uv run streamlit run src/dt_studio/fazzon_api_dashboard.py

Point at a non-default API host with the `OVERTOURISM_API_BASE_URL` env var.
"""

from __future__ import annotations

import os

import streamlit as st

from src.dt_studio.dashboard.app import run_dashboard
from src.dt_studio.dashboard.http_adapter import (
    HttpOvertourismAdapter,
)

_BASE_URL = os.environ.get("OVERTOURISM_API_BASE_URL", "http://localhost:8001")


@st.cache_resource
def _get_adapter() -> HttpOvertourismAdapter:
    return HttpOvertourismAdapter(
        model_key="fazzon",
        title="Overtourism Digital Twin — Altipiano di Fazzon [API]",
        base_url=_BASE_URL,
    )


run_dashboard(_get_adapter())
