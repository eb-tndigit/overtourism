# SPDX-License-Identifier: Apache-2.0
import os
import typing

import requests

MODEL_BACKEND_URL = os.environ.get("MODEL_BACKEND_URL", "http://localhost:8082")
MODEL_BACKEND_USER = os.environ.get("MODEL_BACKEND_USER")
MODEL_BACKEND_PASSWORD = os.environ.get("MODEL_BACKEND_PASSWORD")


def get_model_backend_auth() -> tuple[str, str] | None:
    if MODEL_BACKEND_USER is None or MODEL_BACKEND_PASSWORD is None:
        return None
    return MODEL_BACKEND_USER, MODEL_BACKEND_PASSWORD


def call_executor(
    territory: str,
    param_overrides: dict[str, typing.Any] | None = None,
    as_snapshot: bool = False,
) -> dict[str, typing.Any]:
    """Call the backend executor with the provided parameters."""
    if param_overrides is None:
        param_overrides = {}
    params = {"as_snapshot": str(as_snapshot).lower()}
    base_url = f"{MODEL_BACKEND_URL}/models/{territory}/evaluate"
    return requests.post(
        base_url,
        json={"param_overrides": param_overrides},
        params=params,
        auth=get_model_backend_auth(),
    ).json()


def list_models() -> list[dict[str, typing.Any]]:
    """Call the backend model list endpoint."""
    base_url = f"{MODEL_BACKEND_URL}/models"
    return requests.get(base_url, auth=get_model_backend_auth()).json()


def call_schema(
    territory: str,
) -> dict[str, typing.Any]:
    """Call the backend schema endpoint for the provided territory."""
    base_url = f"{MODEL_BACKEND_URL}/models/{territory}/schema"
    r = requests.get(base_url, auth=get_model_backend_auth())
    r.raise_for_status()
    return r.json()


def call_download_data(model: str) -> dict[str, str]:
    """Ask the Model API to download fresh data for a model."""
    url = f"{MODEL_BACKEND_URL}/models/{model}/download-data"
    response = requests.post(url, auth=get_model_backend_auth())
    response.raise_for_status()
    return response.json()


def call_health_ready() -> requests.Response:
    """Call the model backend readiness endpoint."""
    return requests.get(
        f"{MODEL_BACKEND_URL.rstrip('/')}/ready",
        auth=get_model_backend_auth(),
        timeout=1,
    )


def call_index_diffs(
    territory: str,
    param_overrides_by_scenario: dict[str, dict[str, typing.Any]],
) -> dict[str, dict[str, str]]:
    """Request batched scenario parameter diffs from the model backend."""
    url = f"{MODEL_BACKEND_URL}/models/{territory}/index-diffs"
    response = requests.post(
        url,
        json={"param_overrides_by_scenario": param_overrides_by_scenario},
        auth=get_model_backend_auth(),
    )
    response.raise_for_status()
    return response.json()["index_diffs_by_scenario"]
