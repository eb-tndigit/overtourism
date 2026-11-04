# SPDX-License-Identifier: Apache-2.0
"""REST routes over the Layer 3 computation backends.

One generic route set parameterized by `{model_key}` rather than one router
per model (contrast `overtourism.OLD/backend/api/`, whose routers diverge
per-concern because that Layer 4 was keyed by `problem_id`) — Fazzon and
Molveno backends already share an identical contract (§Layer 3), so a single
router covers both.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.registry import (
    BACKEND_REGISTRY,
    DATA_DOWNLOADERS,
    MODEL_TITLES,
    get_backend,
)
from src.api.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    IndexDiffsRequest,
    IndexDiffsResponse,
    ModelInfo,
    ModelSchema,
)
from src.model.common.sustainability_field import (
    get_index_diffs as calculate_index_diffs,
)

router = APIRouter(prefix="/models", tags=["models"])


def _get_backend_or_404(model_key: str):
    try:
        return get_backend(model_key)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Unknown model: {model_key!r}"
        ) from None


@router.get("", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    """List the available models."""
    return [ModelInfo(key=key, title=MODEL_TITLES[key]) for key in BACKEND_REGISTRY]


# Presentation constants that are identical across every model
_RISK_COLOR_SCALE: list[tuple[float, str]] = [
    (0.0, "rgb(5, 102, 8)"),
    (0.05, "rgb(100, 180, 90)"),
    (0.20, "rgb(180, 230, 170)"),
    (0.40, "rgb(230, 250, 225)"),
    (0.50, "yellow"),
    (0.60, "rgb(255, 242, 242)"),
    (0.80, "rgb(242, 204, 204)"),
    (0.95, "rgb(204, 76, 76)"),
    (1.0, "rgb(180, 4, 38)"),
]

_MONODIMENSIONAL_PLOT_MAPPER: dict[str, dict[str, str]] = {
    "x": {"label": "Giorni (ordinati per utilizzo)"},
    "y": {"label": "Livello di utilizzo della destinazione", "field": "usage"},
}


def _build_metadata(minimal: dict[str, Any]) -> dict[str, Any]:
    """Build the full `/schema` presentation metadata from a backend's minimal set."""
    mapper = minimal["mapper"] | {"default": "Tutti"}
    constraint_kpis = {
        f"constraint level {key}": f"Sostenibilità {label}"
        for key, label in mapper.items()
        if key != "default"
    }
    # display order: the model-specific KPIs, then the sustainability (as SustainabilityFieldOutput.kpis)
    kpi_mapper = {
        "title": "Indici",
        "area": "Area Totale",
        **minimal.get("kpi_labels", {}),
        "sustainability_level": "Sostenibilità complessiva",
        **constraint_kpis,
        "critical constraint": "Vincolo Critico",
    }
    return {
        "mapper": mapper,
        "color_map": _RISK_COLOR_SCALE,
        "kpi_mapper": kpi_mapper,
        "plot_mapper": {
            "monodimensional": _MONODIMENSIONAL_PLOT_MAPPER,
            "bidimensional": {
                "x": {"label": minimal["x_axis_name"], "field": minimal["x_field"]},
                "y": {"label": minimal["y_axis_name"], "field": minimal["y_field"]},
            },
        },
    }


@router.get("/{model_key}/schema", response_model=ModelSchema)
def get_schema(model_key: str) -> ModelSchema:
    """Return the model's ordered, self-describing parameter schema."""
    backend = _get_backend_or_404(model_key)
    schema = backend.schema()
    metadata = _build_metadata(schema["metadata"])
    return ModelSchema.model_validate(
        {"metadata": metadata, "indexes": schema["indexes"]}
    )


@router.post(
    "/{model_key}/index-diffs",
    response_model=IndexDiffsResponse,
)
def get_index_diffs_for_scenarios(
    model_key: str,
    body: IndexDiffsRequest,
) -> IndexDiffsResponse:
    """Return parameter diffs for a batch of scenarios."""
    backend = _get_backend_or_404(model_key)
    if not body.param_overrides_by_scenario:
        return IndexDiffsResponse(index_diffs_by_scenario={})

    schema = backend.schema()
    return IndexDiffsResponse(
        index_diffs_by_scenario={
            scenario_id: calculate_index_diffs(schema, param_overrides)
            for scenario_id, param_overrides in body.param_overrides_by_scenario.items()
        }
    )


@router.post("/{model_key}/evaluate", response_model=Any)
def evaluate(
    model_key: str,
    body: EvaluateRequest,
    as_snapshot: bool = True,
) -> Any:
    """Evaluate the model under the given string-keyed parameter overrides."""
    backend = _get_backend_or_404(model_key)
    output = backend.evaluate(body.param_overrides)
    if as_snapshot:
        return EvaluateResponse.from_output(output)
    return backend.arrange_data(output)


@router.post("/{model_key}/download-data", response_model=dict[str, str])
def download_data(model_key: str) -> dict[str, str]:
    """Download the model's data files from the platform again (overwriting them).

    Not authenticated, like the rest of this API: the backend layers that wrap it are
    expected to restrict access (e.g. to administrators).  On success the cached
    backend is dropped, so the next request builds the model with the new data; on
    failure the current files stay in place.
    """
    if model_key not in BACKEND_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_key!r}")
    downloader = DATA_DOWNLOADERS.get(model_key)
    if downloader is None:
        raise HTTPException(
            status_code=404, detail=f"Model {model_key!r} has no downloadable data"
        )
    try:
        data_dir = downloader(overwrite=True)
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"{model_key} data download failed: {e}"
        ) from e
    get_backend.cache_clear()
    return {"message": f"{model_key} data downloaded", "data_dir": str(data_dir)}
