# SPDX-License-Identifier: Apache-2.0
"""Generic `OvertourismAdapter` that talks to `overtourism.api` over HTTP.

Unlike `fazzon_dashboard.py`/`molveno_dashboard.py`, which import
`FazzonBackend`/`MolvenoBackend` and evaluate in-process, this adapter never
constructs a model or runs a `civic_digital_twins` evaluation itself — and,
unlike an earlier version of this file, it also never imports
`overtourism.model.*` for a type hint. `HttpParameterSpec` below is a local,
dependency-free dataclass satisfying `dashboard.adapter.ParameterSpec`
structurally, built from the API's JSON response rather than sharing
`OvertourismParameterMeta` with the computation layer — see
`overtourism/BACKEND_DESIGN.md` (Layers 1-3 and Layer 5 are slated to run in
separate containers; an HTTP-based Layer 5 client shouldn't need Layers
1-3's dependency chain just to get a schema type).

`run()`'s `/evaluate` response is plain JSON (`overtourism.api.schemas.EvaluateResponse`
— not `SustainabilityFieldOutput.to_snapshot()`), so array fields are
`np.array(...)`-wrapped directly; no base64 decoding is involved.

One adapter class serves every model in the API's registry (`model_key` is
just a constructor argument), mirroring the API's own single generic router.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

from src.dt_studio.dashboard.adapter import (
    OvertourismAdapter,
    ParameterSpec,
    PlotData,
    ScenarioDef,
    format_extra_kpis,
)

__all__ = ["HttpOvertourismAdapter", "HttpParameterSpec"]


@dataclass
class HttpParameterSpec:
    """Local, dependency-free counterpart to `OvertourismParameterMeta`.

    Built from a `GET /schema` JSON entry via `from_json()`. Structurally
    satisfies `dashboard.adapter.ParameterSpec` — deliberately not the same
    class as `overtourism.model.common.sustainability_field.OvertourismParameterMeta`,
    so this module has no import from `overtourism.model.*`.
    """

    name: str
    kind: str
    label: str = ""
    description: str = ""
    unit: str = ""
    category: str = ""
    step: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    default: float | None = None
    default_category: str | None = None
    default_range: tuple[float, float] | None = None
    support: list[str] = field(default_factory=list)
    support_labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> HttpParameterSpec:
        """Build from one `GET /models/{model_key}/schema` JSON entry.

        Reads fields explicitly (rather than `cls(**d)`) so unrelated API
        fields (e.g. `distribution_family`/`distribution_fixed_params`,
        needed server-side to build a `Scenario` but not by this renderer)
        don't have to be mirrored here, and so an API field addition doesn't
        break construction.
        """
        default_range = d.get("default_range")
        return cls(
            name=d["name"],
            kind=d["kind"],
            label=d.get("label", ""),
            description=d.get("description", ""),
            unit=d.get("unit", ""),
            category=d.get("category", ""),
            step=d.get("step"),
            min_value=d.get("min_value"),
            max_value=d.get("max_value"),
            default=d.get("default"),
            default_category=d.get("default_category"),
            default_range=tuple(default_range) if default_range else None,
            support=d.get("support", []),
            support_labels=d.get("support_labels") or {},
        )


class HttpOvertourismAdapter(OvertourismAdapter):
    """`OvertourismAdapter` backed by an `overtourism.api` HTTP endpoint.

    Parameters
    ----------
    model_key : str
        Key identifying the model in the API's registry (e.g. ``"fazzon"``).
    title : str
        Dashboard page title.
    base_url : str
        Base URL of the running `overtourism.api` service (e.g.
        ``"http://localhost:8082"``).
    timeout : float
        Per-request timeout, in seconds. Evaluation requests can take a few
        seconds (full ensemble evaluation), so this is generous by default.
    """

    def __init__(
        self, model_key: str, title: str, base_url: str, timeout: float = 60.0
    ) -> None:
        self._model_key = model_key
        self._title = title
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    @property
    def title(self) -> str:
        return self._title

    @functools.cached_property
    def _schema(self) -> dict[str, Any]:
        """Fetch and cache `GET /models/{model_key}/schema`.

        Reused by `parameter_specs()` (`indexes`) and `run()`
        (`metadata.plot_mapper` axis labels — see its docstring) so
        evaluating repeatedly doesn't refetch it.
        """
        resp = self._client.get(f"/models/{self._model_key}/schema")
        resp.raise_for_status()
        return resp.json()

    def parameter_specs(self) -> list[ParameterSpec]:
        """Return the parameter schema from the cached `/schema` response."""
        return [HttpParameterSpec.from_json(d) for d in self._schema["indexes"]]

    def predefined_scenarios(self) -> list[ScenarioDef]:
        """No `/scenarios` endpoint in v1 — the scenario selector is simply absent."""
        return []

    def run(self, param_overrides: dict[str, Any]) -> PlotData:
        """Evaluate via `POST /models/{model_key}/evaluate` and reshape the plain-JSON response."""
        metadata = self._schema.get("metadata", {})
        bidimensional = metadata.get("plot_mapper", {}).get("bidimensional", {})
        x_label = bidimensional.get("x", {}).get("label", "")
        y_label = bidimensional.get("y", {}).get("label", "")
        constraint_labels = metadata.get("mapper", {})

        resp = self._client.post(
            f"/models/{self._model_key}/evaluate",
            json={"param_overrides": param_overrides},
        )
        resp.raise_for_status()
        data = resp.json()

        return PlotData(
            field=np.array(data["field"]),
            field_elements={k: np.array(v) for k, v in data["field_elements"].items()},
            x_values=np.array(data["x_values"]),
            y_values=np.array(data["y_values"]),
            x_label=x_label,
            y_label=y_label,
            samples_x=data["samples_x"],
            samples_y=data["samples_y"],
            sustainability_index=(
                data["sustainability_index"]["value"],
                data["sustainability_index"]["ci"],
            ),
            sustainability_by_constraint={
                k: (v["value"], v["ci"])
                for k, v in data["sustainability_by_constraint"].items()
            },
            modal_lines={
                k: (np.array(v["x"]), np.array(v["y"]))
                for k, v in data["modal_lines"].items()
            },
            constraint_labels=constraint_labels,
            extra_kpis=format_extra_kpis(
                data.get("kpis", {}), metadata.get("kpi_mapper", {})
            ),
        )
