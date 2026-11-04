# SPDX-License-Identifier: Apache-2.0
"""Streamlit dashboard entry point for the Altipiano di Fazzon overtourism model.

Dev/test tooling only — not part of the application backend. Run with::

    uv run streamlit run src/dt_studio/fazzon_dashboard.py

from the ``executor-backend`` folder, with the Fazzon data in ``FAZZON_DATA_DIR`` (e.g.
``../data_preparation/Output/data/final_data/fazzon``; see ``docs/howto/backend.md``).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.dt_studio.dashboard.adapter import (
    OvertourismAdapter,
    PlotData,
    ScenarioDef,
    format_extra_kpis,
)
from src.dt_studio.dashboard.app import run_dashboard
from src.model.common.sustainability_field import (
    OvertourismParameterMeta,
)
from src.model.fazzon.fazzon_backend import FazzonBackend
from src.model.fazzon.fazzon_scenarios import ALL_SCENARIOS

# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------


def _dist_to_range(frozen_dist: Any) -> tuple[float, float]:
    """Extract ``(lo, hi)`` endpoints from a frozen scipy distribution.

    The convention is ``lo = loc`` and ``hi = loc + scale``, matching the
    convention documented on :func:`civic_digital_twins.dt_model.build_scenario`.
    Used only to translate the *predefined* scenario catalogue
    (``fazzon_scenarios.ALL_SCENARIOS``, which is expressed against live
    ``Index`` objects) into the string-keyed vocabulary the dashboard uses —
    the reverse direction (``(lo, hi) -> frozen distribution``) is
    ``build_scenario``'s job, inside :class:`~fazzon_backend.FazzonBackend`.

    Parameters
    ----------
    frozen_dist : Any
        A frozen scipy distribution (e.g. ``scipy.stats.triang(c=0.5,
        loc=150.0, scale=100.0)``).  The keyword arguments passed at
        construction time are accessed via ``frozen_dist.kwds``.

    Returns
    -------
    tuple of (float, float)
        ``(loc, loc + scale)`` extracted from ``frozen_dist.kwds``.
    """
    loc: float = float(frozen_dist.kwds.get("loc", 0.0))
    scale: float = float(frozen_dist.kwds.get("scale", 1.0))
    return loc, loc + scale


# ---------------------------------------------------------------------------
# FazzonAdapter
# ---------------------------------------------------------------------------


class FazzonAdapter(OvertourismAdapter):
    """Concrete adapter bridging :class:`~fazzon_backend.FazzonBackend` to the generic dashboard.

    Translates between the generic dashboard's string-keyed parameter
    interface (:class:`~overtourism.model.common.sustainability_field.OvertourismParameterMeta`,
    :class:`~overtourism.dt_studio.dashboard.adapter.PlotData`) and the Fazzon model's
    live Python index objects — the boundary crossing itself is entirely the
    backend's job (``build_scenario`` inside :meth:`FazzonBackend.evaluate`);
    this adapter only translates the *predefined scenario catalogue*
    (:data:`~fazzon_scenarios.ALL_SCENARIOS`, expressed against live index
    objects) into the string vocabulary, and maps
    :class:`~overtourism.model.common.sustainability_field.SustainabilityFieldOutput`
    onto :class:`~overtourism.dt_studio.dashboard.adapter.PlotData`.
    """

    def __init__(self) -> None:
        self._backend = FazzonBackend()
        self._spec_by_name: dict[str, OvertourismParameterMeta] = {
            s.name: s for s in self.parameter_specs()
        }

    @property
    def title(self) -> str:
        """Dashboard page title."""
        return "Overtourism Digital Twin — Altipiano di Fazzon"

    def parameter_specs(self) -> list[OvertourismParameterMeta]:
        """Return the backend's parameter schema, in schema-declaration order."""
        return self._backend.parameter_schema()

    def predefined_scenarios(self) -> list[ScenarioDef]:
        """Build :class:`ScenarioDef` objects from the Fazzon what-if scenario catalogue.

        For each scenario in :data:`~fazzon_scenarios.ALL_SCENARIOS`:

        1. Calls ``scenario.overrides_fn(self._backend.model)`` to obtain a
           ``{Index: value}`` override dict.
        2. Seeds ``params`` with spec defaults for all scalar and distribution
           specs (categoricals are skipped).
        3. Replaces entries for each overridden index: distribution values are
           converted to ``(lo, hi)`` tuples via :func:`_dist_to_range`;
           scalars are cast to ``float``.
        4. Appends a :class:`ScenarioDef` for each scenario.

        Returns
        -------
        list of ScenarioDef
            One entry per scenario in :data:`~fazzon_scenarios.ALL_SCENARIOS`.
        """
        specs = self.parameter_specs()
        model = self._backend.model

        scenario_defs: list[ScenarioDef] = []
        for s in ALL_SCENARIOS:
            overrides = s.overrides_fn(model)

            # Seed with spec defaults (skip categoricals — no meaningful default)
            params: dict[str, Any] = {}
            for spec in specs:
                if spec.kind == "scalar" and spec.default is not None:
                    params[spec.name] = spec.default
                elif spec.kind == "distribution" and spec.default_range is not None:
                    params[spec.name] = spec.default_range

            # Apply scenario-specific overrides
            for idx, raw_val in overrides.items():
                spec = self._spec_by_name.get(idx.name)
                if spec is None:
                    continue
                if spec.kind == "distribution":
                    params[spec.name] = _dist_to_range(raw_val)
                else:
                    params[spec.name] = float(raw_val)

            scenario_defs.append(
                ScenarioDef(
                    key=s.key,
                    label=s.label,
                    category=s.category,
                    description=s.description,
                    params=params,
                )
            )
        return scenario_defs

    def run(self, param_overrides: dict[str, Any]) -> PlotData:
        """Evaluate the Fazzon model and return visualisation-ready data.

        Parameters
        ----------
        param_overrides : dict of {str: Any}
            Partial ``{index_name: value}`` mapping from the dashboard.
            Scalars are ``float``, distributions are ``(lo, hi)`` tuples,
            categoricals are ``str``.  Absent keys use model defaults.

        Returns
        -------
        PlotData
            Fully populated data container ready for the dashboard renderer.
        """
        output = self._backend.evaluate(param_overrides)

        return PlotData(
            field=output.field,
            field_elements=output.field_elements,
            x_values=output.x_values,
            y_values=output.y_values,
            x_label=self._backend.X_AXIS_NAME,
            y_label=self._backend.Y_AXIS_NAME,
            samples_x=output.samples_x,
            samples_y=output.samples_y,
            sustainability_index=output.sustainability_index,
            sustainability_by_constraint=output.sustainability_by_constraint,
            modal_lines=output.modal_lines,
            constraint_labels=self._backend.MAPPER,
            extra_kpis=format_extra_kpis(output.extra_kpis, self._backend.KPI_LABELS),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_adapter() -> FazzonAdapter:
    """Create and cache the FazzonAdapter (expensive; runs once per server instance)."""
    return FazzonAdapter()


run_dashboard(_get_adapter())
