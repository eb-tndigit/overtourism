# SPDX-License-Identifier: Apache-2.0
"""Streamlit dashboard entry point for the Molveno overtourism model.

Dev/test tooling only — not part of the application backend. Run with::

    uv run streamlit run src/dt_studio/molveno_dashboard.py
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
from src.model.molveno.molveno_backend import MolvenoBackend

# ---------------------------------------------------------------------------
# Predefined scenarios
# ---------------------------------------------------------------------------

_MOLVENO_SCENARIOS: list[dict[str, Any]] = [
    {
        "key": "typical",
        "label": "Giorno tipico (tutta l'estate)",
        "category": "Riferimento",
        "description": (
            "Nessun vincolo contestuale: tutti i giorni dell'estate (1° giugno – 30 settembre), "
            "tutti i giorni della settimana e il meteo secondo le frequenze osservate."
        ),
        "overrides": {},
    },
    {
        "key": "peak_august_sunday",
        "label": "Domenica di agosto (bel tempo)",
        "category": "Scenari estremi",
        "description": (
            "Condizioni di massima affluenza: 1–25 agosto, domenica, meteo 'good'."
        ),
        "overrides": {
            "day of summer": (61.0, 86.0),
            "weekday": "sunday",
            "weather": "good",
        },
    },
    {
        "key": "rainy_august_sunday",
        "label": "Domenica di agosto con pioggia",
        "category": "Scenari estremi",
        "description": (
            "1–25 agosto, domenica, meteo 'rainy': calano gli escursionisti, la spiaggia si "
            "svuota e la ristorazione ne assorbe una quota maggiore."
        ),
        "overrides": {
            "day of summer": (61.0, 86.0),
            "weekday": "sunday",
            "weather": "rainy",
        },
    },
    {
        "key": "august_no_booking",
        "label": "Agosto, nessun parcheggio su prenotazione",
        "category": "Prenotazione dei parcheggi",
        "description": (
            "1–31 agosto senza posti su prenotazione (regole 2023–2024): chi non trova posto "
            "nei parcheggi a pagamento sosta altrove."
        ),
        "overrides": {"day of summer": (61.0, 92.0), "booked parking spaces": 0.0},
    },
    {
        "key": "august_booking_ischia",
        "label": "Agosto, Ischia su prenotazione",
        "category": "Prenotazione dei parcheggi",
        "description": (
            "1–31 agosto con il parcheggio Ischia (195 posti) solo su prenotazione, come dal 2025."
        ),
        "overrides": {"day of summer": (61.0, 92.0), "booked parking spaces": 195.0},
    },
    {
        "key": "august_booking_all_paid",
        "label": "Agosto, tutti i parcheggi a pagamento su prenotazione",
        "category": "Prenotazione dei parcheggi",
        "description": (
            "1–31 agosto con tutti i 529 posti a pagamento solo su prenotazione: gli escursionisti "
            "in auto oltre i posti disponibili rinunciano (ipotesi A13)."
        ),
        "overrides": {"day of summer": (61.0, 92.0), "booked parking spaces": 529.0},
    },
    {
        "key": "early_june_weekday",
        "label": "Giorno feriale di inizio giugno",
        "category": "Riferimento",
        "description": "1–15 giugno, da lunedì a venerdì, meteo 'good': condizioni di bassa pressione.",
        "overrides": {
            "day of summer": (0.0, 15.0),
            "weekday": "weekday",
            "weather": "good",
        },
    },
]


# ---------------------------------------------------------------------------
# MolvenoAdapter
# ---------------------------------------------------------------------------


class MolvenoAdapter(OvertourismAdapter):
    """Adapter bridging :class:`~molveno_backend.MolvenoBackend` to the generic dashboard."""

    def __init__(self) -> None:
        self._backend = MolvenoBackend()

    @property
    def title(self) -> str:
        """Dashboard page title."""
        return "Overtourism Digital Twin — Molveno (Lago di Molveno)"

    def parameter_specs(self) -> list[OvertourismParameterMeta]:
        """Return the backend's parameter schema, in schema-declaration order."""
        return self._backend.parameter_schema()

    def predefined_scenarios(self) -> list[ScenarioDef]:
        """Build ScenarioDef objects from the Molveno built-in scenario catalogue."""
        specs = self.parameter_specs()
        spec_map = {s.name: s for s in specs}

        scenario_defs: list[ScenarioDef] = []
        for s in _MOLVENO_SCENARIOS:
            # Start from spec defaults
            params: dict[str, Any] = {}
            for spec in specs:
                if spec.kind == "distribution" and spec.default_range is not None:
                    params[spec.name] = spec.default_range
                # categoricals: absent = "(tutte)" — averaged over all values

            # Apply the scenario's CV overrides by matching index name suffixes
            for cv_key, cv_val in s["overrides"].items():
                # cv_key is a short label ("day of summer", "weekday", "weather", ...)
                matching = next(
                    (name for name in spec_map if cv_key in name),
                    None,
                )
                if matching:
                    params[matching] = cv_val

            scenario_defs.append(
                ScenarioDef(
                    key=s["key"],
                    label=s["label"],
                    category=s["category"],
                    description=s["description"],
                    params=params,
                )
            )
        return scenario_defs

    def run(self, param_overrides: dict[str, Any]) -> PlotData:
        """Evaluate the Molveno model and return visualisation-ready data."""
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
def _get_adapter() -> MolvenoAdapter:
    return MolvenoAdapter()


run_dashboard(_get_adapter())
