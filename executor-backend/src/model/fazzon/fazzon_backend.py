# SPDX-License-Identifier: Apache-2.0
"""Layer 3 computation backend for the Fazzon model — see `overtourism/BACKEND_DESIGN.md` §5.

The single evaluation path for the Fazzon model: builds its own seeded
`CrossProductEnsemble`/`Evaluation` and targets
`overtourism.model.common.sustainability_field.SustainabilityFieldOutput`
directly via the shared field-math functions.
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar

import numpy as np
from civic_digital_twins.dt_model import (
    CategoricalIndex,
    CrossProductEnsemble,
    Evaluation,
    Scenario,
    build_scenario,
    sample_across,
)

from src.model.common.labels import MONTH_LABELS, WEATHER_LABELS
from src.model.common.sustainability_field import (
    OvertourismEvaluationConfig,
    OvertourismParameterMeta,
    SustainabilityFieldOutput,
    arrange_frontend_data,
    compute_sustainability_field,
)
from src.model.fazzon.fazzon_model import FazzonModel
from src.model.fazzon.fazzon_presence_stats import (
    PRICE_INDEX_NAME,
    PRICE_LEVELS,
    behaviour_params,
    load_presence_stats,
    policy_kpis,
    shuttle_capacity_level,
    shuttle_usable_capacity,
)


def price_level(price: float) -> str:
    """The price level (5-euro class, D15) of a price in euro: nearest class, clipped to the
    levels' range."""
    step = PRICE_LEVELS[1] - PRICE_LEVELS[0]
    level = int(round(float(price) / step) * step)
    return str(min(max(level, PRICE_LEVELS[0]), PRICE_LEVELS[-1]))


class FazzonBackend:
    """Frontend-agnostic computation backend for the Fazzon model.

    Constructor builds the model, the index map, and holds an
    `OvertourismEvaluationConfig` with fixed seeds and sample counts — seeds
    are a compute-quality concern and live here, not in the frontend.
    """

    # Subsystem/constraint display names and axis labels — the backend's own
    # minimal, structural presentation metadata, hardcoded here the same way
    # `_schema` hardcodes parameter metadata. `schema()` returns these
    # verbatim
    MAPPER: ClassVar[dict[str, str]] = {
        "parking": "Parcheggi",
        "road": "Viabilità",
        "food": "Ristorazione",
    }
    # The axes are the arrivals under the access policy (D15): the policy moves the samples.
    # Total arrivals on the (wider) x axis, car arrivals on y.
    X_AXIS_NAME = "Visitatori arrivati in totale / giorno"
    Y_AXIS_NAME = "Visitatori arrivati in auto / giorno"
    # TODO: check: are thesepointers really needed?
    # Model-specific KPIs (SustainabilityFieldOutput.extra_kpis) and their display names (D18)
    KPI_LABELS: ClassVar[dict[str, str]] = {
        "arrivals mean": "Arrivi rispetto ai visitatori potenziali: media stagionale (persone/giorno)",
        "arrivals peak": "Arrivi rispetto ai visitatori potenziali: giorni di picco (persone/giorno)",
        "vehicles mean": "Veicoli rispetto alle regole 2025: media stagionale (veicoli/giorno)",
        "vehicles peak": "Veicoli rispetto alle regole 2025: giorni di picco (veicoli/giorno)",
    }
    X_FIELD = "visitors_total"
    Y_FIELD = "visitors_car"

    def __init__(self) -> None:
        model = FazzonModel(inputs=FazzonModel.default_inputs())
        # Demand statistics and behaviour parameters, for the policy KPIs
        self._presence_stats = load_presence_stats()
        self._behaviour = behaviour_params()
        self._model = model
        self._index_map: dict[str, Any] = {idx.name: idx for idx in model.indexes}
        self._parameter_axes = [
            model.inputs.pv_visitors_total,
            model.inputs.pv_visitors_car,
        ]
        # Grid-resolution: structural axis choices, not per-evaluation config.
        # x: total arrivals, up to 3000/day; y: car arrivals, up to 2000/day (the busiest
        # August days in 2025 conditions reach about 2500 in total and 2000 by car).
        self._t_max, self._c_max = 3000, 2000
        self._t_sample, self._c_sample = 100, 100
        self._config = OvertourismEvaluationConfig(
            ensemble_size=12,  # 4 seasons x 3 weather classes (D5, D7)
            ensemble_seed=0,
            n_samples_per_combo=10,
            sample_seed=1,
            target_presence_samples=2000,
            confidence=0.8,
        )

    def parameter_schema(self) -> list[OvertourismParameterMeta]:
        """Return the ordered, self-describing parameter schema."""
        return list(self._schema.values())

    def schema(self) -> dict[str, Any]:
        """Return Fazzon indexes and placeholder frontend metadata."""
        return {
            "metadata": {
                "mapper": self.MAPPER,
                "x_axis_name": self.X_AXIS_NAME,
                "y_axis_name": self.Y_AXIS_NAME,
                "x_field": self.X_FIELD,
                "y_field": self.Y_FIELD,
                "kpi_labels": self.KPI_LABELS,
            },
            "indexes": self.parameter_schema(),
        }

    @property
    def model(self) -> FazzonModel:
        """The live `FazzonModel` instance.

        Needed by frontends that resolve predefined what-if scenarios
        (`Index`-keyed ``overrides_fn`` callables) against a live model, e.g.
        `fazzon_scenarios.ALL_SCENARIOS`.
        """
        return self._model

    def evaluate(self, param_overrides: dict[str, Any]) -> SustainabilityFieldOutput:
        """Evaluate the model under the given string-keyed parameter overrides.

        The parking price is a slider in the schema (euro) but a category in the model
        (D15): it is converted to its price level here, outside ``build_scenario``.
        """
        params = dict(param_overrides)
        price = params.pop(PRICE_INDEX_NAME, None)
        scenario = build_scenario(
            self._model,
            params,
            self._index_map,
            self._schema,
            self._parameter_axes,
        )
        if price is not None:
            overrides = {
                **scenario.overrides,
                self._model.cv_parking_price: price_level(price),
            }
            scenario = Scenario(
                self._model, overrides=overrides, parameter_axes=self._parameter_axes
            )
        return self._evaluate_scenario(scenario)

    def _with_policy(self, scenario: Scenario) -> Scenario:
        """Fix the access-policy categories (D15, D16): the price level (default: the 2025
        price, 5 euro) and the usable shuttle capacity class, computed from the shuttle
        timetable parameters (overridden or default).  Left free, the ensemble would mix
        the levels."""
        model = self._model
        inp = model.inputs
        overrides = dict(scenario.overrides)
        overrides.setdefault(model.cv_parking_price, str(PRICE_LEVELS[0]))

        def value(idx):
            return float(overrides.get(idx, idx.concrete_default))

        capacity = shuttle_usable_capacity(
            buses=value(inp.i_shuttle_buses),
            seats_per_bus=value(inp.i_shuttle_seats_per_bus),
            interval_min=value(inp.i_shuttle_interval_min),
            start_hour=value(inp.i_shuttle_start_hour),
            end_hour=value(inp.i_shuttle_end_hour),
            peak_end_hour=value(inp.i_shuttle_peak_end_hour),
            peak_share=value(inp.i_shuttle_peak_share),
        )
        overrides[model.cv_shuttle] = shuttle_capacity_level(capacity)
        return Scenario(model, overrides=overrides, parameter_axes=self._parameter_axes)

    def arrange_data(self, output: SustainabilityFieldOutput) -> dict[str, Any]:
        """Arrange Fazzon output for the frontend presentation format."""
        return arrange_frontend_data(output)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @functools.cached_property
    def _schema(self) -> dict[str, OvertourismParameterMeta]:
        """Hand-authored parameter metadata; the single source of truth (§3.1)."""
        inp = self._model.inputs
        return {
            inp.cv_parking_price.name: OvertourismParameterMeta(
                name=inp.cv_parking_price.name,
                kind="scalar",  # a category in the model, converted by evaluate() (D15)
                label="Tariffa parcheggio (€ per veicolo)",
                description=(
                    "Prezzo medio pagato per veicolo, a passi di 5 €. 5 € = regole 2025; 15 € = "
                    "media pagata con le regole 2026 (20 € giornata, 10 € dopo le 12).  Una tariffa "
                    "più alta riduce gli arrivi in auto (risposta stimata sui dati 2025-2026) e "
                    "aumenta le persone per veicolo; chi rinuncia all'auto prende la navetta (se "
                    "attiva), sale a piedi o va altrove."
                ),
                unit="€",
                category="Parcheggio",
                default=float(PRICE_LEVELS[0]),
                min_value=float(PRICE_LEVELS[0]),
                max_value=float(PRICE_LEVELS[-1]),
                step=float(PRICE_LEVELS[1] - PRICE_LEVELS[0]),
            ),
            inp.i_c_parking.name: OvertourismParameterMeta(
                name=inp.i_c_parking.name,
                kind="distribution",
                label="Cap parcheggio (auto simultanee)",
                description=(
                    "Range [min, max] del numero massimo di auto in parcheggio simultaneo. "
                    "Riferimento 2025: triangolare moda 200, range [150, 250]."
                ),
                unit="auto",
                category="Parcheggio",
                default_range=(150.0, 250.0),
                min_value=50.0,
                max_value=500.0,
                step=10.0,
                distribution_family="triang",
                distribution_fixed_params={"c": 0.5},
            ),
            inp.i_shuttle_buses.name: OvertourismParameterMeta(
                name=inp.i_shuttle_buses.name,
                kind="scalar",
                label="Navetta: numero di bus",
                description="Bus che partono insieme a ogni corsa. 0 = nessuna navetta. Lakebus 2026: 2.",
                unit="bus",
                category="Navetta",
                default=0.0,
                min_value=0.0,
                max_value=6.0,
                step=1.0,
            ),
            inp.i_shuttle_seats_per_bus.name: OvertourismParameterMeta(
                name=inp.i_shuttle_seats_per_bus.name,
                kind="scalar",
                label="Navetta: posti per bus",
                description="Posti a sedere di ogni bus. Lakebus 2026: 30.",
                unit="posti",
                category="Navetta",
                default=30.0,
                min_value=10.0,
                max_value=60.0,
                step=5.0,
            ),
            inp.i_shuttle_interval_min.name: OvertourismParameterMeta(
                name=inp.i_shuttle_interval_min.name,
                kind="scalar",
                label="Navetta: frequenza (minuti)",
                description="Minuti fra due corse. Lakebus 2026: 30 (salita nei primi 15, discesa nei secondi 15).",
                unit="min",
                category="Navetta",
                default=30.0,
                min_value=10.0,
                max_value=60.0,
                step=5.0,
            ),
            inp.i_shuttle_start_hour.name: OvertourismParameterMeta(
                name=inp.i_shuttle_start_hour.name,
                kind="scalar",
                label="Navetta: prima corsa (ora)",
                description="Ora della prima partenza da Pellizzano. Lakebus 2026: 9.",
                unit="h",
                category="Navetta",
                default=9.0,
                min_value=6.0,
                max_value=12.0,
                step=1.0,
            ),
            inp.i_shuttle_end_hour.name: OvertourismParameterMeta(
                name=inp.i_shuttle_end_hour.name,
                kind="scalar",
                label="Navetta: fine servizio (ora)",
                description="Ora di fine servizio, cioè dell'ultimo rientro a Pellizzano. Lakebus 2026: 19 (ultima partenza 18:30).",
                unit="h",
                category="Navetta",
                default=19.0,
                min_value=14.0,
                max_value=22.0,
                step=1.0,
            ),
            inp.i_shuttle_peak_end_hour.name: OvertourismParameterMeta(
                name=inp.i_shuttle_peak_end_hour.name,
                kind="scalar",
                label="Navetta: fine picco del mattino (ora)",
                description="I passeggeri si concentrano fra la prima corsa e quest'ora. ASSUNZIONE: 12.",
                unit="h",
                category="Navetta",
                default=12.0,
                min_value=10.0,
                max_value=15.0,
                step=1.0,
            ),
            inp.i_shuttle_peak_share.name: OvertourismParameterMeta(
                name=inp.i_shuttle_peak_share.name,
                kind="scalar",
                label="Navetta: quota passeggeri nel picco",
                description="Quota dei passeggeri che vuole salire nel picco del mattino: limita la capacità utilizzabile. ASSUNZIONE: 0.6 (da validare; dati 2026: al massimo 0.69).",
                unit="",
                category="Navetta",
                default=0.6,
                min_value=0.1,
                max_value=1.0,
                step=0.05,
            ),
            inp.i_u_food_fraction_car.name: OvertourismParameterMeta(
                name=inp.i_u_food_fraction_car.name,
                kind="scalar",
                label="Ristorazione: quota visitatori in auto che usano i ristoranti",
                description=(
                    "Chi arriva in auto può portarsi il picnic. ASSUNZIONE: 0.50, con 0.66 per gli "
                    "altri visitatori dà la quota 0.55 dell'indagine EETRA 2022 con le regole 2025."
                ),
                unit="",
                category="Ristorazione",
                default=0.50,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
            ),
            inp.i_u_food_fraction_other.name: OvertourismParameterMeta(
                name=inp.i_u_food_fraction_other.name,
                kind="scalar",
                label="Ristorazione: quota altri visitatori che usano i ristoranti",
                description=(
                    "Visitatori arrivati in navetta, a piedi o in bici. ASSUNZIONE: 0.66 (vedi la "
                    "quota dei visitatori in auto)."
                ),
                unit="",
                category="Ristorazione",
                default=0.66,
                min_value=0.0,
                max_value=1.0,
                step=0.02,
            ),
            inp.cv_season.name: OvertourismParameterMeta(
                name=inp.cv_season.name,
                kind="categorical",
                label="Stagione",
                description=(
                    "Vincola la valutazione a una singola stagione. "
                    "Lasciare su '(tutte)' per la media ponderata stagionale."
                ),
                category="Contesto",
                support=list(inp.cv_season.support),
                support_labels=MONTH_LABELS,
            ),
            inp.cv_weather.name: OvertourismParameterMeta(
                name=inp.cv_weather.name,
                kind="categorical",
                label="Meteo",
                description=(
                    "Vincola la valutazione a un tipo di giornata: bel tempo, fresco "
                    "(temperatura massima di oltre 2 °C sotto la media del mese) o pioggia "
                    "(almeno 5 mm). "
                    "Lasciare su '(tutte)' per la media ponderata sulla climatologia."
                ),
                category="Contesto",
                support=list(inp.cv_weather.support),
                support_labels=WEATHER_LABELS,
            ),
        }

    def _evaluate_scenario(self, scenario: Scenario) -> SustainabilityFieldOutput:
        """Run the seeded field + presence-sample evaluation for `scenario`."""
        scenario = self._with_policy(scenario)
        model = self._model
        config = self._config
        pv_car, pv_total = model.inputs.pv_visitors_car, model.inputs.pv_visitors_total
        pv_noncar = model.inputs.pv_visitors_other
        tt = np.linspace(0, self._t_max, self._t_sample + 1)
        cc = np.linspace(0, self._c_max, self._c_sample + 1)

        # Presence samples for the scatter overlay — seeded independently of the field.
        sampling_overrides = {
            idx: (
                [val]
                if isinstance(idx, CategoricalIndex) and isinstance(val, str)
                else val
            )
            for idx, val in scenario.overrides.items()
        }
        sampling_scenario = Scenario(
            model, overrides=sampling_overrides, parameter_axes=[pv_total, pv_car]
        )
        sampling_ensemble = CrossProductEnsemble(
            sampling_scenario,
            max_categorical_size=config.ensemble_size,
            rng=np.random.default_rng(config.sample_seed),
        )
        pv_samples = sample_across(
            sampling_ensemble,
            [pv_car, pv_noncar],
            total=config.target_presence_samples,
            rng=np.random.default_rng(config.sample_seed),
        )

        # Field ensemble — seeded, n_samples_per_combo capacity-distribution replicates
        # per categorical combo. `scenario` already carries parameter_axes (set by
        # build_scenario), so it's used as-is — no internal rebuild.
        field_ensemble = CrossProductEnsemble(
            scenario,
            max_categorical_size=config.ensemble_size,
            n_samples_per_combo=config.n_samples_per_combo,
            rng=np.random.default_rng(config.ensemble_seed),
        )
        result = Evaluation(scenario).evaluate(
            ensemble=field_ensemble,
            parameters={pv_total: tt, pv_car: cc},
        )
        field, field_elements, usage_fields, capacity_distributions = (
            compute_sustainability_field(
                model.constraints, result, pv_total, pv_car, scenario=scenario
            )
        )

        return SustainabilityFieldOutput(
            field=field,
            field_elements=field_elements,
            x_values=tt,
            y_values=cc,
            # total = car + non-car draws of the same day (aligned by sample_across)
            samples_x=list(pv_samples[pv_car] + pv_samples[pv_noncar]),
            samples_y=list(pv_samples[pv_car]),
            usage_fields=usage_fields,
            capacity_distributions=capacity_distributions,
            confidence=config.confidence,
            extra_kpis=self._policy_kpis(scenario),
        )

    def _policy_kpis(self, scenario: Scenario) -> dict[str, dict[str, float]]:
        """Arrivals out of the potential visitors and vehicles out of the 2025-rules
        vehicles (A36), for the scenario's policy and for the seasons / weather classes it
        allows (fixed or restricted categories)."""
        inp = self._model.inputs
        overrides = scenario.overrides

        def allowed(idx) -> dict[str, float]:
            weights = dict(idx.outcomes)
            pin = overrides.get(idx)
            if pin is None:
                return weights
            keep = [pin] if isinstance(pin, str) else list(pin)
            return {k: weights[k] for k in keep}

        seasons, weathers = allowed(inp.cv_season), allowed(inp.cv_weather)
        cell_weights = {
            (s, w): ps * pw for s, ps in seasons.items() for w, pw in weathers.items()
        }
        result = policy_kpis(
            self._presence_stats,
            self._behaviour,
            cell_weights,
            price=float(overrides[inp.cv_parking_price]),
            capacity=float(overrides[inp.cv_shuttle]),
        )
        return {
            f"{quantity} {days}": {
                "value": round(result[quantity][days]["value"], 1),
                "share": round(result[quantity][days]["share"] * 100, 2),
            }
            for quantity in ("arrivals", "vehicles")
            for days in ("mean", "peak")
        }
