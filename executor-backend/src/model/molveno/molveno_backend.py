# SPDX-License-Identifier: Apache-2.0
"""Layer 3 computation backend for the Molveno model — see `overtourism/BACKEND_DESIGN.md` §5.

The single evaluation path for the Molveno model: builds its own seeded
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

from src.model.common.labels import WEATHER_LABELS, WEEKDAY_LABELS
from src.model.common.sustainability_field import (
    OvertourismEvaluationConfig,
    OvertourismParameterMeta,
    SustainabilityFieldOutput,
    arrange_frontend_data,
    compute_sustainability_field,
    expected_period_days,
)
from src.model.molveno.molveno_model import MolvenoModel
from src.model.molveno.molveno_presence_stats import (
    DAY_INDEX_NAME,
    ESTIMATOR_ML,
    ESTIMATORS,
    load_tables,
    weather_weights,
)


def _presence_transformation(
    presence: float,
    reduction_factor: float,
    saturation_level: float,
    sharpness: int = 3,
) -> float:
    """Apply the presence saturation transformation used for scatter-plot samples.

    Molveno-specific (unlike the field math in `overtourism.model.common`): it
    depends on `i_p_*_reduction_factor`/`i_p_*_saturation_level`, parameters
    that only exist on `MolvenoModel`.
    """
    tmp = presence * reduction_factor
    return (
        tmp
        * saturation_level
        / ((tmp**sharpness + saturation_level**sharpness) ** (1 / sharpness))
    )


class MolvenoBackend:
    """Frontend-agnostic computation backend for the Molveno model.

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
        "beach": "Spiaggia",
        "accommodation": "Alberghi",
        "food": "Ristoranti",
    }
    X_AXIS_NAME = "Turisti / giorno"
    Y_AXIS_NAME = "Escursionisti / giorno"
    # TODO: check: are thesepointers really needed?
    X_FIELD = "tourist"
    Y_FIELD = "excursionist"
    # Days of the period sampled per weekday x weather combination: for the field and for
    # the scatter-plot presence samples.
    FIELD_DAYS_PER_COMBO = 6
    SAMPLE_DAYS_PER_COMBO = 20

    def __init__(self, presence_estimator: str = ESTIMATOR_ML) -> None:
        """*presence_estimator*: the presence statistics used (D9): ``ESTIMATOR_ML`` (gradient
        boosting per day, the default) or ``ESTIMATOR_MONTHLY`` (monthly averages, for
        comparison). A backend choice, not a scenario parameter: not in the schema."""
        if presence_estimator not in ESTIMATORS:
            raise ValueError(
                f"Unknown presence estimator {presence_estimator!r}; expected one of {ESTIMATORS}"
            )
        self._presence_estimator = presence_estimator
        model = MolvenoModel(inputs=MolvenoModel.default_inputs())
        self._model = model
        self._tables = load_tables()
        self._index_map: dict[str, Any] = {idx.name: idx for idx in model.indexes}
        self._parameter_axes = [model.inputs.pv_tourists, model.inputs.pv_excursionists]
        # Grid-resolution: structural axis choices, not per-evaluation config.
        self._t_max, self._e_max = 10000, 10000
        self._t_sample, self._e_sample = 100, 100
        # Categorical combinations: 7 weekdays x 3 weather classes (estimator fixed); each
        # combination is replicated with different days of the period and capacities.
        self._config = OvertourismEvaluationConfig(
            ensemble_size=7,
            ensemble_seed=0,
            n_samples_per_combo=self.FIELD_DAYS_PER_COMBO,
            sample_seed=1,
            target_presence_samples=2000,
            confidence=0.8,
        )

    @property
    def model(self) -> MolvenoModel:
        """The live `MolvenoModel` instance.

        Needed by frontends that resolve predefined what-if scenarios against
        a live model.
        """
        return self._model

    def parameter_schema(self) -> list[OvertourismParameterMeta]:
        """Return the ordered, self-describing parameter schema."""
        return list(self._schema.values())

    def schema(self) -> dict[str, Any]:
        """Return Molveno indexes and frontend presentation metadata."""
        return {
            "metadata": {
                "mapper": self.MAPPER,
                "x_axis_name": self.X_AXIS_NAME,
                "y_axis_name": self.Y_AXIS_NAME,
                "x_field": self.X_FIELD,
                "y_field": self.Y_FIELD,
            },
            "indexes": self.parameter_schema(),
        }

    def evaluate(self, param_overrides: dict[str, Any]) -> SustainabilityFieldOutput:
        """Evaluate the model under the given string-keyed parameter overrides.

        The period (day of the summer) is a ``[first, end]`` range: the days from
        ``first`` to ``end`` (excluded); an empty range is widened to one day.
        """
        params = dict(param_overrides)
        period = params.get(DAY_INDEX_NAME)
        if period is not None:
            first, end = (float(v) for v in period)
            params[DAY_INDEX_NAME] = (first, max(end, first + 1.0))
        scenario = build_scenario(
            self._model,
            params,
            self._index_map,
            self._schema,
            self._parameter_axes,
        )
        return self._evaluate_scenario(self._with_period(scenario))

    def period(self, scenario: Scenario) -> tuple[float, float]:
        """``(first, end)`` of the period of *scenario* (default: the whole summer)."""
        dist = scenario.overrides.get(self._model.inputs.i_day_of_summer)
        if dist is None:
            return 0.0, float(self._tables.n_days)
        loc, scale = (
            float(dist.kwds.get("loc", 0.0)),
            float(dist.kwds.get("scale", 1.0)),
        )
        return loc, loc + scale

    def _with_period(self, scenario: Scenario) -> Scenario:
        """Fix the presence estimator of the backend (D9) and, unless the scenario fixes the
        weather, weight the weather classes by the months of the period (A5)."""
        inp = self._model.inputs
        overrides = dict(scenario.overrides)
        overrides[inp.cv_presence_estimator] = self._presence_estimator
        if inp.cv_weather not in overrides:
            first, end = self.period(scenario)
            overrides[inp.cv_weather] = weather_weights(
                self._tables, int(np.floor(first)), int(np.ceil(end))
            )
        return Scenario(
            self._model, overrides=overrides, parameter_axes=self._parameter_axes
        )

    def arrange_data(self, output: SustainabilityFieldOutput) -> dict[str, Any]:
        """Arrange Molveno output for the frontend presentation format."""
        return arrange_frontend_data(output)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @functools.cached_property
    def _schema(self) -> dict[str, OvertourismParameterMeta]:
        """Hand-authored parameter metadata; the single source of truth (§3.1).

        Exposes the three context variables and the full set of model
        parameters, except the presence variables. Percentages are expressed
        as fractions, matching the model's units.
        """
        inp = self._model.inputs
        return {
            inp.cv_weekday.name: OvertourismParameterMeta(
                name=inp.cv_weekday.name,
                kind="categorical",
                label="Giorno della settimana",
                description=(
                    "Filtra la valutazione a un singolo giorno della settimana. "
                    "La domenica registra la maggiore presenza di escursionisti "
                    "(tranne in agosto, quando l'affluenza è alta tutti i giorni)."
                ),
                category="Contesto",
                support=list(inp.cv_weekday.support),
                support_labels=WEEKDAY_LABELS,
            ),
            inp.i_day_of_summer.name: OvertourismParameterMeta(
                name=inp.i_day_of_summer.name,
                kind="distribution",  # the period: the day is uniform over it (D9)
                label="Periodo (giorni dal 1° giugno)",
                description=(
                    "Giorni dell'estate su cui è calcolata la valutazione, contati dal "
                    "1° giugno: 0 = 1° giugno, 30 = 1° luglio, 61 = 1° agosto, "
                    "92 = 1° settembre, 122 = fine settembre. Predefinito: tutta l'estate."
                ),
                category="Contesto",
                default_range=(0.0, float(self._tables.n_days)),
                min_value=0.0,
                max_value=float(self._tables.n_days),
                step=1.0,
                distribution_family="uniform",
                distribution_fixed_params={},
            ),
            inp.cv_weather.name: OvertourismParameterMeta(
                name=inp.cv_weather.name,
                kind="categorical",
                label="Meteo",
                description=(
                    "Meteo del giorno: bel tempo, fresco (temperatura massima di oltre 2 °C "
                    "sotto la media del mese) o pioggia (almeno 5 mm). Senza selezione, "
                    "le frequenze osservate nei mesi del periodo. Con la pioggia "
                    "calano gli escursionisti e la ristorazione ne assorbe una "
                    "quota maggiore."
                ),
                category="Contesto",
                support=list(inp.cv_weather.support),
                support_labels=WEATHER_LABELS,
            ),
            inp.i_c_parking.name: OvertourismParameterMeta(
                name=inp.i_c_parking.name,
                kind="distribution",
                label="Numero di parcheggi disponibili",
                description=(
                    "Numero di posti auto disponibili a Molveno. "
                    "Range [min, max]; riferimento: uniform[350, 450]."
                ),
                category="Parcheggi",
                default_range=(350.0, 450.0),
                min_value=0.0,
                max_value=1000.0,
                step=10.0,
                distribution_family="uniform",
                distribution_fixed_params={},
            ),
            inp.i_u_tourists_parking.name: OvertourismParameterMeta(
                name=inp.i_u_tourists_parking.name,
                kind="scalar",
                label="Percentuale di turisti che usano i parcheggi",
                description=("Frazione di turisti che usano i parcheggi."),
                category="Parcheggi",
                default=0.02,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            ),
            inp.i_u_excursionists_parking.name: OvertourismParameterMeta(
                name=inp.i_u_excursionists_parking.name,
                kind="scalar",
                label="Percentuale di escursionisti che usano i parcheggi",
                description=("Frazione di escursionisti che usano i parcheggi."),
                category="Parcheggi",
                default=0.80,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            ),
            inp.i_xa_tourists_per_vehicle.name: OvertourismParameterMeta(
                name=inp.i_xa_tourists_per_vehicle.name,
                kind="scalar",
                label="Numero medio di turisti per veicolo",
                description=(
                    "Occupazione media (numero medio di persone) nei veicoli "
                    "utilizzati dai turisti."
                ),
                category="Parcheggi",
                default=2.5,
                min_value=0.1,
                max_value=5.0,
                step=0.1,
            ),
            inp.i_xa_excursionists_per_vehicle.name: OvertourismParameterMeta(
                name=inp.i_xa_excursionists_per_vehicle.name,
                kind="scalar",
                label="Numero medio di escursionisti per veicolo",
                description=(
                    "Occupazione media (numero medio di persone) nei veicoli "
                    "utilizzati dagli escursionisti."
                ),
                category="Parcheggi",
                default=2.5,
                min_value=0.1,
                max_value=5.0,
                step=0.1,
            ),
            inp.i_xo_tourists_parking.name: OvertourismParameterMeta(
                name=inp.i_xo_tourists_parking.name,
                kind="scalar",
                label="Ricambi giornalieri per posto auto (turisti)",
                description=(
                    "Numero di veicoli di turisti che possono occupare lo stesso "
                    "posto auto nell'arco della giornata."
                ),
                category="Parcheggi",
                default=1.05,
                min_value=1.0,
                max_value=4.0,
                step=0.05,
            ),
            inp.i_xo_excursionists_parking.name: OvertourismParameterMeta(
                name=inp.i_xo_excursionists_parking.name,
                kind="scalar",
                label="Ricambi giornalieri per posto auto (escursionisti)",
                description=(
                    "Numero di veicoli di escursionisti che possono occupare lo stesso "
                    "posto auto nell'arco della giornata."
                ),
                category="Parcheggi",
                default=3.5,
                min_value=1.0,
                max_value=4.0,
                step=0.05,
            ),
            inp.i_c_beach.name: OvertourismParameterMeta(
                name=inp.i_c_beach.name,
                kind="distribution",
                label="Numero di posti disponibili in spiaggia",
                description=(
                    "Numero massimo di presenze nella spiaggia di Molveno che garantiscono "
                    "un distanziamento adeguato fra le persone. "
                    "Range [min, max]; riferimento: uniform[6000, 7000]."
                ),
                category="Spiaggia",
                default_range=(6000.0, 7000.0),
                min_value=0.0,
                max_value=10000.0,
                step=100.0,
                distribution_family="uniform",
                distribution_fixed_params={},
            ),
            inp.i_u_tourists_beach.name: OvertourismParameterMeta(
                name=inp.i_u_tourists_beach.name,
                kind="scalar",
                label="Percentuale di turisti che usano la spiaggia",
                description=("Frazione di turisti che usano la spiaggia."),
                category="Spiaggia",
                default=0.50,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            ),
            inp.i_u_excursionists_beach.name: OvertourismParameterMeta(
                name=inp.i_u_excursionists_beach.name,
                kind="scalar",
                label="Percentuale di escursionisti che usano la spiaggia",
                description=("Frazione di escursionisti che usano la spiaggia."),
                category="Spiaggia",
                default=0.80,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            ),
            inp.i_xo_tourists_beach.name: OvertourismParameterMeta(
                name=inp.i_xo_tourists_beach.name,
                kind="distribution",
                label="Ricambi giornalieri per posto in spiaggia (turisti)",
                description=(
                    "Numero di turisti che possono occupare lo stesso posto in spiaggia "
                    "nell'arco della giornata. Range [min, max]; riferimento: uniform[1, 3]."
                ),
                category="Spiaggia",
                default_range=(1.0, 3.0),
                min_value=1.0,
                max_value=4.0,
                step=0.05,
                distribution_family="uniform",
                distribution_fixed_params={},
            ),
            inp.i_xo_excursionists_beach.name: OvertourismParameterMeta(
                name=inp.i_xo_excursionists_beach.name,
                kind="scalar",
                label="Ricambi giornalieri per posto in spiaggia (escursionisti)",
                description=(
                    "Numero di escursionisti che possono occupare lo stesso posto in "
                    "spiaggia nell'arco della giornata."
                ),
                category="Spiaggia",
                default=1.05,
                min_value=1.0,
                max_value=4.0,
                step=0.05,
            ),
            inp.i_c_accommodation.name: OvertourismParameterMeta(
                name=inp.i_c_accommodation.name,
                kind="distribution",
                label="Posti letto disponibili",
                description=(
                    "Totale posti letto nelle strutture ricettive alberghiere e "
                    "extralberghiere a Molveno. Distribuzione lognormale (s = 0.125) "
                    "con loc = min e scale = max - min; riferimento: loc 0, scale 5000."
                ),
                category="Alberghi",
                default_range=(0.0, 5000.0),
                min_value=0.0,
                max_value=10000.0,
                step=100.0,
                distribution_family="lognorm",
                distribution_fixed_params={"s": 0.125},
            ),
            inp.i_u_tourists_accommodation.name: OvertourismParameterMeta(
                name=inp.i_u_tourists_accommodation.name,
                kind="scalar",
                label="Percentuale di turisti che alloggiano in strutture ricettive",
                description=(
                    "Frazione di turisti che alloggiano in strutture ricettive alberghiere "
                    "o extralberghiere a Molveno."
                ),
                category="Alberghi",
                default=0.90,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            ),
            inp.i_xa_tourists_accommodation.name: OvertourismParameterMeta(
                name=inp.i_xa_tourists_accommodation.name,
                kind="scalar",
                label="Fattore massimo di allocazione dei posti letto",
                description=(
                    "Fattore di allocazione che rappresenta la frazione di posti letto "
                    "occupati nel caso di alberghi pieni (tenendo conto ad es. di camere "
                    "doppie uso singola)."
                ),
                category="Alberghi",
                default=0.85,
                min_value=0.5,
                max_value=1.25,
                step=0.05,
            ),
            inp.i_c_food.name: OvertourismParameterMeta(
                name=inp.i_c_food.name,
                kind="distribution",
                label="Posti a sedere nei ristoranti",
                description=(
                    "Totale posti a sedere nella ristorazione commerciale a Molveno. "
                    "Range [min, max]; riferimento: triang moda 2800, range [2400, 3200]."
                ),
                category="Ristoranti",
                default_range=(2400.0, 3200.0),
                min_value=0.0,
                max_value=6000.0,
                step=100.0,
                distribution_family="triang",
                distribution_fixed_params={"c": 0.5},
            ),
            inp.i_u_tourists_food.name: OvertourismParameterMeta(
                name=inp.i_u_tourists_food.name,
                kind="scalar",
                label="Percentuale di turisti che usano i ristoranti",
                description=("Frazione di turisti che usano i ristoranti."),
                category="Ristoranti",
                default=0.20,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            ),
            inp.i_u_excursionists_food_rainy_weather.name: OvertourismParameterMeta(
                name=inp.i_u_excursionists_food_rainy_weather.name,
                kind="scalar",
                label="Percentuale di escursionisti che usano i ristoranti (pioggia)",
                description=(
                    "Frazione di escursionisti che usano i ristoranti con la pioggia."
                ),
                category="Ristoranti",
                default=0.80,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            ),
            inp.i_u_excursionists_food_good_cool_weather.name: OvertourismParameterMeta(
                name=inp.i_u_excursionists_food_good_cool_weather.name,
                kind="scalar",
                label="Percentuale di escursionisti che usano i ristoranti (tempo bello / fresco)",
                description=(
                    "Frazione di escursionisti che usano i ristoranti "
                    "con bel tempo o tempo fresco."
                ),
                category="Ristoranti",
                default=0.40,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            ),
            inp.i_xa_visitors_food.name: OvertourismParameterMeta(
                name=inp.i_xa_visitors_food.name,
                kind="scalar",
                label="Fattore massimo di allocazione dei posti a sedere nei ristoranti",
                description=(
                    "Fattore di allocazione che rappresenta la frazione di posti a sedere "
                    "occupati nel caso di ristoranti pieni (tenendo conto ad es. di tavoli "
                    "non completi)."
                ),
                category="Ristoranti",
                default=0.90,
                min_value=0.5,
                max_value=1.5,
                step=0.05,
            ),
            inp.i_xo_visitors_food.name: OvertourismParameterMeta(
                name=inp.i_xo_visitors_food.name,
                kind="scalar",
                label="Tasso di rotazione dei tavoli",
                description=(
                    "Numero di volte in cui un tavolo viene occupato nell'arco di un servizio."
                ),
                category="Ristoranti",
                default=2.0,
                min_value=0.5,
                max_value=4.0,
                step=0.05,
            ),
            inp.i_p_tourists_reduction_factor.name: OvertourismParameterMeta(
                name=inp.i_p_tourists_reduction_factor.name,
                kind="scalar",
                label="Fattore di variazione di presenze turistiche",
                description=(
                    "Aumenta (> 1) o diminuisce (< 1) la presenza stimata di "
                    "turisti rispetto al valore storico."
                ),
                category="Flussi",
                default=1.0,
                min_value=0.05,
                max_value=4.0,
                step=0.05,
            ),
            inp.i_p_excursionists_reduction_factor.name: OvertourismParameterMeta(
                name=inp.i_p_excursionists_reduction_factor.name,
                kind="scalar",
                label="Fattore di variazione di presenze escursionistiche",
                description=(
                    "Aumenta (> 1) o diminuisce (< 1) la presenza stimata di "
                    "escursionisti rispetto al valore storico."
                ),
                category="Flussi",
                default=1.0,
                min_value=0.05,
                max_value=4.0,
                step=0.05,
            ),
            inp.i_p_tourists_saturation_level.name: OvertourismParameterMeta(
                name=inp.i_p_tourists_saturation_level.name,
                kind="scalar",
                label="Soglia di saturazione massima turisti",
                description=(
                    "Livello massimo che può raggiungere la presenza di "
                    "turisti a Molveno durante una giornata."
                ),
                category="Flussi",
                default=10000.0,
                min_value=1000.0,
                max_value=20000.0,
                step=100.0,
            ),
            inp.i_p_excursionists_saturation_level.name: OvertourismParameterMeta(
                name=inp.i_p_excursionists_saturation_level.name,
                kind="scalar",
                label="Soglia di saturazione massima escursionisti",
                description=(
                    "Livello massimo che può raggiungere la presenza di "
                    "escursionisti a Molveno durante una giornata."
                ),
                category="Flussi",
                default=10000.0,
                min_value=1000.0,
                max_value=20000.0,
                step=100.0,
            ),
        }

    def _evaluate_scenario(self, scenario: Scenario) -> SustainabilityFieldOutput:
        """Run the seeded field + presence-sample evaluation for `scenario`."""
        model = self._model
        config = self._config
        pv_t, pv_e = model.inputs.pv_tourists, model.inputs.pv_excursionists
        first, end = self.period(scenario)
        tt = np.linspace(0, self._t_max, self._t_sample + 1)
        ee = np.linspace(0, self._e_max, self._e_sample + 1)

        # Raw presence samples — seeded independently of the field. Transformed
        # below (after the field result is available) via _presence_transformation.
        sampling_overrides = {
            idx: (
                [val]
                if isinstance(idx, CategoricalIndex) and isinstance(val, str)
                else val
            )
            for idx, val in scenario.overrides.items()
        }
        sampling_scenario = Scenario(
            model, overrides=sampling_overrides, parameter_axes=[pv_t, pv_e]
        )
        sampling_ensemble = CrossProductEnsemble(
            sampling_scenario,
            max_categorical_size=config.ensemble_size,
            n_samples_per_combo=self.SAMPLE_DAYS_PER_COMBO,
            rng=np.random.default_rng(config.sample_seed),
        )
        pv_samples = sample_across(
            sampling_ensemble,
            [pv_t, pv_e],
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
            parameters={pv_t: tt, pv_e: ee},
        )
        field, field_elements, usage_fields, capacity_distributions = (
            compute_sustainability_field(
                model.constraints, result, pv_t, pv_e, scenario=scenario
            )
        )

        # Presence-saturation transform: reduction/saturation factors are
        # derived indexes, read from the evaluated result (mean over the ensemble).
        rf_t = float(np.mean(result[model.inputs.i_p_tourists_reduction_factor]))
        sl_t = float(np.mean(result[model.inputs.i_p_tourists_saturation_level]))
        rf_e = float(np.mean(result[model.inputs.i_p_excursionists_reduction_factor]))
        sl_e = float(np.mean(result[model.inputs.i_p_excursionists_saturation_level]))
        samples_x = [_presence_transformation(s, rf_t, sl_t) for s in pv_samples[pv_t]]
        samples_y = [_presence_transformation(s, rf_e, sl_e) for s in pv_samples[pv_e]]

        return SustainabilityFieldOutput(
            field=field,
            field_elements=field_elements,
            x_values=tt,
            y_values=ee,
            samples_x=samples_x,
            samples_y=samples_y,
            usage_fields=usage_fields,
            capacity_distributions=capacity_distributions,
            confidence=config.confidence,
            # the incertezza of the sustainability: from one year to another, over the days
            # of the period
            period_days=expected_period_days(
                end - first,
                scenario.overrides,
                [model.inputs.cv_weekday, model.inputs.cv_weather],
            ),
        )
