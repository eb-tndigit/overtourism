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
)

from src.model.common.labels import WEATHER_LABELS
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
    DAY_TYPE_LABELS,
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


def booking_losses(
    potential: np.ndarray,
    capacity: np.ndarray,
    usage: np.ndarray,
    per_vehicle: float,
    rotation: float,
    booked: float,
) -> np.ndarray:
    """Excursionists who give up the day when *booked* paid spaces are booking-only (D15, A13).

    The *potential* excursionists bring ``potential x usage / per_vehicle`` cars to the paid
    car parks, which take one car per booked space and *rotation* cars per other space; of
    the cars above that, the booked share of the spaces gives up, the rest park elsewhere.
    """
    booked = np.minimum(booked, capacity)
    cars = potential * usage / per_vehicle
    supply = booked + (capacity - booked) * rotation
    return np.maximum(cars - supply, 0.0) * booked / capacity * per_vehicle


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
    KPI_LABELS: ClassVar[dict[str, str]] = {
        "excursionists mean": "Escursionisti rispetto ai potenziali: media del periodo (persone/giorno)",
        "excursionists peak": "Escursionisti rispetto ai potenziali: giorni di picco (persone/giorno)",
    }
    PEAK_QUANTILE = 0.95  # peak days: the busiest 5 % of the simulated days
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
                "kpi_labels": self.KPI_LABELS,
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
                    "Filtra la valutazione a un tipo di giorno: da lunedì a venerdì, sabato "
                    "o domenica. La domenica registra la maggiore presenza di escursionisti "
                    "(tranne in agosto, quando l'affluenza è alta tutti i giorni)."
                ),
                category="Contesto",
                support=list(inp.cv_weekday.support),
                support_labels=DAY_TYPE_LABELS,
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
                label="Posti auto nei parcheggi pubblici",
                description=(
                    "Posti auto dei parcheggi pubblici del comune: 713, di cui 529 a pagamento "
                    "(Ischia 195, Lungolago, Seggiovia, Centro storico, Palazzetto, Piscina, "
                    "Rio Massò, Pozze) e 184 a disco orario o liberi (Comune di Molveno, 2024). "
                    "Range [min, max]; riferimento: uniform[683, 743]."
                ),
                category="Parcheggi",
                default_range=(683.0, 743.0),
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
                kind="distribution",
                label="Percentuale di escursionisti che usano i parcheggi",
                description=(
                    "Frazione di escursionisti che usano i parcheggi, cioè che arrivano in "
                    "auto: 78% nell'indagine dell'estate 2023, confermato dai "
                    "contamacchine (0,31 auto in entrata nella zona spiaggia per "
                    "escursionista, con 2,5 persone per auto). Range [min, max]; "
                    "riferimento: uniform[0.70, 0.86]."
                ),
                category="Parcheggi",
                default_range=(0.70, 0.86),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                distribution_family="uniform",
                distribution_fixed_params={},
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
                    "posto auto nell'arco della giornata: nei giorni affollati del 2023 "
                    "la sosta pagata media è di 5,1 ore, su 10 ore a pagamento (8-18)."
                ),
                category="Parcheggi",
                default=2.0,
                min_value=1.0,
                max_value=4.0,
                step=0.05,
            ),
            inp.i_booked_parking_spaces.name: OvertourismParameterMeta(
                name=inp.i_booked_parking_spaces.name,
                kind="scalar",
                label="Posti auto solo su prenotazione",
                description=(
                    "Posti dei parcheggi a pagamento accessibili solo su prenotazione, "
                    "a giornata intera (un'auto per posto al giorno). Dal 2025 il parcheggio "
                    "Ischia: 195 posti; 529 = tutti i parcheggi a pagamento, 0 = nessuno. "
                    "Gli escursionisti in auto oltre i posti disponibili rinunciano in "
                    "proporzione alla quota di posti su prenotazione; gli altri sostano "
                    "altrove (alberghi, aree private, lungo la strada)."
                ),
                category="Parcheggi",
                default=195.0,
                min_value=0.0,
                max_value=529.0,
                step=1.0,
            ),
            inp.i_c_beach.name: OvertourismParameterMeta(
                name=inp.i_c_beach.name,
                kind="distribution",
                label="Numero di posti disponibili in spiaggia",
                description=(
                    "Numero massimo di presenze nella spiaggia di Molveno che garantiscono "
                    "un distanziamento adeguato fra le persone. Ipotesi, senza fonte: "
                    "almeno il massimo osservato nell'estate 2023 (4.676 presenze "
                    "contemporanee, senza segni di saturazione). Range [min, max]; "
                    "riferimento: uniform[4700, 5500]."
                ),
                category="Spiaggia",
                default_range=(4700.0, 5500.0),
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
                description=(
                    "Frazione di turisti che vanno in spiaggia nella giornata: 0,35 dai "
                    "contapersone ai varchi (estate 2023; 0,25-0,47), contro 0,78 degli "
                    "escursionisti."
                ),
                category="Spiaggia",
                default=0.35,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            ),
            inp.i_u_excursionists_beach.name: OvertourismParameterMeta(
                name=inp.i_u_excursionists_beach.name,
                kind="scalar",
                label="Percentuale di escursionisti che usano la spiaggia",
                description=(
                    "Frazione di escursionisti che vanno in spiaggia nella giornata: 0,78 dai "
                    "contapersone ai varchi (estate 2023; 0,65-0,87)."
                ),
                category="Spiaggia",
                default=0.78,
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
                    "nell'arco della giornata: ingressi giornalieri / massimo di presenti "
                    "ai contapersone (estate 2023), 2,1 (metà dei giorni fra 1,6 e 2,4). "
                    "Range [min, max]; riferimento: uniform[1.6, 2.4]."
                ),
                category="Spiaggia",
                default_range=(1.6, 2.4),
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
                    "spiaggia nell'arco della giornata: ingressi giornalieri / massimo di "
                    "presenti ai contapersone (estate 2023), 2,1."
                ),
                category="Spiaggia",
                default=2.1,
                min_value=1.0,
                max_value=4.0,
                step=0.05,
            ),
            inp.i_c_accommodation.name: OvertourismParameterMeta(
                name=inp.i_c_accommodation.name,
                kind="distribution",
                label="Posti letto disponibili",
                description=(
                    "Posti letto nelle strutture ricettive alberghiere ed extralberghiere "
                    "di Molveno: 3.429 nel 2023, 3.503 nel 2025 (ISPAT). Alloggi privati e "
                    "seconde case non sono considerati. Range [min, max]; riferimento: "
                    "uniform[3429, 3503]."
                ),
                category="Alberghi",
                default_range=(3429.0, 3503.0),
                min_value=0.0,
                max_value=10000.0,
                step=10.0,
                distribution_family="uniform",
                distribution_fixed_params={},
            ),
            inp.i_u_tourists_accommodation.name: OvertourismParameterMeta(
                name=inp.i_u_tourists_accommodation.name,
                kind="distribution",
                label="Percentuale di turisti che alloggiano in strutture ricettive",
                description=(
                    "Frazione di turisti che alloggiano in strutture ricettive alberghiere "
                    "o extralberghiere (campeggio compreso) a Molveno: nell'estate 2023 i "
                    "pernottamenti ISPAT sono da 0,64 a 0,93 volte i turisti stimati da "
                    "Vodafone (10-90% dei giorni; mediana 0,80). Range [min, max]; "
                    "riferimento: uniform[0.65, 0.90]."
                ),
                category="Alberghi",
                default_range=(0.65, 0.90),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                distribution_family="uniform",
                distribution_fixed_params={},
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
                default=1.0,
                min_value=0.5,
                max_value=1.25,
                step=0.05,
            ),
            inp.i_c_food.name: OvertourismParameterMeta(
                name=inp.i_c_food.name,
                kind="distribution",
                label="Posti a sedere nei ristoranti",
                description=(
                    "Posti a sedere di ristoranti e bar dell'area del lago, paese e Pradel "
                    "compresi: da 2.768 a 2.883 (alcuni locali indicano un intervallo; 745 "
                    "nella zona spiaggia). Range [min, max]; riferimento: uniform[2768, 2883]."
                ),
                category="Ristoranti",
                default_range=(2768.0, 2883.0),
                min_value=0.0,
                max_value=6000.0,
                step=10.0,
                distribution_family="uniform",
                distribution_fixed_params={},
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

        # Simulated days for the scatter and the KPIs — seeded independently of the field.
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
        days = self._simulate_days(sampling_ensemble, scenario)

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
        samples_x = [_presence_transformation(s, rf_t, sl_t) for s in days["tourists"]]
        samples_y = [_presence_transformation(s, rf_e, sl_e) for s in days["excursionists"]]
        potential = days["potential_excursionists"]
        peak = potential >= np.quantile(potential, self.PEAK_QUANTILE)
        kpis = {
            which: {
                "value": round(float(days["excursionists"][sel].mean()), 1),
                "share": round(
                    float(days["excursionists"][sel].sum() / potential[sel].sum()) * 100, 2
                ),
            }
            for which, sel in (("mean", np.ones(potential.size, dtype=bool)), ("peak", peak))
        }

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
            extra_kpis={f"excursionists {which}": kpi for which, kpi in kpis.items()},
        )

    def _simulate_days(
        self, ensemble: CrossProductEnsemble, scenario: Scenario
    ) -> dict[str, np.ndarray]:
        """The days of the scenario: tourists and excursionists of each day, with the
        excursionists who give up when the car parks are booking-only (D15, A13).

        Each member of *ensemble* (a weekday x weather combination with a day of the period
        and a draw of the uncertain parameters) gives ``max(1, round(weight x
        target_presence_samples))`` days. The potential excursionists of a day bring
        ``E x u / xa`` cars to the paid car parks, which take ``B`` cars on the booked
        spaces (one per space and day) and ``xo`` per space on the others; of the cars above
        that, the share ``B / capacity`` gives up (the booked share of the spaces), the rest
        park elsewhere (:func:`booking_losses`).
        """
        inp = self._model.inputs
        config = self._config
        values = {id(idx): np.asarray(v) for idx, v in ensemble.assignments().items()}
        weights = ensemble.ensemble_weights[0]
        rng = np.random.default_rng(config.sample_seed)

        def scalar(idx) -> float:
            return float(scenario.overrides.get(idx, idx.concrete_default))

        member, draws = [], {inp.pv_tourists: [], inp.pv_excursionists: []}
        for i, w in enumerate(weights):
            n = max(1, round(float(w) * config.target_presence_samples))
            for idx, out in draws.items():
                parents = {p.name: values[id(p)][i] for p in idx.parents}
                d = idx.distribution_for(**parents).rvs(size=n, random_state=rng)
                out.append(np.asarray(d, dtype=float).ravel())
            member.append(np.full(n, i))
        member = np.concatenate(member)
        tourists = np.concatenate(draws[inp.pv_tourists])
        potential = np.concatenate(draws[inp.pv_excursionists])

        lost = booking_losses(
            potential,
            capacity=values[id(inp.i_c_parking)].astype(float)[member],
            usage=values[id(inp.i_u_excursionists_parking)].astype(float)[member],
            per_vehicle=scalar(inp.i_xa_excursionists_per_vehicle),
            rotation=scalar(inp.i_xo_excursionists_parking),
            booked=scalar(inp.i_booked_parking_spaces),
        )
        return {
            "tourists": tourists,
            "potential_excursionists": potential,
            "excursionists": potential - lost,
        }
