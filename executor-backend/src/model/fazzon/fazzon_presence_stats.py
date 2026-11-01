"""Presence statistics for the Fazzon (Lago dei Caprioli, Val di Sole) overtourism model.

Two presence variables are modelled:

``pv_visitors_car``
    Daily visitors arriving by private car.

``pv_visitors_other``
    Daily visitors arriving by non-car modes (shuttle bus, on foot, bicycle).

Both are the **arrivals** at Fazzon, conditioned on the season, the weather class (good /
cool / rainy, D7 in ``data_preparation/model/fazzon/ASSUMPTIONS.md``), the parking price
level and whether the shuttle runs (D15); there is no day-type split (D5).  They are built
from the demand in 2025 conditions (5 euro parking, no shuttle; ``fazzon_presence_stats``)
and the behaviour model (``fazzon_modal_shift``, D8):

* car arrivals   = car demand × R(price),  R(p) = exp(-β (p - 5))  (A23, A27);
* non-car arrivals = non-car demand + car demand × (1 - R(price)) × walk-up share
  + shuttle riders  (A3, A6);
* shuttle riders = min(car demand × (1 - R(price)) × shuttle share, usable shuttle
  capacity)  (A28, A30, A33, A34);
* the rest of the deterred car demand goes elsewhere and does not arrive.

The data has ``P(weather | season)`` (climatology, A20); the model uses one weather
distribution for all seasons, their weighted average (D13).

Data source
-----------
The tables are the reference datasets produced by the Fazzon data-preparation pipeline
(``data_preparation/model/fazzon``): ``fazzon_presence_stats`` (one row per (season,
weather) cell) and ``fazzon_modal_shift`` (behaviour parameters).  Their calibration and
assumptions are listed in that folder's ``ASSUMPTIONS.md`` (A2: persons per vehicle;
A3: non-car share 0.31 in 2025 conditions; A4: non-car SD 50 % of the mean; A14: season
weights; A19, A20: weather classes and frequencies; A37, A38: demand per season and weather
from the valley traffic; A6, A23: modal shift).  It is read from the Fazzon data folder (see
:mod:`src.model.fazzon.fazzon_data`) every time :func:`presence_inputs` is called,
i.e. every time a model is built: there are no built-in values, and a missing or
malformed file is an error.
"""

# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .fazzon_data import MODAL_SHIFT_FILE, PRESENCE_STATS_FILE, fazzon_data_dir

logger = logging.getLogger(__name__)

SEASONS: tuple[str, ...] = ("june", "july", "august", "september")
WEATHERS: tuple[str, ...] = ("good", "cool", "rainy")
PRESENCE_STATS_COLUMNS: tuple[str, ...] = (
    "season_freq_rel",
    "weather_freq_rel",
    "mean_car_visitors",
    "std_car_visitors",
    "mean_other_visitors",
    "std_other_visitors",
)


# ---------------------------------------------------------------------------
# Loading the reference dataset
# ---------------------------------------------------------------------------


def load_presence_stats(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Return the presence statistics read from *data_dir*, indexed by (season, weather).

    *data_dir* defaults to :func:`~src.model.fazzon.fazzon_data.fazzon_data_dir`.
    A missing file, columns or (season, weather) cells raise an error.
    """
    data_dir = Path(data_dir) if data_dir is not None else fazzon_data_dir()
    path = data_dir / PRESENCE_STATS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Fazzon presence statistics not found: {path}")

    df = pd.read_parquet(path)
    lacking = sorted({"season", "weather", *PRESENCE_STATS_COLUMNS} - set(df.columns))
    if lacking:
        raise ValueError(
            f"Fazzon presence statistics in {data_dir}: missing columns {lacking}"
        )
    df = df.set_index(["season", "weather"])[list(PRESENCE_STATS_COLUMNS)]
    expected = {(s, w) for s in SEASONS for w in WEATHERS}
    if set(df.index) != expected:
        raise ValueError(
            f"Fazzon presence statistics: (season, weather) cells {sorted(df.index)}, "
            f"expected {sorted(expected)}"
        )
    logger.info("Fazzon presence statistics loaded from %s", data_dir)
    order = pd.MultiIndex.from_tuples(
        [(s, w) for s in SEASONS for w in WEATHERS], names=["season", "weather"]
    )
    return df.loc[order]


# ---------------------------------------------------------------------------
# Conditional distribution factories
# ---------------------------------------------------------------------------


PRICE_INDEX_NAME = "parking price (euro per vehicle)"
SHUTTLE_INDEX_NAME = "shuttle capacity (riders per day)"
PRICE_LEVELS: tuple[int, ...] = (5, 10, 15, 20, 25, 30, 35, 40)  # D15: 5-euro classes
SHUTTLE_CAPACITY_STEP = (
    60  # D16: capacity classes of 60 riders (one run of the 2026 Lakebus)
)
SHUTTLE_CAPACITY_LEVELS: tuple[int, ...] = tuple(
    range(0, 3600 + 1, SHUTTLE_CAPACITY_STEP)
)


def shuttle_usable_capacity(
    buses: float,
    seats_per_bus: float,
    interval_min: float,
    start_hour: float,
    end_hour: float,
    peak_end_hour: float,
    peak_share: float,
) -> float:
    """Riders per day the shuttle can carry up (A30, A35).

    All buses leave together every *interval_min* minutes from *start_hour*; each run takes
    one interval (up and back), the last one ends at *end_hour* (end of service), so there
    are ``(end_hour - start_hour) * 60 / interval_min`` runs (Lakebus: 9:00-18:30
    departures, back at 19:00: 20 runs).
    A share *peak_share* of the riders wants to go up between *start_hour* and
    *peak_end_hour*, so the usable capacity is the smaller of the daily seats and the peak
    seats divided by the peak share.
    """
    if buses <= 0 or seats_per_bus <= 0 or interval_min <= 0 or end_hour <= start_hour:
        return 0.0
    seats_per_run = buses * seats_per_bus
    daily = (end_hour - start_hour) * 60.0 / interval_min * seats_per_run
    peak_runs = (
        max(min(peak_end_hour, end_hour) - start_hour, 0.0) * 60.0 / interval_min
    )
    if peak_share <= 0:
        return daily
    return min(daily, peak_runs * seats_per_run / peak_share)


def shuttle_capacity_level(capacity: float) -> str:
    """Capacity class (D16) of a usable capacity: nearest multiple of the step, clipped."""
    level = int(round(capacity / SHUTTLE_CAPACITY_STEP) * SHUTTLE_CAPACITY_STEP)
    return str(min(max(level, 0), SHUTTLE_CAPACITY_LEVELS[-1]))


def price_response(price: float, behaviour: dict[str, float]) -> float:
    """R(p): share of the car demand still arriving by private vehicle (A23; A27: 1 below the
    reference price)."""
    excess = max(price - behaviour["reference_price"], 0.0)
    return float(np.exp(-behaviour["beta_per_euro"] * excess))


def make_presence_factories(stats_table: pd.DataFrame, behaviour: dict[str, float]):
    """Return ``(car_arrivals_stats, other_arrivals_stats)`` closures.

    Both take the parent values as keyword arguments (``season``, ``weather``, the price
    level and the shuttle level, keyed by the index names) and return a **Gamma**
    distribution with mean μ and variance σ²::

        k = μ² / σ²   (shape)
        θ = σ² / μ    (scale)

    Car arrivals scale the car demand by R(price) (mean and SD).  Non-car arrivals add the
    walkers and the shuttle riders to the non-car demand; the riders are capped by the
    shuttle capacity class on the cell mean (A33), and the shifted part keeps the car
    demand's coefficient of variation; non-car demand and shifted part are taken as
    independent (A32), so the variances add.  The mean is floored at 1.0 before computing
    the Gamma parameters.
    """

    def _gamma(mean: float, var: float):
        mean = max(mean, 1.0)
        var = max(var, 1e-6)
        k = mean**2 / var
        theta = var / mean
        return stats.gamma(a=k, scale=theta)

    def _cell(kwargs):
        cell = stats_table.loc[(kwargs["season"], kwargs["weather"])]
        price = float(kwargs[PRICE_INDEX_NAME])
        capacity = float(
            kwargs.get(SHUTTLE_INDEX_NAME, 0)
        )  # not a parent of the car arrivals
        return cell, price_response(price, behaviour), capacity

    def car_arrivals_stats(**kwargs):
        cell, r, _ = _cell(kwargs)
        return _gamma(
            cell["mean_car_visitors"] * r, (cell["std_car_visitors"] * r) ** 2
        )

    def other_arrivals_stats(**kwargs):
        cell, r, capacity = _cell(kwargs)
        deterred = cell["mean_car_visitors"] * (1.0 - r)
        shuttle_demand = deterred * behaviour["share_shuttle"]
        riders = min(
            shuttle_demand, capacity
        )  # A33; A28: no shuttle -> capacity 0; A34: excess elsewhere
        shuttle_part = (
            riders / cell["mean_car_visitors"] if cell["mean_car_visitors"] > 0 else 0.0
        )
        shifted = (1.0 - r) * behaviour["share_walk_up"] + shuttle_part
        mean = cell["mean_other_visitors"] + cell["mean_car_visitors"] * shifted
        var = (
            cell["std_other_visitors"] ** 2 + (cell["std_car_visitors"] * shifted) ** 2
        )  # A32
        return _gamma(mean, var)

    def total_arrivals_stats(**kwargs):
        """All arrivals = car + non-car arrivals, taken as independent (A32): means and
        variances add.  (The backend's scatter samples add a car and a non-car draw of the
        same day instead, so that total >= car on every sampled day.)"""
        car, other = car_arrivals_stats(**kwargs), other_arrivals_stats(**kwargs)
        return _gamma(car.mean() + other.mean(), car.var() + other.var())

    car_arrivals_stats.total = total_arrivals_stats  # see make_presence_factories
    return car_arrivals_stats, other_arrivals_stats


def presence_inputs(data_dir: str | Path | None = None):
    """Return ``(season_weights, weather_weights, behaviour, car_arrivals_stats,
    other_arrivals_stats)``.

    * ``season_weights``: ``{season: P(season)}`` (A14);
    * ``weather_weights``: ``{weather: sum_s P(s) P(weather | s)}`` (A20, D13): the weather
      category does not depend on the season in the model, because the modelling library
      cannot fix the season (or the weather) of a season-dependent category in a scenario;
    * ``behaviour``: the behaviour parameters (:func:`behaviour_params`).

    Reads the reference datasets; called by ``FazzonModel.default_inputs`` each time a model
    is built, so a model built after a data download uses the new statistics.
    """
    table = load_presence_stats(data_dir)
    behaviour = behaviour_params(data_dir)
    season_weights = {
        s: float(table.loc[(s, WEATHERS[0]), "season_freq_rel"]) for s in SEASONS
    }
    weather_weights = {
        w: sum(
            season_weights[s] * float(table.loc[(s, w), "weather_freq_rel"])
            for s in SEASONS
        )
        for w in WEATHERS
    }
    total = sum(weather_weights.values())
    weather_weights = {w: p / total for w, p in weather_weights.items()}
    car_stats, other_stats = make_presence_factories(table, behaviour)
    return season_weights, weather_weights, behaviour, car_stats, other_stats


# ---------------------------------------------------------------------------
# Behaviour parameters (modal shift, D8)
# ---------------------------------------------------------------------------

# parameter name in fazzon_modal_shift -> ``days`` sample it is taken from
BEHAVIOUR_PARAMETERS: dict[str, str] = {
    "reference_price": "",
    "beta_per_euro": "",
    "occupancy_low_price": "",
    "occupancy_low": "",
    "occupancy_high_price": "",
    "occupancy_high": "",
    "share_shuttle": "with_walk_up",  # same days as the walk-up share (from 01/08/2026)
    "share_walk_up": "with_walk_up",
}


def behaviour_params(data_dir: str | Path | None = None) -> dict[str, float]:
    """Point estimates of the behaviour model read from ``fazzon_modal_shift`` (A2, A6, A23).

    A missing file or parameter raises an error.
    """
    data_dir = Path(data_dir) if data_dir is not None else fazzon_data_dir()
    path = data_dir / MODAL_SHIFT_FILE
    if not path.exists():
        raise FileNotFoundError(f"Fazzon modal-shift parameters not found: {path}")
    df = pd.read_parquet(path)
    lacking = sorted({"parameter", "days", "estimate"} - set(df.columns))
    if lacking:
        raise ValueError(
            f"Fazzon modal-shift parameters in {data_dir}: missing columns {lacking}"
        )
    values = df.set_index(["parameter", "days"])["estimate"]
    missing = [p for p, d in BEHAVIOUR_PARAMETERS.items() if (p, d) not in values.index]
    if missing:
        raise ValueError(
            f"Fazzon modal-shift parameters in {data_dir}: missing {missing}"
        )
    return {p: float(values[(p, d)]) for p, d in BEHAVIOUR_PARAMETERS.items()}


# ---------------------------------------------------------------------------
# Dropouts: potential visitors who do not arrive (KPI)
# ---------------------------------------------------------------------------

DROPOUT_PEAK_QUANTILE = 0.95  # "maximum": the 95th percentile of the days (A36)


def dropouts(
    stats_table: pd.DataFrame,
    behaviour: dict[str, float],
    cell_weights: dict[tuple[str, str], float],
    price: float,
    capacity: float,
    n_days: int = 20_000,
    seed: int = 0,
) -> dict[str, float]:
    """Potential visitors who do not arrive, per day: season mean and peak (A36).

    Days are drawn from the (season, weather) cells with *cell_weights*; on each day the car
    demand in 2025 conditions is drawn from the cell's Gamma distribution, and

        deterred  = demand × (1 - R(price))
        riders    = min(deterred × shuttle share, capacity)     (cap per day)
        dropouts  = deterred × (1 - walk-up share) - riders

    i.e. the visitors deterred by the price who neither walk up nor find a seat on the
    shuttle (A28, A34).  The potential visitors of a day are its car demand plus the
    cell's non-car demand in proportion (A3).  Returns ``mean`` and ``share`` (mean
    dropouts / mean potential visitors), ``peak`` (the ``DROPOUT_PEAK_QUANTILE`` of the
    daily dropouts) and ``peak_share`` (dropouts / potential visitors over the days at or
    above the peak).
    """
    rng = np.random.default_rng(seed)
    cells = list(cell_weights)
    weights = np.array([cell_weights[c] for c in cells], dtype=float)
    weights = weights / weights.sum()
    picked = rng.choice(len(cells), size=n_days, p=weights)
    demand = np.empty(n_days)
    potential = np.empty(n_days)
    for k, cell in enumerate(cells):
        mask = picked == k
        row = stats_table.loc[cell]
        mean, sd = float(row["mean_car_visitors"]), float(row["std_car_visitors"])
        shape, scale = (mean / sd) ** 2, sd**2 / mean
        demand[mask] = rng.gamma(shape, scale, size=int(mask.sum()))
        potential[mask] = demand[mask] * (
            1.0 + float(row["mean_other_visitors"]) / mean
        )
    deterred = demand * (1.0 - price_response(price, behaviour))
    riders = np.minimum(deterred * behaviour["share_shuttle"], capacity)
    lost = deterred * (1.0 - behaviour["share_walk_up"]) - riders
    peak = float(np.quantile(lost, DROPOUT_PEAK_QUANTILE))
    peak_days = lost >= peak
    return {
        "mean": float(lost.mean()),
        "share": float(lost.mean() / potential.mean()),
        "peak": peak,
        "peak_share": float(lost[peak_days].sum() / potential[peak_days].sum()),
    }
