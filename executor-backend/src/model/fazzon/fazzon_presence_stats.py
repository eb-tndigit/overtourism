"""Presence statistics for the Altipiano di Fazzon (Val di Sole) overtourism model.

The presence variables (the axes of the field):

``pv_visitors_car``
    Daily visitors arriving by private car.

``pv_visitors_total``
    Daily visitors arriving by any mode (car + shuttle bus, on foot, bicycle).

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

Uncertainty (A39): the arrivals also depend on four uncertain quantities, distribution
indexes of the model sampled by the ensemble: the demand level D (A22), the persons per
vehicle at the low and at the high price L, H (A2) and the vehicle response k (cars / cars
expected at 5 euro on the 2026 regime days, measured; A23).  β and the shuttle and walk-up
shares are derived from them on every draw (:func:`behaviour_for`), consistently with the
2026 observations; at the point values they are the estimates of ``fazzon_modal_shift``.

The scatter and the KPIs come instead from simulated days (A36): the demand of a day in 2025
conditions (:func:`make_demand_factories`) and what happens on it
(:func:`daily_accounting`).

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
PRICE_LEVELS: tuple[int, ...] = tuple(range(5, 31))  # D15: 1-euro classes, 5-30 euro
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


VEHICLE_RESPONSE_NAME = "vehicle response at the 2026 regime price"
OCCUPANCY_LOW_NAME = "persons per vehicle at low price"
OCCUPANCY_HIGH_NAME = "persons per vehicle at high price"
DEMAND_LEVEL_NAME = "demand level"


def behaviour_for(
    behaviour: dict[str, float], k: float, low: float, high: float
) -> dict[str, float]:
    """The behaviour parameters for one draw of the vehicle response *k* and of the persons
    per vehicle at the low and high price *low*, *high* (A39).

    The 2026 regime days are fixed observations: cars C, shuttle riders S and walk-ups W.
    The visitor response at the price paid p26 is R = k x occupancy(p26) / low, and the
    visitors lost per car are (low / k)(1 - R); β = -ln R / (p26 - 5), and the shares are
    S / lost and W / lost (rescaled to sum to 1 when the shuttle and the walk-ups exceed the
    lost visitors).  R, S and W are taken relative to the point estimates, so that at the
    point values the result is ``behaviour`` itself.
    """
    p26, ref = behaviour["price_paid"], behaviour["reference_price"]
    k0, r0 = behaviour["car_response"], behaviour["visitor_response"]
    low0, high0 = behaviour["occupancy_low"], behaviour["occupancy_high"]
    r = (
        r0
        * (k / k0)
        * (
            occupancy(p26, {**behaviour, "occupancy_low": low, "occupancy_high": high})
            / low
        )
        / (occupancy(p26, behaviour) / low0)
    )
    r = min(r, 0.999)
    lost0 = (low0 / k0) * (1.0 - r0)
    lost = (low / k) * (1.0 - r)
    shuttle = behaviour["share_shuttle"] * lost0 / lost
    walk = behaviour["share_walk_up"] * lost0 / lost
    if shuttle + walk > 1.0:
        shuttle, walk = shuttle / (shuttle + walk), walk / (shuttle + walk)
    return {
        **behaviour,
        "occupancy_low": low,
        "occupancy_high": high,
        "beta_per_euro": -float(np.log(r)) / (p26 - ref),
        "share_shuttle": shuttle,
        "share_walk_up": walk,
    }


def price_response(price: float, behaviour: dict[str, float]) -> float:
    """R(p): share of the car demand still arriving by private vehicle (A23; A27: 1 below the
    reference price)."""
    excess = max(price - behaviour["reference_price"], 0.0)
    return float(np.exp(-behaviour["beta_per_euro"] * excess))


def _gamma(mean: float, var: float):
    """Gamma distribution with the given mean and variance (mean floored at 1)."""
    mean = max(mean, 1.0)
    var = max(var, 1e-6)
    return stats.gamma(a=mean**2 / var, scale=var / mean)


def _demand_scale(kwargs, behaviour: dict[str, float]) -> float:
    """D x L / L0: the demand level and the persons per vehicle at 5 euro of a draw (A39); the
    presence statistics count L0 persons per vehicle."""
    low = float(kwargs.get(OCCUPANCY_LOW_NAME, behaviour["occupancy_low"]))
    return float(kwargs.get(DEMAND_LEVEL_NAME, 1.0)) * low / behaviour["occupancy_low"]


def make_demand_factories(stats_table: pd.DataFrame, behaviour: dict[str, float]):
    """Return ``(car_demand_stats, other_demand_stats)``: the visitors of a day in 2025
    conditions (5 euro, no shuttle), by private vehicle and by other means, as Gamma
    distributions of the (season, weather) cell scaled by D x L / L0 (A39).  The days of the
    backend are drawn from them (:func:`daily_accounting`)."""

    def _demand(kwargs, mean_col: str, sd_col: str):
        cell = stats_table.loc[(kwargs["season"], kwargs["weather"])]
        scale = _demand_scale(kwargs, behaviour)
        return _gamma(float(cell[mean_col]) * scale, (float(cell[sd_col]) * scale) ** 2)

    def car_demand_stats(**kwargs):
        return _demand(kwargs, "mean_car_visitors", "std_car_visitors")

    def other_demand_stats(**kwargs):
        return _demand(kwargs, "mean_other_visitors", "std_other_visitors")

    return car_demand_stats, other_demand_stats


def make_presence_factories(stats_table: pd.DataFrame, behaviour: dict[str, float]):
    """Return ``(car_arrivals_stats, other_arrivals_stats)`` closures.

    Both take the parent values as keyword arguments (``season``, ``weather``, the price
    level, the shuttle level and the uncertain quantities of A39, keyed by the index names;
    a missing uncertain quantity takes its point value) and return a **Gamma**
    distribution with mean μ and variance σ²::

        k = μ² / σ²   (shape)
        θ = σ² / μ    (scale)

    The demand (car and non-car) is scaled by D x L / L0: the visitors of the presence
    statistics are cars x L0.  Car arrivals scale the car demand by R(price) (mean and SD).  Non-car arrivals add the
    walkers and the shuttle riders to the non-car demand; the riders are capped by the
    shuttle capacity class on the cell mean (A33), and the shifted part keeps the car
    demand's coefficient of variation; non-car demand and shifted part are taken as
    independent (A32), so the variances add.  The mean is floored at 1.0 before computing
    the Gamma parameters.
    """

    def _cell(kwargs):
        cell = stats_table.loc[(kwargs["season"], kwargs["weather"])]
        low = float(kwargs.get(OCCUPANCY_LOW_NAME, behaviour["occupancy_low"]))
        b = behaviour_for(
            behaviour,
            k=float(kwargs.get(VEHICLE_RESPONSE_NAME, behaviour["car_response"])),
            low=low,
            high=float(kwargs.get(OCCUPANCY_HIGH_NAME, behaviour["occupancy_high"])),
        )
        scale = _demand_scale(kwargs, behaviour)
        demand = {
            col: float(cell[col]) * scale
            for col in (
                "mean_car_visitors",
                "std_car_visitors",
                "mean_other_visitors",
                "std_other_visitors",
            )
        }
        price = float(kwargs[PRICE_INDEX_NAME])
        capacity = float(
            kwargs.get(SHUTTLE_INDEX_NAME, 0)
        )  # not a parent of the car arrivals
        return demand, b, price_response(price, b), capacity

    def car_arrivals_stats(**kwargs):
        cell, _, r, _ = _cell(kwargs)
        return _gamma(
            cell["mean_car_visitors"] * r, (cell["std_car_visitors"] * r) ** 2
        )

    def other_arrivals_stats(**kwargs):
        cell, b, r, capacity = _cell(kwargs)
        deterred = cell["mean_car_visitors"] * (1.0 - r)
        shuttle_demand = deterred * b["share_shuttle"]
        riders = min(
            shuttle_demand, capacity
        )  # A33; A28: no shuttle -> capacity 0; A34: excess elsewhere
        shuttle_part = (
            riders / cell["mean_car_visitors"] if cell["mean_car_visitors"] > 0 else 0.0
        )
        shifted = (1.0 - r) * b["share_walk_up"] + shuttle_part
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
    other_arrivals_stats, demand_stats)``, ``demand_stats`` = ``(car_demand_stats,
    other_demand_stats)`` (:func:`make_demand_factories`).

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
    demand_stats = make_demand_factories(table, behaviour)
    return (
        season_weights,
        weather_weights,
        behaviour,
        car_stats,
        other_stats,
        demand_stats,
    )


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
    # 2026 regime observations, for the behaviour of each draw (A39, behaviour_for)
    "car_response": "",
    "price_paid": "",
    "visitor_response": "",
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
    out = {p: float(values[(p, d)]) for p, d in BEHAVIOUR_PARAMETERS.items()}
    # 90 % interval of the vehicle response (A39)
    if {"low", "high"} <= set(df.columns):
        bounds = df.set_index(["parameter", "days"]).loc[("car_response", "")]
        out["car_response_low"] = float(bounds["low"])
        out["car_response_high"] = float(bounds["high"])
    else:
        out["car_response_low"] = out["car_response_high"] = out["car_response"]
    return out


# ---------------------------------------------------------------------------
# Policy KPIs: arrivals out of the potential visitors, vehicles out of the 2025 rules
# ---------------------------------------------------------------------------

PEAK_QUANTILE = 0.95  # peak days: the busiest 5 % of the days (A36)


def occupancy(price: float, behaviour: dict[str, float]) -> float:
    """Persons per vehicle at *price* (A2): linear between the two occupancy points, clipped
    to them (as ``AccessModel``)."""
    low_p, low = behaviour["occupancy_low_price"], behaviour["occupancy_low"]
    high_p, high = behaviour["occupancy_high_price"], behaviour["occupancy_high"]
    linear = low + (price - low_p) * (high - low) / (high_p - low_p)
    return float(min(max(linear, low), high))


def daily_accounting(
    car_demand: np.ndarray,
    other_demand: np.ndarray,
    *,
    response: np.ndarray,
    share_shuttle: np.ndarray,
    share_walk_up: np.ndarray,
    occupancy_price: np.ndarray,
    occupancy_reference: np.ndarray,
    seats: np.ndarray,
    car_limit: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """What happens on each simulated day (A36); every argument is an array over the days.

    *car_demand*, *other_demand*: the visitors of the day in 2025 conditions by private
    vehicle and by other means; *response*: R(price), the share of the car demand that still
    comes by car; *occupancy_price*, *occupancy_reference*: persons per vehicle at the price
    and at 5 euro; *seats*: usable shuttle seats of the day; *car_limit*: vehicles the
    parking can take in the day (A40), ``None`` = no limit.

        deterred  = car demand x (1 - R)
        riders    = min(deterred x shuttle share, seats)
        walk-up   = deterred x walk-up share
        lost      = deterred - riders - walk-up
        vehicles  = car demand x R / occupancy(price)

    With *car_limit*, the vehicles above it are turned away: their persons take the
    remaining shuttle seats, walk up or are lost with the same shares (A40).
    """
    deterred = car_demand * (1.0 - response)
    riders = np.minimum(deterred * share_shuttle, seats)
    walk = deterred * share_walk_up
    lost = deterred - riders - walk
    car_arrivals = car_demand * response
    vehicles = car_arrivals / occupancy_price
    if car_limit is not None:
        turned_vehicles = np.maximum(vehicles - car_limit, 0.0)
        turned = turned_vehicles * occupancy_price
        more_riders = np.minimum(
            turned * share_shuttle, np.maximum(seats - riders, 0.0)
        )
        more_walk = turned * share_walk_up
        lost = lost + turned - more_riders - more_walk
        riders, walk = riders + more_riders, walk + more_walk
        vehicles = vehicles - turned_vehicles
        car_arrivals = car_arrivals - turned
    other_arrivals = other_demand + riders + walk
    return {
        "potential": car_demand + other_demand,
        "car_arrivals": car_arrivals,
        "other_arrivals": other_arrivals,
        "arrivals": car_arrivals + other_arrivals,
        "shuttle_riders": riders,
        "walk_up": walk,
        "lost": lost,
        "vehicles": vehicles,
        "vehicles_2025": car_demand / occupancy_reference,
    }


def policy_kpis(days: dict[str, np.ndarray]) -> dict[str, dict[str, dict[str, float]]]:
    """Arrivals and vehicles over the simulated *days* (:func:`daily_accounting`), over the
    season and on the peak days (the days at or above the ``PEAK_QUANTILE`` of the potential
    visitors; A36).

    Returns ``{"arrivals" | "vehicles": {"mean" | "peak": {"value", "share"}}}``: ``value``
    is the mean per day; ``share`` the ratio to the potential visitors (arrivals) or to the
    vehicles under the 2025 rules (vehicles) over the same days, so that 1 = nobody lost /
    as many vehicles as in 2025.
    """
    potential = days["potential"]
    every = np.ones(potential.size, dtype=bool)
    peak = potential >= np.quantile(potential, PEAK_QUANTILE)

    def kpi(sel: np.ndarray, value: str, reference: str) -> dict[str, float]:
        return {
            "value": float(days[value][sel].mean()),
            "share": float(days[value][sel].sum() / days[reference][sel].sum()),
        }

    return {
        "arrivals": {
            "mean": kpi(every, "arrivals", "potential"),
            "peak": kpi(peak, "arrivals", "potential"),
        },
        "vehicles": {
            "mean": kpi(every, "vehicles", "vehicles_2025"),
            "peak": kpi(peak, "vehicles", "vehicles_2025"),
        },
    }
