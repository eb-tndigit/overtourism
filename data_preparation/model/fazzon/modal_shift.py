# SPDX-License-Identifier: Apache-2.0
"""
STEP 2b - Modal-shift estimation (D8).

Input : fazzon_daily; the season x weather levels of the calibration year (A21, comparison only)
Output: <final>/fazzon/fazzon_modal_shift (one row per parameter, with a bootstrap 90 % interval)
        and fazzon_modal_shift_daily (the 2026 days used)

Behaviour model (every element is an entry of ASSUMPTIONS.md):

* Vehicles at the 2025 price (5 euro) on a 2026 day: the cars of the calibration year's relation
  between Fazzon cars, cars up valley at Pellizzano in the morning and the weather class (A37),
  applied to the day's traffic and weather, times a 2026 year effect of 1 +- 10 % (A22). The
  year effect measured on the 2026 days at 5 euro without the access rules is reported as a
  check (``year_effect_observed``), not used. Counts at 5 euro include the motorbikes (A26).
  The season x weather level of the calibration year (A21) is kept in the days for comparison.
* Persons per vehicle: a function of the price paid (A2): 2.5 at 5 euro, 3.5 at 20 euro, linear in
  between. Visitors by private vehicle = vehicles x occupancy; under the 2026 rules the motorbikes
  are counted apart (``moto_persons``) and added.
* Response of the visitors by private vehicle to the price (A23):
  visitors(p) = visitors(5) * exp(-beta * (p - 5)), p = average price paid per vehicle (A24).
  The whole 2026 package (price, booking, traffic light, shuttle) is attributed to the price (A25).
* Visitors who no longer come by private vehicle = visitors(5) - visitors(p), split among shuttle
  (bus up, measured), walk-up (bus down - bus up, A6, from 01/08/2026) and elsewhere (the rest).

Uncertainty: bootstrap over days (the calibration days, refitting the relation; the regime days;
the days of the observed year effect) and a year effect drawn uniformly in 1 +- 10 % (A22).
"""

import logging

import numpy as np
import pandas as pd

from model.fazzon.demand import DEMAND_PREDICTOR, calibration_days, expected_cars, fit_demand
from model.fazzon.reference_datasets import (
    OCCUPANCY_HIGH,
    OCCUPANCY_LOW,
    PRICE_2025,
    REFERENCE_OCCUPANCY,
    SEASONS,
    WEATHER_CLASSES,
    occupancy,
)

logger = logging.getLogger(__name__)

N_BOOTSTRAP = 2000
INTERVAL = (0.05, 0.95)
SEED = 0

YEAR_EFFECT = 1.0  # A22
YEAR_EFFECT_UNCERTAINTY = 0.10  # A22: drawn uniformly in YEAR_EFFECT +- this, in the bootstrap


# ------------------------------------------------------------------ days
def modal_shift_days(daily: pd.DataFrame, levels: pd.DataFrame, coef: np.ndarray) -> pd.DataFrame:
    """2026 days with car counts and traffic, the cars expected in 2025 conditions and the price paid.

    ``role``: ``year_effect`` (5 euro, no access rules, no event; A22) or ``regime`` (access
    rules and shuttle running; D2, D3 excluded). ``cars_2025_levels``: the season x weather level
    (A21), for comparison.
    """
    levels = levels.set_index(["season", "weather"])["mean_car_visitors"] / REFERENCE_OCCUPANCY
    f = daily[(daily["year"] == 2026) & daily["cars"].notna() & daily["month"].isin(list(SEASONS))].copy()
    f = f.dropna(subset=["weather"])
    f["season"] = f["month"].map(SEASONS)
    f["cars_2025_levels"] = [levels[(s, w)] for s, w in zip(f["season"], f["weather"])]
    half = f["cars_half"].fillna(0)
    f["price_paid"] = (f["cars_full"] * f["parking_price"] + half * f["parking_price_afternoon"]) / f["cars"]  # A24
    f["walk_up"] = (f["bus_down"] - f["bus_up"]).clip(lower=0)  # A6
    f["private_visitors"] = (  # A2, A26
        f["cars_full"] * occupancy(f["parking_price"]) + half * occupancy(f["parking_price_afternoon"])
        + f["moto_persons"].fillna(0)
    )
    f["role"] = None
    f.loc[~f["regulated_parking"] & ~f["event_day"] & (f["parking_price"] == PRICE_2025), "role"] = "year_effect"
    f.loc[f["regulated_parking"] & f["shuttle"].notna() & ~f["event_day"], "role"] = "regime"
    f = f.dropna(subset=["role"])
    no_traffic = f[DEMAND_PREDICTOR].isna()
    if no_traffic.any():
        logger.warning("Modal shift: %d days without valid traffic at Pellizzano, not used", no_traffic.sum())
    f = f[~no_traffic].copy()
    f["morning_up"] = f[DEMAND_PREDICTOR]
    f["cars_2025_conditions"] = expected_cars(f, coef)
    cols = ["date", "season", "weather", "role", "cars", "morning_up", "cars_2025_conditions", "cars_2025_levels",
            "price_paid", "private_visitors", "bus_up", "bus_down", "walk_up", "moto_persons", "parking_price_assumed"]
    return f[cols].reset_index(drop=True)


# ------------------------------------------------------------------ estimation
def estimate_point(year: pd.DataFrame, regime: pd.DataFrame, year_effect: float) -> dict[str, float]:
    """Point estimates on one (re)sample of days, for a given year effect."""
    visitors5 = regime["cars_2025_conditions"] * year_effect * occupancy(PRICE_2025)
    price = (regime["price_paid"] * regime["cars"]).sum() / regime["cars"].sum()
    response = regime["private_visitors"].sum() / visitors5.sum()
    out = {
        "year_effect": year_effect,
        "year_effect_observed": year["cars"].sum() / year["cars_2025_conditions"].sum(),  # check only (A22)
        "car_response": regime["cars"].sum() / (regime["cars_2025_conditions"] * year_effect).sum(),
        "price_paid": price,
        "visitor_response": response,
        "beta_per_euro": -np.log(response) / (price - PRICE_2025),
    }
    for sub, suffix in ((regime, "all_days"), (regime[regime["bus_down"].notna()], "with_walk_up")):
        v5 = sub["cars_2025_conditions"] * year_effect * occupancy(PRICE_2025)
        lost = (v5 - sub["private_visitors"]).sum()
        shuttle = sub["bus_up"].sum() / lost
        out[f"{suffix}__lost_per_day"] = lost / len(sub)
        out[f"{suffix}__share_shuttle"] = shuttle
        if suffix == "with_walk_up":
            walk = sub["walk_up"].sum() / lost
            out[f"{suffix}__share_walk_up"] = walk
            out[f"{suffix}__share_elsewhere"] = 1.0 - shuttle - walk
    return out


def _with_expected(days: pd.DataFrame, coef: np.ndarray) -> pd.DataFrame:
    return days.assign(cars_2025_conditions=expected_cars(days, coef))


def estimate_modal_shift(daily: pd.DataFrame, levels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(estimates, days)``: one row per parameter with the point estimate and the
    bootstrap interval, and the days used. `levels`: the season x weather levels of the
    calibration year (``level_presence_stats``, A21), kept in the days for comparison."""
    calibration = calibration_days(daily)
    coef = fit_demand(calibration)
    days = modal_shift_days(daily, levels, coef)
    design_days = days.rename(columns={"morning_up": DEMAND_PREDICTOR})
    year = design_days[design_days["role"] == "year_effect"]
    regime = design_days[design_days["role"] == "regime"]
    point = estimate_point(year, regime, YEAR_EFFECT)

    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_BOOTSTRAP):
        c = fit_demand(calibration.iloc[rng.integers(0, len(calibration), len(calibration))])
        y = _with_expected(year.iloc[rng.integers(0, len(year), len(year))], c)
        r = _with_expected(regime.iloc[rng.integers(0, len(regime), len(regime))], c)
        effect = rng.uniform(YEAR_EFFECT - YEAR_EFFECT_UNCERTAINTY, YEAR_EFFECT + YEAR_EFFECT_UNCERTAINTY)
        boot.append(estimate_point(y, r, effect))
    boot = pd.DataFrame(boot)

    rows = []
    for key, value in point.items():
        sample, parameter = key.split("__") if "__" in key else ("", key)
        lo, hi = boot[key].quantile(list(INTERVAL))
        rows.append({"parameter": parameter, "days": sample, "estimate": value, "low": lo, "high": hi})
    # Coefficients of the demand relation (A37, documentation) and constants of the behaviour model,
    # so that the Fazzon model reads them from here (A2)
    for parameter, value in (
        ("demand_intercept", coef[0]), ("demand_per_morning_car", coef[1]),
        *((f"demand_{w}", c) for w, c in zip(WEATHER_CLASSES[1:], coef[2:])),
        ("reference_price", PRICE_2025),
        ("occupancy_low_price", OCCUPANCY_LOW[0]), ("occupancy_low", OCCUPANCY_LOW[1]),
        ("occupancy_high_price", OCCUPANCY_HIGH[0]), ("occupancy_high", OCCUPANCY_HIGH[1]),
    ):
        rows.append({"parameter": parameter, "days": "", "estimate": value, "low": value, "high": value})
    estimates = pd.DataFrame(rows)
    estimates["n_calibration_days"] = len(calibration)
    estimates["n_year_effect_days"] = len(year)
    estimates["n_regime_days"] = len(regime)
    logger.info("Modal shift: visitor response %.2f (%.2f-%.2f), beta %.3f per euro; observed year effect %.2f",
                point["visitor_response"], *boot["visitor_response"].quantile(list(INTERVAL)), point["beta_per_euro"],
                point["year_effect_observed"])
    return estimates, days
