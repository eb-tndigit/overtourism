# SPDX-License-Identifier: Apache-2.0
"""
STEP 2c - Monitoring of the calibration (D20).

Input : fazzon_daily
Output: <final>/fazzon/fazzon_monitoring, one row per indicator x year x period

The presence statistics are calibrated on one year (CALIBRATION_YEAR, D9) and the response to the
access rules on the 2026 regime (D8). Every year of road counts and Fazzon car counts is compared
with the calibration year on the same calendar days, so that a new delivery shows whether the
calibration still holds:

* ``road_<station>``: cars per day, both directions (valid days, D10), and
  ``road_pellizzano_morning_up``: cars up valley at Pellizzano 07:00-11:59 (A16, A17). Drift of
  the demand in the valley; in 2022-2025 the monthly summer means moved within about +-4 %. Road
  works, detours or sensor faults move them too: compare with the control stations.
* ``fazzon_cars_per_morning_car``: Fazzon cars / Pellizzano morning cars up valley (ratio of the
  sums over the days with both; EXCLUDED_DAYS left out: D2, D3, D21). Outside
  the regime: drift specific to Fazzon (the lake more or less popular); inside: the response to
  the access rules.
* ``fazzon_cars_vs_calibration_relation``: Fazzon cars / the cars of the calibration year's
  relation with traffic and weather (A37), same days. The check of the year effect (A22): outside
  the regime it should stay within 1 +- YEAR_EFFECT_UNCERTAINTY; inside, the response to the rules.

Periods: ``regime_window`` = the calendar window of the year's access regime (REGULATED_PARKING; a year
without regime takes the window of the latest regime, so that all years are compared on the same
days); ``rest_of_season`` = the other days of the season (SEASONS months). The reference values are
computed on the same window of the calibration year. Raw ratios, without adjustment for the
weather or weekday mix: the adjusted effects are estimated in modal_shift.py.
"""

import numpy as np
import pandas as pd

from model.fazzon.demand import DEMAND_PREDICTOR, calibration_days, expected_cars, fit_demand

from model.fazzon.reference_datasets import (
    CALIBRATION_YEAR,
    EXCLUDED_DAYS,
    REGULATED_PARKING,
    ROAD_TOTAL_STATIONS,
    SEASONS,
)

REFERENCE_YEAR = CALIBRATION_YEAR
ROAD_INDICATORS = {f"road_{s}": f"strada_{s}_totale" for s in ROAD_TOTAL_STATIONS}
ROAD_INDICATORS["road_pellizzano_morning_up"] = DEMAND_PREDICTOR
CARS_PER_MORNING_CAR = "fazzon_cars_per_morning_car"
CARS_VS_RELATION = "fazzon_cars_vs_calibration_relation"


def _periods(dates: pd.Series, year: int) -> np.ndarray:
    """``regime_window`` / ``rest_of_season`` for `dates`, on the calendar window of `year`'s regime."""
    first, last = REGULATED_PARKING.get(year, REGULATED_PARKING[max(REGULATED_PARKING)])
    month_day = dates.dt.month * 100 + dates.dt.day
    inside = month_day.between(first.month * 100 + first.day, last.month * 100 + last.day)
    return np.where(inside, "regime_window", "rest_of_season")


def _indicators(days: pd.DataFrame, coef: np.ndarray) -> dict[str, tuple[int, float]]:
    """(days, value) of every indicator on `days`; `coef`: the demand relation (A37)."""
    out = {}
    for name, col in ROAD_INDICATORS.items():
        values = days[col].dropna()
        if len(values):
            out[name] = (len(values), values.mean())
    both = days[days["cars"].notna() & days[DEMAND_PREDICTOR].notna()
                & ~days["date"].isin(EXCLUDED_DAYS)]
    if len(both):
        out[CARS_PER_MORNING_CAR] = (len(both), both["cars"].sum() / both[DEMAND_PREDICTOR].sum())
    both = both[both["weather"].notna()]
    if len(both):
        out[CARS_VS_RELATION] = (len(both), both["cars"].sum() / expected_cars(both, coef).sum())
    return out


def build_monitoring(daily: pd.DataFrame) -> pd.DataFrame:
    season = daily[daily["month"].isin(list(SEASONS))].copy()
    reference = season[season["year"] == REFERENCE_YEAR]
    coef = fit_demand(calibration_days(daily))
    rows = []
    for year, days in season.groupby("year"):
        days = days.assign(period=_periods(days["date"], year))
        ref = reference.assign(period=_periods(reference["date"], year))
        for period in ("regime_window", "rest_of_season"):
            ref_values = _indicators(ref[ref["period"] == period], coef)
            for name, (n, value) in _indicators(days[days["period"] == period], coef).items():
                ref_n, ref_value = ref_values.get(name, (0, np.nan))
                rows.append({
                    "indicator": name, "year": year, "period": period, "regime_in_force": year in REGULATED_PARKING,
                    "days": n, "value": value, "reference_year": REFERENCE_YEAR, "reference_days": ref_n,
                    "reference_value": ref_value, "ratio_to_reference": value / ref_value,
                })
    return pd.DataFrame(rows)
