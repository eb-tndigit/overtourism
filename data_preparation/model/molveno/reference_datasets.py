# SPDX-License-Identifier: Apache-2.0
"""
STEP 2 - Molveno reference datasets.

Input : <normalized>/molveno/ tables (step 1)
Output: <final>/molveno/
  - molveno_daily: one row per day, presences, calendar, measured weather and weather class
  - molveno_presence_stats: main statistics of the model, per summer calendar day x weekday x
    weather class (gradient boosting, D9)
  - molveno_presence_stats_monthly: comparison statistics, per month x weekday x weather class
    (least squares, D9)
  - molveno_weather_freq: per summer month, the frequency of each weather class (A5)
  - molveno_counts_daily: one row per day with a count (summers 2022-2023): the presences and weather
    of molveno_daily with the cars entering Molveno and the beach, the beach counters, the paying
    vehicles of the car parks and the ISPAT overnight stays (calibration of the subsystems)

Every choice is an entry of ASSUMPTIONS.md (D*, A*); the estimators are in presence_estimators.py.
"""

import logging

import pandas as pd

from model.molveno.presence_estimators import (
    PRESENCES,
    WEATHER_CLASSES,
    add_calendar,
    add_weather,
    additive_presence_stats,
    gbm_presence_stats,
    training_days,
    weather_climatology,
    weather_frequencies,
)

logger = logging.getLogger(__name__)


def build_molveno_daily(vodafone: pd.DataFrame, meteo: pd.DataFrame) -> pd.DataFrame:
    """D8: tourists and excursionists of the same date, with calendar and weather."""
    daily = vodafone.merge(meteo[["date", "tmax", "rain_mm"]], on="date", how="left")
    daily = add_weather(add_calendar(daily), weather_climatology(meteo))
    return daily.sort_values("date").reset_index(drop=True)


BEACH_GATES = 9
FULL_DAY_HOURS = 24


def build_molveno_counts_daily(daily: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The counts per day, NaN where a day is incomplete: a road direction with fewer than 24
    counted hours, a beach total without all the gates."""
    road = tables["road_counts_daily"]
    road = road[(road["flow"] != "") & (road["hours"] >= FULL_DAY_HOURS)]
    counts = road.pivot_table(index="date", columns="flow", values="cars", aggfunc="sum")
    counts.columns = [f"cars_{c}" for c in counts.columns]
    counts["cars_into_molveno"] = counts["cars_in_from_andalo"] + counts["cars_in_from_san_lorenzo"]

    beach = tables["beach_counts"].groupby("date")
    for column in ("presences", "passages"):
        complete = beach[column].count() == BEACH_GATES
        counts = counts.join(beach[column].sum().where(complete).rename(f"beach_{column}"), how="outer")

    paid = tables["parking_paid"].assign(vehicle_hours=lambda t: t["vehicles"] * t["hours"])
    paid = paid.groupby("date")[["vehicles", "vehicle_hours"]].sum().add_prefix("paid_")
    counts = counts.join(paid, how="outer")
    first, last = counts.index.min(), counts.index.max()
    ispat = tables["ispat_daily"].set_index("date").add_prefix("ispat_")
    counts = counts.join(ispat, how="left")

    out = daily.set_index("date").reindex(pd.date_range(first, last, name="date")).join(counts)
    return out.reset_index()


def _check(daily: pd.DataFrame, stats: pd.DataFrame, monthly: pd.DataFrame, freq: pd.DataFrame) -> None:
    for name, table in (("molveno_presence_stats", stats), ("molveno_presence_stats_monthly", monthly)):
        cols = [f"{s}_{y}" for y in PRESENCES for s in ("mean", "std")]
        if table[cols].isna().any().any() or (table[[f"std_{y}" for y in PRESENCES]] <= 0).any().any():
            raise ValueError(f"{name}: missing or non-positive values")
    if set(stats["weather"]) != set(WEATHER_CLASSES):
        raise ValueError("molveno_presence_stats: a weather class has no observed day")
    if len(stats) != stats["month_day"].nunique() * 7 * len(WEATHER_CLASSES):
        raise ValueError("molveno_presence_stats: a weather class has no observed day in some month")
    sums = freq.groupby("month")["freq_rel"].sum()
    if not ((sums - 1).abs() < 1e-9).all():
        raise ValueError("molveno_weather_freq: frequencies do not sum to 1")
    logger.info("Training days: %d; cells: %d (daily), %d (monthly)",
                len(training_days(daily)), len(stats), len(monthly))


def build_all(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    daily = build_molveno_daily(tables["vodafone_presences"], tables["meteo_station"])
    train = training_days(daily)
    logger.info("Gradient boosting statistics...")
    stats = gbm_presence_stats(train)
    monthly = additive_presence_stats(train)
    freq = weather_frequencies(daily)
    _check(daily, stats, monthly, freq)
    counts = build_molveno_counts_daily(daily, tables)
    logger.info("Count days: %d cars into Molveno, %d beach presences, %d paid parking, %d ISPAT",
                *(counts[c].notna().sum() for c in ("cars_into_molveno", "beach_presences", "paid_vehicles",
                                                    "ispat_overnight_stays")))
    return {
        "molveno_daily": daily,
        "molveno_presence_stats": stats,
        "molveno_presence_stats_monthly": monthly,
        "molveno_weather_freq": freq,
        "molveno_counts_daily": counts,
    }
