"""Presence statistics of the Molveno model, read from the data-preparation pipeline.

The statistics are the reference datasets of ``data_preparation/model/molveno`` (decision
and assumption IDs refer to its ``ASSUMPTIONS.md``):

* ``molveno_presence_stats``: per summer calendar day x weekday x weather class, the mean
  and SD of tourists and excursionists estimated by gradient boosting (D9-D12);
* ``molveno_presence_stats_monthly``: the same per month x weekday x weather class,
  estimated by least squares (comparison, D9);
* ``molveno_weather_freq``: per summer month, the frequency of each weather class (A5).

In the model the day of the summer is a uniform distribution over the period of the
scenario (D9); the presence estimator is a category, fixed by the backend.  There are no
built-in values: missing files or cells raise an error.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy import stats

from .molveno_data import (
    MONTHLY_STATS_FILE,
    PRESENCE_STATS_FILE,
    WEATHER_FREQ_FILE,
    molveno_data_dir,
)

logger = logging.getLogger(__name__)

WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
# The day types of the model: the statistics are per weekday, Monday-Friday merged (A14)
DAY_TYPES = {
    "weekday": ("monday", "tuesday", "wednesday", "thursday", "friday"),
    "saturday": ("saturday",),
    "sunday": ("sunday",),
}
DAY_TYPE_WEIGHTS = {t: len(days) / len(WEEKDAYS) for t, days in DAY_TYPES.items()}
DAY_TYPE_LABELS = {"weekday": "lunedì-venerdì", "saturday": "sabato", "sunday": "domenica"}
WEATHERS = ("good", "cool", "rainy")
PRESENCES = ("tourists", "excursionists")
STAT_COLUMNS = tuple(f"{s}_{p}" for p in PRESENCES for s in ("mean", "std"))

DAY_INDEX_NAME = "day of summer"
ESTIMATOR_INDEX_NAME = "presence estimator"
ESTIMATOR_ML = "gradient boosting"  # main statistics (D9)
ESTIMATOR_MONTHLY = "monthly averages"  # comparison statistics (D9)
ESTIMATORS = (ESTIMATOR_ML, ESTIMATOR_MONTHLY)
TRUNCATION_SDS = (
    10.0  # upper truncation of the presence distributions, in SDs (lower: 0, D12)
)


@dataclass(frozen=True)
class MolvenoTables:
    """The statistics, as lookup tables."""

    daily: dict[
        tuple[int, str, str], tuple[float, ...]
    ]  # (summer day, day type, weather) -> STAT_COLUMNS
    monthly: dict[
        tuple[int, str, str], tuple[float, ...]
    ]  # (month, day type, weather) -> STAT_COLUMNS
    day_month: tuple[int, ...]  # summer day -> month
    day_label: tuple[str, ...]  # summer day -> "MM-DD"
    weather_freq: dict[int, dict[str, float]]  # month -> {weather: frequency}

    @property
    def n_days(self) -> int:
        return len(self.day_month)


def _read(folder: Path, name: str, columns: set[str]) -> pd.DataFrame:
    path = folder / name
    if not path.exists():
        raise FileNotFoundError(f"Molveno statistics not found: {path}")
    df = pd.read_parquet(path)
    lacking = sorted(columns - set(df.columns))
    if lacking:
        raise ValueError(f"Molveno statistics {name}: missing columns {lacking}")
    return df


def _check_cells(name: str, keys: set, expected: set) -> None:
    if keys != expected:
        missing, extra = sorted(expected - keys)[:5], sorted(keys - expected)[:5]
        raise ValueError(
            f"Molveno statistics {name}: cells missing {missing}, unexpected {extra}"
        )


def load_tables(data_dir: str | Path | None = None) -> MolvenoTables:
    """Read and check the three statistics files from *data_dir* (default: the data folder)."""
    folder = Path(data_dir) if data_dir is not None else molveno_data_dir()
    daily = _read(
        folder,
        PRESENCE_STATS_FILE,
        {"summer_day", "month_day", "month", "weekday", "weather", *STAT_COLUMNS},
    )
    monthly = _read(
        folder, MONTHLY_STATS_FILE, {"month", "weekday", "weather", *STAT_COLUMNS}
    )
    freq = _read(folder, WEATHER_FREQ_FILE, {"month", "weather", "freq_rel"})

    days = daily.drop_duplicates("summer_day").sort_values("summer_day")
    if days["summer_day"].tolist() != list(range(len(days))):
        raise ValueError("Molveno statistics: summer days are not consecutive from 0")
    months = sorted(days["month"].unique())
    _check_cells(
        PRESENCE_STATS_FILE,
        set(zip(daily["summer_day"], daily["weekday"], daily["weather"])),
        {(d, w, c) for d in range(len(days)) for w in WEEKDAYS for c in WEATHERS},
    )
    _check_cells(
        MONTHLY_STATS_FILE,
        set(zip(monthly["month"], monthly["weekday"], monthly["weather"])),
        {(m, w, c) for m in months for w in WEEKDAYS for c in WEATHERS},
    )
    _check_cells(
        WEATHER_FREQ_FILE,
        set(zip(freq["month"], freq["weather"])),
        {(m, c) for m in months for c in WEATHERS},
    )
    if (daily[[f"std_{p}" for p in PRESENCES]] <= 0).any().any() or (
        monthly[[f"std_{p}" for p in PRESENCES]] <= 0
    ).any().any():
        raise ValueError("Molveno statistics: non-positive SD")

    def table(df, key):
        return _day_types(
            {
                (int(k), w, c): tuple(float(v) for v in vals)
                for k, w, c, *vals in df[
                    [key, "weekday", "weather", *STAT_COLUMNS]
                ].itertuples(index=False)
            }
        )

    weather_freq = {
        int(m): dict(zip(g["weather"], g["freq_rel"].astype(float)))
        for m, g in freq.groupby("month")
    }
    logger.info("Molveno presence statistics loaded from %s", folder)
    return MolvenoTables(
        daily=table(daily, "summer_day"),
        monthly=table(monthly, "month"),
        day_month=tuple(int(m) for m in days["month"]),
        day_label=tuple(days["month_day"]),
        weather_freq=weather_freq,
    )


def _day_types(
    cells: dict[tuple[int, str, str], tuple[float, ...]],
) -> dict[tuple[int, str, str], tuple[float, ...]]:
    """A14: the statistics per day type, from those per weekday: the mean of the weekdays'
    means, the SD of their equal mixture (mean of the variances + variance of the means)."""
    out = {}
    for key, weather in {(k, c) for k, _, c in cells}:
        for day_type, days in DAY_TYPES.items():
            stats_ = [cells[(key, d, weather)] for d in days]
            row = []
            for j in range(0, len(STAT_COLUMNS), 2):
                means = [s[j] for s in stats_]
                mean = sum(means) / len(means)
                var = sum(s[j + 1] ** 2 + (s[j] - mean) ** 2 for s in stats_) / len(stats_)
                row += [mean, math.sqrt(var)]
            out[(key, day_type, weather)] = tuple(row)
    return out


def summer_day(value: float, n_days: int) -> int:
    """Day of the summer (0 = first day) of a sampled day value: the day it falls in."""
    return min(max(math.floor(value), 0), n_days - 1)


def _presence(mean: float, std: float):
    """D12: normal, truncated at 0."""
    return stats.truncnorm(-mean / std, TRUNCATION_SDS, loc=mean, scale=std)


def make_presence_factories(tables: MolvenoTables):
    """Return ``(tourist_presences_stats, excursionist_presences_stats)``: the distribution
    factories of the presence indexes, keyed by the parents' names (weekday, weather, day of
    summer, presence estimator)."""

    def _cell(kwargs) -> tuple[float, ...]:
        day = summer_day(kwargs[DAY_INDEX_NAME], tables.n_days)
        key = (kwargs["weekday"], kwargs["weather"])
        if kwargs[ESTIMATOR_INDEX_NAME] == ESTIMATOR_MONTHLY:
            return tables.monthly[(tables.day_month[day], *key)]
        return tables.daily[(day, *key)]

    def tourist_presences_stats(**kwargs):
        mean, std, *_ = _cell(kwargs)
        return _presence(mean, std)

    def excursionist_presences_stats(**kwargs):
        *_, mean, std = _cell(kwargs)
        return _presence(mean, std)

    return tourist_presences_stats, excursionist_presences_stats


def weather_weights(
    tables: MolvenoTables, first_day: int = 0, end_day: int | None = None
) -> dict[str, float]:
    """A5: frequency of the weather classes over the days ``first_day .. end_day - 1``: the
    monthly frequencies weighted by the days of each month in the period."""
    end_day = tables.n_days if end_day is None else end_day
    days = range(max(first_day, 0), min(max(end_day, first_day + 1), tables.n_days))
    weights = {
        c: sum(tables.weather_freq[tables.day_month[d]][c] for d in days) / len(days)
        for c in WEATHERS
    }
    total = sum(weights.values())
    return {c: w / total for c, w in weights.items()}


def presence_inputs(data_dir: str | Path | None = None):
    """Return ``(tables, weather_weights, tourist_presences_stats, excursionist_presences_stats)``.

    Reads the reference datasets; called by ``MolvenoModel.default_inputs`` each time a model
    is built, so a model built after a data download uses the new statistics.
    """
    tables = load_tables(data_dir)
    tourists, excursionists = make_presence_factories(tables)
    return tables, weather_weights(tables), tourists, excursionists
