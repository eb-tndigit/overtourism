# SPDX-License-Identifier: Apache-2.0
"""Test data for the Fazzon and Molveno models.

The models have no built-in statistics: they read them from the folders in
``FAZZON_DATA_DIR`` and ``MOLVENO_DATA_DIR``.  The variables are set here, before any
test module imports the models, to temporary folders holding the statistics in the
pipelines' layout (Fazzon: 2025 calibration values; Molveno: synthetic values with the
summer peak, the weekend and the weather effects).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_SEASONS = ["june", "july", "august", "september"]
_WEATHERS = ["good", "cool", "rainy"]
_SEASON_FREQ = {
    "june": 3 / 82,
    "july": 31 / 82,
    "august": 31 / 82,
    "september": 17 / 82,
}
_WEATHER_FREQ = {"good": 0.62, "cool": 0.2, "rainy": 0.18}
_FACTOR = {"good": 1.0, "cool": 0.73, "rainy": 0.6}
_LEVEL = {
    "june": 678.1,
    "july": 611.7,
    "august": 897.3,
    "september": 258.3,
}  # good-day car visitors
_CV = {"june": 0.36, "july": 0.27, "august": 0.3, "september": 0.59}

PRESENCE_STATS = pd.DataFrame(
    [
        {
            "season": s,
            "weather": w,
            "season_freq_rel": _SEASON_FREQ[s],
            "weather_freq_rel": _WEATHER_FREQ[w],
            "mean_car_visitors": _LEVEL[s] * _FACTOR[w],
            "std_car_visitors": _LEVEL[s] * _FACTOR[w] * _CV[s],
            "mean_other_visitors": _LEVEL[s] * _FACTOR[w] * 0.449,
            "std_other_visitors": _LEVEL[s] * _FACTOR[w] * 0.449 * 0.5,
        }
        for s in _SEASONS
        for w in _WEATHERS
    ]
)


MODAL_SHIFT = pd.DataFrame(
    [
        ("reference_price", "", 5.0),
        ("beta_per_euro", "", 0.09),
        ("occupancy_low_price", "", 5.0),
        ("occupancy_low", "", 2.5),
        ("occupancy_high_price", "", 20.0),
        ("occupancy_high", "", 3.5),
        ("share_shuttle", "with_walk_up", 0.77),
        ("share_walk_up", "with_walk_up", 0.155),
        ("share_elsewhere", "with_walk_up", 0.075),
        # 2026 regime observations, consistent with beta: R(15) = exp(-0.09 x 10),
        # vehicle response = R x 2.5 / persons per vehicle at 15 euro (3.1667)
        ("price_paid", "", 15.0),
        ("visitor_response", "", float(np.exp(-0.9))),
        ("car_response", "", float(np.exp(-0.9) * 2.5 / (2.5 + 10.0 / 15.0))),
    ],
    columns=["parameter", "days", "estimate"],
)


def write_fazzon_stats(
    folder: Path, stats=PRESENCE_STATS, modal_shift=MODAL_SHIFT
) -> Path:
    """Write the Fazzon data files (pipeline layout) into *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    stats.assign(calibration="parking_2025").to_parquet(
        folder / "fazzon_presence_stats.parquet"
    )
    modal_shift.assign(
        low=modal_shift["estimate"], high=modal_shift["estimate"]
    ).to_parquet(folder / "fazzon_modal_shift.parquet")
    return folder


FAZZON_TEST_DATA_DIR = write_fazzon_stats(
    Path(tempfile.mkdtemp(prefix="fazzon_test_data_"))
)
os.environ["FAZZON_DATA_DIR"] = str(FAZZON_TEST_DATA_DIR)
os.environ["DT_OVERTURISM_STANDALONE_MODE"] = "true"


# ---------------------------------------------------------------- Molveno
_MOLVENO_WEEKDAYS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]
_MOLVENO_WEATHER_SHIFT = {
    "good": 300.0,
    "cool": -400.0,
    "rainy": -800.0,
}  # excursionists
_MOLVENO_WEATHER_FREQ = {"good": 0.61, "cool": 0.18, "rainy": 0.21}


def _molveno_level(day: int) -> float:
    """Summer profile: low in June and September, peak around mid-August (day 75)."""
    return 1500.0 + 3000.0 * np.exp(-(((day - 75) / 30.0) ** 2))


def molveno_tables() -> dict[str, pd.DataFrame]:
    """The Molveno data files (pipeline layout): daily and monthly statistics, weather frequencies."""
    dates = pd.date_range("2025-06-01", "2025-09-30")
    rows = []
    for day, date in enumerate(dates):
        for wd in _MOLVENO_WEEKDAYS:
            for weather, shift in _MOLVENO_WEATHER_SHIFT.items():
                level = _molveno_level(day)
                weekend = 600.0 if wd in ("saturday", "sunday") else 0.0
                rows.append(
                    {
                        "month_day": date.strftime("%m-%d"),
                        "summer_day": day,
                        "month": date.month,
                        "weekday": wd,
                        "weather": weather,
                        "mean_tourists": level + weekend / 2,
                        "std_tourists": 400.0,
                        "mean_excursionists": level + weekend + shift,
                        "std_excursionists": 800.0,
                        "weather_days": 30,
                    }
                )
    daily = pd.DataFrame(rows)
    stat_cols = [
        "mean_tourists",
        "std_tourists",
        "mean_excursionists",
        "std_excursionists",
    ]
    monthly = daily.groupby(["month", "weekday", "weather"], as_index=False)[
        stat_cols
    ].mean()
    freq = pd.DataFrame(
        [
            {"month": m, "weather": w, "freq_rel": f}
            for m in (6, 7, 8, 9)
            for w, f in _MOLVENO_WEATHER_FREQ.items()
        ]
    )
    return {
        "molveno_presence_stats": daily,
        "molveno_presence_stats_monthly": monthly,
        "molveno_weather_freq": freq,
    }


def write_molveno_stats(
    folder: Path, tables: dict[str, pd.DataFrame] | None = None
) -> Path:
    """Write the Molveno data files (pipeline layout) into *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in (tables or molveno_tables()).items():
        df.to_parquet(folder / f"{name}.parquet")
    return folder


MOLVENO_TEST_DATA_DIR = write_molveno_stats(
    Path(tempfile.mkdtemp(prefix="molveno_test_data_"))
)
os.environ["MOLVENO_DATA_DIR"] = str(MOLVENO_TEST_DATA_DIR)
