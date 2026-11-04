# SPDX-License-Identifier: Apache-2.0
"""
Demand at Fazzon in 2025 conditions (5 euro, no access rules) from the valley traffic (A37, A38).

* A37: cars at Fazzon = a + b x cars up valley at Pellizzano in the morning + c(weather class),
  least squares on the calibration-year days (D4, D9).
* A38: the days scatter around the relation with variance proportional to the expected cars
  (variance = DISPERSION x mean, as for counts).

Used by the presence statistics (every season day with valid traffic and weather, all years:
reference_datasets.build_presence_stats), the modal shift (modal_shift.py) and the monitoring
(monitoring.py).
"""

import numpy as np
import pandas as pd

from model.fazzon.reference_datasets import CALIBRATION_YEAR, SEASONS, WEATHER_CLASSES

DEMAND_PREDICTOR = "strada_pellizzano_salita_mattina"  # A37: cars up valley at Pellizzano 07-12 (A16, A17)


def calibration_days(daily: pd.DataFrame) -> pd.DataFrame:
    """Calibration-year days with Fazzon cars, valid traffic and weather class (D4: all of them)."""
    return daily[(daily["year"] == CALIBRATION_YEAR) & daily["cars"].notna() & daily[DEMAND_PREDICTOR].notna()
                 & daily["weather"].notna()]


def demand_days(daily: pd.DataFrame) -> pd.DataFrame:
    """Season days of every year with valid traffic and weather class: the days the presence
    statistics are built on (A37)."""
    return daily[daily["month"].isin(list(SEASONS)) & daily[DEMAND_PREDICTOR].notna() & daily["weather"].notna()]


def _design(days: pd.DataFrame) -> np.ndarray:
    weather = [(days["weather"] == w).to_numpy(dtype=float) for w in WEATHER_CLASSES[1:]]
    return np.column_stack([np.ones(len(days)), days[DEMAND_PREDICTOR].to_numpy(dtype=float), *weather])


def fit_demand(calibration: pd.DataFrame) -> np.ndarray:
    """A37: cars = a + b x cars up valley + c(weather), least squares; returns (a, b, c_cool, c_rainy)."""
    coef, *_ = np.linalg.lstsq(_design(calibration), calibration["cars"].to_numpy(dtype=float), rcond=None)
    return coef


def expected_cars(days: pd.DataFrame, coef: np.ndarray) -> np.ndarray:
    """Cars at Fazzon in calibration-year conditions, from the day's traffic and weather (A37)."""
    return _design(days) @ coef


def dispersion(calibration: pd.DataFrame, coef: np.ndarray) -> float:
    """A38: variance of the cars around the relation / expected cars, on the calibration days."""
    expected = np.clip(expected_cars(calibration, coef), 1.0, None)
    return float(np.mean((calibration["cars"].to_numpy(dtype=float) - expected) ** 2 / expected))
