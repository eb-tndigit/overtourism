# SPDX-License-Identifier: Apache-2.0
"""Confidence interval of the sustainability over the days of a period."""

import numpy as np
import pytest
from civic_digital_twins.dt_model import CategoricalIndex

from src.model.common.sustainability_field import (
    compute_sustainability_index_with_ci,
    expected_period_days,
)


def test_expected_period_days():
    season = CategoricalIndex("season", {"june": 0.25, "july": 0.25, "august": 0.5})
    weather = CategoricalIndex("weather", {"good": 0.6, "rainy": 0.4})
    assert expected_period_days(120, {}, [season, weather]) == 120
    assert expected_period_days(120, {season: "august"}, [season, weather]) == 60
    assert expected_period_days(
        120, {season: ["june", "july"], weather: "good"}, [season, weather]
    ) == pytest.approx(36)
    # re-weighted, not restricted: the whole period
    assert (
        expected_period_days(120, {weather: {"good": 0.9, "rainy": 0.1}}, [weather])
        == 120
    )


def test_interval_over_the_period_days():
    """The interval is that of the average over the period's days, not over the samples:
    the same samples give a wider interval for a shorter period."""
    x = y = np.array([0.0, 1.0])
    field = np.array([[1.0, 1.0], [0.0, 0.0]])  # sustainable at x = 0, not at x = 1
    rng = np.random.default_rng(0)
    presences = [(float(v), 0.5) for v in rng.integers(0, 2, 2000)]
    m, h_samples = compute_sustainability_index_with_ci(field, x, y, presences)
    _, h_season = compute_sustainability_index_with_ci(
        field, x, y, presences, period_days=122
    )
    _, h_month = compute_sustainability_index_with_ci(
        field, x, y, presences, period_days=31
    )
    assert 0.45 < m < 0.55
    assert h_samples < h_season < h_month
    assert h_season / h_samples == pytest.approx(np.sqrt(2000 / 122), rel=0.05)
