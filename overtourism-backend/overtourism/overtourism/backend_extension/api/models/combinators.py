"""
Python combinators that YAML indicator definitions can refer to by name (``combinator: <name>``).

Use one when the INDICE of an indicator cannot be written as a ``formula:`` (it needs a condition, a ranking, a
rounding option, ...) but the indicator is otherwise ordinary: its metadata and phenomena stay in the YAML file, and
only the maths lives here. (An indicator that must also add columns of its own or override other behaviour is a
Python *class* instead: see ``trentino_indicators.py``.)

How to add one
--------------
1. Write a function and register it with ``@combinator("name")``::

       @combinator("share_above")
       def share_above(df, *, column, threshold, **_extra):
           ...
           return series            # one value per row of ``df``

2. Use it from the YAML file::

       combinator: share_above
       params: {column: arrivals, threshold: 100}

Contract
--------
* ``df``: one row per comune (or per comune and day), one column per phenomenon of the indicator (by ``name``).
* Return a ``pandas.Series`` aligned with ``df`` (one INDICE value per row). A null means "no value": the base
  ``Indicator`` does NOT fill it, so a combinator that wants a value for missing data must say so explicitly.
* Keyword-only arguments that have no default and are not run-time values are the **params** read from the YAML
  ``params:`` mapping (the loader checks at start-up that they are all there and that none is misspelled).
* The run-time values are passed by the ``Indicator`` on every call: ``filtered_data``, ``start_date``,
  ``end_date`` and the request's *extra fields* (those listed in the YAML ``extra_fields:``). A combinator must
  accept ``**_extra`` to ignore the ones it does not use.
* It must be a pure function of its inputs: when an indicator is used inside another one the combinator is run once
  on the whole panel with ``start_date=None``/``end_date=None`` (see ``Indicator.resolve``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

COMBINATORS: dict[str, Callable[..., pd.Series]] = {}


def combinator(name: str) -> Callable[[Callable], Callable]:
    """Register a function under ``name`` so that a YAML file can use it as ``combinator: <name>``."""

    def register(fn: Callable) -> Callable:
        if name in COMBINATORS:
            raise ValueError(f"A combinator named {name!r} is already registered")
        COMBINATORS[name] = fn
        return fn

    return register


# ---------------------------------------------------------------------------
# Quantile classes (discretized PageRank)
# ---------------------------------------------------------------------------


@combinator("quantile_class")
def quantile_class(
    df: pd.DataFrame, *, column: str, n_classes: int = 5, **_extra
) -> pd.Series:
    """
    Quantile class (1..n_classes) of ``column`` among the rows of ``df``; 0 where the value is missing.

    1 = lowest ... n_classes = highest. Classes are quantiles of the comuni of the request (tied values stay in
    the same class, so the classes can have different sizes): they show the *relative* position of a comune. The
    "missing" class 0 is explicit because the base Indicator keeps nulls. With fewer than 2 valid values every row
    is 0.

    Because the classes depend on the comuni of the request, an indicator built on this should set
    ``available_for_variation: false`` (classes of different periods are not comparable as a time series).
    """
    if not isinstance(n_classes, int) or isinstance(n_classes, bool) or n_classes < 2:
        raise ValueError(f"n_classes must be an integer >= 2, got {n_classes!r}")
    value = df[column]
    classes = pd.Series(np.nan, index=df.index, dtype=float)
    valid = value.notna()
    if valid.sum() < 2:
        return classes.fillna(0)
    share_below = value[valid].rank(method="average", pct=True)
    classes[valid] = np.ceil(share_below * n_classes).clip(1, n_classes)
    return classes.fillna(0)


# ---------------------------------------------------------------------------
# Arrival-weighted average travel time
# ---------------------------------------------------------------------------

TRAVEL_TIME_ROUNDING_STEP_MINUTES = 30
MINUTES_PER_HOUR = 60


def _round_to_step(minutes: pd.Series, step: float) -> pd.Series:
    # half up (numpy's round goes to the even one: 45 -> 60 but 15 -> 0); missing values stay missing
    return np.floor(minutes / step + 0.5) * step


@combinator("average_travel_time")
def average_travel_time(
    df: pd.DataFrame,
    *,
    minutes_column: str,
    arrivals_column: str,
    arrotonda_30_minuti=False,
    **_extra,
) -> pd.Series:
    """
    Mean travel time in HOURS: ``minutes_column`` (sum of arrivals * minutes) over ``arrivals_column`` (sum of the
    arrivals), converted from minutes to hours (270 min -> 4.5). Null where there are no arrivals.

    Run-time option ``arrotonda_30_minuti``: round the average to multiples of half an hour (half up).
    """
    average_minutes = df[minutes_column].div(df[arrivals_column].replace(0, np.nan))
    if arrotonda_30_minuti:
        average_minutes = _round_to_step(
            average_minutes, TRAVEL_TIME_ROUNDING_STEP_MINUTES
        )
    return average_minutes / MINUTES_PER_HOUR
