# SPDX-License-Identifier: Apache-2.0
"""Adapter for the weather files (one per year, comune x day, nearest Meteotrentino station)."""

import pandas as pd

from utils.adapters.base import RawData, register_adapter


@register_adapter("meteo_comuni", kind="meteo_raw")
def meteo_comuni(raw: RawData) -> pd.DataFrame:
    """Weather file with columns date, comune_id, t_min, t_max (+ rain_mm, station info): returned unchanged."""
    return raw.df
