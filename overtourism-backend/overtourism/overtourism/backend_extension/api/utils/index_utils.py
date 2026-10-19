from __future__ import annotations

from datetime import timedelta
from math import ceil, floor

import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# Map response helpers
# ---------------------------------------------------------------------------


def _empty_map_response(gdf_base):
    gdf_base["INDICE"] = -1
    return {"geojson": gdf_base.to_json(), "min_value": -1, "max_value": -1}


def _build_geodataframe(gdf_base, result, join_on="ID_COMUNE", area_name_col="COMUNE"):
    return gpd.GeoDataFrame(
        gdf_base[["PRO_COM_T", "COMUNE", "geometry"]]
        .merge(result, left_on="PRO_COM_T", right_on=join_on, how="left")
        .rename(columns={area_name_col: "AREA_NAME"})
        .drop(columns=["PRO_COM_T", join_on]),
        geometry="geometry",
        crs=gdf_base.crs,
    )


def _translate_columns(
    gdf: gpd.GeoDataFrame, translations: dict[str, str]
) -> gpd.GeoDataFrame:
    """
    Rename GeoDataFrame columns using a predefined translation dictionary.

    Only columns whose current name is a key in `translations` are renamed
    (e.g. phenomenon keywords like "beds" / "population"); anything not
    listed there — "AREA_NAME", "INDICE", "geometry", etc. — is left
    untouched. Safe to call even if some mapped keys aren't present in
    `gdf` (e.g. "tasso-variazione" only ever has INDICE).

    Returns a new GeoDataFrame; does not mutate `gdf` in place.
    """
    rename_map = {col: translations[col] for col in gdf.columns if col in translations}
    return gdf.rename(columns=rename_map)


def _compute_min_max(gdf_final):
    # Compute min and max and clamp the minimum to 0.0 when <= 1
    try:
        min_val = float(gdf_final["INDICE"].min())
        max_val = float(gdf_final["INDICE"].max())

        if 0 <= min_val <= 1:
            min_val = 0.0

        if 0 <= max_val <= 1:
            # Round max up to the first decimal
            max_val = ceil(max_val * 10) / 10
        else:
            # Round max up and min down to nearest integer
            max_val = ceil(max_val)
            min_val = floor(min_val)

        if min_val == max_val == 0.0:
            max_val = 1.0

        if min_val == max_val == 1.0:
            min_val = 0.0

        if min_val == max_val == 100.0:
            min_val = 0.0

        return min_val, max_val

    except Exception:
        return -1, -1


# Columns whose name contains one of these markers are ratios / scores, not counts: they are rounded to
# the given number of decimals. Every other numeric column (except AREA_NAME / geometry) is a count and
# is rounded to an integer.
_DECIMAL_COLUMNS = {
    "INDICE": 2,
    "Periodo": 2,
    "quota": 3,
    "PageRank": 5,
    "superficie": 2,
}


def _round_count_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    for col in out.columns:
        decimals = next((d for m, d in _DECIMAL_COLUMNS.items() if m in col), None)
        if decimals is not None and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(decimals)
        elif col not in {
            "AREA_NAME",
            "geometry",
        } and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round().astype("Int64")
    return out


def _to_map_response(gdf_final):
    min_value, max_value = _compute_min_max(gdf_final)
    gdf_final = _round_count_columns(gdf_final)
    return {
        "geojson": gdf_final.to_json(),
        "min_value": min_value,
        "max_value": max_value,
    }


# ---------------------------------------------------------------------------
# Chart labels
# ---------------------------------------------------------------------------


def get_chart_labels(start_date, end_date, granularity: str) -> list[str]:
    if granularity == "giornaliero":
        labels, cur = [], start_date
        while cur <= end_date:
            labels.append(cur.strftime("%Y-%m-%d"))
            cur += timedelta(days=1)

    elif granularity == "settimanale":
        labels, cur = [], start_date - timedelta(days=start_date.weekday())
        while cur <= end_date:
            labels.append(cur.strftime("%Y-%m-%d"))
            cur += timedelta(days=7)

    elif granularity == "mensile":
        labels, cur = [], start_date.replace(day=1)
        while cur <= end_date:
            labels.append(cur.strftime("%Y-%m"))
            cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)

    else:  # annuale
        labels = [str(y) for y in range(start_date.year, end_date.year + 1)]

    return labels


# ---------------------------------------------------------------------------
# Dates filtering
# ---------------------------------------------------------------------------


def _easter_sunday(year: int) -> pd.Timestamp:
    """
    Date of Easter Sunday for `year`, via the Anonymous Gregorian
    (Meeus/Jones/Butcher) algorithm.
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=int(month), day=int(day))


def _italian_festivities(years) -> set:
    """
    Italian national public holidays for the given years: fixed-date
    holidays plus Easter Sunday and Easter Monday (Pasquetta).
    """
    fixed = [
        (1, 1),  # Capodanno
        (1, 6),  # Epifania
        (4, 25),  # Liberazione
        (5, 1),  # Festa dei lavoratori
        (6, 2),  # Festa della Repubblica
        (8, 15),  # Ferragosto
        (11, 1),  # Ognissanti
        (12, 8),  # Immacolata
        (12, 25),  # Natale
        (12, 26),  # Santo Stefano
    ]
    festivities: set = set()
    for year in years:
        year = int(year)
        for month, day in fixed:
            festivities.add(pd.Timestamp(year=year, month=month, day=day))
        easter = _easter_sunday(year)
        festivities.add(easter)
        festivities.add(easter + pd.Timedelta(days=1))
    return festivities
