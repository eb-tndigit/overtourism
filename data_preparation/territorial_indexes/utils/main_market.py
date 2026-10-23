# SPDX-License-Identifier: Apache-2.0
"""Top source countries by Vodafone arrivals for each comune and year."""

import logging

import numpy as np
import pandas as pd

from utils.config import TOP_N_PROVENIENZA_COUNTRIES
from utils.disaggregation import assert_sums_match, beds_weights, disaggregate, is_whole
from utils.datasets import ARRIVI_VALUE_COLS
from utils.markets import country_name

logger = logging.getLogger(__name__)


def area_market_volumes(df: pd.DataFrame) -> pd.DataFrame:
    """Rows (DATA, areaId, userCountry, value) -> yearly area/country arrivals, including countryless rows."""
    out = pd.DataFrame(
        {
            "year": df["DATA"].astype(str).str[:4].astype(int),
            "areaId": df["areaId"],
            "market": df["userCountry"].map(country_name).fillna(""),
            "value": df["value"],
        }
    )
    out = out.groupby(["year", "areaId", "market"], as_index=False, dropna=False)["value"].sum()
    return out[out["value"] > 0].reset_index(drop=True)


def split_areas_to_comuni(
    volumes: pd.DataFrame,
    area_comune: pd.DataFrame,
    strutture=None,
    bed_col="tot_postiletto",
    check=True,
) -> pd.DataFrame:
    """(year, areaId, market, value) -> (year, ID_COMUNE, market, value): the value of an area with several comuni is split
    by the beds of the year (`strutture`: processed strutture; without it, uniformly, with a warning). Integer counts stay
    integer (largest remainder) and the totals are checked. Areas with no comune are dropped with a warning."""
    unmapped = sorted(set(volumes["areaId"]) - set(area_comune["areaId"]))
    if unmapped:
        logger.warning("[main_market] areas with no comune (absent from the comune data): %s", unmapped)
    ids = area_comune.groupby("areaId")["ID_COMUNE"].agg(lambda s: tuple(dict.fromkeys(s)))
    df = volumes[volumes["areaId"].isin(ids.index)].copy()
    df["ID_COMUNE"] = df["areaId"].map(ids)
    df["DATA"] = df["year"].astype(str) + "-01-01"
    df["_TID"] = np.arange(len(df))

    kwargs = {}
    if strutture is not None:
        kwargs = dict(
            space_weights=beds_weights(strutture, df["year"].unique(), bed_col),
            space_weight_col=bed_col,
            space_time_freq="Y",  # beds are yearly
        )
    else:
        logger.warning("[main_market] no strutture given: uniform split among the comuni of an area")
    out = disaggregate(df, cols=["value"], axis="space", integer=is_whole(df, ["value"]), **kwargs)
    if check:
        assert_sums_match(df, out, ["value"], ["_TID"], name="main_market arrivals")
        assert_sums_match(df, out, ["value"], ["year"], name="main_market arrivals/year")
    return out[["year", "ID_COMUNE", "market", "value"]]


def main_market(comune_volumes: pd.DataFrame) -> pd.DataFrame:
    """Return total arrivals and the configured top country names and volumes per comune/year."""
    g = comune_volumes.groupby(["year", "ID_COMUNE", "market"], as_index=False)["value"].sum()
    g = g[g["value"] > 0]
    total = (
        g.groupby(["year", "ID_COMUNE"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "volume_totale"})
    )
    top = (
        g[g["market"] != ""]
        .sort_values(["year", "ID_COMUNE", "value", "market"], ascending=[True, True, False, True])
        .groupby(["year", "ID_COMUNE"], as_index=False, sort=False)
        .head(TOP_N_PROVENIENZA_COUNTRIES)
        .copy()
    )
    top["_rank"] = top.groupby(["year", "ID_COMUNE"]).cumcount() + 1
    out = total
    comune_index = pd.MultiIndex.from_frame(out[["year", "ID_COMUNE"]])
    for rank in range(1, TOP_N_PROVENIENZA_COUNTRIES + 1):
        ranked = top[top["_rank"] == rank].set_index(["year", "ID_COMUNE"])
        out[f"volume_mercato_{rank}"] = ranked["value"].reindex(comune_index).to_numpy()
        out[f"nome_mercato_{rank}"] = ranked["market"].reindex(comune_index).to_numpy()
    out["DATA"] = out["year"].astype(int)
    if is_whole(out, ["volume_totale"]):
        out["volume_totale"] = out["volume_totale"].round().astype("int64")
    return (
        out[["DATA", "ID_COMUNE", *ARRIVI_VALUE_COLS]]
        .sort_values(["DATA", "ID_COMUNE"])
        .reset_index(drop=True)
    )


def comune_main_market(
    df: pd.DataFrame,
    area_comune: pd.DataFrame,
    strutture=None,
    bed_col="tot_postiletto",
) -> pd.DataFrame:
    """Standardized rows (DATA, areaId, userCountry, value = arrivals) -> (DATA = year, ID_COMUNE,
    `volume_totale` and the configured top-country volumes/names). `area_comune`: areaId, ID_COMUNE
    (an area may have several comuni)."""
    if df.empty:
        raise ValueError("[main_market] no arrivals to process")
    volumes = area_market_volumes(df)
    comune = split_areas_to_comuni(volumes, area_comune, strutture, bed_col)
    return main_market(comune)
