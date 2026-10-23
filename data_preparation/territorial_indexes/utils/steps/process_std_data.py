# SPDX-License-Identifier: Apache-2.0
"""
STEP 2 - Processing.

Input : Output/data/normalized/   (+ mapping json files, read from Output/mapping; the ISPAT presences
        arrivals / presences are read from Output/data/raw_data/, they are not normalized)
Output: Output/data/data_processed/   (popolazione_pr, strutture_pr, vodafone_pr,
                                       presenze_alb_pr, presenze_extralb_pr,
                                       ee_prelievi_pr, gas_prelievi_pr, ee_totale_pr, gas_totale_pr,
                                       vodafone_covisits_pr, vodafone_provenienza_pr, vodafone_arrivi_pr)

Transformations: ID_COMUNE resolution, filtering, selection of the columns, computation of
aggregated columns, disaggregation of the presences to comune x day (vodafone: areas -> comuni by beds;
ISPAT: APT / provincia x month -> comune x day, with the average stay of the month, see stay_distribution.py). Every row has a single ID_COMUNE.
Energy categories are summed per comune-month and divided evenly over the calendar days of each month.
The Vodafone OD co-visits are aggregated per YEAR and turned into flow indicators per comune (quota in
ingresso / in uscita, quota intra-ambito, PageRank, per profile): one row per comune and year, see flows.py.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils.cleaning import (
    ids_to_int,
    normalize_id_comune,
    pad_id_comune,
    remove_provincia,
    resolve_id_comune,
    standard_ordering_cols,
)
from utils.config import (
    BASE_WINDOW,
    REFERENCE_CITY_ITALY,
    DATA_START,
    HDD_BASE_TEMP,
    MAPPING_DIR,
    METEO_FILE_PATTERN,
    METEO_FOLDER,
    MAPPING_FILES,
    NORMALIZED_DIR,
    PRESENZE_FILE_PATTERN,
    PRESENZE_MEASURES,
    PRESENZE_TAGS,
    PROCESSED_DIR,
    RAW_DIR,
    REFERENCES,
    STD_SUFFIX,
    TYPE_FORMAT,
    TRAVEL_TIMES_MATRIX_FILE,
    setup_logging,
)
from utils.datasets import (  # noqa: F401  (re-exported for convenience)
    ARRIVI_FULL_YEAR_DAYS,
    ARRIVI_VALUE_COL,
    COVISITS_FULL_YEAR_DAYS,
    PROVENIENZA_FULL_YEAR_DAYS,
    PROVENIENZA_LOC_TYPE,
    PROVENIENZA_MIN_PRESENZE,
    PROVENIENZA_PROFILE,
    COVISITS_METRICS,
    COVISITS_MIN_T,
    COVISITS_PAGERANK_DAMPING,
    COVISITS_PROFILE_WEIGHTS,
    COVISITS_PROFILES,
    COVISITS_VALUE_COLS,
    DATASETS,
    METEO_BASE_YEARS,
    METEO_VALUE_COLS,
    PRESENZE_BASE_YEARS,
    PRESENZE_SPILL_OVER,
    POPOLAZIONE_VALUE_COLS,
    PRESENZE_ALB_VALUE_COLS,
    PRESENZE_XALB_VALUE_COLS,
    EE_PRELIEVI_VALUE_COLS,
    GAS_PRELIEVI_VALUE_COLS,
    EE_TOTALE_VALUE_COLS,
    GAS_TOTALE_VALUE_COLS,
    STRUTTURE_VALUE_COLS,
    VODAFONE_VALUE_COLS,
)
from utils.steps.standardize_raw_data import ENERGY_MONTHS
from utils.window import Window, clip
from utils.adapters.presenze import MONTHS_MAPPING
from utils.disaggregation import (
    assert_sums_match,
    beds_weights,
    disaggregate,
    is_whole,
)
from utils.flows import build_area_comune, comune_flow_indicators
from utils.main_market import comune_main_market, split_areas_to_comuni
from utils.provenienza import comune_hhi
from utils.stay_distribution import distribute_by_stay
from utils.io import read_df, read_json, read_table, save_computed_dfs

from trentino_travel_times import ensure_travel_time_matrix

logger = logging.getLogger(__name__)

BEDS_ALB_COL = "posti_letto_alb"
BEDS_XALB_COL = "posti_letto_xalb"


def _filtering_strutture(df, window, year_col="DATA"):
    """Keeps the years of the time window (nothing before `period.start`: geography changes of the
    municipalities make the older years not comparable)."""
    return clip(df, window, col=year_col, name="strutture")


def process_popolazione(df, mapping_comuni, window=BASE_WINDOW):
    df = clip(df, window, name="popolazione")
    df["ID_COMUNE"] = df["LOCATION"].apply(
        lambda x: resolve_id_comune(x, mapping_comuni)
    )
    df = remove_provincia(df, comune_col="LOCATION")
    df["ID_COMUNE"] = pad_id_comune(df["ID_COMUNE"])
    return standard_ordering_cols(df[["DATA", "ID_COMUNE"] + POPOLAZIONE_VALUE_COLS])


def process_strutture(df, mapping_comuni, window=BASE_WINDOW):
    df = _filtering_strutture(df, window)
    df = remove_provincia(df, comune_col="LOCATION")

    # Rename source columns
    df = df.rename(
        columns={
            "alberghieri posti_letto": "letti_alb",
            "extra alb. Posti_letto": "letti_exalb",
            "all. privati posti_letto": "letti_alloggi_turistici",
            "all. disposizione posti_letto": "letti_alloggi_disp",
            "alberghieri strutture": "strutture_alb",
            "extra alb. Strutture": "strutture_exalb",
            "all. privati numero": "strutture_alloggi_turistici",
            "all.disposizione numero": "strutture_alloggi_disp",
        }
    )

    df["tot_strutture"] = (
        df["strutture_alb"] + df["strutture_exalb"] + df["strutture_alloggi_turistici"]
    )
    df["tot_postiletto"] = (
        df["letti_alb"] + df["letti_exalb"] + df["letti_alloggi_turistici"]
    )

    # Resolve municipality IDs
    df["ID_COMUNE"] = df["LOCATION"].apply(
        lambda x: resolve_id_comune(x, mapping_comuni)
    )

    missing = df.loc[df["ID_COMUNE"].isna(), "LOCATION"].unique()
    if len(missing) > 0:
        logger.warning(
            "[process_strutture] No ID_COMUNE found (even with overrides) for: %s",
            sorted(missing),
        )

    df["ID_COMUNE"] = pad_id_comune(df["ID_COMUNE"])

    return standard_ordering_cols(df[["DATA", "ID_COMUNE"] + STRUTTURE_VALUE_COLS])


def process_vodafone(
    df,
    mapping_vodafone,
    strutture=None,
    bed_col="tot_postiletto",
    check=True,
    window=BASE_WINDOW,
    popolazione=None,
):
    """Vodafone TOURIST and VISITOR presences: areas x day -> comune x day.

    TOURIST counts are split by beds and VISITOR (excursionist) counts by population.
    Single-comune areas are unchanged. With check=True, both totals are checked after
    allocation.
    """
    if popolazione is None:
        raise ValueError(
            "[process_vodafone] population data are required to allocate VISITOR presences"
        )
    df = df[
        df["userProfile"].isin(["TOURIST", "VISITOR"])
        & (df["locType"] == "TN_MKT_AL_3")
    ].copy()
    df["ID_COMUNE"] = df["LOCATION"].map(mapping_vodafone)
    mask = df["LOCATION"] == "SAN GIOVANNI DI FASSA"
    df.loc[mask, "ID_COMUNE"] = pd.Series(
        [[22250]] * mask.sum(), index=df.index[mask], dtype=object
    )
    if df["ID_COMUNE"].isna().any():
        locations = sorted(
            df.loc[df["ID_COMUNE"].isna(), "LOCATION"].dropna().unique().tolist()
        )
        logger.warning(
            "[process_vodafone] %d rows Vodafone (%d aree) with no mapping ID_COMUNE; "
            "rows will be excluded by groupby: %s",
            int(df["ID_COMUNE"].isna().sum()),
            len(locations),
            locations,
        )
    df["DATA"] = pd.to_datetime(df["DATA"].astype(str), errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    df = clip(
        df, window, name="vodafone"
    )  # before the (expensive) split over the comuni
    df["ID_COMUNE"] = pad_id_comune(df["ID_COMUNE"]).apply(
        normalize_id_comune
    )  # hashable

    # Split the raw value by profile. The visitor column must be derived from the ORIGINAL value, before
    # `presenze` is restricted to TOURIST (doing it the other way round zeroes every visitor row).
    raw_value = df["presenze"]
    df["presenze_escursionisti_vodafone"] = raw_value.where(
        df["userProfile"] == "VISITOR", 0
    )
    df["presenze"] = raw_value.where(df["userProfile"] == "TOURIST", 0)
    # Totals of the raw input (mapped areas only, as the groupby below drops the others): the grouped data must
    # keep them. Comparing with the already-derived columns alone cannot catch a profile lost before this point.
    mapped = df["ID_COMUNE"].notna()
    expected = {
        "presenze": raw_value[mapped & (df["userProfile"] == "TOURIST")].sum(),
        "presenze_escursionisti_vodafone": raw_value[
            mapped & (df["userProfile"] == "VISITOR")
        ].sum(),
    }
    df = df.groupby(["DATA", "ID_COMUNE"], as_index=False)[
        VODAFONE_VALUE_COLS
    ].sum()
    for col, total in expected.items():
        if not np.isclose(df[col].sum(), total):
            raise ValueError(
                f"[process_vodafone] {col}: {df[col].sum()} after grouping, {total} in the input profile rows"
            )
    if expected["presenze_escursionisti_vodafone"] > 0:
        logger.info(
            "[process_vodafone] VISITOR presences in input: %s", expected["presenze_escursionisti_vodafone"]
        )
    df["_TID"] = np.arange(len(df))

    shared_visitor_areas = df.loc[
        df["presenze_escursionisti_vodafone"].ne(0)
        & df["ID_COMUNE"].map(lambda ids: isinstance(ids, tuple) and len(ids) > 1),
        ["DATA", "ID_COMUNE"],
    ].explode("ID_COMUNE")
    if not shared_visitor_areas.empty:
        population_weights = popolazione[["DATA", "ID_COMUNE", "popolazione"]].copy()
        shared_visitor_areas["_YEAR"] = pd.to_datetime(
            shared_visitor_areas["DATA"].astype(str), errors="raise"
        ).dt.year
        population_weights["_YEAR"] = pd.to_datetime(
            population_weights["DATA"].astype(str), errors="raise"
        ).dt.year
        shared_visitor_areas["ID_COMUNE"] = (
            shared_visitor_areas["ID_COMUNE"].astype(str).str.zfill(6)
        )
        population_weights["ID_COMUNE"] = (
            population_weights["ID_COMUNE"].astype(str).str.zfill(6)
        )
        population_weights["popolazione"] = pd.to_numeric(
            population_weights["popolazione"], errors="coerce"
        )
        joined = shared_visitor_areas.merge(
            population_weights[["_YEAR", "ID_COMUNE", "popolazione"]],
            on=["_YEAR", "ID_COMUNE"],
            how="left",
        )
        invalid = joined["popolazione"].isna() | joined["popolazione"].le(0)
        if invalid.any():
            missing = joined.loc[invalid, ["_YEAR", "ID_COMUNE"]].drop_duplicates()
            raise ValueError(
                "[process_vodafone] missing positive population for multi-comune "
                f"VISITOR allocation: {missing.to_dict('records')}"
            )

    tourist_kwargs = {}
    if strutture is not None:
        years = pd.to_datetime(df["DATA"]).dt.year.unique()
        tourist_kwargs = dict(
            space_weights=beds_weights(strutture, years, bed_col),
            space_weight_col=bed_col,
            space_time_freq="Y",  # beds are yearly
        )
    else:
        logger.warning("[process_vodafone] no strutture given: uniform TOURIST split")

    tourist = disaggregate(
        df,
        cols=["presenze"],
        axis="space",
        integer=is_whole(df, ["presenze"]),
        **tourist_kwargs,
    )
    visitors = disaggregate(
        df,
        cols=["presenze_escursionisti_vodafone"],
        axis="space",
        space_weights=popolazione,
        space_weight_col="popolazione",
        space_time_freq="Y",
        integer=is_whole(df, ["presenze_escursionisti_vodafone"]),
    )
    keys = ["DATA", "ID_COMUNE"]
    out = tourist[keys + ["presenze"]].merge(
        visitors[keys + ["presenze_escursionisti_vodafone"]],
        on=keys,
        how="outer",
    )

    if check:
        assert_sums_match(
            df, tourist, ["presenze"], ["_TID"], name="vodafone TOURIST"
        )
        assert_sums_match(
            df,
            visitors,
            ["presenze_escursionisti_vodafone"],
            ["_TID"],
            name="vodafone VISITOR",
        )
        assert_sums_match(df, tourist, ["presenze"], ["DATA"], name="vodafone TOURIST/day")
        assert_sums_match(
            df,
            visitors,
            ["presenze_escursionisti_vodafone"],
            ["DATA"],
            name="vodafone VISITOR/day",
        )
    return standard_ordering_cols(out.reset_index(drop=True))


def process_provenienza(df, area_comune, window=BASE_WINDOW):
    """Standardized Vodafone presences (with `userCountry`) -> comune x YEAR concentration of the source markets:
    `hhi_provenienza` (normalized HHI of the tourist presences by country, see utils/provenienza.py) and
    `giorni_coperti`. Only TOURIST, area level; rows without country are ignored. A year with fewer than
    PROVENIENZA_FULL_YEAR_DAYS days is partial: kept, flagged by `giorni_coperti` and by a warning."""
    if "userCountry" not in df.columns:
        raise ValueError(
            "[process_provenienza] the Vodafone presences have no `userCountry` column: "
            "the source markets cannot be computed"
        )
    df = df[
        (df["userProfile"] == PROVENIENZA_PROFILE) & (df["locType"] == PROVENIENZA_LOC_TYPE)
    ]
    df = clip(df, window, name="provenienza")
    df = df[df["userCountry"].notna() & (df["userCountry"].astype(str).str.strip() != "")]
    df = pd.DataFrame(
        {
            "DATA": df["DATA"].astype(str).str[:10],
            "areaId": df["locId"].astype(str).str.replace(r"\.0$", "", regex=True),
            "userCountry": df["userCountry"].astype(str),
            "value": pd.to_numeric(df["presenze"], errors="coerce").fillna(0.0),
        }
    )
    out = comune_hhi(df, area_comune, PROVENIENZA_MIN_PRESENZE)
    partial = out.groupby("DATA")["giorni_coperti"].first()
    for year, days in partial[partial < PROVENIENZA_FULL_YEAR_DAYS].items():
        logger.warning(
            "[process_provenienza] %s is a partial year (%d days): compare it with care", year, days
        )
    return out


def _read_travel_time_routes(path, countries, reference_city_italy):
    """Read only the reference-city and required country rows from the route matrix."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Travel-time matrix not found: {path}. Run gen_phenomenon_dataframes.py to generate it automatically."
        )
    columns = [
        "istat_origine",
        "provenienza",
        "istat_destinazione",
        "tempo_di_percorrenza",
    ]
    wanted_countries = {str(country).strip().upper() for country in countries}
    chunks = []
    for chunk in pd.read_csv(path, usecols=columns, dtype=str, chunksize=250_000):
        country_origin = (chunk["istat_origine"] == "*") & chunk["provenienza"].str.upper().isin(wanted_countries)
        city_origin = (
            (chunk["istat_origine"] != "*")
            & (chunk["provenienza"].str.casefold() == reference_city_italy.casefold())
        )
        selected = chunk.loc[country_origin | city_origin]
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        return pd.DataFrame(columns=columns)
    return pd.concat(chunks, ignore_index=True)


def _mean_tourist_travel_time(df, area_comune, strutture, travel_times, reference_city_italy):
    """Yearly arrival-weighted route time from each origin market to each comune."""
    if "userCountry" not in df.columns:
        raise ValueError("[tempo_medio_turisti] Vodafone arrivals have no userCountry column")
    countries = df["userCountry"].fillna("").astype(str).str.strip().str.upper()
    values = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    valid = countries.ne("") & values.gt(0)
    by_area = pd.DataFrame(
        {
            "year": df.loc[valid, "DATA"].astype(str).str[:4].astype(int),
            "areaId": df.loc[valid, "areaId"].astype(str).str.replace(r"\\.0$", "", regex=True),
            "market": countries[valid],
            "value": values[valid],
        }
    ).groupby(["year", "areaId", "market"], as_index=False)["value"].sum()
    if by_area.empty:
        return pd.DataFrame(columns=["DATA", "ID_COMUNE", "tempo_medio_turisti"])

    comune_arrivals = split_areas_to_comuni(by_area, area_comune, strutture)
    matrix = travel_times
    city_routes = matrix.loc[
        (matrix["istat_origine"].astype(str) != "*")
        & (matrix["provenienza"].astype(str).str.casefold() == reference_city_italy.casefold())
    ].copy()
    city_ids = city_routes["istat_origine"].astype(str).unique()
    if len(city_ids) != 1:
        raise ValueError(
            f"[tempo_medio_turisti] reference city {reference_city_italy!r} resolves to {len(city_ids)} origins in the route matrix"
        )
    city_routes["userCountry"] = "ITA"
    foreign_routes = matrix.loc[matrix["istat_origine"].astype(str) == "*"].copy()
    foreign_routes["userCountry"] = foreign_routes["provenienza"].astype(str).str.upper()
    routes = pd.concat([city_routes, foreign_routes], ignore_index=True)
    routes = routes.rename(
        columns={
            "istat_destinazione": "ID_COMUNE",
            "tempo_di_percorrenza": "tempo_di_percorrenza",
        }
    )[["userCountry", "ID_COMUNE", "tempo_di_percorrenza"]]
    routes["ID_COMUNE"] = routes["ID_COMUNE"].astype(str).str.zfill(6)
    routes["tempo_di_percorrenza"] = pd.to_numeric(routes["tempo_di_percorrenza"], errors="coerce")
    routes = routes.dropna(subset=["tempo_di_percorrenza"]).drop_duplicates(
        ["userCountry", "ID_COMUNE"]
    )

    comune_arrivals["userCountry"] = comune_arrivals["market"].astype(str).str.upper()
    comune_arrivals["ID_COMUNE"] = comune_arrivals["ID_COMUNE"].astype(str).str.zfill(6)
    matched = comune_arrivals.merge(
        routes, on=["userCountry", "ID_COMUNE"], how="left", validate="many_to_one"
    )
    missing = matched[matched["tempo_di_percorrenza"].isna()]
    if not missing.empty:
        logger.warning(
            "[tempo_medio_turisti] arrivals excluded because no route exists for origins: %s",
            sorted(missing["userCountry"].unique()),
        )
    matched = matched.dropna(subset=["tempo_di_percorrenza"])
    matched["_weighted_time"] = matched["value"] * matched["tempo_di_percorrenza"]
    out = matched.groupby(["year", "ID_COMUNE"], as_index=False).agg(
        _weighted_time=("_weighted_time", "sum"), arrivals=("value", "sum")
    )
    out["tempo_medio_turisti"] = out["_weighted_time"] / out["arrivals"]
    out["DATA"] = out["year"]
    return out[["DATA", "ID_COMUNE", "tempo_medio_turisti"]].sort_values(
        ["DATA", "ID_COMUNE"]
    ).reset_index(drop=True)


def process_provenienza_arrivi(
    df,
    area_comune,
    strutture=None,
    window=BASE_WINDOW,
    travel_times=None,
    reference_city_italy=REFERENCE_CITY_ITALY,
):
    """Standardized Vodafone arrivals -> comune x YEAR total and top-country volumes/names.
    Countryless arrivals contribute to `volume_totale` but cannot be assigned a country. Areas with several
    comuni are split by beds (`strutture`). Partial years are kept, with a warning."""
    df = df[
        (df["userProfile"] == PROVENIENZA_PROFILE) & (df["locType"] == PROVENIENZA_LOC_TYPE)
    ]
    df = clip(df, window, name="vodafone_arrivi")
    days = df.assign(year=df["DATA"].astype(str).str[:4].astype(int)).groupby("year")["DATA"].nunique()
    for year, n in days[days < ARRIVI_FULL_YEAR_DAYS].items():
        logger.warning(
            "[process_provenienza_arrivi] %s is a partial year (%d days): compare it with care", year, n
        )
    countries = df["userCountry"] if "userCountry" in df.columns else pd.Series(None, index=df.index)
    df = pd.DataFrame(
        {
            "DATA": df["DATA"].astype(str).str[:10],
            "areaId": df["locId"].astype(str).str.replace(r"\.0$", "", regex=True),
            "userCountry": countries,
            "value": pd.to_numeric(df[ARRIVI_VALUE_COL], errors="coerce").fillna(0.0),
        }
    )
    out = comune_main_market(df, area_comune, strutture)
    countries = df.get("userCountry", pd.Series(dtype=str)).dropna().astype(str).str.strip()
    countries = countries[countries.ne("")].str.upper().unique()
    if travel_times is None:
        travel_times = _read_travel_time_routes(
            TRAVEL_TIMES_MATRIX_FILE, countries, reference_city_italy
        )
    travel_time = _mean_tourist_travel_time(
        df, area_comune, strutture, travel_times, reference_city_italy
    )
    out = out.merge(travel_time, on=["DATA", "ID_COMUNE"], how="left")
    return standard_ordering_cols(out)


def stack_covisits(frames):
    """Standardized OD frames in order: on the dates present in a later frame the later one wins
    (a later file corrects the earlier ones, like vodafone_attendences / _new)."""
    frames = list(frames)
    for k in range(len(frames) - 1):
        later_dates = set().union(*[set(f["DATA"]) for f in frames[k + 1 :]])
        n0 = len(frames[k])
        frames[k] = frames[k][~frames[k]["DATA"].isin(later_dates)]
        if len(frames[k]) < n0:
            logger.info(
                "[covisits] %d rows replaced by the later data", n0 - len(frames[k])
            )
    od = pd.concat(frames, ignore_index=True)
    key = [c for c in od.columns if c != "value"]
    return od.groupby(key, as_index=False, observed=True)[
        "value"
    ].sum()  # duplicated rows


def covisits_area_comune(mapping_dir, mapping_vodafone, mapping_comuni):
    """Area -> ID_COMUNE table of the OD co-visits (geojson of the areas + mapping_vodafone)."""
    geojson = Path(mapping_dir) / Path(REFERENCES["geojson"]).name
    return build_area_comune(geojson, mapping_vodafone, mapping_comuni)


def process_covisits(df, area_comune, window=BASE_WINDOW):
    """Vodafone OD co-visits: area x area x day -> comune x YEAR flow indicators.

    Wide output, one row per comune and year: DATA (year), ID_COMUNE (zero-padded) and
    `<metric>_<profile>` (quota_ingresso, quota_uscita, quota_intra_ambito, pagerank; tourist, visitor and
    combined) + `giorni_coperti`. The `_combined` columns merge the two profiles with the fixed weights
    COVISITS_PROFILE_WEIGHTS (TOURIST 1, VISITOR 0.5). A year with fewer than COVISITS_FULL_YEAR_DAYS days is partial: it
    is kept, flagged by `giorni_coperti` and by a warning."""
    df = clip(
        df, window, name="covisits"
    )  # only the days of the window enter the yearly indicators
    out = comune_flow_indicators(
        df,
        area_comune,
        min_t=COVISITS_MIN_T,
        damping=COVISITS_PAGERANK_DAMPING,
        profiles=COVISITS_PROFILES,
        metrics=COVISITS_METRICS,
        weights=COVISITS_PROFILE_WEIGHTS,
    )
    partial = out.groupby("DATA")["giorni_coperti"].first()
    for year, days in partial[partial < COVISITS_FULL_YEAR_DAYS].items():
        logger.warning(
            "[process_covisits] %s is a partial year (%d days): compare it with care",
            year,
            days,
        )
    out = out[["DATA", "ID_COMUNE"] + COVISITS_VALUE_COLS]
    # Nulls are expected (see METHODOLOGY.md, "phen_flussi_od"): report them so they can be checked per column.
    nulls = out.drop(columns=["DATA", "ID_COMUNE"]).isna().mean()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        logger.info(
            "[process_covisits] share of empty values (%d comune-years): %s",
            len(out),
            {col: f"{share:.0%}" for col, share in nulls.items()},
        )
    return standard_ordering_cols(out)


def process_beds(strutture_std, mapping_comuni, window=BASE_WINDOW):
    """Beds per comune and year from the NORMALIZED strutture: DATA (year), ID_COMUNE, posti_letto_alb
    (`alberghieri posti_letto`), posti_letto_xalb (`extra alb. Posti_letto`). The years up to the end of
    `window` are kept (the update needs the previous years too, hence only the end matters beyond start).
    """
    df = _filtering_strutture(strutture_std, window)
    df = remove_provincia(df, comune_col="LOCATION")
    df["ID_COMUNE"] = df["LOCATION"].apply(
        lambda x: resolve_id_comune(x, mapping_comuni)
    )
    missing = df.loc[df["ID_COMUNE"].isna(), "LOCATION"].unique()
    if len(missing) > 0:
        logger.warning("[process_beds] No ID_COMUNE found for: %s", sorted(missing))
    df = df[df["ID_COMUNE"].notna()].copy()
    df["ID_COMUNE"] = pad_id_comune(df["ID_COMUNE"])
    df = df.rename(
        columns={
            "alberghieri posti_letto": BEDS_ALB_COL,
            "extra alb. Posti_letto": BEDS_XALB_COL,
        }
    )
    return df[["DATA", "ID_COMUNE", BEDS_ALB_COL, BEDS_XALB_COL]].reset_index(drop=True)


def _energy_monthly(df, mapping_comuni, value_col, source):
    """Standardized energy rows -> monthly consumption per comune (DATA = first day of the month, ID_COMUNE,
    value_col), summing the usage categories (`codice_utilizzo`)."""
    df = df.copy()
    df = remove_provincia(df, comune_col="LOCATION", upper=True)
    df["ID_COMUNE"] = df["LOCATION"].apply(
        lambda name: resolve_id_comune(name, mapping_comuni)
    )
    missing = sorted(df.loc[df["ID_COMUNE"].isna(), "LOCATION"].dropna().unique())
    if missing:
        logger.warning(
            "[%s] Excluding locations without municipality IDs: %s", source, missing
        )
    df = df[df["ID_COMUNE"].notna()].copy()
    df["ID_COMUNE"] = pad_id_comune(df["ID_COMUNE"])

    keys = ["DATA", "ID_COMUNE"]
    long = df.melt(
        id_vars=keys,
        value_vars=list(ENERGY_MONTHS),
        var_name="_MONTH",
        value_name=value_col,
    )
    raw_values = long[value_col]
    long[value_col] = pd.to_numeric(raw_values, errors="coerce")
    invalid = raw_values.notna() & long[value_col].isna()
    if invalid.any():
        raise ValueError(
            f"[{source}] non-numeric consumption values: "
            f"{raw_values.loc[invalid].head().tolist()}"
        )
    month_numbers = {month: number for number, month in enumerate(ENERGY_MONTHS, 1)}
    long["DATA"] = pd.to_datetime(
        {
            "year": pd.to_numeric(long["DATA"], errors="raise").astype(int),
            "month": long["_MONTH"].map(month_numbers),
            "day": 1,
        }
    )
    return (
        long.groupby(keys, as_index=False)[value_col]
        .sum(min_count=1)
        .sort_values(keys)
        .reset_index(drop=True)
    )


def process_energy(df, mapping_comuni, value_col, source, window=BASE_WINDOW):
    """Sum energy categories, then spread each municipality-month total evenly over its days.
    Only the days of the time window are kept."""
    df = clip(df, window, name=source)  # years (DATA = anno) first: cheap
    monthly = _energy_monthly(df, mapping_comuni, value_col, source)
    daily = monthly.loc[monthly.index.repeat(monthly["DATA"].dt.days_in_month)].copy()
    if not daily.empty:
        days_in_month = daily["DATA"].dt.days_in_month
        day_offsets = daily.groupby(level=0).cumcount()
        daily["DATA"] = daily["DATA"] + pd.to_timedelta(day_offsets, unit="D")
        daily[value_col] = daily[value_col] / days_in_month
    out = clip(daily.reset_index(drop=True), window, name=source)  # exact days
    # Keep dates as ISO strings in parquet as well: timestamp columns are exposed by
    # some downstream readers as epoch-millisecond integers.
    out["DATA"] = out["DATA"].dt.strftime("%Y-%m-%d")
    return standard_ordering_cols(out)


def meteo_files(raw_dir, years):
    """Weather files of `years` (`files.meteo.pattern` of settings.yaml) present in raw_dir/<meteo folder>/."""
    files = [Path(raw_dir) / METEO_FOLDER / METEO_FILE_PATTERN.format(year=y) for y in years]
    return [f for f in files if f.exists()]


def process_meteo(files, window=BASE_WINDOW, base_temp=HDD_BASE_TEMP):
    """Weather files (date, comune_id, t_min, t_max, rain_mm + station info) -> comune x day temperature of the
    nearest Meteotrentino station: DATA, ID_COMUNE, t_min, t_max, t_mean, hdd, rain_mm.

        t_mean = (t_min + t_max) / 2
        hdd    = max(0, base_temp - t_mean)   heating degree days of the day (degrees C)

    A day without temperature has t_mean / hdd empty (never filled): whoever uses them averages the days that have a
    value. Only the days of the window are kept."""
    if not files:
        raise FileNotFoundError("no weather file found: run the download step")
    raw = pd.concat([read_table(f) for f in files], ignore_index=True)
    missing = {"date", "comune_id", "t_min", "t_max"} - set(raw.columns)
    if missing:
        raise ValueError(f"weather files: missing columns {sorted(missing)}")
    df = pd.DataFrame(
        {
            "ID_COMUNE": pad_id_comune(raw["comune_id"]),
            "DATA": pd.to_datetime(raw["date"], errors="coerce"),
            "t_min": pd.to_numeric(raw["t_min"], errors="coerce"),
            "t_max": pd.to_numeric(raw["t_max"], errors="coerce"),
            "rain_mm": pd.to_numeric(raw.get("rain_mm"), errors="coerce"),
        }
    )
    df = clip(df.dropna(subset=["DATA"]), window, name="meteo")
    dup = df.duplicated(["ID_COMUNE", "DATA"], keep=False)
    if dup.any():  # e.g. several stations per comune, or two files covering the same day
        logger.warning("[meteo] %d rows with a duplicated comune + day: averaged", int(dup.sum()))
        df = df.groupby(["ID_COMUNE", "DATA"], as_index=False).mean(numeric_only=True)
    df["t_mean"] = (df["t_min"] + df["t_max"]) / 2
    df["hdd"] = (base_temp - df["t_mean"]).clip(lower=0)
    n_missing = int(df["t_mean"].isna().sum())
    if n_missing:
        logger.warning("[meteo] %d/%d comune-days without temperature (hdd left empty)", n_missing, len(df))
    df["DATA"] = df["DATA"].dt.strftime("%Y-%m-%d")
    out = df[["DATA", "ID_COMUNE", *METEO_VALUE_COLS]]
    return out.sort_values(["DATA", "ID_COMUNE"]).reset_index(drop=True)


def read_strutture_std(normalized_dir, type_format=TYPE_FORMAT):
    """Every normalized strutture file (base `strutture_std` + the ones of the updates, e.g.
    `strutture_2024_update_std`), stacked: on a duplicated DATA + LOCATION the later file wins.
    The names follow `datasets.strutture.std` and `naming.std_suffix` of settings.yaml.
    """
    normalized_dir = Path(normalized_dir)
    spec = DATASETS["strutture"]
    base = spec.std_name
    names = sorted(
        f.stem
        for f in normalized_dir.glob(f"{spec.name}*{STD_SUFFIX}.{type_format}")
        if f.stem != base
    )
    frames = [read_df(normalized_dir, n, type_format) for n in [base, *names]]
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["DATA", "LOCATION"], keep="last").reset_index(drop=True)


# ---- ISPAT arrivals / presences: read as downloaded, no normalization
def _monthly_long(df, year, source):
    """Rows of the 12 months (the `Anno` total row and the empty row are dropped) -> long frame with
    DATA = first day of the month, `variable` (column name) and `value`."""
    first = df.columns[0]
    month = df[first].astype(str).str.strip().map(MONTHS_MAPPING)
    if month.notna().sum() != 12 or month.dropna().nunique() != 12:
        raise ValueError(
            f"{source}: expected the 12 months in the first column, found {df[first].tolist()}"
        )
    dropped = df.loc[month.isna(), first].tolist()
    logger.info("[%s] rows dropped: %s", source, dropped)
    df = df.loc[month.notna()].copy()
    df["DATA"] = [
        pd.Timestamp(year=int(year), month=int(m), day=1) for m in month[month.notna()]
    ]
    long = df.drop(columns=first).melt(
        id_vars="DATA", var_name="variable", value_name="value"
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    if long["value"].isna().any():
        raise ValueError(
            f"{source}: missing / non numeric values in {sorted(long.loc[long['value'].isna(), 'variable'].unique())}"
        )
    return long


def clean_presenze_alb(df, year, source="presenze_alb"):
    """ISPAT alb table (months x APT): only the `<APT> - Totale` columns are kept (`- Italiani`,
    `- Stranieri` and the `Provincia - ...` columns are ignored). -> (AREA = APT, DATA, value).
    """
    cols = [
        c
        for c in df.columns[1:]
        if str(c).endswith(" - Totale") and not str(c).startswith("Provincia - ")
    ]
    long = _monthly_long(df[[df.columns[0], *cols]], year, source)
    long["AREA"] = long.pop("variable").str.removesuffix(" - Totale").str.strip()
    return long[["AREA", "DATA", "value"]]


def clean_presenze_exalb(df, year, source="presenze_extralb"):
    """ISPAT extra alb table (months x type of structure): only the first column and the last one
    (the provincial total, `Totale - Totale`) are kept. -> (AREA = "PROVINCIA", DATA, value).
    """
    long = _monthly_long(df.iloc[:, [0, -1]], year, source)
    long["AREA"] = "PROVINCIA"
    return long[["AREA", "DATA", "value"]]


def presenze_files(raw_dir, tag, years):
    """Convention of the file names (`files.presenze.pattern` of settings.yaml, e.g. `<measure>_<tag>_<year>.csv`):
    {(measure, year): raw_dir/<file>} (existing files)."""
    raw_dir = Path(raw_dir)
    files = {}
    for y in years:
        for m in PRESENZE_MEASURES:
            f = raw_dir / PRESENZE_FILE_PATTERN.format(measure=m, tag=tag, year=y)
            if f.exists():
                files[(m, int(y))] = f
    return files


def process_presenze_stay(
    files,
    tag,
    years,
    area_ids,
    beds,
    *,
    bed_col,
    out_col,
    spill_over=PRESENZE_SPILL_OVER,
    window=BASE_WINDOW,
):
    """ISPAT arrivals + presences (month x area) of `years` -> comune x day presences (see stay_distribution.py).

    files    {(measure, year): path} with measure in ("arrivi", "presenze"); every year needs both.
             The files of the year before `years` (if present) give the stays that cross the new year.
    tag      "alb" (areas = APT, with the `- Totale` columns) or "exalb" (province).
    area_ids {area: [ids]}; beds: (DATA, ID_COMUNE, bed_col) weights of the split among comuni.
    """
    years = sorted(int(y) for y in years)
    context = years[0] - 1
    has_context = spill_over and all((m, context) in files for m in PRESENZE_MEASURES)
    use = [context, *years] if has_context else years
    if spill_over and not has_context:
        logger.warning(
            "[%s] no files of %s: the stays that cross the beginning of %s are approximated",
            out_col,
            context,
            years[0],
        )
    clean = clean_presenze_alb if tag == "alb" else clean_presenze_exalb
    tables = {m: [] for m in PRESENZE_MEASURES}
    for y in use:
        for m in PRESENZE_MEASURES:
            if (m, y) not in files:
                raise FileNotFoundError(
                    f"missing file for {m} {tag} {y}: run the download step"
                )
            raw = pd.read_csv(files[(m, y)])
            tables[m].append(clean(raw, y, source=f"{m}_{tag}_{y}"))
    arrivals, presences = (
        pd.concat(tables[m], ignore_index=True) for m in PRESENZE_MEASURES
    )

    names = set(arrivals["AREA"])
    if names != set(area_ids):
        raise ValueError(
            f"[{out_col}] areas of the files and of the mapping differ: "
            f"only in files {sorted(names - set(area_ids))}, only in mapping {sorted(set(area_ids) - names)}"
        )
    out = distribute_by_stay(
        arrivals,
        presences,
        area_ids,
        beds,
        bed_col=bed_col,
        out_col=out_col,
        keep_from=pd.Timestamp(years[0], 1, 1),
        spill_over=spill_over,
    )
    return standard_ordering_cols(clip(out, window, name=out_col))


def presenze_alb_area_ids(mapping_apt):
    return {k: [int(i) for i in v] for k, v in mapping_apt.items()}


def presenze_exalb_area_ids(mapping_comuni):
    return {"PROVINCIA": sorted({int(i) for i in mapping_comuni.values()})}


def load_mappings(mapping_dir):
    mapping_dir = Path(mapping_dir)
    return {key: read_json(mapping_dir / fname) for key, fname in MAPPING_FILES.items()}


def process_data(
    normalized_dir=NORMALIZED_DIR,
    mapping_dir=MAPPING_DIR,
    out_dir=PROCESSED_DIR,
    type_format=TYPE_FORMAT,
    raw_dir=RAW_DIR,
    window=BASE_WINDOW,
):
    """Base build: every dataset is restricted to `window` (default: period.start .. period.base_end of
    settings.yaml), so nothing before 2022-01-01 and nothing after 2024-12-31 reaches the processed data."""
    normalized_dir = Path(normalized_dir)
    logger.info("Time window of the base build: %s", window)
    ensure_travel_time_matrix()

    logger.info("Reading standardized data from %s", normalized_dir)
    std = {
        name: read_df(normalized_dir, spec.std_name, type_format)
        for name, spec in DATASETS.items()
        if spec.normalized
    }
    for name in ("ee_prelievi", "gas_prelievi", "ee_totale", "gas_totale", "vodafone_covisits"):  # big
        std[name] = clip(std[name], window, name=name)
    maps = load_mappings(mapping_dir)
    mapping_comuni, mapping_vodafone, mapping_apt = (
        maps["mapping_comuni"],
        maps["mapping_vodafone"],
        maps["mapping_apt"],
    )

    # Order: strutture -> vodafone (areas with several comuni split by beds) -> alb -> extralb.
    # alb / extralb (ISPAT arrivals + presences, read from raw_data) are distributed taking the average
    # stay into account, and split among comuni by `alberghieri` / `extra alb.` beds.
    strutture_pr = process_strutture(std["strutture"], mapping_comuni, window)
    popolazione_pr = process_popolazione(std["popolazione"], mapping_comuni, window)
    vodafone_pr = process_vodafone(
        std["vodafone"],
        mapping_vodafone,
        strutture_pr,
        popolazione=popolazione_pr,
        window=window,
    )
    beds = process_beds(
        read_strutture_std(normalized_dir, type_format), mapping_comuni, window
    )
    years = range(window.start.year, window.end.year + 1)
    alb_pr = process_presenze_stay(
        presenze_files(raw_dir, PRESENZE_TAGS["presenze_alb"], years),
        "alb",
        years,
        presenze_alb_area_ids(mapping_apt),
        beds,
        bed_col=BEDS_ALB_COL,
        out_col=PRESENZE_ALB_VALUE_COLS[0],
        window=window,
    )
    extralb_pr = process_presenze_stay(
        presenze_files(raw_dir, PRESENZE_TAGS["presenze_extralb"], years),
        "exalb",
        years,
        presenze_exalb_area_ids(mapping_comuni),
        beds,
        bed_col=BEDS_XALB_COL,
        out_col=PRESENZE_XALB_VALUE_COLS[0],
        window=window,
    )
    ee_pr = process_energy(
        std["ee_prelievi"],
        mapping_comuni,
        EE_PRELIEVI_VALUE_COLS[0],
        "ee_prelievi",
        window,
    )
    gas_pr = process_energy(
        std["gas_prelievi"],
        mapping_comuni,
        GAS_PRELIEVI_VALUE_COLS[0],
        "gas_prelievi",
        window,
    )
    ee_total_pr = process_energy(
        std["ee_totale"], mapping_comuni, EE_TOTALE_VALUE_COLS[0], "ee_totale", window
    )
    gas_total_pr = process_energy(
        std["gas_totale"], mapping_comuni, GAS_TOTALE_VALUE_COLS[0], "gas_totale", window
    )
    meteo_pr = process_meteo(meteo_files(raw_dir, years), window)
    area_comune = covisits_area_comune(mapping_dir, mapping_vodafone, mapping_comuni)
    covisits_pr = process_covisits(std["vodafone_covisits"], area_comune, window)
    provenienza_pr = process_provenienza(std["vodafone"], area_comune, window)
    arrivi_pr = process_provenienza_arrivi(
        std["vodafone_arrivi"], area_comune, strutture_pr, window
    )
    dict_processed = {
        DATASETS["popolazione"].processed_name: popolazione_pr,
        DATASETS["strutture"].processed_name: strutture_pr,
        DATASETS["vodafone"].processed_name: vodafone_pr,
        DATASETS["presenze_alb"].processed_name: alb_pr,
        DATASETS["presenze_extralb"].processed_name: extralb_pr,
        DATASETS["ee_prelievi"].processed_name: ee_pr,
        DATASETS["gas_prelievi"].processed_name: gas_pr,
        DATASETS["ee_totale"].processed_name: ee_total_pr,
        DATASETS["gas_totale"].processed_name: gas_total_pr,
        DATASETS["meteo"].processed_name: meteo_pr,
        DATASETS["vodafone_covisits"].processed_name: covisits_pr,
        DATASETS["vodafone_provenienza"].processed_name: provenienza_pr,
        DATASETS["vodafone_arrivi"].processed_name: arrivi_pr,
    }

    save_computed_dfs(
        dict_processed, local=True, type_format=type_format, path_saving=out_dir
    )
    logger.info("Processed data saved in %s", out_dir)
    return dict_processed


if __name__ == "__main__":
    setup_logging()
    process_data()
