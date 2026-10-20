# SPDX-License-Identifier: Apache-2.0
"""
STEP 1 - Standardization.

Input : Output/data/raw_data/
Output: Output/data/normalized/   (popolazione_std, strutture_std, vodafone_std, vodafone_arrivi_std,
                                   ee_prelievi_std, gas_prelievi_std, vodafone_covisits_std)

The ISPAT presences (alb / extralb arrivals and presences) are NOT normalized: they are processed from raw_data.

Uniformation of the raw data (names, DATA/LOCATION, dates and comune format).
No filtering and no aggregation, with one exception: the Vodafone OD co-visits file is too big to be
kept whole in memory, so it is read in chunks and only the rows used by the flow indicators are kept
(area level, TOURIST / VISITOR, total over the countries).
"""

import io
import logging
from pathlib import Path

import pandas as pd

from utils.adapters import RawData, get_adapter, validate_schema
from utils.cleaning import (
    customize_unidecode,
    standard_ordering_cols,
    to_data_location,
)
from utils.config import (
    NORMALIZED_DIR,
    RAW_DIR,
    MAPPING_DIR,
    REFERENCES,
    TYPE_FORMAT,
    setup_logging,
)
from utils.io import raw_path, read_df, save_computed_dfs
from utils.datasets import (
    ARRIVI_RAW_FILE,
    ARRIVI_VALUE_COL,
    COVISITS_LOC_TYPE,
    COVISITS_PROFILES,
    COVISITS_RAW_FILE,
    COVISITS_TOTAL_ROWS_ONLY,
    DATASETS,
    ENERGY_TOTAL_FILES,
)

logger = logging.getLogger(__name__)

ENERGY_TOTAL_CODE = "TOTALE"  # codice_utilizzo given to the rows of the files without usage categories

ENERGY_MONTHS = (
    "gennaio",
    "febbraio",
    "marzo",
    "aprile",
    "maggio",
    "giugno",
    "luglio",
    "agosto",
    "settembre",
    "ottobre",
    "novembre",
    "dicembre",
)


def convert_vodafone_comuni(df, geojson_comuni_json_data):
    """Conversion from locId (geojson) to comune name"""
    location_map = (
        geojson_comuni_json_data.set_index("id")["name"].str.upper().to_dict()
    )
    return df["locId"].map(location_map)


def _standardize_columns(df, date_col="anno", df_name=None):
    """Basic standardization: comune/data schema -> DATA/LOCATION."""
    logger.info(
        "Applying standardization to data%s", f" '{df_name}'" if df_name else ""
    )
    return to_data_location(df, date_col=date_col)


def _standardize_comune_yearly(df, comune_col, date_col, df_name):
    """Comune names cleaned + DATA/LOCATION schema (popolazione, strutture: municipality, yearly)."""
    df[comune_col] = df[comune_col].apply(customize_unidecode)
    return standard_ordering_cols(
        _standardize_columns(df, date_col=date_col, df_name=df_name)
    )


def standardize_popolazione_columns(
    df, comune_col="comune", date_col="anno"
) -> pd.DataFrame:
    """popolazione: granularity municipality, yearly"""
    return _standardize_comune_yearly(df, comune_col, date_col, "popolazione_df")


def standardize_strutture_columns(
    df, comune_col="comune", date_col="anno"
) -> pd.DataFrame:
    """strutture: granularity municipality, yearly"""
    return _standardize_comune_yearly(df, comune_col, date_col, "strutture_df")


def read_base_strutture(raw_dir):
    """Combine the historical CSV with the dedicated 2024 ODS source."""
    raw_dir = Path(raw_dir)
    historical = pd.read_csv(raw_dir / "Annuario-TavXIII-per-comune-csv.csv")
    source = pd.read_excel(raw_dir / "strutture_annuario_2024.ods", engine="odf")
    adapter = get_adapter("strutture_annuario_2024")
    strutture_2024 = adapter.func(RawData(source), year=2024)
    validate_schema(strutture_2024, adapter.kind, adapter.name)
    return pd.concat([historical, strutture_2024], ignore_index=True)


def standardize_vodafone_columns(
    df,
    geojson_comuni_json_data,
    comune_col="comune",
    date_col="date",
    value_name="presenze",
):
    """vodafone presences (`value_name="presenze"`) or arrivals (`value_name="arrivi"`): granularity vodafone
    areas, daily. Same layout and same standardization, only the name of the value column changes."""
    df[comune_col] = convert_vodafone_comuni(df, geojson_comuni_json_data)
    logger.info(
        "Unification of Vigo di Fassa and Pozza di Fassa in Vodafone dataset (ID 22250)"
    )
    mask = df[comune_col].isin(["VIGO DI FASSA", "POZZA DI FASSA"])
    df.loc[mask, comune_col] = "SAN GIOVANNI DI FASSA"
    df = _standardize_columns(df, date_col=date_col, df_name="vodafone_df")
    df.rename(columns={"value": value_name}, inplace=True)
    return standard_ordering_cols(df)


def read_base_vodafone(raw_dir, type_format=TYPE_FORMAT) -> pd.DataFrame:
    """Vodafone presences of the base build: `vodafone_attendences` (2022-01-01 .. 2024-03-31) completed with
    `vodafone_attendences_new.csv` (2023-01-01 .. 2025-12-31), which covers the rest of the base window
    (2024-04-01 .. 2024-12-31). On the dates present in both files the NEWER file wins. The time window
    (up to 2024-12-31) is applied in step 2: the 2025 days of the newer file are left to the update."""
    raw_dir = Path(raw_dir)
    old = read_df(raw_dir, "vodafone_attendences", type_format)
    new_path = raw_dir / VODAFONE_NEW_FILE
    if not new_path.exists():
        logger.warning(
            "[vodafone] %s not found in %s: the presences end with %s",
            VODAFONE_NEW_FILE,
            raw_dir,
            old["date"].max(),
        )
        return old
    new = pd.read_csv(new_path)
    replaced = old["date"].astype(str).isin(set(new["date"].astype(str)))
    logger.info(
        "[vodafone] %d rows of vodafone_attendences replaced by %s (%s .. %s)",
        int(replaced.sum()),
        VODAFONE_NEW_FILE,
        new["date"].min(),
        new["date"].max(),
    )
    return pd.concat([old[~replaced], new], ignore_index=True)


def read_base_arrivals(raw_dir) -> pd.DataFrame:
    """Vodafone arrivals of the base build: the single file `vodafone_arrivals.csv` (same layout as
    `vodafone_attendences_new.csv`, but `value` counts arrivals). The time window is applied in step 2."""
    path = Path(raw_dir) / ARRIVI_RAW_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the download step")
    df = pd.read_csv(path)
    logger.info("[vodafone_arrivi] %s: %d rows (%s .. %s)", ARRIVI_RAW_FILE, len(df), df["date"].min(), df["date"].max())
    return df


def read_energy_category(path) -> pd.DataFrame:
    """Read a semicolon-delimited energy category file (a path or a bytes buffer) and repair the header
    variant in 96.csv."""
    path = path if isinstance(path, (str, Path)) else io.StringIO(path.read().decode("utf-8-sig"))
    headers = [
        str(column).strip().lstrip("\ufeff")
        for column in pd.read_csv(path, sep=";", nrows=0).columns
    ]
    if hasattr(path, "seek"):
        path.seek(0)
    n_fields = None
    if "codice_utilizzo" not in headers:
        n_fields = pd.read_csv(path, sep=";", header=None, skiprows=1, nrows=1).shape[1]
        if hasattr(path, "seek"):
            path.seek(0)
    if "codice_utilizzo" not in headers and n_fields == len(headers) + 1:
        # The 96.csv source rows contain the usage code, but its header omits that field.
        headers.insert(2, "codice_utilizzo")
        df = pd.read_csv(path, sep=";", header=None, skiprows=1, names=headers)
    elif "codice_utilizzo" not in headers:
        # "totale" files: no usage category, one row per comune and year
        df = pd.read_csv(path, sep=";")
        df.columns = headers
        df["codice_utilizzo"] = ENERGY_TOTAL_CODE
    else:
        df = pd.read_csv(path, sep=";")
        df.columns = headers

    required = {"anno", "comune", "codice_utilizzo"}
    missing = required - set(df.columns)
    month_columns = {
        str(column).split("[", maxsplit=1)[0].strip().lower(): column
        for column in df.columns
        if str(column).split("[", maxsplit=1)[0].strip().lower() in ENERGY_MONTHS
    }
    if missing or set(month_columns) != set(ENERGY_MONTHS):
        raise ValueError(
            f"{path}: invalid energy columns; missing {sorted(missing)}, "
            f"months found {sorted(month_columns)}"
        )

    df = df.rename(columns={column: month for month, column in month_columns.items()})
    df["DATA"] = pd.to_numeric(df.pop("anno"), errors="raise").astype(int)
    df["LOCATION"] = df.pop("comune").astype(str).apply(customize_unidecode)
    return df[["DATA", "LOCATION", "codice_utilizzo", *ENERGY_MONTHS]]


def standardize_energy_total(raw_dir, dataset) -> pd.DataFrame:
    """Normalizes the "totale" file of `dataset` (ee_totale / gas_totale): same layout as a category file."""
    path = raw_path(ENERGY_TOTAL_FILES[dataset], raw_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the download step")
    return read_energy_category(path)


def standardize_energy_folder(raw_dir, folder) -> pd.DataFrame:
    """Stack all category CSVs in one energy folder without aggregating their values."""
    folder_path = Path(raw_dir) / folder
    totals = {Path(k).name for k in ENERGY_TOTAL_FILES.values()}  # read apart (ee_totale / gas_totale)
    files = sorted(p for p in folder_path.glob("*.csv") if p.name not in totals)
    if not files:
        raise FileNotFoundError(f"No category CSV files found in {folder_path}")
    return pd.concat([read_energy_category(path) for path in files], ignore_index=True)


COVISITS_COLS = [
    "date",
    "locType",
    "sourceId",
    "sourceAmbito",
    "targetId",
    "targetAmbito",
    "covisitType",
    "userCountry",
    "userProfile",
    "value",
]
COVISITS_CHUNK_ROWS = 1_000_000
VODAFONE_NEW_FILE = "vodafone_attendences_new.csv"


def standardize_covisits_columns(df) -> pd.DataFrame:
    """Vodafone OD co-visits: area x area x day. Text columns cleaned (area ids without the '.0' of
    the floats, empty = ''), value numeric, `date` -> DATA. Only the rows used by the flow indicators
    are kept: area level, TOURIST / VISITOR and, if COVISITS_TOTAL_ROWS_ONLY, the total over the countries."""
    df = df[COVISITS_COLS].copy()
    for c in COVISITS_COLS[:-1]:
        df[c] = (
            df[c].astype(str).replace({"nan": "", "None": "", "<NA>": ""}).str.strip()
        )
    for c in ("sourceId", "targetId"):
        df[c] = df[c].str.replace(r"\.0$", "", regex=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)

    keep = (df["locType"] == COVISITS_LOC_TYPE) & df["userProfile"].isin(
        COVISITS_PROFILES
    )
    if COVISITS_TOTAL_ROWS_ONLY:
        keep &= df["userCountry"] == ""
    df = df[keep].drop(columns=["locType", "userCountry"])
    return df.rename(columns={"date": "DATA"}).reset_index(drop=True)


def read_covisits_raw(path) -> pd.DataFrame:
    """Reads the raw OD co-visits (csv in chunks, or parquet) and standardizes it."""
    path = Path(path)
    if path.suffix == ".parquet":
        return standardize_covisits_columns(pd.read_parquet(path, columns=COVISITS_COLS))
    parts = [
        standardize_covisits_columns(chunk)
        for chunk in pd.read_csv(
            path,
            usecols=COVISITS_COLS,
            dtype=str,
            keep_default_na=False,
            chunksize=COVISITS_CHUNK_ROWS,
        )
    ]
    return pd.concat(parts, ignore_index=True)


def standardize_raw_data(
    raw_dir=RAW_DIR,
    mapping_dir=MAPPING_DIR,
    out_dir=NORMALIZED_DIR,
    type_format=TYPE_FORMAT,
):
    from utils.readers import read_geojson

    raw_dir = Path(raw_dir)
    mapping_dir = Path(MAPPING_DIR)

    logger.info("Reading raw data from %s", raw_dir)
    popolazione_df = read_df(raw_dir, "popolazione_2020_2024", type_format)
    strutture_df = read_base_strutture(raw_dir)
    vodafone_df = read_base_vodafone(raw_dir, type_format)
    geojson = read_geojson(mapping_dir / Path(REFERENCES["geojson"]).name)
    arrivals_df = read_base_arrivals(raw_dir)
    covisits_df = read_covisits_raw(raw_dir / COVISITS_RAW_FILE)

    dict_std = {
        "popolazione_std": standardize_popolazione_columns(popolazione_df),
        "strutture_std": standardize_strutture_columns(strutture_df),
        "vodafone_std": standardize_vodafone_columns(vodafone_df, geojson),
        DATASETS["vodafone_arrivi"].std_name: standardize_vodafone_columns(
            arrivals_df, geojson, value_name=ARRIVI_VALUE_COL
        ),
        DATASETS["ee_prelievi"].std_name: standardize_energy_folder(
            raw_dir, "ee_prelievi"
        ),
        DATASETS["gas_prelievi"].std_name: standardize_energy_folder(
            raw_dir, "gas_prelievi"
        ),
        DATASETS["ee_totale"].std_name: standardize_energy_total(raw_dir, "ee_totale"),
        DATASETS["gas_totale"].std_name: standardize_energy_total(raw_dir, "gas_totale"),
        DATASETS["vodafone_covisits"].std_name: covisits_df,
    }
    save_computed_dfs(
        dict_std, local=True, type_format=type_format, path_saving=out_dir
    )
    logger.info("Standardized data saved in %s", out_dir)
    return dict_std


if __name__ == "__main__":
    setup_logging()
    standardize_raw_data()
