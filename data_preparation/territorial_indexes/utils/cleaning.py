# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for cleaning names / ids / frames shared by every step."""

import ast
import logging

import numpy as np
import pandas as pd
from unidecode import unidecode

logger = logging.getLogger(__name__)

# Explicit overrides for comuni whose official Italian name differs from
# a naive "before the dash" split of the bilingual name in the source CSV.
COMUNE_NAME_OVERRIDES = {
    "CAMPITELLO DI FASSA-CIAMPEDEL": "CAMPITELLO DI FASSA",
    "CAMPODENNO": "CAMPODENNO",  # no dash present, check exact spelling/accents in mapping
    "CANAL SAN BOVO": "CANAL SAN BOVO",
    "CANAZEI-CIANACEI": "CANAZEI",
    "FIEROZZO-VLAROTZ": "FIEROZZO",
    "FRASSILONGO-GARAIT": "FRASSILONGO",
    "LUSERNA-LUSERN": "LUSERNA",
    "MAZZIN-MAZIN": "MAZZIN",
    "MOENA-MOENA": "MOENA",
    "PALU DEL FERSINA-PALAI EN BERSNTOL": "PALU DEL FERSINA",
    "PALU' DEL FERSINA": "PALU DEL FERSINA",
    "ROVERE' DELLA LUNA": "ROVERE DELLA LUNA",
    "RUFFRE'-MENDOLA": "RUFFRE-MENDOLA",
    "SAN MICHELE ALL' ADIGE": "SAN MICHELE ALL'ADIGE",
    "SANT' ORSOLA TERME": "SANT'ORSOLA TERME",
    "SORAGA": "SORAGA DI FASSA",
    "SAN GIOVANNI DI FASSA-SEN JAN": "SAN GIOVANNI DI FASSA",
    "SORAGA DI FASSA-SORAGA": "SORAGA DI FASSA",
}


def customize_unidecode(x):
    """Remove accents, upper-case and strip a comune name."""
    if x.endswith("'"):  # removes also trailing apostrophe if present
        x = x.removesuffix("'")
    return unidecode(x.strip().upper()).replace("0", "-")


def normalize_id_comune(x):
    """Canonical hashable form of ID_COMUNE: lists, tuples, arrays (parquet) and
    strings like "['022001', '022002']" (csv) all become a sorted tuple;
    scalars are left untouched."""
    if isinstance(x, str) and x.strip().startswith("["):
        x = ast.literal_eval(x)
    if isinstance(x, (list, tuple, np.ndarray)):
        return tuple(sorted(x))
    return x


def make_hashable(value):
    """Canonical representation used exclusively for deduplication."""
    value = normalize_id_comune(value)

    if isinstance(value, tuple):
        return tuple(str(x).zfill(6) for x in value)

    if pd.isna(value):
        return value

    return str(value).zfill(6)


def pad_id_comune(series, width=6):
    """Zero-pad an ID_COMUNE column to `width` digits (e.g. 22001 -> '022001').

    Missing / unmapped values (NaN) are left untouched. Works for scalars
    (int, float, str, NaN) and lists of IDs.
    """

    def _pad(x):
        if isinstance(x, list):
            return [_pad(i) for i in x]

        if pd.isna(x):
            return x
        return str(int(x)).zfill(width)

    return series.apply(_pad)


def ids_to_int(x):
    """[ '22001', ... ] -> [22001, ...]; anything that is not a list is returned unchanged."""
    return [int(i) for i in x] if isinstance(x, list) else x


def resolve_id_comune(name, mapping_comuni, overrides=COMUNE_NAME_OVERRIDES):
    """Map a comune name to its ISTAT ID, falling back to the bilingual-name overrides."""
    id_comune = mapping_comuni.get(name)
    if id_comune is None and name in overrides:
        id_comune = mapping_comuni.get(overrides[name])
    return id_comune


def remove_provincia(df, comune_col="comune", upper=False):
    """Drop rows whose comune starts with 'PROVINCIA', logging what gets removed."""
    df = df.copy()
    series = df[comune_col].str.upper() if upper else df[comune_col]
    mask = series.str.startswith("PROVINCIA")
    if mask.any():
        logger.info(
            "Comune %s is a PROVINCIA, removing it from the analysis",
            df.loc[mask, comune_col].unique(),
        )
        df = df[~mask]
    return df


def to_data_location(df, date_col, drop_cols=None):
    """Standardize a phenomenon dataframe to DATA/LOCATION column naming."""
    df = df.drop(columns=drop_cols) if drop_cols else df
    return df.rename(columns={date_col: "DATA", "comune": "LOCATION"})


def standard_ordering_cols(df):
    """DATA and ID_COMUNE first, then every other column in the original order."""
    existing_first = [col for col in ["DATA", "ID_COMUNE"] if col in df.columns]
    remaining = [col for col in df.columns if col not in ["DATA", "ID_COMUNE"]]
    return df[existing_first + remaining]


def remove_unnamed(df):
    """Flatten a 2-level header, dropping 'Unnamed: ...' levels.
    If the columns are already flat strings this is a no-op (returns a copy)."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df.copy()

    top = pd.Series([c[0] for c in df.columns])
    top = top.where(~top.astype(str).str.startswith("Unnamed"), pd.NA).ffill()
    bottom = pd.Series([c[1] for c in df.columns])

    df = df.copy()
    df.columns = [
        (
            str(t).strip()
            if str(b).startswith("Unnamed") or pd.isna(b)
            else f"{str(t).strip()} {str(b).strip()}"
        )
        for t, b in zip(top, bottom)
    ]
    return df


def grouped_presenze_columns(groups):
    """Flat column names of the ISPAT presences tables: Mese, '<group> Italiani/Stranieri/Totale', ..."""
    names = ["Mese"]
    for group in groups:
        names.extend([f"{group} Italiani", f"{group} Stranieri", f"{group} Totale"])
    return names


def concat_keep_last(frames, name="dataset", key=("DATA", "ID_COMUNE")):
    """Stacks dataframes; on a duplicated key (DATA + ID_COMUNE) the LAST frame wins.
    ID_COMUNE may hold lists, hence make_hashable. Result sorted by key, columns as in the first frame.
    """
    frames = list(frames)
    if len(frames) == 1 and not frames[0].duplicated(list(key)).any():
        return frames[0]
    merged = pd.concat(frames, ignore_index=True)
    merged["_ID_KEY"] = merged["ID_COMUNE"].map(make_hashable)
    n_before = len(merged)
    merged = (
        merged.drop_duplicates(subset=["DATA", "_ID_KEY"], keep="last")
        .sort_values(["DATA", "_ID_KEY"])
        .drop(columns="_ID_KEY")
        .reset_index(drop=True)
    )
    if n_before != len(merged):
        logger.info(
            "[%s] %d rows with a duplicated DATA + ID_COMUNE replaced by the last version",
            name,
            n_before - len(merged),
        )
    return standard_ordering_cols(merged)


def concat_replace_groups(frames, name="dataset"):
    """concat_keep_last for data with several rows per DATA + ID_COMUNE (e.g. one per origin market): where a key is in
    several frames, ALL its rows come from the LAST one (a row missing from it does not survive)."""
    merged = pd.concat([f.assign(_FRAME=i) for i, f in enumerate(frames)], ignore_index=True)
    merged["_ID_KEY"] = merged["ID_COMUNE"].map(make_hashable)
    last = merged.groupby(["DATA", "_ID_KEY"], dropna=False)["_FRAME"].transform("max")
    n_before = len(merged)
    merged = (
        merged[merged["_FRAME"] == last]
        .sort_values(["DATA", "_ID_KEY"], kind="stable")
        .drop(columns=["_FRAME", "_ID_KEY"])
        .reset_index(drop=True)
    )
    if n_before != len(merged):
        logger.info("[%s] %d rows of a DATA + ID_COMUNE replaced by the last version", name, n_before - len(merged))
    return standard_ordering_cols(merged)
