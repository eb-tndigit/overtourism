# SPDX-License-Identifier: Apache-2.0
"""Local read/write of dataframes and JSON. No network / platform dependency at import time."""

import json
import logging
from pathlib import Path

import pandas as pd

from utils.cleaning import normalize_id_comune
from utils.config import ENERGY_ROOT, METEO_FOLDER, TYPE_FORMATS

logger = logging.getLogger(__name__)


def ensure_dir(path) -> Path:
    """Creates `path` (and parents) if needed and returns it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_path(key, raw_dir) -> Path:
    """Where a data-lake object `key` lives in raw_data: flat (its base name), except
      - the energy files, which keep their folder (`<energy root>/ee_prelievi/49_3.csv` -> raw_data/ee_prelievi/49_3.csv):
        electricity and gas have identically named category files. The energy root is `files.energy.root` of settings.yaml;
      - the weather files, which go to their own folder (`<meteo folder>/meteo_trentino_comuni_2024.csv` ->
        raw_data/meteo/meteo_trentino_comuni_2024.csv). The folder is `files.meteo.folder` of settings.yaml."""
    key = Path(key)
    sub = key.parent.name if key.parts[0] in (ENERGY_ROOT, METEO_FOLDER) else ""
    return Path(raw_dir) / sub / key.name


def read_df(path, name, type_format="csv", parse_ids=False):
    """Reads path/name in type_format format. With parse_ids=True, ID_COMUNE is
    brought back to its canonical form (lists are serialized as strings in csv)."""
    file = Path(path) / f"{name}.{type_format}"
    if not file.exists():
        raise FileNotFoundError(
            f"{file} not found: run the previous step (download raw data)"
        )
    if type_format == "csv":
        df = pd.read_csv(file, dtype={"ID_COMUNE": str})
    else:
        df = pd.read_parquet(file)
    if parse_ids and "ID_COMUNE" in df.columns:
        df["ID_COMUNE"] = df["ID_COMUNE"].apply(normalize_id_comune)
    return df


def read_table(path) -> pd.DataFrame:
    """Reads a raw table (csv or parquet) choosing the reader from the file extension, so the file names
    of settings.yaml are used exactly as written."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the previous step (download raw data)")
    match path.suffix.lower():
        case ".parquet":
            return pd.read_parquet(path)
        case ".csv":
            return pd.read_csv(path)
        case other:
            raise NotImplementedError(f"read_table: unsupported extension {other!r} ({path})")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def put_dataframe(df: pd.DataFrame, name: str, path, type: str = "parquet") -> str:
    """Saves a dataframe as <path>/<name>.<type> and returns the file path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    path = path / name
    match type:
        case "json":
            path = path.with_suffix(".json")
            df.to_json(path, orient="index", indent=4)
        case "csv":
            path = path.with_suffix(".csv")
            df.to_csv(path, index=False)
        case "parquet":
            path = path.with_suffix(".parquet")
            df.to_parquet(path, index=False)
        case _:
            raise NotImplementedError(f"Unsupported type: {type}")
    return str(path)


def save_computed_dfs(dict_dfs, local=False, type_format="parquet", path_saving=None):
    """Writes every dataframe in `path_saving`; with local=False it also logs it to the platform.

    `path_saving` is mandatory: there is deliberately no default location.
    """
    if path_saving is None:
        raise ValueError("path_saving is required")
    if type_format not in TYPE_FORMATS:
        raise ValueError(
            f"type_format must be one of {TYPE_FORMATS}, got {type_format!r}"
        )
    for key, value in dict_dfs.items():
        logger.info(
            "Saving dataframe '%s' in %s/%s.%s...", key, path_saving, key, type_format
        )
        file_path = put_dataframe(value, key, path=path_saving, type=type_format)
        if not local:
            from utils.remote import log_dataframe

            logger.info(
                "Logging dataframe '%s.%s' to the platform...", key, type_format
            )
            log_dataframe(file_path, key, type=type_format)
    logger.info("## Saved.")
