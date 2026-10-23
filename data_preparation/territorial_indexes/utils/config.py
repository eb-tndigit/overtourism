# SPDX-License-Identifier: Apache-2.0
"""Loads config/settings.yaml and exposes it as module constants (the file is the only place to edit)."""

import logging
from pathlib import Path

import pandas as pd
import yaml

PACKAGE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_DIR / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.yaml"

TYPE_FORMATS = ("csv", "parquet")

_PATH_KEYS = (
    "mapping",
    "raw",
    "normalized",
    "processed",
    "final",
)
_PLATFORM_KEYS = ("project", "s3_bucket", "s3_prefix", "dhcore_env", "credentials_file")
_REFERENCE_KEYS = ("mapping_comuni", "mapping_vodafone", "mapping_apt", "geojson")
_PERIOD_KEYS = ("start", "base_end", "update_start", "update_end")
_DATASET_FILE_KEYS = ("std", "processed", "update")
_NAMING_KEYS = ("std_suffix", "update_part_suffix")
_FILES_KEYS = (
    "popolazione",
    "strutture_historical",
    "strutture_2024",
    "vodafone_presenze_old",
    "vodafone_presenze_new",
    "vodafone_arrivi",
    "vodafone_covisits",
    "presenze",
    "meteo",
    "energy",
)
_PRESENZE_FILES_KEYS = ("folder", "pattern", "measures", "tags")
_METEO_FILES_KEYS = ("folder", "pattern")
_ENERGY_FILES_KEYS = ("root", "folders", "category_files", "category_glob", "totals")
# references that are json mappings (the other one, the geojson, is not read by `load_mappings`)
_MAPPING_REFERENCE_KEYS = ("mapping_comuni", "mapping_vodafone", "mapping_apt")


def load_settings(path=SETTINGS_FILE) -> dict:
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    missing = [
        f"{section}.{key}" if section else key
        for section, keys in (
            (
                "",
                (
                    "output_dir",
                    "type_format",
                    "paths",
                    "period",
                    "platform",
                    "references",
                    "datasets",
                    "phenomena",
                    "derived_phenomena",
                    "naming",
                    "files",
                ),
            ),
            ("paths", _PATH_KEYS),
            ("period", _PERIOD_KEYS),
            ("platform", _PLATFORM_KEYS),
            ("references", _REFERENCE_KEYS),
            ("naming", _NAMING_KEYS),
            ("files", _FILES_KEYS),
        )
        for key in keys
        if key not in (settings.get(section, {}) if section else settings)
    ]
    for dataset_name, dataset_cfg in (settings.get("datasets") or {}).items():
        if not isinstance(dataset_cfg, dict):
            raise TypeError(f"{path}: datasets.{dataset_name} must be a mapping")
        for key in _DATASET_FILE_KEYS:
            if key not in dataset_cfg:
                missing.append(f"datasets.{dataset_name}.{key}")
    files = settings.get("files") or {}
    for section, keys in (
        ("presenze", _PRESENZE_FILES_KEYS),
        ("meteo", _METEO_FILES_KEYS),
        ("energy", _ENERGY_FILES_KEYS),
    ):
        body = files.get(section)
        if not isinstance(body, dict):
            continue  # already reported as missing above
        missing += [f"files.{section}.{key}" for key in keys if key not in body]
    update_cfg = settings.get("update") or {}
    if "default_config" not in update_cfg:
        missing.append("update.default_config")
    for phenomenon_name, deps in (settings.get("phenomena") or {}).items():
        if not isinstance(deps, list):
            raise TypeError(
                f"{path}: phenomena.{phenomenon_name} must be a list of dataset names"
            )
    for phenomenon_name, deps in (settings.get("derived_phenomena") or {}).items():
        if not isinstance(deps, list):
            raise TypeError(
                f"{path}: derived_phenomena.{phenomenon_name} must be a list of phenomenon names"
            )
    if "hdd_base_temp" not in (settings.get("meteo") or {}):
        missing.append("meteo.hdd_base_temp")
    if missing:
        raise KeyError(f"{path}: missing setting(s) {missing}")
    if settings["type_format"] not in TYPE_FORMATS:
        raise ValueError(f"{path}: type_format must be one of {TYPE_FORMATS}")
    period = {k: pd.Timestamp(str(v)) for k, v in settings["period"].items()}
    if not (
        period["start"] <= period["base_end"] < period["update_start"] <= period["update_end"]
    ):
        raise ValueError(
            f"{path}: period must satisfy start <= base_end < update_start <= update_end, got {settings['period']}"
        )
    return settings


def _resolve(base: Path, value) -> Path:
    p = Path(value).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


_S = load_settings()

OUTPUT_DIR = _resolve(PACKAGE_DIR, _S["output_dir"])
_P = {k: _resolve(OUTPUT_DIR, _S["paths"][k]) for k in _PATH_KEYS}

# reference mappings, shared by base build and update
MAPPING_DIR = _P["mapping"]
# base build
RAW_DIR = _P["raw"]
NORMALIZED_DIR = _P["normalized"]
PROCESSED_DIR = _P["processed"]
FINAL_DIR = _P["final"]
REFERENCES = _S["references"]

# time window (utils/window.py): base build = [start, base_end], update = [update_start, update_end]
from utils.window import Window  # noqa: E402

_PER = {k: pd.Timestamp(str(v)) for k, v in _S["period"].items()}
DATA_START = _PER["start"]
BASE_WINDOW = Window(_PER["start"], _PER["base_end"])
UPDATE_WINDOW = Window(_PER["update_start"], _PER["update_end"])

TYPE_FORMAT = _S["type_format"]
TOP_N_PROVENIENZA_COUNTRIES = (_S.get("provenienza") or {}).get("top_n_countries", 3)
if (
    isinstance(TOP_N_PROVENIENZA_COUNTRIES, bool)
    or not isinstance(TOP_N_PROVENIENZA_COUNTRIES, int)
    or TOP_N_PROVENIENZA_COUNTRIES < 1
):
    raise ValueError(
        f"{SETTINGS_FILE}: provenienza.top_n_countries must be a positive integer"
    )
DATASET_FILES = _S["datasets"]
PHENOMENA_FILES = _S["phenomena"]
TRAVEL_TIMES = _S.get("travel_times") or {}
TRAVEL_TIMES_MATRIX_FILE = _resolve(
    PACKAGE_DIR,
    TRAVEL_TIMES.get("matrix_file", "../Output/travel_times/matrice_tempi_verso_trentino.csv"),
)
REFERENCE_CITY_ITALY = str(TRAVEL_TIMES.get("reference_city_italy", "Milano")).strip()  # default: Milano 
if not REFERENCE_CITY_ITALY:
    raise ValueError(f"{SETTINGS_FILE}: travel_times.reference_city_italy must not be empty")
DERIVED_PHENOMENA_FILES = _S["derived_phenomena"]
HDD_BASE_TEMP = float(_S["meteo"]["hdd_base_temp"])

# platform (DigitalHub) / data lake
PROJECT = _S["platform"]["project"]
S3_BUCKET = _S["platform"]["s3_bucket"]
S3_PREFIX = _S["platform"]["s3_prefix"]
S3_ENV = _S["platform"]["dhcore_env"]

CREDENTIALS_FILE = _S["platform"]["credentials_file"]

_LOCAL = (_S.get("update") or {}).get("local_source_dir")
LOCAL_SOURCE_DIR = _resolve(PACKAGE_DIR, _LOCAL) if _LOCAL else None
DEFAULT_UPDATE_CONFIG = _resolve(PACKAGE_DIR, _S["update"]["default_config"])

# ---- file names (everything comes from settings.yaml: no file name is written in the code)
# json mappings: reference key -> file name in MAPPING_DIR (derived from `references`)
MAPPING_FILES = {k: Path(REFERENCES[k]).name for k in _MAPPING_REFERENCE_KEYS}

STD_SUFFIX = _S["naming"]["std_suffix"]
UPDATE_PART_SUFFIX = _S["naming"]["update_part_suffix"]

_FILES = _S["files"]
# raw base-build objects (data-lake keys)
POPOLAZIONE_RAW_FILE = _FILES["popolazione"]
STRUTTURE_HISTORICAL_RAW_FILE = _FILES["strutture_historical"]
STRUTTURE_2024_RAW_FILE = _FILES["strutture_2024"]
VODAFONE_OLD_RAW_FILE = _FILES["vodafone_presenze_old"]
VODAFONE_NEW_RAW_FILE = _FILES["vodafone_presenze_new"]
ARRIVI_RAW_FILE = _FILES["vodafone_arrivi"]
COVISITS_RAW_FILE = _FILES["vodafone_covisits"]
# ISPAT arrivals / presences: <PRESENZE_FOLDER>/<PRESENZE_FILE_PATTERN.format(measure, tag, year)>
PRESENZE_FOLDER = _FILES["presenze"]["folder"]
PRESENZE_FILE_PATTERN = _FILES["presenze"]["pattern"]
PRESENZE_MEASURES = tuple(_FILES["presenze"]["measures"])
PRESENZE_TAGS = dict(_FILES["presenze"]["tags"])  # dataset -> tag in the file names
# weather: <METEO_FOLDER>/<METEO_FILE_PATTERN.format(year)>
METEO_FOLDER = _FILES["meteo"]["folder"]
METEO_FILE_PATTERN = _FILES["meteo"]["pattern"]
# energy
ENERGY_ROOT = _FILES["energy"]["root"]
ENERGY_FOLDERS = dict(_FILES["energy"]["folders"])  # dataset -> sub-folder
ENERGY_CATEGORY_FILES = tuple(_FILES["energy"]["category_files"])
ENERGY_CATEGORY_GLOB = _FILES["energy"]["category_glob"]
ENERGY_TOTAL_FILES = dict(_FILES["energy"]["totals"])  # dataset -> data-lake key


def setup_logging(level=logging.INFO):
    """Call from entry points only (libraries must not configure logging)."""
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
