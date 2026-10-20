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
_PLATFORM_KEYS = ("project", "s3_bucket", "s3_prefix", "dhcore_env")
_REFERENCE_KEYS = ("mapping_comuni", "mapping_vodafone", "mapping_apt", "geojson")
_PERIOD_KEYS = ("start", "base_end", "update_start", "update_end")
_DATASET_FILE_KEYS = ("std", "processed", "update")


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
                ),
            ),
            ("paths", _PATH_KEYS),
            ("period", _PERIOD_KEYS),
            ("platform", _PLATFORM_KEYS),
            ("references", _REFERENCE_KEYS),
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
    for phenomenon_name, deps in (settings.get("phenomena") or {}).items():
        if not isinstance(deps, list):
            raise TypeError(
                f"{path}: phenomena.{phenomenon_name} must be a list of dataset names"
            )
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

# platform (DigitalHub) / data lake
PROJECT = _S["platform"]["project"]
S3_BUCKET = _S["platform"]["s3_bucket"]
S3_PREFIX = _S["platform"]["s3_prefix"]
S3_ENV = _S["platform"]["dhcore_env"]

_LOCAL = (_S.get("update") or {}).get("local_source_dir")
LOCAL_SOURCE_DIR = _resolve(PACKAGE_DIR, _LOCAL) if _LOCAL else None


MAPPING_FILES = {
    "mapping_comuni": "mapping_comuni_ISTAT.json",
    "mapping_vodafone": "mapping_comuni_into_vodafone_Trento.json",
    "mapping_apt": "map_comuni_into_apt.json",
}


def setup_logging(level=logging.INFO):
    """Call from entry points only (libraries must not configure logging)."""
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
