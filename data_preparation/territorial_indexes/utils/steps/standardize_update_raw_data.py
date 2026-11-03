# SPDX-License-Identifier: Apache-2.0
"""
STEP 1 (update) - Standardization of the update raw data.

Input : Output/data/raw_data/   (files downloaded by download_update_raw_data)
Output: Output/data/normalized/ (<dataset>_update_std, or <dataset>_<label>_update_std; suffixes in settings.yaml `naming`)

For every source of the update config:
  1. read the raw file
  2. CHECK that its column structure is compatible with the expected one (the layout of the base
     raw data of its kind, see KIND_SCHEMAS in utils/adapters/base.py) and RENAME the columns to the
     standard names: this is what the adapter of the source does, a mismatch stops the run
  3. apply the SAME standardization of the base build (standardize_raw_data.py): names, DATA/LOCATION,
     dates, comune format. No filtering and no aggregation
  4. save the result with the standard name `<part>_std`: same format as the base normalized files
"""

import logging
from pathlib import Path

import pandas as pd

from utils.adapters import (
    KIND_DATASETS,
    get_adapter,
    validate_schema,
)
from utils.config import (
    MAPPING_DIR,
    NORMALIZED_DIR,
    RAW_DIR,
    REFERENCES,
    STD_SUFFIX,
    TYPE_FORMAT,
    setup_logging,
)
from utils.datasets import ARRIVI_VALUE_COL
from utils.io import ensure_dir, save_computed_dfs
from utils.steps.standardize_raw_data import (
    standardize_covisits_columns,
    standardize_popolazione_columns,
    standardize_strutture_columns,
    standardize_vodafone_columns,
)
from utils.update.sources import read_source
from utils.update.spec import config_from_argv, load_config

logger = logging.getLogger(__name__)

# kind -> standardization (same functions of the base build)
STANDARDIZERS = {
    "popolazione": lambda df, geojson: standardize_popolazione_columns(df),
    "strutture": lambda df, geojson: standardize_strutture_columns(df),
    "vodafone": lambda df, geojson: standardize_vodafone_columns(df, geojson),
    "vodafone_covisits": lambda df, geojson: standardize_covisits_columns(df),
    # energy: the reader (energy_csv) already gives the standard layout of the base normalized files
    "energia": lambda df, geojson: df,
}
# datasets of a kind with a different standardization (same raw layout as the kind)
DATASET_STANDARDIZERS = {
    # arrivals: same layout as the presences, the value column is `arrivi`
    "vodafone_arrivi": lambda df, geojson: standardize_vodafone_columns(
        df, geojson, value_name=ARRIVI_VALUE_COL
    ),
}
# arrivals x travel time: the same arrivals file, standardized in the same way
DATASET_STANDARDIZERS["vodafone_tempi"] = DATASET_STANDARDIZERS["vodafone_arrivi"]
# kinds used as downloaded (no normalization): ISPAT arrivals / presences, weather
RAW_KINDS = {"presenze_raw", "meteo_raw"}
assert set(STANDARDIZERS) | RAW_KINDS == set(KIND_DATASETS)


def check_and_rename_columns(raw, src, strict=True) -> tuple:
    """Checks that the raw source has the expected column structure and renames its columns to the
    standard names (adapter of the source). Returns (aligned dataframe, kind).

    The adapter raises if the layout does not match; then the aligned frame is validated against the
    columns required by its kind (strict=False: warning instead of error)."""
    adapter = get_adapter(src.adapter)
    aligned = adapter.func(raw, **src.adapter_kwargs)
    validate_schema(aligned, adapter.kind, adapter.name, strict=strict)
    return aligned, adapter.kind


def standardize_update_raw_data(
    config,
    raw_dir=RAW_DIR,
    mapping_dir=MAPPING_DIR,
    out_dir=NORMALIZED_DIR,
    type_format=TYPE_FORMAT,
    strict=True,
):
    """Standardizes the sources of the datasets enabled in the config; returns {<part>_std: dataframe}.

    Sources of the same part (see SourceSpec.part_name) are stacked; a source with a `label` is kept
    in its own part."""
    config = load_config(config)
    raw_dir, mapping_dir = Path(raw_dir), Path(mapping_dir)

    geojson = None
    if {"vodafone", "vodafone_provenienza", "vodafone_arrivi", "vodafone_tempi"} & set(config.datasets):
        from utils.readers import read_geojson

        geojson = read_geojson(mapping_dir / Path(REFERENCES["geojson"]).name)

    frames = {}
    for dataset in config.datasets:
        for src in config.sources[dataset]:
            logger.info("[%s] %s with adapter '%s'", dataset, src.file, src.adapter)
            raw = read_source(src.file, src.reader, src.reader_kwargs, raw_dir=raw_dir)
            aligned, kind = check_and_rename_columns(raw, src, strict=strict)
            if kind in RAW_KINDS:  # layout checked, nothing to normalize
                logger.info(
                    "[%s] %s is used as downloaded (no normalization)",
                    dataset,
                    src.file,
                )
                continue
            std = (DATASET_STANDARDIZERS.get(dataset) or STANDARDIZERS[kind])(
                aligned, geojson
            )
            frames.setdefault(src.part_name(dataset), []).append(std)

    dict_std = {
        f"{name}{STD_SUFFIX}": (
            parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
        )
        for name, parts in frames.items()
    }
    save_computed_dfs(
        dict_std,
        local=True,
        type_format=type_format,
        path_saving=ensure_dir(out_dir),
    )
    logger.info("Standardized update data saved in %s", out_dir)
    return dict_std


if __name__ == "__main__":
    setup_logging()
    standardize_update_raw_data(config_from_argv("Standardize the update raw data."))
