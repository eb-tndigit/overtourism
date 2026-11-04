# SPDX-License-Identifier: Apache-2.0
"""Molveno data-preparation pipeline (entry point).

    python -m model.molveno.prepare                    # steps 0-2, sources from molveno.yaml
    python -m model.molveno.prepare --skip-download    # steps 0-1 skipped: reuse normalized tables
    python -m model.molveno.prepare --refresh          # download the data-lake files again
    python -m model.molveno.prepare --local-sources DIR    # read the sources from DIR, not the data lake
    python -m model.molveno.prepare --upload           # also log the reference datasets to the platform

Run from the data_preparation folder (as the other entry points).

| Step | Module | Output |
|------|--------|--------|
| 0 Sources | model/common/sources.py | raw files (local copies or <raw>/molveno/) |
| 1 Standardize | standardize.py | <normalized>/molveno/*.parquet |
| 2 Reference datasets | reference_datasets.py | <final>/molveno/*.parquet |
| 3 Upload (optional) | model.common.remote.log_dataframe | platform dataitems |
"""

import argparse
import logging

import pandas as pd

from model.common.sources import resolve_sources
from model.molveno.config import MOLVENO_FINAL_DIR, MOLVENO_NORMALIZED_DIR, MOLVENO_RAW_DIR, load_config
from model.molveno.reference_datasets import build_all
from model.molveno.standardize import standardize_all
from model.common.config import setup_logging
from model.common.io import save_computed_dfs

logger = logging.getLogger(__name__)

NORMALIZED_TABLES = ("vodafone_presences", "meteo_station", "road_counts_daily", "beach_counts", "parking_paid",
                     "ispat_daily", "beds", "restaurants", "parking_lots")
REFERENCE_TABLES = ("molveno_daily", "molveno_presence_stats", "molveno_presence_stats_monthly", "molveno_weather_freq",
                    "molveno_counts_daily")


def read_normalized(normalized_dir=MOLVENO_NORMALIZED_DIR) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(normalized_dir / f"{name}.parquet") for name in NORMALIZED_TABLES}


def read_reference(final_dir=MOLVENO_FINAL_DIR) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(final_dir / f"{name}.parquet") for name in REFERENCE_TABLES}


def prepare(skip_download: bool = False, refresh: bool = False, upload: bool = False, local_sources=None) -> dict[str, pd.DataFrame]:
    if skip_download:
        logger.info("Reusing the normalized tables in %s", MOLVENO_NORMALIZED_DIR)
        tables = read_normalized()
    else:
        paths = resolve_sources(load_config(local_source_dir=local_sources), MOLVENO_RAW_DIR, refresh=refresh)
        tables = standardize_all(paths)
        save_computed_dfs(tables, local=True, type_format="parquet", path_saving=MOLVENO_NORMALIZED_DIR)

    reference = build_all(tables)
    save_computed_dfs(reference, local=not upload, type_format="parquet", path_saving=MOLVENO_FINAL_DIR)
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-download", action="store_true", help="reuse <normalized>/molveno/ (skip steps 0-1)")
    parser.add_argument("--refresh", action="store_true", help="download the data-lake files again")
    parser.add_argument("--upload", action="store_true", help="log the reference datasets to the platform")
    parser.add_argument("--local-sources", metavar="DIR", help="read the sources from DIR (overrides molveno.yaml)")
    args = parser.parse_args()
    setup_logging()
    prepare(skip_download=args.skip_download, refresh=args.refresh, upload=args.upload,
            local_sources=args.local_sources)


if __name__ == "__main__":
    main()
