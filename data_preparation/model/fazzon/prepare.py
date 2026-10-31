# SPDX-License-Identifier: Apache-2.0
"""Fazzon data-preparation pipeline (entry point).

    python -m model.fazzon.prepare                    # steps 0-2, sources from fazzon.yaml
    python -m model.fazzon.prepare --skip-download    # steps 1-2 skipped too: reuse normalized tables
    python -m model.fazzon.prepare --refresh          # download the data-lake files again
    python -m model.fazzon.prepare --local-sources DIR    # read the sources from DIR, not the data lake
    python -m model.fazzon.prepare --upload           # also log the reference datasets to the platform

Run from the data_preparation folder (as the other entry points).

| Step | Module | Output |
|------|--------|--------|
| 0 Sources | sources.py | raw files (local copies or <raw>/fazzon/) |
| 1 Standardize | standardize.py | <normalized>/fazzon/*.parquet |
| 2 Reference datasets | reference_datasets.py | <final>/fazzon/*.parquet |
| 3 Upload (optional) | model.common.remote.log_dataframe | platform dataitems |
"""

import argparse
import logging

import pandas as pd

from model.fazzon.config import FAZZON_FINAL_DIR, FAZZON_NORMALIZED_DIR, load_config
from model.fazzon.reference_datasets import build_all
from model.fazzon.sources import resolve_sources
from model.fazzon.standardize import standardize_all
from model.common.config import setup_logging
from model.common.io import save_computed_dfs

logger = logging.getLogger(__name__)

NORMALIZED_TABLES = (
    "stations", "fazzon_parking", "fazzon_parking_2025", "fazzon_greenbus_2025", "road_counts_hourly",
    "road_counts_daily", "meteo_station", "meteo_bulletin", "vodafone_presences", "bike_counts", "tgc_accessi",
)


def read_normalized(normalized_dir=FAZZON_NORMALIZED_DIR) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(normalized_dir / f"{name}.parquet") for name in NORMALIZED_TABLES}


def read_reference(final_dir=FAZZON_FINAL_DIR) -> dict[str, pd.DataFrame]:
    names = ("fazzon_daily", "fazzon_presence_stats", "fazzon_modal_shift", "fazzon_modal_shift_daily",
             "fazzon_monitoring")
    return {name: pd.read_parquet(final_dir / f"{name}.parquet") for name in names}


def prepare(skip_download: bool = False, refresh: bool = False, upload: bool = False, local_sources=None) -> dict[str, pd.DataFrame]:
    config = load_config(local_source_dir=local_sources)
    if skip_download:
        logger.info("Reusing the normalized tables in %s", FAZZON_NORMALIZED_DIR)
        tables = read_normalized()
    else:
        paths = resolve_sources(config, refresh=refresh)
        tables = standardize_all(paths)
        save_computed_dfs(tables, local=True, type_format="parquet", path_saving=FAZZON_NORMALIZED_DIR)

    reference = build_all(tables)
    save_computed_dfs(reference, local=not upload, type_format="parquet", path_saving=FAZZON_FINAL_DIR)
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-download", action="store_true", help="reuse <normalized>/fazzon/ (skip steps 0-1)")
    parser.add_argument("--refresh", action="store_true", help="download the data-lake files again")
    parser.add_argument("--upload", action="store_true", help="log the reference datasets to the platform")
    parser.add_argument("--local-sources", metavar="DIR", help="read the sources from DIR (overrides fazzon.yaml)")
    args = parser.parse_args()
    setup_logging()
    prepare(skip_download=args.skip_download, refresh=args.refresh, upload=args.upload,
            local_sources=args.local_sources)


if __name__ == "__main__":
    main()
