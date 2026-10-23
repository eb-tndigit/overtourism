# SPDX-License-Identifier: Apache-2.0
"""
Base build of the phenomenon dataframes, from scratch (entry point). Runs ALL the steps:

    0 download        server (S3 / platform)       -> Output/data/raw_data/
    1 standardize     raw_data                     -> Output/data/normalized/
    2 process         normalized (+ mappings)      -> Output/data/data_processed/
    3 phenomena       data_processed               -> Output/data/final_data/
                      (phen_popolazione, phen_strutture, phen_presenze, phen_consumption,
                      phen_flussi_od, phen_provenienza, phen_meteo, + derived: phen_ee_tourism_share)

Every step reads the output of the previous one. Directories, file format (`type_format`, parquet by
default) and platform settings are in config/settings.yaml.

The download step NEVER fetches a file that is already in raw_data / mapping (checked file by file, always);
--force-download overrides it.

Usage:
    python -m gen_phenomenon_dataframes                  # base build (2022-2024), all steps
    python -m gen_phenomenon_dataframes --with-update    # base build, then the 2025 update (appended)
    python -m gen_phenomenon_dataframes --update-only    # ONLY the 2025 update (the base build must exist)
    python -m gen_phenomenon_dataframes --skip-download  # raw_data already there
    python -m gen_phenomenon_dataframes --force-download # download again the files already in raw_data
    python -m gen_phenomenon_dataframes --upload         # also log final data to the platform

Each step can also be run on its own, see utils/steps/.
"""

import argparse
import logging
from pathlib import Path

from utils.config import DEFAULT_UPDATE_CONFIG, TYPE_FORMAT, setup_logging
from utils.steps.create_phenomena_df import compute_phenomena
from utils.steps.download_data import download_raw_base_data
from utils.steps.process_std_data import process_data
from utils.steps.standardize_raw_data import standardize_raw_data

logger = logging.getLogger(__name__)


def run_update_2025(
    config=DEFAULT_UPDATE_CONFIG,
    type_format=TYPE_FORMAT,
    local=True,
    skip_download=False,
    force_download=False,
):
    """Runs the yearly update of 2025 only (default: `update.default_config` of settings.yaml): appends the 2025 rows to the
    phenomena of the base build. Files already in raw_data are not downloaded again."""
    from update_phenomenon_dataframes import update_pipeline

    return update_pipeline(
        config,
        type_format=type_format,
        local=local,
        skip_download=skip_download,
        force_download=force_download,
    )


def main_compute_phenomena_dfs(
    type_format=TYPE_FORMAT,
    local=True,
    skip_download=False,
    force_download=False,
    with_update=False,
    update_only=False,
    update_config=DEFAULT_UPDATE_CONFIG,
):
    """Runs the base pipeline and returns the final phenomenon dataframes.
    local=False also uploads the final dataframes to the platform;
    skip_download=True reuses the raw data already in Output/data/raw_data (without it, the files that are
    already there are anyway not downloaded again; force_download=True downloads them again);
    with_update=True runs, after the base build, the update of 2025 only (appended to the phenomena);
    update_only=True runs ONLY the 2025 update (the base build must already exist)."""
    if update_only:
        return run_update_2025(
            update_config, type_format, local, skip_download, force_download
        )
    if skip_download:
        logger.info("Step 0/3: download skipped, using the existing raw data")
    else:
        logger.info("Step 0/3: download raw data")
        download_raw_base_data(type_format=type_format, force=force_download)
    logger.info("Step 1/3: standardize raw data")
    standardize_raw_data(type_format=type_format)
    logger.info("Step 2/3: process standardized data")
    process_data(type_format=type_format)
    logger.info("Step 3/3: compute final phenomena")
    dict_dfs = compute_phenomena(type_format=type_format, local=local)
    logger.info("Base build finished!")
    if with_update:
        logger.info("Running the update of 2025 on top of the base build")
        dict_dfs = run_update_2025(
            update_config, type_format, local, skip_download, force_download
        )
    logger.info("Pipeline finished!")
    return dict_dfs


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Base build of the phenomenon dataframes (all steps)."
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="reuse the raw data already downloaded",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="download again the files already in raw_data (by default they are skipped)",
    )
    p.add_argument(
        "--with-update",
        action="store_true",
        help="after the base build, also run the update of 2025 only",
    )
    p.add_argument(
        "--update-only",
        action="store_true",
        help="run ONLY the update of 2025 (the base build must already exist)",
    )
    p.add_argument(
        "--update-config",
        type=Path,
        default=DEFAULT_UPDATE_CONFIG,
        help="update yaml used by --with-update / --update-only (default: `update.default_config` of settings.yaml)",
    )
    p.add_argument(
        "--upload",
        action="store_true",
        help="log the final phenomena to the platform too",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    setup_logging()
    args = _parse_args()
    if args.with_update and args.update_only:
        raise SystemExit("--with-update and --update-only are mutually exclusive")
    main_compute_phenomena_dfs(
        local=not args.upload,
        skip_download=args.skip_download,
        force_download=args.force_download,
        with_update=args.with_update,
        update_only=args.update_only,
        update_config=args.update_config,
    )
