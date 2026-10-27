# SPDX-License-Identifier: Apache-2.0
"""
Redistribution indicator "Squilibrio di carico e ridistribuzione dei flussi" (entry point).

Run it AFTER gen_phenomenon_dataframes.py (and after the update, if you use it): it reads phen_presenze,
phen_strutture and phen_popolazione from final_data. Steps:

    0 download   the extra mapping inputs (ISTAT comuni shapefile, ISTAT typology) -> Output/mapping/
    1 compute    nodes, load ceilings, daily min-cost redistribution            -> Output/data/data_processed/redistribution/
                 daily panels read by the backend                               -> Output/data/final_data/
                 (phen_ridistribuzione, phen_ridistribuzione_flussi, ridistribuzione_nodi)

All the parameters (inputs, model, transport cost) are in config/settings.yaml, section `redistribution`.
Files already in mapping are not downloaded again; --force-download overrides it.
Run it again whenever the presences change (e.g. after the yearly update): the ceilings and the daily solution are
recomputed for every year of the data (about 2 minutes).

Usage:
    python -m gen_redistribution_dataframes                  # download the extra inputs, then compute
    python -m gen_redistribution_dataframes --skip-download  # inputs already in mapping
    python -m gen_redistribution_dataframes --force-download
    python -m gen_redistribution_dataframes --upload         # also log the final tables to the platform
"""

import argparse
import logging

from utils.config import TYPE_FORMAT, setup_logging
from utils.steps.create_redistribution_df import (
    compute_redistribution_dfs,
    download_redistribution_inputs,
)

logger = logging.getLogger(__name__)


def main_compute_redistribution_dfs(type_format=TYPE_FORMAT, local=True, skip_download=False, force_download=False):
    """Returns the three final tables {name: dataframe}. local=False also uploads them to the platform."""
    if skip_download:
        logger.info("Step 0/1: download skipped, using the existing mapping inputs")
    else:
        logger.info("Step 0/1: download the extra inputs")
        download_redistribution_inputs(force=force_download)
    logger.info("Step 1/1: compute the redistribution indicator")
    dfs = compute_redistribution_dfs(type_format=type_format, local=local)
    logger.info("Redistribution indicator finished!")
    return dfs


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Redistribution indicator: daily panels read by the backend.")
    p.add_argument("--skip-download", action="store_true", help="reuse the mapping inputs already downloaded")
    p.add_argument(
        "--force-download",
        action="store_true",
        help="download again the mapping files (by default they are skipped)",
    )
    p.add_argument("--upload", action="store_true", help="log the final tables to the platform too")
    return p.parse_args(argv)


if __name__ == "__main__":
    setup_logging()
    args = _parse_args()
    main_compute_redistribution_dfs(
        local=not args.upload, skip_download=args.skip_download, force_download=args.force_download
    )
