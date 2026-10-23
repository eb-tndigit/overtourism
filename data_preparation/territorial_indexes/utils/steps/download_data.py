# SPDX-License-Identifier: Apache-2.0
"""
STEP 0 - Download.

Two entry points, one per pipeline:

    download_raw_base_data()      base build   -> Output/data/raw_data/  (+ Output/mapping/)
    download_update_raw_data(cfg) yearly update -> Output/data/raw_data/  (+ Output/mapping/)

Data are saved exactly as downloaded: no renaming, no normalization.
The reference files (json mappings + geojson) go to Output/mapping/ and are downloaded in BOTH
cases, by the same function (`download_mappings`).
"""

import io
import logging
from pathlib import Path

from utils.config import (
    ARRIVI_RAW_FILE,
    COVISITS_RAW_FILE,
    ENERGY_CATEGORY_FILES,
    ENERGY_FOLDERS,
    ENERGY_ROOT,
    ENERGY_TOTAL_FILES,
    LOCAL_SOURCE_DIR,
    METEO_FILE_PATTERN,
    METEO_FOLDER,
    MAPPING_DIR,
    POPOLAZIONE_RAW_FILE,
    PRESENZE_FILE_PATTERN,
    PRESENZE_FOLDER,
    PRESENZE_MEASURES,
    PRESENZE_TAGS,
    RAW_DIR,
    REFERENCES,
    STRUTTURE_2024_RAW_FILE,
    STRUTTURE_HISTORICAL_RAW_FILE,
    TYPE_FORMAT,
    VODAFONE_NEW_RAW_FILE,
    VODAFONE_OLD_RAW_FILE,
    setup_logging,
)
from utils.datasets import METEO_BASE_YEARS, PRESENZE_BASE_YEARS
from utils.io import ensure_dir, raw_path, save_computed_dfs

logger = logging.getLogger(__name__)

# Data-lake objects of the base build. EVERY name comes from config/settings.yaml (section `files`).
S3_DATA = [
    STRUTTURE_HISTORICAL_RAW_FILE,
    STRUTTURE_2024_RAW_FILE,
    POPOLAZIONE_RAW_FILE,
    VODAFONE_OLD_RAW_FILE,
    VODAFONE_NEW_RAW_FILE,  # completes the presences up to the end of the base window
    ARRIVI_RAW_FILE,  # Vodafone arrivals (source countries of the tourists)
    COVISITS_RAW_FILE,  # Vodafone OD co-visits (flows between areas)
    # ISPAT arrivals / presences (alb, extralb) of the base years; later years come with the updates
    *(
        f"{PRESENZE_FOLDER}/{PRESENZE_FILE_PATTERN.format(measure=measure, tag=tag, year=year)}"
        for year in PRESENZE_BASE_YEARS
        for tag in PRESENZE_TAGS.values()
        for measure in PRESENZE_MEASURES
    ),
    # weather of the base years (one file per year); later years come with the updates
    *(f"{METEO_FOLDER}/{METEO_FILE_PATTERN.format(year=year)}" for year in METEO_BASE_YEARS),
    *ENERGY_TOTAL_FILES.values(),  # consumption of all the usage categories
    *(
        f"{ENERGY_ROOT}/{folder}/{filename}"
        for folder in ENERGY_FOLDERS.values()
        for filename in ENERGY_CATEGORY_FILES
    ),
]


# ------------------------------------------------------------------ low level
def fetch_bytes(key: str, *, local_dir=None, save_to=None) -> io.BytesIO:
    """Content of `key` (relative to the data-lake prefix, or to `local_dir` if given).
    If `save_to` is set, a copy is stored there under the file's base name."""
    if local_dir is not None:
        buffer = io.BytesIO((Path(local_dir) / key).read_bytes())
    else:
        from utils.remote import get_s3

        buffer = get_s3(key)
    if save_to is not None:
        save_to = ensure_dir(save_to)
        (save_to / Path(key).name).write_bytes(buffer.getvalue())
    return buffer


def already_downloaded(key: str, target_dir, force=False) -> bool:
    """True if `key` is already in `target_dir` (non-empty file named as the object): it is NOT downloaded
    again, unless force=True. This check is done for every file, in the base build and in the update."""
    target = Path(target_dir) / Path(key).name
    return not force and target.is_file() and target.stat().st_size > 0


# ------------------------------------------------------------------ shared
def download_mappings(mapping_dir=MAPPING_DIR, local_dir=None, force=False) -> None:
    """Downloads the reference files (json mappings + geojson, `references` in settings.yaml)
    into `mapping_dir`, skipping the ones already there (force=True: download all).
    Used by the base build AND by the update."""
    mapping_dir = ensure_dir(mapping_dir)
    for key in REFERENCES.values():
        if already_downloaded(key, mapping_dir, force):
            logger.info("Mapping %s already in %s: not downloaded", key, mapping_dir)
            continue
        logger.info("Downloading mapping %s...", key)
        fetch_bytes(key, local_dir=local_dir, save_to=mapping_dir)
    logger.info("Mappings saved in %s", mapping_dir)


# ------------------------------------------------------------------ base build
def download_raw_base_data(
    out_dir=RAW_DIR, type_format=TYPE_FORMAT, mapping_dir=MAPPING_DIR, force=False
):
    """Base build: the historical raw files + the mappings. Files already in raw_data are not downloaded
    again (force=True downloads everything)."""
    out_dir = ensure_dir(out_dir)

    for name in S3_DATA:
        # The energy folders contain identically named category files; keep their
        # folder names in raw_data so electricity and gas are never overwritten.
        target_dir = raw_path(name, out_dir).parent
        if already_downloaded(name, target_dir, force):
            logger.info("Object %s already in %s: not downloaded", name, target_dir)
            continue
        logger.info("Downloading object %s from S3...", name)
        fetch_bytes(name, save_to=target_dir)

    download_mappings(mapping_dir, force=force)
    logger.info("Raw data saved in %s, mappings in %s", out_dir, mapping_dir)


# ------------------------------------------------------------------ update
def download_update_raw_data(
    config,
    out_dir=RAW_DIR,
    mapping_dir=MAPPING_DIR,
    local_dir=LOCAL_SOURCE_DIR,
    force=False,
):
    """Update: the source files listed in the update config + the mappings. Files already in raw_data
    (e.g. downloaded by the base build) are not downloaded again (force=True downloads everything).
    `local_dir` (settings.yaml: update.local_source_dir) reads them from a folder instead of S3.
    """
    from utils.update.spec import load_config

    config = load_config(config)
    out_dir = ensure_dir(out_dir)

    for dataset in config.datasets:
        for src in config.sources[dataset]:
            target_dir = raw_path(src.file, out_dir).parent
            if already_downloaded(src.file, target_dir, force):
                logger.info(
                    "[%s] %s already in %s: not downloaded", dataset, src.file, target_dir
                )
                continue
            logger.info("[%s] downloading %s...", dataset, src.file)
            fetch_bytes(src.file, local_dir=local_dir, save_to=target_dir)

    download_mappings(mapping_dir, local_dir=local_dir, force=force)
    logger.info("Update raw data saved in %s, mappings in %s", out_dir, mapping_dir)


if __name__ == "__main__":
    setup_logging()
    download_raw_base_data()
