# SPDX-License-Identifier: Apache-2.0
"""
STEP 2 (update) - Processing of the update standardized data.

Input : Output/data/normalized/     (<part>_std, written by standardize_update_raw_data)
        Output/data/raw_data/       (ISPAT arrivals / presences of alb and extralb: not normalized)
        Output/mapping/             (mapping json files, downloaded by the download step)
Output: Output/data/data_processed/ (<dataset>_update_pr)

The processing functions are THE SAME of the base build (process_std_data.py), so every
`<dataset>_update_pr` has exactly the format of the matching base `<dataset>_pr`:
DATA, ID_COMUNE (zero-padded, one comune per row) + the value columns of the dataset, comune x day for
the presences. Parts of the same dataset (e.g. one per year, see `label`) are stacked; later sources
win on a duplicated DATA + ID_COMUNE.
"""

import logging
from pathlib import Path

import pandas as pd

from utils.adapters import KIND_DATASETS
from utils.cleaning import concat_keep_last, concat_replace_groups
from utils.config import (
    DATA_START,
    MAPPING_DIR,
    NORMALIZED_DIR,
    PRESENZE_MEASURES,
    PRESENZE_TAGS,
    PROCESSED_DIR,
    RAW_DIR,
    TYPE_FORMAT,
    setup_logging,
)
from utils.datasets import DATASETS, MULTI_ROW_DATASETS
from utils.io import ensure_dir, raw_path, read_df, save_computed_dfs
from utils.steps.process_std_data import (
    BEDS_ALB_COL,
    BEDS_XALB_COL,
    load_mappings,
    presenze_alb_area_ids,
    presenze_exalb_area_ids,
    presenze_files,
    covisits_area_comune,
    process_beds,
    process_covisits,
    process_energy,
    process_meteo,
    process_provenienza,
    process_provenienza_arrivi,
    process_tempi_arrivi,
    process_popolazione,
    process_presenze_stay,
    process_strutture,
    process_vodafone,
    read_strutture_std,
    stack_covisits,
)
from utils.update.spec import config_from_argv, load_config, plan_parts
from utils.window import Window, clip

logger = logging.getLogger(__name__)

# kind -> processing of the normalized parts (same functions of the base build), restricted to `window`
# (the period of the update). `strutture` = processed strutture (beds for the split of the vodafone areas)
PROCESSORS = {
    "popolazione": lambda df, maps, strutture, window, dataset: process_popolazione(
        df, maps["mapping_comuni"], window
    ),
    "strutture": lambda df, maps, strutture, window, dataset: process_strutture(
        df, maps["mapping_comuni"], window
    ),
    "vodafone": lambda df, maps, strutture, window, dataset: process_vodafone(
        df, maps["mapping_vodafone"], strutture, window=window
    ),
    # the same function for electricity and gas: only the value column / source name change
    "energia": lambda df, maps, strutture, window, dataset: process_energy(
        df,
        maps["mapping_comuni"],
        DATASETS[dataset].value_cols[0],
        dataset,
        window,
    ),
}
# datasets of a kind with a different processing (same standardized layout as the kind)
DATASET_PROCESSORS = {
    "vodafone_provenienza": lambda df, maps, strutture, window, dataset: process_provenienza(
        df, maps["area_comune"], window
    ),
    # arrivals -> top source countries; areas with several comuni are split by beds
    "vodafone_arrivi": lambda df, maps, strutture, window, dataset: process_provenienza_arrivi(
        df, maps["area_comune"], strutture, window
    ),
    # arrivals -> pairs arrivals / travel time per origin market (areas split by beds as above)
    "vodafone_tempi": lambda df, maps, strutture, window, dataset: process_tempi_arrivi(
        df, maps["area_comune"], strutture, window
    ),
}
# not processed part by part, see below: the ISPAT presences (alb / extralb) and the weather are not normalized and are
# processed from the raw files; the OD co-visits are yearly indicators, computed on all the days of a year
# (base + every part of the update)
assert set(PROCESSORS) | {"presenze_raw", "meteo_raw", "vodafone_covisits"} == set(KIND_DATASETS)

PROCESS_ORDER = (
    "popolazione",
    "strutture",
    "vodafone",
    "vodafone_provenienza",
    "vodafone_arrivi",
    "vodafone_tempi",
    "presenze_alb",
    "presenze_extralb",
    "vodafone_covisits",
    "ee_prelievi",
    "gas_prelievi",
    "ee_totale",
    "gas_totale",
    "meteo",
)


def _process_covisits(parts, normalized_dir, mapping_dir, maps, type_format, window):
    """Flow indicators of the update. They are yearly, so they are computed on ALL the days of each year:
    the standardized co-visits of the base build (if any) + every part of the update (later ones win on
    the common dates). Only the years touched by the update are returned."""
    spec = DATASETS["vodafone_covisits"]
    new = [read_df(normalized_dir, part.std_name, type_format) for part in parts]
    try:
        base = [read_df(normalized_dir, spec.std_name, type_format)]
    except FileNotFoundError:
        logger.warning("[vodafone_covisits] no base data: only the update is used")
        base = []
    # whole days of the years touched by the update (base std of earlier years + update), never after its end
    od = clip(
        stack_covisits(base + new),
        Window(max(DATA_START, pd.Timestamp(window.start.year, 1, 1)), window.end),
        name="covisits",
    )
    area_comune = covisits_area_comune(
        mapping_dir, maps["mapping_vodafone"], maps["mapping_comuni"]
    )
    out = process_covisits(od, area_comune, window)
    # the indicators are yearly: only the years of the window (the base years stay as they are)
    return clip(out, window, name="vodafone_covisits")


def _processed_reference(dataset, done, processed_dir, type_format):
    """Processed frame of `dataset`: base `<dataset>_pr` (if any) updated by this run."""
    frames = []
    try:
        frames.append(
            read_df(processed_dir, DATASETS[dataset].processed_name, type_format)
        )
    except FileNotFoundError:
        pass
    if dataset in done:
        frames.append(done[dataset])
    if not frames:
        logger.warning("[%s] no processed data available: uniform split", dataset)
        return None
    return concat_keep_last(frames, name=f"{dataset} reference")


def _process_presenze(dataset, config, maps, beds, raw_dir, window):
    """alb / extralb of the update: arrivals + presences files listed in the config -> comune x day."""
    tag = PRESENZE_TAGS[dataset]
    files, years = {}, set()
    for src in config.sources[dataset]:
        kw = src.adapter_kwargs
        files[(kw["measure"], int(kw["year"]))] = Path(raw_dir) / Path(src.file).name
        years.add(int(kw["year"]))
    for y in years:
        missing = [m for m in PRESENZE_MEASURES if (m, y) not in files]
        if missing:
            raise ValueError(
                f"[{dataset}] year {y}: missing {missing} in the update config"
            )
    # files of the year before (e.g. from the previous update / the base build), if already downloaded
    ctx = min(years) - 1
    files = {**presenze_files(raw_dir, tag, [ctx]), **files}
    if dataset == "presenze_alb":
        area_ids, bed_col = presenze_alb_area_ids(maps["mapping_apt"]), BEDS_ALB_COL
    else:
        area_ids, bed_col = (
            presenze_exalb_area_ids(maps["mapping_comuni"]),
            BEDS_XALB_COL,
        )
    return process_presenze_stay(
        files,
        "alb" if dataset == "presenze_alb" else "exalb",
        sorted(years),
        area_ids,
        beds,
        bed_col=bed_col,
        out_col=DATASETS[dataset].value_cols[0],
        window=window,
    )


def process_update_std_data(
    config,
    normalized_dir=NORMALIZED_DIR,
    mapping_dir=MAPPING_DIR,
    out_dir=PROCESSED_DIR,
    type_format=TYPE_FORMAT,
    raw_dir=RAW_DIR,
):
    """Processes the standardized update parts; returns {<dataset>_update_pr: dataframe}."""
    config = load_config(config)
    normalized_dir = Path(normalized_dir)
    window = config.window
    logger.info("Time window of the update: %s", window)

    maps = load_mappings(mapping_dir)
    if {"vodafone_provenienza", "vodafone_arrivi", "vodafone_tempi"} & set(
        config.datasets
    ):  # areaId -> ID_COMUNE (as for the OD flows)
        maps = {
            **maps,
            "area_comune": covisits_area_comune(
                mapping_dir, maps["mapping_vodafone"], maps["mapping_comuni"]
            ),
        }
    parts = plan_parts(config)
    done = {}  # dataset -> stacked processed frame, filled following PROCESS_ORDER
    beds = None
    for dataset in PROCESS_ORDER:
        if dataset in PRESENZE_TAGS:
            if dataset not in config.datasets:
                continue
            if beds is None:  # beds of every normalized strutture file (base + updates)
                beds = process_beds(
                    read_strutture_std(normalized_dir, type_format),
                    maps["mapping_comuni"],
                    Window(
                        DATA_START, window.end
                    ),  # also the earlier years: beds of the nearest year
                )
            logger.info("[%s] processing the raw arrivals / presences", dataset)
            done[dataset] = _process_presenze(
                dataset, config, maps, beds, raw_dir, window
            )
            continue
        if dataset == "meteo":
            if dataset in config.datasets:  # the files listed in the update config (not normalized)
                files = [raw_path(src.file, raw_dir) for src in config.sources[dataset]]
                logger.info("[%s] processing the raw weather files %s", dataset, [f.name for f in files])
                done[dataset] = process_meteo(files, window)
            continue
        if dataset == "vodafone_covisits":
            ds_parts = [p for p in parts if p.dataset == dataset]
            if ds_parts:
                logger.info(
                    "[%s] processing %s", dataset, [p.std_name for p in ds_parts]
                )
                done[dataset] = _process_covisits(
                    ds_parts,
                    normalized_dir,
                    mapping_dir,
                    maps,
                    type_format,
                    window,
                )
            continue
        frames = []
        for part in (p for p in parts if p.dataset == dataset):
            logger.info("[%s] processing %s", part.dataset, part.std_name)
            std = read_df(normalized_dir, part.std_name, type_format)
            strutture = (
                _processed_reference("strutture", done, out_dir, type_format)
                if dataset in ("vodafone", "vodafone_arrivi", "vodafone_tempi")
                else None
            )
            if dataset == "vodafone":
                popolazione = _processed_reference(
                    "popolazione", done, out_dir, type_format
                )
                frames.append(
                    process_vodafone(
                        std,
                        maps["mapping_vodafone"],
                        strutture,
                        popolazione=popolazione,
                        window=window,
                    )
                )
                continue
            processor = DATASET_PROCESSORS.get(dataset) or PROCESSORS[part.kind]
            frames.append(processor(std, maps, strutture, window, part.dataset))
        if frames:
            stack = concat_replace_groups if dataset in MULTI_ROW_DATASETS else concat_keep_last
            done[dataset] = stack(frames, name=dataset)

    empty = sorted(d for d, df in done.items() if df.empty)
    if empty:
        raise ValueError(
            f"No {empty} data in the update window {window}: check the files of the update config "
            "(do they contain that period?)"
        )
    dict_processed = {DATASETS[d].update_name: df for d, df in done.items()}
    save_computed_dfs(
        dict_processed,
        local=True,
        type_format=type_format,
        path_saving=ensure_dir(out_dir),
    )
    logger.info("Processed update data saved in %s", out_dir)
    return dict_processed


if __name__ == "__main__":
    setup_logging()
    process_update_std_data(config_from_argv("Process the update standardized data."))
