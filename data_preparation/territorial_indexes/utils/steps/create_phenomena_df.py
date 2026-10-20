# SPDX-License-Identifier: Apache-2.0
"""
STEP 3 - Final phenomenon dataframes.

Input : Output/data/data_processed/   (already disaggregated to comune x day, see process_std_data.py)
Output: Output/data/final_data/       (phen_popolazione, phen_strutture, phen_presenze,
                                       phen_consumption, phen_flussi_od, phen_provenienza)
        and, with local=False, the platform

Two modes:
  - base build (datasets=None): reads `<dataset>_pr` and CREATES the phenomenon dataframes
  - update (datasets=[...]):    reads `<dataset>_update_pr` of the updated datasets and APPENDS the
    phenomena to the existing ones in final_data. On a duplicated DATA + ID_COMUNE the update wins.
    A phenomenon is computed only if ALL its input datasets are in the update.

Each phenomenon dataframe contains:
  - DATA: YYYY for yearly data (popolazione, strutture, phen_flussi_od, phen_provenienza), YYYY-MM-DD for daily data
  - ID_COMUNE: zero-padded ISTAT code (e.g. 22001 -> "022001")
plus the phenomenon's value columns.
"""

import logging
from pathlib import Path

from utils.cleaning import concat_keep_last
from utils.config import (
    BASE_WINDOW,
    DATA_START,
    FINAL_DIR,
    PROCESSED_DIR,
    TYPE_FORMAT,
    setup_logging,
)
from utils.datasets import (
    ARRIVI_VALUE_COLS,
    CONSUMPTION_VALUE_COLS,
    DATASETS,
    PHENOMENA,
    PROVENIENZA_VALUE_COLS,
    phenomena_to_recompute,
)
from utils.io import ensure_dir, read_df, save_computed_dfs
from utils.window import Window, clip

logger = logging.getLogger(__name__)


def compute_presenze_trentino(df_alb, df_extralb, df_vodafone):
    """Daily x comune presences dataframe: ISPAT alb + xalb and vodafone presences."""
    df = df_alb.merge(
        df_extralb[["DATA", "ID_COMUNE", "presenze_xalb"]],
        on=["DATA", "ID_COMUNE"],
        how="inner",
    )

    len_pre = len(df)
    df_mg = df.merge(
        df_vodafone[
            [
                "DATA",
                "ID_COMUNE",
                "presenze",
                "presenze_escursionisti_vodafone",
            ]
        ].rename(columns={"presenze": "presenze_vodafone"}),
        on=["DATA", "ID_COMUNE"],
        how="inner",
    )

    if len(df_mg) < len_pre:
        records_lost = len_pre - len(df_mg)
        perc_lost = (records_lost / len_pre) * 100
        missing_dates = set(df["DATA"].dropna()) - set(df_vodafone["DATA"].dropna())

        rows_missing_dates = int(df["DATA"].isin(missing_dates).sum())
        rows_missing_keys = records_lost - rows_missing_dates

        logger.warning(
            "Filtered Vodafone: %d/%d records discarded (%.2f%%).",
            records_lost,
            len_pre,
            perc_lost,
        )
        if missing_dates:
            min_d, max_d = min(missing_dates), max(missing_dates)
            logger.warning(
                "Records lost on dates not covered by Vodafone: %d records across %d days (%s..%s).",
                rows_missing_dates,
                len(missing_dates),
                min_d,
                max_d,
            )
            logger.warning("First day missing: %s | Last: %s", min_d, max_d)
        if rows_missing_keys:
            logger.warning(
                "%d Records lost because of missing ID_COMUNE on common dates.",
                rows_missing_keys,
            )
    return df_mg.sort_values(by=["DATA", "ID_COMUNE"]).reset_index(drop=True)


def compute_consumption(df_ee, df_gas, df_ee_total, df_gas_total):
    """Combine daily electricity and gas consumption (services categories + totals), retaining the keys
    present in any source. Columns: DATA, ID_COMUNE, ee_services, ee_total, gas_services, gas_total."""
    keys = ["DATA", "ID_COMUNE"]
    out = df_ee
    for df in (df_gas, df_ee_total, df_gas_total):
        out = out.merge(df, on=keys, how="outer")
    return (
        out[keys + CONSUMPTION_VALUE_COLS]
        .sort_values(keys)
        .reset_index(drop=True)
    )


def compute_provenienza(df_hhi, df_arrivi):
    """phen_provenienza: concentration of the source markets (HHI, from the presences) + top source countries (from the
    arrivals). Outer join on DATA + ID_COMUNE: a year / comune present in only one source keeps the other columns empty."""
    keys = ["DATA", "ID_COMUNE"]
    df_arrivi = df_arrivi.copy()
    df_arrivi["DATA"] = df_arrivi["DATA"].astype(df_hhi["DATA"].dtype)
    out = df_hhi.merge(df_arrivi[keys + ARRIVI_VALUE_COLS], on=keys, how="outer")
    return out[keys + PROVENIENZA_VALUE_COLS + ARRIVI_VALUE_COLS].sort_values(keys).reset_index(drop=True)


# phenomenon -> how it is built from the processed dataframes {dataset: df}
BUILDERS = {
    "phen_popolazione": lambda pr: pr["popolazione"],
    "phen_strutture": lambda pr: pr["strutture"],
    "phen_presenze": lambda pr: compute_presenze_trentino(
        pr["presenze_alb"], pr["presenze_extralb"], pr["vodafone"]
    ),
    "phen_consumption": lambda pr: compute_consumption(
        pr["ee_prelievi"], pr["gas_prelievi"], pr["ee_totale"], pr["gas_totale"]
    ),
    # yearly flow indicators per comune (quota in ingresso / uscita, quota intra-ambito, PageRank)
    "phen_flussi_od": lambda pr: pr["vodafone_covisits"],
    # yearly concentration (normalized HHI) of the source markets + top source countries (arrivals), per comune
    "phen_provenienza": lambda pr: compute_provenienza(pr["vodafone_provenienza"], pr["vodafone_arrivi"]),
}
assert set(BUILDERS) == set(PHENOMENA)


def append_to_existing(df_old, df_new, name):
    """df_old + df_new on DATA + ID_COMUNE: where the key is in both, df_new is correct."""
    df_new = df_new.copy()
    if df_new["DATA"].dtype != df_old["DATA"].dtype:  # e.g. year as int vs str
        df_new["DATA"] = df_new["DATA"].astype(df_old["DATA"].dtype)
    missing = set(df_old.columns) ^ set(df_new.columns)
    if missing:
        raise ValueError(
            f"[{name}] cannot append: columns differ between existing and update data: {sorted(missing)}"
        )
    df_new = df_new[df_old.columns]
    n_old = len(df_old)
    out = concat_keep_last([df_old, df_new], name=name)
    logger.info(
        "[%s] %d existing rows + %d update rows -> %d rows (%d new keys)",
        name,
        n_old,
        len(df_new),
        len(out),
        len(out) - n_old,
    )
    return out


def compute_phenomena(
    processed_dir=PROCESSED_DIR,
    out_dir=FINAL_DIR,
    type_format=TYPE_FORMAT,
    local=True,
    datasets=None,
    window=None,
):
    """Reads the processed dataframes, computes the phenomena and saves them.

    datasets=None: base build, every `<dataset>_pr` -> phenomena CREATED in out_dir.
    datasets=[...]: update, the `<dataset>_update_pr` of those datasets -> phenomena APPENDED to the
    existing files of out_dir (update wins on a duplicated DATA + ID_COMUNE).
    local=False also uploads the phenomena to the platform.

    window: base build = [period.start, period.base_end] (default); update = the period of the update.
    Whatever the processed files contain, the final phenomena never hold data before `period.start` or after
    the end of the window: in the update the existing phenomena are clipped too (earlier rows stay as they are,
    unless they are older than `period.start`)."""
    processed_dir, out_dir = Path(processed_dir), ensure_dir(out_dir)
    update = datasets is not None
    datasets = list(datasets) if update else list(DATASETS)
    window = window or BASE_WINDOW
    keep = Window(DATA_START, window.end)  # what the final files may contain
    logger.info("Time window of the %s: %s", "update" if update else "base build", window)

    logger.info(
        "Reading %s processed data from %s",
        "update" if update else "base",
        processed_dir,
    )
    pr = {
        name: read_df(
            processed_dir,
            DATASETS[name].update_name if update else DATASETS[name].processed_name,
            type_format,
        )
        for name in datasets
    }

    to_build = phenomena_to_recompute(pr)
    for skipped in sorted(set(PHENOMENA) - set(to_build)):
        logger.warning(
            "%s not computed: it needs %s, the update has only %s",
            skipped,
            sorted(PHENOMENA[skipped]),
            sorted(pr),
        )
    if not to_build:
        raise ValueError(
            f"No phenomenon can be computed from the datasets {sorted(pr)}"
        )

    dict_dfs = {}
    for phen in to_build:
        logger.info("## Computing %s", phen)
        df = clip(BUILDERS[phen](pr), window, name=phen)
        if update:
            try:
                existing = read_df(out_dir, phen, type_format)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"{phen}.{type_format} not found in {out_dir}: the update appends to the "
                    "final data of the base build, run the base build first"
                ) from None
            df = append_to_existing(existing, df, phen)
        dict_dfs[phen] = clip(df, keep, name=phen)

    save_computed_dfs(
        dict_dfs, local=local, type_format=type_format, path_saving=out_dir
    )
    return dict_dfs


if __name__ == "__main__":
    setup_logging()
    compute_phenomena()
