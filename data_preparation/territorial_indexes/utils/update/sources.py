# SPDX-License-Identifier: Apache-2.0
"""How the raw update files (already downloaded in raw_data, see steps/download_data.py) are read.

Readers turn bytes into a RawData(df, meta). To support a new file layout add a function to READERS.
"""

import io
import logging
from pathlib import Path

import pandas as pd

from utils.adapters import RawData
from utils.io import raw_path
from utils.readers import read_grouped_presenze_tsv

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ readers
def read_csv(buffer, **kwargs) -> RawData:
    return RawData(pd.read_csv(buffer, **kwargs))


def read_excel(buffer, **kwargs) -> RawData:
    """.xlsx / .ods (pass engine='odf' for ods); kwargs go to pandas.read_excel (header, sheet_name...)."""
    return RawData(pd.read_excel(buffer, **kwargs))


def read_ispat_apt_tsv(buffer, sep="\t", skiprows=2) -> RawData:
    """ISPAT APT-level TSV: 2-row header, no header parsing; first row (the APT names) goes to meta['apts']."""
    df = pd.read_csv(buffer, sep=sep, header=None, skiprows=skiprows, dtype=str)
    apts = [
        x.strip() for x in buffer.getvalue().decode("utf-8").splitlines()[0].split(sep)
    ]
    return RawData(df, {"apts": apts})


def read_ispat_grouped_tsv(buffer, sep="\t") -> RawData:
    """ISPAT TSV with grouped header, flattened to 'Mese', '<group> Italiani/Stranieri/Totale'."""
    return RawData(read_grouped_presenze_tsv(buffer, sep=sep))


def read_energy_csv(buffer) -> RawData:
    """Energy category file (semicolon-delimited, same layout as the base build, see
    standardize_raw_data.read_energy_category): already in the standard energy layout."""
    from utils.steps.standardize_raw_data import read_energy_category

    return RawData(read_energy_category(buffer))


def read_covisits_csv(buffer, start=None, end=None, chunksize=1_000_000) -> RawData:
    """Vodafone OD co-visits csv, read in chunks (the file is too big to be kept whole): only the rows of the
    time window and the ones used by the flow indicators (area level, profiles, TOTAL rows) are kept, see
    standardize_covisits_columns / covisits_total_mask.
    Default window: the whole year(s) of the update period (the indicators are yearly)."""
    from utils.config import UPDATE_WINDOW
    from utils.datasets import COVISITS_LOC_TYPE, COVISITS_PROFILES
    from utils.steps.standardize_raw_data import covisits_total_mask

    start = pd.Timestamp(str(start)) if start else pd.Timestamp(UPDATE_WINDOW.start.year, 1, 1)
    end = pd.Timestamp(str(end)) if end else UPDATE_WINDOW.end
    parts = []
    for chunk in pd.read_csv(buffer, dtype=str, keep_default_na=False, chunksize=chunksize):
        day = pd.to_datetime(chunk["date"], errors="coerce")
        keep = (
            (day >= start)
            & (day <= end)
            & (chunk["locType"].str.strip() == COVISITS_LOC_TYPE)
            & chunk["userProfile"].str.strip().isin(COVISITS_PROFILES)
        )
        keep &= covisits_total_mask(chunk)  # total rows only, the breakdown by country is discarded
        parts.append(chunk[keep])
    return RawData(pd.concat(parts, ignore_index=True))


READERS = {
    "energy_csv": read_energy_csv,
    "covisits_csv": read_covisits_csv,
    "csv": read_csv,
    "excel": read_excel,
    "ispat_apt_tsv": read_ispat_apt_tsv,
    "ispat_grouped_tsv": read_ispat_grouped_tsv,
}


def read_source(file: str, reader: str, reader_kwargs=None, *, raw_dir) -> RawData:
    """Parses `file` (already downloaded in `raw_dir`) with the named reader."""
    if reader not in READERS:
        raise KeyError(f"Unknown reader {reader!r}. Available: {sorted(READERS)}")
    path = raw_path(file, raw_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the download step first")
    logger.info("Reading %s (reader=%s)", path, reader)
    return READERS[reader](io.BytesIO(path.read_bytes()), **(reader_kwargs or {}))
