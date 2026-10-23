"""Conventions shared by all the download scripts in this folder.

Every script uses this module so that they agree on:
  * where files are written: ``../Output/Download`` (relative to the folder of the scripts, NOT to the
    current working directory, so the result is the same wherever the script is launched from);
  * how HTTP requests are made (same User-Agent, retries with exponential backoff);
  * the command-line options ``--output-dir``, ``--start-year`` and ``--end-year``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (SCRIPT_DIR / ".." / "Output" / "Download").resolve()

# First year of the consumption / presences data of the project: nothing before 2022-01-01.
DEFAULT_START_YEAR = 2022

USER_AGENT = "Mozilla/5.0 (TrentinoDataDownload/3.0)"


def make_session(user_agent: str = USER_AGENT) -> requests.Session:
    """Session with retries and exponential backoff, so that a temporary error is not mistaken for 'no data'."""
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def add_common_args(
    parser: argparse.ArgumentParser, *, start_year: int, end_year: int
) -> None:
    """Add --output-dir, --start-year, --end-year with the given defaults."""
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"where to write the files (default: {OUTPUT_DIR})",
    )
    parser.add_argument("--start-year", type=int, default=start_year)
    parser.add_argument("--end-year", type=int, default=end_year)


def parse_args(parser: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    args = parser.parse_args(argv)
    if args.start_year > args.end_year:
        parser.error("--start-year must be <= --end-year")
    return args


def prepare_output_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
