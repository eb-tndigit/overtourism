"""Monthly arrivals and presences in complementary accommodation (esercizi extra-alberghieri) by type and provenance.

    python download_arrivi_presenze_exalb.py [--start-year 2022] [--end-year 2025] [--output-dir DIR]

Writes, in ../Output/Download/ (see download_common.py):
    arrivi_exalb_<year>.csv
    presenze_exalb_<year>.csv
"""

from __future__ import annotations

import argparse
import io
import time

import pandas as pd

from download_common import (
    DEFAULT_START_YEAR,
    add_common_args,
    make_session,
    parse_args,
    prepare_output_dir,
)

BASE = "https://statweb.provincia.tn.it/movturistico/data.asp"
SP = "spArrPresEsComplXTipoProvMese"
N_COLS = 13  # Mese + 4 tipologie x 3
# The yearbook of a year is published after the year ends: the current year is not available yet.
DEFAULT_END_YEAR = 2025
PAUSE_SECONDS = 1.0


def fetch_tables(session, year, var=0):
    r = session.get(
        BASE,
        params={"db": "annuarioturismo", "sp": SP, "var": var, "a": year},
        timeout=30,
    )
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    return [
        t for t in tables if t.shape[1] == N_COLS and str(t.iloc[0, 0]).strip() == "Mese"
    ]


def to_wide(t):
    tipologie = t.iloc[0, 1:].astype(str).str.strip().tolist()  # header row 1
    prov = t.iloc[1, 1:].astype(str).str.strip().tolist()  # header row 2
    body = t.iloc[2:].reset_index(drop=True)

    out = pd.DataFrame({"mese": body.iloc[:, 0].astype(str).str.strip()})
    for j in range(1, N_COLS):
        col = f"{tipologie[j-1]} - {prov[j-1]}"
        out[col] = pd.to_numeric(
            body.iloc[:, j].astype(str).str.replace(".", "", regex=False),
            errors="coerce",
        ).astype("Int64")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(ap, start_year=DEFAULT_START_YEAR, end_year=DEFAULT_END_YEAR)
    args = parse_args(ap, argv)

    out_dir = prepare_output_dir(args.output_dir)
    session = make_session()
    for year in range(args.start_year, args.end_year + 1):
        try:
            tabs = fetch_tables(session, year)
            if len(tabs) != 2:
                print(year, f"expected 2 tables, found {len(tabs)}")
                continue
            # first table = arrivi, second = presenze
            for misura, t in zip(["arrivi", "presenze"], tabs):
                to_wide(t).to_csv(out_dir / f"{misura}_exalb_{year}.csv", index=False)
            print(year, "ok")
        except Exception as e:  # noqa: BLE001 - one bad year must not stop the others
            print(year, "skipped:", e)
        time.sleep(PAUSE_SECONDS)
    print(f"Files in {out_dir}")


if __name__ == "__main__":
    main()
