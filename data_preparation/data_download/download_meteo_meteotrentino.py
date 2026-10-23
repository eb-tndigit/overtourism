"""Daily measured weather of a Meteotrentino station (open-data service), 2023 to today.

    python data_download/download_meteo_meteotrentino.py [STATION]    # default T0074

Writes Download/meteo_stazione_<code>.csv (date, tmin, tmax, rain_mm, quality), to be uploaded to
the data lake under <model>/meteo/ (see model/<model>/<model>.yaml). Stations used:
T0074 = Male', reference station of the area "Sole, Peio e Rabbi" (Fazzon); T0414 = San Lorenzo
in Banale (Pergoletti), reference station of the area "Giudicarie esteriori e Molveno" (Molveno).
"""

import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from .download_common import OUTPUT_DIR

BASE = "https://dati.meteotrentino.it/service.asmx/datiStoriciCaratteristiciSerie{measure}UnArea"
STATION = sys.argv[1] if len(sys.argv) > 1 else "T0074"
YEARS = range(2023, dt.date.today().year + 1)
OUTPUT_DIR = Path("Download")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch(measure, year):
    r = requests.get(
        BASE.format(measure=measure),
        params={"codice": STATION, "anno": year},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
    )
    r.raise_for_status()
    return pd.DataFrame(r.json()["dati"])


temps, rains = [], []
for year in YEARS:
    temps.append(fetch("Temperature", year))
    rains.append(fetch("Precipitazioni", year))
    print(year, "ok")
    time.sleep(1)

temp = pd.concat(temps).rename(
    columns={"Min": "tmin", "Max": "tmax", "QualityMax": "quality"}
)
rain = pd.concat(rains).rename(columns={"Valore": "rain_mm"})
# dates are ISO yyyy-mm-dd (the rain series is listed by day, then month)
out = temp[["giorno", "tmin", "tmax", "quality"]].merge(
    rain[["giorno", "rain_mm"]], on="giorno", how="outer"
)
out = out.rename(columns={"giorno": "date"}).sort_values("date")[
    ["date", "tmin", "tmax", "rain_mm", "quality"]
]
path = OUTPUT_DIR / f"meteo_stazione_{STATION}.csv"
out.to_csv(path, index=False)
print(f"{path}: {len(out)} days, {out['date'].min()} .. {out['date'].max()}")
