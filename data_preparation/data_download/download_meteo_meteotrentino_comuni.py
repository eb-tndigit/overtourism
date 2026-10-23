"""Daily weather (t_min, t_max, rain_mm) of every Trentino comune, from the Meteotrentino open-data service.

This script replaces the two old ones (download_meteo_meteotrentino_all_stations.py and
create_mapping_comuni_stations.py) and runs the whole chain in one go:

  1. downloads the daily t_min / t_max / rain of every station of the Meteotrentino catalogue;
  2. computes the real coverage of each station's temperature series;
  3. places each comune (ISTAT shapefile centroid, or Nominatim geocoding);
  4. assigns to each comune the nearest station with a (nearly) complete temperature series;
  5. joins the two: one row per comune and day.

Files written to ../Output/Download/ (see download_common.py):

    meteo_trentino_comuni_YEAR.csv  <-- one final CSV per requested year
    All intermediate files and caches are kept in a temporary directory and
    automatically removed when the script finishes.

The station catalogue is provided by Meteotrentino's Provincia di Trento service, so stations in Alto
Adige/Südtirol are not included.

Usage:
    pip install requests pandas            # + geopandas if you use --shapefile
    python download_meteo_meteotrentino_comuni.py
    python download_meteo_meteotrentino_comuni.py --start-year 2022 --end-year 2025

Download
  * requests are retried; a request that still fails is recorded as a FAILURE (it is not confused with a
    station that has no temperature);
  * physically impossible values are set to missing, not clipped.

Comune position
  * with --shapefile: centroid of the ISTAT boundary (no network, no name ambiguity); the comuni are the
    ones of the shapefile with COD_PROV == 22, checked against the COMUNI table below;
  * otherwise: OpenStreetMap Nominatim, restricted to a bounding box of Trentino, so that a homonym in
    Alto Adige cannot be chosen; results are cached in comuni_coordinates_cache.json.

Station choice
  Only stations with at least --min-coverage (default 0.95) of the days of the window with BOTH t_min and
  t_max are eligible, whether or not the catalogue marks them as ended. Nearest-by-distance is still a
  starting point, not the best climatological match in mountainous terrain: the details file reports the
  distance and the elevation of the station so that the far / high ones can be checked.

Missing days
  A comune-day row exists only if its station has a row for that day (a day for which the station returned
  neither temperature nor rain is absent, not filled with empty values).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

from download_common import (
    DEFAULT_START_YEAR,
    add_common_args,
    make_session,
    parse_args,
    prepare_output_dir,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVICE = "https://dati.meteotrentino.it/service.asmx"
STATIONS_URL = f"{SERVICE}/listaStazioni"
DATA_URL = f"{SERVICE}/datiStoriciCaratteristiciSerie{{measure}}UnArea"
MEASURES = ("Temperature", "Precipitazioni")

COMUNE_WEATHER_CSV_NAME = "meteo_trentino_comuni.csv"
WEATHER_CSV_NAME = "meteo_comuni.csv"
STATIONS_CSV_NAME = "meteo_stations.csv"
FAILURES_CSV_NAME = "meteo_download_failures.csv"
MAPPING_JSON_NAME = "comune_stazione_meteotrentino.json"
DETAILS_NAME = "comune_stazione_meteotrentino_details.csv"
CACHE_NAME = "comuni_coordinates_cache.json"
GEOCODING_ERRORS_NAME = "comuni_geocoding_errors.csv"

WEATHER_COLUMNS = ["date", "station", "t_min", "t_max", "rain_mm"]
STATION_COLUMNS = [
    "station_code",
    "station",
    "station_short_name",
    "station_lat",
    "station_lon",
    "station_elevation_m",
    "start_date",
    "end_date",
]
COMUNE_WEATHER_COLUMNS = [
    "date",
    "comune_id",
    "comune",
    "station_code",
    "station_rank",
    "distance_km",
    "t_min",
    "t_max",
    "rain_mm",
]

# Plausibility limits (degrees C, mm/day). Values outside are set to missing, not clipped.
T_LIMITS = (-50.0, 50.0)
RAIN_LIMITS = (0.0, 600.0)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "TrentinoComuneStationMapper/3.0"
# Bounding box of the Provincia di Trento (lon_min, lat_max, lon_max, lat_min, as Nominatim `viewbox`).
TRENTINO_BBOX = (10.38, 46.60, 12.05, 45.65)
PROVINCE_CODE = 22
DEFAULT_MIN_COVERAGE = 0.95
DEFAULT_MAX_DISTANCE_KM = 15.0

COMUNI = {
    "ALA": 22001,
    "ALBIANO": 22002,
    "ALDENO": 22003,
    "ANDALO": 22005,
    "ARCO": 22006,
    "AVIO": 22007,
    "BASELGA DI PINE": 22009,
    "BEDOLLO": 22011,
    "BESENELLO": 22013,
    "BIENO": 22015,
    "BLEGGIO SUPERIORE": 22017,
    "BOCENAGO": 22018,
    "BONDONE": 22021,
    "BORGO VALSUGANA": 22022,
    "BRENTONICO": 22025,
    "BRESIMO": 22026,
    "CADERZONE TERME": 22029,
    "CALCERANICA AL LAGO": 22032,
    "CALDES": 22033,
    "CALDONAZZO": 22034,
    "CALLIANO": 22035,
    "CAMPITELLO DI FASSA": 22036,
    "CAMPODENNO": 22037,
    "CANAL SAN BOVO": 22038,
    "CANAZEI": 22039,
    "CAPRIANA": 22040,
    "CARISOLO": 22042,
    "CARZANO": 22043,
    "CASTEL CONDINO": 22045,
    "CASTELLO-MOLINA DI FIEMME": 22047,
    "CASTELLO TESINO": 22048,
    "CASTELNUOVO": 22049,
    "CAVALESE": 22050,
    "CAVARENO": 22051,
    "CAVEDAGO": 22052,
    "CAVEDINE": 22053,
    "CAVIZZANA": 22054,
    "CIMONE": 22058,
    "CINTE TESINO": 22059,
    "CIS": 22060,
    "CIVEZZANO": 22061,
    "CLES": 22062,
    "COMMEZZADURA": 22064,
    "CROVIANA": 22068,
    "DAMBEL": 22071,
    "DENNO": 22074,
    "DRENA": 22078,
    "DRO": 22079,
    "FAI DELLA PAGANELLA": 22081,
    "FIAVE": 22083,
    "FIEROZZO": 22085,
    "FOLGARIA": 22087,
    "FORNACE": 22089,
    "FRASSILONGO": 22090,
    "GARNIGA TERME": 22091,
    "GIOVO": 22092,
    "GIUSTINO": 22093,
    "GRIGNO": 22095,
    "IMER": 22097,
    "ISERA": 22098,
    "LAVARONE": 22102,
    "LAVIS": 22103,
    "LEVICO TERME": 22104,
    "LIVO": 22106,
    "LONA-LASES": 22108,
    "LUSERNA": 22109,
    "MALE": 22110,
    "MASSIMENO": 22112,
    "MAZZIN": 22113,
    "MEZZANA": 22114,
    "MEZZANO": 22115,
    "MEZZOCORONA": 22116,
    "MEZZOLOMBARDO": 22117,
    "MOENA": 22118,
    "MOLVENO": 22120,
    "MORI": 22123,
    "NAGO-TORBOLE": 22124,
    "NOGAREDO": 22127,
    "NOMI": 22128,
    "NOVALEDO": 22129,
    "OSPEDALETTO": 22130,
    "OSSANA": 22131,
    "PALU DEL FERSINA": 22133,
    "PANCHIA": 22134,
    "RONZO-CHIENIS": 22135,
    "PEIO": 22136,
    "PELLIZZANO": 22137,
    "PELUGO": 22138,
    "PERGINE VALSUGANA": 22139,
    "PIEVE TESINO": 22142,
    "PINZOLO": 22143,
    "POMAROLO": 22144,
    "PREDAZZO": 22147,
    "RABBI": 22150,
    "RIVA DEL GARDA": 22153,
    "ROMENO": 22155,
    "RONCEGNO TERME": 22156,
    "RONCHI VALSUGANA": 22157,
    "RONZONE": 22159,
    "ROVERE DELLA LUNA": 22160,
    "ROVERETO": 22161,
    "RUFFRE-MENDOLA": 22162,
    "RUMO": 22163,
    "SAGRON MIS": 22164,
    "SAMONE": 22165,
    "SAN MICHELE ALL'ADIGE": 22167,
    "SANT'ORSOLA TERME": 22168,
    "SANZENO": 22169,
    "SARNONICO": 22170,
    "SCURELLE": 22171,
    "SEGONZANO": 22172,
    "SFRUZ": 22173,
    "SORAGA DI FASSA": 22176,
    "SOVER": 22177,
    "SPIAZZO": 22179,
    "SPORMAGGIORE": 22180,
    "SPORMINORE": 22181,
    "STENICO": 22182,
    "STORO": 22183,
    "STREMBO": 22184,
    "TELVE": 22188,
    "TELVE DI SOPRA": 22189,
    "TENNA": 22190,
    "TENNO": 22191,
    "TERRAGNOLO": 22193,
    "TERZOLAS": 22195,
    "TESERO": 22196,
    "TIONE DI TRENTO": 22199,
    "TON": 22200,
    "TORCEGNO": 22202,
    "TRAMBILENO": 22203,
    "TRENTO": 22205,
    "VALFLORIANA": 22209,
    "VALLARSA": 22210,
    "VERMIGLIO": 22213,
    "VIGNOLA-FALESINA": 22216,
    "VILLA LAGARINA": 22222,
    "VOLANO": 22224,
    "ZIANO DI FIEMME": 22226,
    "COMANO TERME": 22228,
    "LEDRO": 22229,
    "PREDAIA": 22230,
    "SAN LORENZO DORSINO": 22231,
    "VALDAONE": 22232,
    "DIMARO FOLGARIDA": 22233,
    "PIEVE DI BONO-PREZZO": 22234,
    "ALTAVALLE": 22235,
    "ALTOPIANO DELLA VIGOLANA": 22236,
    "AMBLAR-DON": 22237,
    "BORGO CHIESE": 22238,
    "BORGO LARES": 22239,
    "CASTEL IVANO": 22240,
    "CEMBRA LISIGNAGO": 22241,
    "CONTA": 22242,
    "MADRUZZO": 22243,
    "PORTE DI RENDENA": 22244,
    "PRIMIERO SAN MARTINO DI CASTROZZA": 22245,
    "SELLA GIUDICARIE": 22246,
    "TRE VILLE": 22247,
    "VALLELAGHI": 22248,
    "VILLE D'ANAUNIA": 22249,
    "SAN GIOVANNI DI FASSA": 22250,
    "TERRE D'ADIGE": 22251,
    "BORGO D'ANAUNIA": 22252,
    "NOVELLA": 22253,
    "VILLE DI FIEMME": 22254,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _to_float(value: str) -> float:
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return float("nan")


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle (haversine) distance in km."""
    rad = math.pi / 180
    p1, p2 = lat1 * rad, lat2 * rad
    h = (
        math.sin((lat2 - lat1) * rad / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin((lon2 - lon1) * rad / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(math.sqrt(h))


def format_comune_id(code: int | str, width: int = 5) -> str:
    """ISTAT comune id as a zero-padded string (width 5 -> '22001', width 6 -> '022001', the ID_COMUNE of the
    phen_* files)."""
    digits = str(int(str(code).strip()))
    return digits.zfill(width)


def expected_days(start_year: int, end_year: int, today: dt.date | None = None) -> int:
    """Number of days of the requested window that can already have data (the window is cut at yesterday)."""
    today = today or dt.date.today()
    start = dt.date(start_year, 1, 1)
    end = min(dt.date(end_year, 12, 31), today - dt.timedelta(days=1))
    return max((end - start).days + 1, 0)


# ---------------------------------------------------------------------------
# 1. Station catalogue and download
# ---------------------------------------------------------------------------


def parse_catalogue(xml_bytes: bytes) -> pd.DataFrame:
    """Parse the XML of ``listaStazioni`` into one row per station (columns: STATION_COLUMNS).

    A node is a station when it has a ``codice``. XML namespaces are ignored. A station listed more than
    once keeps its last row. Coordinates / elevation that cannot be read are NaN (the station is kept: the
    mapper decides if it is usable).
    """
    root = ET.fromstring(xml_bytes)
    stations: dict[str, dict] = {}
    for node in root.iter():
        fields = {
            _local(child.tag): (child.text or "").strip()
            for child in node
            if len(child) == 0
        }
        code = fields.get("codice", "")
        if not code:
            continue
        lat, lon = _to_float(fields.get("latitudine")), _to_float(
            fields.get("longitudine")
        )
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            lat = lon = float("nan")
        stations[code] = {
            "station_code": code,
            "station": fields.get("nome") or fields.get("nomebreve") or code,
            "station_short_name": fields.get("nomebreve") or fields.get("nome") or code,
            "station_lat": lat,
            "station_lon": lon,
            "station_elevation_m": _to_float(fields.get("quota")),
            "start_date": fields.get("inizio", ""),
            "end_date": fields.get("fine", ""),
        }
    if not stations:
        raise RuntimeError(
            "No stations were parsed from the Meteotrentino catalogue; the response format may have changed."
        )
    return pd.DataFrame(list(stations.values()), columns=STATION_COLUMNS)


def fetch_catalogue(session: requests.Session) -> pd.DataFrame:
    response = session.get(STATIONS_URL, timeout=60)
    response.raise_for_status()
    return parse_catalogue(response.content)


class SeriesFailure(Exception):
    """A request failed even after the retries: the data may exist but could not be read."""


def fetch_series(session, measure, station_code, year, cache_dir, refresh):
    """One station / measure / year. Returns a DataFrame (empty = the service answered 'no data').

    Years before the current one are cached: they do not change. Failures are never cached.
    """
    cache_file = cache_dir / f"{measure}_{station_code}_{year}.json"
    cacheable = year < dt.date.today().year
    if cacheable and not refresh and cache_file.exists():
        return pd.DataFrame(json.loads(cache_file.read_text(encoding="utf-8")))

    try:
        response = session.get(
            DATA_URL.format(measure=measure),
            params={"codice": station_code, "anno": year},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise SeriesFailure(str(exc)) from exc

    rows = payload.get("dati", []) if isinstance(payload, dict) else []
    if cacheable:
        cache_file.write_text(json.dumps(rows), encoding="utf-8")
    return pd.DataFrame(rows)


def _numeric(series, limits):
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.between(*limits))


def station_daily_data(session, station, years, cache_dir, refresh, pause, failures):
    """Temperature and rainfall of one station over all the years, one row per day."""
    frames = {"Temperature": [], "Precipitazioni": []}
    for year in years:
        for measure in MEASURES:
            try:
                frame = fetch_series(
                    session, measure, station["station_code"], year, cache_dir, refresh
                )
                if not frame.empty:
                    frames[measure].append(frame)
            except SeriesFailure as exc:
                print(
                    f"  FAILED {station['station_code']} ({station['station']}), {measure} {year}: {exc}"
                )
                failures.append(
                    {
                        "station": station["station_code"],
                        "measure": measure,
                        "year": year,
                        "error": str(exc),
                    }
                )
            time.sleep(pause)

    temp = (
        pd.concat(frames["Temperature"], ignore_index=True)
        if frames["Temperature"]
        else pd.DataFrame(columns=["giorno", "Min", "Max"])
    )
    rain = (
        pd.concat(frames["Precipitazioni"], ignore_index=True)
        if frames["Precipitazioni"]
        else pd.DataFrame(columns=["giorno", "Valore"])
    )
    temp = temp.rename(columns={"Min": "t_min", "Max": "t_max"})
    rain = rain.rename(columns={"Valore": "rain_mm"})
    for column in ("t_min", "t_max"):
        if column not in temp.columns:
            temp[column] = pd.NA
    if "rain_mm" not in rain.columns:
        rain["rain_mm"] = pd.NA
    if "giorno" not in temp.columns:
        temp["giorno"] = pd.NA
    if "giorno" not in rain.columns:
        rain["giorno"] = pd.NA

    temp = temp[["giorno", "t_min", "t_max"]].drop_duplicates("giorno", keep="last")
    rain = rain[["giorno", "rain_mm"]].drop_duplicates("giorno", keep="last")
    out = temp.merge(rain, on="giorno", how="outer").rename(columns={"giorno": "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out["t_min"] = _numeric(out["t_min"], T_LIMITS)
    out["t_max"] = _numeric(out["t_max"], T_LIMITS)
    # t_min > t_max is an error of the source: do not trust either value
    inverted = out["t_min"] > out["t_max"]
    out.loc[inverted, ["t_min", "t_max"]] = float("nan")
    out["rain_mm"] = _numeric(out["rain_mm"], RAIN_LIMITS)
    out["station"] = station["station_code"]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out[WEATHER_COLUMNS]


# ---------------------------------------------------------------------------
# 2. Coverage
# ---------------------------------------------------------------------------


def station_coverage(weather: pd.DataFrame, n_expected: int) -> pd.DataFrame:
    """Days with data per station. A temperature day counts only if BOTH t_min and t_max are present."""
    if weather.empty:
        return pd.DataFrame(
            columns=[
                "station_code",
                "n_days_temp",
                "n_days_rain",
                "coverage_temp",
                "coverage_rain",
                "first_date",
                "last_date",
            ]
        )
    has_temp = weather["t_min"].notna() & weather["t_max"].notna()
    grouped = weather.assign(has_temp=has_temp, has_rain=weather["rain_mm"].notna())
    cov = grouped.groupby("station").agg(
        n_days_temp=("has_temp", "sum"),
        n_days_rain=("has_rain", "sum"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )
    denominator = max(n_expected, 1)
    cov["coverage_temp"] = (cov["n_days_temp"] / denominator).clip(upper=1).round(4)
    cov["coverage_rain"] = (cov["n_days_rain"] / denominator).clip(upper=1).round(4)
    return cov.reset_index().rename(columns={"station": "station_code"})


def download_all_stations(args, out_dir: Path):
    """Steps 1-2: returns (weather, stations) and writes meteo_comuni.csv / meteo_stations.csv."""
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = make_session()
    years = range(args.start_year, args.end_year + 1)

    catalogue = fetch_catalogue(session)
    print(f"Found {len(catalogue)} stations in the Meteotrentino catalogue.")

    all_data, failures = [], []
    for index, station in enumerate(catalogue.to_dict("records"), start=1):
        print(
            f"[{index}/{len(catalogue)}] {station['station_code']} - {station['station']}"
        )
        data = station_daily_data(
            session, station, years, cache_dir, args.refresh, args.pause, failures
        )
        if not data.empty:
            all_data.append(data)

    if all_data:
        weather = pd.concat(all_data, ignore_index=True)
        first, last = f"{args.start_year}-01-01", f"{args.end_year}-12-31"
        weather = weather[weather["date"].between(first, last)]
        weather = weather.drop_duplicates(["date", "station"], keep="last")
        weather = weather.sort_values(["date", "station"], kind="stable")
    else:
        weather = pd.DataFrame(columns=WEATHER_COLUMNS)
    weather = weather[WEATHER_COLUMNS]
    weather.to_csv(out_dir / WEATHER_CSV_NAME, index=False)

    n_expected = expected_days(args.start_year, args.end_year)
    coverage = station_coverage(weather, n_expected)
    stations = catalogue[STATION_COLUMNS].merge(coverage, on="station_code", how="left")
    for column in ("n_days_temp", "n_days_rain"):
        stations[column] = stations[column].fillna(0).astype(int)
    for column in ("coverage_temp", "coverage_rain"):
        stations[column] = stations[column].fillna(0.0)
    stations["expected_days"] = n_expected
    stations["window_start_year"] = args.start_year
    stations["window_end_year"] = args.end_year
    stations.to_csv(out_dir / STATIONS_CSV_NAME, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(out_dir / FAILURES_CSV_NAME, index=False)
        print(
            f"WARNING: {len(failures)} requests failed; see {out_dir / FAILURES_CSV_NAME}. "
            "Run again: the finished years are cached, only the missing ones are requested."
        )
    with_temp = int((stations["n_days_temp"] > 0).sum())
    complete = int((stations["coverage_temp"] >= 0.95).sum())
    print(
        f"Saved {out_dir / WEATHER_CSV_NAME}: {len(weather):,} rows, {weather['station'].nunique()} stations."
    )
    print(
        f"Saved {out_dir / STATIONS_CSV_NAME}: {len(stations)} catalogue stations, {with_temp} with "
        f"temperature, {complete} with temperature coverage >= 95% of {n_expected} days."
    )
    return weather, stations


def load_existing(out_dir: Path):
    """--skip-download: read the files written by a previous run."""
    weather_csv, stations_csv = out_dir / WEATHER_CSV_NAME, out_dir / STATIONS_CSV_NAME
    for path in (weather_csv, stations_csv):
        if not path.exists():
            raise SystemExit(f"{path} not found: run once without --skip-download.")
    weather = pd.read_csv(weather_csv, dtype={"station": str})
    stations = pd.read_csv(stations_csv, dtype={"station_code": str})
    print(
        f"Reusing {weather_csv.name} ({len(weather):,} rows) and {stations_csv.name}."
    )
    return weather, stations


# ---------------------------------------------------------------------------
# 3. Comune positions
# ---------------------------------------------------------------------------


def inside_trentino(lat: float, lon: float) -> bool:
    lon_min, lat_max, lon_max, lat_min = TRENTINO_BBOX
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def geocode(name, code, session):
    query = name.title().replace("Pine", "Piné").replace("Fiave", "Fiavè")
    if name == "MALE":
        query = "Malè"
    lon_min, lat_max, lon_max, lat_min = TRENTINO_BBOX
    r = session.get(
        NOMINATIM_URL,
        params={
            "q": f"{query}, Provincia di Trento, Italia",
            "format": "jsonv2",
            "limit": 5,
            "countrycodes": "it",
            "viewbox": f"{lon_min},{lat_max},{lon_max},{lat_min}",
            "bounded": 1,
        },
        timeout=30,
    )
    r.raise_for_status()
    results = [
        x
        for x in r.json()
        if inside_trentino(float(x["lat"]), float(x["lon"]))
        and "BOLZANO" not in x.get("display_name", "").upper()
    ]
    if not results:
        raise RuntimeError(f"No geocoding result inside Trentino for {name} ({code})")

    def score(x):
        d = x.get("display_name", "").upper()
        return (
            3 if "PROVINCIA DI TRENTO" in d else 0,
            1 if x.get("class") in ("boundary", "place") else 0,
            float(x.get("importance", 0)),
        )

    item = max(results, key=score)
    return {
        "lat": float(item["lat"]),
        "lon": float(item["lon"]),
        "display_name": item.get("display_name", ""),
    }


def points_from_nominatim(out_dir: Path) -> dict[int, dict]:
    cache_file = out_dir / CACHE_NAME
    cache = (
        json.loads(cache_file.read_text(encoding="utf-8"))
        if cache_file.exists()
        else {}
    )
    session = make_session(NOMINATIM_USER_AGENT)
    points, errors = {}, []
    for i, (comune, code) in enumerate(COMUNI.items(), 1):
        try:
            cached = cache.get(str(code))
            if cached is not None and not inside_trentino(cached["lat"], cached["lon"]):
                cached = None  # an old cache entry outside Trentino is a wrong homonym: redo it
            if cached is None:
                cache[str(code)] = geocode(comune, code, session)
                cache_file.write_text(
                    json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                time.sleep(
                    1.1
                )  # Nominatim public endpoint: no bulk / concurrent requests
            pt = cache[str(code)]
            points[code] = {"comune": comune, "lat": pt["lat"], "lon": pt["lon"]}
            print(f"[{i}/{len(COMUNI)}] {comune}: OK")
        except Exception as e:  # noqa: BLE001 - one bad comune must not stop the others
            errors.append({"comune": comune, "codice_comune": code, "error": str(e)})
            print(f"[{i}/{len(COMUNI)}] {comune}: ERROR: {e}")
    if errors:
        path = out_dir / GEOCODING_ERRORS_NAME
        pd.DataFrame(errors).to_csv(path, index=False)
        print(
            f"{len(errors)} errors saved to {path}; coordinates found so far are cached."
        )
    return points


def points_from_shapefile(path: Path) -> dict[int, dict]:
    import geopandas as gpd  # imported here: only needed with --shapefile

    gdf = gpd.read_file(path)
    gdf = gdf[gdf["COD_PROV"].astype(int) == PROVINCE_CODE].copy()
    if gdf.empty:
        raise RuntimeError(f"No comune with COD_PROV == {PROVINCE_CODE} in {path}")
    # centroid in a metric CRS (UTM 32N), then back to WGS84
    centroids = gdf.to_crs(32632).geometry.centroid.to_crs(4326)
    name_col = "COMUNE" if "COMUNE" in gdf.columns else None
    points = {}
    for idx, row in gdf.iterrows():
        code = int(str(row["PRO_COM_T"]).strip())
        points[code] = {
            "comune": str(row[name_col]) if name_col else str(code),
            "lat": float(centroids.loc[idx].y),
            "lon": float(centroids.loc[idx].x),
        }
    missing = sorted(set(COMUNI.values()) - set(points))
    extra = sorted(set(points) - set(COMUNI.values()))
    if missing:
        print(f"WARNING: in the COMUNI table but not in the shapefile: {missing}")
    if extra:
        print(f"WARNING: in the shapefile but not in the COMUNI table: {extra}")
    return points


# ---------------------------------------------------------------------------
# 4. Comune -> station mapping
# ---------------------------------------------------------------------------


def eligible_stations(stations: pd.DataFrame, min_coverage: float) -> pd.DataFrame:
    has_position = stations["station_lat"].notna() & stations["station_lon"].notna()
    enough = stations["coverage_temp"] >= min_coverage
    usable = stations[has_position & enough].copy()
    print(
        f"{len(stations)} catalogue stations, {int(has_position.sum())} with coordinates, "
        f"{int((stations['n_days_temp'] > 0).sum())} with some temperature data, "
        f"{len(usable)} usable (temperature coverage >= {min_coverage:.0%})."
    )
    if usable.empty:
        raise SystemExit(
            "No usable station: check meteo_download_failures.csv, or lower --min-coverage."
        )
    return usable


def rank_stations(point: dict, stations: pd.DataFrame, n: int) -> list[tuple]:
    ranked = sorted(
        (
            distance_km(point["lat"], point["lon"], s.station_lat, s.station_lon),
            s.station_code,
        )
        for s in stations.itertuples()
    )
    return ranked[:n]


def build_mapping(points, stations, n_per_comune, max_distance_km):
    by_code = stations.set_index("station_code")
    rows = []
    for code, point in points.items():
        for rank, (dist, station_code) in enumerate(
            rank_stations(point, stations, n_per_comune), 1
        ):
            s = by_code.loc[station_code]
            rows.append(
                {
                    "comune": point["comune"],
                    "codice_comune": code,
                    "rank": rank,
                    "station_code": station_code,
                    "station": s["station"],
                    "distance_km": round(dist, 2),
                    "station_elevation_m": s["station_elevation_m"],
                    "coverage_temp": s["coverage_temp"],
                    "far": dist > max_distance_km,
                    "comune_lat": point["lat"],
                    "comune_lon": point["lon"],
                    "station_lat": s["station_lat"],
                    "station_lon": s["station_lon"],
                }
            )
    return pd.DataFrame(rows)


def write_mapping(details, out_dir: Path, n_per_comune: int, id_width: int):
    details.sort_values(["codice_comune", "rank"]).to_csv(
        out_dir / DETAILS_NAME, index=False
    )
    # JSON keyed by comune id; value = station code (or list of codes with --stations-per-comune > 1)
    mapping = {}
    for code, group in details.groupby("codice_comune"):
        codes = group.sort_values("rank")["station_code"].tolist()
        key = format_comune_id(code, id_width)
        mapping[key] = codes[0] if n_per_comune == 1 else codes
    path = out_dir / MAPPING_JSON_NAME
    path.write_text(
        json.dumps(dict(sorted(mapping.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {path}: {len(mapping)} municipalities mapped.")


# ---------------------------------------------------------------------------
# 5. Join: weather of each comune
# ---------------------------------------------------------------------------


def build_comune_weather(details: pd.DataFrame, weather: pd.DataFrame, id_width: int):
    """One row per comune, assigned station and day: the weather of the station that serves the comune."""
    mapped = details[
        ["codice_comune", "comune", "rank", "station_code", "distance_km"]
    ].copy()
    mapped["comune_id"] = mapped["codice_comune"].map(
        lambda c: format_comune_id(c, id_width)
    )
    mapped = mapped.rename(columns={"rank": "station_rank"})
    out = mapped.merge(
        weather.rename(columns={"station": "station_code"}),
        on="station_code",
        how="inner",
    )
    out = out[COMUNE_WEATHER_COLUMNS].sort_values(
        ["date", "comune_id", "station_rank"], kind="stable"
    )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\\n\\n")[0])
    add_common_args(ap, start_year=DEFAULT_START_YEAR, end_year=dt.date.today().year)
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="reuse the existing weather/station CSVs from --output-dir instead of downloading.",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="redownload cached year/measure files even if they already exist.",
    )
    ap.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="pause in seconds between individual station/year requests.",
    )
    ap.add_argument(
        "--shapefile",
        type=Path,
        help="use municipality centroids from an ISTAT shapefile instead of Nominatim geocoding.",
    )
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help="minimum temperature coverage (0..1) required for a station to be eligible.",
    )
    ap.add_argument(
        "--max-distance-km",
        type=float,
        default=DEFAULT_MAX_DISTANCE_KM,
        help="maximum distance in km for a candidate station before it is considered too far.",
    )
    ap.add_argument(
        "--stations-per-comune",
        type=int,
        default=1,
        help="number of nearest eligible stations to keep for each comune.",
    )
    args = parse_args(ap, argv)
    if not 0.0 <= args.min_coverage <= 1.0:
        ap.error("--min-coverage must be between 0 and 1")
    if args.max_distance_km < 0:
        ap.error("--max-distance-km must be >= 0")
    if args.stations_per_comune < 1:
        ap.error("--stations-per-comune must be >= 1")
    out_dir = prepare_output_dir(args.output_dir)

    # Intermediate downloads, station metadata, mappings and caches are required
    # by the processing pipeline, but are kept in a temporary directory and
    # removed automatically. Only the final municipality-level CSVs are retained.
    with tempfile.TemporaryDirectory(prefix="meteo_trentino_comuni_") as temp:
        work_dir = Path(temp)
        if args.skip_download:
            weather, stations = load_existing(out_dir)
        else:
            weather, stations = download_all_stations(args, work_dir)
            weather.to_csv(out_dir / WEATHER_CSV_NAME, index=False)
            stations.to_csv(out_dir / STATIONS_CSV_NAME, index=False)
        if weather.empty:
            raise SystemExit(
                "No weather data downloaded: check the download log above for failures."
            )

        usable = eligible_stations(stations, args.min_coverage)
        if args.shapefile:
            points = points_from_shapefile(Path(args.shapefile))
        else:
            points = points_from_nominatim(work_dir)
        if not points:
            raise RuntimeError("No comune position available.")
        details = build_mapping(points, usable, args.stations_per_comune, args.max_distance_km)
        write_mapping(details, out_dir, args.stations_per_comune, 5)

        used = set(details["station_code"])
        missing_stations = used - set(weather["station"])
        if missing_stations:
            raise RuntimeError(
                f"Mapped stations missing from downloaded weather: {sorted(missing_stations)}."
            )

        comune_weather = build_comune_weather(details, weather, 5)
        if comune_weather.empty:
            raise SystemExit("No comune weather rows could be generated.")

        for year in range(args.start_year, args.end_year + 1):
            yearly = comune_weather[
                pd.to_datetime(comune_weather["date"]).dt.year == year
            ].copy()
            path = out_dir / f"meteo_trentino_comuni_{year}.csv"
            yearly.to_csv(path, index=False)
            print(f"Saved {path}: {len(yearly):,} rows.")


if __name__ == "__main__":
    main()
