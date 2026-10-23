import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from haversine import Unit, haversine_vector

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# Everything stored on disk keeps its Italian name: file names, columns and keys of the caches, OSRM block
# labels and the columns of the final dataset. Renaming them would invalidate the caches and change the output.
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = BASE_DIR.parent / "Output"
TRAVEL_TIMES_DIR = OUTPUT_DIR / "travel_times"
WORKSPACE_DIR = BASE_DIR.parents[1]

CONFIG = {
    "shapefile_path": WORKSPACE_DIR / "overtourism-backend/overtourism/overtourism/database/index_data/Com01012026_g/Com01012026_g_WGS84.shp",
    "province_code": "022",  # ISTAT prefix of Trentino (3 digits)

    # None = all of Italy. To test/limit: list of province prefixes,
    # e.g. ["021", "025", "024", "023", "017", "014"] (BZ, BL, VI, VR, BS, SO)
    "extra_provinces": None,

    # --- Cache ---
    # Overpass is only needed for the Trentino admin centres: the raw response is saved and the query
    # is never repeated. To redo it, delete the file (force_recompute does not touch it).
    "cache_overpass_path": TRAVEL_TIMES_DIR / "cache_overpass_admin_centre_trentino.json",
    "cache_points_path": TRAVEL_TIMES_DIR / "capoluoghi_trentino_cache.csv",
    "cache_osrm_dir": TRAVEL_TIMES_DIR / "cache_osrm",
    # As for Overpass: raw Wikidata response, redone only when new countries appear in the Vodafone data
    "cache_wikidata_path": TRAVEL_TIMES_DIR / "cache_wikidata_capitali.json",
    "force_recompute": False,  # recomputes the points; server responses (OSRM, Overpass, Wikidata) stay cached

    # --- Output ---
    # Italian municipalities and foreign countries in a single table, rebuilt from the OSRM cache on every run
    "output_matrix_path": TRAVEL_TIMES_DIR / "matrice_tempi_verso_trentino.csv",

    # --- Servers ---
    "user_agent": "overtourism-trentino/0.1",
    "osrm_url": "https://router.project-osrm.org",
    "osrm_max_coord_per_req": 100,  # demo: 100. Can be raised with a local OSRM
    "osrm_delay_sec": 1.0,          # minimum interval between requests. With a local OSRM: 0
    # If the server answers "busy" (429/502/503/504) retry with exponential backoff, as Overpass asks;
    # on other errors move on to the next server, then fall back to Nominatim
    "overpass_urls": [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
    ],
    "overpass_attempts": 3,      # attempts per server
    "overpass_wait_sec": 10,     # wait before the 2nd attempt, then it doubles
    "overpass_timeout_sec": 25,  # server-side timeout: on an idle server the query takes a few seconds
    "nominatim_url": "https://nominatim.openstreetmap.org/search",
    "nominatim_delay_sec": 1.0,
    "wikidata_url": "https://query.wikidata.org/sparql",
    "max_point_offset_m": 2000,  # point farther than this from its own municipality -> discarded

    # --- Foreign tourists ---
    # Each country starts from its capital. Below the threshold (great-circle capital -> Trento) it drives
    # straight to the municipality, above it flies to the airport and continues by car.
    "vodafone_arrivals_path": OUTPUT_DIR / "data/raw_data/vodafone_arrivals.csv",
    "istat_trento": "022205",
    "car_threshold_km": 1000,
    "airport": {"name": "Bergamo Orio al Serio (BGY)", "lon": 9.69948, "lat": 45.66547},  # Wikidata Q517873
    "flight_speed_kmh": 800,
    "flight_fixed_time_min": 60,  # take-off, landing, taxiing; check-in and security excluded

}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =============================================================================
# 2. MUNICIPALITY POINTS
# =============================================================================
def init_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session

def cache_path(key: str) -> Path:
    p = CONFIG[key]
    extra = CONFIG["extra_provinces"]
    sfx = "italia" if extra is None else "_".join(sorted(extra))
    return p.with_name(f"{p.stem}_{sfx}{p.suffix}")

def read_cache_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Only an empty cell is a missing value: with pandas defaults the municipality "None" (TO) would become NaN."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""], **kwargs)

def load_municipalities() -> gpd.GeoDataFrame:
    """Reads the shapefile, flags the Trentino municipalities and (optionally) filters the extra provinces.
    Trentino municipalities are put at the TOP of the list: the block computation relies on it."""
    municipalities = gpd.read_file(CONFIG["shapefile_path"], encoding="utf-8").to_crs(epsg=4326)
    municipalities = municipalities[["PRO_COM_T", "COMUNE", "geometry"]].copy()

    prefix = municipalities["PRO_COM_T"].str[:3]
    municipalities["is_trentino"] = prefix == CONFIG["province_code"]
    if CONFIG["extra_provinces"] is not None:
        municipalities = municipalities[municipalities["is_trentino"] | prefix.isin(CONFIG["extra_provinces"])]

    return municipalities.sort_values(["is_trentino", "COMUNE"], ascending=[False, True], ignore_index=True)


def parse_admin_centres(data: dict) -> Dict[str, Tuple[float, float, str]]:
    """foreach returns each municipal boundary followed by its admin_centre (if any)."""
    centres = {}
    current_istat = None
    for el in data.get("elements", []):
        if el["type"] == "relation":
            current_istat = el.get("tags", {}).get("ref:ISTAT")
        elif el["type"] == "node" and current_istat:
            centres.setdefault(current_istat, (el["lon"], el["lat"], el.get("tags", {}).get("name")))
    return centres


def fetch_overpass_admin_centres(session: requests.Session) -> Optional[dict]:
    """admin_centre of the Trentino municipalities. Returns the raw JSON, or None if no server answers.

    Selection is by ISTAT code, without a bbox: ref:ISTAT has a few thousand values, so the first step is
    light; the boundary/admin_level filters only apply to the ~170 relations found. A bbox +
    boundary=administrative filter instead forces the server to intersect hundreds of thousands of
    boundaries with the box: on busy servers it ends in a timeout or a 502."""
    prov = CONFIG["province_code"]
    query = f'''[out:json][timeout:{CONFIG['overpass_timeout_sec']}];
    rel["ref:ISTAT"~"^{prov}[0-9][0-9][0-9]$"];
    rel._["boundary"="administrative"]["admin_level"="8"];
    foreach(out tags; node(r:"admin_centre"); out;);'''
    timeout = (10, CONFIG["overpass_timeout_sec"] + 15)  # (connect, read): read > server timeout

    for url in CONFIG["overpass_urls"]:
        for attempt in range(CONFIG["overpass_attempts"]):
            try:
                r = session.post(url, data={"data": query}, timeout=timeout)
                if r.status_code == 400:  # invalid query: no point retrying it on other servers
                    logging.error(f"Overpass rejected the query: {r.text[:500]}")
                    return None
                if r.status_code in (429, 502, 503, 504) and attempt + 1 < CONFIG["overpass_attempts"]:
                    wait = CONFIG["overpass_wait_sec"] * 2 ** attempt
                    logging.warning(f"Overpass busy on {url} ({r.status_code}): retrying in {wait} s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                if "runtime error" in data.get("remark", ""):
                    raise RuntimeError(data["remark"])
                n = len(parse_admin_centres(data))
                if n == 0:  # e.g. an instance with partial data
                    raise RuntimeError("no admin centre in the response")
                logging.info(f"Overpass ok from {url}: {n} admin centres")
                return data
            except (requests.RequestException, RuntimeError, KeyError, ValueError) as err:
                logging.warning(f"Overpass error on {url}: {err}")
                break  # error not caused by load: move on to the next server
    return None


def get_or_download_admin_centres(session: requests.Session) -> Dict[str, Tuple[float, float, str]]:
    cache_file = CONFIG["cache_overpass_path"]
    if cache_file.exists():
        logging.info(f"Loading the Overpass response from cache: {cache_file.name}")
        return parse_admin_centres(json.loads(cache_file.read_text()))

    logging.info("Querying Overpass (a single query, then it stays cached)...")
    data = fetch_overpass_admin_centres(session)
    if data is None:
        logging.warning("Overpass unavailable: falling back to Nominatim for all Trentino municipalities (~3 min)")
        return {}
    cache_file.write_text(json.dumps(data))
    return parse_admin_centres(data)


def nominatim_fallback(session: requests.Session, municipality: str) -> Optional[Tuple[float, float, str]]:
    """Only for TRENTINO municipalities without admin_centre. Returns None if not found."""
    params = {"q": f"{municipality}, Provincia di Trento, Italia", "format": "json", "limit": 1, "countrycodes": "it"}
    try:
        r = session.get(CONFIG["nominatim_url"], params=params, timeout=30)
        r.raise_for_status()
        res = r.json()
    except (requests.RequestException, ValueError) as err:
        logging.warning(f"Nominatim error for {municipality}: {err}")
        res = []
    finally:
        time.sleep(CONFIG["nominatim_delay_sec"])  # policy: max 1 request per second

    if not res:
        return None
    return float(res[0]["lon"]), float(res[0]["lat"]), res[0].get("name", municipality)


def discard_far_points(points: gpd.GeoDataFrame) -> None:
    """Clears the Trentino points that fall far from their own municipality (outdated ref:ISTAT in OSM,
    homonyms on Nominatim). The ISTAT shapefile is generalized: a few hundred metres outside is normal."""
    idx = points.index[points["is_trentino"] & points["lon"].notna()]
    xy = gpd.GeoSeries(gpd.points_from_xy(points.loc[idx, "lon"], points.loc[idx, "lat"]), index=idx, crs=4326)
    offset = xy.to_crs(32632).distance(points.geometry[idx].to_crs(32632))
    far = offset.index[offset > CONFIG["max_point_offset_m"]]
    for i in far:
        logging.warning(f"{points.at[i, 'COMUNE']}: point '{points.at[i, 'fonte']}' {offset[i]:.0f} m from the municipality, discarded")
    points.loc[far, ["lon", "lat"]] = np.nan
    points.loc[far, "fonte"] = ""


def get_or_compute_points(session: requests.Session) -> pd.DataFrame:
    cache_file = cache_path("cache_points_path")
    if cache_file.exists() and not CONFIG["force_recompute"]:
        logging.info(f"Loading the points from cache: {cache_file.name}")
        return read_cache_csv(cache_file, dtype={"PRO_COM_T": str})

    municipalities = load_municipalities()
    n_tn = int(municipalities["is_trentino"].sum())
    logging.info(f"{len(municipalities)} municipalities in total, {n_tn} of them in Trentino.")

    centres = get_or_download_admin_centres(session)
    centres_df = pd.DataFrame.from_dict(centres, orient="index", columns=["lon", "lat", "capoluogo"]).astype({"lon": float, "lat": float})
    points = municipalities.merge(centres_df, left_on="PRO_COM_T", right_index=True, how="left").reset_index(drop=True)
    points["fonte"] = np.where(points["lon"].notna(), "osm_admin_centre", "")
    discard_far_points(points)

    # 1) Trentino municipalities without a valid admin_centre -> Nominatim
    missing_tn = points.index[points["is_trentino"] & points["lon"].isna()]
    logging.info(f"{len(missing_tn)} Trentino municipalities without admin_centre: trying Nominatim (~{len(missing_tn)} s)")
    for i in missing_tn:
        found = nominatim_fallback(session, points.at[i, "COMUNE"])
        if found is not None:
            points.at[i, "lon"], points.at[i, "lat"], points.at[i, "capoluogo"] = found
            points.at[i, "fonte"] = "nominatim"
    discard_far_points(points)

    # 2) all the others (outside the province, or Nominatim failed) -> point inside the polygon
    rest = points["lon"].isna()
    rp = points.geometry.representative_point()
    points.loc[rest, "lon"] = rp.x[rest]
    points.loc[rest, "lat"] = rp.y[rest]
    points.loc[rest, "capoluogo"] = points.loc[rest, "COMUNE"]
    points.loc[rest, "fonte"] = "punto_interno"

    assert points["lon"].notna().all() and points["lat"].notna().all()

    # If many Trentino municipalities are left without an admin centre, Overpass and Nominatim were not answering:
    # better to stop than to cache approximate points and run the whole OSRM computation on them.
    tn_sources = points.loc[points["is_trentino"], "fonte"].value_counts()
    logging.info(f"Source of the Trentino points: {tn_sources.to_dict()}")
    if tn_sources.get("punto_interno", 0) > 0.1 * n_tn:
        raise RuntimeError("Too many Trentino municipalities without an admin centre (Overpass/Nominatim unreachable?): try again later.")

    out = pd.DataFrame(points[["PRO_COM_T", "COMUNE", "capoluogo", "lon", "lat", "is_trentino", "fonte"]])
    out.to_csv(cache_file, index=False)
    logging.info(f"Points saved to {cache_file}")
    return out


# =============================================================================
# 3. OSRM ROUTING
# =============================================================================
def fetch_osrm_table(session, origins, destinations, attempts: int = 5) -> dict:
    coords = ";".join(f"{lon},{lat}" for lon, lat in origins + destinations)
    url = f"{CONFIG['osrm_url']}/table/v1/driving/{coords}"
    params = {
        "sources": ";".join(str(i) for i in range(len(origins))),
        "destinations": ";".join(str(i) for i in range(len(origins), len(origins) + len(destinations))),
        "annotations": "duration,distance",
    }
    error = ""
    for t in range(attempts):
        try:
            r = session.get(url, params=params, timeout=120)
        except requests.RequestException as err:
            error = str(err)
        else:
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()  # other 4xx errors: permanent, no point retrying
                js = r.json()
                if js.get("code") != "Ok":
                    raise RuntimeError(f"OSRM: {js.get('code')} {js.get('message', '')}")
                return js
            error = f"HTTP {r.status_code}"
        wait = 5 * 2 ** t
        logging.warning(f"OSRM {error}: retrying in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"OSRM is not responding: {error}")


def osrm_block(session, coords, o, d, cache_dir: Path, label: str):
    """Computes one block (o = origin indices, d = destination indices), cached on disk.
    The label and the npz keys (durate, distanze, aggancio) are part of the cached files."""
    f = cache_dir / f"{label}_{o[0]}_{d[0]}_{len(o)}_{len(d)}.npz"
    if f.exists():
        z = np.load(f)
        return z["durate"], z["distanze"], z["aggancio"]

    t0 = time.monotonic()
    resp = fetch_osrm_table(session, [coords[i] for i in o], [coords[j] for j in d])
    durations = np.array(resp["durations"], dtype=float)  # seconds; None -> NaN
    distances = np.array(resp["distances"], dtype=float)  # metres
    snap = np.array([s["distance"] for s in resp["sources"]], dtype=float)
    np.savez(f, durate=durations, distanze=distances, aggancio=snap)
    # demo policy: max 1 request/s. The response time already counts towards the interval
    time.sleep(max(0.0, CONFIG["osrm_delay_sec"] - (time.monotonic() - t0)))
    return durations, distances, snap


def _chunks(idx: np.ndarray, step: int) -> list:
    """Indices split into groups of at most `step`: every OSRM request has a cap on coordinates."""
    return [] if len(idx) == 0 else np.array_split(idx, int(np.ceil(len(idx) / step)))


def compute_osrm_matrix(session, points_df: pd.DataFrame):
    """Computes only the pairs with a destination in Trentino: origins = all municipalities (N x nT)."""
    coords = list(zip(points_df["lon"], points_df["lat"]))
    is_tn = points_df["is_trentino"].to_numpy()
    N, nT = len(coords), int(is_tn.sum())
    assert is_tn[:nT].all() and not is_tn[nT:].any(), "Trentino municipalities must come first"

    step = CONFIG["osrm_max_coord_per_req"] // 2
    idx_T, idx_all = np.arange(nT), np.arange(N)

    tag = hashlib.md5(points_df[["lon", "lat"]].round(6).to_csv(index=False).encode()).hexdigest()[:8]
    cache_dir = CONFIG["cache_osrm_dir"] / f"{tag}_{step}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(o, d) for o in _chunks(idx_all, step) for d in _chunks(idx_T, step)]
    logging.info(f"{len(jobs)} OSRM requests (all municipalities x destinations in Trentino)")

    durations, distances = np.full((N, nT), np.nan), np.full((N, nT), np.nan)
    snap = np.full(N, np.nan)

    t0 = time.time()
    for k, (o, d) in enumerate(jobs, start=1):
        dur, dist, block_snap = osrm_block(session, coords, o, d, cache_dir, "verso_tn")
        durations[np.ix_(o, d)], distances[np.ix_(o, d)] = dur, dist
        snap[o] = block_snap
        if k % 25 == 0 or k == len(jobs):
            logging.info(f"  {k}/{len(jobs)} blocks ({(time.time() - t0) / 60:.1f} min)")

    return durations, distances, snap

def italy_matrix(session, points_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """km and minutes from every Italian municipality (rows, in points order) to every Trentino municipality (columns)."""
    durations, distances, snap = compute_osrm_matrix(session, points_df)
    nT = durations.shape[1]

    # --- diagnostics ---
    sub = durations[:nT, :]  # Trentino x Trentino block
    asym = np.abs(sub - sub.T) / 60
    logging.info(f"Round-trip asymmetry (TN-TN): median {np.nanmedian(asym):.1f} min, max {np.nanmax(asym):.1f} min")
    logging.info(f"Snapping to the road network: median {np.nanmedian(snap):.0f} m, "
                 f"max {np.nanmax(snap):.0f} m ({points_df['COMUNE'].iloc[np.nanargmax(snap)]})")

    return np.round(distances / 1000, 2), np.round(durations / 60, 1)


# =============================================================================
# 4. FOREIGN TOURISTS
# =============================================================================
def load_foreign_countries() -> List[str]:
    """ISO3 origin codes among Vodafone tourist arrivals, Italy excluded."""
    arrivals = pd.read_csv(
        CONFIG["vodafone_arrivals_path"],
        usecols=["userCountry", "userProfile", "locType"],
        engine="pyarrow",
    )
    tourist_municipal_arrivals = arrivals[
        (arrivals["userProfile"] == "TOURIST")
        & (arrivals["locType"] == "TN_MKT_AL_3")
    ]
    countries = tourist_municipal_arrivals["userCountry"]
    countries = countries.dropna().astype(str).str.strip().str.upper()
    return sorted(set(countries[countries.ne("")]) - {"ITA"})


def fetch_wikidata_capitals(session: requests.Session, iso3: List[str]) -> dict:
    """For each ISO3 code: capital with coordinates and population, plus the coordinates of the country itself
    (city-states such as HKG and MAC have no separate capital). The SPARQL variable names end up in the
    cached response, so they keep their Italian names."""
    values = " ".join(f'"{c}"' for c in iso3)
    query = f'''SELECT ?iso3 ?paeseLabel ?capitaleLabel ?coordCapitale ?pop ?coordPaese WHERE {{
      VALUES ?iso3 {{ {values} }}
      ?paese wdt:P298 ?iso3 .
      OPTIONAL {{ ?paese wdt:P36 ?capitale . ?capitale wdt:P625 ?coordCapitale .
                 OPTIONAL {{ ?capitale wdt:P1082 ?pop }} }}
      OPTIONAL {{ ?paese wdt:P625 ?coordPaese }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "it,en". }}
    }}'''
    r = session.post(CONFIG["wikidata_url"], data={"query": query, "format": "json"}, timeout=120)
    r.raise_for_status()
    return r.json()


def _wkt_point(wkt: str) -> Tuple[float, float]:
    lon, lat = wkt.removeprefix("Point(").removesuffix(")").split()
    return float(lon), float(lat)


def parse_capitals(data: dict) -> pd.DataFrame:
    """One point per country: the most populous capital (BOL: La Paz rather than Sucre), otherwise the country coordinates."""
    rows = []
    for b in data["results"]["bindings"]:
        v = {k: x["value"] for k, x in b.items()}
        if "coordCapitale" in v:
            rows.append((v["iso3"], v["paeseLabel"], v["capitaleLabel"], *_wkt_point(v["coordCapitale"]),
                         "wikidata_capital", float(v.get("pop", 0))))
        elif "coordPaese" in v:
            rows.append((v["iso3"], v["paeseLabel"], v["paeseLabel"], *_wkt_point(v["coordPaese"]),
                         "wikidata_country", 0.0))
    df = pd.DataFrame(rows, columns=["iso3", "country", "capital", "lon", "lat", "source", "pop"])
    df["has_capital"] = df["source"] == "wikidata_capital"
    df = df.sort_values(["iso3", "has_capital", "pop"], ascending=[True, False, False]).drop_duplicates("iso3")
    return df.drop(columns=["has_capital", "pop"]).reset_index(drop=True)


def get_or_download_capitals(session: requests.Session, iso3: List[str]) -> pd.DataFrame:
    cache_file = CONFIG["cache_wikidata_path"]
    # the requested codes are cached too: a country Wikidata does not have must not repeat the query on every run
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {"iso3": []}
    if set(iso3) <= set(cache["iso3"]):
        logging.info(f"Loading the capitals from cache: {cache_file.name}")
    else:
        logging.info(f"Querying Wikidata for the capitals of {len(iso3)} countries...")
        cache = {"iso3": sorted(iso3), "risposta": fetch_wikidata_capitals(session, iso3)}
        cache_file.write_text(json.dumps(cache))

    capitals = parse_capitals(cache["risposta"])
    capitals = capitals[capitals["iso3"].isin(iso3)].reset_index(drop=True)
    missing = sorted(set(iso3) - set(capitals["iso3"]))
    if missing:
        logging.warning(f"Countries without coordinates on Wikidata, excluded: {missing}")
    return capitals


def great_circle_km(lon, lat, lon0: float, lat0: float) -> np.ndarray:
    """Great-circle distance (km) from every point (lon, lat) to the fixed point (lon0, lat0)."""
    points = np.column_stack([lat, lon])  # haversine wants (lat, lon)
    return haversine_vector(points, np.tile([lat0, lon0], (len(points), 1)), Unit.KILOMETERS)


def road_to_trentino(session, origins: list, destinations: list):
    """OSRM durations (s) and distances (m) from arbitrary points to the Trentino municipalities, in cached blocks."""
    coords = origins + destinations
    nO = len(origins)
    step = CONFIG["osrm_max_coord_per_req"] // 2

    tag = hashlib.md5(np.round(coords, 6).tobytes()).hexdigest()[:8]
    cache_dir = CONFIG["cache_osrm_dir"] / f"{tag}_{step}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    durations = np.full((nO, len(destinations)), np.nan)
    distances = np.full((nO, len(destinations)), np.nan)
    for o in _chunks(np.arange(nO), step):
        for d in _chunks(np.arange(nO, len(coords)), step):
            dur, dist, _ = osrm_block(session, coords, o, d, cache_dir, "esteri")
            durations[np.ix_(o, d - nO)], distances[np.ix_(o, d - nO)] = dur, dist
    return durations, distances


def foreign_matrix(session, points_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ISO3 codes, km and minutes from every foreign country (rows) to every Trentino municipality (columns).
    For countries arriving by plane: flight capital -> airport + road airport -> municipality."""
    capitals = get_or_download_capitals(session, load_foreign_countries())
    trento = points_df.loc[points_df["PRO_COM_T"] == CONFIG["istat_trento"]].iloc[0]
    km_trento = great_circle_km(capitals["lon"], capitals["lat"], trento["lon"], trento["lat"])
    capitals["mode"] = np.where(km_trento < CONFIG["car_threshold_km"], "car", "plane")

    # Road legs in a single table: rows = capitals of the countries by car + airport (last row)
    tn = points_df[points_df["is_trentino"]]
    by_car = np.flatnonzero(capitals["mode"] == "car")
    airport = CONFIG["airport"]
    origins = list(zip(capitals.loc[by_car, "lon"], capitals.loc[by_car, "lat"])) + [(airport["lon"], airport["lat"])]
    durations, distances = road_to_trentino(session, origins, list(zip(tn["lon"], tn["lat"])))
    assert not np.isnan(durations[-1]).any(), f"OSRM: Trentino municipalities unreachable from {airport['name']}"

    road_row = np.full(len(capitals), len(by_car))  # by default the road leg starts from the airport
    road_row[by_car] = np.arange(len(by_car))
    no_road = by_car[np.isnan(durations[: len(by_car)]).any(axis=1)]
    if len(no_road):
        logging.warning(f"No road route for {capitals.loc[no_road, 'iso3'].tolist()}: they switch to plane")
        capitals.loc[no_road, "mode"] = "plane"
        road_row[no_road] = len(by_car)

    is_car = capitals["mode"] == "car"
    logging.info(f"By car (< {CONFIG['car_threshold_km']} km great-circle from Trento): {is_car.sum()} countries "
                 f"({', '.join(capitals.loc[is_car, 'iso3'])}); by plane: {(~is_car).sum()} countries")

    by_plane = (~is_car).to_numpy()
    flight_km = np.where(by_plane, great_circle_km(capitals["lon"], capitals["lat"], airport["lon"], airport["lat"]), 0.0)
    flight_min = np.where(by_plane, CONFIG["flight_fixed_time_min"] + flight_km / CONFIG["flight_speed_kmh"] * 60, 0.0)

    # distances[road_row] = road leg of each country (nP x nT); the flight (0 by car) is added to the whole row.
    # Each leg is rounded before the sum
    km = np.round(np.round(flight_km, 2)[:, None] + np.round(distances[road_row] / 1000, 2), 2)
    minutes = np.round(np.round(flight_min, 1)[:, None] + np.round(durations[road_row] / 60, 1), 1)
    return capitals["iso3"].to_numpy(), km, minutes


# =============================================================================
# 5. FINAL MATRIX
# =============================================================================
def save_final_matrix(points_df: pd.DataFrame, iso3: np.ndarray, km: np.ndarray, minutes: np.ndarray) -> pd.DataFrame:
    """One row per (origin, Trentino municipality) pair. km and minutes hold the Italian municipalities first,
    in points order, then the foreign countries. provenienza = municipality name or ISO3 country code;
    a country has no ISTAT code, so istat_origine = "*"."""
    tn = points_df[points_df["is_trentino"]]
    istat = np.concatenate([points_df["PRO_COM_T"].to_numpy(), np.full(len(iso3), "*")])
    origin = np.concatenate([points_df["COMUNE"].to_numpy(), iso3])
    o, d = np.divmod(np.arange(km.size), km.shape[1])  # row and column of every cell, in ravel() order

    matrix = pd.DataFrame({
        "istat_origine": istat[o],
        "provenienza": origin[o],
        "istat_destinazione": tn["PRO_COM_T"].to_numpy()[d],
        "comune_destinazione": tn["COMUNE"].to_numpy()[d],
        "distanza": km.ravel(),                   # km
        "tempo_di_percorrenza": minutes.ravel(),  # minutes
    })
    before = len(matrix)
    matrix = matrix.dropna(subset=["tempo_di_percorrenza"]).reset_index(drop=True)

    matrix.to_csv(CONFIG["output_matrix_path"], index=False)
    n_foreign = int((matrix["istat_origine"] == "*").sum())
    logging.info(f"Matrix saved to {CONFIG['output_matrix_path']}: {len(matrix)} rows ({len(matrix) - n_foreign} from "
                 f"Italian municipalities, {n_foreign} from foreign countries; dropped {before - len(matrix)} pairs without a route)")
    return matrix


# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    TRAVEL_TIMES_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG["cache_osrm_dir"].mkdir(parents=True, exist_ok=True)
    session = init_session(CONFIG["user_agent"])
    points_df = get_or_compute_points(session)
    km_it, min_it = italy_matrix(session, points_df)
    iso3, km_abroad, min_abroad = foreign_matrix(session, points_df)
    matrix = save_final_matrix(points_df, iso3, np.vstack([km_it, km_abroad]), np.vstack([min_it, min_abroad]))

    return matrix, points_df


def ensure_travel_time_matrix() -> pd.DataFrame:
    """Return the cached matrix, building it only when it is missing or lacks a source country."""
    matrix_path = CONFIG["output_matrix_path"]
    arrivals_path = CONFIG["vodafone_arrivals_path"]
    if not arrivals_path.is_file():
        raise FileNotFoundError(
            f"Vodafone arrivals not found: {arrivals_path}. Run the download step first."
        )

    required_countries = set(load_foreign_countries())
    if matrix_path.is_file():
        existing = pd.read_csv(
            matrix_path,
            usecols=["istat_origine", "provenienza"],
            dtype=str,
        )
        cached_countries = set(
            existing.loc[existing["istat_origine"] == "*", "provenienza"].str.upper()
        )
        missing_countries = required_countries - cached_countries
        if not missing_countries:
            logging.info("Travel-time matrix already cached: %s", matrix_path)
            return read_cache_csv(matrix_path)
        logging.info(
            "Travel-time matrix misses %d country origins; extending it from cached routes where possible: %s",
            len(missing_countries),
            sorted(missing_countries),
        )

    return main()[0]


if __name__ == "__main__":
    matrix = ensure_travel_time_matrix()
