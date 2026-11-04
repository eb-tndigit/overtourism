# SPDX-License-Identifier: Apache-2.0
"""
STEP 1 - Standardization of the Molveno sources.

Input : raw files (step 0, model/common/sources.py)
Output: <normalized>/molveno/<table>.parquet, one clean table per source

Only cleaning here; modelling choices belong to step 2 (IDs refer to ASSUMPTIONS.md).
"""

import logging
import re
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from model.common.weather import standardize_meteo_station

logger = logging.getLogger(__name__)

# vodafone_aree.csv: 27;comune;TN_MKT_AL_3;Molveno
VODAFONE_LOC_ID = "27"
VODAFONE_LOC_TYPE = "TN_MKT_AL_3"
VODAFONE_PROFILES = {"TOURIST": "tourists", "VISITOR": "excursionists"}


def _read_vodafone(path, columns) -> pd.DataFrame:
    table = pacsv.read_csv(
        path,
        convert_options=pacsv.ConvertOptions(
            column_types={"locId": pa.string(), "date": pa.string()}, include_columns=columns,
        ),
    )
    keep = pc.and_(pc.equal(table["locId"], VODAFONE_LOC_ID), pc.equal(table["locType"], VODAFONE_LOC_TYPE))
    raw = table.filter(keep).to_pandas()
    return raw[raw["userProfile"].isin(list(VODAFONE_PROFILES))]


def _per_day(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.pivot_table(index="date", columns="userProfile", values="value", aggfunc="sum")


def standardize_vodafone(path_attendences, path_str) -> pd.DataFrame:
    """D13: daily presences in the comune of Molveno per profile: Italians (``userCountry ==
    "ITA"``) plus the aggregated foreign total (``*_STR``), since single foreign countries are
    masked to 0 below the privacy threshold of 15."""
    att = _read_vodafone(path_attendences, ["date", "locType", "locId", "userCountry", "userProfile", "value"])
    foreign = _per_day(_read_vodafone(path_str, ["date", "locType", "locId", "userProfile", "value"]))
    ita = _per_day(att[att["userCountry"] == "ITA"])
    out = ita.add(foreign.reindex(ita.index).fillna(0), fill_value=0)  # no STR row = no foreigners
    out = out.rename(columns=VODAFONE_PROFILES).reset_index()
    out.columns.name = None
    out["date"] = pd.to_datetime(out["date"])
    return out[["date", *VODAFONE_PROFILES.values()]].sort_values("date").reset_index(drop=True)


# PAT sensors (contamezzi_descrizione_sensore.csv): direction of the vehicles entering Molveno or the beach
ROAD_ENTRIES = {("M40", 1): "in_from_andalo", ("M42", 2): "in_from_san_lorenzo", ("M47", 1): "in_beach"}
ROAD_CLASSES = ["cl1", "cl2", "cl3", "cl4", "cl5", "cl6", "cl7", "nonrilevato"]  # cl1 = cars (< 5 m)


def _read_zip_csv(path, member_suffix) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        (name,) = [n for n in z.namelist() if n.endswith(member_suffix)]
        with z.open(name) as f:
            return pd.read_csv(f)


def standardize_road_counts(path_zip) -> pd.DataFrame:
    """Daily vehicles per sensor and direction from the hourly PAT counts: ``cars`` (class 1, below
    5 m), ``vehicles`` (all classes), ``hours`` (hours with a count; a complete day has 24)."""
    raw = _read_zip_csv(path_zip, "contamezzi_Molveno.csv")
    raw["data"] = pd.to_datetime(raw["data"])
    raw["date"] = raw["data"].dt.normalize()
    raw["vehicles"] = raw[ROAD_CLASSES].sum(axis=1)
    out = (raw.groupby(["sensore", "direzione", "date"])
           .agg(cars=("cl1", "sum"), vehicles=("vehicles", "sum"), hours=("data", "nunique"))
           .reset_index().rename(columns={"sensore": "sensor", "direzione": "direction"}))
    out["flow"] = [ROAD_ENTRIES.get((s, int(d)), "") for s, d in zip(out["sensor"], out["direction"])]
    return out.sort_values(["sensor", "direction", "date"]).reset_index(drop=True)


def standardize_beach_counts(path_zip) -> pd.DataFrame:
    """Daily counts per beach gate: ``presences`` (people present, as computed by BS4) and
    ``passages`` (people passing under the gate); NaN where a file has no row for the day."""
    tables = []
    for member, column in (("contapersone_presenze_Molveno.csv", "presences"),
                           ("contapersone_passaggi_Molveno.csv", "passages")):
        t = _read_zip_csv(path_zip, member).rename(columns={"data": "date", "varco": "gate", "presenze": column})
        t["date"] = pd.to_datetime(t["date"])
        tables.append(t.set_index(["date", "gate"])[column])
    return pd.concat(tables, axis=1).reset_index().sort_values(["date", "gate"]).reset_index(drop=True)


def _tariff_hours(label: str) -> float:
    """Stay of a parking tariff in hours: '20 MIN', '1 ORA E 40 MIN', '7 ORE', '1 GIORNO', '3 GIORNI'."""
    hours = 0.0
    for value, unit in re.findall(r"(\d+)\s*(MIN|ORA|ORE|GIORNO|GIORNI)", label.upper()):
        hours += int(value) * {"MIN": 1 / 60, "ORA": 1, "ORE": 1, "GIORNO": 24, "GIORNI": 24}[unit]
    if hours == 0:
        raise ValueError(f"Unknown parking tariff: {label!r}")
    return hours


def standardize_parking_paid(path) -> pd.DataFrame:
    """Paying vehicles per day and tariff (stay duration), all the paid car parks of the comune.
    The sheet has one column per day and one row per tariff, plus a TOTALE row (checked, dropped)."""
    sheet = pd.read_excel(path, header=None)
    dates = pd.to_datetime(sheet.iloc[0, 1:].values)
    body = sheet.iloc[1:].set_index(0).fillna(0)  # an empty cell = no vehicle with that tariff
    body.columns = dates
    is_total = body.index.astype(str).str.upper() == "TOTALE"
    if not (body[~is_total].sum() == body[is_total].sum()).all():
        raise ValueError("parking_paid: the tariffs do not add up to the TOTALE row")
    body = body[~is_total]
    out = body.stack().rename("vehicles").reset_index()
    out.columns = ["tariff", "date", "vehicles"]
    out["hours"] = out["tariff"].map(_tariff_hours)
    out["vehicles"] = out["vehicles"].astype(int)
    return out[["date", "tariff", "hours", "vehicles"]].sort_values(["date", "hours"]).reset_index(drop=True)


def standardize_ispat_daily(path) -> pd.DataFrame:
    """ISPAT daily arrivals and overnight stays (hotel + extra-hotel) of the comune of Molveno."""
    sheet = pd.read_excel(path, sheet_name="molveno", header=2)
    sheet.columns = ["year", "month", "date", "comune", "arrivals", "overnight_stays"]
    sheet = sheet[sheet["comune"].astype(str).str.upper() == "MOLVENO"]
    out = sheet[["date", "arrivals", "overnight_stays"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.astype({"arrivals": int, "overnight_stays": int}).sort_values("date").reset_index(drop=True)


BED_COLUMNS = ["hotel_beds", "extra_hotel_beds", "private_beds", "second_home_beds"]


def standardize_beds(path_yearbook, path_2025, comune="Molveno") -> pd.DataFrame:
    """Beds of the comune per year: hotel, extra-hotel, private tourist rentals ("alloggi turistici"
    / "privati") and second homes ("alloggi a disposizione"). Yearbook table XIII for 2019-2023,
    the 2025 ISPAT table for 2025."""
    yb = pd.read_csv(path_yearbook, encoding="utf-8-sig")
    yb = yb[yb["comune"] == comune]
    rows = pd.DataFrame({"year": yb["anno"].astype(int),
                         "hotel_beds": yb["alberghieri posti_letto"],
                         "extra_hotel_beds": yb["extra alb. Posti_letto"],
                         "private_beds": yb["all. privati posti_letto"],
                         "second_home_beds": yb["all. disposizione posti_letto"]})
    t25 = pd.read_excel(path_2025, header=None)
    (r,) = t25.index[t25[0].astype(str).str.strip() == comune]
    beds_2025 = pd.to_numeric(t25.loc[r, [2, 4, 8, 10]].replace("-", 0))
    rows.loc[len(rows)] = [2025, *beds_2025.astype(int)]
    return rows.astype(int).sort_values("year").reset_index(drop=True)


PARKING_LOT_COLUMNS = ["id", "name", "tariff", "monitoring", "gps", "cars", "disabled", "motorbikes",
                       "coaches", "campers", "total"]


def standardize_parking_lots(path) -> pd.DataFrame:
    """Public car parks of the comune with their spaces per vehicle type; ``tariff``: P paid, D
    parking disc, D - L disc or free; ``car_spaces`` = cars + disabled. The totals row and the
    legend are dropped."""
    sheet = pd.read_excel(path)
    sheet.columns = PARKING_LOT_COLUMNS
    sheet = sheet[pd.to_numeric(sheet["id"], errors="coerce").notna()].copy()
    lat_lon = sheet["gps"].str.split(",", expand=True).astype(float)
    out = pd.DataFrame({"name": sheet["name"].str.strip(), "tariff": sheet["tariff"].str.strip(),
                        "paid": sheet["tariff"].str.strip().str.startswith("P"),
                        "lat": lat_lon[0], "lon": lat_lon[1]})
    for column in ("cars", "disabled", "motorbikes", "coaches", "campers"):
        out[column] = pd.to_numeric(sheet[column]).fillna(0).astype(int)
    out["car_spaces"] = out["cars"] + out["disabled"]
    return out.reset_index(drop=True)


def _seats(value) -> tuple[float, float]:
    """'150', '45/50', '55 max.', '120 e poco più' -> (min, max) seats."""
    numbers = [float(n) for n in re.findall(r"\d+", str(value))]
    if not numbers:
        raise ValueError(f"Unreadable seats: {value!r}")
    return min(numbers), max(numbers)


def standardize_restaurants(path) -> pd.DataFrame:
    """Restaurants and bars of the lake area with their seats (min-max where the source gives a range)."""
    sheet = pd.read_excel(path)
    sheet.columns = ["name", "zone", "seats", "note"]
    sheet = sheet.dropna(subset=["name", "seats"])
    seats = sheet["seats"].map(_seats)
    out = pd.DataFrame({"name": sheet["name"].str.strip(), "zone": sheet["zone"].str.strip(),
                        "seats_min": [s[0] for s in seats], "seats_max": [s[1] for s in seats],
                        "note": sheet["note"]})
    return out.reset_index(drop=True)


def standardize_all(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    logger.info("Vodafone...")
    tables = {"vodafone_presences": standardize_vodafone(paths["vodafone_attendences"], paths["vodafone_attendences_str"])}
    logger.info("Weather...")
    tables["meteo_station"] = standardize_meteo_station(paths["meteo_station"])
    logger.info("Counts (roads, beach, paid parking), ISPAT, beds, restaurants, car parks...")
    tables["road_counts_daily"] = standardize_road_counts(paths["road_counts"])
    tables["beach_counts"] = standardize_beach_counts(paths["beach_counts"])
    tables["parking_paid"] = standardize_parking_paid(paths["parking_paid"])
    tables["ispat_daily"] = standardize_ispat_daily(paths["ispat_daily"])
    tables["beds"] = standardize_beds(paths["beds"], paths["beds_2025"])
    tables["restaurants"] = standardize_restaurants(paths["restaurants"])
    tables["parking_lots"] = standardize_parking_lots(paths["parking_lots"])
    return tables
