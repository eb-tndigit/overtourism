# SPDX-License-Identifier: Apache-2.0
"""
STEP 2 - Fazzon reference datasets.

Input : <normalized>/fazzon/ tables (step 1)
Output: <final>/fazzon/
  - fazzon_daily: one row per day, measured Fazzon data + access regime flags + predictors
  - fazzon_presence_stats: the presence statistics of the Fazzon model
    (executor-backend/src/model/fazzon/fazzon_presence_stats.py), per season and weather class
  - fazzon_modal_shift, fazzon_modal_shift_daily (modal_shift.py, step 2b)
  - fazzon_monitoring (monitoring.py, step 2c): drift indicators, every year vs the calibration year

`fazzon_daily` holds measured values and the access rules of each day; assumptions
(occupancy, non-car share, ...) are applied in the statistics. Every decision and
assumption is listed in ASSUMPTIONS.md; the IDs in the comments below refer to it.
"""

import calendar
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ access regimes
# 2025: parking at 5 euro/day until full, no booking. Greenbus pilot 28/06-20/07 (1 euro,
#       wider loop through Fucine and Ossana).
# 2026: regulated access 04/07-31/08 (traffic light 9-19, booking, 20 euro/day, 10 euro after
#       12:00, 5 euro motorbikes, free after 17:00); Lakebus shuttle 04/07-30/08 every 30 min
#       from Pellizzano Parco Sama.
SHUTTLE_PERIODS = {
    "greenbus_pilot": (pd.Timestamp("2025-06-28"), pd.Timestamp("2025-07-20")),
    "lakebus": (pd.Timestamp("2026-07-04"), pd.Timestamp("2026-08-30")),
}
# Regulated access (traffic light, booking, regulated price), per year: add a line for a new season.
REGULATED_PARKING = {
    2026: (pd.Timestamp("2026-07-04"), pd.Timestamp("2026-08-31")),
}
# D3: 21/08/2026, bad weather all day: shuttle not running, parking at 5 euro.
SHUTTLE_NOT_RUNNING = pd.to_datetime(["2026-08-21"])
REDUCED_PRICE_DAYS = {pd.Timestamp("2026-08-21"): 5.0}
# D2: events that make the car count not comparable ("5days": >2000 people, extra meadows
# used as parking, cars not counted).
EVENT_DAYS = pd.to_datetime(["2026-07-02", "2026-07-03"])
# D21: days whose access rules are uncertain (31/08/2026: in the regulated period but after the
# Lakebus, recorded and counted like the 5-euro days).
UNCERTAIN_RULES_DAYS = pd.to_datetime(["2026-08-31"])
# Days left out of the comparisons between years and periods (D2, D3, D21).
EXCLUDED_DAYS = EVENT_DAYS.union(SHUTTLE_NOT_RUNNING).union(UNCERTAIN_RULES_DAYS)

# Parking price per car (euro), on the days with Fazzon parking data.
PRICE_2025 = 5.0  # 2025 rules (operator): 5 euro/day until the car park is full
PRICE_2026_REGULATED = 20.0  # 2026 rules (operator) 04/07-31/08: 20 euro/day ...
PRICE_2026_REGULATED_AFTERNOON = 10.0  # ... 10 euro for entries after 12:00
PRICE_2026_UNREGULATED = 5.0  # A7: 2026 outside 04/07-31/08, ASSUMPTION (as 2025)

# ------------------------------------------------------------------ predictors
ROAD_UP_VALLEY_DIR = 2  # A16: SS42 km increase downhill (Ossana -> Pellizzano -> Mezzana)
ROAD_MORNING_HOURS = range(7, 12)  # A17
ROAD_TOTAL_STATIONS = ("pellizzano", "pracorno", "celledizzo", "tonale", "revo")


def _road_predictors(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    cars = daily.assign(cars=daily["cars"].where(daily["valid"]))
    by_dir = cars.pivot_table(index="date", columns=["station", "direction"], values="cars", dropna=False)
    out = pd.DataFrame(index=by_dir.index)
    for station in ROAD_TOTAL_STATIONS:
        out[f"strada_{station}_totale"] = by_dir[station].sum(axis=1, min_count=2)  # NaN if a direction is missing
    up = hourly[(hourly["station"] == "pellizzano") & (hourly["direction"] == ROAD_UP_VALLEY_DIR)
                & hourly["hour"].isin(ROAD_MORNING_HOURS)]
    morning_up = up.groupby("date")["cars"].sum()
    out["strada_pellizzano_salita_mattina"] = morning_up.where(by_dir[("pellizzano", ROAD_UP_VALLEY_DIR)].notna())
    return out


def _vodafone_predictors(vodafone: pd.DataFrame) -> pd.DataFrame:
    wide = vodafone.pivot(index="date", columns="area", values="presences")
    return wide.add_prefix("vf_pres_")


def _bike_predictors(bikes: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    near = set(stations.loc[(stations["kind"] == "bike") & (stations["role"] == "near_fazzon"), "code"])
    return pd.DataFrame({
        "bici_near_fazzon": bikes[bikes["sensor"].isin(near)].groupby("date")["bikes_peds"].sum(),
        "bici_valley_total": bikes.groupby("date")["bikes_peds"].sum(),
    })


def _tgc_predictors(tgc: pd.DataFrame) -> pd.DataFrame:
    tonale = tgc[tgc["service"].str.contains("Tonale", case=False, na=False)]
    return tonale.groupby("date")["entrants"].sum().rename("tgc_accessi_tonale").to_frame()


# ------------------------------------------------------------------ fazzon_daily
def _fazzon_counts(parking: pd.DataFrame, parking_2025: pd.DataFrame, greenbus: pd.DataFrame) -> pd.DataFrame:
    fz = parking.set_index("date")
    # the 2025-only sheet has days missing from the combined one (late September)
    old = parking_2025.set_index("date")["cars"]
    both = old.index.intersection(fz.index)
    mismatch = (fz.loc[both, "cars_full"] != old.loc[both]).sum()
    if mismatch:
        raise ValueError(f"{mismatch} days with different 2025 car counts in the two parking sheets")
    extra = old.index.difference(fz.index)
    fz = pd.concat([fz, pd.DataFrame({"cars_full": old.loc[extra]})]).sort_index()

    fz["cars"] = fz["cars_full"] + fz["cars_half"].fillna(0)
    valid_bus = greenbus[~greenbus["other_area"]].set_index("date")["tickets"]
    fz["greenbus_tickets"] = valid_bus
    return fz


def _add_parking_price(daily: pd.DataFrame, parking_days: pd.DatetimeIndex) -> None:
    """``parking_price`` (full day) and ``parking_price_afternoon`` (entry after 12:00), euro per
    car, on the days with Fazzon parking data; ``parking_price_assumed`` marks A7."""
    d = daily.index.to_series()
    season = d.isin(parking_days)
    regulated = daily["regulated_parking"]
    unregulated_2026 = season & (d.dt.year == 2026) & ~regulated
    price = pd.Series(np.nan, index=daily.index)
    price[season & (d.dt.year == 2025)] = PRICE_2025
    price[season & regulated] = PRICE_2026_REGULATED
    price[unregulated_2026] = PRICE_2026_UNREGULATED
    afternoon = price.copy()
    afternoon[season & regulated] = PRICE_2026_REGULATED_AFTERNOON
    for day, reduced in REDUCED_PRICE_DAYS.items():  # D3
        price[day] = afternoon[day] = reduced
    daily["parking_price"] = price
    daily["parking_price_afternoon"] = afternoon
    daily["parking_price_assumed"] = unregulated_2026


# ------------------------------------------------------------------ weather class (D7)
WEATHER_CLASSES = ("good", "cool", "rainy")
RAINY_MM = 5.0  # A19: rainy day = rain >= 5 mm
COOL_DEGREES = 2.0  # A19: cool day = not rainy, tmax below the month's climatological mean - 2 C
CLIMATOLOGY_YEARS = (2023, 2026)  # A20: years of the Male' station used as climatology


def _add_weather_class(daily: pd.DataFrame) -> None:
    """``weather``: good / cool / rainy (A19), for the days with station data; ``tmax_clim``:
    the month's mean tmax over CLIMATOLOGY_YEARS (A20)."""
    clim_days = daily["year"].between(*CLIMATOLOGY_YEARS) & daily["tmax"].notna()
    tmax_clim = daily[clim_days].groupby("month")["tmax"].mean()
    daily["tmax_clim"] = daily["month"].map(tmax_clim)
    rainy = daily["rain_mm"] >= RAINY_MM
    cool = ~rainy & (daily["tmax"] < daily["tmax_clim"] - COOL_DEGREES)
    weather = np.select([rainy, cool], ["rainy", "cool"], "good").astype(object)
    weather[(daily["tmax"].isna() | daily["rain_mm"].isna()).to_numpy()] = None
    daily["weather"] = weather


def build_fazzon_daily(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per day from 2023-01-01 to the last day with any data."""
    fz = _fazzon_counts(tables["fazzon_parking"], tables["fazzon_parking_2025"], tables["fazzon_greenbus_2025"])
    meteo = tables["meteo_station"].set_index("date")
    bulletin = tables["meteo_bulletin"].set_index("date")
    parts = [
        fz[["cars_full", "cars_half", "cars", "moto_persons", "bus_up", "bus_down", "greenbus_tickets", "note"]],
        meteo[["tmin", "tmax", "rain_mm", "quality"]].rename(columns={"quality": "meteo_quality"}),
        bulletin.rename(columns={c: f"{c}_bollettino" for c in ("tmin", "tmax", "rain_mm")}),
        _road_predictors(tables["road_counts_hourly"], tables["road_counts_daily"]),
        _vodafone_predictors(tables["vodafone_presences"]),
        _bike_predictors(tables["bike_counts"], tables["stations"]),
        _tgc_predictors(tables["tgc_accessi"]),
    ]
    start = pd.Timestamp("2023-01-01")
    end = max(p.index.max() for p in parts)
    daily = pd.DataFrame(index=pd.date_range(start, end, freq="D", name="date"))
    for p in parts:
        daily = daily.join(p)

    d = daily.index.to_series()
    daily["shuttle"] = None
    for name, (first, last) in SHUTTLE_PERIODS.items():
        daily.loc[d.between(first, last) & ~d.isin(SHUTTLE_NOT_RUNNING), "shuttle"] = name
    daily["regulated_parking"] = False
    for first, last in REGULATED_PARKING.values():
        daily.loc[d.between(first, last), "regulated_parking"] = True
    daily["event_day"] = d.isin(EVENT_DAYS)
    _add_parking_price(daily, parking_days=fz.index)
    daily["year"] = d.dt.year
    daily["month"] = d.dt.month
    _add_weather_class(daily)
    daily["dow"] = d.dt.day_name().str.lower()
    daily["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    return daily.reset_index()


# ------------------------------------------------------------------ presence statistics
# A2: persons per vehicle, linear in the price paid per vehicle between these two points
# (A26: at 5 euro the vehicle count includes the motorbikes, so 2.5 is an average over cars and
# motorbikes). Replaces A1 (2.83 persons per car).
OCCUPANCY_LOW = (PRICE_2025, 2.5)
OCCUPANCY_HIGH = (PRICE_2026_REGULATED, 3.5)


def occupancy(price):
    """Persons per vehicle at the given price paid per vehicle (A2)."""
    (p0, o0), (p1, o1) = OCCUPANCY_LOW, OCCUPANCY_HIGH
    return np.clip(o0 + (np.asarray(price, dtype=float) - p0) * (o1 - o0) / (p1 - p0), o0, o1)


REFERENCE_OCCUPANCY = float(occupancy(PRICE_2025))  # 2025 conditions: 2.5 persons per vehicle
NON_CAR_SHARE = 0.31  # A3: 2022 EETRA survey, non-car share of visitors in 2025 conditions
OTHER_STD_REL = 0.50  # A4: relative SD of non-car visitors in 2025 conditions
CALIBRATION_YEAR = 2025  # D9: calibration on the 2025 parking data (D4: all days included)
SEASONS = {6: "june", 7: "july", 8: "august", 9: "september"}  # D9
# D5: no day-type (weekday) split: no evidence of a weekday effect.


def weather_climatology(daily: pd.DataFrame) -> pd.DataFrame:
    """P(weather class | season) from the Male' station over CLIMATOLOGY_YEARS (A20)."""
    w = daily[daily["year"].between(*CLIMATOLOGY_YEARS) & daily["month"].isin(list(SEASONS))
              & daily["weather"].notna()]
    freq = pd.crosstab(w["month"].map(SEASONS), w["weather"], normalize="index")
    return freq.reindex(columns=list(WEATHER_CLASSES), fill_value=0.0)


def weather_factors(cal: pd.DataFrame, n_iter: int = 50) -> tuple[pd.Series, pd.Series]:
    """Multiplicative weather effect, the same in every season (A21):
    E[cars | season, weather] = level[season] * factor[weather], factor["good"] = 1.

    Fitted by alternating means on the calibration days (``season``, ``weather``, ``cars``).
    """
    factor = pd.Series(1.0, index=list(WEATHER_CLASSES))
    for _ in range(n_iter):
        level = (cal["cars"] / cal["weather"].map(factor)).groupby(cal["season"]).mean()
        ratio = cal["cars"] / cal["season"].map(level)
        factor = ratio.groupby(cal["weather"]).mean().reindex(list(WEATHER_CLASSES))
        factor = factor / factor["good"]
    return level, factor


def level_presence_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """Season x weather levels of the calibration year (A21), in the layout of the presence
    statistics. Replaced by ``build_presence_stats`` (A37, A38); kept for comparison (modal-shift
    days, exploratory analysis).

    * ``season_freq_rel``: share of the calibration days in the season (A14).
    * ``weather_freq_rel``: P(weather | season), climatology (A20).
    * Visitors by private vehicle in 2025 conditions (5 euro; A2, A26): mean = level[season] x
      factor[weather] x REFERENCE_OCCUPANCY (A21);
      SD = mean x CV of the season's weather-adjusted daily cars.
    * Non-car visitors: car visitors x NON_CAR_SHARE / (1 - NON_CAR_SHARE) (A3),
      SD = OTHER_STD_REL x mean (A4).
    """
    other_ratio = NON_CAR_SHARE / (1.0 - NON_CAR_SHARE)
    cal = daily[(daily["year"] == CALIBRATION_YEAR) & daily["cars"].notna()].copy()
    cal["season"] = cal["month"].map(SEASONS)
    cal = cal.dropna(subset=["season", "weather"])
    level, factor = weather_factors(cal)
    missing = factor[factor.isna()].index.tolist()
    if missing:
        raise ValueError(f"no {CALIBRATION_YEAR} days to calibrate the weather classes {missing}")
    adjusted = cal["cars"] / (cal["season"].map(level) * cal["weather"].map(factor))
    cv = adjusted.groupby(cal["season"]).std()
    climatology = weather_climatology(daily)

    rows = []
    for season in [s for s in SEASONS.values() if s in set(cal["season"])]:
        n_season = int((cal["season"] == season).sum())
        for weather in WEATHER_CLASSES:
            mean_car = level[season] * factor[weather] * REFERENCE_OCCUPANCY
            rows.append({
                "season": season,
                "weather": weather,
                "season_freq_rel": n_season / len(cal),
                "weather_freq_rel": climatology.loc[season, weather],
                "mean_car_visitors": mean_car,
                "std_car_visitors": mean_car * cv[season],
                "mean_other_visitors": mean_car * other_ratio,
                "std_other_visitors": mean_car * other_ratio * OTHER_STD_REL,
                "weather_factor": factor[weather],
                "n_days": int(((cal["season"] == season) & (cal["weather"] == weather)).sum()),
            })
    stats = pd.DataFrame(rows)
    stats.insert(0, "calibration", f"parking_{CALIBRATION_YEAR}")
    return stats


MIN_CELL_DAYS = 10  # fewer traffic days in a season x weather cell: error, not a thin estimate


def season_weights() -> pd.Series:
    """A14: share of the calendar days of each season (month) in June-September."""
    days = pd.Series({season: calendar.monthrange(CALIBRATION_YEAR, month)[1] for month, season in SEASONS.items()})
    return days / days.sum()


def build_presence_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """Presence statistics per (season, weather) cell, in the layout read by
    fazzon_presence_stats.py.

    * ``season_freq_rel``: share of the calendar days in the season (A14).
    * ``weather_freq_rel``: P(weather | season), climatology (A20).
    * Visitors by private vehicle in 2025 conditions (5 euro; A2, A26): the cars of every season
      day of every year with valid traffic and weather class, from the calibration year's relation
      with the traffic (A37; below zero: zero); mean = their mean x REFERENCE_OCCUPANCY;
      variance = their variance + DISPERSION x mean (the scatter of the days around the
      relation, A38), x REFERENCE_OCCUPANCY^2.
    * Non-car visitors: car visitors x NON_CAR_SHARE / (1 - NON_CAR_SHARE) (A3),
      SD = OTHER_STD_REL x mean (A4).
    """
    from model.fazzon.demand import calibration_days, demand_days, dispersion, expected_cars, fit_demand

    other_ratio = NON_CAR_SHARE / (1.0 - NON_CAR_SHARE)
    cal = calibration_days(daily)
    coef = fit_demand(cal)
    phi = dispersion(cal, coef)
    days = demand_days(daily).copy()
    days["season"] = days["month"].map(SEASONS)
    days["cars"] = np.clip(expected_cars(days, coef), 0.0, None)
    climatology = weather_climatology(daily)
    weights = season_weights()

    rows = []
    for season in SEASONS.values():
        for weather in WEATHER_CLASSES:
            cell = days.loc[(days["season"] == season) & (days["weather"] == weather), "cars"]
            if len(cell) < MIN_CELL_DAYS:
                raise ValueError(f"{season} / {weather}: {len(cell)} days with traffic, fewer than {MIN_CELL_DAYS}")
            mean_car = cell.mean() * REFERENCE_OCCUPANCY
            std_car = np.sqrt(cell.var() + phi * cell.mean()) * REFERENCE_OCCUPANCY
            rows.append({
                "season": season,
                "weather": weather,
                "season_freq_rel": weights[season],
                "weather_freq_rel": climatology.loc[season, weather],
                "mean_car_visitors": mean_car,
                "std_car_visitors": std_car,
                "mean_other_visitors": mean_car * other_ratio,
                "std_other_visitors": mean_car * other_ratio * OTHER_STD_REL,
                "n_days": len(cell),
            })
    stats = pd.DataFrame(rows)
    years = sorted(days["year"].unique())
    stats.insert(0, "calibration", f"traffic_{years[0]}_{years[-1]}_relation_{CALIBRATION_YEAR}")
    return stats


def build_all(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    daily = build_fazzon_daily(tables)
    stats = build_presence_stats(daily)
    _check(daily, stats)
    from model.fazzon.modal_shift import estimate_modal_shift  # step 2b

    modal_shift, modal_shift_days = estimate_modal_shift(daily, level_presence_stats(daily))
    from model.fazzon.monitoring import build_monitoring  # step 2c

    return {
        "fazzon_daily": daily,
        "fazzon_presence_stats": stats,
        "fazzon_modal_shift": modal_shift,
        "fazzon_modal_shift_daily": modal_shift_days,
        "fazzon_monitoring": build_monitoring(daily),
    }


def _check(daily: pd.DataFrame, stats: pd.DataFrame) -> None:
    if daily["date"].duplicated().any():
        raise ValueError("fazzon_daily: duplicated dates")
    seasons = stats.drop_duplicates("season")
    if not np.isclose(seasons["season_freq_rel"].sum(), 1.0):
        raise ValueError("season_freq_rel does not sum to 1")
    by_season = stats.groupby("season")["weather_freq_rel"].sum()
    if not np.allclose(by_season, 1.0) or (stats["weather_freq_rel"] <= 0).any():
        raise ValueError(f"weather_freq_rel must be > 0 and sum to 1 per season: {by_season.to_dict()}")
    for year, n in daily.groupby("year")["cars"].count().items():
        if n:
            logger.info("fazzon_daily: %d days with Fazzon car counts in %d", n, year)
