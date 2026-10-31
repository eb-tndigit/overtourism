import numpy as np
import pandas as pd
import pytest

from model.fazzon.config import load_config
from model.fazzon.reference_datasets import (
    REFERENCE_OCCUPANCY,
    build_fazzon_daily,
    build_presence_stats,
    weather_factors,
)
from model.fazzon.standardize import (
    road_counts_daily,
    standardize_greenbus_2025,
    standardize_parking,
    standardize_road_counts,
)


def test_config_lists_all_sources():
    config = load_config()
    assert {"parking", "road_counts", "vodafone_attendences", "meteo_station", "meteo_bulletin"} <= set(config.sources)
    assert isinstance(config.sources["road_counts"], tuple)  # one entry per delivery


def _road_csv(path, rows):
    """rows: (station, date, direction, cars per hour) -> PAT file with one car row per hour."""
    lines = ['"PUNTO","DATA","ORA","DIREZIONE","CLASSE","VELOCITA","N_VEICOLI"']
    for station, date, direction, cars in rows:
        lines += [f'"{station}","{date}","{hour:02d}",{direction},2,5,{cars}' for hour in range(24)]
    path.write_text("\n".join(lines))
    return path


_ROAD_STATIONS = pd.DataFrame({"kind": ["road", "road"], "code": ["110", "155"], "name": ["pellizzano", "pracorno"]})


def test_road_counts_later_delivery_replaces_earlier(tmp_path):
    first = _road_csv(tmp_path / "a.csv", [("110", "01/08/2026", 2, 10), ("110", "02/08/2026", 2, 10),
                                           ("155", "01/08/2026", 2, 7)])
    second = _road_csv(tmp_path / "b.csv", [("110", "02/08/2026", 2, 20)])
    hourly = standardize_road_counts([first, second], _ROAD_STATIONS)
    cars = hourly.groupby(["station", "date"])["cars"].sum()
    assert cars[("pellizzano", pd.Timestamp("2026-08-01"))] == 240  # only in the first delivery
    assert cars[("pellizzano", pd.Timestamp("2026-08-02"))] == 480  # replaced, not added
    assert cars[("pracorno", pd.Timestamp("2026-08-01"))] == 168


def test_road_counts_undercounting_direction_is_invalid(tmp_path):
    path = _road_csv(tmp_path / "r.csv", [("155", "01/08/2026", 1, 3), ("155", "01/08/2026", 2, 60),
                                          ("110", "01/08/2026", 1, 50), ("110", "01/08/2026", 2, 70)])
    daily = road_counts_daily(standardize_road_counts(path, _ROAD_STATIONS)).set_index(["station", "direction"])
    assert not daily.loc[("pracorno", 1), "valid"]  # 72 cars, above the minimum, but 1/20 of direction 2
    assert daily.loc[("pracorno", 2), "valid"]
    assert daily.loc[("pellizzano", 1), "valid"] and daily.loc[("pellizzano", 2), "valid"]


def _parking_sheet(tmp_path):
    header = [[None] * 18, [None] * 18, ["DATA"] + [None] * 17]
    rows = [
        # 2025 block (8 cols) | 2026 block (8 cols) | note | note2
        ["2025-07-02", 99, None, None, 53, 0, None, None,
         "2026-07-01", 47, None, None, None, 0, 0, None, None, None],
        ["2025-07-03", 176, None, None, 38, 0, None, None,
         "2026-07-02", "5days", None, None, None, None, None, None, "5days - evento", None],
        ["2025-08-12", 422, None, None, None, 0, None, None,
         "2026-08-11", 73, None, 77, None, 428, 16, None, "487 (discesa)", "Brutto tempo"],
    ]
    path = tmp_path / "parking.xlsx"
    pd.DataFrame(header + rows).to_excel(path, header=False, index=False)
    return path


def test_standardize_parking_reads_both_years_and_bus_down(tmp_path):
    out = standardize_parking(_parking_sheet(tmp_path)).set_index("date")
    assert out.loc["2025-07-02", "cars_full"] == 99
    assert out.loc["2025-07-02", "bus_up"] == 53
    assert np.isnan(out.loc["2026-07-02", "cars_full"])  # "5days" is not a count
    assert out.loc["2026-07-02", "note"] == "5days - evento"
    assert out.loc["2026-08-11", "cars_half"] == 77
    assert out.loc["2026-08-11", "bus_down"] == 487
    assert out.loc["2026-08-11", "note"] == "Brutto tempo"


def test_greenbus_rows_after_pilot_are_flagged(tmp_path):
    sheet = pd.DataFrame([
        [None, "RIEPILOGO", None, None, None, None],
        [None, "Data", "Ticket Venduti", None, "Data", "Ticket Venduti"],
        [None, "2025-07-20", 90, None, None, None],
        [None, "2025-08-06", 90, None, None, None],
    ])
    path = tmp_path / "bus.xlsx"
    sheet.to_excel(path, header=False, index=False)
    out = standardize_greenbus_2025(path).set_index("date")
    assert not out.loc["2025-07-20", "other_area"]
    assert out.loc["2025-08-06", "other_area"]


def test_road_counts_keep_cars_only_and_flag_dead_sensors(tmp_path):
    lines = ['"PUNTO","DATA","ORA","DIREZIONE","CLASSE","VELOCITA","N_VEICOLI"']
    for hour in range(24):
        lines.append(f'"110","01/08/2025","{hour:02d}",2,2,5,10')   # cars
        lines.append(f'"110","01/08/2025","{hour:02d}",2,2,6,5')    # cars, another speed band
        lines.append(f'"110","01/08/2025","{hour:02d}",2,9,5,1')    # bus: excluded
        lines.append(f'"110","01/08/2025","{hour:02d}",1,2,5,0')    # direction 1 reads zero
    path = tmp_path / "road.csv"
    path.write_text("\n".join(lines))
    stations = pd.DataFrame({"kind": ["road"], "code": ["110"], "name": ["pellizzano"]})
    hourly = standardize_road_counts(path, stations)
    assert set(hourly["station"]) == {"pellizzano"}
    assert hourly.loc[hourly["direction"] == 2, "cars"].eq(15).all()
    daily = road_counts_daily(hourly).set_index("direction")
    assert daily.loc[2, "cars"] == 24 * 15 and daily.loc[2, "valid"]
    assert not daily.loc[1, "valid"]


def _tables_for_daily(cars_by_date: dict[str, float], rain_by_date: dict[str, float] | None = None,
                      tmax_by_date: dict[str, float] | None = None) -> dict[str, pd.DataFrame]:
    dates = pd.to_datetime(list(cars_by_date))
    rain = [(rain_by_date or {}).get(d, 0.0) for d in cars_by_date]
    tmax = [(tmax_by_date or {}).get(d, 25.0) for d in cars_by_date]
    parking = pd.DataFrame({"date": dates, "cars_full": list(cars_by_date.values()), "cars_half": np.nan,
                            "moto_persons": 0.0, "bus_up": np.nan, "bus_down": np.nan, "note": pd.NA})
    empty_dates = pd.DatetimeIndex([], name="date")
    hourly = pd.DataFrame({"station": pd.Series(dtype=str), "date": empty_dates, "direction": pd.Series(dtype=int),
                           "hour": pd.Series(dtype=int), "cars": pd.Series(dtype=int)})
    daily_road = pd.DataFrame({
        "station": [s for s in ("pellizzano", "pracorno", "celledizzo", "tonale", "revo") for _ in (1, 2)],
        "date": dates[0], "direction": [1, 2] * 5, "cars": 1000, "valid": True,
    })
    return {
        "fazzon_parking": parking,
        "fazzon_parking_2025": parking[["date"]].assign(cars=parking["cars_full"]),
        "fazzon_greenbus_2025": pd.DataFrame({"date": dates[:1], "tickets": [9.0], "other_area": [False]}),
        "meteo_station": pd.DataFrame({"date": dates, "tmin": 10.0, "tmax": tmax, "rain_mm": rain, "quality": 145}),
        "meteo_bulletin": pd.DataFrame({"date": dates, "weather_desc": "Sereno", "tmin": 10, "tmax": 25, "rain_mm": 0}),
        "road_counts_hourly": hourly,
        "road_counts_daily": daily_road,
        "vodafone_presences": pd.DataFrame({"date": dates, "area": "apt_valdisole", "presences": 100.0}),
        "bike_counts": pd.DataFrame({"date": dates, "sensor": "TFXV3R06", "bikes_peds": 50}),
        "stations": pd.DataFrame({"kind": ["bike"], "code": ["TFXV3R06"], "role": ["near_fazzon"]}),
        "tgc_accessi": pd.DataFrame({"date": dates, "service": "Passo Tonale", "entrants": 3}),
    }


def test_weather_factors_recover_multiplicative_effect():
    cal = pd.DataFrame({
        "season": ["july"] * 4 + ["august"] * 4,
        "weather": ["good", "good", "rainy", "rainy"] * 2,
        "cars": [200.0, 200.0, 100.0, 100.0, 400.0, 400.0, 200.0, 200.0],
    })
    level, factor = weather_factors(cal)
    assert level["july"] == pytest.approx(200.0) and level["august"] == pytest.approx(400.0)
    assert factor["good"] == 1.0 and factor["rainy"] == pytest.approx(0.5)


def test_presence_stats_per_season_and_weather():
    days = pd.date_range("2025-07-01", "2025-08-31")
    cars = {str(d.date()): (300.0 if d.month == 8 else 150.0) for d in days}
    rain = {str(d.date()): 20.0 for d in days if d.day % 4 == 0}  # rainy days: half the cars
    tmax = {str(d.date()): 15.0 for d in days if d.day % 4 == 1}  # cool days: 80 % of the cars
    cars.update({k: v / 2 for k, v in cars.items() if k in rain})
    cars.update({k: v * 0.8 for k, v in cars.items() if k in tmax})
    cars["2025-08-28"] = 27.0  # D4: bad-weather days are part of the calibration
    rain["2025-08-28"] = 34.0
    daily = build_fazzon_daily(_tables_for_daily(cars, rain, tmax))
    stats = build_presence_stats(daily).set_index(["season", "weather"])
    assert set(stats.index.get_level_values("season")) == {"july", "august"}
    assert set(stats.index.get_level_values("weather")) == {"good", "cool", "rainy"}
    assert stats.loc[("august", "good"), "mean_car_visitors"] == pytest.approx(300.0 * REFERENCE_OCCUPANCY, rel=0.05)
    assert stats.loc[("august", "rainy"), "weather_factor"] == pytest.approx(0.5, rel=0.1)
    assert stats.loc[("july", "cool"), "weather_factor"] == pytest.approx(0.8, rel=0.1)
    assert (stats["weather_freq_rel"] > 0).all()
    seasons = stats.reset_index().drop_duplicates("season")
    assert seasons["season_freq_rel"].sum() == pytest.approx(1.0)
    assert stats.groupby("season")["weather_freq_rel"].sum().tolist() == pytest.approx([1.0, 1.0])


def test_presence_stats_fail_without_days_of_a_weather_class():
    cars = {str(d.date()): 100.0 for d in pd.date_range("2025-07-01", "2025-07-31")}
    daily = build_fazzon_daily(_tables_for_daily(cars))  # all days good: no cool / rainy days
    with pytest.raises(ValueError, match="weather classes"):
        build_presence_stats(daily)


def test_parking_price_per_day():
    days = ["2025-07-10", "2026-06-30", "2026-07-10", "2026-08-21", "2026-09-05"]
    daily = build_fazzon_daily(_tables_for_daily({d: 100.0 for d in days})).set_index("date")
    price = daily.loc[pd.to_datetime(days), ["parking_price", "parking_price_afternoon", "parking_price_assumed"]]
    assert price["parking_price"].tolist() == [5.0, 5.0, 20.0, 5.0, 5.0]
    assert price["parking_price_afternoon"].tolist() == [5.0, 5.0, 10.0, 5.0, 5.0]
    assert price["parking_price_assumed"].tolist() == [False, True, False, False, True]
    assert np.isnan(daily.loc["2026-07-11", "parking_price"])  # no parking data that day


def test_modal_shift_recovers_known_response():
    from model.fazzon.modal_shift import estimate_modal_shift

    stats = pd.DataFrame({  # season x weather levels: only kept for comparison (A21)
        "season": ["june", "july", "august", "september"], "weather": ["good"] * 4,
        "mean_car_visitors": [100 * REFERENCE_OCCUPANCY] * 4,
    })
    beta = 0.1
    rows = []
    for i, day in enumerate(pd.date_range("2025-06-01", "2025-09-30")):  # calibration: cars = 0.1 x traffic
        traffic = 1000.0 + 20 * (i % 50)
        rows.append({"date": day, "year": 2025, "month": day.month, "traffic": traffic, "cars": 0.1 * traffic,
                     "parking_price": 5.0, "regulated_parking": False, "shuttle": None})
    for i, day in enumerate(pd.date_range("2026-07-04", "2026-08-30")):  # rules: 20 euro, shuttle
        traffic = 1500.0 + 30 * (i % 20)
        rows.append({"date": day, "year": 2026, "month": day.month, "traffic": traffic,
                     "cars": 0.1 * traffic * np.exp(-beta * 15), "parking_price": 20.0, "regulated_parking": True,
                     "shuttle": "lakebus", "bus_up": 50.0, "bus_down": 60.0})
    for i, day in enumerate(pd.date_range("2026-09-01", "2026-09-10")):  # 5 euro, no rules: 1.2 x the relation
        traffic = 1000.0 + 40 * i
        rows.append({"date": day, "year": 2026, "month": 9, "traffic": traffic, "cars": 0.12 * traffic,
                     "parking_price": 5.0, "regulated_parking": False, "shuttle": None})
    daily = pd.DataFrame(rows).rename(columns={"traffic": "strada_pellizzano_salita_mattina"})
    daily = daily.assign(cars_full=daily["cars"], cars_half=np.nan, parking_price_afternoon=daily["parking_price"],
                         weather="good", event_day=False, moto_persons=0.0, parking_price_assumed=False)
    daily.loc[daily["shuttle"].isna(), ["bus_up", "bus_down"]] = np.nan
    daily.loc[daily["regulated_parking"], "parking_price_afternoon"] = 10.0
    estimates, days = estimate_modal_shift(daily, stats)
    e = estimates.set_index(["parameter", "days"])["estimate"]
    assert e[("demand_per_morning_car", "")] == pytest.approx(0.1) and e[("demand_intercept", "")] == pytest.approx(0, abs=1e-6)
    assert e[("year_effect_observed", "")] == pytest.approx(1.2)  # reported, not used (A22)
    assert e[("year_effect", "")] == 1.0
    assert e[("car_response", "")] == pytest.approx(np.exp(-beta * 15))
    assert e[("price_paid", "")] == pytest.approx(20.0)
    # visitors: 3.5 persons per car at 20 euro vs 2.5 at 5 euro (A2)
    assert e[("visitor_response", "")] == pytest.approx(np.exp(-beta * 15) * 3.5 / 2.5)
    assert e[("share_walk_up", "with_walk_up")] > 0
    assert set(days["role"]) == {"regime", "year_effect"}
    assert e[("occupancy_high", "")] == 3.5 and e[("reference_price", "")] == 5.0
    low, high = estimates.set_index("parameter").loc["year_effect", ["low", "high"]]
    assert 0.9 <= low < 1.0 < high <= 1.1


def test_occupancy_interpolates_between_low_and_high_price():
    from model.fazzon.reference_datasets import occupancy

    assert occupancy([0.0, 5.0, 10.0, 20.0, 40.0]).tolist() == pytest.approx([2.5, 2.5, 2.5 + 1 / 3, 3.5, 3.5])


def test_monitoring_compares_every_year_with_the_calibration_year():
    from model.fazzon.monitoring import build_monitoring

    dates = pd.date_range("2025-06-01", "2025-09-30").append(pd.date_range("2026-06-01", "2026-09-30"))
    daily = pd.DataFrame({"date": dates, "year": dates.year, "month": dates.month, "weather": "good"})
    regime = (daily["year"] == 2026) & daily["date"].between("2026-07-04", "2026-08-31")
    daily["strada_pellizzano_salita_mattina"] = np.where(daily["year"] == 2026, 1100.0, 1000.0)
    daily["cars"] = np.where(regime, 33.0, daily["strada_pellizzano_salita_mattina"] * 0.1)
    for station in ("pellizzano", "pracorno", "celledizzo", "tonale", "revo"):
        daily[f"strada_{station}_totale"] = np.nan
    m = build_monitoring(daily).set_index(["indicator", "year", "period"])
    up = m.loc[("road_pellizzano_morning_up", 2026, "regime_window")]
    assert up["ratio_to_reference"] == pytest.approx(1.1) and up["regime_in_force"]
    ratio = m.loc[("fazzon_cars_per_morning_car", 2026, "regime_window"), "ratio_to_reference"]
    assert ratio == pytest.approx(0.03 / 0.1)
    assert m.loc[("fazzon_cars_per_morning_car", 2026, "rest_of_season"), "ratio_to_reference"] == pytest.approx(1.0)
    assert m.loc[("fazzon_cars_per_morning_car", 2025, "regime_window"), "ratio_to_reference"] == pytest.approx(1.0)
    assert "road_tonale" not in m.index.get_level_values("indicator")  # no data, no row
    # 2025: cars = 0.1 x traffic, so the relation (A37) predicts the 2026 days outside the regime exactly
    assert m.loc[("fazzon_cars_vs_calibration_relation", 2026, "rest_of_season"), "value"] == pytest.approx(1.0)
