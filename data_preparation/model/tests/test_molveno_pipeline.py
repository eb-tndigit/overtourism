import numpy as np
import pandas as pd

from model.molveno.config import load_config
from model.molveno.presence_estimators import (
    TRAINING_YEARS,
    WEATHER_CLASSES,
    WEEKDAYS,
    add_calendar,
    add_weather,
    additive_presence_stats,
    gbm_presence_stats,
    training_days,
    weather_climatology,
    weather_frequencies,
)
import zipfile

import pytest

from model.molveno.reference_datasets import build_molveno_counts_daily, build_molveno_daily
from model.molveno.standardize import (
    _seats,
    _tariff_hours,
    standardize_beach_counts,
    standardize_parking_paid,
    standardize_road_counts,
    standardize_vodafone,
)


def test_config_lists_all_sources():
    assert {"vodafone_attendences", "vodafone_attendences_str", "meteo_station", "road_counts", "beach_counts",
            "parking_paid", "ispat_daily", "beds", "beds_2025", "restaurants", "parking_lots"} <= set(load_config().sources)


def _vodafone_csv(path, rows, columns):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def test_standardize_vodafone_is_italians_plus_foreign_total(tmp_path):
    cols = ["date", "locType", "locId", "userCountry", "userProfile", "value"]
    att = _vodafone_csv(tmp_path / "att.csv", [
        ("2023-07-01", "TN_MKT_AL_3", "27", "ITA", "TOURIST", 100),
        ("2023-07-01", "TN_MKT_AL_3", "27", "DEU", "TOURIST", 20),  # single countries: not used
        ("2023-07-01", "TN_MKT_AL_3", "27", "ITA", "VISITOR", 50),
        ("2023-07-01", "TN_MKT_AL_3", "27", "ITA", "INHABITANT", 900),
        ("2023-07-01", "TN_MKT_AL_3", "8", "ITA", "TOURIST", 999),
        ("2023-07-01", "AL_2", "27", "ITA", "TOURIST", 999),
        ("2023-07-02", "TN_MKT_AL_3", "27", "ITA", "TOURIST", 10),
        ("2023-07-02", "TN_MKT_AL_3", "27", "ITA", "VISITOR", 5),
    ], cols)
    foreign = _vodafone_csv(tmp_path / "str.csv", [
        ("2023-07-01", "TN_MKT_AL_3", "27", "STR", "TOURIST", 70),
        ("2023-07-01", "TN_MKT_AL_3", "27", "STR", "VISITOR", 30),
    ], cols)
    out = standardize_vodafone(att, foreign).set_index("date")
    assert out.loc["2023-07-01"].tolist() == [170, 80]
    assert out.loc["2023-07-02"].tolist() == [10, 5]  # no STR row: no foreigners


def test_summer_day_ignores_leap_years():
    d = add_calendar(pd.DataFrame({"date": pd.to_datetime(["2024-06-01", "2025-06-01", "2024-08-15", "2025-08-15"])}))
    assert d["summer_day"].tolist() == [0, 0, 75, 75]
    assert d["holiday"].tolist() == [0, 0, 1, 1]


def test_weather_classes():
    meteo = pd.DataFrame({"date": pd.to_datetime(["2023-07-01", "2023-07-02", "2023-07-03"]),
                          "tmax": [30.0, 24.0, 20.0], "rain_mm": [0.0, 0.0, 12.0]})
    clim = pd.Series({7: 27.0})
    out = add_weather(meteo, clim)
    assert out["weather"].tolist() == ["good", "cool", "rainy"]


def _synthetic(seed=0):
    """Summer days of the training years: higher in August, on weekends; rain lowers excursionists."""
    rng = np.random.default_rng(seed)
    dates = pd.concat([pd.Series(pd.date_range(f"{y}-06-01", f"{y}-09-30")) for y in TRAINING_YEARS])
    n = len(dates)
    rain = np.where(rng.random(n) < 0.2, 15.0, 0.0)
    tmax = 25 + rng.normal(0, 2, n)
    peak = np.where(dates.dt.month == 8, 2000.0, 0.0)
    weekend = np.where(dates.dt.dayofweek >= 5, 800.0, 0.0)
    vodafone = pd.DataFrame({"date": dates.values,
                             "tourists": 2000 + peak + 0.3 * weekend + rng.normal(0, 100, n),
                             "excursionists": 1500 + peak + weekend - 60 * rain + rng.normal(0, 100, n)})
    meteo = pd.DataFrame({"date": dates.values, "tmax": tmax, "rain_mm": rain})
    return build_molveno_daily(vodafone, meteo)


def test_gbm_presence_stats_cover_every_cell_and_learn_the_effects():
    stats = gbm_presence_stats(training_days(_synthetic())).set_index(["month_day", "weekday", "weather"])
    assert len(stats) == 122 * 7 * len(WEATHER_CLASSES)
    assert (stats[["std_tourists", "std_excursionists"]] > 0).all().all()
    aug, jun = stats.loc[("08-10", "wednesday", "good")], stats.loc[("06-10", "wednesday", "good")]
    assert aug["mean_tourists"] > jun["mean_tourists"] + 1000
    sunday, rainy = stats.loc[("08-10", "sunday", "good")], stats.loc[("08-10", "wednesday", "rainy")]
    assert sunday["mean_excursionists"] > aug["mean_excursionists"] + 400
    assert rainy["mean_excursionists"] < aug["mean_excursionists"] - 400


def test_additive_presence_stats_cells():
    stats = additive_presence_stats(training_days(_synthetic()))
    assert len(stats) == 4 * len(WEEKDAYS) * len(WEATHER_CLASSES)
    assert stats["std_tourists"].nunique() == 1  # one residual SD


def test_weather_frequencies_sum_to_one_per_month():
    freq = weather_frequencies(_synthetic())
    assert np.allclose(freq.groupby("month")["freq_rel"].sum(), 1.0)
    assert set(freq["weather"]) == set(WEATHER_CLASSES)


def test_climatology_uses_the_given_years():
    meteo = pd.DataFrame({"date": pd.to_datetime(["2023-07-01", "2026-07-01"]), "tmax": [20.0, 40.0]})
    assert weather_climatology(meteo, (2023,)).loc[7] == 20.0


def _zip(path, members: dict[str, pd.DataFrame]):
    with zipfile.ZipFile(path, "w") as z:
        for name, table in members.items():
            z.writestr(name, table.to_csv(index=False))
    return path


def test_road_counts_daily_cars_and_complete_hours(tmp_path):
    hours = pd.date_range("2023-07-01", periods=25, freq="h")  # 24 hours of 1/7, 1 of 2/7
    rows = [(h, "M40", 1, "MOLVENO", 10, 2, 0, 0, 0, 0, 0, 1) for h in hours]
    rows += [(hours[0], "M40", 2, "MOLVENO", 7, 0, 0, 0, 0, 0, 0, 0)]
    cols = ["data", "sensore", "direzione", "comune", "cl1", "cl2", "cl3", "cl4", "cl5", "cl6", "cl7", "nonrilevato"]
    path = _zip(tmp_path / "c.zip", {"contamezzi_Molveno.csv": pd.DataFrame(rows, columns=cols)})
    out = standardize_road_counts(path).set_index(["sensor", "direction", "date"])
    day = out.loc[("M40", 1, pd.Timestamp("2023-07-01"))]
    assert (day["cars"], day["vehicles"], day["hours"], day["flow"]) == (240, 312, 24, "in_from_andalo")
    assert out.loc[("M40", 1, pd.Timestamp("2023-07-02")), "hours"] == 1
    assert out.loc[("M40", 2, pd.Timestamp("2023-07-01")), "flow"] == ""  # leaving Molveno


def test_parking_tariffs_and_total_check(tmp_path):
    assert _tariff_hours("20 MIN") == pytest.approx(1 / 3)
    assert _tariff_hours("1 ORA E 40 MIN") == pytest.approx(5 / 3)
    assert _tariff_hours("3 GIORNI") == 72
    sheet = pd.DataFrame([[None, pd.Timestamp("2023-07-01"), pd.Timestamp("2023-07-02")],
                          ["1 ORA", 3, None], ["2 ORE", 1, 4], ["TOTALE", 4, 4]])
    sheet.to_excel(tmp_path / "p.xlsx", header=False, index=False)
    out = standardize_parking_paid(tmp_path / "p.xlsx")
    assert out.groupby("date")["vehicles"].sum().tolist() == [4, 4]
    sheet.iloc[3, 1] = 5
    sheet.to_excel(tmp_path / "p.xlsx", header=False, index=False)
    with pytest.raises(ValueError):
        standardize_parking_paid(tmp_path / "p.xlsx")


def test_seats_ranges():
    assert _seats(150) == (150, 150)
    assert _seats("45/50") == (45, 50)
    assert _seats("120 e poco più") == (120, 120)


def test_counts_daily_needs_complete_days(tmp_path):
    gates = [f"Varco {i}" for i in range(1, 10)]
    presences = pd.DataFrame([("2023-07-01", g, 10) for g in gates] + [("2023-07-02", "Varco 1", 10)],
                             columns=["data", "varco", "presenze"])
    passages = presences.iloc[:0]
    beach = standardize_beach_counts(_zip(tmp_path / "b.zip", {"contapersone_presenze_Molveno.csv": presences,
                                                              "contapersone_passaggi_Molveno.csv": passages}))
    road = pd.DataFrame({"sensor": ["M40", "M42", "M40", "M42"], "direction": [1, 2, 1, 2],
                         "date": pd.to_datetime(["2023-07-01"] * 2 + ["2023-07-02"] * 2),
                         "cars": [100, 50, 80, 40], "vehicles": [0] * 4, "hours": [24, 24, 24, 23],
                         "flow": ["in_from_andalo", "in_from_san_lorenzo"] * 2})
    parking = pd.DataFrame({"date": pd.to_datetime(["2023-07-01"]), "tariff": ["2 ORE"], "hours": [2.0],
                            "vehicles": [5]})
    ispat = pd.DataFrame({"date": pd.to_datetime(["2023-07-01", "2023-07-02"]), "arrivals": [1, 2],
                          "overnight_stays": [10, 20]})
    daily = pd.DataFrame({"date": pd.to_datetime(["2023-07-01", "2023-07-02"]), "tourists": [1, 2]})
    out = build_molveno_counts_daily(daily, {"road_counts_daily": road, "beach_counts": beach,
                                             "parking_paid": parking, "ispat_daily": ispat}).set_index("date")
    assert out.loc["2023-07-01", "cars_into_molveno"] == 150
    assert np.isnan(out.loc["2023-07-02", "cars_into_molveno"])  # 23 hours on one road
    assert out.loc["2023-07-01", "beach_presences"] == 90
    assert np.isnan(out.loc["2023-07-02", "beach_presences"])  # one gate only
    assert out.loc["2023-07-01", "paid_vehicle_hours"] == 10
    assert out.loc["2023-07-02", "ispat_overnight_stays"] == 20
