import pandas as pd
import pytest

from utils.config import BASE_WINDOW
from utils.datasets import ARRIVI_VALUE_COLS, phenomena_to_recompute
from utils.provenienza import comune_hhi, normalized_hhi
from utils.steps.create_phenomena_df import compute_provenienza
from utils.steps.process_std_data import process_provenienza, process_provenienza_arrivi


def _hhi(values):
    df = pd.DataFrame({"key": "a", "value": values})
    return float(normalized_hhi(df)["a"])


def test_normalized_hhi_extremes_and_example():
    assert _hhi([5, 5, 5, 5]) == pytest.approx(0.0)  # evenly spread
    assert _hhi([100]) == 1.0  # a single market
    assert _hhi([70, 30]) == pytest.approx((0.58 - 0.5) / 0.5)
    assert 0 < _hhi([90, 5, 5]) < 1


def _rows(area, year, values, days=("-01-10", "-01-11")):
    return [
        {"DATA": f"{year}{d}", "areaId": area, "userCountry": c, "value": v}
        for d in days
        for c, v in values.items()
    ]


def test_comune_hhi_threshold_split_areas_and_weights():
    df = pd.DataFrame(
        _rows("1", 2023, {"ITA": 70, "DEU": 30})  # area 1: 200 user-days
        + _rows("2", 2023, {"ITA": 50, "DEU": 50})  # area 2: 200 user-days, hhi 0
        + _rows("3", 2023, {"ITA": 20, "DEU": 10}, days=("-01-10",))  # 30 user-days: below the threshold
    )
    ac = pd.DataFrame(
        {"areaId": ["1", "2", "3", "2"], "ID_COMUNE": ["022001", "022001", "022003", "022002"]}
    )
    out = comune_hhi(df, ac, min_presenze=100).set_index("ID_COMUNE")
    assert out.loc["022001", "hhi_provenienza"] == pytest.approx(0.16 * 0.5)  # mean weighted by user-days
    assert out.loc["022002", "hhi_provenienza"] == pytest.approx(0.0)  # an area passes its value to all its comuni
    assert "022003" not in out.index  # too few presences
    assert out["DATA"].unique().tolist() == [2023]


def _std(country_values, **kw):
    rows = [
        {
            "DATA": "2023-05-01",
            "locId": "7",
            "locType": "TN_MKT_AL_3",
            "userProfile": kw.get("profile", "TOURIST"),
            "userCountry": c,
            "presenze": v,
        }
        for c, v in country_values.items()
    ]
    return pd.DataFrame(rows)


def test_process_provenienza_only_tourists_and_areas_level():
    ac = pd.DataFrame({"areaId": ["7"], "ID_COMUNE": ["022007"]})
    std = pd.concat(
        [
            _std({"ITA": 600, "DEU": 400}),
            _std({"FRA": 1000}, profile="VISITOR"),  # ignored
            _std({"ITA": 600, "DEU": 400}).assign(locType="TN_MKT_AM_22"),  # ignored
        ]
    )
    out = process_provenienza(std, ac, BASE_WINDOW)
    assert out["hhi_provenienza"].iloc[0] == pytest.approx((0.52 - 0.5) / 0.5)
    assert list(out.columns) == ["DATA", "ID_COMUNE", "hhi_provenienza", "giorni_coperti"]


def test_process_provenienza_needs_country():
    with pytest.raises(ValueError, match="userCountry"):
        process_provenienza(
            _std({"ITA": 1}).drop(columns="userCountry"),
            pd.DataFrame({"areaId": ["7"], "ID_COMUNE": ["022007"]}),
        )


def test_arrivals_count_countryless_rows_and_report_top_country_shares():
    rows = [
        {
            "DATA": "2023-05-01",
            "locId": "7",
            "locType": "TN_MKT_AL_3",
            "userProfile": "TOURIST",
            "userCountry": country,
            "arrivi": value,
        }
        for country, value in [
            ("ITA", 50),
            ("DEU", 30),
            ("FRA", 10),
            (None, 10),
        ]
    ]
    out = process_provenienza_arrivi(
        pd.DataFrame(rows),
        pd.DataFrame({"areaId": ["7"], "ID_COMUNE": ["022007"]}),
    )

    row = out.iloc[0]
    assert row["volume_totale"] == 100
    assert [row[f"nome_mercato_{rank}"] for rank in range(1, 4)] == [
        "Italia",
        "Germania",
        "Francia",
    ]
    assert [row[f"volume_mercato_{rank}"] for rank in range(1, 4)] == [50, 30, 10]
    assert list(out.columns) == ["DATA", "ID_COMUNE", *ARRIVI_VALUE_COLS]
    phenomenon = compute_provenienza(
        pd.DataFrame(
            {
                "DATA": [2023],
                "ID_COMUNE": ["022007"],
                "hhi_provenienza": [0.2],
                "giorni_coperti": [1],
            }
        ),
        out,
    )
    assert list(phenomenon.columns) == [
        "DATA",
        "ID_COMUNE",
        "hhi_provenienza",
        "giorni_coperti",
        *ARRIVI_VALUE_COLS,
    ]
    assert phenomenon.loc[0, "volume_mercato_1"] == 50


def test_arrivals_without_country_column_still_count_in_total():
    df = pd.DataFrame(
        [
            {
                "DATA": "2023-05-01",
                "locId": "7",
                "locType": "TN_MKT_AL_3",
                "userProfile": "TOURIST",
                "arrivi": 17,
            }
        ]
    )
    out = process_provenienza_arrivi(df, pd.DataFrame({"areaId": ["7"], "ID_COMUNE": ["022007"]}))

    assert out["volume_totale"].tolist() == [17]
    assert out[[f"nome_mercato_{rank}" for rank in range(1, 4)]].isna().all().all()
    assert out[[f"volume_mercato_{rank}" for rank in range(1, 4)]].isna().all().all()


def test_phenomenon_needs_only_vodafone_provenienza():
    assert "phen_provenienza" in phenomena_to_recompute({"vodafone_provenienza"})
