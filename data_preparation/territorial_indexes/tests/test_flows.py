import json

import numpy as np
import pandas as pd

from utils.datasets import COVISITS_METRICS, COVISITS_PROFILE_WEIGHTS, COVISITS_PROFILES
from utils.flows import AREA_COMUNE_OVERRIDES, build_area_comune, comune_flow_indicators

# every comune named by AREA_COMUNE_OVERRIDES must be in the mapping
MAPPING_COMUNI = {
    name: 22000 + k
    for k, name in enumerate(sorted(set(AREA_COMUNE_OVERRIDES.values())))
}
MAPPING_COMUNI["TRENTO"] = 22205


def _geojson(tmp_path, areas):
    gj = {
        "features": [
            {"properties": {"id": i, "name": name}} for i, name in areas.items()
        ]
    }
    path = tmp_path / "areas.geojson"
    path.write_text(json.dumps(gj), encoding="utf-8")
    return path


def test_area_with_several_comuni_keeps_all_of_them(tmp_path):
    path = _geojson(tmp_path, {1: "AREA UNO", 2: "AREA DUE", 3: "AREA TRE"})
    mapping_vodafone = {
        "AREA UNO": [10001, 10002, 10003],  # one area, three comuni
        "AREA DUE": [20001],
        "AREA TRE": [30001, 30002],
    }
    ac = build_area_comune(path, mapping_vodafone, MAPPING_COMUNI)

    by_area = ac.groupby("areaId")["ID_COMUNE"].apply(sorted).to_dict()
    assert by_area["1"] == ["010001", "010002", "010003"]
    assert by_area["2"] == ["020001"]
    assert by_area["3"] == ["030001", "030002"]
    # network-only areas resolved by name (STAZ_TREN -> Trento) are added without touching the others
    assert by_area["STAZ_TREN"] == ["022205"]
    assert not ac.duplicated().any()


def test_override_replaces_the_geojson_rows_of_the_area(tmp_path):
    path = _geojson(tmp_path, {"STAZ_TREN": "STAZIONE"})
    ac = build_area_comune(path, {"STAZIONE": [1, 2]}, MAPPING_COMUNI)
    assert ac[ac["areaId"] == "STAZ_TREN"]["ID_COMUNE"].tolist() == ["022205"]


def _od(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "DATA",
            "sourceId",
            "sourceAmbito",
            "targetId",
            "targetAmbito",
            "covisitType",
            "userProfile",
            "value",
        ],
    )


def test_every_comune_of_a_multi_comune_area_gets_the_indicators_and_combined_uses_half_weight():
    rows = []
    # areas A (two comuni) and B (one comune); tourists and visitors, same ambito
    for prof, t_a, t_b, ud_ab in (("TOURIST", 1000, 600, 100), ("VISITOR", 400, 300, 200)):
        rows += [
            ("2024-01-01", "*", "X", "A", "X", "ND", prof, t_a),
            ("2024-01-01", "*", "X", "B", "X", "ND", prof, t_b),
            ("2024-01-01", "*", "X", "A", "X", "UD", prof, 50),
            ("2024-01-01", "A", "X", "*", "X", "UD", prof, ud_ab),
            ("2024-01-01", "A", "X", "B", "X", "UD", prof, ud_ab),
            ("2024-01-01", "B", "X", "A", "X", "UD", prof, 30),
        ]
    ac = pd.DataFrame(
        {"areaId": ["A", "A", "B"], "ID_COMUNE": ["000001", "000002", "000003"]}
    )
    out = comune_flow_indicators(
        _od(rows),
        ac,
        min_t=100,
        damping=0.85,
        profiles=COVISITS_PROFILES,
        metrics=COVISITS_METRICS,
        weights=COVISITS_PROFILE_WEIGHTS,
    )
    assert COVISITS_PROFILE_WEIGHTS == {"TOURIST": 1.0, "VISITOR": 0.5}
    assert out["ID_COMUNE"].tolist() == ["000001", "000002", "000003"]
    assert not out.drop(columns="giorni_coperti").isna().any().any()
    # the two comuni of area A share the value of the area
    for col in out.columns.drop(["DATA", "ID_COMUNE"]):
        assert out.loc[0, col] == out.loc[1, col]

    # combined quota_uscita of A = (100 + 0.5 * 200) / (1000 + 0.5 * 400) = 200 / 1200
    a = out.loc[0]
    assert np.isclose(a["quota_uscita_tourist"], 100 / 1000)
    assert np.isclose(a["quota_uscita_visitor"], 200 / 400)
    assert np.isclose(a["quota_uscita_combined"], (100 + 0.5 * 200) / (1000 + 0.5 * 400))


def test_intra_ambito_is_computed_without_presence_denominator():
    rows = []
    for profile in ("TOURIST", "VISITOR"):
        rows += [
            ("2024-01-01", "*", "X", "A", "X", "ND", profile, 200),
            ("2024-01-01", "A", "X", "B", "X", "UD", profile, 50),
            ("2024-01-01", "B", "X", "A", "X", "UD", profile, 30),
        ]
    area_comune = pd.DataFrame(
        {"areaId": ["A", "B"], "ID_COMUNE": ["000001", "000002"]}
    )
    out = comune_flow_indicators(
        _od(rows),
        area_comune,
        min_t=100,
        damping=0.85,
        profiles=COVISITS_PROFILES,
        metrics=COVISITS_METRICS,
        weights=COVISITS_PROFILE_WEIGHTS,
    )

    b = out[out["ID_COMUNE"] == "000002"].iloc[0]
    assert pd.isna(b["quota_ingresso_tourist"])
    assert pd.isna(b["quota_uscita_tourist"])
    assert np.isclose(b["quota_intra_ambito_tourist"], 1.0)
    assert np.isclose(b["quota_intra_ambito_combined"], 1.0)
