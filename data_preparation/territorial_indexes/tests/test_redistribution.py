# SPDX-License-Identifier: Apache-2.0
"""Redistribution indicator: node building, solver, split to comuni, config. Synthetic data, no network, no S3."""
import numpy as np
import pandas as pd
import pytest

from utils import config
from utils.redistribution import nodes as nd
from utils.redistribution import panel as pn
from utils.redistribution.flow import Graph
from utils.steps.create_redistribution_df import fix_encoding

COST = dict(alpha=2.0, d_ref_km=10.0, d_max_km=40.0, d_min_km=3.0)


def test_redistribution_inputs_are_saved_in_mapping_layout(monkeypatch, tmp_path):
    from utils.steps import create_redistribution_df as step

    downloads = []
    monkeypatch.setattr(step, "download_mappings", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        step,
        "download_mapping_artifact",
        lambda name, mapping_dir, **kwargs: downloads.append((name, mapping_dir)),
    )

    step.download_redistribution_inputs(mapping_dir=tmp_path)

    assert ("tipologia_comuni.csv", tmp_path) in downloads
    assert ("Com01012026_g", tmp_path) in downloads


# ------------------------------------------------------------------ nodes
def test_areas_sharing_a_comune_form_one_node_and_numbering_is_stable():
    links = pd.DataFrame(
        {"area": ["1", "1", "2", "3"], "ID_COMUNE": ["022001", "022002", "022002", "022003"]}
    )
    cm, area_node = nd.link_nodes(links)
    node = cm.set_index("ID_COMUNE")["node"]
    assert node["022001"] == node["022002"] != node["022003"]
    assert area_node["1"] == area_node["2"] == node["022001"] and area_node["3"] == node["022003"]
    assert sorted(cm["node"].unique()) == ["N000", "N001"]


def test_node_centroid_is_the_centroid_of_the_union_of_its_areas():
    import geopandas as gpd
    from shapely.geometry import box

    areas = gpd.GeoDataFrame(
        {"id": ["1", "2", "3"], "geometry": [box(0, 0, 2000, 2000), box(2000, 0, 4000, 2000), box(10000, 0, 12000, 2000)]},
        crs=32632,
    )
    out = nd.node_centroids(areas, {"1": "N000", "2": "N000", "3": "N001"}).set_index("node")
    assert out.loc["N000", "cx"] == pytest.approx(2.0) and out.loc["N000", "cy"] == pytest.approx(1.0)  # km
    assert out.loc["N001", "cx"] == pytest.approx(11.0)


def test_typology_of_merged_comuni_is_the_most_frequent_of_the_constituents():
    raw = pd.DataFrame(
        {"PROCOM": ["022001", "099001", "099002", "099003"], "COMUNE": ["Ala", "Brez", "Cloz", "Revò"], "COD_CAT": ["B", "E", "E", "P"]}
    )
    out = nd.current_typology(raw, ["022001", "022253"], {"022253": ["Brez", "Cloz", "Revò"]}).set_index("PROCOM")["COD_CAT"]
    assert out["022001"] == "B" and out["022253"] == "E" and "099001" not in out.index
    with pytest.raises(ValueError):
        nd.current_typology(raw, ["022001"], {"022254": ["Nowhere"]})


def test_shapefile_names_encoding_is_repaired_and_clean_names_are_kept():
    assert fix_encoding("Baselga di PinÃ¨") == "Baselga di Pinè"
    assert fix_encoding("Trento") == "Trento"


# ------------------------------------------------------------------ solver
def _line(n=3, step_km=10.0):
    return np.array([[i * step_km, 0.0] for i in range(n)])


def test_excess_moves_to_the_node_with_room_and_conserves_volume():
    g = Graph(_line(), None, **COST)
    P, C = np.array([300.0, 100.0, 100.0]), np.array([100.0, 100.0, 100.0])
    r = g.solve(P, C)
    assert r["ubar"] == pytest.approx(500 / 300)
    assert r["out"].sum() == pytest.approx(r["inn"].sum())  # what leaves one node enters another
    assert r["out"][0] > 0 and r["inn"][1:].sum() == pytest.approx(r["out"][0])  # node 0 is over, 1 and 2 have room
    P2 = np.array([200.0, 20.0, 20.0])
    r2 = g.solve(P2, C)
    assert r2["out"][0] > 0 and r2["inn"][0] == 0 and r2["inn"][1:].sum() == pytest.approx(r2["out"][0])


def test_nodes_farther_than_dmax_never_exchange_flow_and_nan_nodes_are_left_out():
    g = Graph(_line(3, 100.0), None, **COST)  # 100 km apart > d_max
    r = g.solve(np.array([200.0, 10.0, 10.0]), np.array([100.0, 100.0, 100.0]))
    assert r["out"].sum() == 0 and r["unabs"][0] > 0
    r = Graph(_line(), None, **COST).solve(np.array([np.nan, 100.0, 100.0]), np.array([100.0, 100.0, 100.0]))
    assert np.isnan(r["out"][0]) and not np.isnan(r["out"][1])


# ------------------------------------------------------------------ tables
def test_persons_are_split_over_the_comuni_and_sum_back_to_the_node_total():
    cm = pd.DataFrame({"ID_COMUNE": ["022001", "022002"], "node": ["N000", "N000"]})
    st = pd.DataFrame({"DATA": [2024, 2024], "ID_COMUNE": cm.ID_COMUNE, "tot_postiletto": [300.0, 100.0]})
    pop = pd.DataFrame({"DATA": [2024, 2024], "ID_COMUNE": cm.ID_COMUNE, "popolazione": [500.0, 500.0]})
    w = pn.comune_weights(cm, st, pop)
    assert w.groupby("node")[["sb", "sp"]].sum().iloc[0].tolist() == pytest.approx([1.0, 1.0])
    nd_day = pd.DataFrame(
        {"node": ["N000"], "date": [pd.Timestamp("2024-08-15")], "squilibrio_pp_combined": [-20.0],
         "squilibrio_pp_tourist": [-30.0], "squilibrio_pp_visitor": [-10.0],
         "gap_tourist": [-400.0], "gap_visitor": [-200.0], "moved_tourist": [0.0], "moved_visitor": [0.0],
         "unabs_tourist": [0.0], "unabs_visitor": [0.0]}
    )
    p = pn.build_comune_panel(nd_day, w, visitor_weight=0.5).set_index("ID_COMUNE")
    assert p["persone_gap_tourist"].sum() == pytest.approx(-400) and p.loc["022001", "persone_gap_tourist"] == pytest.approx(-300)
    assert p["persone_gap_visitor"].sum() == pytest.approx(-200) and p.loc["022002", "persone_gap_visitor"] == pytest.approx(-100)
    assert p["persone_gap_combined"].sum() == pytest.approx(-400 + 0.5 * -200)
    assert (p["squilibrio_pp_combined"] == -20.0).all()  # the index is a value of the node: shared by its comuni


def test_node_days_without_data_have_no_row():
    cm = pd.DataFrame({"ID_COMUNE": ["022001"], "node": ["N000"]})
    nd_day = pd.DataFrame(
        {"node": ["N000", "N000"], "date": pd.to_datetime(["2024-08-14", "2024-08-15"]),
         "squilibrio_pp_combined": [np.nan, 5.0], "squilibrio_pp_tourist": [np.nan, 5.0], "squilibrio_pp_visitor": [np.nan, 5.0],
         **{c: [np.nan, 1.0] for c in ("gap_tourist", "gap_visitor", "moved_tourist", "moved_visitor", "unabs_tourist", "unabs_visitor")}}
    )
    w = pd.DataFrame({"node": ["N000"], "year": [2024], "ID_COMUNE": ["022001"], "sb": [1.0], "sp": [1.0]})
    assert pn.build_comune_panel(nd_day, w, 0.5)["DATA"].tolist() == ["2024-08-15"]


def test_arcs_panel_carries_the_combined_flow_with_the_given_weight():
    arcs = pd.DataFrame(
        {"date": pd.to_datetime(["2024-08-15"] * 2), "seg": ["T", "E"], "node_from": ["N000"] * 2, "node_to": ["N001"] * 2,
         "flow": [100.0, 40.0]}
    )
    a = pn.build_arcs_panel(arcs, visitor_weight=0.5).iloc[0]
    assert (a.flow_tourist, a.flow_visitor, a.flow_combined) == (100.0, 40.0, 120.0)


# ------------------------------------------------------------------ config
def test_redistribution_settings_are_valid_and_a_missing_key_is_reported(monkeypatch):
    cfg = config.redistribution_settings()
    assert cfg["output"]["panel"] == "phen_ridistribuzione" and set(cfg["transport_cost"]["affinity"]) == {"T", "E"}
    broken = {k: v for k, v in config._S.items() if k != "redistribution"}
    broken["redistribution"] = {k: v for k, v in cfg.items() if k != "ceiling"}
    monkeypatch.setattr(config, "_S", broken)
    with pytest.raises(KeyError, match="redistribution.ceiling"):
        config.redistribution_settings()
