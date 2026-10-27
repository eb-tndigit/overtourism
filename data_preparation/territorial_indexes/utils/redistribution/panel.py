# SPDX-License-Identifier: Apache-2.0
"""Daily redistribution results -> the tables read by the backend indicator.

For every day and segment (tourists T, excursionists E) the min-cost redistribution (flow.py) is solved once; the
backend never runs the solver, it only slices these panels by date (mask + mean, like any daily phenomenon):

  comune_panel  comune x day: imbalance index (saturation points) and persons to attract / move / not absorbable,
                for the tourist profile, the visitor profile and their weighted union (`_combined`)
  arcs_panel    arc x day:    optimal flow node_from -> node_to (the arrows of the map)
  nodes_table   node:         WGS84 position, label and comuni (the ends of the arrows)

A node-day without data is absent (never a zero), so a window mean does not count it.
Pure functions, no I/O. Parameters: settings.yaml, `redistribution.transport_cost` / `redistribution.model`.
"""

import logging
import time

import numpy as np
import pandas as pd

from utils.redistribution.flow import Graph

logger = logging.getLogger(__name__)

SUFFIX = {"T": "tourist", "E": "visitor"}  # same suffixes as the OD flow phenomena (+ "combined")
SEGMENTS = ("combined", "tourist", "visitor")


def _differs(values: np.ndarray) -> np.ndarray:
    v = values.astype(object)
    return (v[:, None] != v[None, :]).astype(float)


def build_graphs(nodes: pd.DataFrame, cost: dict) -> dict:
    """One graph per segment. The cost of an arc is multiplied by exp(a_cluster * [different seasonal profile] +
    a_cat * [different ISTAT typology]): moving visitors between dissimilar nodes is harder (affinities estimated on
    the co-visits, settings.yaml `redistribution.transport_cost.affinity`)."""
    d_cluster, d_cat = _differs(nodes["cluster"].to_numpy()), _differs(nodes["cod_cat"].to_numpy())
    graphs = {}
    for seg, aff in cost["affinity"].items():
        mult = np.exp(aff["cluster"] * d_cluster + aff["cat"] * d_cat)
        graphs[seg] = Graph(
            nodes[["cx", "cy"]].values,
            mult,
            alpha=cost["alpha"],
            d_ref_km=cost["d_ref_km"],
            d_max_km=cost["d_max_km"],
            d_min_km=cost["d_min_km"],
        )
    return graphs


def solve_all_days(nodes, demand, ceilings, cost: dict, min_arc_flow: float):
    """Solve the daily problem for T and E. Returns (node-day frame, arcs frame)."""
    node_ids = nodes["node"].to_numpy()
    graphs = build_graphs(nodes, cost)
    node_rows, arc_rows, t0 = [], [], time.time()
    for seg, g in graphs.items():
        wide = demand.pivot(index="date", columns="node", values=seg)[node_ids]
        ceil = {
            y: ceilings[ceilings["year"] == y].set_index("node")["C_" + seg].reindex(node_ids).to_numpy()
            for y in ceilings["year"].unique()
        }
        for day, row in wide.iterrows():
            if day.year not in ceil:
                raise KeyError(f"no ceilings for year {day.year}: the ceiling step must cover every year of the demand")
            P, C = row.to_numpy(float), ceil[day.year]
            r = g.solve(P, C)
            node_rows.append(
                pd.DataFrame(
                    {
                        "date": day, "seg": seg, "node": node_ids, "P": P, "C": C, "ubar": r["ubar"],
                        "out": r["out"], "inn": r["inn"], "unabs": r["unabs"],
                    }
                )
            )
            ai, aj, f = r["arcs"]
            keep = f >= min_arc_flow  # below: numerical dust
            if keep.any():
                arc_rows.append(
                    pd.DataFrame(
                        {
                            "date": day, "seg": seg, "node_from": node_ids[ai[keep]],
                            "node_to": node_ids[aj[keep]], "flow": f[keep],
                        }
                    )
                )
        logger.info("redistribution %s: %d days solved (%.0fs)", seg, len(wide), time.time() - t0)
    return pd.concat(node_rows, ignore_index=True), pd.concat(arc_rows, ignore_index=True)


def node_day_indicators(r: pd.DataFrame, visitor_weight: float) -> pd.DataFrame:
    """Per node and day, one column per segment. Imbalance (pp) = 100 * (balanced saturation - saturation);
    combined = tourists + visitor_weight * excursionists (an excursionist-day weighs less than a tourist-day)."""
    r = r.assign(gap=r["ubar"] * r["C"] - r["P"], moved=r["inn"] - r["out"])
    r["gap_pp"] = 100 * (r["ubar"] - r["P"] / r["C"])
    w = r.pivot(index=["node", "date"], columns="seg", values=["C", "gap", "gap_pp", "moved", "unabs"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    out = pd.DataFrame({"node": w["node"], "date": w["date"]})
    out["squilibrio_pp_combined"] = 100 * (w["gap_T"] + visitor_weight * w["gap_E"]) / (
        w["C_T"] + visitor_weight * w["C_E"]
    )
    out["squilibrio_pp_tourist"], out["squilibrio_pp_visitor"] = w["gap_pp_T"], w["gap_pp_E"]
    # persons at node level; they are split over the comuni of the node afterwards
    out["gap_tourist"], out["gap_visitor"] = w["gap_T"], w["gap_E"]
    out["moved_tourist"], out["moved_visitor"] = w["moved_T"], w["moved_E"]
    out["unabs_tourist"], out["unabs_visitor"] = w["unabs_T"], w["unabs_E"]
    return out


def comune_weights(comune_node: pd.DataFrame, strutture: pd.DataFrame, popolazione: pd.DataFrame) -> pd.DataFrame:
    """Share of the persons of a node that goes to each comune: tourists by beds, excursionists by residents."""
    yearly = lambda df: df.assign(DATA=df["DATA"].astype(str).str[:4].astype(int))  # noqa: E731
    w = (
        yearly(strutture)[["DATA", "ID_COMUNE", "tot_postiletto"]]
        .merge(yearly(popolazione), on=["DATA", "ID_COMUNE"])
        .merge(comune_node, on="ID_COMUNE")
        .rename(columns={"DATA": "year"})
    )
    w["sb"] = w["tot_postiletto"] / w.groupby(["node", "year"])["tot_postiletto"].transform("sum")
    w["sp"] = w["popolazione"] / w.groupby(["node", "year"])["popolazione"].transform("sum")
    w["sb"] = w["sb"].fillna(w["sp"])  # a node without beds: split by residents
    return w[["node", "year", "ID_COMUNE", "sb", "sp"]]


def build_comune_panel(nd: pd.DataFrame, weights: pd.DataFrame, visitor_weight: float) -> pd.DataFrame:
    o = nd.assign(year=nd["date"].dt.year).merge(weights, on=["node", "year"])
    p = pd.DataFrame({"DATA": o["date"].dt.strftime("%Y-%m-%d"), "ID_COMUNE": o["ID_COMUNE"]})
    # the pp index is a value of the NODE (normalised by its ceiling): every comune of the node gets it
    for s in SEGMENTS:
        p[f"squilibrio_pp_{s}"] = o[f"squilibrio_pp_{s}"]
    # persons are split over the comuni (they sum back to the node total); combined = T + visitor_weight * E
    for name, col in (("persone_gap", "gap"), ("persone_spostate_nette", "moved"), ("eccesso_non_assorbibile", "unabs")):
        t, v = o[f"{col}_tourist"] * o["sb"], o[f"{col}_visitor"] * o["sp"]
        p[f"{name}_tourist"], p[f"{name}_visitor"], p[f"{name}_combined"] = t, v, t + visitor_weight * v
    num = p.columns.drop(["DATA", "ID_COMUNE"])
    p[num] = p[num].astype("float64").round(3)
    # a node-day without data is NaN everywhere: no row (a window mean must skip it, not count it as a zero)
    p = p.dropna(subset=["squilibrio_pp_combined"])
    return p.sort_values(["DATA", "ID_COMUNE"], ignore_index=True)


def build_arcs_panel(arcs: pd.DataFrame, visitor_weight: float) -> pd.DataFrame:
    """arc x day: flow_tourist, flow_visitor and flow_combined = tourist + visitor_weight * visitor (persons/day).
    The weight is applied here, once: the backend only reads the column of the requested segment."""
    a = arcs.assign(seg=arcs["seg"].map(SUFFIX)).pivot_table(
        index=["date", "node_from", "node_to"], columns="seg", values="flow", aggfunc="sum", fill_value=0.0
    )
    a = a.reindex(columns=["tourist", "visitor"], fill_value=0.0).rename(columns=lambda c: f"flow_{c}").reset_index()
    a["flow_combined"] = a["flow_tourist"] + visitor_weight * a["flow_visitor"]
    a["DATA"] = a["date"].dt.strftime("%Y-%m-%d")
    cols = ["flow_tourist", "flow_visitor", "flow_combined"]
    for c in cols:
        a[c] = a[c].astype("float32").round(1)
    return a[["DATA", "node_from", "node_to", *cols]].sort_values(["DATA", "node_from", "node_to"], ignore_index=True)


def build_nodes_table(nodes: pd.DataFrame, comune_node: pd.DataFrame, names: pd.Series, utm_epsg: int = 32632):
    """node, lon, lat (WGS84), label (up to 3 comune names), comuni (list of ID_COMUNE). Centroids are in km."""
    from pyproj import Transformer

    to_wgs = Transformer.from_crs(utm_epsg, 4326, always_xy=True)
    lon, lat = to_wgs.transform(nodes["cx"].to_numpy() * 1000, nodes["cy"].to_numpy() * 1000)
    members = comune_node.groupby("node")["ID_COMUNE"].apply(lambda s: sorted(s))
    label = members.apply(lambda ids: " + ".join(names.reindex(ids).fillna("?").head(3)))
    t = pd.DataFrame({"node": nodes["node"].to_numpy(), "lon": lon, "lat": lat})
    t["label"] = t["node"].map(label)
    t["comuni"] = t["node"].map(members)
    return t
