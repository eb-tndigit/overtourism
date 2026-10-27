# SPDX-License-Identifier: Apache-2.0
"""Nodes of the redistribution analysis and their daily load / yearly attributes (ported from the notebook).

A NODE is a connected component of the area <-> comune links: the Vodafone areas that share a comune (and the comuni
of a multi-comune area) form one unit, because the mobile data cannot tell them apart. Everything of the analysis
(load, ceiling, flows) lives at node level; the comuni only get the result back (see panel.py).

Pure functions on dataframes: no file name, no I/O (utils/steps/create_redistribution_df.py reads and writes).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from utils.flows import build_area_comune, load_area_names

logger = logging.getLogger(__name__)

N_CLUSTERS = 4  # seasonal profiles: annuale, sci, estate_alpina, estate_spalle (the names below assume 4)
KMEANS_SEED = 0
KMEANS_N_INIT = 50


# ------------------------------------------------------------------ area <-> comune <-> node
def area_comune_links(geojson_path, mapping_vodafone, mapping_comuni) -> pd.DataFrame:
    """Table (area, ID_COMUNE) of the Vodafone areas that HAVE a polygon in the geojson (the nodes need a position).

    The rule is `utils.flows.build_area_comune` (the same used for the presences and the flow indicators); the
    network-only areas that it adds through AREA_COMUNE_OVERRIDES have no polygon, so they are left out here.
    """
    links = build_area_comune(geojson_path, mapping_vodafone, mapping_comuni)
    with_polygon = set(load_area_names(geojson_path)["id"])
    return (
        links[links["areaId"].isin(with_polygon)]
        .rename(columns={"areaId": "area"})
        .reset_index(drop=True)
    )


def link_nodes(links: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Connected components of the area-comune graph (union-find).

    Returns (comune_node: ID_COMUNE, node) and (area_node: area id -> node). Node ids are N000, N001, ... in the order
    of the component roots: they depend on the ORDER of the area rows (the geojson), so keep that file as it is.
    """
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in links.itertuples():
        parent[find("A" + r.area)] = find("C" + r.ID_COMUNE)
    roots = pd.Series([find("C" + c) for c in links.ID_COMUNE], index=links.index)
    ids = {n: f"N{i:03d}" for i, n in enumerate(sorted(roots.unique()))}
    comune_node = (
        pd.DataFrame({"ID_COMUNE": links.ID_COMUNE.to_numpy(), "node": roots.map(ids).to_numpy()})
        .drop_duplicates("ID_COMUNE")
        .reset_index(drop=True)
    )
    area_node = {a: ids.get(find("A" + a)) for a in links.area.unique()}
    return comune_node, area_node


def node_centroids(areas_utm, area_node: dict) -> pd.DataFrame:
    """node, cx, cy (km, UTM 32N): centroid of the union of the polygons of the areas of each node.

    `areas_utm`: GeoDataFrame of the areas in EPSG:32632 with an `id` column (same id as `area_node`).
    """
    areas = areas_utm.copy()
    areas["node"] = areas["id"].map(lambda a: area_node.get(a))
    geo = areas.dropna(subset=["node"]).dissolve(by="node")
    cent = geo.geometry.centroid
    return pd.DataFrame({"node": geo.index, "cx": cent.x.to_numpy() / 1000, "cy": cent.y.to_numpy() / 1000})


# ------------------------------------------------------------------ load and attributes
def node_demand(presenze: pd.DataFrame, comune_node: pd.DataFrame) -> pd.DataFrame:
    """node x day: T (tourists, Vodafone presences) and E (excursionists, Vodafone). Columns: node, date, T, E, year."""
    p = presenze.merge(comune_node, on="ID_COMUNE")
    p["date"] = pd.to_datetime(p["DATA"])
    dem = p.groupby(["node", "date"], as_index=False).agg(
        T=("presenze_vodafone", "sum"), E=("presenze_escursionisti_vodafone", "sum")
    )
    dem["year"] = dem["date"].dt.year
    return dem


def _yearly(df: pd.DataFrame) -> pd.DataFrame:
    """DATA of the yearly phenomena as int year."""
    return df.assign(DATA=df["DATA"].astype(str).str[:4].astype(int))


def node_year(strutture, popolazione, comune_node) -> tuple[pd.DataFrame, pd.DataFrame]:
    """node x year: beds, residents, second homes available. Also returns the comune x year table it is built from."""
    att = _yearly(strutture).merge(_yearly(popolazione), on=["DATA", "ID_COMUNE"]).merge(comune_node, on="ID_COMUNE")
    ny = (
        att.groupby(["node", "DATA"], as_index=False)
        .agg(beds=("tot_postiletto", "sum"), pop=("popolazione", "sum"), disp=("letti_alloggi_disp", "sum"))
        .rename(columns={"DATA": "year"})
    )
    return ny, att


def seasonal_clusters(dem: pd.DataFrame) -> pd.Series:
    """node -> seasonal profile (KMeans on the monthly shape of the tourists), named by the shape of the cluster."""
    monthly = dem.assign(m=dem["date"].dt.month).groupby(["node", "m"]).T.mean().unstack()
    prof = monthly.div(monthly.sum(axis=1), axis=0).fillna(1 / 12)
    km = KMeans(N_CLUSTERS, n_init=KMEANS_N_INIT, random_state=KMEANS_SEED).fit(prof)
    shape = prof.groupby(km.labels_).mean()
    stats = {}
    for c, r in shape.iterrows():
        stats[c] = (r[[7, 8]].sum(), r[[12, 1, 2]].sum(), r.std())  # summer, winter, flatness
    order = sorted(stats, key=lambda c: stats[c][2])  # flattest first
    names = {order[0]: "annuale"}
    rest_sorted = sorted(order[1:], key=lambda c: stats[c][1] - stats[c][0], reverse=True)  # most winter-heavy first
    names[rest_sorted[0]] = "sci"
    names[rest_sorted[-1]] = "estate_alpina"
    names[rest_sorted[1]] = "estate_spalle"
    logger.info("seasonal clusters: %s", pd.Series(km.labels_).map(names).value_counts().to_dict())
    return pd.Series(km.labels_, index=prof.index).map(names)


def current_typology(tip_raw: pd.DataFrame, comune_ids, merged: dict) -> pd.DataFrame:
    """PROCOM, COD_CAT (ISTAT tourist typology) for the CURRENT comuni.

    The comuni born from a merger are absent from the typology file: their class is the most frequent one among the
    comuni they were made of (`merged`: new ID_COMUNE -> names of the constituent comuni, settings.yaml).
    """
    tip = tip_raw.assign(PROCOM=tip_raw["PROCOM"].astype(str).str.zfill(6))
    current = tip[tip["PROCOM"].isin(set(comune_ids))][["PROCOM", "COD_CAT"]]
    rows = []
    for new_id, constituents in merged.items():
        sub = tip[tip["COMUNE"].isin(constituents)]
        if sub.empty:
            raise ValueError(f"typology: none of the constituent comuni of {new_id} found: {constituents}")
        if len(sub) != len(constituents):
            logger.warning("typology of %s: %d of %d constituent comuni found", new_id, len(sub), len(constituents))
        rows.append({"PROCOM": new_id, "COD_CAT": sub["COD_CAT"].mode().iloc[0]})
    return pd.concat([current, pd.DataFrame(rows)], ignore_index=True)


def dominant_typology(typology: pd.DataFrame, att: pd.DataFrame, comune_node, year: int) -> pd.Series:
    """node -> COD_CAT of its comune with the most beds in `year`."""
    tip = typology.merge(comune_node, left_on="PROCOM", right_on="ID_COMUNE")
    beds = att[att["DATA"] == year].set_index("ID_COMUNE")["tot_postiletto"]
    tip["b"] = tip["ID_COMUNE"].map(beds).fillna(0)
    return tip.sort_values("b").groupby("node").tail(1).set_index("node")["COD_CAT"]


def build_nodes(
    *,
    comune_node: pd.DataFrame,
    centroids: pd.DataFrame,
    comuni: pd.DataFrame,
    presenze: pd.DataFrame,
    strutture: pd.DataFrame,
    popolazione: pd.DataFrame,
    typology: pd.DataFrame,
    typology_year: int,
) -> dict:
    """Everything the next steps need. `comuni`: ID_COMUNE, Shape_Area (m2) of the current comuni."""
    dem = node_demand(presenze, comune_node)
    ny, att = node_year(strutture, popolazione, comune_node)
    nodes = centroids.copy()
    surface = comuni.merge(comune_node, on="ID_COMUNE").groupby("node")["Shape_Area"].sum() / 1e6
    nodes["surface_km2"] = nodes["node"].map(surface)
    nodes["cluster"] = seasonal_clusters(dem).reindex(nodes["node"]).to_numpy()
    nodes["cod_cat"] = nodes["node"].map(dominant_typology(typology, att, comune_node, typology_year))
    missing = nodes.columns[nodes.isna().any()].tolist()
    if missing:
        raise ValueError(f"nodes with missing {missing}: check the typology / shapefile / area inputs")
    return {"nodes": nodes, "comune_node": comune_node, "demand": dem, "node_year": ny}
