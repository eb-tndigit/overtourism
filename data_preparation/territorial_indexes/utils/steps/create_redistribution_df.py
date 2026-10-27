# SPDX-License-Identifier: Apache-2.0
"""
STEP - Redistribution indicator ("Squilibrio di carico e ridistribuzione dei flussi").

Input : Output/data/final_data/   phen_presenze, phen_strutture, phen_popolazione   (gen_phenomenon_dataframes.py)
        Output/mapping/          the Vodafone areas geojson + the area <-> comune mappings (`references`)
                                ISTAT comuni shapefile + ISTAT tourist typology (`redistribution.files`)
Output: Output/data/final_data/  phen_ridistribuzione, phen_ridistribuzione_flussi, ridistribuzione_nodi (parquet)
        Output/data/data_processed/redistribution/  nodes, comune_node, demand, node_year, ceilings, model tables

What it does (all the parameters are in settings.yaml, `redistribution`):

  1 nodes      the Vodafone areas that share a comune form a NODE (utils/redistribution/nodes.py); load per node and day
  2 ceilings   load ceiling of every node and year, quantile regression on the observed peaks (ceiling.py)
  3 solve      for every day and segment, the min-cost redistribution of the excess towards the nodes with room (flow.py)
  4 tables     the daily panels read by the backend indicator `squilibrio-ridistribuzione` (panel.py)

The solver runs ONCE here: the backend only slices the daily panels by date.
"""

import logging
import shutil
from pathlib import Path

import pandas as pd

from utils.config import (
    FINAL_DIR,
    MAPPING_DIR,
    MAPPING_FILES,
    PROCESSED_DIR,
    PROJECT,
    REFERENCES,
    TYPE_FORMAT,
    redistribution_settings,
)
from utils.datasets import COVISITS_VISITOR_COEFFICIENT
from utils.io import ensure_dir, put_dataframe, read_df, read_json, save_computed_dfs
from utils.redistribution import nodes as nd
from utils.redistribution import panel as pn
from utils.redistribution.ceiling import fit_ceilings
from utils.steps.download_data import download_mappings

logger = logging.getLogger(__name__)

TRENTINO_PREFIX = "022"  # ISTAT province code of Trento
TRENTINO_KM2 = (5500, 7000)  # plausibility of the total surface: Shape_Area must be in m2


def download_mapping_artifact(name: str, mapping_dir, *, local_dir=None, force=False) -> None:
    """Download a DigitalHub artifact into mapping_dir, or copy it from a local mirror."""
    target = Path(mapping_dir) / name
    if not force and (
        (target.is_file() and target.stat().st_size > 0)
        or (target.is_dir() and any(target.iterdir()))
    ):
        logger.info("%s already in %s: not downloaded", name, mapping_dir)
        return

    if local_dir is not None:
        source = Path(local_dir) / name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
        return

    import digitalhub as dh

    logger.info("Downloading artifact %s...", name)
    dh.get_project(PROJECT).get_artifact(name).download(str(mapping_dir), overwrite=force)


# ------------------------------------------------------------------ download
def download_redistribution_inputs(mapping_dir=MAPPING_DIR, local_dir=None, force=False) -> None:
    """Download shared mappings and the two redistribution artifacts into mapping_dir."""
    cfg = redistribution_settings()
    download_mappings(mapping_dir, local_dir=local_dir, force=force)
    mapping_dir = ensure_dir(mapping_dir)

    download_mapping_artifact(
        cfg["files"]["tipologia_comuni"], mapping_dir, local_dir=local_dir, force=force
    )
    shapefile_path = Path(cfg["files"]["comuni_shapefile"])
    download_mapping_artifact(
        shapefile_path.parts[0], mapping_dir, local_dir=local_dir, force=force
    )


# ------------------------------------------------------------------ input readers
def fix_encoding(name: str) -> str:
    """The ISTAT shapefile names come out as UTF-8 read as cp1252 ('Baselga di PinÃ¨'): repair them."""
    try:
        return name.encode("cp1252").decode("utf-8")
    except Exception:
        return name


def read_comuni(shapefile) -> pd.DataFrame:
    """ID_COMUNE, COMUNE, Shape_Area (m2) of the comuni of the province (ISTAT shapefile)."""
    import geopandas as gpd

    g = gpd.read_file(shapefile)
    g["ID_COMUNE"] = g["PRO_COM_T"].astype(str).str.zfill(6)
    g = g[g["ID_COMUNE"].str.startswith(TRENTINO_PREFIX)]
    km2 = g["Shape_Area"].sum() / 1e6
    if not TRENTINO_KM2[0] < km2 < TRENTINO_KM2[1]:
        raise ValueError(f"{shapefile}: the comuni sum to {km2:.0f} km2, expected about 6200: is Shape_Area in m2?")
    return pd.DataFrame({"ID_COMUNE": g["ID_COMUNE"], "COMUNE": g["COMUNE"].map(fix_encoding), "Shape_Area": g["Shape_Area"]})


def read_areas(geojson, utm_epsg: int):
    """Polygons of the Vodafone areas in UTM (metres), with the area `id` as text (same rule as utils.flows)."""
    import geopandas as gpd

    areas = gpd.read_file(geojson).to_crs(utm_epsg)
    areas["id"] = areas["id"].astype(str).str.replace(".0", "", regex=False)
    return areas


# ------------------------------------------------------------------ the computation (no I/O)
def compute_redistribution(
    *, areas_utm, links, comuni, tipologia_raw, presenze, strutture, popolazione, cfg, visitor_weight
) -> dict:
    """All the steps, from dataframes to dataframes. Returns {"outputs": {...3 tables...}, "intermediate": {...}}."""
    comune_node, area_node = nd.link_nodes(links)
    typology = nd.current_typology(tipologia_raw, comuni["ID_COMUNE"], cfg["merged_comuni"])
    built = nd.build_nodes(
        comune_node=comune_node,
        centroids=nd.node_centroids(areas_utm, area_node),
        comuni=comuni,
        presenze=presenze,
        strutture=strutture,
        popolazione=popolazione,
        typology=typology,
        typology_year=cfg["model"]["typology_year"],
    )
    nodes, demand, node_year = built["nodes"], built["demand"], built["node_year"]
    logger.info("%d nodes, %d comuni, %d node-days", len(nodes), len(comune_node), len(demand))

    logger.info("Fitting the load ceilings...")
    ceilings, cv, coef = fit_ceilings(demand, node_year, nodes, cfg["ceiling"])

    logger.info("Solving the daily redistribution...")
    solved, arcs = pn.solve_all_days(nodes, demand, ceilings, cfg["transport_cost"], cfg["model"]["min_arc_flow"])
    day = pn.node_day_indicators(solved, visitor_weight)
    weights = pn.comune_weights(comune_node, strutture, popolazione)
    panel = pn.build_comune_panel(day, weights, visitor_weight)
    names = comuni.set_index("ID_COMUNE")["COMUNE"]
    return {
        "outputs": {
            cfg["output"]["panel"]: panel,
            cfg["output"]["flows"]: pn.build_arcs_panel(arcs, visitor_weight),
            cfg["output"]["nodes"]: pn.build_nodes_table(nodes, comune_node, names, cfg["model"]["utm_epsg"]),
        },
        "intermediate": {
            "nodes": nodes, "comune_node": comune_node, "demand_node_day": demand, "node_year": node_year,
            "ceilings": ceilings, "ceiling_cv_models": cv, "ceiling_coefficients": coef,
        },
    }


# ------------------------------------------------------------------ the step
def compute_redistribution_dfs(
    type_format=TYPE_FORMAT,
    local=True,
    final_dir=FINAL_DIR,
    mapping_dir=MAPPING_DIR,
    processed_dir=PROCESSED_DIR,
) -> dict:
    """Reads the inputs, runs the pipeline, writes the three final tables (parquet) and the intermediate results.
    local=False also logs the final tables to the platform. Returns the three final tables."""
    cfg = redistribution_settings()
    inp = cfg["inputs"]
    presenze = read_df(final_dir, inp["presenze"], type_format, parse_ids=True)
    strutture = read_df(final_dir, inp["strutture"], type_format, parse_ids=True)
    popolazione = read_df(final_dir, inp["popolazione"], type_format, parse_ids=True)

    mapping_dir = Path(mapping_dir)
    geojson = mapping_dir / Path(REFERENCES["geojson"]).name
    mappings = {k: read_json(mapping_dir / f) for k, f in MAPPING_FILES.items()}
    links = nd.area_comune_links(geojson, mappings["mapping_vodafone"], mappings["mapping_comuni"])
    comuni = read_comuni(mapping_dir / f"{cfg['files']['comuni_shapefile']}.shp")
    tipologia_raw = pd.read_csv(mapping_dir / Path(cfg["files"]["tipologia_comuni"]).name, dtype={"PROCOM": str})

    result = compute_redistribution(
        areas_utm=read_areas(geojson, cfg["model"]["utm_epsg"]),
        links=links,
        comuni=comuni,
        tipologia_raw=tipologia_raw,
        presenze=presenze,
        strutture=strutture,
        popolazione=popolazione,
        cfg=cfg,
        visitor_weight=COVISITS_VISITOR_COEFFICIENT,
    )

    mid = ensure_dir(Path(processed_dir) / cfg["intermediate_dir"])
    for name, df in result["intermediate"].items():
        put_dataframe(df, name, mid, type="parquet")
    # the backend reads these three as parquet, whatever `type_format` is
    save_computed_dfs(result["outputs"], local=local, type_format="parquet", path_saving=final_dir)
    return result["outputs"]
