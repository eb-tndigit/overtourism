"""Seasonality cluster of every Trentino comune: mapping_comuni_cluster_stagionalita.json.

Same clustering as notebook_ridistribuzione/indicatore_ridistribuzione.ipynb (section 3, computed by
notebook_ridistribuzione/scripts/step2_prepare.py):
  1. the Vodafone areas and the comuni they cover are joined into NODES (connected components of the area-comune
     links, e.g. Vigo + Pozza di Fassa -> San Giovanni di Fassa): inside a multi-comune area the presences of a comune
     are a split of the area value, not an independent observation, so the clustering works on the nodes;
  2. monthly profile of each node: mean daily tourist presences (Vodafone) of every calendar month over all the
     years, as a share of the sum of the 12 months;
  3. KMeans with 4 clusters on the profiles, each cluster named by the shape of its mean profile: the flattest is
     "annuale", the others by weight of the winter (Dec-Feb) minus the summer (Jul-Aug): "sci", "estate_spalle",
     "estate_alpina".
Every comune gets the cluster of its node.

The script is run ONCE, by hand, like trentino_travel_times.py, and the file is uploaded to the platform (project
"overtourism") as an artifact with the same name: the backend indicator "gini-stagionalita" downloads it and shows
the cluster next to the Gini of every comune. Run it again (and upload the new file) when the presences change.

Input : Output/data/final_data/phen_presenze.parquet (territorial-indexes pipeline)
        Output/mapping/mapping_comuni_into_vodafone_Trento.json (Vodafone area -> ISTAT codes of its comuni)
Output: Output/mapping/mapping_comuni_cluster_stagionalita.json
        {"022001": {"cluster": 2, "tipo_cluster": "estate_spalle"}, ...}

Usage:
    python data_download/seasonality_clusters.py [--presences ...] [--mapping ...] [--output ...]
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "Output"

PRESENCES_PATH = OUTPUT_DIR / "data" / "final_data" / "phen_presenze.parquet"
AREA_MAPPING_PATH = OUTPUT_DIR / "mapping" / "mapping_comuni_into_vodafone_Trento.json"
OUTPUT_PATH = OUTPUT_DIR / "mapping" / "mapping_comuni_cluster_stagionalita.json"

PRESENCE_COLUMN = "presenze_vodafone"  # tourists
# Vigo and Pozza di Fassa (Vodafone areas) are today the comune of San Giovanni di Fassa: "NA" in the mapping
SAN_GIOVANNI_DI_FASSA = "022250"

# KMeans as in step2_prepare.py
N_CLUSTERS = 4
N_INIT = 50
RANDOM_STATE = 0

# numeric code of every cluster, fixed by its name (the KMeans labels change from one run to another). Keep the codes
# in the alphabetical order of the names: on a tie the backend takes the smallest code and the first name of a macro
# area, which must be the same cluster
CLUSTER_IDS = {"annuale": 0, "estate_alpina": 1, "estate_spalle": 2, "sci": 3}
SUMMER_MONTHS = [7, 8]
WINTER_MONTHS = [12, 1, 2]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def comune_nodes(area_mapping: dict) -> pd.Series:
    """Node of every comune (ISTAT code): connected component of the Vodafone area <-> comune links, named after the
    smallest ISTAT code it contains."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    comuni = set()
    for area, codes in area_mapping.items():
        for code in codes:
            comune = SAN_GIOVANNI_DI_FASSA if code == "NA" else f"{int(code):06d}"
            comuni.add(comune)
            parent[find("A" + area)] = find("C" + comune)

    components = pd.Series({comune: find("C" + comune) for comune in comuni})
    return components.map(components.groupby(components).apply(lambda s: s.index.min())).rename("node")


def monthly_profiles(presences: pd.DataFrame, nodes: pd.Series) -> pd.DataFrame:
    """Node x month: mean daily tourist presences of the month (all the years), as a share of the 12 months."""
    daily = (
        presences.assign(node=presences["ID_COMUNE"].map(nodes), date=pd.to_datetime(presences["DATA"]))
        .groupby(["node", "date"])[PRESENCE_COLUMN]
        .sum()
        .reset_index()
    )
    monthly = daily.assign(month=daily["date"].dt.month).groupby(["node", "month"])[PRESENCE_COLUMN].mean().unstack()
    return monthly.div(monthly.sum(axis=1), axis=0).fillna(1 / 12)


def name_clusters(profiles: pd.DataFrame, labels) -> dict:
    """KMeans label -> name, from the shape of the mean profile of the cluster (rule of step2_prepare.py)."""
    shape = profiles.groupby(labels).mean()
    flattest_first = shape.std(axis=1).sort_values().index.tolist()
    winter_minus_summer = shape[WINTER_MONTHS].sum(axis=1) - shape[SUMMER_MONTHS].sum(axis=1)
    rest = winter_minus_summer[flattest_first[1:]].sort_values(ascending=False).index.tolist()
    return {flattest_first[0]: "annuale", rest[0]: "sci", rest[1]: "estate_spalle", rest[-1]: "estate_alpina"}


def main(presences_path: Path, mapping_path: Path, output_path: Path) -> dict:
    presences = pd.read_parquet(presences_path, columns=["ID_COMUNE", "DATA", PRESENCE_COLUMN])
    presences["ID_COMUNE"] = presences["ID_COMUNE"].astype(str).str.zfill(6)
    logging.info(f"Presences {presences['DATA'].min()} -> {presences['DATA'].max()}, {presences['ID_COMUNE'].nunique()} comuni")

    nodes = comune_nodes(json.loads(mapping_path.read_text(encoding="utf-8")))
    missing = sorted(set(presences["ID_COMUNE"]) - set(nodes.index))
    if missing:
        raise ValueError(f"comuni of the presences without a Vodafone area in the mapping: {missing}")
    logging.info(f"{len(nodes)} comuni in {nodes.nunique()} nodes")

    profiles = monthly_profiles(presences, nodes)
    labels = KMeans(N_CLUSTERS, n_init=N_INIT, random_state=RANDOM_STATE).fit(profiles).labels_
    node_cluster = pd.Series(labels, index=profiles.index).map(name_clusters(profiles, labels))
    logging.info(f"Nodes per cluster: {node_cluster.value_counts().to_dict()}")

    # every comune of the mapping gets the cluster of its node
    tipo = nodes.map(node_cluster)
    if tipo.isna().any():
        raise ValueError(f"comuni without presences, hence without a cluster: {sorted(tipo.index[tipo.isna()])}")
    clusters = {
        comune: {"cluster": CLUSTER_IDS[name], "tipo_cluster": name} for comune, name in sorted(tipo.items())
    }
    logging.info(f"Comuni per cluster: {tipo.value_counts().to_dict()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(clusters, indent=1, ensure_ascii=False), encoding="utf-8")
    logging.info(f"Saved {len(clusters)} comuni to {output_path}")
    return clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--presences", type=Path, default=PRESENCES_PATH, help=f"default: {PRESENCES_PATH}")
    parser.add_argument("--mapping", type=Path, default=AREA_MAPPING_PATH, help=f"default: {AREA_MAPPING_PATH}")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help=f"default: {OUTPUT_PATH}")
    args = parser.parse_args()
    main(args.presences, args.mapping, args.output)
