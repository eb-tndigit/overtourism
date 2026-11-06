# SPDX-License-Identifier: Apache-2.0
"""Flow indicators from the Vodafone OD co-visits (ported from notebooks/flow_indicators.ipynb).

Pure functions, no I/O besides reading the geojson of the areas. Everything is computed per YEAR,
separately for every user profile (TOURIST / VISITOR):

  quota_ingresso     UD(*->j) / T(j)   share of the users present in j who come from another area
  quota_uscita       UD(j->*) / T(j)   share of the users present in j who go on to another area
  quota_intra_ambito share of the UD flow between areas of the SAME ambito (inter = 1 - intra)
  pagerank           PageRank of the weighted directed UD network between areas

with T(j) = ND(*->j): users present in the area j (rows with sourceId = '*' and covisitType = ND).
The values are user-days summed over the year; the privacy thresholds zero the small counts.

Besides the two profiles, a COMBINED profile merges them with a weight per profile (see
`combine_profiles`; the weight of VISITOR is COVISITS_PROFILE_WEIGHTS in datasets.py): the user-days of every profile are multiplied by its weight
BEFORE the indicators are computed (equivalent users: share = sum_p w_p UD_p / sum_p w_p T_p, PageRank on the
network of the weighted flows), so a profile counts in proportion to its volume AND to its weight.
The reliability threshold `min_t` is always applied on the raw (unweighted) user-days.

The indices are born at the level of the Vodafone AREAS. For the comune an area that covers several
comuni passes its value to all of them; a comune with several areas gets the weighted mean
(weights: T for the two mobility shares, intra-ambito flow for the intra share, in+out flow for the
PageRank).
"""

import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STAR = "*"  # "any area"
COMBINED = "COMBINED"  # userProfile of the weighted union of the profiles

# the SAN GIOVANNI DI FASSA unification of the vodafone standardization (see standardize_vodafone_columns)
SAN_GIOVANNI = "SAN GIOVANNI DI FASSA"
SAN_GIOVANNI_ID = 22250
UNIFIED_AREAS = ("VIGO DI FASSA", "POZZA DI FASSA")

# Vodafone network areas that represent specific comuni but are absent from the area GeoJSON.
AREA_COMUNE_OVERRIDES = {
    "LAGO_CAPR": "PELLIZZANO",
    "LAGO_LEDR": "LEDRO",
    "LAGO_MOLV": "MOLVENO",
    "LAGO_TENN": "TENNO",
    "LAGO_TOVE": "VILLE D'ANAUNIA",
    "MADO_CAMP": "PINZOLO",
    "PINZ_REND": "PINZOLO",
    "STAZ_ROVE": "ROVERETO",
    "STAZ_TREN": "TRENTO",
}

# metric -> weight column used to average the areas of a comune
METRIC_WEIGHT = {
    "quota_ingresso": "T",
    "quota_uscita": "T",
    "quota_intra_ambito": "w_intra",
    "pagerank": "w_pagerank",
}


# ------------------------------------------------------------------ areas -> comuni
def load_area_names(geojson_path) -> pd.DataFrame:
    """id -> NAME (upper case) of the Vodafone areas. Plain json, no geopandas needed."""
    with open(geojson_path, encoding="utf-8") as f:
        gj = json.load(f)
    rows = [
        (
            str(ft["properties"]["id"]).replace(".0", ""),
            str(ft["properties"]["name"]).upper(),
        )
        for ft in gj["features"]
    ]
    return pd.DataFrame(rows, columns=["id", "name"]).drop_duplicates("id")


def _as_id_list(v) -> list:
    """Flattens (possibly nested) lists of ID_COMUNE into a list of int."""
    if v is None:
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return [i for x in v for i in _as_id_list(x)]
    try:
        return [int(v)]
    except (TypeError, ValueError):
        return []


def build_area_comune(geojson_path, mapping_vodafone, mapping_comuni) -> pd.DataFrame:
    """Table areaId, ID_COMUNE (zero-padded): same criterion of the vodafone presences
    (area id -> name from the geojson -> mapping_vodafone -> ID_COMUNE), plus known
    network-only area IDs resolved through the ISTAT comune mapping."""
    names = load_area_names(geojson_path)
    names.loc[names["name"].isin(UNIFIED_AREAS), "name"] = SAN_GIOVANNI
    rows = []
    for _, r in names.iterrows():
        ids = (
            [SAN_GIOVANNI_ID]
            if r["name"] == SAN_GIOVANNI
            else _as_id_list(mapping_vodafone.get(r["name"]))
        )
        rows.extend((r["id"], f"{cid:06d}") for cid in ids)
    # Overrides take precedence if an area also appears in the GeoJSON: its GeoJSON rows are replaced.
    rows = [r for r in rows if r[0] not in AREA_COMUNE_OVERRIDES]
    rows.extend(
        (area_id, f"{mapping_comuni[comune]:06d}")
        for area_id, comune in AREA_COMUNE_OVERRIDES.items()
    )
    # An area that covers several comuni keeps ALL of them (it passes its value to each one, see to_comuni):
    # only exact duplicates are dropped, never the other comuni of the same area.
    rows = list(dict.fromkeys(rows))
    return pd.DataFrame(rows, columns=["areaId", "ID_COMUNE"]).reset_index(drop=True)


# ------------------------------------------------------------------ yearly aggregation
def yearly_flows(od: pd.DataFrame) -> tuple:
    """(yearly, coverage, ambito) from the standardized OD rows.

    yearly:   year, userProfile, covisitType, sourceId, targetId, value (user-days of the year)
    coverage: days with data per year (a year < 360 days is partial)
    ambito:   areaId -> ambito
    """
    od = od.copy()
    od["year"] = od["DATA"].astype(str).str[:4].astype(int)
    coverage = od.groupby("year")["DATA"].nunique().rename("giorni_coperti")
    yearly = od.groupby(
        ["year", "userProfile", "covisitType", "sourceId", "targetId"], as_index=False
    )["value"].sum()

    amb = pd.concat(
        [
            od[["sourceId", "sourceAmbito"]].set_axis(["id", "ambito"], axis=1),
            od[["targetId", "targetAmbito"]].set_axis(["id", "ambito"], axis=1),
        ]
    )
    amb = amb[(amb["id"] != STAR) & (amb["ambito"] != "")].drop_duplicates()
    multi = amb.groupby("id")["ambito"].nunique()
    if (multi > 1).any():
        logger.warning(
            "[flows] areas with more than one ambito (the first is used): %s",
            multi[multi > 1].index.tolist(),
        )
    ambito = amb.drop_duplicates("id").set_index("id")["ambito"].to_dict()
    return yearly, coverage, ambito


# ------------------------------------------------------------------ indicators per area
def _presence(nd, col) -> pd.Series:
    """T(j) = ND(*->j) per area (rows with source '*'; else the rows with target '*')."""
    T = nd[nd.sourceId == STAR].groupby("targetId")[col].sum()
    T_src = nd[nd.targetId == STAR].groupby("sourceId")[col].sum()
    return T.combine_first(T_src)


def area_indicators(yearly, ambito, min_t) -> pd.DataFrame:
    """quota_ingresso, quota_uscita, quota_intra_ambito per (year, userProfile, areaId).

    `yearly` has `value` (the values used by the ratios: weighted for the COMBINED profile) and `value_raw`
    (unweighted user-days). min_t: minimum yearly raw user-days in an area to compute the two mobility shares."""
    out = []
    for (year, prof), g in yearly.groupby(["year", "userProfile"]):
        nd = g[g.covisitType == "ND"]
        ud = g[g.covisitType == "UD"]

        T = _presence(nd, "value").rename("T")
        # unweighted user-days: the reliability threshold refers to the real counts, not to the weighted ones
        T_raw = _presence(nd, "value_raw").rename("T_raw")

        ud_in = (
            ud[ud.sourceId == STAR].groupby("targetId")["value"].sum().rename("UD_in")
        )
        ud_out_star = ud[ud.targetId == STAR].groupby("sourceId")["value"].sum()

        pr = ud[
            (ud.sourceId != STAR) & (ud.targetId != STAR) & (ud.sourceId != ud.targetId)
        ].copy()
        pr["amb_s"] = pr["sourceId"].map(ambito)
        pr["amb_t"] = pr["targetId"].map(ambito)
        pr = pr.dropna(subset=["amb_s", "amb_t"])
        pr["intra"] = pr["amb_s"] == pr["amb_t"]

        out_pairs = pr.groupby("sourceId")["value"].sum()
        # exit: prefer the aggregated row target='*', else the sum of the pairs (an underestimate)
        ud_out = ud_out_star.combine_first(out_pairs).rename("UD_out")

        tot_in = pr.groupby("targetId")["value"].sum().rename("pair_in")
        intra_in = pr[pr.intra].groupby("targetId")["value"].sum().rename("intra_in")
        tot_out = out_pairs.rename("pair_out")
        intra_out = pr[pr.intra].groupby("sourceId")["value"].sum().rename("intra_out")

        d = pd.concat(
            [T, T_raw, ud_in, ud_out, tot_in, intra_in, tot_out, intra_out], axis=1
        ).fillna(
            {
                k: 0
                for k in (
                    "UD_in",
                    "UD_out",
                    "pair_in",
                    "intra_in",
                    "pair_out",
                    "intra_out",
                )
            }
        )
        n_pairs_only = int((~d.index.isin(ud_out_star.index)).sum())
        if n_pairs_only:
            logger.warning(
                "[flows] %s %s: %d areas without the aggregated exit row (target='*'): "
                "quota_uscita is the sum of the pairs, an underestimate because of the censoring",
                year,
                prof,
                n_pairs_only,
            )
        ok = d["T_raw"] >= min_t
        coverage = ok.mean() if len(ok) else 0.0
        if coverage < 0.5:
            logger.warning(
                "[flows] %s %s: only %d/%d areas meet the %s user-day presence threshold; "
                "inbound/outbound shares are unavailable for most areas",
                year,
                prof,
                int(ok.sum()),
                len(ok),
                min_t,
            )
        t = d["T"].replace(0, np.nan)
        d["quota_ingresso"] = np.where(ok, d["UD_in"] / t, np.nan)
        d["quota_uscita"] = np.where(ok, d["UD_out"] / t, np.nan)
        d["w_intra"] = d["pair_in"] + d["pair_out"]
        d["quota_intra_ambito"] = (d["intra_in"] + d["intra_out"]) / d[
            "w_intra"
        ].replace(0, np.nan)
        d["year"], d["userProfile"] = year, prof
        out.append(d.rename_axis("areaId").reset_index())
    if not out:
        return pd.DataFrame(
            columns=["areaId", "year", "userProfile", "T", "T_raw", "w_intra", *METRIC_WEIGHT]
        )
    return pd.concat(out, ignore_index=True)


def pagerank(edges, nodes, damping, tol=1e-10, max_iter=500) -> pd.Series:
    """PageRank of a weighted directed graph; edges = [(source, target, weight)]."""
    idx = {n: k for k, n in enumerate(nodes)}
    n = len(nodes)
    M = np.zeros((n, n))
    for s, t, v in edges:
        M[idx[t], idx[s]] += v
    colsum = M.sum(axis=0)
    dangling = colsum == 0
    M = np.divide(M, colsum, out=np.zeros_like(M), where=colsum > 0)
    r = np.full(n, 1 / n)
    for _ in range(max_iter):
        r_new = damping * (M @ r + r[dangling].sum() / n) + (1 - damping) / n
        done = np.abs(r_new - r).sum() < tol
        r = r_new
        if done:
            break
    return pd.Series(r, index=nodes)


def pagerank_areas(yearly, damping) -> pd.DataFrame:
    """PageRank of the UD network per (year, userProfile): areaId, pagerank, w_pagerank (S_in + S_out)."""
    res = []
    for (year, prof), g in yearly[yearly.covisitType == "UD"].groupby(
        ["year", "userProfile"]
    ):
        e = g[
            (g.sourceId != STAR)
            & (g.targetId != STAR)
            & (g.sourceId != g.targetId)
            & (g.value > 0)
        ]
        if e.empty:
            continue
        nodes = sorted(set(e.sourceId) | set(e.targetId))
        pr = pagerank(list(zip(e.sourceId, e.targetId, e.value)), nodes, damping)
        s_in = e.groupby("targetId")["value"].sum().reindex(nodes).fillna(0)
        s_out = e.groupby("sourceId")["value"].sum().reindex(nodes).fillna(0)
        res.append(
            pd.DataFrame(
                {
                    "areaId": nodes,
                    "pagerank": pr.values,
                    "w_pagerank": (s_in + s_out).values,
                    "year": year,
                    "userProfile": prof,
                }
            )
        )
    if not res:
        return pd.DataFrame(
            columns=["areaId", "pagerank", "w_pagerank", "year", "userProfile"]
        )
    return pd.concat(res, ignore_index=True)


# ------------------------------------------------------------------ areas -> comuni
def to_comuni(ind_area, area_comune) -> pd.DataFrame:
    """Area indicators -> comune indicators (long: year, userProfile, ID_COMUNE + the metrics)."""
    m = ind_area.merge(area_comune, on="areaId", how="inner")
    rows = []
    for (year, prof, cid), g in m.groupby(["year", "userProfile", "ID_COMUNE"]):
        rec = {"DATA": int(year), "userProfile": prof, "ID_COMUNE": cid}
        for metric, weight in METRIC_WEIGHT.items():
            v, w = g[metric], g[weight]
            ok = v.notna() & (w > 0)
            rec[metric] = np.average(v[ok], weights=w[ok]) if ok.any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def combine_profiles(yearly, weights, name=COMBINED) -> pd.DataFrame:
    """Adds the weighted union of the profiles in `weights` ({userProfile: weight}) as profile `name`.

    Returns `yearly` + `value_raw` (the original user-days) where, for the new profile, `value` is
    sum_p weight_p * user-days_p on every (year, covisitType, sourceId, targetId) and `value_raw` the plain
    sum. Every ratio / network computed on the new profile therefore treats a user-day of profile p as
    weight_p "equivalent" user-days (e.g. TOURIST = 1, VISITOR = 0.3)."""
    yearly = yearly.copy()
    yearly["value_raw"] = yearly["value"]
    missing = sorted(set(weights) - set(yearly["userProfile"].unique()))
    if missing:
        raise ValueError(f"[flows] no data for the weighted profile(s) {missing}")
    if any(w < 0 for w in weights.values()) or not any(w > 0 for w in weights.values()):
        raise ValueError(f"[flows] the weights must be >= 0, not all zero: {weights}")
    sel = yearly[yearly["userProfile"].isin(weights)].copy()
    sel["value"] = sel["value"] * sel["userProfile"].map(weights)
    comb = sel.groupby(
        ["year", "covisitType", "sourceId", "targetId"], as_index=False
    )[["value", "value_raw"]].sum()
    comb["userProfile"] = name
    return pd.concat([yearly, comb[yearly.columns]], ignore_index=True)


def comune_flow_indicators(
    od, area_comune, *, min_t, damping, profiles, metrics, weights=None
) -> pd.DataFrame:
    """Standardized OD rows -> one row per (DATA = year, ID_COMUNE), wide:
    `<metric>_<profile in lower case>` for every metric / profile, plus `giorni_coperti`.

    weights: {userProfile: weight}. If given, the columns `<metric>_combined` (the weighted union of
    the profiles, see `combine_profiles`) are added too."""
    if od.empty:
        raise ValueError("[flows] no OD rows to process")
    yearly, coverage, ambito = yearly_flows(od)
    if weights:
        yearly = combine_profiles(yearly, weights)
        out_profiles = [*profiles, COMBINED]
    else:
        yearly["value_raw"] = yearly["value"]
        out_profiles = list(profiles)
    ia = area_indicators(yearly, ambito, min_t)
    pr = pagerank_areas(yearly, damping)
    area = ia.merge(pr, on=["areaId", "year", "userProfile"], how="outer")

    unmapped = sorted(
        (set(yearly["sourceId"]) | set(yearly["targetId"]))
        - set(area_comune["areaId"])
        - {STAR}
    )
    if unmapped:
        logger.warning(
            "[flows] areas with no comune (kept in the networks, absent from the comune data): %s",
            unmapped,
        )

    long = to_comuni(area, area_comune)
    long = long[long["userProfile"].isin(out_profiles)]
    wide = long.pivot(
        index=["DATA", "ID_COMUNE"], columns="userProfile", values=list(metrics)
    )
    wide.columns = [f"{metric}_{prof.lower()}" for metric, prof in wide.columns]
    wide = wide.reindex(
        columns=[f"{m}_{p.lower()}" for m in metrics for p in out_profiles]
    ).reset_index()
    wide["giorni_coperti"] = wide["DATA"].map(coverage).astype(int)
    return wide.sort_values(["DATA", "ID_COMUNE"]).reset_index(drop=True)
