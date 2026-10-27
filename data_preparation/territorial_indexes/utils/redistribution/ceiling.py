# SPDX-License-Identifier: Apache-2.0
"""Load ceiling of every node and year (ported from the notebook, same maths).

The ceiling of a node is the load it can carry before it is 'full', estimated from the observed peaks of the comparable
nodes: a quantile regression at `tau` of the yearly 95th percentile of the daily load on structural features
(beds, residents, second homes, surface, seasonal profile). Tourists and excursionists have their own model.
The model of each segment is chosen by cross-validation by node (the simplest within `tolerance` of the best pinball
loss) and the uncertainty is a bootstrap by node. Only the estimates used downstream are kept: the point ceiling and
its bootstrap interval.

Pure functions, no I/O. Parameters: settings.yaml, `redistribution.ceiling`.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools.sm_exceptions import IterationLimitWarning

logger = logging.getLogger(__name__)

SEGMENTS = (("T", "T95"), ("E", "E95"))  # segment -> column of the yearly peak
CLUSTER_DUMMIES = ["c_estate_alpina", "c_estate_spalle", "c_sci"]  # "annuale" is the reference
# Candidate models per segment (simplest first). The notebook also tried POI features (Google reviews): they did not
# improve the cross-validated loss, so the POI inputs are not part of this pipeline.
MODEL_SETS = {
    "T": {
        "letti": ["lb"],
        "letti+cluster": ["lb"] + CLUSTER_DUMMIES,
        "letti+pop+cluster": ["lb", "lp"] + CLUSTER_DUMMIES,
        "letti+pop+seconde+cluster": ["lb", "lp", "ld"] + CLUSTER_DUMMIES,
    },
    "E": {
        "pop": ["lp"],
        "pop+cluster": ["lp"] + CLUSTER_DUMMIES,
        "pop+cluster+superficie": ["lp", "lsurf"] + CLUSTER_DUMMIES,
    },
}


def _features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lb"] = np.log(df["beds"].clip(lower=1))
    df["lp"] = np.log(df["pop"].clip(lower=1))
    df["ld"] = np.log1p(df["disp"])
    df["lsurf"] = np.log(df["surface_km2"])
    for c in ("estate_alpina", "estate_spalle", "sci"):
        df["c_" + c] = (df["cluster"] == c).astype(float)
    return df


def _xy(df, cols, target):
    return np.c_[np.ones(len(df)), df[cols].values], np.log(df[target].clip(lower=1)).values


def _pinball(y, p, tau):
    e = y - p
    return np.mean(np.maximum(tau * e, (tau - 1) * e))


def _cv(df, cols, target, tau, k, seeds):
    """Mean pinball loss and coverage of a K-fold cross-validation BY NODE (all the years of a node in one fold)."""
    ok = df[df["beds"] > 0] if target == "T95" else df
    ids = ok["node"].unique()
    res, cov = [], []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        fold = dict(zip(ids, rng.integers(0, k, len(ids))))
        f = ok["node"].map(fold).values
        for q in range(k):
            tr, te = ok[f != q], ok[f == q]
            X, y = _xy(tr, cols, target)
            b = QuantReg(y, X).fit(q=tau, max_iter=5000).params
            Xt, yt = _xy(te, cols, target)
            p = Xt @ b
            res.append(_pinball(yt, p, tau))
            cov.append(np.mean(yt <= p))
    return np.mean(res), np.mean(cov)


def fit_ceilings(demand: pd.DataFrame, node_year: pd.DataFrame, nodes: pd.DataFrame, params: dict):
    """Returns (ceilings, cv_table, coef_table).

    ceilings: node, year, C_T, C_T_lo, C_T_hi, C_E, C_E_lo, C_E_hi (persons/day; lo/hi = bootstrap interval)
    params:   tau, cv_folds, cv_seeds, tolerance, bootstrap, bootstrap_seed, ci_low, ci_high
    """
    # a few bootstrap resamples hit the iteration cap of the solver (as in the notebook): the estimate is still usable
    warnings.filterwarnings("ignore", category=IterationLimitWarning)
    tau, k, seeds, tol = params["tau"], params["cv_folds"], params["cv_seeds"], params["tolerance"]
    peak = (
        demand.groupby(["node", "year"])
        .agg(T95=("T", lambda s: s.quantile(0.95)), E95=("E", lambda s: s.quantile(0.95)))
        .reset_index()
    )
    d = _features(peak.merge(node_year, on=["node", "year"]).merge(nodes, on="node"))

    best, cv_rows = {}, []
    for seg, tgt in SEGMENTS:
        r = {}
        for name, cols in MODEL_SETS[seg].items():
            loss, cover = _cv(d, cols, tgt, tau, k, seeds)
            r[name] = loss
            cv_rows.append(dict(segmento=seg, modello=name, pinball_loss=loss, copertura=cover))
        floor = min(r.values())
        best[seg] = [n for n, v in r.items() if v <= floor * (1 + tol)][0]  # simplest within `tol` of the best
        for row in cv_rows:
            if row["segmento"] == seg:
                row["scelto"] = row["modello"] == best[seg]
        logger.info("ceiling model %s: %s", seg, best[seg])

    # final fit + bootstrap by node
    rng = np.random.default_rng(params["bootstrap_seed"])
    coefs, boots, coef_rows = {}, {}, []
    for seg, tgt in SEGMENTS:
        cols = MODEL_SETS[seg][best[seg]]
        ok = d[d["beds"] > 0] if seg == "T" else d
        X, y = _xy(ok, cols, tgt)
        m = QuantReg(y, X).fit(q=tau, max_iter=5000)
        coefs[seg] = (cols, m.params)
        ids = ok["node"].unique()
        draws = []
        for _ in range(params["bootstrap"]):
            sample = rng.choice(ids, len(ids))
            sub = pd.concat([ok[ok["node"] == n] for n in sample])
            Xb, yb = _xy(sub, cols, tgt)
            draws.append(QuantReg(yb, Xb).fit(q=tau, max_iter=5000).params)
        boots[seg] = np.array(draws)
        lo, hi = np.percentile(boots[seg], [params["ci_low"], params["ci_high"]], axis=0)
        for nm, b, l, h in zip(["const"] + cols, m.params, lo, hi):
            coef_rows.append(dict(segmento=seg, termine=nm, coef=b, ic90_lo=l, ic90_hi=h))

    # ceilings of every node-year (also of the nodes/years that were not in the fit)
    allny = _features(node_year.merge(nodes, on="node"))
    res = allny[["node", "year"]].copy()
    for seg, _ in SEGMENTS:
        cols, b = coefs[seg]
        X = np.c_[np.ones(len(allny)), allny[cols].values]
        res["C_" + seg] = np.exp(X @ b)
        res[f"C_{seg}_lo"], res[f"C_{seg}_hi"] = np.exp(
            np.percentile(X @ boots[seg].T, [params["ci_low"], params["ci_high"]], axis=1)
        )
    return res, pd.DataFrame(cv_rows), pd.DataFrame(coef_rows)
