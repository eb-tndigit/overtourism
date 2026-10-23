"""Share of the electricity consumption attributable to tourism, per comune and YEAR (data preparation of
`phen_ee_tourism_share`).

Model. For every comune and every calendar year, on the 12 monthly means (WLS, weight = days of the month):

    y_m = a + b * x_m + c * hdd_m + e_m

    y_m    mean daily consumption of the month (kWh/day)
    x_m    mean daily tourist presences of the month (Vodafone TOURIST profile)
    hdd_m  mean daily heating degree days of the month (temperature control)
    b      kWh consumed per tourist-day: an intermediate value, it is NOT published

    share = b * mean(x) / mean(y)      the part of the mean consumption that goes with the tourists (0-1)

A year is estimated only if the comune has all its 12 months (a month needs MIN_DAYS_PER_MONTH days with consumption,
presences and temperature): a missing month means no value for that comune-year, never a zero.

The estimate of b of ONE comune, on 12 months, is too unstable to be published as it is (on the data of the project
the single-comune fits gave slopes of 6-17 kWh per tourist-day and a share of 100% in several comuni). Hence:

 1. Partial pooling (empirical Bayes / ridge). Every b is pulled towards the province-wide coefficient b_pool of
    THE SAME YEAR, with a strength that depends on how well the comune alone determines it: a comune with clean data
    keeps its own slope, a noisy one gets b_pool. b_pool comes from a panel regression on all the comuni with comune
    AND month fixed effects, i.e. from how much MORE the consumption rises where the tourists rise more, after
    removing the seasonality common to everybody (weather, holidays). The prior spread TAU was chosen by
    leave-one-year-out cross-validation (see TAU).
 2. Tourists only. The excursionists add nothing: their coefficient is ~0 and giving them a weight inflates the
    shares. EXCURSIONIST_WEIGHT = 0.
 3. Monthly, not daily. The energy data are monthly totals spread evenly over the days, so a daily series has one
    independent value per month.
 4. Temperature. Tourists and heating are both seasonal; without HDD the heating is mistaken for a tourist effect.

Reliability (`affidabilita`: "alta" / "media" / "bassa") is judged on the evidence of the comune ALONE, not on the
pooled estimate (shrinking towards b_pool makes the interval narrower and pushes it away from zero, so a comune with no
data of its own would look reliable). With a single year every fit has the same 12 months, so the levels depend on how
clearly the own slope of the comune is above zero (one-sided interval of the plain WLS slope):
  - "alta":  the lower bound of the 95% interval is above zero;
  - "media": only the lower bound of the 80% interval is above zero;
  - "bassa": otherwise (the share is then mostly the prior).
A share of IMPLAUSIBLE_SHARE or more is always "bassa": it would mean that residents and the other activities consume
(almost) nothing. The standard errors are Newey-West (HAC): consecutive months are correlated.

Only pandas / numpy: no dependency on the Indicator framework or on the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Excursionists add nothing (see above). The parameter is kept to be able to test other weights.
EXCURSIONIST_WEIGHT = 0.0
# The model has 3 parameters and the estimate is yearly: all the 12 months of the year are needed.
MIN_MONTHS = 12
# A month of the window with fewer days than this (a window that starts / ends inside a month) is ignored.
MIN_DAYS_PER_MONTH = 7
HAC_LAGS = 2
Z_ONE_SIDED_95 = 1.645  # "alta": the own slope is above zero at this level
Z_ONE_SIDED_80 = 0.842  # "media": it is above zero at this (weaker) level only
# A tourism share at or above this is not credible (nothing left for the residents and the other activities).
IMPLAUSIBLE_SHARE = 0.9
# Standard deviation of the true slopes of the comuni around the province-wide one, in kWh per tourist-day (the
# per-comune slopes of the project data have median 0.6 and inter-quartile range 0.2-1.2). A smaller value pulls
# harder on b_pool. Leave-one-year-out held-out error (mean abs. error of the relative
# consumption): 0.1883 with no pooling, 0.1830 at 1.0, 0.1842 at 0.5, 0.1862 at 0.3, 0.1978 when fully pooled;
# year-to-year Spearman of the shares 0.61 / 0.79 / 0.80 / 0.83. 0.5 keeps the held-out error and most of the stability.
TAU = 0.5
# With fewer comuni than this there is no province to pool on: the comuni are fitted on their own.
MIN_COMUNI_FOR_POOLING = 10
RELIABILITY_HIGH, RELIABILITY_MEDIUM, RELIABILITY_LOW = "alta", "media", "bassa"

# Columns of the result of the fits (after ID_COMUNE)
RESULT_COLUMNS = ["quota_ee_turismo", "presenze_medie", "consumo_medio", "affidabilita"]


def monthly_panel(daily: pd.DataFrame, excursionist_weight: float = EXCURSIONIST_WEIGHT):
    """Daily rows -> one row per comune and calendar month.

    `daily` columns: ID_COMUNE, DATA (datetime), y (consumption), tourists, hdd and, optionally, excursionists.
    Days without consumption are dropped (a null is not a zero). Returns: ID_COMUNE, month, days, y, x, hdd with
    y, x, hdd = mean over the days of the month that are in the window, x = tourists + weight * excursionists.
    """
    d = daily.dropna(subset=["y", "tourists"]).copy()
    excursionists = d["excursionists"].fillna(0) if "excursionists" in d else 0.0
    d["x"] = d["tourists"] + excursionist_weight * excursionists
    d["month"] = d["DATA"].dt.to_period("M")
    monthly = (
        d.groupby(["ID_COMUNE", "month"], sort=True)
        .agg(days=("y", "size"), y=("y", "mean"), x=("x", "mean"), hdd=("hdd", "mean"))
        .reset_index()
    )
    monthly = monthly[monthly["days"] >= MIN_DAYS_PER_MONTH]
    return monthly.dropna(subset=["hdd"])


def _eligible(monthly: pd.DataFrame) -> pd.DataFrame:
    """Months of the comuni that can be estimated: enough months, a consumption, variation in the presences."""
    stats = monthly.groupby("ID_COMUNE").agg(
        n=("y", "size"),
        y_mean=("y", "mean"),
        y_ptp=("y", np.ptp),
        x_max=("x", "max"),
        x_ptp=("x", np.ptp),
    )
    ok = (
        (stats["n"] >= MIN_MONTHS)
        & (stats["y_mean"] > 0)
        & (stats["y_ptp"] > 0)
        & (stats["x_ptp"] > 1e-9 * stats["x_max"].clip(lower=1))
    )
    return monthly[monthly["ID_COMUNE"].isin(stats.index[ok])]


def _two_way_demean(values: pd.DataFrame, comune: pd.Series, month: pd.Series, tol=1e-10, max_iter=200):
    """Remove the comune means and the month means (alternating projections: the panel is unbalanced)."""
    v = values.copy()
    for _ in range(max_iter):
        before = v.to_numpy().copy()
        v = v - v.groupby(comune.values).transform("mean")
        v = v - v.groupby(month.values).transform("mean")
        if np.max(np.abs(v.to_numpy() - before)) < tol:
            break
    return v


def pooled_slope(monthly_rel: pd.DataFrame) -> float | None:
    """Province-wide slope b_pool (in relative units): regression of the relative consumption on the relative
    presences with comune and month fixed effects. None if there are too few comuni to pool."""
    if monthly_rel["ID_COMUNE"].nunique() < MIN_COMUNI_FOR_POOLING:
        return None
    v = _two_way_demean(
        monthly_rel[["yr", "xr"]], monthly_rel["ID_COMUNE"], monthly_rel["month"].astype(str)
    )
    denominator = float((v["xr"] ** 2).sum())
    if denominator <= 0:
        return None
    return float((v["xr"] * v["yr"]).sum() / denominator)


def _hac_covariance(A: np.ndarray, X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West sandwich covariance A^-1 S A^-1 (X and resid already weighted, A = X'X + penalty)."""
    n, k = X.shape
    scores = X * resid[:, None]
    s = scores.T @ scores
    for lag in range(1, min(lags, n - 1) + 1):
        gamma = scores[lag:].T @ scores[:-lag]
        s += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    bread = np.linalg.inv(A)
    return bread @ s @ bread * n / max(n - k, 1)


def fit_months(monthly: pd.DataFrame, b_pool: float | None = None, tau: float = TAU) -> dict | None:
    """Fit of one comune (rows = months). `b_pool` None: no pooling (plain WLS). None if it cannot be estimated.

    Consumption and presences are divided by the mean consumption of the comune: the slope (kWh per tourist-day) does
    not change, but every comune weighs the same, so the big ones do not decide the pooled slope and `tau` has the
    same meaning everywhere.
    """
    m = monthly.dropna(subset=["y", "x", "hdd"]).sort_values("month")
    n = len(m)
    if n < MIN_MONTHS:
        return None
    w = m["days"].to_numpy(float)
    y_bar = float(np.average(m["y"], weights=w))
    if y_bar <= 0:
        return None
    yr, xr = m["y"].to_numpy(float) / y_bar, m["x"].to_numpy(float) / y_bar
    h = m["hdd"].to_numpy(float)
    if np.ptp(xr) <= 1e-9 * max(abs(xr).max(), 1) or np.ptp(yr) == 0:
        return None
    X = np.column_stack([np.ones(n), xr, h])
    if np.linalg.matrix_rank(X) < 3:  # e.g. hdd constant: the temperature cannot be separated
        X = X[:, :2]
    k = X.shape[1]
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], yr * sw

    beta_ols, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid_ols = yw - Xw @ beta_ols
    sigma2 = float(resid_ols @ resid_ols) / max(n - k, 1)
    penalty = np.zeros((k, k))
    center = 0.0
    if b_pool is not None and tau > 0 and sigma2 > 0:
        penalty[1, 1] = sigma2 / tau**2
        center = b_pool
    A = Xw.T @ Xw + penalty
    beta = np.linalg.solve(A, Xw.T @ (yw - center * Xw[:, 1])) + np.array([0, center] + [0] * (k - 2))
    resid = yw - Xw @ beta
    se = np.sqrt(np.diag(_hac_covariance(A, Xw, resid, HAC_LAGS)))

    slope_rel, se_rel = float(beta[1]), float(se[1])
    x_bar_rel = float(np.average(xr, weights=w))
    raw_share = slope_rel * x_bar_rel
    share = min(max(raw_share, 0.0), 1.0)
    # evidence of the comune alone: plain WLS slope and its HAC interval (no pull towards b_pool)
    se_own = np.sqrt(np.diag(_hac_covariance(Xw.T @ Xw, Xw, resid_ols, HAC_LAGS)))
    lower_95 = (float(beta_ols[1]) - Z_ONE_SIDED_95 * float(se_own[1])) * x_bar_rel
    lower_80 = (float(beta_ols[1]) - Z_ONE_SIDED_80 * float(se_own[1])) * x_bar_rel
    if raw_share >= IMPLAUSIBLE_SHARE or lower_80 <= 0:
        reliability = RELIABILITY_LOW
    elif lower_95 <= 0:
        reliability = RELIABILITY_MEDIUM
    else:
        reliability = RELIABILITY_HIGH
    return {
        "quota_ee_turismo": share,  # share of the consumption attributable to tourism (0-1)
        "presenze_medie": float(np.average(m["x"], weights=w)),
        "consumo_medio": y_bar,
        "affidabilita": reliability,
    }


def fit_comuni(
    daily: pd.DataFrame, excursionist_weight: float = EXCURSIONIST_WEIGHT, tau: float = TAU
) -> pd.DataFrame:
    """One row per comune with a valid estimate (ID_COMUNE + RESULT_COLUMNS) on the months of `daily` (one year);
    comuni that cannot be estimated (no data, < MIN_MONTHS months, no variation) are absent, they are not zeros."""
    monthly = _eligible(monthly_panel(daily, excursionist_weight))
    empty = pd.DataFrame(columns=["ID_COMUNE"] + RESULT_COLUMNS)
    if monthly.empty:
        return empty
    scale = monthly.groupby("ID_COMUNE").apply(
        lambda g: np.average(g["y"], weights=g["days"]), include_groups=False
    )
    rel = monthly.assign(yr=monthly["y"] / monthly["ID_COMUNE"].map(scale), xr=monthly["x"] / monthly["ID_COMUNE"].map(scale))
    b_pool = pooled_slope(rel)
    rows = []
    for comune, group in monthly.groupby("ID_COMUNE", sort=True):
        fit = fit_months(group, b_pool, tau)
        if fit is not None:
            rows.append({"ID_COMUNE": comune, **fit})
    return pd.DataFrame(rows) if rows else empty


def fit_years(
    daily: pd.DataFrame, excursionist_weight: float = EXCURSIONIST_WEIGHT, tau: float = TAU
) -> pd.DataFrame:
    """The model run separately on every calendar year of `daily`.

    `daily` columns: ID_COMUNE, DATA (datetime), y, tourists, hdd and, optionally, excursionists.
    Returns DATA (year, str), ID_COMUNE + RESULT_COLUMNS, one row per comune-year with a valid estimate."""
    parts = []
    for year, group in daily.groupby(daily["DATA"].dt.year, sort=True):
        fits = fit_comuni(group, excursionist_weight, tau)
        if not fits.empty:
            parts.append(fits.assign(DATA=str(year)))
    columns = ["DATA", "ID_COMUNE"] + RESULT_COLUMNS
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)[columns]
