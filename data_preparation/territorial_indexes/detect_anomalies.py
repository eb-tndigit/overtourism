"""
Detect unusual DAILY observations in the final territorial-index phenomena.

Run from:
    data_preparation/territorial_indexes/

Examples:
    python detect_anomalies.py
    python detect_anomalies.py --datasets phen_presenze phen_consumption
    python detect_anomalies.py --contamination 0.02 --min-observations 120
    python detect_anomalies.py --severity 2

The script is read-only with respect to the original phenomena. It writes:
    Output/data/final_data/anomaly_detection_scores.<format>

Each municipality is modelled separately through time. This helps avoid flagging
large municipalities simply because their absolute values are larger. Calendar
features help the model distinguish seasonal patterns. Annual phenomena are
not included by default because the current time series is too short for
reliable municipality-level anomaly detection.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from utils.config import FINAL_DIR, TYPE_FORMAT

LOG = logging.getLogger("anomaly_detection")

DEFAULT_DATASETS = [
    "phen_presenze",
    "phen_consumption",
    "phen_meteo",
]

KEY_COLUMNS = ["DATA", "ID_COMUNE"]


def read_phenomenon(name: str, file_format: str) -> pd.DataFrame:
    path = FINAL_DIR / f"{name}.{file_format}"
    if not path.exists():
        LOG.warning("Skipping %s: file not found (%s)", name, path)
        return pd.DataFrame()

    if file_format == "parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={"ID_COMUNE": "string"})

    missing = [col for col in KEY_COLUMNS if col not in df.columns]
    if missing:
        LOG.warning("Skipping %s: missing key columns %s", name, missing)
        return pd.DataFrame()

    df = df.copy()
    df["ID_COMUNE"] = df["ID_COMUNE"].astype("string").str.zfill(6)
    df["_parsed_date"] = pd.to_datetime(df["DATA"], errors="coerce")
    df = df[df["_parsed_date"].notna()].copy()
    return df


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(KEY_COLUMNS + ["_parsed_date"])
    features = []
    for col in df.columns:
        if col in excluded:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= 10 and values.nunique(dropna=True) > 1:
            features.append(col)
    return features


def prepare_features(group: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    """Prepare numeric variables and cyclical calendar features."""
    x = pd.DataFrame(index=group.index)

    for col in value_columns:
        values = pd.to_numeric(group[col], errors="coerce")
        # Log transform non-negative, strongly right-skewed quantities.
        if values.notna().any() and values.min(skipna=True) >= 0:
            values = np.log1p(values)
        x[col] = values

    dates = group["_parsed_date"]
    month = dates.dt.month
    weekday = dates.dt.dayofweek
    day_of_year = dates.dt.dayofyear

    x["month_sin"] = np.sin(2 * np.pi * month / 12)
    x["month_cos"] = np.cos(2 * np.pi * month / 12)
    x["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    x["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    x["year_day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    x["year_day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Median imputation is performed within the municipality's own time series.
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(axis=1, how="all")
    x = x.fillna(x.median())
    return x


def detect_in_phenomenon(
    name: str,
    df: pd.DataFrame,
    contamination: float,
    min_observations: int,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    value_columns = numeric_feature_columns(df)
    if not value_columns:
        LOG.warning("%s: no usable numeric feature columns", name)
        return pd.DataFrame()

    results = []
    skipped = 0

    # Fit one model per municipality: this detects unusual changes relative to
    # that municipality's own history, rather than differences in municipality size.
    for comune_id, group in df.groupby("ID_COMUNE", dropna=True, sort=False):
        group = group.sort_values("_parsed_date").copy()

        if len(group) < min_observations:
            skipped += 1
            continue

        x = prepare_features(group, value_columns)
        if x.shape[1] < 2 or len(x) < min_observations:
            skipped += 1
            continue

        scaler = RobustScaler()
        x_scaled = scaler.fit_transform(x)

        model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_scaled)

        # Larger score = more anomalous. IsolationForest predicts -1 for anomalies.
        anomaly_score = -model.decision_function(x_scaled)
        prediction = model.predict(x_scaled)

        out = group[KEY_COLUMNS].copy()
        out["phenomenon"] = name
        out["anomaly_score"] = anomaly_score
        out["is_anomaly"] = prediction == -1
        out["features_used"] = ", ".join(x.columns)
        results.append(out)

    if skipped:
        LOG.info(
            "%s: skipped %d municipalities with insufficient observations",
            name,
            skipped,
        )

    if not results:
        LOG.warning("%s: no municipalities had enough data to fit a model", name)
        return pd.DataFrame()

    result = pd.concat(results, ignore_index=True)
    result["DATA"] = result["DATA"].astype(str)
    return result


def apply_severity_filter(scores: pd.DataFrame, severity: int) -> pd.DataFrame:
    """
    Assign a severity level to every scored observation, then apply the
    requested output filter.

    severity_level:
        0 = not flagged by any of the rules
        1 = is_anomaly is True
        2 = abs(anomaly_score) > 0.1
        3 = abs(anomaly_score) > 0.3

    Higher score thresholds take precedence in the severity_level column.
    The output filter follows the user's exact requested rules:
        severity 0: keep everything
        severity 1: keep is_anomaly == True
        severity 2: keep abs(anomaly_score) > 0.1
        severity 3: keep abs(anomaly_score) > 0.3
    """
    scores = scores.copy()
    abs_score = scores["anomaly_score"].abs()

    scores["severity_level"] = 0
    scores.loc[scores["is_anomaly"], "severity_level"] = 1
    scores.loc[abs_score > 0.1, "severity_level"] = 2
    scores.loc[abs_score > 0.3, "severity_level"] = 3

    if severity == 0:
        return scores
    if severity == 1:
        return scores[scores["is_anomaly"]].copy()
    if severity == 2:
        return scores[abs_score > 0.1].copy()
    if severity == 3:
        return scores[abs_score > 0.3].copy()

    raise ValueError("severity must be one of: 0, 1, 2, 3")


def run(
    datasets: list[str],
    file_format: str,
    contamination: float,
    min_observations: int,
    severity: int = 3,
) -> pd.DataFrame:
    all_results = []

    for name in datasets:
        LOG.info("Analysing %s", name)
        df = read_phenomenon(name, file_format)
        result = detect_in_phenomenon(
            name=name,
            df=df,
            contamination=contamination,
            min_observations=min_observations,
        )
        if not result.empty:
            all_results.append(result)

    if not all_results:
        raise RuntimeError(
            "No anomaly results produced. Check the final_data files, dataset names "
            "and minimum-observations setting."
        )

    scores = pd.concat(all_results, ignore_index=True)
    total_observations = len(scores)

    # Add severity_level to all scored rows, then retain only the requested
    # severity class. Filtering uses absolute anomaly_score as specified.
    scores = apply_severity_filter(scores, severity)
    scores = scores.sort_values(
        ["severity_level", "is_anomaly", "anomaly_score"],
        ascending=[False, False, False],
    )

    output_path = FINAL_DIR / f"anomaly_detection_scores.{file_format}"
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    if file_format == "parquet":
        scores.to_parquet(output_path, index=False)
    else:
        scores.to_csv(output_path, index=False)

    flagged = int(scores["is_anomaly"].sum())
    LOG.info(
        "Severity filter %d: saved %d of %d observations (%d marked is_anomaly=True) to %s",
        severity,
        len(scores),
        total_observations,
        flagged,
        output_path,
    )
    return scores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run municipality-level Isolation Forest anomaly detection."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Final phenomenon names to analyse (daily datasets recommended).",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default=TYPE_FORMAT,
        help="Input/output format (defaults to config/settings.yaml).",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.01,
        help="Expected proportion of anomalies, between 0 and 0.5 (default: 0.01).",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=90,
        help="Minimum observations per municipality (default: 90).",
    )
    parser.add_argument(
        "--severity",
        type=int,
        choices=[0, 1, 2, 3],
        default=3,
        help=(
            "Output filter: 0=all observations; 1=is_anomaly; "
            "2=abs(anomaly_score)>0.1; 3=abs(anomaly_score)>0.3 (default: 3)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    if not 0 < args.contamination < 0.5:
        raise SystemExit("--contamination must be greater than 0 and less than 0.5")
    if args.min_observations < 20:
        raise SystemExit("--min-observations must be at least 20")

    run(
        datasets=args.datasets,
        file_format=args.format,
        contamination=args.contamination,
        min_observations=args.min_observations,
        severity=args.severity,
    )
