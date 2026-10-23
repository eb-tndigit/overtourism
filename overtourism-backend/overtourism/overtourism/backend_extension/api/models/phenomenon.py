from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Process-level raw-file cache
# Only the *raw* bytes from disk are shared here; each Phenomenon gets its
# own resolved panel in self._panel (set during resolve()).
# ---------------------------------------------------------------------------
_FILE_CACHE: dict[str, pd.DataFrame] = {}


# ---------------------------------------------------------------------------
# Phenomenon
# ---------------------------------------------------------------------------


class Phenomenon:
    """
    Base class for a single measurable phenomenon.

    Resolution model
    ----------------
    Every phenomenon declares its *native* resolution on two independent axes:

    Temporal axis
    ~~~~~~~~~~~~~
    ``temporal_resolution`` : 'daily' | 'monthly' | 'yearly'
        Granularity of the source data.

    ``temporal_strategy`` : 'identity' | 'constant' | 'weighted'
        How to bring the phenomenon to daily grain.

        identity  — data is already daily; no transformation needed.
        constant  — a period value is broadcast uniformly to every day in
                    that period (cheap, vectorised; done lazily at startup).
        weighted  — the period total is distributed across days in proportion
                    to a reference phenomenon's daily profile. Computationally
                    expensive; should be done *eagerly* offline (preprocessing
                    script writes a daily Parquet that is loaded as-is).
                    At runtime, the source path is expected to already contain
                    pre-disaggregated daily data, so ``temporal_strategy``
                    effectively becomes 'identity' after preprocessing.

    Spatial axis
    ~~~~~~~~~~~~
    ``spatial_resolution`` : 'comune' | 'macro-area' | 'provincia'
        Granularity of the source data.

    ``spatial_strategy`` : 'identity' | 'constant' | 'weighted'
        How to bring the phenomenon to comune grain.

        identity  — data already has one row per comune.
        constant  — a coarser-level value is broadcast uniformly to all
                    comuni in that unit (cheap, vectorised; done at startup).
        weighted  — the coarser total is apportioned across comuni according
                    to each comune's share of a reference phenomenon.
                    Same eager/offline recommendation as temporal weighted.

    After ``resolve()`` every phenomenon exposes a uniform panel:

        self._panel : pd.DataFrame
            Columns: [ID_COMUNE (str), DATA (datetime64), <self.name> (float)]
            One row per (comune, day) combination, already at the canonical
            finest grain. ``_filter_by_date`` is then a trivial boolean mask.

    Parameters
    ----------
    source : str | Path
        Path to a CSV or Parquet file.
    col : str
        Physical column name in the source file that holds the values.
    agg : str
        Aggregation function when collapsing multiple rows per comune per day
        after resolution: ``'sum'`` or ``'mean'`` or ``'max'`` or ``'min'``.
    municipality_id_col : str, optional
        Name of the municipality key column (default ``"ID_COMUNE"``).
    date_col : str, optional
        Name of the date / period column (default ``"DATA"``).
    temporal_resolution : str, optional
        Overrides the class-level default.
    temporal_strategy : str, optional
        Overrides the class-level default.
    spatial_resolution : str, optional
        Overrides the class-level default.
    spatial_strategy : str, optional
        Overrides the class-level default.
    support_temporal : Phenomenon, optional
        Reference phenomenon for ``temporal_strategy='weighted'``.
    support_spatial : Phenomenon, optional
        Reference phenomenon for ``spatial_strategy='weighted'``.
    assign_temporal_fn : callable, optional
        Custom ``(period_value, support_daily_series) -> daily_series``
        used instead of simple proportional split for temporal weighted.
    assign_spatial_fn : callable, optional
        Custom ``(region_value, support_comune_series) -> comune_series``
        used instead of simple proportional split for spatial weighted.
    all_comuni : list[str], optional
        Ordered list of all comune IDs in the region. Required when
        ``spatial_strategy != 'identity'`` so the broadcast/weight knows
        the full target set.
    sep : str, optional
        CSV column separator (default ``";"``).
    dtype : dict, optional
        Column-to-dtype mapping passed to the CSV reader.
    """

    # ── Class-level defaults (override in subclasses or via __init__) ──────
    name: str = ""

    temporal_resolution: str = "daily"  # 'daily' | 'monthly' | 'yearly'
    temporal_strategy: str = "identity"  # 'identity' | 'constant' | 'weighted'

    spatial_resolution: str = "comune"  # 'comune' | 'macro-area' | 'provincia'
    spatial_strategy: str = "identity"  # 'identity' | 'constant' | 'weighted'

    # False: the value column is text (e.g. a label), kept as it is; `agg` must then be a text-friendly one
    # ('first' / 'last'). Only for descriptive columns shown next to an index, never for values to compute with.
    numeric: bool = True

    support_temporal: Phenomenon | None = None
    support_spatial: Phenomenon | None = None
    assign_temporal_fn: Callable | None = None
    assign_spatial_fn: Callable | None = None

    def __init__(
        self,
        source: str | Path,
        col: str,
        agg: str = "sum",
        municipality_id_col: str | None = None,
        date_col: str | None = None,
        temporal_resolution: str | None = None,
        temporal_strategy: str | None = None,
        spatial_resolution: str | None = None,
        spatial_strategy: str | None = None,
        support_temporal: Phenomenon | None = None,
        support_spatial: Phenomenon | None = None,
        assign_temporal_fn: Callable | None = None,
        assign_spatial_fn: Callable | None = None,
        all_comuni: list[str] | None = None,
        sep: str = ";",
        dtype: dict | None = None,
    ):
        self.source = source
        self.col = col
        self.agg = agg
        self.municipality_id_col = municipality_id_col or "ID_COMUNE"
        self.date_col = date_col or "DATA"
        self.sep = sep
        self.dtype = dtype
        self.all_comuni = all_comuni  # needed for non-identity spatial strategies

        if temporal_resolution is not None:
            self.temporal_resolution = temporal_resolution
        if temporal_strategy is not None:
            self.temporal_strategy = temporal_strategy
        if spatial_resolution is not None:
            self.spatial_resolution = spatial_resolution
        if spatial_strategy is not None:
            self.spatial_strategy = spatial_strategy
        if support_temporal is not None:
            self.support_temporal = support_temporal
        if support_spatial is not None:
            self.support_spatial = support_spatial
        if assign_temporal_fn is not None:
            self.assign_temporal_fn = assign_temporal_fn
        if assign_spatial_fn is not None:
            self.assign_spatial_fn = assign_spatial_fn

        # Resolved daily × comune panel — populated once by resolve()
        self._panel: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self) -> Phenomenon:
        """
        Materialise self._panel: the full daily × comune DataFrame.

        Calling this more than once is a no-op.  The Indicator layer calls
        this at startup (or on first use) so that every subsequent
        _filter_by_date() is a cheap boolean mask with no disaggregation.
        """
        if self._panel is not None:
            return self

        raw = self._read_source()
        df = self._normalise_columns(raw)

        # ── Step 1: resolve temporal axis ─────────────────────────────────
        df = self._resolve_temporal(df)

        # ── Step 2: resolve spatial axis ──────────────────────────────────
        df = self._resolve_spatial(df)

        # ── Step 3: final tidy-up ─────────────────────────────────────────
        df = df[[self.municipality_id_col, "DATA", self.name]].rename(
            columns={self.municipality_id_col: "ID_COMUNE"}
        )
        df["ID_COMUNE"] = df["ID_COMUNE"].astype(str)
        df["DATA"] = pd.to_datetime(df["DATA"])
        if self.numeric:
            df[self.name] = pd.to_numeric(df[self.name], errors="coerce")

        # Collapse any residual duplicate (comune, date) pairs that might
        # arise from overlapping source rows after disaggregation.
        df = (
            df.groupby(["ID_COMUNE", "DATA"], sort=False)[self.name]
            .agg(self.agg)
            .reset_index()
        )

        self._panel = df
        return self

    # Keep load() as an alias so existing call-sites don't break.
    def load(self) -> Phenomenon:
        return self.resolve()

    def _filter_by_date(
        self,
        start_date=None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Return the slice of self._panel within [start_date, end_date].

        After resolve() this is a plain vectorised boolean mask — no loops,
        no disaggregation.  Call resolve() before calling this.
        """
        if self._panel is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.resolve() must be called before _filter_by_date()"
            )
        if start_date is None or end_date is None:
            return pd.DataFrame(columns=self._panel.columns)

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        mask = (self._panel["DATA"] >= start) & (self._panel["DATA"] <= end)
        return self._panel.loc[mask]

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse a filtered slice to one row per comune."""
        return df.groupby("ID_COMUNE")[self.name].agg(self.agg).reset_index()

    def date_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Return (min_date, max_date) available in this phenomenon's resolved
        panel. Calls resolve() if it hasn't run yet.
        """
        self.resolve()
        if self._panel is None or self._panel.empty:
            raise ValueError(f"{self.name}: no data available to compute date bounds")
        return self._panel["DATA"].min(), self._panel["DATA"].max()

    # ------------------------------------------------------------------
    # Source loading
    # ------------------------------------------------------------------

    def _read_source(self) -> pd.DataFrame:
        src = str(self.source)
        if src not in _FILE_CACHE:
            if src.endswith(".parquet"):
                _FILE_CACHE[src] = pd.read_parquet(src)
            elif src.endswith(".csv"):
                _FILE_CACHE[src] = pd.read_csv(src, sep=self.sep, dtype=self.dtype)
            else:
                raise NotImplementedError(f"Unsupported file type: {src}")
        return _FILE_CACHE[src].copy()

    def _normalise_columns(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Rename the value column to self.name and keep only needed cols."""
        if self.col not in raw.columns:
            raise KeyError(
                f"{self.__class__.__name__}: column '{self.col}' not found. "
                f"Available: {list(raw.columns)}"
            )
        needed = [self.municipality_id_col, self.date_col, self.col]
        # Some sources may omit the municipality col (e.g. regional-level)
        needed = [c for c in needed if c in raw.columns]
        df = raw[needed].copy()
        df = df.rename(columns={self.col: self.name})
        return df

    # ------------------------------------------------------------------
    # Temporal resolution
    # ------------------------------------------------------------------

    def _resolve_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bring df to daily grain on the temporal axis."""
        if self.temporal_resolution == "daily" or self.temporal_strategy == "identity":
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            return df

        if self.temporal_strategy == "constant":
            return self._temporal_constant(df)

        if self.temporal_strategy == "weighted":
            return self._temporal_weighted(df)

        raise ValueError(
            f"{self.name}: unknown temporal_strategy={self.temporal_strategy!r}"
        )

    def _temporal_constant(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Broadcast each period value uniformly to every constituent day.
        Fully vectorised — no iterrows().
        """
        rows = []
        for period_val, group in df.groupby(self.date_col):
            p_start, p_end = self._period_bounds(str(period_val))
            dates = pd.date_range(p_start, p_end, freq="D")

            # Repeat each comune's row for every day in the period
            expanded = group.loc[group.index.repeat(len(dates))].copy()
            expanded[self.date_col] = np.tile(dates, len(group))
            rows.append(expanded)

        if not rows:
            return df.iloc[0:0]  # empty, same columns

        result = pd.concat(rows, ignore_index=True)
        result[self.date_col] = pd.to_datetime(result[self.date_col])
        return result

    def _temporal_weighted(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Distribute each period value across days proportionally to a
        support phenomenon's daily profile.

        NOTE: For genuinely large phenomena this should be run *offline*
        (preprocessing script → Parquet) and the daily Parquet loaded
        directly as temporal_resolution='daily' / temporal_strategy='identity'.
        This path is provided for convenience on smaller datasets.
        """
        if self.support_temporal is None:
            raise ValueError(
                f"{self.name}: temporal_strategy='weighted' requires support_temporal"
            )

        support = self.support_temporal
        support.resolve()

        rows = []
        for period_val, group in df.groupby(self.date_col):
            p_start, p_end = self._period_bounds(str(period_val))
            dates = pd.date_range(p_start, p_end, freq="D")

            for _, row in group.iterrows():
                comune = row[self.municipality_id_col]
                value = row[self.name]

                sup_slice = support._filter_by_date(
                    p_start.strftime("%Y-%m-%d"), p_end.strftime("%Y-%m-%d")
                )
                sup_series = (
                    sup_slice[sup_slice["ID_COMUNE"] == str(comune)]
                    .set_index("DATA")[support.name]
                    .reindex(dates)
                    .fillna(0.0)
                )

                if self.assign_temporal_fn is not None:
                    daily_values = self.assign_temporal_fn(value, sup_series).to_numpy()
                else:
                    total = sup_series.sum()
                    daily_values = (
                        value * (sup_series / total).to_numpy()
                        if total != 0
                        else np.full(len(dates), value / len(dates))
                    )

                rows.append(
                    pd.DataFrame(
                        {
                            self.municipality_id_col: comune,
                            self.date_col: dates,
                            self.name: daily_values,
                        }
                    )
                )

        if not rows:
            return df.iloc[0:0]
        result = pd.concat(rows, ignore_index=True)
        result[self.date_col] = pd.to_datetime(result[self.date_col])
        return result

    # ------------------------------------------------------------------
    # Spatial resolution
    # ------------------------------------------------------------------

    def _resolve_spatial(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bring df to comune grain on the spatial axis."""
        if self.spatial_resolution == "comune" or self.spatial_strategy == "identity":
            return df

        if self.spatial_strategy == "constant":
            return self._spatial_constant(df)

        if self.spatial_strategy == "weighted":
            return self._spatial_weighted(df)

        raise ValueError(
            f"{self.name}: unknown spatial_strategy={self.spatial_strategy!r}"
        )

    def _spatial_constant(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Broadcast a coarser spatial value uniformly to all comuni.
        Requires self.all_comuni to be set.
        """
        if not self.all_comuni:
            raise ValueError(
                f"{self.name}: spatial_strategy='constant' requires all_comuni"
            )

        # df has no municipality_id_col at this point (regional/provincial source)
        # Repeat every date-row once per comune.
        rows = []
        for date_val, group in df.groupby(self.date_col):
            # One aggregate value per date across all spatial units in source
            agg_value = group[self.name].agg(self.agg)
            frame = pd.DataFrame(
                {
                    self.municipality_id_col: self.all_comuni,
                    self.date_col: date_val,
                    self.name: agg_value,
                }
            )
            rows.append(frame)

        if not rows:
            return df.iloc[0:0]
        return pd.concat(rows, ignore_index=True)

    def _spatial_weighted(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apportion a coarser spatial value across comuni according to each
        comune's share of a reference phenomenon.
        """
        if self.support_spatial is None:
            raise ValueError(
                f"{self.name}: spatial_strategy='weighted' requires support_spatial"
            )
        if not self.all_comuni:
            raise ValueError(
                f"{self.name}: spatial_strategy='weighted' requires all_comuni"
            )

        support = self.support_spatial
        support.resolve()

        rows = []
        for date_val, group in df.groupby(self.date_col):
            agg_value = group[self.name].agg(self.agg)

            sup_slice = support._filter_by_date(date_val, date_val)
            sup_series = (
                sup_slice.set_index("ID_COMUNE")[support.name]
                .reindex(self.all_comuni)
                .fillna(0.0)
            )

            if self.assign_spatial_fn is not None:
                comune_values = self.assign_spatial_fn(agg_value, sup_series).to_numpy()
            else:
                total = sup_series.sum()
                comune_values = (
                    agg_value * (sup_series / total).to_numpy()
                    if total != 0
                    else np.full(len(self.all_comuni), agg_value / len(self.all_comuni))
                )

            rows.append(
                pd.DataFrame(
                    {
                        self.municipality_id_col: self.all_comuni,
                        self.date_col: date_val,
                        self.name: comune_values,
                    }
                )
            )

        if not rows:
            return df.iloc[0:0]
        return pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _period_bounds(self, period_value: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return (period_start, period_end) for a 'YYYY' or 'YYYY-MM' string."""
        if self.temporal_resolution == "yearly":
            start = pd.Timestamp(year=int(period_value), month=1, day=1)
            end = pd.Timestamp(year=int(period_value), month=12, day=31)
        elif self.temporal_resolution == "monthly":
            start = pd.to_datetime(period_value + "-01")
            end = start + pd.offsets.MonthEnd(0)
        else:
            raise ValueError(
                f"_period_bounds called for temporal_resolution={self.temporal_resolution!r}"
            )
        return start, end

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"temporal={self.temporal_resolution}/{self.temporal_strategy} "
            f"spatial={self.spatial_resolution}/{self.spatial_strategy}>"
        )
