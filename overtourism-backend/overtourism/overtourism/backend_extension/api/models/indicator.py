from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from overtourism.overtourism.backend_extension.api.models.phenomenon import Phenomenon
from overtourism.overtourism.backend_extension.api.utils.index_utils import get_chart_labels

# ---------------------------------------------------------------------------
# Indicator
# ---------------------------------------------------------------------------


class Indicator:
    """
    Owns a list of Phenomena, a combinator g_k, and all computation logic.

    After each Phenomenon is resolved (daily × comune panel), every operation
    here is a cheap vectorised slice + groupby — no disaggregation happens at
    query time.

    Composability (macro-indicators)
    ---------------------------------
    An Indicator can itself be passed inside another Indicator's
    ``phenomena`` list, to build a macro-indicator out of several
    sub-indicators. This works because Indicator
    implements the same duck-typed interface Phenomenon exposes — `.name`,
    `.agg`, `.resolve()`, `._filter_by_date()`, `.aggregate()`,
    `.date_bounds()` — so `_ensure_resolved()` / `_compute_indicator()` /
    `_compute_label_metrics()` treat a sub-Indicator exactly like any other
    Phenomenon, no adapter class required. See `resolve()` below for the
    one important caveat this implies for combinators.

    Parameters
    ----------
    phenomena : list[Phenomenon | Indicator]
        Data sources, already configured. May include other Indicators to
        build a macro-indicator (see "Composability" above).
    combinator : Callable[[pd.DataFrame, ...], pd.Series]
        Pure function that receives a DataFrame (one column per phenomenon
        name, one row per comune or per (comune, date)) and returns the
        INDICE Series.  Signature:
            fn(df, *, filtered_data, start_date, end_date, **extra) -> Series
    """

    name: str = ""
    type: str = ""
    description: str = ""
    category: str = ""
    order: int | None = None
    index_value_unit_description = "Valore indice territoriale"
    availableForVariation: bool = True
    extraFields: list[str] = []
    internal_only = False
    year_only = False
    _years_range: dict | None = None

    # How this indicator's own daily INDICE values collapse to one row per
    # comune when *it* is used as a phenomenon by a parent Indicator (see
    # `aggregate()` below). 'mean' is the sensible default for an index —
    # summing a ratio across days rarely means anything.
    agg: str = "mean"

    # How the columns next to INDICE collapse to one row per macro area / province (see
    # TerritorialConfig.column_aggs): {column: aggregation}, e.g. "mode" for a category. Columns not listed
    # follow the TerritorialConfig defaults (numeric -> sum, text -> dropped).
    column_aggs: dict[str, str] = {}

    def __init__(
        self,
        phenomena: list[Phenomenon | Indicator],
        combinator: Callable[[pd.DataFrame], pd.Series],
    ):
        self.phenomena = phenomena
        self.combinator = combinator

        # Result caches keyed by call arguments.
        # Indicators are process-level singletons; source data doesn't change
        # at runtime, so caching by (start_date, end_date, ...) is safe.
        self._indicator_cache: dict = {}
        self._variation_cache: dict = {}

        # Resolved daily × comune panel — only populated (via resolve()) if
        # this Indicator is used as a phenomenon inside another Indicator.
        self._panel: pd.DataFrame | None = None

    @staticmethod
    def _cache_key(*args, **kwargs) -> tuple:
        return (args, tuple(sorted(kwargs.items())))

    def phenomenon_names(self) -> list[str]:
        return [p.name for p in self.phenomena]

    @property
    def years_range(self) -> dict[str, int]:
        """
        {min_year, max_year} for which *every* phenomenon in this indicator
        has data for the *entire* calendar year (Jan 1 to Dec 31) — i.e. the
        intersection of each phenomenon's own [min_date, max_date], not the
        union, restricted to full years only.

        This matters because get_indicator()/get_temporal_variation() return
        None for any window where even one phenomenon is empty, so the range
        actually usable for this indicator is the latest of all min dates to
        the earliest of all max dates. Partial years at either edge of that
        overlap are excluded, since they don't have a full 01-01 to 31-12
        span covered by every phenomenon.
        """
        if self._years_range is None:
            self._ensure_resolved()
            bounds = [p.date_bounds() for p in self.phenomena]

            overall_start = max(b[0] for b in bounds)
            overall_end = min(b[1] for b in bounds)

            if overall_start > overall_end:
                raise ValueError(
                    f"{self!r}: phenomena have no overlapping date range "
                    f"(latest start={overall_start.date()}, "
                    f"earliest end={overall_end.date()})"
                )

            # If the overlap doesn't start on Jan 1, the first partial year
            # isn't fully covered, so bump to the next year.
            min_year = overall_start.year
            if (overall_start.month, overall_start.day) != (1, 1):
                min_year += 1

            # If the overlap doesn't end on Dec 31, the last partial year
            # isn't fully covered, so drop to the previous year.
            max_year = overall_end.year
            if (overall_end.month, overall_end.day) != (12, 31):
                max_year -= 1

            if min_year > max_year:
                raise ValueError(
                    f"{self!r}: phenomena have no overlapping *full* calendar "
                    f"year (overlap is {overall_start.date()} to "
                    f"{overall_end.date()}, which contains no complete "
                    f"01-01 to 31-12 span)"
                )

            self._years_range = {
                "min_year": min_year,
                "max_year": max_year,
            }
        return self._years_range

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"phenomena={self.phenomenon_names()} "
            f"combinator={getattr(self.combinator, '__name__', '?')}>"
        )

    # ------------------------------------------------------------------
    # Numerical safety
    # ------------------------------------------------------------------

    @classmethod
    def _safe_std(cls, value) -> float:
        """
        Normalize a computed std value: NaN -> 0.0, and any value that's
        within floating-point noise of zero -> 0.0 (pandas' .std() can leave
        tiny non-zero residues like 1.4e-16 for groups of identical values
        due to catastrophic cancellation in the variance formula).
        """
        _STD_EPS = 1e-9  # below this, treat as floating-point noise around zero
        if pd.isna(value):
            return 0.0
        value = float(value)
        return 0.0 if abs(value) < _STD_EPS else value

    @staticmethod
    def _round_or_none(value, ndigits: int = 2):
        """Rounded value for the chart series; None (null) when there is no value (label without data, NaN)."""
        return None if value is None or pd.isna(value) else round(float(value), ndigits)

    # ------------------------------------------------------------------
    # Built-in combinators
    # ------------------------------------------------------------------

    def divide(self, numerator: str, denominator: str) -> Callable:
        """Return a combinator that computes numerator / denominator per row."""

        def _fn(df: pd.DataFrame, **_extra) -> pd.Series:
            return df[numerator] / df[denominator].replace(0, np.nan)

        _fn.__name__ = f"divide({numerator}, {denominator})"
        return _fn

    # ------------------------------------------------------------------
    # Ensure all phenomena are resolved (call once at startup or on
    # first use; subsequent calls are no-ops inside resolve()).
    # ------------------------------------------------------------------

    def _ensure_resolved(self) -> None:
        for p in self.phenomena:
            p.resolve()

    # ------------------------------------------------------------------
    # Phenomenon-compatible interface (lets this Indicator be used as a
    # phenomenon inside another Indicator — see class docstring).
    # ------------------------------------------------------------------

    def resolve(self) -> Indicator:
        """
        Materialise self._panel: [ID_COMUNE, DATA, self.name], one row per
        (comune, day), by merging every phenomenon's own resolved daily
        panel and applying this indicator's combinator row-wise.

        Only relevant when this Indicator is used as a phenomenon inside a
        parent Indicator; get_indicator()/get_temporal_variation() don't go
        through this path. Calling
        this more than once is a no-op, mirroring Phenomenon.resolve().

        Caveat: this panel is computed *once*, over the full range each
        underlying phenomenon has data for, with start_date=None,
        end_date=None and no extra kwargs (there is no single query these
        values could come from — the panel has to hold for every future
        date-range slice a parent Indicator might request). Combinators
        used on an Indicator that will be composed this way must therefore
        be pure functions of the row values alone (like `divide()`); they
        must not rely on `start_date`/`end_date`/`filtered_data`/`**extra`
        for anything beyond ignoring them, or the composed values will be
        wrong.
        """
        if self._panel is not None:
            return self

        self._ensure_resolved()

        daily_frames: dict[str, pd.DataFrame] = {}
        for phenom in self.phenomena:
            panel = phenom._panel
            if panel is None:
                raise RuntimeError(f"{phenom!r}: resolve() did not populate _panel")
            daily_frames[phenom.name] = (
                panel.groupby(["ID_COMUNE", "DATA"])[phenom.name]
                .agg(phenom.agg)
                .reset_index()
            )

        merged = self._merge_daily(daily_frames, on=["ID_COMUNE", "DATA"])
        if merged.empty:
            self._panel = pd.DataFrame(columns=["ID_COMUNE", "DATA", self.name])
            return self

        merged[self.name] = self.combinator(
            merged,
            filtered_data={p.name: p._panel for p in self.phenomena},
            start_date=None,
            end_date=None,
        )
        self._panel = merged[["ID_COMUNE", "DATA", self.name]]
        return self

    def _filter_by_date(
        self,
        start_date=None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Same contract as Phenomenon._filter_by_date(); requires resolve() first."""
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
        """Same contract as Phenomenon.aggregate(): collapse a filtered slice to one row per comune."""
        return df.groupby("ID_COMUNE")[self.name].agg(self.agg).reset_index()

    def date_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Same contract as Phenomenon.date_bounds()."""
        self.resolve()
        if self._panel is None or self._panel.empty:
            raise ValueError(f"{self.name}: no data available to compute date bounds")
        return self._panel["DATA"].min(), self._panel["DATA"].max()

    # ------------------------------------------------------------------
    # get_indicator
    # ------------------------------------------------------------------

    def get_indicator(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        **extra,
    ) -> pd.DataFrame | None:
        key = self._cache_key(start_date, end_date, **extra)
        if key in self._indicator_cache:
            cached = self._indicator_cache[key]
            return cached.copy() if cached is not None else None

        result = self._compute_indicator(start_date, end_date, **extra)
        self._indicator_cache[key] = result
        return result.copy() if result is not None else None

    def _compute_indicator(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        **extra,
    ) -> pd.DataFrame | None:
        self._ensure_resolved()

        # ── 1. Filter and aggregate each phenomenon ──
        aggregated: dict[str, pd.DataFrame] = {}
        filtered_data: dict[str, pd.DataFrame] = {}

        for phenom in self.phenomena:
            filtered = phenom._filter_by_date(start_date, end_date)
            if filtered.empty:
                return None
            filtered_data[phenom.name] = filtered
            aggregated[phenom.name] = phenom.aggregate(filtered)

        # ── 2. Merge all phenomena into one wide DataFrame ────────────────
        result = self._merge_phenomena(aggregated)
        if result.empty:
            return None

        # ── 3. Apply combinator g_k ───────────────────────────────────────
        result["INDICE"] = self.combinator(
            result,
            filtered_data=filtered_data,
            start_date=start_date,
            end_date=end_date,
            **extra,
        )
        # INDICE is NOT filled: a null means "no value" (missing data, division by zero) and is returned as such.
        # A combinator that wants a value for a missing input must say so explicitly.
        return result

    # ------------------------------------------------------------------
    # get_temporal_variation
    # ------------------------------------------------------------------

    def get_temporal_variation(
        self,
        *,
        start_date,
        end_date,
        granularity: str,
        region_id: str = "-1",
        **extra,
    ) -> list[dict]:
        key = self._cache_key(
            start_date, end_date, granularity, region_id, **extra
        )
        if key in self._variation_cache:
            return self._variation_cache[key]

        self._ensure_resolved()

        labels = get_chart_labels(start_date, end_date, granularity)
        ranges = self._label_date_ranges(labels, granularity, start_date, end_date)

        per_comune: dict = {}
        per_comune_std: dict = {}
        region_series: dict = {}
        region_std_series: dict = {}

        for label, (sub_start, sub_end) in zip(labels, ranges):
            metrics = self._compute_label_metrics(
                sub_start, sub_end, **extra
            )
            if metrics is None:
                continue
            comune_result, std_by_comune, region_value, region_std = metrics

            for _, row in comune_result.iterrows():
                comune = row["ID_COMUNE"]
                per_comune.setdefault(comune, {})[label] = row["INDICE"]

            for comune, std_val in std_by_comune.items():
                per_comune_std.setdefault(comune, {})[label] = std_val

            region_series[label] = region_value
            region_std_series[label] = region_std

        series = []
        for comune, values in per_comune.items():
            std_values = per_comune_std.get(comune, {})
            series.append(
                {
                    "label": comune,
                    "data": [self._round_or_none(values.get(l)) for l in labels],
                    "std": [self._round_or_none(std_values.get(l)) for l in labels],
                }
            )

        series.append(
            {
                "label": region_id,
                "data": [self._round_or_none(region_series.get(l)) for l in labels],
                "std": [self._round_or_none(region_std_series.get(l)) for l in labels],
            }
        )

        self._variation_cache[key] = series
        return series

    def _compute_label_metrics(
        self, start_date, end_date, **extra
    ) -> tuple | None:
        """
        Single-pass computation for one time-label window.

        Because all phenomena are already resolved to daily × comune panels,
        _filter_by_date() here is just a boolean mask. All four outputs
        (per-comune INDICE, per-comune daily std, region value, region std)
        are built from one consistently-filtered pass per phenomenon.

        Returns None if any phenomenon has no data in the window, otherwise
        (comune_result, std_by_comune, region_value, region_std).
        """
        filtered_data: dict[str, pd.DataFrame] = {}
        aggregated: dict[str, pd.DataFrame] = {}
        daily_by_comune_date: dict[str, pd.DataFrame] = {}
        daily_by_date: dict[str, pd.DataFrame] = {}
        region_total: dict[str, float] = {}

        for phenom in self.phenomena:
            filtered = phenom._filter_by_date(start_date, end_date)
            if filtered.empty:
                return None
            filtered_data[phenom.name] = filtered

            aggregated[phenom.name] = phenom.aggregate(filtered)

            # daily per-comune (already at daily grain after resolve())
            daily_by_comune_date[phenom.name] = (
                filtered.groupby(["ID_COMUNE", "DATA"])[phenom.name]
                .agg(phenom.agg)
                .reset_index()
            )

            # daily region-wide (collapse comuni)
            daily_by_date[phenom.name] = (
                filtered.groupby("DATA")[phenom.name].agg(phenom.agg).reset_index()
            )

            region_total[phenom.name] = float(filtered[phenom.name].agg(phenom.agg))

        # ── per-comune INDICE ──────────────────────────────────────────────
        comune_result = self._merge_phenomena(aggregated)
        if comune_result.empty:
            return None
        comune_result["INDICE"] = self.combinator(
            comune_result,
            filtered_data=filtered_data,
            start_date=start_date,
            end_date=end_date,
            **extra,
        )

        # ── per-comune daily std ───────────────────────────────────────────
        merged_cd = self._merge_daily(daily_by_comune_date, on=["ID_COMUNE", "DATA"])
        merged_cd["INDICE"] = self.combinator(
            merged_cd,
            filtered_data=filtered_data,
            start_date=start_date,
            end_date=end_date,
            **extra,
        )
        # std of a comune with no value at all is null; a single value has no spread: 0 (see _safe_std)
        n_values = merged_cd.groupby("ID_COMUNE")["INDICE"].count()
        std_by_comune = {
            comune: self._safe_std(v) if n_values[comune] > 0 else np.nan
            for comune, v in merged_cd.groupby("ID_COMUNE")["INDICE"].std().items()
        }

        # ── region-wide INDICE ────────────────────────────────────────────
        region_df = pd.DataFrame([region_total])
        region_indice = self.combinator(
            region_df,
            filtered_data=filtered_data,
            start_date=start_date,
            end_date=end_date,
            **extra,
        )
        region_val = (
            region_indice.iloc[0] if hasattr(region_indice, "iloc") else region_indice
        )
        region_value = self._safe_std(region_val) if pd.notna(region_val) else np.nan

        # ── region-wide daily std ─────────────────────────────────────────
        merged_d = self._merge_daily(daily_by_date, on=["DATA"])
        merged_d["INDICE"] = self.combinator(
            merged_d,
            filtered_data=filtered_data,
            start_date=start_date,
            end_date=end_date,
            **extra,
        )
        region_std = (
            self._safe_std(merged_d["INDICE"].std())
            if merged_d["INDICE"].notna().any()
            else np.nan
        )

        return comune_result, std_by_comune, region_value, region_std

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _merge_phenomena(self, aggregated: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Outer-join all per-phenomenon aggregated frames on ID_COMUNE."""
        frames = [df.set_index("ID_COMUNE") for df in aggregated.values()]
        if not frames:
            return pd.DataFrame()
        merged = frames[0]
        for df in frames[1:]:
            merged = merged.join(df, how="outer")
        return merged.reset_index()

    def _merge_daily(
        self, daily_frames: dict[str, pd.DataFrame], on: list[str]
    ) -> pd.DataFrame:
        """Outer-join per-phenomenon daily frames on the given key columns."""
        frames = list(daily_frames.values())
        if not frames:
            return pd.DataFrame()
        merged = frames[0]
        for df in frames[1:]:
            merged = merged.merge(df, how="outer", on=on)
        return merged

    def _label_date_ranges(
        self, labels, granularity, start_date, end_date
    ) -> list[tuple[str, str]]:
        """Map each chart label to its (sub_start, sub_end) ISO date strings."""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        ranges = []

        if granularity == "giornaliero":
            for label in labels:
                ranges.append((label, label))

        elif granularity == "settimanale":
            for label in labels:
                week_start = pd.to_datetime(label)
                week_end = week_start + pd.Timedelta(6, unit="D")
                sub_start = max(week_start, start_date)
                sub_end = min(week_end, end_date)
                ranges.append(
                    (
                        sub_start.strftime("%Y-%m-%d"),
                        sub_end.strftime("%Y-%m-%d"),
                    )
                )

        elif granularity == "mensile":
            for label in labels:
                month_start = pd.to_datetime(label + "-01")
                month_end = month_start + pd.offsets.MonthEnd(0)
                sub_start = max(month_start, start_date)
                sub_end = min(month_end, end_date)
                ranges.append(
                    (
                        sub_start.strftime("%Y-%m-%d"),
                        sub_end.strftime("%Y-%m-%d"),
                    )
                )

        else:  # annuale
            for label in labels:
                year = int(label)
                year_start = pd.Timestamp(year=year, month=1, day=1)
                year_end = pd.Timestamp(year=year, month=12, day=31)
                sub_start = max(year_start, start_date)
                sub_end = min(year_end, end_date)
                ranges.append(
                    (
                        sub_start.strftime("%Y-%m-%d"),
                        sub_end.strftime("%Y-%m-%d"),
                    )
                )

        return ranges
