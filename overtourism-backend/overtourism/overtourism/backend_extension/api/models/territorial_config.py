"""Territorial granularity configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Granularity = Literal["comune", "macro_area", "provincia"]
Aggregation = Literal["mean", "sum"]


@dataclass(frozen=True)
class MacroArea:
    """A named province-level group of comuni."""

    name: str
    comuni: list[str]


@dataclass
class TerritorialConfig:
    """
    Request-scoped territorial configuration.

    Aggregation of columns at macro_area / provincia level
    -------------------------------------------------------
    - ``value_col`` ("INDICE")  -> ``macro_area_agg`` ("mean" | "sum")
    - any column listed in ``column_aggs`` -> the function given there
      (any pandas aggregation name: "sum", "mean", "max", "min", "first", ...,
      or "mode": the most frequent value, the smallest one on a tie)
    - any other numeric column (e.g. presences, number of structures)
      -> "sum", i.e. the area total
    - any other non-numeric column -> dropped (a label taken from a single
      comune would be misleading for the whole area); opt in via
      ``column_aggs={"col": "first"}``.
    """

    areas: list[MacroArea] = field(default_factory=list)
    spatial_granularity: Granularity = "comune"
    allowed_comuni: set[str] | None = None
    macro_area_agg: Aggregation = "mean"
    column_aggs: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        areas: list[MacroArea],
        spatial_granularity: str = "comune",
        allowed_comuni: set[str] | None = None,
        macro_area_agg: str = "mean",
        column_aggs: dict[str, str] | None = None,
    ) -> TerritorialConfig:
        """Validated constructor used by all endpoints."""

        if spatial_granularity not in ("comune", "macro_area", "provincia"):
            raise ValueError(
                f"Unknown spatial_granularity={spatial_granularity!r}. "
                "Expected 'comune' or 'macro_area' or 'provincia'."
            )

        if macro_area_agg not in ("mean", "sum"):
            raise ValueError(
                f"Unknown macro_area_agg={macro_area_agg!r}. Expected 'mean' or 'sum'."
            )

        return cls(
            areas=areas,
            spatial_granularity=spatial_granularity,
            allowed_comuni=allowed_comuni,
            macro_area_agg=macro_area_agg,
            column_aggs=dict(column_aggs or {}),
        )

    # ------------------------------------------------------------------
    # Core: filter + aggregate
    # ------------------------------------------------------------------

    def filter_comuni(
        self,
        df: pd.DataFrame,
        id_col: str = "ID_COMUNE",
    ) -> pd.DataFrame:
        """Apply the permission filter."""

        if self.allowed_comuni is None:
            return df

        return df[df[id_col].astype(str).isin(self.allowed_comuni)].copy()

    def _aggregate_group(
        self,
        subset: pd.DataFrame,
        value_col: str,
        id_col: str,
    ) -> dict:
        """
        Collapse a group of comune rows into one dict of {column: value},
        applying the column rules described in the class docstring.
        """
        out: dict = {}
        for col in subset.columns:
            if col == id_col:
                continue

            if col == value_col:
                fn = self.macro_area_agg
            elif col in self.column_aggs:
                fn = self.column_aggs[col]
            elif pd.api.types.is_numeric_dtype(subset[col]):
                fn = "sum"
            else:
                continue  # non-numeric, no explicit rule: not meaningful for an area

            series = subset[col]
            # sum() of an all-NaN column is 0 in pandas; keep it null instead
            # so "no data" is not shown as "zero".
            if not series.notna().any():
                out[col] = float("nan")
            else:
                valid = series.dropna()
                if fn == "first":
                    out[col] = valid.iloc[0]
                elif fn == "last":
                    out[col] = valid.iloc[-1]
                elif fn == "mode":
                    out[col] = valid.mode().iloc[0]  # mode() is sorted: a tie takes the smallest value
                else:
                    out[col] = series.agg(fn)
        return out

    def _finalize(
        self,
        rows: list[dict],
        source: pd.DataFrame,
        value_col: str,
        id_col: str,
    ) -> pd.DataFrame:
        """Build the result frame, keeping the source column order."""
        out = pd.DataFrame(rows)
        if value_col not in out.columns:
            out[value_col] = float("nan")
        ordered = [id_col] + [
            c for c in source.columns if c != id_col and c in out.columns
        ]
        return out[ordered]

    def aggregate_to_macro_areas(
        self,
        df: pd.DataFrame,
        value_col: str = "INDICE",
        id_col: str = "ID_COMUNE",
    ) -> pd.DataFrame:
        """Aggregate a comune dataframe into macro areas (all columns kept)."""

        ids = df[id_col].astype(str)
        present = set(ids)
        rows = []

        for area in self.areas:
            members = [c for c in area.comuni if c in present]

            if not members:
                # Other columns end up as NaN once the frame is assembled.
                rows.append({id_col: area.name, value_col: float("nan")})
                continue

            subset = df[ids.isin(members)]
            rows.append(
                {id_col: area.name, **self._aggregate_group(subset, value_col, id_col)}
            )

        return self._finalize(rows, df, value_col, id_col)

    def aggregate_to_province(
        self,
        df: pd.DataFrame,
        value_col: str = "INDICE",
        id_col: str = "ID_COMUNE",
    ) -> pd.DataFrame:
        """Aggregate all available comuni into the province-level unit."""

        comuni = df[df[id_col].astype(str) != "-1"]
        if comuni.empty:
            rows = [{id_col: "-1", value_col: float("nan")}]
        else:
            rows = [{id_col: "-1", **self._aggregate_group(comuni, value_col, id_col)}]

        return self._finalize(rows, df, value_col, id_col)

    def apply(
        self,
        df: pd.DataFrame,
        value_col: str = "INDICE",
        id_col: str = "ID_COMUNE",
    ) -> pd.DataFrame:
        """Filter and optionally aggregate."""

        filtered = self.filter_comuni(df, id_col=id_col)

        if self.spatial_granularity == "provincia":
            return self.aggregate_to_province(
                filtered,
                value_col=value_col,
                id_col=id_col,
            )

        if self.spatial_granularity == "macro_area":
            return self.aggregate_to_macro_areas(
                filtered,
                value_col=value_col,
                id_col=id_col,
            )

        return filtered

    # ------------------------------------------------------------------
    # Time series
    # ------------------------------------------------------------------

    def apply_to_series(
        self,
        series: list[dict],
        label_key: str = "label",
        data_key: str = "data",
        std_key: str = "std",
    ) -> list[dict]:
        """Apply filtering/aggregation to temporal series."""

        import numpy as np

        region_series = [s for s in series if str(s[label_key]) == "-1"]
        comune_series = [s for s in series if str(s[label_key]) != "-1"]

        if self.allowed_comuni is not None:
            comune_series = [
                s for s in comune_series if str(s[label_key]) in self.allowed_comuni
            ]

        if self.spatial_granularity == "comune":
            return comune_series + region_series

        if self.spatial_granularity == "provincia":
            return region_series

        by_comune = {str(s[label_key]): s for s in comune_series}

        agg_fn = np.mean if self.macro_area_agg == "mean" else np.sum

        area_series = []

        for area in self.areas:
            members = [c for c in area.comuni if c in by_comune]

            if not members:
                continue

            data = np.array([by_comune[c][data_key] for c in members])
            std = np.array([by_comune[c][std_key] for c in members])

            area_series.append(
                {
                    label_key: area.name,
                    data_key: [round(float(x), 2) for x in agg_fn(data, axis=0)],
                    std_key: [round(float(x), 2) for x in np.mean(std, axis=0)],
                }
            )

        return area_series + region_series
