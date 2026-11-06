from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from overtourism.overtourism.backend_extension.api.models.indicator import Indicator
from overtourism.overtourism.backend_extension.api.models.indicator_loader import (
    indicator_class,
    load_registry,
    load_snippets,
    load_sources,
)
from overtourism.overtourism.backend_extension.api.models.trenitno_phenomena import (
    PresencesPhenomenon,
    ProvenienzaVolumePhenomenon,
    RedistributionPhenomenon,
)
from overtourism.overtourism.backend_extension.api.utils.index_utils_trentino import (
    get_comuni_clusters,
)
from overtourism.overtourism.setup import index_data_path

# ---------------------------------------------------------------------------
# Indicator definitions
# ---------------------------------------------------------------------------
# Every indicator is declared in a YAML file of ``api/indicators/definitions/`` (the file name is the key used by
# the API): metadata, phenomena and how INDICE is computed, either with a formula written in the YAML file, or with a
# Python combinator (``combinators.py``), or with one of the Python classes below (``@indicator_class``).
# How to add one: ``api/indicators/README.md``.
#
# This module only holds the Python classes that YAML definitions refer to with ``class: <Name>``: the indicators
# that need more than a function (extra columns next to INDICE, arrows between areas, a composite of other
# indicators, ...). A class takes its presentation metadata (name, category, description, ...) from its YAML file;
# what is tied to the columns its code produces (``column_aggs``, ``custom_legend``, ``extraFields``) stays here.

INDICATORS_DIR = Path(__file__).resolve().parents[1] / "indicators"

# ---------------------------------------------------------------------------
# Source paths
# ---------------------------------------------------------------------------

# The data files read by the phenomena: name -> path (indicators/sources.yaml, relative to index_data_path)
SOURCES = load_sources(INDICATORS_DIR / "sources.yaml", data_root=index_data_path)

MAP_SHAPEFILE = SOURCES["surface"]
MAP_IDS = index_data_path / "cities_gdf_base_columns.geojson"
MACRO_AREAS_FILE = index_data_path / "map_comuni_into_apt.json"
CODICI_COMUNI_FILE = index_data_path / "mapping_comuni_ISTAT.json"

# Comune-year (or comune) with fewer yearly arrivals than this has no main-market share
# (the text of the indicator's description in dipendenza-mercato-principale.yaml says the same number)
MAIN_MARKET_MIN_ARRIVALS = 100

_VARIATION_RATE_KEY = "tasso-variazione"

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_indicator(key: str) -> Indicator:
    if key not in _INDICATOR_CACHE:
        if key not in _REGISTRY:
            raise KeyError(f"Unknown indicator '{key}'. Available: {sorted(_REGISTRY)}")
        _INDICATOR_CACHE[key] = _REGISTRY[key]()
    return _INDICATOR_CACHE[key]


_INDICATOR_CACHE: dict[str, Indicator] = {}

# ---------------------------------------------------------------------------
# Python indicator classes (referenced by `class:` in the YAML definitions)
# ---------------------------------------------------------------------------


@indicator_class
class TouristPresenceGiniIndicator(Indicator):
    """
    Normalized Gini of the temporal distribution of daily tourist presences.

    Next to INDICE, ``CLUSTER_COLUMN`` holds the seasonality cluster of the comune (an integer code) and
    ``CLUSTER_TYPE_COLUMN`` its name (e.g. "sci"), read from a comune -> cluster mapping computed once in the data
    preparation on the whole period (data_download/seasonality_clusters.py): they are the same for any requested
    window.
    """

    _PRESENCE_COLUMN = "presenze_vodafone"
    _LOW_COVERAGE_VALUE = np.nan
    _MIN_COVERAGE = 0.8
    CLUSTER_COLUMN = "cluster_stagionalita"
    CLUSTER_TYPE_COLUMN = "tipo_cluster"
    custom_legend = {
        "type": "cluster-value",
        "cluster_property": "cluster stagionale",
        "clusters": [
            {"id": 0, "label": "annuale", "color": "#984ea3"},
            {"id": 1, "label": "estate_alpina", "color": "#fcc306"},
            {"id": 2, "label": "estate_spalle", "color": "#1a9850"},
            {"id": 3, "label": "sci", "color": "#4575b4"},
        ],
    }
    # macro area / province: INDICE = mean Gini of the comuni, cluster = the most frequent one. The codes follow the
    # alphabetical order of the names, so on a tie both columns take the same cluster (the smallest)
    column_aggs = {CLUSTER_COLUMN: "mode", CLUSTER_TYPE_COLUMN: "mode"}

    def __init__(self, source_file, clusters_file):
        self.clusters_file = clusters_file
        super().__init__(
            phenomena=[
                PresencesPhenomenon(
                    source_file,
                    col=self._PRESENCE_COLUMN,
                    name=self._PRESENCE_COLUMN,
                )
            ],
            combinator=self._compute_gini,
        )

    def _compute_indicator(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        **extra,
    ) -> pd.DataFrame | None:
        result = super()._compute_indicator(start_date, end_date, **extra)
        if result is None:
            return None

        # the cluster does not depend on the window; a comune missing from the mapping has no cluster (null)
        clusters = get_comuni_clusters(self.clusters_file)
        comuni = result["ID_COMUNE"].astype(str)
        result[self.CLUSTER_COLUMN] = comuni.map(
            {c: v["cluster"] for c, v in clusters.items()}
        )
        result[self.CLUSTER_TYPE_COLUMN] = comuni.map(
            {c: v["tipo_cluster"] for c, v in clusters.items()}
        )
        return result

    @staticmethod
    def _gini(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        # Validity check: if less than 2 days are available, either negative values are present, or total sum is zero, NaN is returned
        if (
            values.size < 2
            or not np.isfinite(values).all()
            or (values < 0).any()
            or values.sum() <= 0
        ):
            return np.nan

        values = np.sort(values)  # It has to be computed on the ordered time series
        count = values.size
        weighted_sum = np.dot(2 * np.arange(1, count + 1) - count - 1, values)
        return float(
            weighted_sum / (count * values.sum()) * count / (count - 1)
        )  # Normalized (index between 0 and 1)

    def _compute_gini(
        self,
        df: pd.DataFrame,
        *,
        filtered_data: dict[str, pd.DataFrame],
        start_date,
        end_date,
        **_extra,
    ) -> pd.Series:
        if start_date is None or end_date is None:
            return pd.Series(np.nan, index=df.index, dtype=float)

        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        data_range = pd.date_range(start, end, freq="D")
        if data_range.empty:
            return pd.Series(np.nan, index=df.index, dtype=float)

        source = self.phenomena[0]._panel
        if source is None:
            raise RuntimeError(
                "Tourist presences must be resolved before Gini calculation"
            )
        source = source[source["DATA"].between(start, end)][
            ["ID_COMUNE", "DATA", self._PRESENCE_COLUMN]
        ]
        observed = filtered_data[self._PRESENCE_COLUMN].copy()
        observed["DATA"] = pd.to_datetime(observed["DATA"])
        source_by_comune = {
            str(comune): group.set_index("DATA")[self._PRESENCE_COLUMN]
            for comune, group in source.groupby("ID_COMUNE", sort=False)
        }
        observed_by_comune = {
            str(comune): group.groupby("DATA")[self._PRESENCE_COLUMN].mean()
            for comune, group in observed.groupby("ID_COMUNE", sort=False)
        }

        gini_by_comune = {}
        for comune in df["ID_COMUNE"].astype(str):
            if (
                source_by_comune.get(comune, pd.Series(dtype=float))
                .reindex(data_range)
                .notna()
                .mean()
                < self._MIN_COVERAGE
            ):
                gini_by_comune[comune] = self._LOW_COVERAGE_VALUE
                continue

            gini_by_comune[comune] = self._gini(
                observed_by_comune.get(comune, pd.Series(dtype=float))
                .dropna()
                .to_numpy(dtype=float)
            )

        return df["ID_COMUNE"].astype(str).map(gini_by_comune)


# ---------------------------------------------------------------------------
# Redistribution of the flows (imbalance index + optimal moves between areas)
# ---------------------------------------------------------------------------
# Reads `phen_ridistribuzione`: one row per comune and DAY, computed offline (data_preparation/
# gen_redistribution_dataframes.py) by solving, for each day, a min-cost redistribution of the visitors above the
# balanced saturation towards the areas with room. The solver never runs in the backend: a requested window is a
# mask on the daily panel + the mean of the daily values, exactly like the other daily phenomena. Besides the
# map (INDICE) the indicator exposes the arrows (`get_flows`, read from `phen_ridistribuzione_flussi`).

_REDISTRIBUTION_SEGMENTS = {
    "combinato": "combined",
    "turisti": "tourist",
    "escursionisti": "visitor",
}
_REDISTRIBUTION_DEFAULT_SEGMENT = "combinato"
_REDISTRIBUTION_COMPANIONS = (
    "persone_gap",
    "persone_spostate_nette",
    "eccesso_non_assorbibile",
)


@indicator_class
class RedistributionBalanceIndicator(Indicator):
    """
    INDICE = mean over the days of the window of the daily imbalance: (balanced saturation - saturation of the area)
    in percentage points. Positive = under-represented (room for more visitors), negative = overloaded.

    ``segmento`` (extra field): ``combinato`` (default: tourists + 0.5 * excursionists), ``turisti``, ``escursionisti``.
    The columns next to INDICE (persons to attract / move / not absorbable) follow the same segment. They are
    persons per day (window mean), split over the comuni of a multi-comune area by beds (tourists) and residents
    (excursionists). INDICE is a value of the AREA: all the comuni of a multi-comune area share it.

    Not available as time series (``availableForVariation = False``): the province value of an index averaged
    over comuni would not be the province imbalance (that is zero by construction).
    """

    FLOWS_TOP_N = 50  # arrows returned with the map (the biggest ones)

    def __init__(self, source_file, flows_file, nodes_file):
        self.flows_file = flows_file
        self.nodes_file = nodes_file
        self._flows_data: tuple[pd.DataFrame, pd.DataFrame] | None = (
            None  # (arcs, nodes), loaded on first use
        )
        super().__init__(
            phenomena=[
                RedistributionPhenomenon(source_file, f"{metric}_{suffix}")
                for metric in ("squilibrio_pp", *_REDISTRIBUTION_COMPANIONS)
                for suffix in _REDISTRIBUTION_SEGMENTS.values()
            ],
            combinator=self._balance,
        )

    @staticmethod
    def _suffix(segmento: str | None) -> str:
        key = segmento or _REDISTRIBUTION_DEFAULT_SEGMENT
        if key not in _REDISTRIBUTION_SEGMENTS:
            raise ValueError(
                f"Unknown segmento={key!r}. Expected one of {sorted(_REDISTRIBUTION_SEGMENTS)}."
            )
        return _REDISTRIBUTION_SEGMENTS[key]

    def _balance(
        self, df: pd.DataFrame, *, segmento: str | None = None, **_extra
    ) -> pd.Series:
        return df[f"squilibrio_pp_{self._suffix(segmento)}"]

    def _compute_indicator(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        segmento: str | None = None,
        **extra,
    ) -> pd.DataFrame | None:
        suffix = self._suffix(segmento)
        result = super()._compute_indicator(
            start_date, end_date, segmento=segmento, **extra
        )
        if result is None:
            return None
        # keep only the columns of the requested segment, under segment-free names
        out = result[["ID_COMUNE"]].copy()
        for metric in _REDISTRIBUTION_COMPANIONS:
            out[metric] = result[f"{metric}_{suffix}"]
        out["INDICE"] = result["INDICE"]
        return out

    # ------------------------------------------------------------------
    # Arrows: optimal flows between the nodes, averaged over the window
    # ------------------------------------------------------------------

    def _load_flows(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._flows_data is None:
            arcs = pd.read_parquet(self.flows_file)
            arcs["DATA"] = pd.to_datetime(arcs["DATA"])
            nodes = pd.read_parquet(self.nodes_file).set_index("node")
            self._flows_data = (
                arcs,
                nodes,
            )  # one assignment: never half-loaded for another thread
        return self._flows_data

    def get_flows(
        self,
        start_date: str | None,
        end_date: str | None,
        *,
        allowed_comuni: set[str] | None = None,
        segmento: str | None = None,
        top_n: int | None = None,
        **_ignored,
    ) -> dict | None:
        """
        The biggest flows of the window as GeoJSON LineStrings (node centroid -> node centroid, WGS84), value =
        mean persons per day moved along the arc (a day without that flow counts as zero). For a one-day window
        it is that day's optimal flow. Only arcs between nodes that contain at least one allowed comune are
        returned (an arrow must not reveal a comune the caller cannot see).

        Returns None when the window has no data.
        """
        suffix = self._suffix(segmento)
        self._ensure_resolved()
        n_days = (
            self.phenomena[0]._filter_by_date(start_date, end_date)["DATA"].nunique()
        )
        if n_days == 0:
            return None

        arcs, nodes = self._load_flows()
        start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
        window = arcs[(arcs["DATA"] >= start) & (arcs["DATA"] <= end)]
        flow = window[
            f"flow_{suffix}"
        ]  # the weight of the visitors in `combined` is applied offline, once
        mean_flow = (
            window.assign(flow=flow)
            .groupby(["node_from", "node_to"], as_index=False)["flow"]
            .sum()
            .assign(flow=lambda d: d["flow"] / n_days)
        )

        if allowed_comuni is not None:
            visible = {
                node
                for node, comuni in nodes["comuni"].items()
                if allowed_comuni.intersection(map(str, comuni))
            }
            mean_flow = mean_flow[
                mean_flow["node_from"].isin(visible)
                & mean_flow["node_to"].isin(visible)
            ]

        mean_flow = mean_flow[mean_flow["flow"] > 0].nlargest(
            top_n or self.FLOWS_TOP_N, "flow"
        )

        features = []
        for rank, row in enumerate(mean_flow.itertuples(index=False), start=1):
            a, b = nodes.loc[row.node_from], nodes.loc[row.node_to]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",  # from -> to: the arrow head is at the destination
                        "coordinates": [
                            [round(float(a["lon"]), 5), round(float(a["lat"]), 5)],
                            [round(float(b["lon"]), 5), round(float(b["lat"]), 5)],
                        ],
                    },
                    "properties": {
                        "rank": rank,
                        "value": round(float(row.flow), 1),
                        "from": row.node_from,
                        "to": row.node_to,
                        "from_label": a["label"],
                        "to_label": b["label"],
                    },
                }
            )
        unit = "presenze al giorno (media del periodo)"
        if suffix == "combined":
            unit = "presenze equivalenti al giorno (media del periodo)"
        return {
            "data": {"type": "FeatureCollection", "features": features},
            "max_value": float(mean_flow["flow"].max()) if len(mean_flow) else 0.0,
            "unit": unit,
            "n_days": int(n_days),
        }


@indicator_class
class MainMarketDependenceIndicator(Indicator):
    """Share of the yearly tourist arrivals that come from the main source market: volume_mercato_1 / volume_totale."""

    MAIN_MARKETS_COLUMN = "main_markets"

    def __init__(self, source_file):
        super().__init__(
            phenomena=[
                ProvenienzaVolumePhenomenon(
                    source_file,
                    "volume_mercato_1",
                    min_total=MAIN_MARKET_MIN_ARRIVALS,
                ),
                ProvenienzaVolumePhenomenon(
                    source_file, "volume_totale", min_total=MAIN_MARKET_MIN_ARRIVALS
                ),
            ],
            combinator=self._dependence,
        )

    def _compute_indicator(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        **extra,
    ) -> pd.DataFrame | None:
        result = super()._compute_indicator(start_date, end_date, **extra)
        if result is None:
            return None

        source = self.phenomena[0]._read_source()
        source_years = source["DATA"].astype(str).str[:4].astype(int)
        market_names = source.loc[
            source_years == pd.to_datetime(end_date).year,
            ["ID_COMUNE", "nome_mercato_1", "nome_mercato_2", "nome_mercato_3"],
        ].copy()
        market_names["ID_COMUNE"] = market_names["ID_COMUNE"].astype(str)
        if market_names["ID_COMUNE"].duplicated().any():
            raise ValueError(
                "phen_provenienza must have only one row per comune and year"
            )

        def describe_markets(row: pd.Series) -> str | None:
            names = [
                str(value).strip()
                for value in row.iloc[1:]
                if pd.notna(value) and str(value).strip()
            ]
            return ", ".join(names) if names else None

        market_names[self.MAIN_MARKETS_COLUMN] = market_names.apply(
            describe_markets, axis=1
        )
        market_names_by_comune = market_names.set_index("ID_COMUNE")[
            self.MAIN_MARKETS_COLUMN
        ]
        result[self.MAIN_MARKETS_COLUMN] = (
            result["ID_COMUNE"].astype(str).map(market_names_by_comune)
        )
        return result

    @staticmethod
    def _dependence(df: pd.DataFrame, **_extra) -> pd.Series:
        """main / total. The comune-years below MAIN_MARKET_MIN_ARRIVALS are already dropped by the phenomena."""
        return df["volume_mercato_1"] / df["volume_totale"].replace(0, np.nan)


def _parse_bool_flag(value, name: str) -> bool:
    """Boolean extra field: it arrives as a query-string text (\"true\", \"1\", ...), or as a bool when called from code."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "si", "sì", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Invalid value for {name!r}: {value!r}. Expected true or false.")


# ---------------------------------------------------------------------------
# Territorial sustainability (explainable composite index)
# ---------------------------------------------------------------------------
# A transparent composite: every COMPONENT is an indicator that already exists in the registry, turned into a 0-100
# score with a linear scale between a "sustainable" value (score 100) and a "critical" one (score 0); the scores are
# averaged inside each PILLAR, and the pillars are averaged into INDICE. Everything that drives the result (pillars,
# components, weights, reference values, framework references) is declared in the tables below and is written in the
# description shown to the user: there is no other parameter.
#
# Framework. Pillars and components follow the GSTC Destination Criteria v2.0 (sections: A sustainable management,
# B socio-economic, C cultural, D environmental) and the issue areas of the UN Tourism INSTO observatories (e.g.
# Local Satisfaction, Energy Management, Climate Action). GSTC and UN Tourism publish criteria and indicators, NOT
# numeric thresholds or weights: the reference values and weights below are the choice of this index, with equal
# weights between pillars (no consensus on a different one) and they are PROVISIONAL, to be calibrated on the data
# and validated with the stakeholders (see TerritorialSustainabilityIndicator.reference_value_summary).
#
# Selection rule: ONE component per sustainability theme, not "all the indicators". A candidate is left out when it is
#   (a) year-only (it cannot be computed for a window of days);
#   (b) a variant of a ratio that another component already measures (same numerator: tourists per resident, tourist
#       density per km², beds per resident, hospitality shares, ... measure the same pressure, and together they would
#       weigh it several times);
#   (c) the same theme as a component that is already in (electricity and gas: one energy theme), or built on data
#       that is too thin to be a pillar of the index (gas);
#   (d) not an impact of tourism (e.g. "hidden tourism" measures how well the official statistics capture tourism: it
#       is a monitoring-quality measure).
# The check of (b) on real data is TerritorialSustainabilityIndicator.component_correlation.
#
# Scope. With these data only three themes can be measured: pressure on the residents, energy, transport / climate.
# GSTC sections A (management) and C (cultural), the local economic benefits and the pressure on the land are NOT
# scored, and a high score is not a statement of overall sustainability.


@dataclass(frozen=True)
class SustainabilityPillar:
    key: str
    label: str
    weight: float  # relative weight in INDICE
    framework: str


@dataclass(frozen=True)
class SustainabilityComponent:
    key: str  # name of the component; its score column is ``score_<key>``
    indicator: str  # registry key of the indicator that gives the raw value
    pillar: str
    weight: float  # relative weight inside the pillar
    best: float  # raw value at (or beyond) which the score is 100
    worst: float  # raw value at (or beyond) which the score is 0
    label: str
    unit: str
    framework: str


# INDICE is not computed when the components that are available weigh less than this share of the total
SUSTAINABILITY_MIN_COVERAGE = 0.5

# The weights give every THEME the same weight (1/3): the environmental pillar covers two themes, so it weighs twice
# the socio-economic one.
SUSTAINABILITY_PILLARS = (
    SustainabilityPillar(
        key="socioeconomic",
        label="Impatto socio-economico",
        weight=1.0,
        framework="GSTC-D sezione B; UN Tourism INSTO: Local Satisfaction",
    ),
    SustainabilityPillar(
        key="environmental",
        label="Impatto ambientale e uso delle risorse",
        weight=2.0,
        framework="GSTC-D sezione D; UN Tourism INSTO: Energy Management, Climate Action",
    ),
)

# best / worst: PROVISIONAL reference values (see the comment above). best > worst is allowed (higher is better).
SUSTAINABILITY_COMPONENTS = (
    SustainabilityComponent(
        key="tourists_per_resident",
        indicator="turisticita",
        pillar="socioeconomic",
        weight=1,
        best=0.1,
        worst=3.0,
        label="Turisti per residente",
        unit="turisti per residente",
        framework="GSTC-D A(c) gestione della pressione e B(b) benessere sociale; INSTO Local Satisfaction",
    ),
    SustainabilityComponent(
        key="electricity_services_share",
        indicator="impatto-turismo-consumi-elettrici",
        pillar="environmental",
        weight=1,
        best=0.2,
        worst=0.8,
        label="Impatto sui consumi elettrici",
        unit="quota dei consumi elettrici dei servizi sul totale",
        framework="GSTC-D D(b) gestione delle risorse; INSTO Energy Management",
    ),
    SustainabilityComponent(
        key="travel_time_hours",
        indicator="distanza-media-turisti",
        pillar="environmental",
        weight=1,
        best=2.0,
        worst=8.0,
        label="Tempo medio di viaggio dei turisti",
        unit="ore",
        framework="GSTC-D D(c) gestione di rifiuti ed emissioni; INSTO Climate Action",
    ),
)


def _validate_sustainability_config() -> None:
    pillar_keys = [p.key for p in SUSTAINABILITY_PILLARS]
    component_keys = [c.key for c in SUSTAINABILITY_COMPONENTS]
    if len(set(pillar_keys)) != len(pillar_keys) or len(set(component_keys)) != len(
        component_keys
    ):
        raise ValueError("Sustainability pillars / components must have unique keys")
    if any(p.weight <= 0 for p in SUSTAINABILITY_PILLARS):
        raise ValueError("Sustainability pillar weights must be positive")
    for c in SUSTAINABILITY_COMPONENTS:
        if c.pillar not in pillar_keys:
            raise ValueError(f"Component {c.key!r}: unknown pillar {c.pillar!r}")
        if c.weight <= 0:
            raise ValueError(f"Component {c.key!r}: weight must be positive")
        if c.best == c.worst:
            raise ValueError(f"Component {c.key!r}: best and worst must differ")
    for key in pillar_keys:
        if not any(c.pillar == key for c in SUSTAINABILITY_COMPONENTS):
            raise ValueError(f"Pillar {key!r} has no components")


def _it_number(value: float) -> str:
    return f"{value:g}".replace(".", ",")


@indicator_class
class TerritorialSustainabilityIndicator(Indicator):
    """
    Explainable composite of tourism sustainability, 0-100 (100 = most sustainable), for any window of days.

    Construction (all declared in SUSTAINABILITY_PILLARS / SUSTAINABILITY_COMPONENTS):

    1. component score = 100 * clip((worst - value) / (worst - best), 0, 1): linear between the "critical" value
       (0) and the "sustainable" one (100). ``value`` is the one of the standalone indicator.
    2. pillar score = mean of its component scores, weighted by the component weights; a component without a value
       is left out and the weights of the others are rescaled.
    3. INDICE = mean of the pillar scores, weighted by the pillar weights (a pillar without any value is left out).
       INDICE is null when the components that are available weigh less than SUSTAINABILITY_MIN_COVERAGE.

    The component values are NOT recomputed: the indicator shares the phenomena of the component indicators and
    calls their own combinators on the same data, so each value is the one the user sees on that indicator's map,
    and the macro-indicator works for any window and in the time series like any other indicator.

    Next to INDICE: every component score (``score_<key>``), the pillar scores (``score_<pillar>``),
    ``data_coverage`` (share of the total weight that has a value, 0-1) and ``weakest_component`` (name of the
    component with the lowest score, empty when all the scores are 100). The raw phenomena are not returned.

    Year-only indicators are rejected when the indicator is built. See the selection rule above the config.
    """

    SCORE_PREFIX = "score_"
    COVERAGE_COLUMN = "data_coverage"
    WEAKEST_COLUMN = "weakest_component"
    # the scores and the coverage are means over the comuni at macro area / province level, not sums (the default
    # of the numeric columns); the weakest component is the most frequent one
    column_aggs = {
        **{f"score_{c.key}": "mean" for c in SUSTAINABILITY_COMPONENTS},
        **{f"score_{p.key}": "mean" for p in SUSTAINABILITY_PILLARS},
        "data_coverage": "mean",
        "weakest_component": "mode",
    }

    def __init__(self):
        _validate_sustainability_config()
        self._component_indicators: dict[str, Indicator] = {}
        for c in SUSTAINABILITY_COMPONENTS:
            indicator = get_indicator(c.indicator)
            if indicator.year_only:
                raise ValueError(
                    f"Component {c.key!r}: indicator {c.indicator!r} is year-only, "
                    "so it cannot be used for a window of days"
                )
            self._component_indicators[c.key] = indicator

        # effective weight of each component in INDICE (they sum to 1)
        pillar_total = sum(p.weight for p in SUSTAINABILITY_PILLARS)
        self._effective_weight: dict[str, float] = {}
        for p in SUSTAINABILITY_PILLARS:
            members = [c for c in SUSTAINABILITY_COMPONENTS if c.pillar == p.key]
            members_total = sum(c.weight for c in members)
            for c in members:
                self._effective_weight[c.key] = (
                    p.weight / pillar_total * c.weight / members_total
                )

        super().__init__(phenomena=self._shared_phenomena(), combinator=self._indice)
        self.description = self._build_description()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _shared_phenomena(self) -> list:
        """The phenomena of all the components, without duplicates (the same objects: they are resolved once)."""
        unique: dict[str, tuple] = {}
        for indicator in self._component_indicators.values():
            for phenomenon in indicator.phenomena:
                signature = (
                    type(phenomenon).__name__,
                    str(phenomenon.source),
                    phenomenon.col,
                    phenomenon.agg,
                    getattr(phenomenon, "measure", None),
                    getattr(phenomenon, "max_minutes", None),
                    getattr(phenomenon, "min_total", None),
                )
                known = unique.get(phenomenon.name)
                if known is None:
                    unique[phenomenon.name] = (phenomenon, signature)
                elif known[1] != signature:
                    raise ValueError(
                        f"Two different phenomena share the name {phenomenon.name!r}: "
                        "the component indicators cannot be merged"
                    )
        return [phenomenon for phenomenon, _ in unique.values()]

    @staticmethod
    def _build_description() -> str:
        return (
            "L'indice di sostenibilità territoriale è un <strong>punteggio da 0 a 100</strong> "
            "(100 = massima sostenibilità) che sintetizza la pressione del turismo sul territorio. "
            "Si ispira ai criteri GSTC per le destinazioni e alle aree tematiche degli osservatori "
            "INSTO di UN Tourism."
            "<br><strong>Dimensioni e indicatori.</strong> L'indice considera due dimensioni, "
            "rappresentate da tre indicatori:"
            "<ul>"
            "<li><strong>Impatto socio-economico:</strong> turisti per residente, per misurare "
            "la pressione del turismo sulla popolazione locale.</li>"
            "<li><strong>Impatto ambientale e uso delle risorse:</strong> impatto del turismo sui "
            "consumi elettrici e tempo medio di viaggio dei turisti, per rappresentare rispettivamente "
            "la pressione sui consumi energetici e la componente della mobilità turistica.</li>"
            "</ul>"
            "<strong>Calcolo.</strong> Ogni indicatore è convertito in un punteggio da 0 a 100 "
            "sulla base di valori di riferimento sostenibili e critici. Le tre componenti hanno "
            "lo stesso peso nell'indice finale."
            "<br><strong>Dati mancanti.</strong> Gli indicatori non disponibili sono esclusi dal "
            "calcolo e i pesi vengono ricalcolati. L'indice non è calcolato se la copertura dei dati "
            "è inferiore al 50%."
            "<br><strong>Limiti.</strong> L'indice considera la pressione sui residenti, i consumi "
            "elettrici e la mobilità turistica. Non costituisce una valutazione complessiva della "
            "sostenibilità del territorio né una verifica di conformità ai criteri GSTC."
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _linear_score(values: pd.Series, best: float, worst: float) -> pd.Series:
        """100 at ``best`` (or beyond), 0 at ``worst`` (or beyond), linear in between; missing stays missing."""
        share = ((worst - values) / (worst - best)).clip(0, 1)
        return share * 100

    @staticmethod
    def _weighted_mean(
        values: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Row-wise weighted mean of the available (non-NaN) values; also returns the available weight per row."""
        available = ~np.isnan(values)
        used = (available * weights).sum(axis=1)
        total = (np.nan_to_num(values) * weights).sum(axis=1)
        mean = np.divide(total, used, out=np.full(len(values), np.nan), where=used > 0)
        return mean, used

    def _score_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """All the scores of a wide frame (one column per phenomenon): components, pillars, coverage, INDICE."""
        out = pd.DataFrame(index=df.index)

        for c in SUSTAINABILITY_COMPONENTS:
            raw = pd.to_numeric(
                self._component_indicators[c.key].combinator(df), errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)
            out[f"{self.SCORE_PREFIX}{c.key}"] = self._linear_score(
                raw, c.best, c.worst
            )

        pillar_scores = []
        for p in SUSTAINABILITY_PILLARS:
            members = [c for c in SUSTAINABILITY_COMPONENTS if c.pillar == p.key]
            values = out[[f"{self.SCORE_PREFIX}{c.key}" for c in members]].to_numpy(
                dtype=float
            )
            weights = np.array([c.weight for c in members], dtype=float)
            mean, _ = self._weighted_mean(values, weights)
            out[f"{self.SCORE_PREFIX}{p.key}"] = mean
            pillar_scores.append(mean)

        indice, _ = self._weighted_mean(
            np.column_stack(pillar_scores),
            np.array([p.weight for p in SUSTAINABILITY_PILLARS], dtype=float),
        )

        component_scores = out[
            [f"{self.SCORE_PREFIX}{c.key}" for c in SUSTAINABILITY_COMPONENTS]
        ].to_numpy(dtype=float)
        coverage = (
            ~np.isnan(component_scores)
            * np.array(
                [self._effective_weight[c.key] for c in SUSTAINABILITY_COMPONENTS]
            )
        ).sum(axis=1)
        out[self.COVERAGE_COLUMN] = coverage
        out["INDICE"] = np.where(
            coverage >= SUSTAINABILITY_MIN_COVERAGE, indice, np.nan
        )

        # weakest component: the lowest score, named only if it is below 100
        labels = np.array([c.label for c in SUSTAINABILITY_COMPONENTS], dtype=object)
        none_available = np.isnan(component_scores).all(axis=1)
        lowest = np.where(np.isnan(component_scores), np.inf, component_scores)
        position = lowest.argmin(axis=1)
        weakest = labels[position]
        weakest[none_available | (lowest.min(axis=1) >= 100)] = None
        out[self.WEAKEST_COLUMN] = weakest
        return out

    def _indice(self, df: pd.DataFrame, **_extra) -> pd.Series:
        return self._score_table(df)["INDICE"]

    def _compute_indicator(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        **extra,
    ) -> pd.DataFrame | None:
        result = super()._compute_indicator(start_date, end_date, **extra)
        if result is None:
            return None
        table = self._score_table(result)
        columns = [
            *(f"{self.SCORE_PREFIX}{c.key}" for c in SUSTAINABILITY_COMPONENTS),
            *(f"{self.SCORE_PREFIX}{p.key}" for p in SUSTAINABILITY_PILLARS),
            self.COVERAGE_COLUMN,
            self.WEAKEST_COLUMN,
            "INDICE",
        ]
        return pd.concat([result[["ID_COMUNE"]], table[columns]], axis=1)

    def _compute_label_metrics(self, start_date, end_date, **extra) -> tuple | None:
        """
        Same as the base class, except for the region ("Provincia") value and its std: the mean of the comune
        values, like the map at province / macro area level (TerritorialConfig). The base class would score the
        province-wide totals, and a clipped score of an average is not the average of the scores (it hides the
        overloaded comuni): the two views of the same index would show different numbers.
        """
        metrics = super()._compute_label_metrics(start_date, end_date, **extra)
        if metrics is None:
            return None
        comune_result, std_by_comune, _, _ = metrics
        values = comune_result["INDICE"].dropna()
        stds = [v for v in std_by_comune.values() if pd.notna(v)]
        region_value = float(values.mean()) if len(values) else np.nan
        region_std = float(np.mean(stds)) if stds else np.nan
        return comune_result, std_by_comune, region_value, region_std

    # ------------------------------------------------------------------
    # Calibration support
    # ------------------------------------------------------------------

    def component_correlation(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame | None:
        """
        Spearman correlation, across the comuni, of the raw values of the components in a window. The check of the
        selection rule (b): two components that are strongly correlated (say |rho| > 0.7) on the real data measure
        the same thing and should not both be in the index. Not used by the API.
        """
        series = {}
        for c in SUSTAINABILITY_COMPONENTS:
            values = self._component_indicators[c.key].get_indicator(
                start_date=start_date, end_date=end_date
            )
            if values is not None:
                series[c.key] = values.set_index("ID_COMUNE")["INDICE"].replace(
                    [np.inf, -np.inf], np.nan
                )
        if len(series) < 2:
            return None
        return pd.DataFrame(series).corr(method="spearman", min_periods=3)

    def reference_value_summary(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame | None:
        """
        Distribution over the comuni (10th / 50th / 90th percentile) of the raw value of every component in a
        window, next to the current "sustainable" / "critical" reference values: the base to calibrate them
        (e.g. best ~ a low percentile, worst ~ a high one) before sharing the index. Not used by the API.
        """
        rows = []
        for c in SUSTAINABILITY_COMPONENTS:
            values = self._component_indicators[c.key].get_indicator(
                start_date=start_date, end_date=end_date
            )
            if values is None:
                continue
            series = values["INDICE"].replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "component": c.key,
                    "unit": c.unit,
                    "n_comuni": int(series.size),
                    "p10": series.quantile(0.1),
                    "p50": series.quantile(0.5),
                    "p90": series.quantile(0.9),
                    "best": c.best,
                    "worst": c.worst,
                }
            )
        return pd.DataFrame(rows) if rows else None


@indicator_class
class VariationRateIndicator(Indicator):
    """
    Generic "Tasso di variazione" (rate of variation) index.

    Computes the percentage change of *any other registered indicator*
    between two periods: the request's regular baseline period
    (``start_date`` / ``end_date``) and a comparison period
    (``start_date_comparison`` / ``end_date_comparison``). The same
    indicator is used as the phenomenon in both periods.

    Rather than hard-coding one variation subclass per wrapped indicator,
    the wrapped indicator is itself a parameter: it is selected at request
    time via the ``indicator`` extra field (its registry key, e.g.
    ``"turisticita"``), together with the dates of the comparison period.
    This keeps a single
    "tasso-variazione" entry in the registry that works for every
    combinable indicator.

    Because the wrapped indicator is only known once a request comes in,
    this class doesn't use the phenomena/combinator pipeline from the base
    class at all — ``_compute_indicator`` is fully overridden below, and
    ``phenomena``/``combinator`` are left empty/no-op.
    """

    extraFields = ["indicator", "start_date_comparison", "end_date_comparison"]

    def __init__(self):
        # No fixed phenomena: which indicator (and therefore which
        # phenomena) to use is only known at request time, via the
        # "indicator" extra field handled in _compute_indicator().
        super().__init__(phenomena=[], combinator=self._unused_combinator)

    @staticmethod
    def _unused_combinator(df: pd.DataFrame, **_extra) -> pd.Series:  # pragma: no cover
        # Never invoked: _compute_indicator() is fully overridden below and
        # bypasses the base class's phenomena-driven pipeline entirely.
        raise RuntimeError(
            "VariationRateIndicator.combinator should never be called directly"
        )

    @property
    def years_range(self) -> dict[str, int]:
        """
        Broadest year range across every combinable indicator in the
        registry (excluding this one), since the actual wrapped indicator
        isn't known until request time. Validity of the *chosen* indicator
        for the requested dates is enforced in ``_compute_indicator``.
        """
        if self._years_range is None:
            min_years, max_years = [], []
            for key in _REGISTRY:
                if key == _VARIATION_RATE_KEY:
                    continue
                try:
                    yr = get_indicator(key).years_range
                except ValueError:
                    continue
                min_years.append(yr["min_year"])
                max_years.append(yr["max_year"])

            if not min_years:
                raise ValueError(
                    f"{self!r}: no combinable indicators available to compute "
                    "a years_range for"
                )

            self._years_range = {
                "min_year": min(min_years),
                "max_year": max(max_years),
            }
        return self._years_range

    def _compute_indicator(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        **extra,
    ) -> pd.DataFrame | None:
        indicator_key = extra.pop("indicator", None)
        start_date_comparison = extra.pop("start_date_comparison", None)
        end_date_comparison = extra.pop("end_date_comparison", None)

        if not indicator_key:
            raise ValueError(
                "'indicator' extra field is required for 'tasso-variazione'"
            )
        if indicator_key == _VARIATION_RATE_KEY:
            raise ValueError("'tasso-variazione' cannot wrap itself")
        if start_date_comparison is None or end_date_comparison is None:
            raise ValueError(
                "'start_date_comparison' and 'end_date_comparison' extra "
                "fields are required for 'tasso-variazione'"
            )

        wrapped = get_indicator(indicator_key)

        # Current period (e.g. 2025)
        current_df = wrapped.get_indicator(
            start_date=start_date,
            end_date=end_date,
            **extra,
        )

        # Reference period (e.g. 2024)
        previous_df = wrapped.get_indicator(
            start_date=start_date_comparison,
            end_date=end_date_comparison,
            **extra,
        )

        if current_df is None or previous_df is None:
            return None

        merged = (
            current_df[["ID_COMUNE", "INDICE"]]
            .rename(columns={"INDICE": "Periodo base"})
            .merge(
                previous_df[["ID_COMUNE", "INDICE"]].rename(
                    columns={"INDICE": "Periodo comparazione"}
                ),
                on="ID_COMUNE",
                how="outer",
            )
        )

        # Percentage variation: (baseline - comparison) / comparison * 100
        merged["INDICE"] = (
            (merged["Periodo base"] - merged["Periodo comparazione"])
            / merged["Periodo comparazione"].replace(0, np.nan)
        ) * 100

        return merged[["ID_COMUNE", "INDICE", "Periodo base", "Periodo comparazione"]]


# ---------------------------------------------------------------------------
# Registry (built last: every class above is registered, so the whole YAML is validated at import time)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[[], Indicator]] = load_registry(
    INDICATORS_DIR / "definitions",
    sources=SOURCES,
    snippets=load_snippets(INDICATORS_DIR / "snippets.yaml"),
)
