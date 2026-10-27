import pandas as pd
import geopandas as gpd

from overtourism.overtourism.backend_extension.api.models.phenomenon import Phenomenon

# ---------------------------------------------------------------------------
# Italian translations for phenomenon keywords returned in API responses
# ---------------------------------------------------------------------------
# The GeoJSON properties returned by /get_index_data carry one column per
# phenomenon that makes up the requested indicator (e.g. "beds" and
# "population" for "tasso-ricettivita"), plus "INDICE" itself. Those column
# names come straight from each Phenomenon's `.name` (see the phenomenon
# classes below) and are English by convention, since they're also used
# internally as pandas/DataFrame keys throughout indicator.py.
#
# This dictionary is the single predefined mapping from those internal
# keywords to the Italian labels exposed to the frontend. It's applied via
# `index_utils._translate_columns()` right before a GeoDataFrame is
# serialised to GeoJSON, so indicator/phenomenon internals never need to
# know about it.
#
# Only keys listed here get renamed — "AREA_NAME", "INDICE", and "geometry"
# are intentionally left out and pass through unchanged. Add an entry here
# whenever a new Phenomenon (or a custom `name=...` override) is introduced
# and should be human-readable in the API response.

PHENOMENON_LABELS_IT: dict[str, str] = {
    # AccommodationCapacityIndicator ("tasso-ricettivita")
    "beds": "posti letto",
    "population": "popolazione",
    # TourismIndexIndicator ("indice-turisticita")
    "presences": "presenze",
    # TouristDensityIndicator ("densita-turistica")
    "superficie": "superficie (km²)",
    # Hospitality indicators
    "strutture_alloggi_turistici": "strutture alloggi turistici",
    "strutture_alloggi_disp": "strutture alloggi a disposizione",
    "strutture_alb": "strutture alberghiere",
    "strutture_exalb": "strutture extra-alberghiere",
    "letti_alloggi_turistici": "posti letto alloggi turistici",
    "letti_alloggi_disp": "posti letto alloggi a disposizione",
    "letti_alb": "posti letto alberghieri",
    "letti_exalb": "posti letto extra-alberghieri",
    # HiddenTourismIndicator ("indice-turismo-sommerso")
    "presenze_vodafone": "presenze turisti rete mobile",
    "presenze_escursionisti_vodafone": "presenze escrusionisti rete mobile",
    "presenze_alb": "presenze alb.",
    "presenze_xalb": "presenze extra-alb.",
    # FlowPhenomenon-based indicators (RatioFlowsIndicator, FlowsIndicatorLevel;
    # currently only reachable if INCLUDE_TEMP_REGISTRY is enabled)
    "flows": "flussi",
    "flow_value": "valore flusso",
    # OD flow indicators (phen_flussi_od): quota in ingresso / uscita, quota intra-ambito, PageRank.
    # Every label contains "quota" / "PageRank": index_utils rounds those columns to decimals, not to integers.
    "quota_ingresso_combined": "quota ingresso (turisti + escursionisti)",
    "quota_ingresso_tourist": "quota ingresso turisti",
    "quota_ingresso_visitor": "quota ingresso escursionisti",
    "quota_uscita_combined": "quota uscita (turisti + escursionisti)",
    "quota_uscita_tourist": "quota uscita turisti",
    "quota_uscita_visitor": "quota uscita escursionisti",
    "quota_intra_ambito_combined": "quota intra-ambito (turisti + escursionisti)",
    "quota_intra_ambito_tourist": "quota intra-ambito turisti",
    "quota_intra_ambito_visitor": "quota intra-ambito escursionisti",
    "pagerank_combined": "PageRank (turisti + escursionisti)",
    "pagerank_tourist": "PageRank turisti",
    "pagerank_visitor": "PageRank escursionisti",
    # Provenienza (phen_provenienza): normalized HHI of the source markets of the tourists.
    # The label contains "quota" on purpose: index_utils rounds the columns with "quota" / "PageRank" in their
    # label to decimals (an index in [0, 1] must not be rounded to an integer).
    "hhi_provenienza": "quota di concentrazione dei mercati (HHI normalizzato)",
    # Main source market (phen_provenienza): yearly arrivals of the main market and of all the markets (integers)
    "volume_mercato_1": "arrivi dal mercato principale",
    "volume_totale": "arrivi totali",
    "main_markets": "Mercati principali",
    # Consumption indicators (phen_consumption): daily electricity / gas consumption, services vs total
    "ee_services": "consumi elettrici servizi",
    "ee_total": "consumi elettrici totali",
    "gas_services": "consumi gas servizi",
    "gas_total": "consumi gas totali",
    # Share of the electricity consumption attributable to tourism (phen_ee_tourism_share, computed in the data preparation).
    # Labels with "quota" are rounded to decimals by index_utils (see above): the others are rounded to integers.
    "hdd": "gradi giorno di riscaldamento",
    "quota_ee_turismo": "quota consumi elettrici attribuibile al turismo",
    "presenze_medie": "presenze turistiche medie giornaliere",
    "consumo_medio": "consumo elettrico medio giornaliero (kWh)",
    "affidabilita": "affidabilità della stima",
    # Seasonality cluster of the comune (TouristPresenceGiniIndicator, read from a comune -> cluster mapping).
    # The label has no "quota" / "PageRank": index_utils rounds it to an integer.
    "cluster_stagionalita": "cluster stagionale",
    "tipo_cluster": "tipo di cluster stagionale",  # text (e.g. "sci"): not rounded
    # Redistribution indicator (phen_ridistribuzione): persons, equivalent presences per day (window mean).
    # No "quota" in these labels on purpose: they are counts, index_utils rounds them to integers.
    "persone_gap": "presenze da attrarre (+) o da spostare (−)",
    "persone_spostate_nette": "presenze spostate entro il raggio (ingresso − uscita)",
    "eccesso_non_assorbibile": "presenze non assorbibili entro il raggio",
}

# ---------------------------------------------------------------------------
# Concrete phenomena
# ---------------------------------------------------------------------------


class PresencesPhenomenon(Phenomenon):
    """Total presences (tourists, excursionists, or combined)."""

    name = "presences"
    # Native resolution: daily × comune  →  both axes are 'identity'
    temporal_resolution = "daily"
    temporal_strategy = "identity"
    spatial_resolution = "comune"
    spatial_strategy = "identity"

    def __init__(self, source, col="presenze", name=None):
        super().__init__(source, col, agg="mean")
        if name is not None:
            self.name = name


class BedsPhenomenon(Phenomenon):
    """Total beds (all accommodation types)."""

    name = "beds"
    # Beds don't change day-to-day within a period; treat as constant daily.
    # If the source is already daily, temporal_strategy='identity' is fine too.
    temporal_resolution = "yearly"
    temporal_strategy = "constant"
    spatial_resolution = "comune"
    spatial_strategy = "identity"

    def __init__(self, source, col="tot_postiletto", name=None):
        super().__init__(source, col, agg="mean")
        if name is not None:
            self.name = name


class ExtraBedsPhenomenon(Phenomenon):
    """Beds in non-hotel (extra-alberghiero) facilities."""

    name = "extra_beds"
    temporal_resolution = "yearly"
    temporal_strategy = "constant"
    spatial_resolution = "comune"
    spatial_strategy = "identity"

    def __init__(self, source, col="tot_postiletto_non_conv", name=None):
        super().__init__(source, col, agg="mean")
        if name is not None:
            self.name = name


class FacilitiesTotalPhenomenon(Phenomenon):
    """Total accommodation facilities."""

    name = "Facilities_total"
    temporal_resolution = "yearly"
    temporal_strategy = "constant"
    spatial_resolution = "comune"
    spatial_strategy = "identity"

    def __init__(self, source, col="tot_strutture", name=None):
        super().__init__(source, col, agg="mean")
        if name is not None:
            self.name = name


class ExtraFacilitiesPhenomenon(Phenomenon):
    """Non-hotel facilities."""

    name = "extra_Facilities"
    temporal_resolution = "yearly"
    temporal_strategy = "constant"
    spatial_resolution = "comune"
    spatial_strategy = "identity"

    def __init__(self, source, col="tot_strutture_non_conv", name=None):
        super().__init__(source, col, agg="mean")
        if name is not None:
            self.name = name


class ConsumptionPhenomenon(Phenomenon):
    """
    One column of ``phen_consumption`` (``ee_services``, ``ee_total``, ``gas_services``, ``gas_total``):
    daily energy / gas consumption per comune. Nulls are kept (a comune without gas network has no gas
    consumption data): each indicator decides what a null means.
    """

    name = "consumption"
    temporal_resolution = "daily"
    temporal_strategy = "identity"
    spatial_resolution = "comune"
    spatial_strategy = "identity"

    def __init__(self, source, col, name=None):
        super().__init__(source, col, agg="mean", municipality_id_col="ID_COMUNE")
        self.name = name or col


class WeatherPhenomenon(Phenomenon):
    """
    One column of ``phen_meteo`` (``hdd``: daily heating degree days; ``t_mean``, ``t_min``, ``t_max``): daily
    temperature per comune, from the nearest Meteotrentino station. Days without temperature are empty
    (they are ignored when averaging, never treated as zeros).
    """

    name = "weather"
    temporal_resolution = "daily"
    temporal_strategy = "identity"
    spatial_resolution = "comune"
    spatial_strategy = "identity"

    def __init__(self, source, col="hdd", name=None):
        super().__init__(source, col, agg="mean", municipality_id_col="ID_COMUNE")
        self.name = name or col


class PopulationPhenomenon(Phenomenon):
    """
    Resident population.
    """

    name = "population"
    temporal_resolution = "yearly"
    temporal_strategy = "constant"  # broadcast uniformly across days
    spatial_resolution = "comune"
    spatial_strategy = "identity"

    def __init__(self, source, col="popolazione"):
        super().__init__(
            source,
            col,
            agg="mean",
            dtype={"LOCATION_ID": str},
        )


class SurfacePhenomenon(Phenomenon):
    """Comune surface in km² (ISTAT shapefile), constant over time."""

    name = "superficie"
    temporal_resolution = "yearly"
    temporal_strategy = "constant"
    spatial_resolution = "comune"
    spatial_strategy = "identity"

    def __init__(self, source, col="Shape_Area", name=None, province_code=22):
        super().__init__(source, col, agg="mean", municipality_id_col="PRO_COM_T")
        self.province_code = province_code
        if name is not None:
            self.name = name

    def resolve(self) -> "SurfacePhenomenon":
        if self._panel is not None:
            return self

        raw = gpd.read_file(self.source, ignore_geometry=True)
        raw = raw[raw["COD_PROV"] == self.province_code]
        self._panel = pd.DataFrame(
            {
                "ID_COMUNE": raw[self.municipality_id_col].astype(str),
                self.name: pd.to_numeric(raw[self.col], errors="coerce") / 1e6,
            }
        ).reset_index(drop=True)
        return self

    def _filter_by_date(
        self,
        start_date=None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if self._panel is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.resolve() must be called before _filter_by_date()"
            )
        if start_date is None or end_date is None:
            return pd.DataFrame(columns=["ID_COMUNE", "DATA", self.name])

        dates = pd.DataFrame({"DATA": pd.date_range(start_date, end_date, freq="D")})
        return self._panel.merge(dates, how="cross")[["ID_COMUNE", "DATA", self.name]]

    def date_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return pd.Timestamp.min, pd.Timestamp.max


class FlowIndicatorPhenomenon(Phenomenon):
    """
    One column of ``phen_flussi_od``: a yearly flow indicator per comune (quota in ingresso / uscita,
    quota intra-ambito, PageRank), for TOURIST, VISITOR or their weighted union (``_combined``).

    The value is yearly, so it is broadcast to every day of the year and averaged over the requested window
    (a window inside one year returns that year's value). ``name`` must be unique inside an Indicator.
    """

    temporal_resolution = "yearly"
    temporal_strategy = "constant"
    spatial_resolution = "comune"
    spatial_strategy = "identity"

    def __init__(self, source, col, name=None):
        super().__init__(source, col, agg="mean", municipality_id_col="ID_COMUNE")
        self.name = name or col


class ProvenienzaPhenomenon(FlowIndicatorPhenomenon):
    """
    ``hhi_provenienza`` of ``phen_provenienza``: yearly normalized Herfindahl-Hirschman index of the source markets
    (countries) of the tourist presences, per comune (0 = presences spread over many markets, 1 = a single market).
    Same yearly resolution as the OD flow indicators: broadcast to the days of the year and averaged over the window.
    """

    def __init__(self, source, col="hhi_provenienza", name=None):
        super().__init__(source, col, name=name or "hhi_provenienza")


class ProvenienzaVolumePhenomenon(FlowIndicatorPhenomenon):
    """
    One numeric arrivals column of ``phen_provenienza``: ``volume_mercato_1`` (arrivals of the main source
    market of the comune) or ``volume_totale`` (arrivals of all the markets), yearly, per comune.

    Same yearly resolution as the other provenienza phenomena: the yearly value is broadcast to every day of the year
    and averaged over the requested window (a window inside one year returns that year's arrivals). The comune-years
    without arrivals data (the arrivals file may cover fewer years than the presences, so the row exists in the file
    with an empty value) are dropped, so that they do not extend ``date_bounds()`` / ``years_range``.
    The comune-years whose ``volume_totale`` is below ``min_total`` (too few arrivals for a reliable share) are dropped
    too: both volume phenomena of an indicator must be given the same ``min_total``, so they drop the same rows
    and the regional value is computed on the reliable comuni only.
    ``nome_mercato_principale`` is text and is NOT a phenomenon: it cannot be averaged over a window.
    """

    def __init__(self, source, col, name=None, min_total: float = 0):
        super().__init__(source, col, name=name or col)
        self.min_total = min_total

    def _normalise_columns(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = super()._normalise_columns(raw)
        keep = df[self.name].notna() & (
            raw.loc[df.index, "volume_totale"] >= self.min_total
        )
        return df[keep]


class ElectricityTourismSharePhenomenon(Phenomenon):
    """
    One column of ``phen_ee_tourism_share``: the share of the electricity consumption attributable to tourism of
    each comune and YEAR (``quota_ee_turismo``) and its companions ``presenze_medie`` / ``consumo_medio`` (mean daily
    tourists / kWh of the year) and ``affidabilita`` (text: alta / media / bassa, ``text=True``). The model runs in
    the data preparation (``utils/consumption_share.py``) on the 12 months of the year; the comune-years that cannot
    be estimated are absent from the file.

    Yearly like the OD flow indicators: broadcast to the days of the year and averaged over the requested window
    (a window inside one year returns that year's value). The text column cannot be averaged: over a window with
    several years it takes the value of the latest year.
    """

    temporal_resolution = "yearly"
    temporal_strategy = "constant"
    spatial_resolution = "comune"
    spatial_strategy = "identity"

    def __init__(self, source, col, name=None, text=False):
        super().__init__(
            source, col, agg="last" if text else "mean", municipality_id_col="ID_COMUNE"
        )
        self.numeric = not text
        self.name = name or col


class RedistributionPhenomenon(Phenomenon):
    """
    One column of ``phen_ridistribuzione``: a DAILY value per comune of the redistribution analysis (index of
    imbalance in saturation points, or persons to attract / move / not absorbable), for the tourist profile, the
    visitor (excursionist) profile or their weighted union (``_combined``).

    The panel is computed offline (``data_preparation/gen_redistribution_dataframes.py``): the min-cost redistribution
    is solved once per day, so here a window is just a mask + the mean of the daily values. Days without data of a
    node are absent from the file (never zeros), so they do not enter the mean. ``name`` must be unique inside an
    Indicator.
    """

    temporal_resolution = "daily"
    temporal_strategy = "identity"
    spatial_resolution = "comune"
    spatial_strategy = "identity"

    def __init__(self, source, col, name=None):
        super().__init__(source, col, agg="mean", municipality_id_col="ID_COMUNE")
        self.name = name or col
