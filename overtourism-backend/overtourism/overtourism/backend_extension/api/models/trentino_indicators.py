from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from overtourism.overtourism.backend_extension.api.models.indicator import Indicator
from overtourism.overtourism.backend_extension.api.models.trenitno_phenomena import (
    BedsPhenomenon,
    ConsumptionPhenomenon,
    ElectricityTourismSharePhenomenon,
    FacilitiesTotalPhenomenon,
    FlowIndicatorPhenomenon,
    PopulationPhenomenon,
    PresencesPhenomenon,
    ProvenienzaPhenomenon,
    ProvenienzaVolumePhenomenon,
    SurfacePhenomenon,
)
from overtourism.overtourism.setup import index_data_path

# ---------------------------------------------------------------------------
# Source paths
# ---------------------------------------------------------------------------

MAP_SHAPEFILE = index_data_path / "Com01012026_g" / "Com01012026_g_WGS84.shp"
MAP_IDS = index_data_path / "cities_gdf_base_columns.geojson"
POPULATION_SOURCE = index_data_path / "phen_popolazione.parquet"
STRUCTURES_SOURCE = index_data_path / "phen_strutture.parquet"
PRESENCES_SOURCE = index_data_path / "phen_presenze.parquet"
FLOWS_OD_SOURCE = (
    index_data_path / "phen_flussi_od.parquet"
)  # yearly OD flow indicators per comune
PROVENIENZA_SOURCE = (
    index_data_path / "phen_provenienza.parquet"
)  # yearly concentration (normalized HHI) of the source markets per comune
CONSUMPTION_SOURCE = (
    index_data_path / "phen_consumption.parquet"
)  # daily electricity / gas consumption per comune (services and total)
EE_TOURISM_SHARE_SOURCE = (
    index_data_path / "phen_ee_tourism_share.parquet"
)  # yearly share of the electricity consumption attributable to tourism per comune (computed in the data preparation)
MACRO_AREAS_FILE = index_data_path / "map_comuni_into_apt.json"
CODICI_COMUNI_FILE = index_data_path / "mapping_comuni_ISTAT.json"

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Number of classes of the discretized PageRank map (quantiles, 1 = lowest ... N = highest)
PAGERANK_N_CLASSES = 5
MAIN_MARKET_MIN_ARRIVALS = 100


_VARIATION_RATE_KEY = "tasso-variazione"
_PERIOD_PERCENTAGE_IMPACT_KEY = "incidenza-periodo"

_REGISTRY: dict[str, Callable[[], Indicator]] = {
    "ricettivita": lambda: AccommodationCapacityIndicator(
        STRUCTURES_SOURCE, POPULATION_SOURCE
    ),
    "turisticita": lambda: TourismIndexIndicator(
        POPULATION_SOURCE, PRESENCES_SOURCE, "presenze_vodafone"
    ),
    "densita-turistica": lambda: TouristDensityIndicator(
        PRESENCES_SOURCE, MAP_SHAPEFILE
    ),
    "densita-antropica": lambda: AnthropogenicDensityIndicator(
        PRESENCES_SOURCE, POPULATION_SOURCE, MAP_SHAPEFILE
    ),
    "incidenza-alloggi-turistici-strutture": lambda: TouristAccommodationFacilitiesIndicator(
        STRUCTURES_SOURCE
    ),
    "incidenza-alloggi-disposizione-strutture": lambda: AvailableAccommodationFacilitiesIndicator(
        STRUCTURES_SOURCE
    ),
    "incidenza-alloggi-turistici-letti": lambda: TouristAccommodationBedsIndicator(
        STRUCTURES_SOURCE
    ),
    "incidenza-alloggi-disposizione-letti": lambda: AvailableAccommodationBedsIndicator(
        STRUCTURES_SOURCE
    ),
    "turismo-sommerso": lambda: HiddenTourismIndicator(PRESENCES_SOURCE),
    "carico-antropico": lambda: AnthropogenicLoadIndicator(
        POPULATION_SOURCE, PRESENCES_SOURCE
    ),
    "gini-stagionalita": lambda: TouristPresenceGiniIndicator(PRESENCES_SOURCE),
    "quota-mobilita-ingresso": lambda: InboundMobilityShareIndicator(FLOWS_OD_SOURCE),
    "quota-mobilita-uscita": lambda: OutboundMobilityShareIndicator(FLOWS_OD_SOURCE),
    "quota-intra-ambito": lambda: IntraAmbitoShareIndicator(FLOWS_OD_SOURCE),
    "pagerank-discretizzato": lambda: DiscretizedPageRankIndicator(
        FLOWS_OD_SOURCE, PAGERANK_N_CLASSES
    ),
    "concentrazione-mercati-provenienza": lambda: SourceMarketsConcentrationIndicator(
        PROVENIENZA_SOURCE
    ),
    "dipendenza-mercato-principale": lambda: MainMarketDependenceIndicator(
        PROVENIENZA_SOURCE
    ),
    "impatto-turismo-consumi-elettrici": lambda: TourismElectricityConsumptionImpactIndicator(
        CONSUMPTION_SOURCE
    ),
    "impatto-turismo-consumi-gas": lambda: TourismGasConsumptionImpactIndicator(
        CONSUMPTION_SOURCE
    ),
    "quota-consumi-elettrici-attribuibili-turismo": lambda: TourismElectricityShareIndicator(
        EE_TOURISM_SHARE_SOURCE
    ),
    "indice-stagionalita-presenze": lambda: SeasonalityOfPresencesIndicator(),
    "distanza-media-turisti": lambda: AverageTouristTravelTimeIndicator(),
    "dipendenza-domanda-prossimita": lambda: ProximityDemandDependenceIndicator(),
    "affollamento-turistico": lambda: TouristCrowdingIndicator(),
    "sostenibilita-territoriale": lambda: TerritorialSustainabilityIndicator(),
    _VARIATION_RATE_KEY: lambda: VariationRateIndicator(),
    _PERIOD_PERCENTAGE_IMPACT_KEY: lambda: PeriodPercentageImpactIndicator(),
}

_INDICATOR_CACHE: dict[str, Indicator] = {}


def get_indicator(key: str) -> Indicator:
    if key not in _INDICATOR_CACHE:
        if key not in _REGISTRY:
            raise KeyError(f"Unknown indicator '{key}'. Available: {sorted(_REGISTRY)}")
        _INDICATOR_CACHE[key] = _REGISTRY[key]()
    return _INDICATOR_CACHE[key]


# ---------------------------------------------------------------------------
# Concrete indicators
# ---------------------------------------------------------------------------


class AccommodationCapacityIndicator(Indicator):
    category = "Pressione turistica"
    order = 10
    """Beds per resident: NUMLETTI_TOT / POPOLAZIONE."""

    name = "Indice di ricettività"
    description = "L'indice di ricettività definisce il <strong>rapporto</strong> fra i <strong>letti presenti negli esercizi ricettivi</strong> e gli <strong>abitanti</strong> di una stessa area. L'indice è una misura della capacità turistica rispetto alla dimensione, in termini di popolazione, di un'area. L'indice è calcolato partendo dai dati ISPAT relativi alla popolazione residente e alla consistenza degli esercizi alberghieri e extra-alberghieri."
    index_value_unit_description = "Rapporto posti letto / popolazione"

    def __init__(self, source_file1, source_file2):
        super().__init__(
            phenomena=[
                BedsPhenomenon(source_file1),
                PopulationPhenomenon(source_file2),
            ],
            combinator=self.divide("beds", "population"),
        )


class TourismIndexIndicator(Indicator):
    category = "Pressione turistica"
    order = 10
    """Total presences per resident."""

    name = "Indice di turisticità"
    description = "L'indice di turisticità definisce il <strong>rapporto</strong> fra il <strong>numero medio giornaliero di turisti negli esercizi ricettivi</strong> di una specifica area e il<strong>numero di abitanti</strong> della stessa area. L'indice fornisce una misura dell'effettiva incidenza del turismo rispetto alla dimensione, in termini di popolazione, di un'area. L'indice è calcolato partendo dai dati Vodafone per quanto riguarda le presenze turistiche e dai dati ISPAT relativi alla popolazione residente."
    index_value_unit_description = "Rapporto fra numero turisti e popolazione"

    def __init__(self, source_file1, source_file2, presences_col_name):
        super().__init__(
            phenomena=[
                PresencesPhenomenon(source_file2, presences_col_name),
                PopulationPhenomenon(source_file1),
            ],
            combinator=self.divide("presences", "population"),
        )


class TouristDensityIndicator(Indicator):
    category = "Pressione turistica"
    order = 10
    """Vodafone tourists + visitors presences per km² of the comune's surface."""

    name = "Indice di densità turistica"
    description = "L'indice di densità turistica definisce il <strong>rapporto</strong> fra il <strong>numero medio giornaliero di turisti ed escursionisti</strong> presenti in un comune e la sua <strong>superficie</strong> in km². L'indice fornisce una misura della pressione turistica sul territorio. L'indice è calcolato partendo dai dati Vodafone per quanto riguarda le presenze di turisti ed escursionisti e dai confini amministrativi ISTAT per la superficie dei comuni."
    index_value_unit_description = "Presenze turistiche per km²"

    def __init__(self, presences_source, surface_source):
        super().__init__(
            phenomena=[
                PresencesPhenomenon(
                    presences_source,
                    col="presenze_vodafone",
                    name="presenze_vodafone",
                ),
                PresencesPhenomenon(
                    presences_source,
                    col="presenze_escursionisti_vodafone",
                    name="presenze_escursionisti_vodafone",
                ),
                SurfacePhenomenon(surface_source),
            ],
            combinator=self.compute_density,
        )

    def compute_density(
        self,
        df: pd.DataFrame,
        **extra,
    ) -> pd.Series:
        numerator = df[["presenze_vodafone", "presenze_escursionisti_vodafone"]].sum(
            axis=1, min_count=1
        )  # one of the two missing = 0; both missing = no data (null)
        return numerator.div(df["superficie"].replace(0, np.nan))


class AnthropogenicDensityIndicator(Indicator):
    category = "Pressione turistica"
    order = 10
    """Inhabitants and Vodafone tourist presences per km² of comune area."""

    name = "Indice di densità antropica"
    description = "L'indice di densità antropica misura il rapporto fra la popolazione residente e il numero medio giornaliero di turisti ed escursionisti rilevati da Vodafone, rispetto alla superficie del comune in km²."
    index_value_unit_description = "Abitanti e presenze turistiche per km²"

    def __init__(self, presences_source, population_source, surface_source):
        super().__init__(
            phenomena=[
                PresencesPhenomenon(
                    presences_source,
                    col="presenze_vodafone",
                    name="presenze_vodafone",
                ),
                PresencesPhenomenon(
                    presences_source,
                    col="presenze_escursionisti_vodafone",
                    name="presenze_escursionisti_vodafone",
                ),
                PopulationPhenomenon(population_source),
                SurfacePhenomenon(surface_source),
            ],
            combinator=self.compute_density_anthropogenic,
        )

    def compute_density_anthropogenic(
        self,
        df: pd.DataFrame,
        **extra,
    ) -> pd.Series:
        vodafone_presences = df[
            ["presenze_vodafone", "presenze_escursionisti_vodafone"]
        ].sum(
            axis=1, min_count=1
        )  # one of the two missing = 0; both missing = no data (null)
        numerator = vodafone_presences + df["population"]
        return numerator.div(df["superficie"].replace(0, np.nan))


def _hospitality_share(numerator: str, denominator_components: tuple[str, ...]):
    def combine(df: pd.DataFrame, **_extra) -> pd.Series:
        denominator = df[list(denominator_components)].sum(axis=1)
        return df[numerator].div(denominator.replace(0, np.nan))

    combine.__name__ = f"hospitality_share({numerator})"
    return combine


class _HospitalityShareIndicator(Indicator):
    """Common construction for a hospitality component's share of a total."""

    component_columns: tuple[str, ...] = ()
    numerator_column: str = ""
    phenomenon_class = None

    def __init__(self, source_file):
        super().__init__(
            phenomena=[
                self.phenomenon_class(source_file, col=column, name=column)
                for column in self.component_columns
            ],
            combinator=_hospitality_share(
                self.numerator_column, self.component_columns
            ),
        )


class _HospitalityFacilitiesShareIndicator(_HospitalityShareIndicator):
    phenomenon_class = FacilitiesTotalPhenomenon
    index_value_unit_description = "Quota delle strutture sul totale"


class TouristAccommodationFacilitiesIndicator(_HospitalityFacilitiesShareIndicator):
    category = "Strutture"
    order = 30
    """Tourist accommodation facilities over tourist, hotel and extra-hotel facilities."""

    name = "Incidenza Alloggi turistici (strutture)"
    description = (
        "Misura la quota delle <strong>strutture degli alloggi turistici</strong> "
        "sul totale composto da alloggi turistici, strutture alberghiere ed "
        "extra-alberghiere."
    )
    numerator_column = "strutture_alloggi_turistici"
    component_columns = (
        "strutture_alloggi_turistici",
        "strutture_alb",
        "strutture_exalb",
    )

class TouristPresenceGiniIndicator(Indicator):
    """Normalized Gini of the temporal distribution of daily tourist presences."""

    name = "Concentrazione temporale delle presenze turistiche (Gini)"
    type = INDICATORS_TYPES[1]
    _PRESENCE_COLUMN = "presenze_vodafone"
    _LOW_COVERAGE_VALUE = -1.0
    _MIN_COVERAGE = 0.8

    description = (
        "L'indice di Gini misura quanto le presenze turistiche giornaliere in un comune "
        "sono concentrate nel periodo selezionato. "
        "Il valore 0 indica una distribuzione uniforme, il valore 1 la massima concentrazione. "
        f"Il calcolo è disponibile solo con almeno l'{_MIN_COVERAGE*100}% di copertura temporale; "
        f"i comuni sotto questa soglia hanno valore {_LOW_COVERAGE_VALUE} "
    )
    index_value_unit_description = "Indice di Gini normalizzato (0 = uniforme, 1 = concentrato)"
    availableForVariation = False

    def __init__(self, source_file):
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

    @staticmethod
    def _gini(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        # Validity check: if less than 2 days are available, either negative values are present, or total sum is zero, NaN is returned
        if (values.size < 2 or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0):
            return np.nan
        
        values = np.sort(values) # It has to be computed on the ordered time series 
        count = values.size
        weighted_sum = np.dot(2 * np.arange(1, count + 1) - count - 1, values)
        return float(weighted_sum / (count * values.sum()) * count / (count - 1))  # Normalized (index between 0 and 1)

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
            raise RuntimeError("Tourist presences must be resolved before Gini calculation")
        source = source[source["DATA"].between(start, end)][["ID_COMUNE", "DATA", self._PRESENCE_COLUMN]]
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
            if source_by_comune.get(comune, pd.Series(dtype=float)).reindex(data_range).notna().mean() < self._MIN_COVERAGE:
                gini_by_comune[comune] = self._LOW_COVERAGE_VALUE
                continue

            gini_by_comune[comune] = self._gini(observed_by_comune.get(comune, pd.Series(dtype=float)).dropna().to_numpy(dtype=float))

        return df["ID_COMUNE"].astype(str).map(gini_by_comune)


class AvailableAccommodationFacilitiesIndicator(_HospitalityFacilitiesShareIndicator):
    category = "Strutture"
    order = 30
    """Available accommodation facilities over all accommodation facilities."""

    name = "Incidenza Alloggi a disposizione (strutture)"
    description = (
        "Misura la quota delle <strong>strutture degli alloggi a disposizione</strong> "
        "sul totale composto da alloggi turistici, alloggi a disposizione, "
        "strutture alberghiere ed extra-alberghiere."
    )
    numerator_column = "strutture_alloggi_disp"
    component_columns = (
        "strutture_alloggi_turistici",
        "strutture_alloggi_disp",
        "strutture_alb",
        "strutture_exalb",
    )


class _HospitalityBedsShareIndicator(_HospitalityShareIndicator):
    phenomenon_class = BedsPhenomenon
    index_value_unit_description = "Quota dei posti letto sul totale"


class TouristAccommodationBedsIndicator(_HospitalityBedsShareIndicator):
    category = "Strutture"
    order = 30
    """Tourist accommodation beds over tourist, hotel and extra-hotel beds."""

    name = "Incidenza Alloggi turistici (posti letto)"
    description = (
        "Misura la quota dei <strong>posti letto degli alloggi turistici</strong> "
        "sul totale composto da alloggi turistici, strutture alberghiere ed "
        "extra-alberghiere."
    )
    numerator_column = "letti_alloggi_turistici"
    component_columns = (
        "letti_alloggi_turistici",
        "letti_alb",
        "letti_exalb",
    )


class AvailableAccommodationBedsIndicator(_HospitalityBedsShareIndicator):
    category = "Strutture"
    order = 30
    """Available accommodation beds over all accommodation beds."""

    name = "Incidenza Alloggi a disposizione (posti letto)"
    description = (
        "Misura la quota dei <strong>posti letto degli alloggi a disposizione</strong> "
        "sul totale composto da alloggi turistici, alloggi a disposizione, "
        "strutture alberghiere ed extra-alberghiere."
    )
    numerator_column = "letti_alloggi_disp"
    component_columns = (
        "letti_alloggi_turistici",
        "letti_alloggi_disp",
        "letti_alb",
        "letti_exalb",
    )


class HiddenTourismIndicator(Indicator):
    category = "Pressione turistica"
    order = 10
    """Calculates the hidden tourism indicator as the ratio between Vodafone registered attendences and accomodancy ones"""

    name = "Turismo non intercettato"
    description = "L'indice misura il <strong>rapporto</strong> fra le <strong>presenze di turisti misurate</strong> attraverso l'analisi di dati da rete di telefonia mobile e le <strong>presenze ufficiali</strong> di turisti in strutture alberghiere e extra-alberghiere. L'indice è stato calcolato partendo dai dati ISPAT sul movimento turistico e dai dati Vodafone relativi alle presenze misurate."
    index_value_unit_description = "Rapporto fra presenze misurate e ufficiali"

    def __init__(self, source_file):
        super().__init__(
            phenomena=[
                PresencesPhenomenon(
                    source_file, col="presenze_vodafone", name="presenze_vodafone"
                ),
                PresencesPhenomenon(
                    source_file,
                    col="presenze_alb",
                    name="presenze_alb",
                ),
                PresencesPhenomenon(
                    source_file,
                    col="presenze_xalb",
                    name="presenze_xalb",
                ),
            ],
            combinator=self.compute_hidden_factor,
        )

    def compute_hidden_factor(
        self,
        df: pd.DataFrame,
        **extra,
    ) -> pd.Series:
        denominator = df["presenze_alb"] + df["presenze_xalb"]
        return (
            df["presenze_vodafone"]
            .div(denominator.where(denominator != 0))
            .fillna(df["presenze_vodafone"])
        )


class AnthropogenicLoadIndicator(Indicator):
    category = "Pressione turistica"
    order = 10
    """Calculates the anthropogenic load as the ratio between Vodafone
    registered presences (tourists and excursionists) and resident population.
    """

    name = "Carico Antropico"
    description = (
        "L'indice misura il rapporto fra le presenze complessive rilevate "
        "attraverso i dati Vodafone, comprendenti turisti ed escursionisti, "
        "e la popolazione residente."
    )
    index_value_unit_description = (
        "Rapporto fra presenze Vodafone complessive e popolazione residente"
    )

    def __init__(self, population_source_file, presences_source_file):
        super().__init__(
            phenomena=[
                PopulationPhenomenon(
                    population_source_file,
                ),
                PresencesPhenomenon(
                    presences_source_file,
                    col="presenze_vodafone",
                    name="presenze_vodafone",
                ),
                PresencesPhenomenon(
                    presences_source_file,
                    col="presenze_escursionisti_vodafone",
                    name="presenze_escursionisti_vodafone",
                ),
            ],
            combinator=self.compute_anthropogenic_load,
        )

    def compute_anthropogenic_load(
        self,
        df: pd.DataFrame,
        **extra,
    ) -> pd.Series:
        total_presences = (
            df["presenze_vodafone"] + df["presenze_escursionisti_vodafone"]
        )

        population = df["population"]

        return total_presences.div(population.where(population != 0))


# ---------------------------------------------------------------------------
# OD flow indicators (Vodafone co-visits between areas)
# ---------------------------------------------------------------------------
# They read `phen_flussi_od`: one row per comune and YEAR with, for every metric, a column for the TOURIST
# profile, one for the VISITOR (excursionist) profile and one for their weighted union (`<metric>_combined`).
# The union is computed in the data preparation on the user-days (a visitor-day counts `w` tourist-days,
# w = 0.5 fixed, see COVISITS_PROFILE_WEIGHTS in data_preparation), so the indicators here only read it.
# The values are yearly: a requested window returns the value of its year(s) (mean over the years of the window),
# and the seasonality filter does not change them.

_FLOW_PROFILE_SUFFIXES = ("combined", "tourist", "visitor")
_FLOW_COMMON_DESCRIPTION = (
    " Turisti ed escursionisti sono <strong>aggregati con un peso per ciascun profilo</strong>, "
    "stimato dai dati (una giornata-utente di escursionista non pesa come una di turista); "
    "i valori per singolo profilo sono riportati a fianco. "
    "L'indice è annuale e si basa sui dati Vodafone di co-visitazione fra aree."
)


class _FlowIndicator(Indicator):
    """Base of the indicators that read one metric of phen_flussi_od (tourists + excursionists, weighted)."""

    metric: str = ""
    year_only = True

    def __init__(self, source_file, combinator=None):
        super().__init__(
            phenomena=[
                FlowIndicatorPhenomenon(
                    source_file, f"{self.metric}_{suffix}", f"{self.metric}_{suffix}"
                )
                for suffix in _FLOW_PROFILE_SUFFIXES
            ],
            combinator=combinator or self._weighted_union,
        )

    def _weighted_union(self, df: pd.DataFrame, **_extra) -> pd.Series:
        return df[f"{self.metric}_combined"]


class InboundMobilityShareIndicator(_FlowIndicator):
    category = "Flussi"
    order = 50
    """UD(*->j) / T(j): share of the users present in j who arrive from another area."""

    metric = "quota_ingresso"
    name = "Flussi - Quota di mobilità in ingresso"
    description = (
        "La quota di mobilità in ingresso misura la <strong>quota degli utenti presenti in un'area che "
        "arrivano da un'altra area</strong> (rapporto fra gli utenti in arrivo da altre aree e gli utenti "
        "presenti). Valori alti indicano località di tappa, raggiunte durante uno spostamento; valori bassi "
        "indicano località di destinazione o di pernottamento stabile."
        + _FLOW_COMMON_DESCRIPTION
    )
    index_value_unit_description = (
        "Rapporto fra utenti in arrivo da altre aree e utenti presenti"
    )


class OutboundMobilityShareIndicator(_FlowIndicator):
    category = "Flussi"
    order = 50
    """UD(j->*) / T(j): share of the users present in j who go on to another area."""

    metric = "quota_uscita"
    name = "Quota di mobilità in uscita"
    description = (
        "La quota di mobilità in uscita misura la <strong>quota degli utenti presenti in un'area che "
        "proseguono verso un'altra area</strong> (rapporto fra gli utenti diretti verso altre aree e gli "
        "utenti presenti). Valori alti indicano località da cui i visitatori ripartono per visitare altre aree."
        + _FLOW_COMMON_DESCRIPTION
    )
    index_value_unit_description = (
        "Rapporto fra utenti diretti verso altre aree e utenti presenti"
    )


class IntraAmbitoShareIndicator(_FlowIndicator):
    category = "Flussi"
    order = 50
    """Share of the UD flow of an area that stays inside its own ambito."""

    metric = "quota_intra_ambito"
    name = "Quota intra-ambito"
    description = (
        "La quota intra-ambito misura la <strong>quota dei flussi di un'area (in ingresso e in uscita) "
        "scambiati con aree dello stesso ambito turistico</strong>; il resto è mobilità inter-ambito. "
        "Valori alti indicano un'area i cui flussi restano all'interno del proprio ambito."
        + _FLOW_COMMON_DESCRIPTION
    )
    index_value_unit_description = "Quota dei flussi con aree dello stesso ambito"


class DiscretizedPageRankIndicator(_FlowIndicator):
    category = "Flussi"
    order = 100
    """
    PageRank of the network of the flows between areas, discretized in ``n_classes`` quantile classes.

    INDICE is the class: 1 = lowest PageRank ... n_classes = highest, 0 = no data. Classes are quantiles of
    the comuni of the request (values that are tied stay in the same class, so the classes can have different
    sizes), so they show the *relative* position of a comune. The continuous PageRank values stay in the
    columns next to INDICE. Classes of different years are not comparable as a time series, hence
    ``availableForVariation = False``.
    """

    metric = "pagerank"
    name = "Centralità nella rete"
    description = (
        "Questo indice misura la <strong>centralità di un'area nella rete dei flussi</strong> fra aree: è alto "
        "per le tappe che ricevono flusso da aree a loro volta rilevanti. Per la lettura su mappa il valore è "
        "<strong>discretizzato in classi di quantili</strong> (1 = centralità più bassa, "
        "{n} = più alta; 0 = dato non disponibile): le classi indicano la posizione relativa fra i comuni."
        + _FLOW_COMMON_DESCRIPTION
    )
    index_value_unit_description = "Classe di PageRank (quantili: 1 = più basso)"
    availableForVariation = False

    def __init__(self, source_file, n_classes: int = PAGERANK_N_CLASSES):
        if (
            not isinstance(n_classes, int)
            or isinstance(n_classes, bool)
            or n_classes < 2
        ):
            raise ValueError(f"n_classes must be an integer >= 2, got {n_classes!r}")
        self.n_classes = n_classes
        self.description = self.description.format(n=n_classes)
        super().__init__(source_file, combinator=self._discretize)

    def _discretize(self, df: pd.DataFrame, **_extra) -> pd.Series:
        """Quantile class (1..n_classes) of the combined PageRank among the comuni of ``df``; 0 (documented class
        "dato non disponibile") where missing: explicit, the base Indicator keeps nulls.
        """
        value = df["pagerank_combined"]
        classes = pd.Series(np.nan, index=df.index, dtype=float)
        valid = value.notna()
        if valid.sum() < 2:
            return classes.fillna(0)
        share_below = value[valid].rank(method="average", pct=True)
        classes[valid] = np.ceil(share_below * self.n_classes).clip(1, self.n_classes)
        return classes.fillna(0)


# ---------------------------------------------------------------------------
# Provenienza (source markets of the tourists)
# ---------------------------------------------------------------------------
# Reads `phen_provenienza`: one row per comune and YEAR with `hhi_provenienza`, the normalized
# Herfindahl-Hirschman index of the tourist presences by country of origin (computed in the data preparation,
# `territorial_indexes/utils/provenienza.py`). Yearly like the OD flow indicators: a requested window returns the
# value of its year(s) and the seasonality filter does not change it.


class SourceMarketsConcentrationIndicator(Indicator):
    category = "Provenienza"
    order = 100
    """Normalized HHI of the source markets of the tourists: 0 = many markets evenly, 1 = a single market."""

    name = "Provenienza - Indice di concentrazione mercati provenienza"
    description = (
        "L'indice di concentrazione dei mercati di provenienza misura quanto le <strong>presenze di turisti "
        "provengono da pochi paesi</strong>. È l'<strong>indice di Herfindahl-Hirschman (HHI) normalizzato</strong> "
        "calcolato sulle quote di presenze per paese di provenienza (l'Italia è un mercato): vale 0 quando le "
        "presenze sono distribuite in modo uniforme fra molti mercati e 1 quando provengono da un solo mercato. "
        "Valori alti indicano una forte dipendenza da pochi mercati. L'indice è annuale, riguarda solo i turisti "
        "ed è calcolato sui dati Vodafone; è lasciato vuoto per le aree con presenze troppo basse. "
        "I conteggi molto piccoli non sono forniti da Vodafone: l'indice descrive i mercati osservati."
    )
    index_value_unit_description = (
        "Indice HHI normalizzato (0 = mercati diversificati, 1 = un solo mercato)"
    )
    year_only = True

    def __init__(self, source_file):
        super().__init__(
            phenomena=[ProvenienzaPhenomenon(source_file)],
            combinator=self._hhi,
        )

    @staticmethod
    def _hhi(df: pd.DataFrame, **_extra) -> pd.Series:
        return df["hhi_provenienza"]


class MainMarketDependenceIndicator(Indicator):
    category = "Provenienza"
    order = 100
    """Share of the yearly tourist arrivals that come from the main source market: volume_mercato_1 / volume_totale."""

    name = "Indice di dipendenza dal principale mercato di origine"
    MAIN_MARKETS_COLUMN = "main_markets"
    MAIN_MARKET_MARKETS_DESCRIPTION = "Italia, altri paesi europei, paesi extra-europei"
    description = (
        "L'indice di dipendenza dal principale mercato di origine misura la <strong>quota degli arrivi di turisti "
        "che proviene dal mercato di origine più importante</strong> di un comune (rapporto fra gli arrivi dal mercato "
        "principale e gli arrivi totali). Valori alti indicano una forte dipendenza da un solo mercato; il mercato "
        "principale può essere diverso da un comune all'altro. I mercati sono: "
        + MAIN_MARKET_MARKETS_DESCRIPTION
        + ". Sono contati gli <strong>arrivi</strong> e non le presenze, così un soggiorno lungo non pesa più "
        "di uno breve. L'indice è annuale, riguarda solo i turisti ed è calcolato sui dati Vodafone; è lasciato vuoto "
        "per i comuni con arrivi troppo bassi. I conteggi molto piccoli non sono forniti da Vodafone: l'indice descrive "
        "i mercati osservati."
    )
    index_value_unit_description = (
        "Quota degli arrivi dal mercato principale (0-1; 1 = un solo mercato)"
    )
    year_only = True

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


# ---------------------------------------------------------------------------
# Consumption indicators (impact of tourism on electricity / gas consumption)
# ---------------------------------------------------------------------------
# They read `phen_consumption`: one row per comune and DAY with the electricity (`ee_*`) and gas (`gas_*`)
# consumption of the services sector (`*_services`, the one driven by tourism) and of all sectors (`*_total`).
# The indicator is the share of the services consumption over the total one, computed on the averages of the
# requested window. Where the total consumption is missing or zero (e.g. a comune without gas network) the index has no
# value (null); services consumption missing while the total is known counts as zero.


class _ConsumptionImpactIndicator(Indicator):
    """Base: share of the services consumption over the total consumption (nulls = 0)."""

    services_column: str = ""
    total_column: str = ""
    index_value_unit_description = (
        "Rapporto fra consumi del settore servizi e consumi totali"
    )

    def __init__(self, source_file):
        super().__init__(
            phenomena=[
                ConsumptionPhenomenon(source_file, col, name=col)
                for col in (self.services_column, self.total_column)
            ],
            combinator=self._services_share,
        )

    def _services_share(self, df: pd.DataFrame, **_extra) -> pd.Series:
        # no total consumption (e.g. a comune without gas network) or a zero one: no value (null), not 0.
        # Services missing while the total is known: the comune has no consumption in those categories (0).
        services = df[self.services_column].fillna(0)
        total = df[self.total_column]
        return services.div(total.where(total != 0))


class TourismElectricityConsumptionImpactIndicator(_ConsumptionImpactIndicator):
    category = "Consumi"
    order = 150
    """Electricity consumption of the services sector over the total electricity consumption."""

    name = "Impatto turismo sui consumi elettrici"
    description = (
        "L'indice misura l'impatto del turismo sui consumi elettrici come <strong>rapporto</strong> fra i "
        "<strong>consumi elettrici del settore servizi</strong> e i <strong>consumi elettrici totali</strong> "
        "dell'area. Valori alti indicano un'area in cui i consumi sono dominati dalle attività legate ai servizi "
        "e quindi al turismo. I comuni senza consumi totali non hanno valore; se mancano solo i consumi del settore servizi "
        "sono considerati pari a zero."
    )
    services_column = "ee_services"
    total_column = "ee_total"


class TourismGasConsumptionImpactIndicator(_ConsumptionImpactIndicator):
    category = "Consumi"
    order = 150
    """Gas consumption of the services sector over the total gas consumption."""

    name = "Impatto turismo sui consumi di gas"
    description = (
        "L'indice misura l'impatto del turismo sui consumi di gas come <strong>rapporto</strong> fra i "
        "<strong>consumi di gas del settore servizi</strong> e i <strong>consumi di gas totali</strong> "
        "dell'area. Valori alti indicano un'area in cui i consumi sono dominati dalle attività legate ai servizi "
        "e quindi al turismo. I comuni senza consumi totali non hanno valore; se mancano solo i consumi del settore servizi "
        "sono considerati pari a zero."
    )
    services_column = "gas_services"
    total_column = "gas_total"


# ---------------------------------------------------------------------------
# Share of the electricity consumption attributable to tourism
# ---------------------------------------------------------------------------
# It reads `phen_ee_tourism_share`: one row per comune and YEAR with the share of the services-sector electricity
# consumption that goes with the tourist presences (`quota_ee_turismo`), computed in the data preparation
# (`utils/consumption_share.py`: monthly regression of the 12 months of the year, controlling for the temperature)
# from `phen_consumption`, `phen_presenze` and `phen_meteo`. Yearly like the other yearly indicators: a requested
# window returns the value of its year(s) and the seasonality filter does not change it. The comuni that cannot
# be estimated in a year (a month without data) are absent from the file, so they have no value, not a zero.


class TourismElectricityShareIndicator(Indicator):
    category = "Consumi"
    order = 150
    """Share (0-1) of the services-sector electricity consumption that goes with the tourist presences."""

    name = "Quota consumi elettrici attribuibili al turismo"
    description = (
        "L'indice misura la <strong>quota dei consumi elettrici del settore servizi attribuibile al "
        "turismo</strong>: la parte del consumo medio giornaliero che varia insieme alle presenze di turisti. "
        "La quota è stimata, per ogni comune e per ogni anno, con una <strong>regressione sui 12 mesi</strong> "
        "dell'anno che tiene conto della temperatura (gradi giorno di riscaldamento): i comuni con dati mancanti "
        "in qualche mese non hanno valore. L'indice è un'associazione statistica, non una misura diretta: il campo "
        "<em>affidabilità</em> (alta, media, bassa) indica quanto i dati del solo comune sostengono un effetto "
        "positivo delle presenze; per i comuni vicini, che condividono le presenze di una stessa area Vodafone, le stime non "
        "sono indipendenti. Il filtro stagionale non modifica l'indice. I consumi sono mensili e distribuiti "
        "uniformemente sui giorni."
    )
    index_value_unit_description = (
        "Quota dei consumi elettrici attribuibile al turismo (0-1)"
    )
    year_only = True
    availableForVariation = False

    SHARE_COLUMN = "quota_ee_turismo"
    TEXT_COLUMN = "affidabilita"  # alta / media / bassa
    COMPANION_COLUMNS = ("presenze_medie", "consumo_medio", TEXT_COLUMN)

    def __init__(self, source_file):
        super().__init__(
            phenomena=[
                ElectricityTourismSharePhenomenon(
                    source_file, col, text=(col == self.TEXT_COLUMN)
                )
                for col in (self.SHARE_COLUMN, *self.COMPANION_COLUMNS)
            ],
            combinator=self._share,
        )

    @staticmethod
    def _share(df: pd.DataFrame, **_extra) -> pd.Series:
        return df["quota_ee_turismo"]


class _NotImplementedIndexIndicator(Indicator):
    """Metadata-only placeholder for an index whose methodology is not implemented yet."""

    category = "Indicatori in sviluppo"
    availableForVariation = False
    _years_range = {
        "min_year": 2022,
        "max_year": 2025,
    }

    def __init__(self):
        super().__init__(phenomena=[], combinator=self._not_implemented)

    @staticmethod
    def _not_implemented(df: pd.DataFrame, **_extra) -> pd.Series:
        # #TODO Implement the index methodology and required source data.
        raise NotImplementedError(
            "This indicator is a placeholder; its calculation has not been implemented yet."
        )


class SeasonalityOfPresencesIndicator(_NotImplementedIndexIndicator):
    # #TODO Define and document the seasonality index methodology.
    name = "Indice di stagionalità delle presenze"
    description = "TODO"
    index_value_unit_description = "TODO"
    category = "Pressione turistica"
    order = 10
    year_only = True


class AverageTouristTravelTimeIndicator(_NotImplementedIndexIndicator):
    # #TODO Estimate travel time using tourist origin nationality at the available data granularity.
    name = "Distanza media turisti"
    description = (
        "Stima del tempo medio impiegato dai turisti per raggiungere la località, "
        "basata sulla nazionalità di provenienza, data la granularità dei dati disponibili."
    )
    index_value_unit_description = "Tempo medio stimato di viaggio"
    category = "Provenienza"
    order = 100


class ProximityDemandDependenceIndicator(_NotImplementedIndexIndicator):
    # #TODO Calculate the share of tourists originating within a two-hour travel distance.
    name = "Indice di dipendenza dalla domanda di prossimità"
    description = (
        "Misura la quota di turisti che proviene da destinazioni raggiungibili entro "
        "due ore di distanza. La metodologia di stima della distanza è da definire."
    )
    index_value_unit_description = "Quota di turisti provenienti entro 2 ore (0-1)"
    category = "Provenienza"
    order = 100
    year_only = True


class TouristCrowdingIndicator(_NotImplementedIndexIndicator):
    # #TODO Define and normalize the component metrics and aggregation method.
    name = "Affollamento turistico"
    description = (
        "Indice composito che combina diverse metriche per offrire una panoramica "
        "complessiva del livello di affollamento turistico di una località."
    )
    index_value_unit_description = "Indice composito (scala da definire)"
    category = "Pressione turistica"
    order = 10


class TerritorialSustainabilityIndicator(_NotImplementedIndexIndicator):
    # #TODO Define component metrics, normalization, weighting, and aggregation.
    name = "Indice di sostenibilità territoriale"
    description = (
        "Macro-indicatore composito che integra diverse metriche per fornire una "
        "valutazione sintetica della sostenibilità territoriale in relazione al turismo."
    )
    index_value_unit_description = "Indice composito (scala da definire)"
    category = "Pressione turistica"
    order = 100


class VariationRateIndicator(Indicator):
    category = "Internal"
    order = 900
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
    ``"indice-turisticita"``) — exactly like ``seasonality`` is an extra
    field on ``SeasonalityIndicator`` above. This keeps a single
    "tasso-variazione" entry in the registry that works for every
    combinable indicator.

    Because the wrapped indicator is only known once a request comes in,
    this class doesn't use the phenomena/combinator pipeline from the base
    class at all — ``_compute_indicator`` is fully overridden below, and
    ``phenomena``/``combinator`` are left empty/no-op.
    """

    name = "Tasso di variazione"
    description = "Il tasso di variazione misura la <strong>variazione percentuale</strong> di un indicatore fra il <strong>periodo selezionato</strong> ed un <strong>periodo storicizzato di confronto</strong>. È utile per confrontare due intervalli temporali (es. due stagioni, due anni) sullo stesso indice."
    availableForVariation = False
    extraFields = ["indicator", "start_date_comparison", "end_date_comparison"]
    internal_only = True
    index_value_unit_description = "Percentuale di variazione tra due periodi"

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


class PeriodPercentageImpactIndicator(Indicator):
    category = "Internal"
    order = 910
    """
    Percentage weight that a seasonal sub-period has on the total value of
    another indicator over the full requested period.

    Hybrid of VariationRateIndicator (wrapped indicator selected at request
    time via the "indicator" extra field) and SeasonalityIndicator (result
    expresses a sub-period's share of a total period, via "seasonality").

    result = 100 * INDICE(wrapped, full_period, seasonality=X)
                 / INDICE(wrapped, full_period, no seasonality filter)
    """

    name = "Incidenza percentuale del periodo"
    description = (
        "L'incidenza percentuale del periodo misura il <strong>peso "
        "percentuale</strong> che un <strong>sotto-periodo stagionale</strong> "
        "(es. alta stagione, periodo di spalla, weekend) ha sul "
        "<strong>totale del periodo selezionato</strong>, per un indicatore "
        "a scelta."
    )
    availableForVariation = False
    extraFields = ["indicator", "seasonality"]
    internal_only = True
    index_value_unit_description = (
        "Percentuale di impatto del sottoperiodo rispetto al periodo totale"
    )

    def __init__(self):
        # Same rationale as VariationRateIndicator: the wrapped indicator is
        # only known at request time via the "indicator" extra field, so the
        # base class's phenomena/combinator pipeline is bypassed entirely.
        super().__init__(phenomena=[], combinator=self._unused_combinator)

    @staticmethod
    def _unused_combinator(df: pd.DataFrame, **_extra) -> pd.Series:  # pragma: no cover
        raise RuntimeError(
            "PeriodPercentageImpactIndicator.combinator should never be called directly"
        )

    @property
    def years_range(self) -> dict[str, int]:
        """Broadest year range across every combinable indicator, same
        rationale as VariationRateIndicator.years_range."""
        if self._years_range is None:
            min_years, max_years = [], []
            for key in _REGISTRY:
                if key == _PERIOD_PERCENTAGE_IMPACT_KEY:
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
        seasonality = extra.pop("seasonality", None)

        if not indicator_key:
            raise ValueError(
                "'indicator' extra field is required for 'incidenza-periodo'"
            )
        if indicator_key == _PERIOD_PERCENTAGE_IMPACT_KEY:
            raise ValueError("'incidenza-periodo' cannot wrap itself")
        if indicator_key == _VARIATION_RATE_KEY:
            # Not strictly wrong, but tasso-variazione already needs its own
            # extra fields (start/end_date_comparison); reject early with a
            # clear message rather than a confusing missing-field error.
            raise ValueError("'incidenza-periodo' cannot wrap 'tasso-variazione'")
        if not seasonality:
            raise ValueError(
                "'seasonality' extra field is required for 'incidenza-periodo'"
            )

        wrapped = get_indicator(indicator_key)

        total_df = wrapped.get_indicator(
            start_date=start_date,
            end_date=end_date,
            **extra,
        )
        seasonal_df = wrapped.get_indicator(
            start_date=start_date,
            end_date=end_date,
            seasonality=seasonality,
            **extra,
        )

        if total_df is None or seasonal_df is None:
            return None

        merged = (
            total_df[["ID_COMUNE", "INDICE"]]
            .rename(columns={"INDICE": "INDICE_periodo"})
            .merge(
                seasonal_df[["ID_COMUNE", "INDICE"]].rename(
                    columns={"INDICE": "INDICE_sottoperiodo"}
                ),
                on="ID_COMUNE",
                how="outer",
            )
        )
        # a comune with no value in the sub-period (no data) has no incidence: null, not 0%

        merged["INDICE"] = (
            merged["INDICE_sottoperiodo"] / merged["INDICE_periodo"].replace(0, np.nan)
        ) * 100

        return merged[["ID_COMUNE", "INDICE", "INDICE_periodo", "INDICE_sottoperiodo"]]
