from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
    RedistributionPhenomenon,
    SurfacePhenomenon,
    TravelTimeArrivalsPhenomenon,
)
from overtourism.overtourism.backend_extension.api.utils.index_utils_trentino import (
    get_comuni_clusters,
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
TRAVEL_TIME_SOURCE = (
    index_data_path / "phen_tempo_medio_turisti.parquet"
)  # daily arrivals and travel time (minutes) of the tourists per comune and provenienza
CONSUMPTION_SOURCE = (
    index_data_path / "phen_consumption.parquet"
)  # daily electricity / gas consumption per comune (services and total)
EE_TOURISM_SHARE_SOURCE = (
    index_data_path / "phen_ee_tourism_share.parquet"
)  # yearly share of the electricity consumption attributable to tourism per comune (computed in the data preparation)
REDISTRIBUTION_SOURCE = (
    index_data_path / "phen_ridistribuzione.parquet"
)  # daily redistribution results per comune (computed offline: data_preparation/gen_redistribution_dataframes.py)
REDISTRIBUTION_FLOWS_SOURCE = (
    index_data_path / "phen_ridistribuzione_flussi.parquet"
)  # daily optimal flows between the Vodafone nodes (the arrows of the map)
REDISTRIBUTION_NODES_SOURCE = (
    index_data_path / "ridistribuzione_nodi.parquet"
)  # the nodes: WGS84 position, label and comuni
MACRO_AREAS_FILE = index_data_path / "map_comuni_into_apt.json"
CODICI_COMUNI_FILE = index_data_path / "mapping_comuni_ISTAT.json"
SEASONALITY_CLUSTERS_FILE = (
    index_data_path / "mapping_comuni_cluster_stagionalita.json"
)  # seasonality cluster of each comune, computed once in the data preparation (notebook gini_clustering)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Number of classes of the discretized PageRank map (quantiles, 1 = lowest ... N = highest)
PAGERANK_N_CLASSES = 5
MAIN_MARKET_MIN_ARRIVALS = 100
# Travel time of the tourists: a tourist is a "proximity" one when the origin is within this many hours
PROXIMITY_MAX_TRAVEL_TIME = 5
# Step (minutes) of the optional rounding of the average travel time
TRAVEL_TIME_ROUNDING_STEP_MINUTES = 30
MINUTES_PER_HOUR = 60


_VARIATION_RATE_KEY = "tasso-variazione"

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
    "incidenza-alloggi-turistici-strutture": lambda: (
        TouristAccommodationFacilitiesIndicator(STRUCTURES_SOURCE)
    ),
    "incidenza-alloggi-disposizione-strutture": lambda: (
        AvailableAccommodationFacilitiesIndicator(STRUCTURES_SOURCE)
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
    "gini-stagionalita": lambda: TouristPresenceGiniIndicator(
        PRESENCES_SOURCE, SEASONALITY_CLUSTERS_FILE
    ),
    "quota-mobilita-ingresso": lambda: InboundMobilityShareIndicator(FLOWS_OD_SOURCE),
    "quota-mobilita-uscita": lambda: OutboundMobilityShareIndicator(FLOWS_OD_SOURCE),
    "quota-intra-ambito": lambda: IntraAmbitoShareIndicator(FLOWS_OD_SOURCE),
    "squilibrio-ridistribuzione": lambda: RedistributionBalanceIndicator(
        REDISTRIBUTION_SOURCE, REDISTRIBUTION_FLOWS_SOURCE, REDISTRIBUTION_NODES_SOURCE
    ),
    "pagerank-discretizzato": lambda: DiscretizedPageRankIndicator(
        FLOWS_OD_SOURCE, PAGERANK_N_CLASSES
    ),
    "concentrazione-mercati-provenienza": lambda: SourceMarketsConcentrationIndicator(
        PROVENIENZA_SOURCE
    ),
    "dipendenza-mercato-principale": lambda: MainMarketDependenceIndicator(
        PROVENIENZA_SOURCE
    ),
    "impatto-turismo-consumi-elettrici": lambda: (
        TourismElectricityConsumptionImpactIndicator(CONSUMPTION_SOURCE)
    ),
    "impatto-turismo-consumi-gas": lambda: TourismGasConsumptionImpactIndicator(
        CONSUMPTION_SOURCE
    ),
    "quota-consumi-elettrici-attribuibili-turismo": lambda: (
        TourismElectricityShareIndicator(EE_TOURISM_SHARE_SOURCE)
    ),
    "distanza-media-turisti": lambda: AverageTouristTravelTimeIndicator(
        TRAVEL_TIME_SOURCE
    ),
    "dipendenza-domanda-prossimita": lambda: ProximityDemandDependenceIndicator(
        TRAVEL_TIME_SOURCE
    ),
    "sostenibilita-territoriale": lambda: TerritorialSustainabilityIndicator(),
    _VARIATION_RATE_KEY: lambda: VariationRateIndicator(),
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
    description = (
        "L'indice di ricettività definisce il <strong>rapporto</strong> fra i <strong>letti presenti negli esercizi ricettivi</strong> e gli <strong>abitanti</strong> di una stessa area. "
        "L'indice è una misura della capacità turistica rispetto alla dimensione, in termini di popolazione, di un'area. "
        "<strong>Fonti dei dati</strong>: ISPAT (Istituto di statistica della Provincia autonoma di Trento) per la consistenza delle strutture ricettive "
        "(posti letto totali = alberghieri + extra-alberghieri + alloggi turistici privati; gli alloggi a disposizione non sono inclusi nel totale, "
        "che è ricalcolato dalle componenti) e per la popolazione residente al 1° gennaio dell'anno. "
        "L'indice è annuale."
    )
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
    description = (
        "L'indice di turisticità definisce il <strong>rapporto</strong> fra il <strong>numero medio giornaliero di turisti</strong> di una specifica area e il <strong>numero di abitanti</strong> della stessa area. "
        "L'indice fornisce una misura dell'effettiva incidenza del turismo rispetto alla dimensione, in termini di popolazione, di un'area. "
        "<strong>Fonti dei dati</strong>: Vodafone (analisi dei dati della rete di telefonia mobile) per le presenze giornaliere di turisti, cioè le persone che pernottano; "
        "ISPAT per la popolazione residente al 1° gennaio. "
        "Le presenze Vodafone, fornite per aree che possono comprendere più comuni, sono attribuite ai comuni in proporzione ai posti letto (fonte ISPAT) dell'anno."
    )
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
    description = (
        "L'indice di densità turistica definisce il <strong>rapporto</strong> fra il <strong>numero medio giornaliero di turisti ed escursionisti</strong> presenti in un comune e la sua <strong>superficie</strong> in km². "
        "L'indice fornisce una misura della pressione turistica sul territorio. "
        "<strong>Fonti dei dati</strong>: Vodafone (dati della rete di telefonia mobile) per le presenze giornaliere di turisti (persone che pernottano) ed escursionisti (visitatori giornalieri); "
        "confini amministrativi ISTAT per la superficie dei comuni. "
        "Per le aree Vodafone che comprendono più comuni, i turisti sono attribuiti ai comuni in proporzione ai posti letto (ISPAT) e gli escursionisti in proporzione alla popolazione residente. "
        "Se manca uno dei due profili è considerato pari a zero; se mancano entrambi il comune non ha valore."
    )
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
    description = (
        "L'indice di densità antropica misura il rapporto fra la <strong>somma di popolazione residente e numero medio giornaliero di turisti ed escursionisti</strong> "
        "rilevati da Vodafone e la <strong>superficie</strong> del comune in km². Misura quindi quante persone, residenti e presenti, insistono su ogni km² di territorio. "
        "<strong>Fonti dei dati</strong>: ISPAT per la popolazione residente al 1° gennaio; Vodafone (dati della rete di telefonia mobile) per le presenze di turisti ed escursionisti; "
        "confini amministrativi ISTAT per la superficie. "
        "Per le aree Vodafone con più comuni, i turisti sono attribuiti in proporzione ai posti letto (ISPAT) e gli escursionisti alla popolazione residente."
    )
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
        "extra-alberghiere (numero di strutture, non di posti letto). Valori alti indicano un'offerta ricettiva "
        "orientata agli alloggi turistici privati rispetto alle strutture tradizionali. "
        "<strong>Fonte dei dati</strong>: ISPAT, statistiche sulla consistenza delle strutture ricettive (annuario del turismo). "
        "L'indice è annuale; le aree senza strutture non hanno valore."
    )
    numerator_column = "strutture_alloggi_turistici"
    component_columns = (
        "strutture_alloggi_turistici",
        "strutture_alb",
        "strutture_exalb",
    )


class TouristPresenceGiniIndicator(Indicator):
    """
    Normalized Gini of the temporal distribution of daily tourist presences.

    Next to INDICE, ``CLUSTER_COLUMN`` holds the seasonality cluster of the comune (an integer code) and
    ``CLUSTER_TYPE_COLUMN`` its name (e.g. "sci"), read from a comune -> cluster mapping computed once in the data
    preparation on the whole period (data_download/seasonality_clusters.py): they are the same for any requested
    window.
    """

    name = "Indice di stagionalità delle presenze"
    category = "Pressione turistica"
    order = 10
    year_only = True

    _PRESENCE_COLUMN = "presenze_vodafone"
    _LOW_COVERAGE_VALUE = np.nan
    _MIN_COVERAGE = 0.8
    CLUSTER_COLUMN = "cluster_stagionalita"
    CLUSTER_TYPE_COLUMN = "tipo_cluster"
    # macro area / province: INDICE = mean Gini of the comuni, cluster = the most frequent one. The codes follow the
    # alphabetical order of the names, so on a tie both columns take the same cluster (the smallest)
    column_aggs = {CLUSTER_COLUMN: "mode", CLUSTER_TYPE_COLUMN: "mode"}

    description = (
        "L'indice di Gini quantifica la <strong>concentrazione temporale delle presenze turistiche giornaliere</strong> nel periodo selezionato. "
        "Il valore 0 indica una distribuzione uniforme, il valore 1 la concentrazione in un unico giorno. \n"
        "<strong>Fonte dei dati</strong>: presenze giornaliere di turisti (persone che pernottano) rilevate da Vodafone con i dati della rete di telefonia mobile. "
        "L'indice è calcolato sulla serie giornaliera dei comuni, ordinata, ed è normalizzato tra 0 e 1. "
        "Il comune non ha valore se i giorni con dato coprono meno dell'80% del periodo o se i giorni disponibili sono meno di due. "
        "Per le aree Vodafone con più comuni, le presenze sono attribuite ai comuni in proporzione ai posti letto (ISPAT). "
        "Per macro aree e province il valore è la media degli indici dei comuni. \n"
        "Per ogni comune è indicato anche il <strong>cluster stagionale</strong> (codice e tipologia): "
        "annuale, sci, estate + spalle, estate alpina. Il cluster è calcolato una sola volta sul profilo mensile dei flussi Vodafone "
        "dell'intero periodo disponibile, quindi non dipende dal periodo selezionato."
    )
    index_value_unit_description = (
        "Indice di Gini normalizzato (0 = uniforme, 1 = concentrato)"
    )
    availableForVariation = False

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


class AvailableAccommodationFacilitiesIndicator(_HospitalityFacilitiesShareIndicator):
    category = "Strutture"
    order = 30
    """Available accommodation facilities over all accommodation facilities."""

    name = "Incidenza Alloggi a disposizione (strutture)"
    description = (
        "Misura la quota delle <strong>strutture degli alloggi a disposizione</strong> "
        "sul totale composto da alloggi turistici, alloggi a disposizione, "
        "strutture alberghiere ed extra-alberghiere (numero di strutture, non di posti letto). "
        "Gli alloggi a disposizione (ad esempio le seconde case) non fanno parte dell'offerta ricettiva ufficiale: "
        "valori alti indicano un territorio con una forte presenza di abitazioni non stabilmente occupate. "
        "<strong>Fonte dei dati</strong>: ISPAT, statistiche sulla consistenza delle strutture ricettive (annuario del turismo). "
        "L'indice è annuale; le aree senza strutture non hanno valore."
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
        "extra-alberghiere (posti letto, non numero di strutture). Valori alti indicano una capacità ricettiva "
        "concentrata negli alloggi turistici privati rispetto alle strutture tradizionali. "
        "<strong>Fonte dei dati</strong>: ISPAT, statistiche sulla consistenza delle strutture ricettive (annuario del turismo). "
        "L'indice è annuale; le aree senza posti letto non hanno valore."
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
        "strutture alberghiere ed extra-alberghiere (posti letto, non numero di strutture). "
        "Gli alloggi a disposizione (ad esempio le seconde case) non fanno parte dell'offerta ricettiva ufficiale: "
        "valori alti indicano un territorio con molti posti letto in abitazioni non stabilmente occupate. "
        "<strong>Fonte dei dati</strong>: ISPAT, statistiche sulla consistenza delle strutture ricettive (annuario del turismo). "
        "L'indice è annuale; le aree senza posti letto non hanno valore."
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
    description = (
        "L'indice misura il <strong>rapporto</strong> fra le <strong>presenze di turisti misurate</strong> attraverso l'analisi di dati da rete di telefonia mobile e le <strong>presenze ufficiali</strong> di turisti in strutture alberghiere e extra-alberghiere. "
        "Valori superiori a 1 indicano la presenza di turismo non registrato dalle statistiche ufficiali (ad esempio seconde case, alloggi privati, ospiti di amici e parenti). "
        "<strong>Fonti dei dati</strong>: Vodafone per le presenze di turisti misurate; ISPAT per le presenze ufficiali (movimento turistico) di alberghi ed extra-alberghieri. "
        "Le presenze ISPAT sono mensili: sono distribuite sui giorni con un modello basato sulla durata media del soggiorno e poi attribuite ai comuni in proporzione ai posti letto "
        "del tipo di struttura. Dove le presenze ufficiali sono nulle, l'indice coincide con le presenze Vodafone."
    )
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
        "L'indice misura il <strong>rapporto fra le presenze complessive</strong> rilevate "
        "attraverso i dati Vodafone, comprendenti turisti ed escursionisti, "
        "e la <strong>popolazione residente</strong>. Indica quante persone presenti in media ogni giorno, oltre ai residenti, "
        "gravano su ogni abitante. "
        "<strong>Fonti dei dati</strong>: Vodafone (dati della rete di telefonia mobile) per le presenze giornaliere di turisti ed escursionisti; "
        "ISPAT per la popolazione residente al 1° gennaio. "
        "Per le aree Vodafone con più comuni, i turisti sono attribuiti in proporzione ai posti letto (ISPAT) e gli escursionisti alla popolazione residente."
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
# The values are yearly: a requested window returns the value of its year(s) (mean over the years of the window).

_FLOW_PROFILE_SUFFIXES = ("combined", "tourist", "visitor")
_FLOW_COMMON_DESCRIPTION = (
    " Turisti ed escursionisti sono <strong>aggregati con un peso per ciascun profilo</strong>, "
    "stimato dai dati (una giornata-utente di escursionista non pesa come una di turista); "
    "i valori per singolo profilo sono riportati a fianco (turisti peso 1, escursionisti peso 0,5: coefficienti scelti, non stimati). "
    "<strong>Fonte dei dati</strong>: dati Vodafone di co-visitazione fra aree (flussi origine-destinazione di utenti della rete di telefonia mobile "
    "presenti in due aree nello stesso periodo), considerando solo le righe di totale, che Vodafone fornisce al netto delle soglie di privacy (conteggi inferiori a circa 15 utenti sono azzerati). "
    "L'indice è annuale e calcolato sulle giornate-utente dell'anno per ciascuna area Vodafone; "
    "le aree che comprendono più comuni passano il proprio valore a tutti i loro comuni. "
    "Le aree con meno di 100 giornate-utente annue di presenza sono lasciate senza valore, dove indicato."
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
        "indicano località di destinazione o di pernottamento stabile. Il valore può superare 1 perché un utente può essere contato in più aree; "
        "è lasciato vuoto per le aree con meno di 100 giornate-utente annue di presenza."
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
        "utenti presenti). Valori alti indicano località da cui i visitatori ripartono per visitare altre aree. "
        "Il valore può superare 1 perché un utente può essere contato in più aree; "
        "è lasciato vuoto per le aree con meno di 100 giornate-utente annue di presenza."
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
        "Valori alti indicano un'area i cui flussi restano all'interno del proprio ambito. "
        "Per i comuni con più aree Vodafone il valore è la media pesata sui flussi intra-ambito; "
        "è vuoto per le aree senza flussi verso altre aree o con ambito sconosciuto."
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
        "{n} = più alta; 0 = dato non disponibile): le classi indicano la posizione relativa fra i comuni. "
        "Il PageRank (fattore di smorzamento 0,85) è calcolato sulla rete orientata dei flussi pesati fra aree Vodafone; "
        "hanno valore solo le aree che fanno parte della rete, cioè con almeno un flusso in entrata o in uscita."
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


class RedistributionBalanceIndicator(Indicator):
    category = "Flussi"
    order = 100
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

    name = "Squilibrio di carico e ridistribuzione dei flussi"
    description = (
        "L'indice confronta il <strong>carico di ogni area</strong> (turisti ed escursionisti rilevati da Vodafone) "
        "con il suo <strong>tetto di carico</strong>, stimato dallo storico delle aree comparabili, e lo mette a "
        "confronto con una <strong>distribuzione bilanciata</strong>, in cui tutte le aree hanno la stessa "
        "saturazione (carico / tetto). Valori <strong>positivi</strong> indicano un'area <strong>sotto-rappresentata</strong>, "
        "con margine per accogliere più visitatori; valori <strong>negativi</strong> un'area <strong>sovraccarica</strong>. "
        "È espresso in punti di saturazione ed è la <strong>media giornaliera del periodo selezionato</strong>. "
        "Le <strong>frecce</strong> sulla mappa mostrano gli spostamenti a costo minimo con cui l'eccesso potrebbe essere "
        "riassorbito entro circa 40 km (il costo cresce con la distanza e con la diversità di tipo fra le aree): "
        "la parte che non trova posto entro il raggio resta «non assorbibile». "
        "Non è un costo in euro. Il tetto è una stima incerta (intervallo al 90%: ×1,5 per i turisti, ×1,9 per gli "
        "escursionisti) e «bilanciato = stessa saturazione» è una scelta normativa. Per le aree Vodafone che "
        "comprendono più comuni il valore è quello dell'area. "
        "<strong>Fonti dei dati</strong>: Vodafone (presenze giornaliere di turisti ed escursionisti e co-visitazioni fra aree, usate per stimare "
        "il decadimento con la distanza e l'affinità fra aree); ISPAT (posti letto e popolazione residente); ISTAT (confini e superficie dei comuni, "
        "tipologia turistica dei comuni). Il tetto di carico è stimato con una regressione quantile (quantile 0,9) sui picchi annuali di "
        "carico delle aree comparabili, in funzione di posti letto, residenti, seconde case o superficie e profilo stagionale. "
        "Nella versione combinata gli escursionisti pesano 0,5 rispetto ai turisti. "
        "Il calcolo è eseguito una volta nella preparazione dei dati; le colonne affiancate indicano le persone al giorno da attrarre (+) o da "
        "spostare (−), gli spostamenti netti e l'eccesso non assorbibile entro il raggio."
    )
    index_value_unit_description = (
        "Punti di saturazione (positivo = sotto-rappresentato, negativo = sovraccarico)"
    )
    availableForVariation = False
    year_only = True

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


# ---------------------------------------------------------------------------
# Provenienza (source markets of the tourists)
# ---------------------------------------------------------------------------
# Reads `phen_provenienza`: one row per comune and YEAR with `hhi_provenienza`, the normalized
# Herfindahl-Hirschman index of the tourist presences by country of origin (computed in the data preparation,
# `territorial_indexes/utils/provenienza.py`). Yearly like the OD flow indicators: a requested window returns the
# value of its year(s).


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
        "ed è calcolato sui dati Vodafone; è lasciato vuoto per le aree con presenze troppo basse (meno di 100 giornate-utente annue). "
        "<strong>Fonte dei dati</strong>: presenze giornaliere di turisti per paese di provenienza rilevate da Vodafone con i dati della rete di telefonia mobile, "
        "sommate sull'anno per area. L'indice non è additivo, quindi non viene ripartito: le aree che comprendono più comuni passano il proprio valore a tutti i loro comuni, "
        "mentre un comune con più aree ha la media pesata sulle giornate-utente. "
        "Le presenze contano ogni giorno di soggiorno, quindi i mercati con soggiorni lunghi pesano di più. "
        "I conteggi molto piccoli (sotto circa 15 utenti) non sono forniti da Vodafone: l'indice descrive i mercati osservati."
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
        "per i comuni con arrivi troppo bassi (meno di 100 arrivi annui). "
        "<strong>Fonte dei dati</strong>: arrivi giornalieri di turisti per paese di provenienza rilevati da Vodafone con i dati della rete di telefonia mobile, "
        "sommati sull'anno; per le aree Vodafone che comprendono più comuni gli arrivi sono attribuiti ai comuni in proporzione ai posti letto (ISPAT). "
        "Il mercato principale è calcolato sull'insieme dei paesi di provenienza (l'Italia è un mercato) e sono riportati i principali mercati di ciascun comune. "
        "I conteggi molto piccoli (sotto circa 15 utenti) non sono forniti da Vodafone: l'indice descrive "
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
        "sono considerati pari a zero. "
        "<strong>Fonte dei dati</strong>: prelievi mensili di energia elettrica per comune (kWh), per il settore servizi (categorie ATECO legate al turismo: "
        "49.3, 55.1, 56.1, 93.1, 96) e per il totale di tutti gli usi. I consumi mensili sono distribuiti uniformemente sui giorni del mese; "
        "l'indice è il rapporto fra le medie del periodo selezionato. È una misura indiretta: i consumi dei servizi non sono solo turistici."
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
        "e quindi al turismo. I comuni senza consumi totali non hanno valore (ad esempio i comuni privi di rete del gas); se mancano solo i consumi del settore servizi "
        "sono considerati pari a zero. "
        "<strong>Fonte dei dati</strong>: prelievi mensili di gas per comune (Smc), per il settore servizi (categorie ATECO legate al turismo: "
        "49.3, 55.1, 56.1, 93.1, 96) e per il totale di tutti gli usi. I consumi mensili sono distribuiti uniformemente sui giorni del mese; "
        "l'indice è il rapporto fra le medie del periodo selezionato. È una misura indiretta: i consumi dei servizi non sono solo turistici."
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
# window returns the value of its year(s). The comuni that cannot
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
        "sono indipendenti. I consumi sono mensili e distribuiti uniformemente sui giorni. "
        "<strong>Fonti dei dati</strong>: prelievi mensili di energia elettrica del settore servizi (categorie ATECO legate al turismo); "
        "presenze giornaliere di turisti Vodafone (i turisti sono il regressore; gli escursionisti hanno peso zero); "
        "temperature giornaliere della stazione Meteotrentino più vicina a ciascun comune, da cui sono ricavati i gradi giorno di riscaldamento (base 18 °C). "
        "La stima usa i dati dell'anno e la pendenza è attratta verso quella provinciale dello stesso anno (pooling parziale). "
        "Affidabilità: «alta» se il limite inferiore dell'intervallo al 95% dell'effetto è positivo, «media» se lo è solo quello all'80%, «bassa» altrimenti "
        "(la quota è allora determinata soprattutto dalla stima provinciale); una quota pari o superiore a 0,9 è sempre «bassa»."
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


class AverageTouristTravelTimeIndicator(Indicator):
    """
    Mean travel time (HOURS) of the tourists to reach the comune, weighted by the arrivals.

    Source: ``phen_tempo_medio_turisti``, one row per (day, comune, provenienza): ``tempo_di_percorrenza`` is the time
    that each tourist of that origin takes, so the comune value is sum(arrivi * tempo) / sum(arrivi) over the rows of
    the window. Both sums are daily phenomena, so a window of many days is a ratio of sums (arrival-weighted), not a
    mean of daily means, and the region value of the time series is weighted the same way.

    The source is in minutes; INDICE is converted to hours (270 min -> 4.5).

    Extra field ``arrotonda_30_minuti`` (default false): the final average is rounded to the nearest multiple of 30
    minutes (half up), i.e. to 0.5 h steps, before the conversion to hours. The tourists' ``arrivals`` stay next to
    INDICE.
    """

    category = "Provenienza"
    order = 100

    name = "Distanza media turisti"
    description = (
        "L'indice stima il <strong>tempo medio impiegato dai turisti per raggiungere la località</strong>. "
        "Per ogni paese o area di provenienza il dato riporta il tempo di percorrenza dei turisti verso il comune: "
        "i tempi sono combinati con una <strong>media pesata sugli arrivi</strong>, così le provenienze con più "
        "arrivi contano di più. Se il periodo selezionato comprende più giorni, la media è pesata su tutti gli "
        "arrivi del periodo. Il valore è espresso in <strong>ore</strong> (es. 4,5 = 4 ore e 30 minuti) e può essere "
        "opzionalmente <strong>arrotondato a multipli di mezz'ora</strong>. "
        "Sono considerati solo i turisti, e solo le provenienze con arrivi e tempo di percorrenza noti. "
        "<strong>Fonti dei dati</strong>: arrivi giornalieri di turisti per paese di provenienza rilevati da Vodafone con i dati della rete di telefonia mobile "
        "(attribuiti ai comuni in proporzione ai posti letto ISPAT per le aree con più comuni); matrice dei tempi di percorrenza verso i comuni trentini, "
        "calcolata con OpenStreetMap/OSRM (percorso in auto più veloce, senza traffico, verso il capoluogo di ciascun comune). "
        "È una <strong>stima</strong> e non un viaggio misurato: ogni arrivo italiano è rappresentato da un'unica città di riferimento, ogni arrivo estero dalla capitale del paese "
        "(in auto se a meno di 1000 km da Trento, altrimenti volo fino all'aeroporto di Bergamo e poi auto; il tempo di volo è stimato dalla distanza a velocità costante, "
        "senza attese aeroportuali). Gli arrivi senza paese di origine non sono considerati."
    )
    index_value_unit_description = "Ore (media pesata sugli arrivi)"
    extraFields = ["arrotonda_30_minuti"]

    ARRIVALS_COLUMN = "tourist_arrivals"
    MINUTES_COLUMN = (
        "arrival_minutes"  # sum of arrivals * minutes: helper, not returned
    )

    def __init__(self, source_file):
        super().__init__(
            phenomena=[
                TravelTimeArrivalsPhenomenon(
                    source_file, name=self.ARRIVALS_COLUMN, measure="arrivals"
                ),
                TravelTimeArrivalsPhenomenon(
                    source_file, name=self.MINUTES_COLUMN, measure="arrival_minutes"
                ),
            ],
            combinator=self._average_travel_time,
        )

    @staticmethod
    def _round_to_step(minutes: pd.Series, step: float) -> pd.Series:
        # half up (numpy's round goes to the even one: 45 -> 60 but 15 -> 0); missing values stay missing
        return np.floor(minutes / step + 0.5) * step

    def _average_travel_time(
        self,
        df: pd.DataFrame,
        *,
        arrotonda_30_minuti=None,
        **_extra,
    ) -> pd.Series:
        average_minutes = df[self.MINUTES_COLUMN].div(
            df[self.ARRIVALS_COLUMN].replace(0, np.nan)
        )
        if _parse_bool_flag(arrotonda_30_minuti, "arrotonda_30_minuti"):
            average_minutes = self._round_to_step(
                average_minutes, TRAVEL_TIME_ROUNDING_STEP_MINUTES
            )
        return average_minutes / MINUTES_PER_HOUR

    def _compute_indicator(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        **extra,
    ) -> pd.DataFrame | None:
        result = super()._compute_indicator(start_date, end_date, **extra)
        if result is None:
            return None
        return result.drop(columns=[self.MINUTES_COLUMN])


class ProximityDemandDependenceIndicator(Indicator):
    """
    Share (0-1) of the tourist arrivals that come from origins within ``PROXIMITY_MAX_TRAVEL_TIME``.

    Same source as AverageTouristTravelTimeIndicator: arrivals of the origins whose travel time is <= the threshold
    over all the arrivals, both summed over the window. A comune whose tourists all come from farther away has 0;
    one without arrivals has no value (null). Both arrival counts stay next to INDICE.
    """

    category = "Provenienza"
    order = 100

    name = "Indice di dipendenza dalla domanda di prossimità"
    description = (
        f"Misura la <strong>quota di turisti che proviene da destinazioni raggiungibili entro {PROXIMITY_MAX_TRAVEL_TIME} ore</strong> "
        f"di viaggio: il rapporto fra gli arrivi dalle provenienze con tempo di percorrenza fino a {PROXIMITY_MAX_TRAVEL_TIME} ore "
        "e gli arrivi totali. Valori alti indicano una forte dipendenza dalla domanda di prossimità. "
        "Sono considerati solo i turisti, e solo le provenienze con arrivi e tempo di percorrenza noti. "
        "<strong>Fonti dei dati</strong>: arrivi giornalieri di turisti per paese di provenienza rilevati da Vodafone con i dati della rete di telefonia mobile; "
        "tempi di percorrenza stimati con OpenStreetMap/OSRM dalla città di riferimento (Italia) o dalla capitale del paese (estero), "
        "in auto o, oltre 1000 km, in aereo fino a Bergamo più auto. La quota è calcolata sugli arrivi dell'intero periodo selezionato; "
        "i comuni senza arrivi non hanno valore."
    )
    index_value_unit_description = (
        f"Quota di turisti provenienti entro {PROXIMITY_MAX_TRAVEL_TIME} ore (0-1)"
    )

    def __init__(self, source_file):
        super().__init__(
            phenomena=[
                TravelTimeArrivalsPhenomenon(
                    source_file,
                    name="arrivals_within",
                    measure="arrivals_within",
                    max_minutes=PROXIMITY_MAX_TRAVEL_TIME * MINUTES_PER_HOUR,
                ),
                TravelTimeArrivalsPhenomenon(
                    source_file, name="tourist_arrivals", measure="arrivals"
                ),
            ],
            combinator=self.divide("arrivals_within", "tourist_arrivals"),
        )


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

    category = "Pressione turistica"
    order = 10

    name = "Indice di sostenibilità territoriale"
    index_value_unit_description = "Punteggio 0-100 (100 = massima sostenibilità)"

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
        total_pillars = sum(p.weight for p in SUSTAINABILITY_PILLARS)
        parts = [
            "L'indice di sostenibilità territoriale è un <strong>punteggio da 0 a 100</strong> "
            "(100 = massima sostenibilità) che sintetizza la <strong>pressione del turismo</strong> su un comune "
            "in un periodo, usando solo indicatori calcolabili per qualsiasi intervallo di giorni (sono esclusi gli "
            "indicatori annuali). Si ispira ai <strong>criteri GSTC per le destinazioni (GSTC-D v2.0)</strong> e "
            "alle <strong>aree tematiche degli osservatori INSTO di UN Tourism</strong>. \n"
            "<strong>Come si calcola.</strong> Ogni componente è un indicatore già disponibile, trasformato in un "
            "punteggio da 0 a 100 con una scala lineare fra un valore «sostenibile» (100) e un valore «critico» (0). "
            "I punteggi sono mediati, con i pesi indicati, in due dimensioni, e le dimensioni sono mediate "
            "nell'indice: ogni tema pesa un terzo. "
        ]
        for pillar in SUSTAINABILITY_PILLARS:
            members = [c for c in SUSTAINABILITY_COMPONENTS if c.pillar == pillar.key]
            members_total = sum(c.weight for c in members)
            parts.append(
                f"\n<strong>{pillar.label}</strong> (peso {pillar.weight / total_pillars:.0%}; "
                f"{pillar.framework}): "
            )
            for c in members:
                if c.best < c.worst:
                    rule = (
                        f"100 se ≤ {_it_number(c.best)}, 0 se ≥ {_it_number(c.worst)}"
                    )
                else:  # higher is better
                    rule = (
                        f"100 se ≥ {_it_number(c.best)}, 0 se ≤ {_it_number(c.worst)}"
                    )
                parts.append(
                    f"\n• {c.label} (peso {c.weight / members_total:.0%} nella dimensione): "
                    f"{rule} ({c.unit}). Rif.: {c.framework}. "
                )
        parts.append(
            f"\n<strong>Dati mancanti.</strong> Un componente senza valore (ad esempio i consumi elettrici di un "
            f"comune senza dati) è escluso e i pesi degli altri sono ricalcolati; se i componenti disponibili pesano meno del "
            f"{SUSTAINABILITY_MIN_COVERAGE:.0%} del totale l'indice non è calcolato. Accanto all'indice sono riportati "
            "i punteggi di ogni componente e dimensione, la quota di dati disponibili e il componente con il punteggio "
            "più basso. \n"
            "<strong>Limiti.</strong> Sono misurati solo tre temi (pressione sui residenti, energia, trasporti e "
            "clima), uno per ciascuno: indicatori molto simili fra loro non sono sommati per non contare più volte lo "
            "stesso fenomeno. Le sezioni GSTC sulla gestione (A) e sulla cultura (C), i benefici economici locali e "
            "la pressione sugli spazi non sono valutati. GSTC e UN Tourism non definiscono soglie numeriche: i valori "
            "di riferimento e i pesi sono scelte di prima impostazione, da validare con i portatori di interesse. "
            "Il punteggio descrive la pressione del turismo e non è una valutazione di conformità GSTC."
        )
        return "".join(parts)

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
    ``"turisticita"``), together with the dates of the comparison period.
    This keeps a single
    "tasso-variazione" entry in the registry that works for every
    combinable indicator.

    Because the wrapped indicator is only known once a request comes in,
    this class doesn't use the phenomena/combinator pipeline from the base
    class at all — ``_compute_indicator`` is fully overridden below, and
    ``phenomena``/``combinator`` are left empty/no-op.
    """

    name = "Tasso di variazione"
    description = "Il tasso di variazione misura la <strong>variazione percentuale</strong> di un indicatore fra il <strong>periodo selezionato</strong> ed un <strong>periodo storicizzato di confronto</strong>. È utile per confrontare due intervalli temporali (es. due stagioni, due anni) sullo stesso indice. Il valore è (periodo selezionato − periodo di confronto) / periodo di confronto × 100; non ha valore dove l'indice del periodo di confronto è zero o mancante. Le fonti dei dati sono quelle dell'indicatore confrontato."
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
