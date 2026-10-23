# SPDX-License-Identifier: Apache-2.0
"""Single source of truth about the datasets of the pipeline.

Everything that used to be repeated (dataset keys, value columns used for merging, which reference mappings
each dataset needs, which datasets feed which phenomenon) is defined here once.

NO FILE NAME is written here (nor anywhere else in the code): raw files, folders, file-name patterns and the
names of the standardized / processed / update files are all in config/settings.yaml (sections `files`,
`naming`, `datasets`, `references`) and are loaded by utils/config.py.
"""

from dataclasses import dataclass

from utils.config import (
    BASE_WINDOW,
    DATASET_FILES,
    DERIVED_PHENOMENA_FILES,
    PHENOMENA_FILES,
    TOP_N_PROVENIENZA_COUNTRIES,
)

# value columns of the processed dataframes
POPOLAZIONE_VALUE_COLS = ["popolazione"]
STRUTTURE_VALUE_COLS = [
    "letti_alb",
    "letti_exalb",
    "letti_alloggi_turistici",
    "letti_alloggi_disp",
    "strutture_alb",
    "strutture_exalb",
    "strutture_alloggi_turistici",
    "strutture_alloggi_disp",
    "tot_strutture",
    "tot_postiletto",
]
VODAFONE_VALUE_COLS = ["presenze", "presenze_escursionisti_vodafone"]
PRESENZE_ALB_VALUE_COLS = ["presenze_alb"]
PRESENZE_XALB_VALUE_COLS = ["presenze_xalb"]
EE_PRELIEVI_VALUE_COLS = ["ee_services"]
GAS_PRELIEVI_VALUE_COLS = ["gas_services"]
# "totale" files: withdrawals of ALL the usage categories (not only the services ones above)
EE_TOTALE_VALUE_COLS = ["ee_total"]
GAS_TOTALE_VALUE_COLS = ["gas_total"]
# Weather (`phen_meteo`, comune x day): final columns after DATA, ID_COMUNE. The base temperature of the heating degree
# days is `meteo.hdd_base_temp` of settings.yaml.
METEO_VALUE_COLS = ["t_min", "t_max", "t_mean", "hdd", "rain_mm"]
# Final column order of `phen_consumption` (after DATA, ID_COMUNE)
CONSUMPTION_VALUE_COLS = ["ee_services", "ee_total", "gas_services", "gas_total"]
# The raw files of the energy totals are in settings.yaml (`files.energy.totals`)

# Provenienza: concentration (normalized HHI) of the source markets of the TOURIST presences, see utils/provenienza.py
PROVENIENZA_VALUE_COLS = ["hhi_provenienza", "giorni_coperti"]
PROVENIENZA_PROFILE = "TOURIST"  # only tourists
PROVENIENZA_LOC_TYPE = "TN_MKT_AL_3"  # area level (same as the vodafone presences)
PROVENIENZA_MIN_PRESENZE = 100  # minimum yearly user-days of an area to compute the index (else empty)
PROVENIENZA_FULL_YEAR_DAYS = 360  # fewer days with data: the year is partial (warning, `giorni_coperti`)

# Top source countries, see utils/main_market.py. Source: Vodafone ARRIVALS (raw file in settings.yaml, `files.vodafone_arrivi`),
# same layout as the presences but `value` counts arrivals (a person once, however long the stay).
# Same profile / area level as the HHI. Final columns of `phen_provenienza` after the HHI ones:
ARRIVI_VALUE_COL = "arrivi"  # name of `value` in the standardized dataset (the presences one is `presenze`)
ARRIVI_VALUE_COLS = ["volume_totale"] + [
    col
    for rank in range(1, TOP_N_PROVENIENZA_COUNTRIES + 1)
    for col in (f"volume_mercato_{rank}", f"nome_mercato_{rank}")
]
ARRIVI_FULL_YEAR_DAYS = 360  # fewer days with data: the year is partial (warning)

# Vodafone OD co-visits (flows between areas) -> yearly flow indicators per comune, see utils/flows.py.
# One column per metric and profile (`<metric>_<profile in lower case>`), plus the days covered by the year
# Raw file of the base build: settings.yaml, `files.vodafone_covisits`
COVISITS_LOC_TYPE = "TN_MKT_AL_3"  # area level kept (same as the vodafone presences)
COVISITS_PROFILES = (
    "TOURIST",
    "VISITOR",
)  # kept separate, the other profiles are discarded
# Only the TOTAL rows are used (no userCountry, no userCluster): see utils/steps/standardize_raw_data.py and METHODOLOGY.md.
# There is deliberately no switch for this: the country rows are an anonymized breakdown of the total rows, so keeping
# both would double count, and the total rows are the only reliable figure.
COVISITS_METRICS = ("quota_ingresso", "quota_uscita", "quota_intra_ambito", "pagerank")
COVISITS_MIN_T = (
    100  # minimum yearly user-days in an area to compute the mobility shares
)
COVISITS_PAGERANK_DAMPING = 0.85
COVISITS_FULL_YEAR_DAYS = (
    360  # fewer days with data: the year is partial (warning, `giorni_coperti`)
)
# Weighted union of the two profiles (a tourist-day and a visitor-day do not weigh the same): columns
# `<metric>_combined`, computed on user-days multiplied by the weight of their profile (see utils/flows.py).
# Simple fixed coefficient: a visitor-day (excursionist, no overnight stay) counts half of a tourist-day.
COVISITS_COMBINED = "combined"
COVISITS_VISITOR_COEFFICIENT = 0.5
COVISITS_PROFILE_WEIGHTS = {"TOURIST": 1.0, "VISITOR": COVISITS_VISITOR_COEFFICIENT}
COVISITS_VALUE_COLS = [
    f"{metric}_{profile.lower()}"
    for metric in COVISITS_METRICS
    for profile in (*COVISITS_PROFILES, COVISITS_COMBINED)
] + ["giorni_coperti"]

# Share of the electricity consumption attributable to tourism (`phen_ee_tourism_share`, comune x YEAR), see
# utils/consumption_share.py. `affidabilita` is text: alta / media / bassa.
EE_TOURISM_SHARE_VALUE_COLS = ["quota_ee_turismo", "presenze_medie", "consumo_medio", "affidabilita"]

# ISPAT presences (alb / extralb): arrivals + presences per month, read from the raw files (no normalization).
# File names (folder, pattern, measures, tags): settings.yaml, `files.presenze`. The base build downloads the years of its
# window (period.start .. period.base_end), the updates the others.
PRESENZE_BASE_YEARS = tuple(
    range(BASE_WINDOW.start.year, BASE_WINDOW.end.year + 1)
)  # 2022..2024
# True: a stay that starts at the end of a month goes on in the next one (uses the previous month's arrivals and stay);
# False: every month is distributed using only its own arrivals and stay
PRESENZE_SPILL_OVER = True
# Weather files: one per year; the base build reads the years of its window
METEO_BASE_YEARS = PRESENZE_BASE_YEARS

# reference files that a dataset may need to be standardized / processed
REF_MAPPING_COMUNI = "mapping_comuni"
REF_MAPPING_VODAFONE = "mapping_vodafone"
REF_MAPPING_APT = "mapping_apt"
REF_GEOJSON = "geojson"


@dataclass(frozen=True)
class DatasetSpec:
    name: str  # key used for selection, e.g. "presenze_alb"
    value_cols: tuple  # columns merged on (DATA, ID_COMUNE) when updating
    references: tuple  # reference files needed to process it
    # names of the files of the dataset, all from settings.yaml (`datasets.<name>`)
    std_name: str = ""  # standardized dataframe name (step 1); "" = not normalized, processed from raw_data
    processed_name: str = ""  # processed dataframe name (step 2)
    update_name: str = ""  # processed dataframe name in the yearly update

    @property
    def normalized(self) -> bool:
        return bool(self.std_name)


def _spec(name, value_cols, references):
    """Every dataset must have its file names in settings.yaml (`datasets.<name>`, checked by load_settings):
    there are no defaults here. `std: ""` means "not normalized" (processed from raw_data or derived)."""
    if name not in DATASET_FILES:
        raise KeyError(f"settings.yaml: datasets.{name} (file names) is missing")
    files = DATASET_FILES[name]
    return DatasetSpec(
        name=name,
        value_cols=tuple(value_cols),
        references=tuple(references),
        std_name=files["std"],
        processed_name=files["processed"],
        update_name=files["update"],
    )


DATASETS = {
    spec.name: spec
    for spec in (
        _spec("popolazione", POPOLAZIONE_VALUE_COLS, [REF_MAPPING_COMUNI]),
        _spec("strutture", STRUTTURE_VALUE_COLS, [REF_MAPPING_COMUNI]),
        _spec("vodafone", VODAFONE_VALUE_COLS, [REF_MAPPING_VODAFONE, REF_GEOJSON]),
        # derived from the vodafone presences (same raw files): the base build reads the `vodafone` standardized one
        # (`std: ""` in settings.yaml)
        _spec(
            "vodafone_provenienza",
            PROVENIENZA_VALUE_COLS,
            [REF_MAPPING_VODAFONE, REF_GEOJSON],
        ),
        # Vodafone arrivals -> top source countries (normalized like the presences, value column `arrivi`)
        _spec("vodafone_arrivi", ARRIVI_VALUE_COLS, [REF_MAPPING_VODAFONE, REF_GEOJSON]),
        _spec("ee_prelievi", EE_PRELIEVI_VALUE_COLS, [REF_MAPPING_COMUNI]),
        _spec("gas_prelievi", GAS_PRELIEVI_VALUE_COLS, [REF_MAPPING_COMUNI]),
        _spec("ee_totale", EE_TOTALE_VALUE_COLS, [REF_MAPPING_COMUNI]),
        _spec("gas_totale", GAS_TOTALE_VALUE_COLS, [REF_MAPPING_COMUNI]),
        _spec(
            "vodafone_covisits",
            COVISITS_VALUE_COLS,
            [REF_MAPPING_VODAFONE, REF_GEOJSON],
        ),
        # daily temperature of the comuni: used as downloaded (no normalization), processed from raw_data
        _spec("meteo", METEO_VALUE_COLS, [REF_MAPPING_COMUNI]),
        _spec(
            "presenze_alb",
            PRESENZE_ALB_VALUE_COLS,
            [REF_MAPPING_APT, REF_MAPPING_COMUNI],
        ),
        _spec(
            "presenze_extralb",
            PRESENZE_XALB_VALUE_COLS,
            [REF_MAPPING_COMUNI],
        ),
    )
}
ALL_DATASETS = tuple(DATASETS)

# final phenomenon -> datasets that must ALL be updated for it to be recomputed
PHENOMENA = {name: frozenset(deps) for name, deps in PHENOMENA_FILES.items()}
PRESENZE_DATASETS = PHENOMENA["phen_presenze"]
# derived phenomenon -> phenomena that must ALL be computed in the same run for it to be computed
DERIVED_PHENOMENA = {name: tuple(deps) for name, deps in DERIVED_PHENOMENA_FILES.items()}


def phenomena_to_recompute(selected) -> list:
    """Phenomena whose inputs are all among the selected datasets."""
    return [p for p, required in PHENOMENA.items() if required <= set(selected)]
