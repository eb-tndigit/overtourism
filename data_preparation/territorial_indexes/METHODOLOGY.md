# Methodology: how each phenomenon is computed

This document explains, for every file in `Output/data/final_data/`, **which data it starts from, which assumptions
are made and how the values are computed**. How to run the code is in [README.md](README.md).

Every final file has `DATA` and `ID_COMUNE` (6 digits, zero-padded ISTAT code, one comune per row) plus the value
columns below. `DATA` is `YYYY` for yearly phenomena (`phen_popolazione`, `phen_strutture`, `phen_flussi_od`,
`phen_provenienza`) and `YYYY-MM-DD` for daily ones (`phen_presenze`, `phen_consumption`).

| Phenomenon | Granularity | Starting data | Main assumption |
|---|---|---|---|
| `phen_popolazione` | comune × year | ISPAT resident population | value at 1 January |
| `phen_strutture` | comune × year | ISPAT accommodation statistics | totals recomputed from the components |
| `phen_presenze` | comune × day | Vodafone presences (tourists, visitors) + ISPAT hotel / extra-hotel presences | disaggregated (see *Disaggregation*): areas split by beds (tourists) or population (visitors); monthly ISPAT presences spread over days with the average length of stay |
| `phen_consumption` | comune × day | monthly electricity and gas withdrawals | disaggregated in time only: each month spread evenly over its days |
| `phen_flussi_od` | comune × year | Vodafone OD co-visits | only the total rows are used (the country breakdown is anonymized and does not add up); indicators computed per area, then passed to comuni (not split); visitors weigh 0.5 |
| `phen_provenienza` | comune × year | Vodafone presences (HHI) and arrivals (main market) by country of origin | normalized HHI of tourist user-days; main market from tourist arrivals (areas split by beds); small areas left empty (HHI) |
| `phen_tempo_medio_turisti` | comune × year | Vodafone tourist arrivals by origin country + route-time matrix | arrival-weighted mean journey time; Italy uses one configurable reference city, foreign origins use their capitals |

## `phen_tempo_medio_turisti` (comune × year)

Column: `tempo_medio_turisti`, in minutes. The phenomenon is computed from the same processed Vodafone arrivals as
`phen_provenienza`, retaining origin-country volumes before the top-country aggregation discards them.

* **Starting data**: `vodafone_arrivals.csv`, tourist profile, municipality-area level, grouped by year, destination area
  and `userCountry`. Area arrivals are allocated to comuni proportionally to yearly `tot_postiletto`, using the same
  largest-remainder split as the main-market volumes.
* **Travel times**: `Output/travel_times/matrice_tempi_verso_trentino.csv`, generated automatically during processing by
  `territorial_indexes/trentino_travel_times.py`. Each origin-destination duration is joined to the destination comune. Countries
  outside Italy use their ISO3 capital-origin row. `ITA` uses the origin municipality configured as
  `travel_times.reference_city_italy` in `config/settings.yaml` (default `Milano`); it can be changed to another
  municipality present in the matrix, such as `Bologna` or `Firenze`.
* **Computation**: for each destination comune and year, `Σ(arrivals × travel_time_minutes) / Σ(arrivals)` over
  countries with a matching route. Countryless arrivals remain part of `phen_provenienza.volume_totale`, but cannot be
  included in this index. Missing routes are logged and excluded from both numerator and denominator.
* **Interpretation and limits**: this is an estimate of average travel time from an assumed origin, not a measured
  journey. Every Italian arrival is represented by the configured Italian municipality; every foreign arrival by the
  capital. The result does not model the actual starting city, transport mode choice, or individual route. It is yearly
  and arrival-weighted (people arriving), not weighted by days of stay.

## Common rules

### Time window
The window is set in `config/settings.yaml` (`period`). Nothing before **2022-01-01** is kept anywhere.

| Run | From | To | Effect |
|---|---|---|---|
| base build (gen) | 2022-01-01 | 2024-12-31 | creates the phenomena; later days in the raw files (e.g. 2025 Vodafone days) are ignored |
| update 2025 | 2025-01-01 | 2025-12-31 | appends the 2025 rows; other periods in the same files are ignored |

The window is applied in step 2 (before the heavy computations) and again on the final phenomena, so an old
`final_data` file cannot bring back older rows. Yearly data are kept if their **year** is inside the window, daily
data by exact date. An update that finds no data in its window stops with an error.

### Pipeline conventions
* **Standardization** only renames columns, unifies `DATA` / `LOCATION` and cleans comune names; it does not filter or
  aggregate. Exceptions: the Vodafone OD co-visits are reduced while reading (see `phen_flussi_od`), and the ISPAT
  arrivals / presences are read directly from `raw_data` (not normalized).
* **Comune ids**: comune names are resolved to ISTAT ids with `mapping_comuni`; rows of the whole provincia are removed.
* **Vodafone areas** (`locType = TN_MKT_AL_3`, hereafter "areas") may cover one or several comuni, as defined by
  `mapping_vodafone`. Vigo di Fassa and Pozza di Fassa are unified in **San Giovanni di Fassa** (22250).
* **Splitting an area over its comuni** (and every other disaggregation) follows the rules of the next section,
  *Disaggregation*.
* **File names** (raw files, folders, file-name patterns, names of the intermediate files) are not in the code: they are
  all in `config/settings.yaml` (`files`, `naming`, `datasets`, `references`) and in the update yaml.
* **Update**: the update runs the same processing functions as the base build, appends its rows to the existing
  `final_data` and, on a duplicated `DATA` + `ID_COMUNE`, the update row wins. A phenomenon is recomputed only if **all**
  its datasets are in the update. Every source of the update first goes through an *adapter* that checks the file layout
  (a mismatch stops the run) and renames the columns to the standard ones.

### Disaggregation

Several sources are coarser than the final granularity (comune × day): Vodafone gives **areas** that may cover several
comuni, ISPAT gives **months** for APT / the whole provincia, and the energy files give **months** per comune.
*Disaggregation* turns one coarse value into several finer ones. Territorial splits (area → comuni) are all done by the
same function, `disaggregate` in `utils/disaggregation.py`, called by `process_vodafone` (`process_std_data.py`),
`stay_distribution.py` (ISPAT) and `main_market.py` (arrivals). The spread over time (month → days) is done by two
dedicated methods, described below, because it needs more than a weight.

**Principle.** A parent row (one value of a coarse area and / or period) is split among its *children* (the comuni of the
area, the days of the month) **in proportion to a weight**: `child = parent × w_child / Σ w`. The values are therefore
**redistributed, never created or lost**: the children add up to the parent. The weights are a supporting variable, not
an estimate of the real distribution: where the real split is unknown, the result is a plausible allocation, not an
observation.

**Axes.**
* *Space*: area → comuni, with `disaggregate(..., axis="space")`. This is the only axis of that function used by the
  pipeline. The parent row lists the comuni of the area (`mapping_vodafone`, `mapping_apt`; for the whole provincia, all
  the comuni). An area that is a single comune is unchanged.
* *Time*: month → days. ISPAT presences use the **length-of-stay model** (see `presenze_alb` and `presenze_xalb`); the
  energy withdrawals are divided **equally** over the days (see `phen_consumption`, done in `process_energy`).
* `disaggregation.py` also contains a time axis (`disaggregate_time`) and a combined one (`distribute_like`: a phenomenon
  distributed over comuni *and* days following another phenomenon as weight). They are generic tools, **not used** by the
  current steps, so no published value depends on them.

**Weights.** They are always a non-negative number per child:

| Where | Axis | Weight |
|---|---|---|
| Vodafone tourists (`presenze_vodafone`) | space | `tot_postiletto` of the year (beds) |
| Vodafone visitors (`presenze_escursionisti_vodafone`) | space | resident population of the year |
| ISPAT hotels (`presenze_alb`) | space, after the time step | `alberghieri posti_letto` of the year |
| ISPAT extra-hotels (`presenze_xalb`) | space, after the time step | `extra alb. Posti_letto` of the year |
| Vodafone arrivals (`volume_*` of `phen_provenienza`) | space | `tot_postiletto` of the year |
| ISPAT monthly presences | time | the stay model of `stay_distribution.py` (see `presenze_alb` and `presenze_xalb`), not a plain weight |
| Energy (`phen_consumption`) | time | none: **equal** share for every calendar day of the month (leap days included) |

**Rules of the territorial splits** (`disaggregate`; the largest-remainder rounding, rule 3, is also used inside the
stay model, and the energy is a plain division of continuous values).
1. *Weight of the right period.* Weights that are yearly (beds, population) are looked up for the year of the day. If that
   year is missing, the **nearest available year** is used (the earlier one on a tie), with a warning.
2. *No usable weight.* If all the weights of a parent are zero or missing, the parent is split **uniformly** among its
   children, with a warning (when there are several children and something to split).
3. *Integers stay integers.* Counts are rounded with the **largest-remainder method**: every child gets the integer part of
   its share and the remaining units go to the children with the largest fractional parts, so the children add up to the
   parent **exactly**. Continuous values (energy) are not rounded.
4. *Sum check.* After every territorial split the children are summed back for every input row (and, for Vodafone, per day;
   for the arrivals, per year) and compared with the original values. A mismatch **stops the run**; the totals are logged
   otherwise.
5. *Keys.* Comune ids are compared in normalized form (`22001`, `'22001'` and `'022001'` are the same id), and the output has
   one `ID_COMUNE` per row, zero-padded.

**Order for the ISPAT presences.** Time first, then space: the monthly arrivals and presences of every area are spread
over the days with the length-of-stay model (see `presenze_alb` and `presenze_xalb`), and the resulting area × day values
are split among the comuni with the bed weights. The check is done on the final result: the comune × day values, summed
back by area and month, must equal the official monthly presences (a mismatch stops the run).

**What is *not* disaggregated.** Quantities that are not additive cannot be split without changing their meaning, so
they are computed on the area and **passed unchanged to every comune of the area** (a comune with several areas gets a
weighted mean): the concentration `hhi_provenienza` and the flow indicators of `phen_flussi_od` (shares and PageRank).
Population, structures and the energy withdrawals are already at comune level and are not split in space.

## `phen_popolazione` (comune × year)

* **Starting data**: ISPAT resident population (`popolazione_2020_2024` for the base build, `popolazione_2026_ISPAT.csv`
  for the 2025 update).
* **Assumption**: the population of year *Y* is the resident population **at 1 January of *Y*** (for the 2025 update:
  the column `Popolazione residente al 1.1.2025`).
* **Computation**: none beyond id resolution; values are kept as they are. The population is also used as weight to split
  the visitors of multi-comune areas (see `phen_presenze`).

## `phen_strutture` (comune × year)

* **Starting data**: ISPAT accommodation statistics: `Annuario-TavXIII-per-comune-csv.csv` combined with
  `strutture_annuario_2024.ods` (base build), `numero_strutture_ISPAT_2025.xlsx` (update). `-` is read as 0.
* **Computation**: for each comune and year, number of structures (`strutture_*`) and beds (`letti_*`) of four types:
  hotels (`alb`), extra-hotels (`exalb`), tourist private accommodation (`alloggi_turistici`) and accommodation at
  disposal (`alloggi_disp`).
  * `tot_strutture` = alb + exalb + alloggi turistici
  * `tot_postiletto` = letti_alb + letti_exalb + letti_alloggi_turistici
* **Assumptions**: the totals of the raw files are discarded and recomputed from the components, for consistency between
  years and layouts. The `alloggi_disp` are kept in their own columns but are **not** part of the totals.
* **Use**: the beds are the weights used to split areas and ISPAT presences over comuni.
* **Update**: the default adapter expects the 2025 layout (XLSX with two-row header and `Comune`); the 2024 ODS has its
  own adapter. A file that does not match its adapter stops the run rather than being guessed.

## `phen_presenze` (comune × day)

Columns: `presenze_vodafone`, `presenze_escursionisti_vodafone`, `presenze_alb`, `presenze_xalb`. The four measures are
joined with an **inner join** on `DATA` + `ID_COMUNE`: the file only has the days and comuni present in all of the
sources (dropped records are counted in warnings).

### `presenze_vodafone` (tourists) and `presenze_escursionisti_vodafone` (excursionists)
* **Starting data**: Vodafone presences per area and day, `userProfile` `TOURIST` (people staying overnight) and `VISITOR`
  (day visitors / excursionists), area level. Base build: `vodafone_attendences` (2022-01-01 .. 2024-03-31) completed with
  `vodafone_attendences_new.csv` (2023-01-01 .. 2025-12-31); on dates present in both files the newer file wins. The
  2025 update reads `vodafone_attendences_new.csv`.
* **Assumptions**: other profiles (e.g. commuters) are discarded. An area that is a single comune is unchanged. For an
  area covering several comuni:
  * tourists are allocated **proportionally to `tot_postiletto`** of the year (supply of overnight accommodation);
  * visitors are allocated **proportionally to the resident population** of the same year (they do not use beds);
    a missing or non-positive population for any comune of a shared area stops the run.
* **Computation**: the two profiles are separated before allocation, each is split over the comuni of its area with the
  weights above, and the totals are checked per area-day and per day.

### `presenze_alb` and `presenze_xalb` (ISPAT hotels / extra-hotels)
* **Starting data**: ISPAT **monthly** arrivals and presences, files `Presenze/<arrivi|presenze>_<alb|exalb>_<year>.csv`
  (one year per file; the base build reads 2022–2024, the update the later years). Hotels (`alb`) are reported per APT
  (only the `<APT> - Totale` columns are used); extra-hotels (`exalb`) as a provincial total (first and last column).
* **Assumptions**: the monthly arrivals are spread **uniformly over the days of the month**; a guest arriving on day *d*
  stays `s = presences / arrivals` nights (the average stay of the month), also across the end of the month and of the
  year. Files of the previous year give the stays crossing 31/12; if absent (e.g. 2021) they are approximated, with a
  warning. The comuni of an area share the presences in proportion to the beds of the type
  (`alberghieri posti_letto` for alb, `extra alb. Posti_letto` for exalb).
* **Computation**:
  1. average stay of the month `s = presences / arrivals`;
  2. arrivals spread uniformly over the days of the month;
  3. each arrival adds one presence on each of the following `s` days (the fractional part on the last day);
  4. the daily values of each month are rescaled and rounded (largest remainder) so that each area-month adds up
     **exactly** to the official monthly presences;
  5. area × day is split over comuni by beds.
  Arrivals are not in the output.

## `phen_consumption` (comune × day)

Columns: `ee_services`, `ee_total`, `gas_services`, `gas_total` (electricity in kWh, gas in Smc, as in the sources; no
unit conversion).

* **Starting data**: monthly withdrawals per comune (semicolon CSV, one column per month).
  * `ee_services`, `gas_services`: one file per usage category in `Energia/ee_prelievi/` and `Energia/gas_prelievi/`
    (`49_3`, `55_1`, `56_1`, `93_1`, `96`: the ATECO codes of the tourism-related services). The categories of a
    comune-month are **summed**.
  * `ee_total`, `gas_total`: `Energia/ee_prelievi/totale_ee.csv` and `Energia/gas_prelievi/totale_gas.csv`, withdrawals of
    all the usages. They are excluded from the category folders, otherwise they would be summed into `*_services`.
* **Assumptions**: the total files have the same layout as the category files, with **one row per comune and year**
  (if they also listed single categories next to a grand total, summing would double count). Comuni without an ISTAT id
  are dropped, with a warning.
* **Computation**: the monthly value of a comune is **divided evenly over the calendar days** of the month (leap days
  included), so the days add up to the month; there is no daily profile. The four measures are combined with an
  **outer join** on `DATA` + `ID_COMUNE` (a measure missing for a key is left empty).
* **Update**: the energy datasets are currently `enabled: false` in `update_2025.yaml`; since `phen_consumption`
  needs all four, it is not appended by the 2025 update until they are enabled together.

## `phen_meteo` (comune × day)

Columns: `t_min`, `t_max`, `t_mean`, `hdd`, `rain_mm`.

* **Starting data**: `meteo/meteo_trentino_comuni_<year>.csv` (one file per year, `files.meteo` in settings.yaml): for every
  comune and day the temperatures and rain of the nearest Meteotrentino station (`date`, `comune_id`, `t_min`, `t_max`,
  `rain_mm` + station info). Downloaded to `raw_data/meteo/` and used as downloaded (not normalized), processed in step 2 from there.
* **Computation**: `t_mean = (t_min + t_max) / 2`; `hdd = max(0, base_temp - t_mean)`, the heating degree days of the day,
  with `base_temp = meteo.hdd_base_temp` (18 °C, a chosen value, not an estimate). A day without temperature has `t_mean` and
  `hdd` empty (never filled): whoever uses them averages the days that have a value. A duplicated comune + day is averaged.
* **Window**: like every phenomenon, only the days of the window of the run are kept (the weather files may go beyond it).

## `phen_ee_tourism_share` (comune × year)

Columns: `quota_ee_turismo` (0-1), `presenze_medie`, `consumo_medio`, `affidabilita`.

* **Derived phenomenon**: computed from `phen_consumption` (`ee_services`), `phen_presenze` (`presenze_vodafone`,
  `presenze_escursionisti_vodafone`) and `phen_meteo` (`hdd`), in the same run. Not computed if one of them is missing.
* **Model** (`utils/consumption_share.py`): for every comune and every calendar year, on the 12 monthly means,
  `y = a + b·x + c·hdd` (WLS, weight = days of the month; `y` consumption, `x` tourists, `hdd` heating degree days); share =
  `b · mean(x) / mean(y)`. The slope `b` is shrunk towards the province-wide slope **of the same year** (partial pooling,
  `TAU`), the excursionists have weight 0, the standard errors are Newey-West. See the module docstring for the reasons.
* **Years**: all the 12 months of the year are needed (a month needs 7 days with consumption, presences and temperature). A
  comune-year that cannot be estimated is **absent** (never a zero). The estimate does not use other years.
* **`affidabilita`** (text: `alta` / `media` / `bassa`): judged on the comune's OWN evidence (plain WLS slope, no pull towards
  the pooled one), one-sided: `alta` if the lower bound of the 95% interval of the slope is above zero, `media` if only the
  80% lower bound is, `bassa` otherwise (the share is then mostly the prior). A share of 0.9 or more is always `bassa`.
  The thresholds are `Z_ONE_SIDED_95` / `Z_ONE_SIDED_80` in `consumption_share.py`.
* **Not published**: R², number of months, the slope.

## `phen_flussi_od` (comune × year)

Columns: `quota_ingresso`, `quota_uscita`, `quota_intra_ambito`, `pagerank`, each with suffix `_tourist`, `_visitor`,
`_combined`, plus `giorni_coperti`.

* **Starting data**: Vodafone OD co-visits `vodafone_covisits.csv`: day × source area × target area, with `covisitType`
  `ND` (users present) or `UD` (users who visit both areas), profile, country and cluster. Only area-level rows, profiles
  `TOURIST` and `VISITOR`, and the **total rows** are kept; the file is read in chunks. The 2025 update reads the same
  object, filtered to 2025.
* **Total rows and anonymization**: for every day × source × target × `covisitType` the file starts with **one total row per
  profile** (`TOURIST`, then `VISITOR`; `userCountry` and `userCluster` empty), followed by the breakdown by country
  (with its cluster). The total rows are the only ones used, and **the totals are never recomputed by summing the
  countries**, because the breakdown does not add up to the total: Vodafone's privacy rule sets to 0 every cell below
  a threshold (about 15 users; the smallest non-zero value, in totals and countries alike, is 15), so each country row
  only shows its visible part. On the sample file (one day, 460 total rows, 173 of them non-zero) the countries add up
  exactly to the total in only 32% of the non-zero totals and to **less** than it in all the others, never more; overall
  the breakdown shows 91.6% of the user-days (e.g. 1,594 of 2,091 tourist user-days for one pair), and 9 pairs with totals
  of 16 to 27 have an all-zero breakdown. Keeping both the totals and the countries would roughly **double** the
  user-days (the sample sums to 41,684 with the totals only and to 79,878 with every row), so there is no option to keep
  them: a row is a total only if both `userCountry` and `userCluster` are empty. The total is itself censored (it is 0
  below the threshold), so very small flows are invisible (see the reliability threshold below).
* **Computation**, per year and profile, on user-days summed over the year, with `T(j)` = users present in area *j*
  (`ND`, source `*`):
  * `quota_ingresso` = UD(\*→j) / T(j): share of the users present in *j* who come from another area;
  * `quota_uscita` = UD(j→\*) / T(j): share who go on to another area (if the aggregated exit row is missing, the sum of
    the pairs is used, an underestimate);
  * `quota_intra_ambito` = flow between areas of the same ambito / total flow between areas, on the in + out pairs of *j*;
  * `pagerank` = PageRank (damping 0.85) of the directed network of UD flows between areas.
* **Combined profile** (`_combined`): a tourist-day and a visitor-day do not weigh the same, so the two profiles are
  merged with fixed weights, **TOURIST 1 and VISITOR 0.5** (`COVISITS_PROFILE_WEIGHTS` in `utils/datasets.py`). The weights
  multiply the user-days **before** the indicators: share = Σ w·UD / Σ w·T, and PageRank on the network of the weighted
  flows. The weight is a chosen coefficient, not an estimate; changing it requires rerunning the base build and the update.
* **Areas → comuni**: an area covering several comuni passes its value to **all** of them (not split); a comune with
  several areas gets the weighted mean (weights: `T` for the two shares, intra-ambito flow for `quota_intra_ambito`, in +
  out flow for `pagerank`).
* **Reliability**: the two mobility shares are empty where the raw yearly user-days `T` are below 100 (`COVISITS_MIN_T`);
  they can exceed 1 because a user can be counted in several areas.
* **Partial years**: `giorni_coperti` is the number of days with data in the year. A year with fewer than 360 days is
  kept but flagged (warning). In the update the indicators are recomputed on all the days of the year.
* **Where empty values come from** (they are expected, not errors; `process_covisits` logs the share of empty values per
  column):
  1. `quota_ingresso` / `quota_uscita`: area with `T` below 100 user-days (or zero) in that year and profile. This
     is most frequent for `_visitor` and for small areas, and rarer for `_combined`;
  2. `quota_intra_ambito`: area with no flow towards other areas in that year and profile (or with unknown ambito);
  3. `pagerank`: area with no UD flow to / from other areas, since only the nodes of the network get a value;
  4. any `_tourist` / `_visitor` column: no area of the comune has data for that profile-year;
  5. a comune whose areas are all empty for a metric gets an empty value; a comune with no area in the area map is
     absent from the file altogether.

## `phen_provenienza` (comune × year)

Columns: `hhi_provenienza`, `giorni_coperti` (concentration, from the presences) and `volume_totale` plus
`volume_mercato_11`, `nome_mercato_1`, ..., `volume_mercato_N`, `nome_mercato_N` (top source countries, from the arrivals;
`N` is `provenienza.top_n_countries` in `config/settings.yaml`, default 3). The two parts are joined with an **outer join**
on `DATA` + `ID_COMUNE`: a year / comune present in only one source keeps the other columns empty. In an update both
datasets (`vodafone_provenienza`, `vodafone_arrivi`) must be updated together.

### `hhi_provenienza` (concentration of the source markets)
* **Starting data**: the standardized Vodafone presences (same raw files as `phen_presenze`), `TOURIST` profile, area
  level, per day × area × `userCountry`.
* **Computation**: yearly user-days per area and country → shares `s_c` → `H = Σ s_c²` → normalized
  `hhi = (H − 1/N) / (1 − 1/N)`, with `N` the countries with presences > 0 (`N = 1` → 1). 0 means presences evenly spread
  over the markets, 1 a single market. Italy is one market; rows without country are ignored.
* **Assumptions**: areas with fewer than 100 yearly user-days (`PROVENIENZA_MIN_PRESENZE`) are left empty: Vodafone hides
  counts below a privacy threshold (about 15), so shares only cover the visible part of the presences and small areas are
  biased upwards. The 100 threshold is a chosen value. The index is therefore the concentration of the *observed* markets.
  The HHI is not additive, so an area is **not split**: it passes its value to all its comuni, and a comune with several
  areas gets the mean weighted by the areas' user-days.
* **Partial years**: `giorni_coperti` below 360 flags a partial year (warning). In the update the file must contain all
  the days of the year, because the index is computed on the update data only.

### Top source-country volumes and total arrivals
* **Starting data**: Vodafone **arrivals** `vodafone_arrivals.csv`, same layout as `vodafone_attendences_new.csv` (`locId`, `date`,
  `value`, `userProfile`, `locType`, `userCountry`) but `value` counts arrivals. `TOURIST` profile, area level
  (`TN_MKT_AL_3`), per day × area × `userCountry`; standardized as `vodafone_arrivi_std` (value column `arrivi`).
  Arrivals, not presences, because a presence counts the same person again on every day of the stay: with presences a market
  with long stays weighs more than the number of people it brings.
* **Countries** (`utils/markets.py`): recognized Italian and European alpha-3 codes and names are rendered as
  English country names (for example, `ITA` → `Italia`, `DEU` → `Germania`). Other supplied country labels are retained.
* **Computation**: (1) arrivals per year, area and country; (2) an area with several comuni is **split** among them
  proportionally to `tot_postiletto` of the year, like the tourist presences of `phen_presenze` (arrivals are additive, so,
  unlike the HHI, they can be split; integer counts stay integer with the largest-remainder rule and totals are checked);
  (3) per comune, countries are ranked by arrivals descending (ties alphabetically). The top `N` countries and their
  arrival volumes are written to the parquet; unused ranks are empty.
* **Assumptions**: `volume_totale` includes every visible arrival, including rows with no country. Country volumes do not
  include arrivals with no country or countries outside the top `N`.
  Vodafone hides counts below a privacy threshold (about 15), so volumes are the *visible* arrivals. Arrivals are smaller
  than presences (by the average length of stay), so privacy censoring bites more. **No minimum volume is applied in the
  file**: small comuni are kept with their volumes; consumers can filter by `volume_totale`. Country volumes of comuni in
  a shared area are equal by construction (same area, same split).
* **Partial years**: a year with fewer than 360 days (`ARRIVI_FULL_YEAR_DAYS`) in the file is kept but a warning is logged;
  there is no coverage column for the arrivals (`giorni_coperti` is the one of the presences). In the update the file must contain
  all the days of the year.

## Presences or arrivals: which indicators are affected
Presences are user-days: a person who stays 5 days is counted 5 times. This is right for the stock indicators (how many
people are present on an average day) and wrong when the question is "how many people / from where". Checked one by one:

| Phenomenon | Uses | Affected? |
|---|---|---|
| `phen_presenze` (and the indicators on it: turisticità, densità, carico antropico, turismo sommerso) | daily presences, compared with ISPAT presences / residents / km² | No. They measure the average daily load, which is what presences are for; the hidden-tourism ratio compares presences with presences |
| `phen_flussi_od` | user-days of co-visits (`ND`, `UD`), shares and PageRank on user-days | No. Ratios of user-days with the same unit; there is no arrivals counterpart in the OD file. The shares are *of user-days* (a long stay weighs more), by design |
| `phen_consumption`, `phen_popolazione`, `phen_strutture` | not Vodafone | No |
| `phen_provenienza.hhi_provenienza` | share of each country in the **presences** | **Yes**, see below |

`hhi_provenienza` measures how concentrated the *people* are across the markets, but it is computed on user-days, so a market
with long stays weighs more. Example: Italians 50 arrivals × 2 nights, Germans 50 × 7 nights. On arrivals the two markets are
equal (normalized HHI 0); on presences Germans are 78% and the index is 0.31. The new index (arrivals) and the HHI (presences)
can therefore disagree in the same file. Recomputing the HHI on the arrivals is a small change (the standardized arrivals have
the same layout, only `value` is `arrivi`), but it changes the values already produced and `PROVENIENZA_MIN_PRESENZE` (100
user-days) would have to be revisited, because arrivals are fewer; it has **not** been done here.

## Points still to verify on the real data

These parts were written without access to the real files; check them before trusting the numbers:
* the layout and content of `totale_ee.csv` / `totale_gas.csv` (one row per comune-year assumed) and of the 2025 energy
  files;
* that `vodafone_covisits.csv` contains the 2025 days, and that the whole file has the structure seen in the 999-row sample
  (total rows with empty `userCountry` and `userCluster` first, no duplicated total rows; the code warns on duplicates);
* the ISPAT presences of 2022–2024 and the choice of reading `vodafone_attendences_new.csv` for April–December 2024;
* `vodafone_arrivals.csv`: that `userCountry` is ISO alpha-3 / alpha-2 / English names (otherwise the run stops with the list of
  the values found), the period it covers (all the days of every year), that `TOURIST` / `TN_MKT_AL_3` are present as in the
  presences, and that rows with an empty `userCountry` (if any) are totals that must be ignored;
* the full pipeline has been tested on synthetic data only (`python -m pytest tests`), not run end to end.
