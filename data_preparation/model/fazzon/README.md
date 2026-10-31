# Fazzon data preparation

Reference datasets for the Fazzon (Lago dei Caprioli, Val di Sole) model. Same structure as the
territorial indexes pipeline (`territorial_indexes/README.md`), reusing `model/common/` (config, io, remote) and
`model/config/settings.yaml` (directories, platform). Run from the `data_preparation` folder:

```bash
python -m model.fazzon.prepare                  # steps 0-2
python -m model.fazzon.prepare --skip-download  # reuse <normalized>/fazzon/ (steps 2 only)
python -m model.fazzon.prepare --refresh        # download the data-lake files again
python -m model.fazzon.prepare --local-sources model/fazzon/data  # local copies instead of the data lake
python -m model.fazzon.prepare --upload         # also log the reference datasets to the platform
python -m model.fazzon.exploratory_analysis     # exploratory analysis and plots (Italian labels)
```

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 0 Sources | `sources.py` | data lake or `local_source_dir` (`fazzon.yaml`) | raw files |
| 1 Standardize | `standardize.py` | raw files | `Output/data/normalized/fazzon/*.parquet` |
| 2 Reference datasets | `reference_datasets.py` | normalized tables | `Output/data/final_data/fazzon/*.parquet` |
| 3 Upload | `model.common.remote.log_dataframe` | reference datasets | platform dataitems (`--upload`) |

* **Sources** (`fazzon.yaml`): each source has its data-lake key and its local path. With
  `local_source_dir` set, or with `--local-sources DIR`, the files are read in place;
  otherwise (the default) they are streamed from the data lake to `Output/data/raw_data/fazzon/`.
  As in the main pipeline only the data lake is read: the measured weather (Male' station, T0074) is fetched from the Meteotrentino open-data service
  by `data_download/download_meteo_meteotrentino.py` and uploaded to `fazzon/meteo/`.
* **Standardize**: one table per source, cleaned and quality-flagged, no modelling choices:
  `fazzon_parking` (2025 + 2026), `fazzon_parking_2025`, `fazzon_greenbus_2025` (rows after the
  pilot flagged `other_area`), `road_counts_hourly` / `road_counts_daily` (cars, classes 2+3, all
  the deliveries; `valid` = 23+ hours, 50+ cars and not below half the other direction), `meteo_station`, `meteo_bulletin`, `vodafone_presences`
  (tourist + visitor, Italian + foreign total), `bike_counts`, `tgc_accessi`, `stations`
  (from `reference/stations.csv`).
* **Reference datasets**:
  * `fazzon_daily`: one row per day from 2023-01-01: measured Fazzon data (cars, bus, motorbikes),
    access rules (`shuttle`, `regulated_parking`, `event_day`, `parking_price`,
    `parking_price_afternoon`, `parking_price_assumed`), weather and weather class (`weather`,
    `tmax_clim`), predictors (road counts,
    Vodafone, bikes, Guest Card).
  * `fazzon_presence_stats`: the presence statistics of the Fazzon model, per season and weather
    class (good / cool / rainy), with the season weights and the weather climatology.
  * `fazzon_modal_shift` (`modal_shift.py`, step 2b): the behaviour model of the 2026 rules —
    the cars expected in 2025 conditions on each 2026 day come from the 2025 relation between
    Fazzon cars, morning traffic up valley at Pellizzano and weather (A37), with a year effect of
    1 ± 10 % (A22; the measured one is reported as a check); response of the visitors by private
    vehicle to the price (β; persons per vehicle 2.5 at 5 euro to 3.5 at 20 euro), and the split
    of the visitors who no longer come by private vehicle (shuttle / walk-up / elsewhere), each
    with a bootstrap 90 % interval; `fazzon_modal_shift_daily`: the 2026 days used, with both the
    traffic-based and the season × weather expectation.
  * `fazzon_monitoring` (`monitoring.py`, step 2c): drift indicators, every year against the
    calibration year on the same calendar window (regime window, rest of the season): cars per day
    at the road stations, cars up valley at Pellizzano in the morning (valley demand), Fazzon cars
    per morning car (drift specific to Fazzon; the response to the access rules). One row per
    indicator × year × period, with the reference value and the ratio to it.

### Adding a new year

* **Road counts** (a new PAT delivery): upload it to `dati_transiti/` on the data lake with its
  delivered name, add an entry at the end of `road_counts` in `fazzon.yaml` (data-lake key and
  local copy), run `prepare` (a new key is downloaded on the next run). A later delivery replaces
  an earlier one on the same station and day, so a delivery can also re-send months already loaded.
* **Fazzon parking / shuttle data**: the readers in `standardize.py` follow the operator
  spreadsheets as delivered (2025 and 2026 side by side): a new spreadsheet needs its reader or a
  new column block.
* **Access rules of the year**: add its window to `REGULATED_PARKING`, its shuttle to
  `SHUTTLE_PERIODS`, special days to `EVENT_DAYS` / `SHUTTLE_NOT_RUNNING` / `REDUCED_PRICE_DAYS`
  and, if they change, the prices (`reference_datasets.py`; record them in `ASSUMPTIONS.md`).
* `fazzon_monitoring` then includes the new year with no code change: read its
  `ratio_to_reference`. The calibration (D9) and the modal-shift estimate (2026) do not change by
  themselves: recalibrate when an indicator drifts.

The Fazzon model (`executor-backend/src/model/fazzon/`) reads the two statistics files from its
data folder (`FAZZON_DATA_DIR`, e.g. `<repo>/data_preparation/Output/data/final_data/fazzon` for a
local run), or downloads them from the platform at startup; it has no built-in values. See
`executor-backend/src/model/fazzon/fazzon_data.py` and `docs/howto/backend.md` (Model API data).

Every decision and assumption, with its evidence, status (e.g. "to validate" with the domain
experts) and where it is used: `ASSUMPTIONS.md` (the code refers to its IDs). The exploratory
analysis writes its tables and plots to `data/analysis_output/` (not versioned). `local-docs/`
holds local working notes and is never versioned.
Tests: `tests/test_fazzon_pipeline.py` (synthetic fixtures, no data needed).
