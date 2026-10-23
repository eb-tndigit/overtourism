# Territorial indexes: data preparation pipeline

Builds the phenomenon dataframes behind the territorial indexes:
`phen_popolazione`, `phen_strutture`, `phen_presenze`, `phen_consumption`, `phen_flussi_od`, `phen_provenienza`,
`phen_tempo_medio_turisti`.
`phen_meteo`, `phen_ee_tourism_share`.
There are two entry points: a **base build** (2022-2024, `gen_phenomenon_dataframes.py`) and a **yearly update**
(`update_phenomenon_dataframes.py`, currently 2025). The base build can also run the 2025 update.

This README explains how to run the code and the steps that lead to the final files. **How each phenomenon is computed
(starting data, assumptions, formulas) is in [METHODOLOGY.md](METHODOLOGY.md).** This folder is part of
`data_preparation/`; the model pipelines live in `../model/`. All commands are run from this folder
(`territorial_indexes/`).

## Requirements

* Python 3 with `pandas`, `numpy`, `pyyaml`, `unidecode`, `pyarrow` (parquet) and, for the tests, `pytest`.
* `boto3`, `geopandas` and `digitalhub` are imported only when needed (download from the data lake / platform upload).
  The S3 credentials are read from the section `platform.dhcore_env` of the file `platform.credentials_file` (default `.dhcore.ini`) in the home folder.
* To work without the data lake, put the files in a local folder and set `update.local_source_dir` in `config/settings.yaml`
  (update only), or copy them in `Output/data/raw_data/` and run with `--skip-download`.

## Folder layout

```
territorial_indexes/
├── README.md, METHODOLOGY.md
├── gen_phenomenon_dataframes.py     # entry point: base build, all steps (--with-update / --update-only: also the 2025 update)
├── update_phenomenon_dataframes.py  # entry point: yearly update
├── config/
│   ├── settings.yaml                # directories, format, period, platform, ALL the file names
│   └── updates/                     # one yaml per yearly update
├── tests/
└── utils/
    ├── config.py, datasets.py, window.py          settings (incl. file names) as constants, dataset registry, time window
    ├── cleaning.py, io.py, readers.py, remote.py  helpers, read/write, file readers, S3 / DigitalHub
    ├── disaggregation.py, stay_distribution.py    area -> comune split, monthly -> daily presences
    ├── flows.py, provenienza.py                   flow indicators, concentration of the source markets
    ├── markets.py, main_market.py                 country names and top source countries (from arrivals)
    ├── adapters/                                  layout of each yearly source -> standard layout
    ├── update/                                    readers and config parsing of the update
    └── steps/                                     download_data, standardize_raw_data, process_std_data,
                                                   create_phenomena_df (+ the update versions of standardize / process)
```

## Configuration

Directories, file format, time window and platform are all in **`config/settings.yaml`** (neither script takes
directories as arguments). The main entries:

| Entry | Meaning |
|---|---|
| `output_dir` | root of everything the pipeline writes (`../Output`, shared with the model pipelines) |
| `paths.*` | sub-folders: `mapping`, `raw`, `normalized`, `processed`, `final` |
| `type_format` | `parquet` (default) or `csv`, for **every** file; keep it the same in the base build and in the update |
| `period` | time window: `start` (nothing before it is kept), `base_end`, `update_start`, `update_end` |
| `references` | data-lake keys of the json mappings and of the geojson of the Vodafone areas |
| `files` | the RAW files of the base build (data-lake keys): popolazione, strutture, Vodafone presences / arrivals / co-visits, the ISPAT presences (`files.presenze`: folder, file-name pattern, measures, tags) and the energy files (`files.energy`: root, folders, category files, totals) |
| `datasets` | names of the standardized (`std`), processed and update files of every dataset |
| `naming` | suffixes of the intermediate files (`_std`, `_update`) |
| `platform` | project, bucket and prefix of the data lake, name of the credentials file |
| `update.default_config` | update yaml used by `gen_phenomenon_dataframes.py` when `--update-config` is not given |

**No file name is written in the Python code** (`utils/datasets.py` included): to add, rename or replace a file, edit the
yaml (`settings.yaml` for the base build, `config/updates/<round>.yaml` for an update). The only name left in the code is
that of `settings.yaml` itself (`utils/config.py`), which cannot be read from itself. A missing key stops the run at
start-up, naming the key.

The year-specific parts of the yearly update (files, readers, adapters, column names) are in `config/updates/<round>.yaml`.

## Steps that produce the final files

```
data lake -> raw_data -> normalized -> data_processed -> final_data
    (0)         (1)          (2)             (3)
```

All folders are under `Output/` (`data/raw_data`, `data/normalized`, `data/data_processed`, `data/final_data`; the
json mappings are in `Output/mapping`). Every step reads the output of the previous one.

| Step | Module | Reads | Writes |
|---|---|---|---|
| 0 Download | `utils/steps/download_data.py` | data lake (S3) | `raw_data/` and `mapping/` |
| 1 Standardization | `utils/steps/standardize_raw_data.py` | `raw_data/` | `normalized/<dataset>_std` |
| 2 Processing | `utils/steps/process_std_data.py` | `normalized/` + mappings | `data_processed/<dataset>_pr` |
| 3 Phenomena | `utils/steps/create_phenomena_df.py` | `data_processed/` | `final_data/phen_*` (+ platform with `--upload`) |

1. **Download** copies the files as they are (no renaming). A file already present and non-empty in `raw_data/` or
   `mapping/` is **never downloaded again** (`--force-download` overrides this, `--skip-download` skips the step). A partially
   written or outdated file is not detected: delete it or force the download.
2. **Standardization** uniforms column names, the `DATA` / `LOCATION` schema, dates and comune names, without filtering or
   aggregating. Output: `popolazione_std`, `strutture_std`, `vodafone_std`, `vodafone_arrivi_std`, `ee_prelievi_std`, `gas_prelievi_std`,
   `ee_totale_std`, `gas_totale_std`, `vodafone_covisits_std` (the Vodafone OD co-visits are reduced to the rows the indicators
   use). The ISPAT arrivals / presences are read directly from `raw_data` in step 2.
3. **Processing** applies the time window, resolves `ID_COMUNE` and moves every dataset to its final granularity (comune ×
   year or comune × day), including the splits of Vodafone areas and the monthly → daily spread of ISPAT presences and
   energy. The datasets are processed in this order: strutture and popolazione, Vodafone (needs beds and population),
   ISPAT alb and extralb, energy, provenienza, co-visits.
4. **Phenomena** builds the final files from the processed datasets (joins, column order) and applies the time window again.
   `phen_presenze` is an inner join of Vodafone, alb and extralb; `phen_consumption` an outer join of the four energy
   measures; the other phenomena are the processed dataset as it is.

Before processing arrivals, the pipeline prepares `Output/travel_times/matrice_tempi_verso_trentino.csv` automatically.
The matrix and routing caches are stored under `Output/travel_times/`; a complete existing matrix is reused, while
previously unseen Vodafone country origins are added using cached routing blocks where possible. The standalone router
is `trentino_travel_times.py` in this folder.

The dataset → phenomenon mapping is in the `phenomena` section of `config/settings.yaml`.

## Running the base build

```bash
python -m gen_phenomenon_dataframes                   # base build 2022-2024, all steps, parquet
python -m gen_phenomenon_dataframes --with-update     # base build, then the update of 2025 (appended)
python -m gen_phenomenon_dataframes --update-only     # ONLY the update of 2025 (the base build must already exist)
python -m gen_phenomenon_dataframes --update-config config/updates/update_2025.yaml   # yaml used by the two options above
python -m gen_phenomenon_dataframes --skip-download   # reuse Output/data/raw_data, step 0 not run
python -m gen_phenomenon_dataframes --force-download  # download again also the files already in raw_data
python -m gen_phenomenon_dataframes --upload          # also log the final data to the platform
```

Each step can also be run alone, e.g. to recompute from step 2 onwards after a code change, with the raw and normalized
data already in place:

```bash
python -m utils.steps.process_std_data        # step 2
python -m utils.steps.create_phenomena_df     # step 3
```

From Python: `main_compute_phenomena_dfs(with_update=True)` or `run_update_2025()`.

## Running a yearly update

The update runs the same four steps with update-specific standardization and processing, then **appends** its rows to the
existing `final_data` files (on a duplicated `DATA` + `ID_COMUNE` the update row wins; earlier rows are untouched).
The base build must have been run before, in the same `Output/` folder and with the same `type_format`.

```bash
python -m update_phenomenon_dataframes --config config/updates/update_2025.yaml
python -m update_phenomenon_dataframes --config ... --skip-download   # reuse raw_data and mappings
python -m update_phenomenon_dataframes --config ... --force-download  # download again the files already in raw_data
python -m update_phenomenon_dataframes --config ... --upload          # also log the final data to the platform
python -m update_phenomenon_dataframes --config ... --no-strict       # adapter schema mismatch becomes a warning
python -m utils.steps.process_update_std_data --config ...            # a single step
```

| Step | Module | Writes |
|---|---|---|
| 0 Download | `download_data.py` (`download_update_raw_data`) | files listed in the update yaml, skipping those already in `raw_data/` |
| 1 Standardization | `standardize_update_raw_data.py` | `normalized/<dataset>_update_std` |
| 2 Processing | `process_update_std_data.py` | `data_processed/<dataset>_update_pr` (the base `_pr` files are not modified) |
| 3 Phenomena | `create_phenomena_df.py` | `final_data/phen_*` with the update rows appended |

Notes:
* A phenomenon is recomputed only if **all** its datasets are in the update. In `update_2025.yaml` the four energy
  datasets are `enabled: false` (no 2025 energy files confirmed), so `phen_consumption` is not appended until they are
  enabled together.
* The final files of the base build must have the same columns as the update: after a change of the output columns (e.g.
  the addition of `presenze_escursionisti_vodafone`, `ee_total`, `gas_total`) **rerun the base build** before the update.
* `phen_provenienza` needs **both** `vodafone_provenienza` and `vodafone_arrivi` in the update (`vodafone_arrivals.csv`, with
  ALL the days of the year). It gained the three main-market columns: rerun the base build before the update.
* The 2024 ISPAT presence files (downloaded by the base build) are needed by the 2025 update for the stays crossing the new year.

### The update config (`config/updates/<round>.yaml`)

Nothing year-specific is in the code: which datasets, file names, readers, adapters and their parameters are here.

```yaml
datasets:                                      # only these are updated
  popolazione:
    enabled: false                             # optional (default true): skip it this round
  strutture:
    sources:                                   # several sources are stacked (e.g. one per year)
      - file: numero_strutture_ISPAT_2026.xlsx
        reader: excel                          # csv | excel | ispat_apt_tsv | ispat_grouped_tsv | covisits_csv | energy_csv
        reader_kwargs: {header: [0, 1]}        # passed to the reader
        adapter: strutture_annuario            # handler of the file layout
        adapter_kwargs: {year: 2026}
        # label: "2026"                        # optional: keeps this source separate until the processed parts are stacked
```

Sources of the same dataset are stacked (later ones win on overlaps). To update a subset, list only those datasets or
set `enabled: false` on the others; their final data remain unchanged. Config errors (unknown dataset, adapter or reader,
wrong adapter parameters) are reported before anything is downloaded.

### When a new year arrives

1. Upload the raw files to the data lake.
2. Copy `config/updates/update_2025.yaml` to `update_2026.yaml` and change file names, `year` and column names. Set the
   window with `period: {start, end}` in the yaml, or edit `period.update_*` in `settings.yaml`.
3. If the provider changed a layout, add an adapter in `utils/adapters/` and nothing else:
   ```python
   @register_adapter("popolazione_ispat_2027", kind="popolazione")
   def popolazione_ispat_2027(raw: RawData, *, year: int) -> pd.DataFrame:
       return raw.df.rename(columns={"Territorio": "comune", "Residenti": "popolazione"}).assign(anno=year)
   ```
   The output must contain the columns of its `kind` (`KIND_SCHEMAS` in `utils/adapters/base.py`); a mismatch stops the
   standardization with a clear error.
4. Run the update and inspect `data_processed/<dataset>_update_pr` and the appended rows in `final_data/`.

The layouts of future yearly files cannot be known in advance (data have been retrieved on request), so new files
may need a new adapter.

## Outputs

| File in `Output/data/final_data/` | Granularity | Columns (after `DATA`, `ID_COMUNE`) |
|---|---|---|
| `phen_popolazione` | comune × year | `popolazione` |
| `phen_strutture` | comune × year | beds and structures by type, `tot_strutture`, `tot_postiletto` |
| `phen_presenze` | comune × day | `presenze_alb`, `presenze_xalb`, `presenze_vodafone`, `presenze_escursionisti_vodafone` |
| `phen_consumption` | comune × day | `ee_services`, `ee_total`, `gas_services`, `gas_total` |
| `phen_flussi_od` | comune × year | `quota_ingresso`, `quota_uscita`, `quota_intra_ambito`, `pagerank` × (`tourist`, `visitor`, `combined`), `giorni_coperti` |
| `phen_provenienza` | comune × year | `hhi_provenienza`, `giorni_coperti`, `volume_totale`, top-country `volume_mercato_N` / `nome_mercato_N` pairs (`N` configurable in `config/settings.yaml`) |
| `phen_tempo_medio_turisti` | comune × year | `tempo_medio_turisti` (minutes; weighted by arrivals; Italian reference city configurable in `config/settings.yaml`) |
| `phen_meteo` | comune × day | `t_min`, `t_max`, `t_mean`, `hdd`, `rain_mm` |
| `phen_ee_tourism_share` | comune × year | `quota_ee_turismo`, `presenze_medie`, `consumo_medio`, `affidabilita` (`alta` / `media` / `bassa`) |

`phen_meteo` comes from the weather files `meteo/meteo_trentino_comuni_<year>.csv` (data lake, saved in `raw_data/meteo/`, one per year: the base build
downloads the years of its window, the updates list the later ones in their yaml). `phen_ee_tourism_share` is **derived**
from `phen_consumption`, `phen_presenze` and `phen_meteo` (`derived_phenomena` in `config/settings.yaml`), right after
them and in the same run; it is computed only if all three are computed in that run.

Empty values in `phen_flussi_od` and `phen_provenienza` are expected in some cases (small areas, missing flows); see
METHODOLOGY.md. The step 2 log reports the share of empty values per column of `phen_flussi_od`.

## Tests

```bash
python -m pytest tests        # from territorial_indexes/
```

The tests use synthetic data (window logic, Vodafone allocation, flows, provenienza, energy totals, config files); they do not
replace a run on the real data.
