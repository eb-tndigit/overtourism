# data_preparation

Data preparation for the Overtourism project: it turns the raw files of the data lake into the datasets used by the
backend. The folder holds two independent pipelines that write in the same `Output/` folder.

```
data_preparation/
├── README.md                  this file
├── pyproject.toml, uv.lock    dependencies (uv), shared by the two pipelines
├── Output/                    everything the pipelines write (not versioned): mapping/, data/{raw_data,normalized,data_processed,final_data}
├── territorial_indexes/       territorial indexes: base build (gen) + yearly update of the phenomena
│   ├── README.md              how the pipeline works, steps, configuration, how to run it
│   ├── METHODOLOGY.md         the steps of gen / update, in short
│   ├── ASSUMPTIONS.md         decisions and assumptions (time window, files, methods)
│   ├── gen_phenomenon_dataframes.py, update_phenomenon_dataframes.py
│   ├── config/                settings.yaml (directories, format, period, platform), updates/<round>.yaml
│   ├── utils/                 the code of the pipeline (steps, adapters, flows, disaggregation, ...)
│   └── tests/
└── model/                     reference datasets of the local models
    ├── common/                code shared by the models: config, io, remote (data lake / platform), sources, weather
    ├── config/settings.yaml   directories and platform of the models
    ├── molveno/               Molveno model (README.md, ASSUMPTIONS.md, molveno.yaml)
    ├── fazzon/                Fazzon model (Lago dei Caprioli; README.md, ASSUMPTIONS.md, fazzon.yaml)
    └── tests/
```

## The two pipelines

| | Territorial indexes | Models |
|---|---|---|
| What | `phen_popolazione`, `phen_strutture`, `phen_presenze`, `phen_consumption`, `phen_flussi_od` per comune (yearly / daily) | reference datasets (daily presences, calendar, weather, ...) of Molveno and Fazzon |
| Period | gen: 2022-01-01 .. 2024-12-31; update: 2025 | defined by each model (see its ASSUMPTIONS.md) |
| Run from | `territorial_indexes/` | `data_preparation/` |
| Entry points | `python -m gen_phenomenon_dataframes`, `python -m update_phenomenon_dataframes --config config/updates/update_2025.yaml` | `python -m model.molveno.prepare`, `python -m model.fazzon.prepare` |
| Settings | `territorial_indexes/config/settings.yaml` | `model/config/settings.yaml` |
| Documentation | `territorial_indexes/README.md`, `METHODOLOGY.md`, `ASSUMPTIONS.md` | `model/<name>/README.md`, `ASSUMPTIONS.md` |

Both follow the same flow: `server (data lake) -> raw_data -> normalized -> processed / reference -> final_data`
(the models write in a sub-folder of their own, e.g. `final_data/molveno`).


## Setup

```bash
uv sync                      # from data_preparation/ (pyproject.toml, uv.lock); extras: cloud, download
```
S3 and platform access need the credentials of the DHCLI in `~/.dhcore.ini` (section `platform.dhcore_env` of the settings).

## Generate the data currently in use

From `territorial_indexes/`, generate the phenomenon data, then the redistribution indicator that reads those phenomena:

```bash
python gen_phenomenon_dataframes.py --all --with-update
```

`--all` runs the base build and then `gen_redistribution_dataframes.py`. Add `--with-update` to append the 2025 update
before redistribution. Without `--all`, `--with-update` runs only the base build and update. The redistribution step
checks that its required phenomenon dataframes exist and reports a `FileNotFoundError` if any are missing.

Add `--skip-download` to `--all` to reuse the existing raw and mapping inputs. In this checkout, `gen_data` is not
defined as a command; the equivalent update-only command is `python -m gen_phenomenon_dataframes --update-only`. The
base build must already exist in the same `Output/` folder when using update-only.

## Tests

```bash
cd territorial_indexes && python -m pytest tests     # time window, config files
cd .. && python -m pytest model/tests                # Molveno and Fazzon pipelines
```
No test needs the data lake or the real data.

## Data

Raw and generated data are not versioned: `Output/` is created by the pipelines. The raw files come from the data lake
(bucket and prefix in the settings); external sources are fetched by their own scripts and uploaded to the data lake.
