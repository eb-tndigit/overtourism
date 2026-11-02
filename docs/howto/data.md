# Data preparation

The [`data_preparation/`](../../data_preparation/) directory contains two independent pipelines that turn raw data into
datasets used by the Overtourism backend. Both write to the shared, unversioned `data_preparation/Output/` directory
and follow the general flow:

`data lake -> raw_data -> normalized -> processed / reference -> final_data`

## Pipelines

| Pipeline | Purpose | Run from | Entry point |
|---|---|---|---|
| Territorial indexes | Generate and update territorial phenomena, including population, structures, presences, consumption, and OD flows by municipality | `data_preparation/territorial_indexes/` | `python -m gen_phenomenon_dataframes` for the base dataset; `python -m update_phenomenon_dataframes --config config/updates/update_2025.yaml` for the 2025 update |
| Local models | Prepare reference datasets, such as daily presences, calendars, and weather, for Molveno and Fazzon (Lago dei Caprioli) | `data_preparation/` | `python -m model.molveno.prepare` or `python -m model.fazzon.prepare` |

The territorial-index base generation covers 2022-01-01 through 2024-12-31; the documented update covers 2025. Model
periods are defined separately for each model.

## Setup and configuration

From `data_preparation/`, install the shared dependencies with:

```bash
uv sync
```

Both pipelines read their settings from separate YAML files: `territorial_indexes/config/settings.yaml` and
`model/config/settings.yaml`. If changing shared configuration such as output paths, the platform project,
or the bucket, keep the corresponding values in both files in sync.

Access to S3 and the platform requires DHCLI credentials in `~/.dhcore.ini`, configured for the `platform.dhcore_env`
section in the relevant settings.

## Tests

Run the territorial-index and model tests from `data_preparation/`:

```bash
cd territorial_indexes && python -m pytest tests
cd .. && python -m pytest model/tests
```

These tests do not require access to the data lake or real data.

## Data and further documentation

Raw and generated data are not versioned. Pipelines create `Output/` as needed; raw inputs are read from the data lake
using the configured bucket and prefix. External sources are fetched by their own download scripts and uploaded to the
data lake.

For pipeline-specific steps, assumptions, and configuration, see the
[territorial-index README](../../data_preparation/territorial_indexes/README.md) and
[model documentation](../../data_preparation/model/).
The [`data_preparation/README.md`](../../data_preparation/README.md) has the full directory overview and explains
which code and settings are intentionally duplicated between the pipelines.
