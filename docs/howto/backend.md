# Overtourism Backend

The backend runs as two separate services:

| Service | Project | Responsibility | Port |
| --- | --- | --- | --- |
| CRUD API | `overtourism-backend` | Manages users, territories, problems, scenarios, and evaluations. | `8000` |
| Model API | `executor-backend` | Lists and evaluates the Fazzon and Molveno models. | `8001` |

The CRUD API calls the Model API over HTTP. Start the Model API first: the CRUD application reads its model registry during startup and its readiness check also depends on the Model API. Expose the CRUD API to clients; keep the Model API on a private network because it does not validate end-user JWTs.

At startup, the CRUD API also asks the optional agent backend to generate a summary for each stored scenario. The agent backend is meant to be active for this step; if it is not present, summary generation is skipped. The same operation can be requested with `POST /api/{territory}/configuration/update/summaries`.

## Configuration

Backend requires the datasets and artefacts to build a Digital Twin model on startup.
The data may be either provided manually (standalone mode) or may be downloaded from the platform.

To enable standalone mode, run the application with the following configuration:

- `DT_OVERTURISM_STANDALONE_MODE` environment variable set to true. Ensure that the datasets are available on startup (see below)

The data is expected to be found at the `data/index_data` folder relative to the working directory. To change the default location,
use `DT_OVERTURISM_INDEX_DATA_PATH` environment variable.

Note that when the backend is run in a non standalone mode, the platform credentials should be available (via environment or CLI setup). It is also
necessary to specify the project from which the data items and artefacts should be downloaded. The name of the project is defined with
`PROJECT_NAME` variable.

### Model API data

The Model API also needs data on startup: each model reads the dataitems produced by its data-preparation
pipeline and has no built-in values, so **the Model API does not start without them**:

| Model | Dataitems | Pipeline |
|-------|-----------|----------|
| Fazzon | `fazzon_presence_stats.parquet` (presence statistics), `fazzon_modal_shift.parquet` (behaviour parameters of the access policy) | `data_preparation/model/fazzon` |
| Molveno | `molveno_presence_stats.parquet` (presence statistics per summer day, gradient boosting), `molveno_presence_stats_monthly.parquet` (monthly comparison statistics), `molveno_weather_freq.parquet` (weather class frequencies) | `data_preparation/model/molveno` |

The same variables as above apply, set on the Model API:

- `DT_OVERTURISM_STANDALONE_MODE=true` (default): the files must already be in the data folder.
- `DT_OVERTURISM_STANDALONE_MODE=false`: the files are downloaded from the platform project `PROJECT_NAME` at
  startup (platform credentials needed, as for the CRUD API), unless they are already in the data folder.

The data folder of each model is `FAZZON_DATA_DIR` / `MOLVENO_DATA_DIR`, or `executor-backend/database/<model>`
if unset. A failed download stops the startup. `POST /models/<model>/download-data` downloads the files again on
demand (see the
[Model API endpoints](../../executor-backend/src/BACKEND_DESIGN.md#layer-5--rest-api)); like the rest of the Model
API it is not authenticated, so the CRUD API should expose it only to administrators.

For a local run, point the variables at the pipeline outputs, e.g.
`FAZZON_DATA_DIR=<repo>/data_preparation/Output/data/final_data/fazzon` and
`MOLVENO_DATA_DIR=<repo>/data_preparation/Output/data/final_data/molveno` after
`python -m model.fazzon.prepare` and `python -m model.molveno.prepare` (run from `data_preparation`).

## Local execution

When running outside standalone mode, the CRUD API and the Model API need credentials to download data from the DigitalHub platform. This platform login is separate from end-user authentication. Install the [DigitalHub CLI](https://github.com/scc-digitalhub/digitalhub-cli/releases), then register and log in:

```bash
./dhcli register https://core.digitalhub-test.smartcommunitylab.it/
./dhcli login dhcore
```

Before starting the services, initialize the database schema with the standalone bootstrap project:

```bash
cd db-bootstrap
uv sync
source .venv/bin/activate
python bootstrap_db.py
```

The script creates the schema and, for SQLite, the database file and parent directories. Pass the configured identifier claim value to also provision the first admin, for example `python bootstrap_db.py YOUR_IDENTIFIER`. It uses `OVERTOURISM_DATABASE` when set; otherwise it targets the same local SQLite file as the CRUD API.

To add or update users in bulk, create a semicolon-delimited CSV with an `email;role` header, then run `python import_users.py users.csv` from `db-bootstrap`. Add an optional `territories` column for comma-separated territory names, such as `email;role;territories`. The importer uses the same `OVERTOURISM_DATABASE` setting and local SQLite default as the schema bootstrap. It normalizes email addresses, inserts missing users, and updates existing users only when their role or territory assignments differ. Existing subject links and active status are preserved. If the `territories` column is omitted, existing assignments are preserved; an empty cell clears assignments. Changing a user to `admin` also clears their territory assignments, matching the API's role behavior. Supported roles are `admin`, `multieditor`, `editor`, and `viewer`; `multieditor` users must have at least one territory.

Run the services in separate terminals from the repository root. Start the Model API first:

```bash
cd executor-backend
uv sync
source .venv/bin/activate
FAZZON_DATA_DIR=../data_preparation/Output/data/final_data/fazzon \
MOLVENO_DATA_DIR=../data_preparation/Output/data/final_data/molveno \
    uvicorn src.api.main:app --host 127.0.0.1 --port 8001 \
        --proxy-headers --log-config uvicorn_log_config.json
```

Wait for the Model API readiness probe to return `200`, then start the CRUD API in another terminal:

```bash
curl --fail http://127.0.0.1:8001/health/ready
```

```bash
cd overtourism-backend
uv sync
source .venv/bin/activate
MODEL_BACKEND_URL=http://127.0.0.1:8001 \
    uvicorn overtourism.overtourism.app:app --host 127.0.0.1 --port 8000 \
        --proxy-headers --log-config uvicorn_log_config.json
```

## Authentication flow

The CRUD API does not provide a login page or authenticate user credentials. An external identity provider authenticates the user and issues a JWT access token. The frontend sends it to the CRUD API as `Authorization: Bearer <token>`; it does not send the user's token to the Model API.

With `AUTH_ENABLED=true`, the CRUD API verifies the token signature using the identity provider's JWKS, checks the issuer and audience, and uses the `sub` claim to identify the user. The user's active status, role, and territory assignments are read from the backend database; territory claims in the token do not grant access. The first admin must be provisioned in that database, and users are linked to their identity-provider subject on their first request using the claim selected by `AUTH_IDENTIFIER_CLAIM` (default: `email`). When using the default, the email must also be verified. See [Authentication setup](auth.md) for admin provisioning and authorization details.

## Environment variables

Set `CORS_ALLOWED_ORIGINS` on both API processes. Set the other variables on the CRUD API unless otherwise noted. Origins are comma-separated full browser origins, including scheme and port when applicable, for example `https://app.example.com,http://localhost:5173`. If unset, the default is `*`; configure an explicit list for deployment. Authentication is disabled by default, so enable and configure it before exposing the CRUD API beyond local development.

| Variable | Required when | Default / purpose |
| --- | --- | --- |
| `CORS_ALLOWED_ORIGINS` | Optional; set on both APIs for deployment. | Comma-separated allowed origins; defaults to `*`. |
| `AUTH_ENABLED` | Always choose explicitly for deployment. | `false`; set to `true` to validate bearer tokens. |
| `AUTH_JWKS_URL` | `AUTH_ENABLED=true` | Identity provider's JWKS endpoint. |
| `AUTH_ISSUER` | `AUTH_ENABLED=true` | Expected JWT issuer. |
| `AUTH_AUDIENCE` | `AUTH_ENABLED=true` | Expected JWT audience. |
| `AUTH_IDENTIFIER_CLAIM` | Optional | JWT claim matched against the database `identifier`; defaults to `email`. |
| `AUTH_ALGORITHMS` | Optional | Accepted JWT algorithms; defaults to `RS256`. |
| `AUTH_LEEWAY_SECONDS` | Optional | JWT clock-skew allowance; defaults to `30` seconds. |
| `MODEL_BACKEND_URL` | When the Model API is not at the local default. | Defaults to `http://localhost:8001`; for containers, use the Model API's private service address. |
| `AGENT_BACKEND_URL` | When the agent backend is not at the local default. | Defaults to `http://localhost:9000`; used to generate scenario summaries. |
| `OVERTOURISM_DATABASE` | Optional locally; set to a persistent database in deployment. | SQLAlchemy database URL; defaults to a SQLite file under `overtourism/overtourism/database/`. |
| `MODEL_BACKEND_USER`, `MODEL_BACKEND_PASSWORD` | Only if an upstream proxy requires HTTP Basic authentication. | The CRUD API sends Basic credentials only when both are set. The Model API itself does not validate them. |
| `FAZZON_DATA_DIR` | Model API; optional. | Folder of the Fazzon model data; defaults to `executor-backend/database/fazzon`. |
| `MOLVENO_DATA_DIR` | Model API; optional. | Folder of the Molveno model data; defaults to `executor-backend/database/molveno`. |

The data-related settings `DT_OVERTURISM_STANDALONE_MODE` and `PROJECT_NAME` are described in [Configuration](#configuration);
they apply to both APIs (see [Model API data](#model-api-data)).

## Health probes

Both services expose unauthenticated operational probes:

- `GET /health/live` returns `200` while the process can answer requests. It does not check external dependencies.
- `GET /health/ready` returns `200` when ready, or `503` otherwise. The CRUD API checks its SQL store with `SELECT 1` and the Model API's readiness. The Model API checks that its model registry is available and consistent.

## With Docker

Build and run the Model API on a private Docker network:

```bash
docker network create overtourism

cd executor-backend
docker build -f Dockerfile.model -t overtourism-model .
docker run -d --rm --name overtourism-model --network overtourism \
    -p 127.0.0.1:8001:8001 \
    -e CORS_ALLOWED_ORIGINS \
    -v "$PWD/../data_preparation/Output/data/final_data/fazzon:/data/fazzon:ro" \
    -e FAZZON_DATA_DIR=/data/fazzon \
    -v "$PWD/../data_preparation/Output/data/final_data/molveno:/data/molveno:ro" \
    -e MOLVENO_DATA_DIR=/data/molveno \
    overtourism-model
```

The Model API needs the model data (see [Model API data](#model-api-data)): mount it as above, or run with
`-e DT_OVERTURISM_STANDALONE_MODE=false -e PROJECT_NAME` and platform credentials to download it at startup.

Wait for `curl --fail http://127.0.0.1:8001/health/ready` to return `200`. In another terminal, from the repository root, build and run the CRUD API. Export `CORS_ALLOWED_ORIGINS`, the OIDC variables, and a persistent `OVERTOURISM_DATABASE` URL before starting it:

```bash
cd overtourism-backend
docker build -f Dockerfile.crud -t overtourism-crud .
docker run -d --rm --name overtourism-crud --network overtourism -p 8000:8000 \
    -e CORS_ALLOWED_ORIGINS \
    -e MODEL_BACKEND_URL=http://overtourism-model:8001 \
    -e AUTH_ENABLED=true \
    -e AUTH_JWKS_URL \
    -e AUTH_ISSUER \
    -e AUTH_AUDIENCE \
    -e OVERTOURISM_DATABASE \
    overtourism-crud
```

`OVERTOURISM_DATABASE` should point to a persistent database for deployment. The model container is not published to the host; the CRUD API reaches it through the Docker network.
