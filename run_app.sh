#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ $# -gt 1 ]]; then
	printf 'Usage: %s [admin-email]\n' "$0" >&2
	exit 2
fi

EXECUTOR_DIR="$ROOT_DIR/executor-backend"
API_DIR="$ROOT_DIR/overtourism-backend"
DB_BOOTSTRAP_DIR="$ROOT_DIR/db-bootstrap"
DB_BOOTSTRAP_PYTHON="$DB_BOOTSTRAP_DIR/.venv/bin/python"
EXECUTOR_UVICORN="$EXECUTOR_DIR/.venv/bin/uvicorn"
API_UVICORN="$API_DIR/.venv/bin/uvicorn"
EXECUTOR_LOG_CONFIG="$EXECUTOR_DIR/uvicorn_log_config.json"
API_LOG_CONFIG="$API_DIR/uvicorn_log_config.json"

for project_dir in "$EXECUTOR_DIR" "$API_DIR"; do
	if [[ ! -f "$project_dir/pyproject.toml" ]]; then
		printf 'Project pyproject.toml not found: %s\n' "$project_dir" >&2
		exit 1
	fi
	if [[ ! -d "$project_dir/.venv" ]]; then
		uv venv "$project_dir/.venv"
	fi
	uv sync --project "$project_dir"
done

if [[ ! -f "$DB_BOOTSTRAP_DIR/pyproject.toml" ]]; then
	printf 'Project pyproject.toml not found: %s\n' "$DB_BOOTSTRAP_DIR/pyproject.toml" >&2
	exit 1
fi
if [[ ! -d "$DB_BOOTSTRAP_DIR/.venv" ]]; then
	uv venv "$DB_BOOTSTRAP_DIR/.venv"
fi
uv sync --project "$DB_BOOTSTRAP_DIR"

export OVERTOURISM_DATABASE="${OVERTOURISM_DATABASE:-sqlite:///$API_DIR/overtourism/overtourism/database/overtourism.sqlite}"
export DT_OVERTURISM_STANDALONE_MODE="${DT_OVERTOURISM_STANDALONE_MODE:-true}"
export AUTH_ENABLED="${AUTH_ENABLED:-true}"
export AUTH_ISSUER="${AUTH_ISSUER:-https://aac.platform.smartcommunitylab.it}"
export AUTH_JWKS_URL="${AUTH_JWKS_URL:-https://aac.platform.smartcommunitylab.it/jwk}"
export AUTH_AUDIENCE="${AUTH_AUDIENCE:-c_50e8e205e30243588df8f1ad9425831a}"
export AUTH_ALGORITHMS="${AUTH_ALGORITHMS:-RS256}"
export AUTH_LEEWAY_SECONDS="${AUTH_LEEWAY_SECONDS:-30}"
export MODEL_BACKEND_URL="${MODEL_BACKEND_URL:-http://localhost:8001}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAIN_PORT="${MAIN_PORT:-8001}"
HEALTH_CHECK_TIMEOUT_SECONDS="${HEALTH_CHECK_TIMEOUT_SECONDS:-60}"

kill -9 $(lsof -t -i:$MAIN_PORT) >/dev/null 2>&1 || true
kill -9 $(lsof -t -i:$PORT) >/dev/null 2>&1 || true

if ! [[ "$HEALTH_CHECK_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
	printf 'HEALTH_CHECK_TIMEOUT_SECONDS must be a positive integer.\n' >&2
	exit 2
fi

if [[ $# -eq 1 ]]; then
	"$DB_BOOTSTRAP_PYTHON" "$DB_BOOTSTRAP_DIR/bootstrap_db.py" "$1"
else
	"$DB_BOOTSTRAP_PYTHON" "$DB_BOOTSTRAP_DIR/bootstrap_db.py"
fi

wait_for_readiness() {
	local service_name="$1"
	local service_port="$2"
	local process_id="$3"
	local python_executable="$4"
	local attempt

	for ((attempt = 0; attempt < HEALTH_CHECK_TIMEOUT_SECONDS; attempt++)); do
		if "$python_executable" -c 'import sys; from urllib.request import urlopen; urlopen(f"http://127.0.0.1:{sys.argv[1]}/health/ready", timeout=1)' "$service_port" >/dev/null 2>&1; then
			return 0
		fi
		if ! kill -0 "$process_id" >/dev/null 2>&1; then
			wait "$process_id" || true
			printf '%s exited before becoming ready.\n' "$service_name" >&2
			return 1
		fi
		sleep 1
	done
	printf '%s did not become ready within %s seconds.\n' \
		"$service_name" "$HEALTH_CHECK_TIMEOUT_SECONDS" >&2
	return 1
}

(
	cd "$EXECUTOR_DIR"
	exec "$EXECUTOR_UVICORN" src.api.main:app \
		--host "$HOST" --port "$MAIN_PORT" \
		--proxy-headers --log-config "$EXECUTOR_LOG_CONFIG"
) &
MAIN_PID=$!
API_PID=""

cleanup() {
	kill "$MAIN_PID" >/dev/null 2>&1 || true
	if [[ -n "$API_PID" ]]; then
		kill "$API_PID" >/dev/null 2>&1 || true
	fi
}

trap cleanup EXIT INT TERM

if ! wait_for_readiness "Executor backend" "$MAIN_PORT" "$MAIN_PID" "$EXECUTOR_DIR/.venv/bin/python"; then
	exit 1
fi

(
	cd "$API_DIR"
	exec "$API_UVICORN" overtourism.overtourism.app:app \
		--host "$HOST" --port "$PORT" \
		--proxy-headers --log-config "$API_LOG_CONFIG"
) &
API_PID=$!

if [[ $# -eq 1 ]]; then
	if ! wait_for_readiness "API backend" "$PORT" "$API_PID" "$API_DIR/.venv/bin/python"; then
		exit 1
	fi
fi

wait "$API_PID"
