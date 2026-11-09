# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient
from src.api import health as health_api
from src.api.main import app, create_app
from src.model.common.sustainability_field import (
    OvertourismParameterMeta,
    arrange_frontend_data,
)


def test_layer_3_cors_uses_configured_origins(monkeypatch) -> None:
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        " https://frontend.example,https://admin.example, ",
    )
    configured_app = create_app()

    with TestClient(configured_app) as client:
        frontend_response = client.get(
            "/live", headers={"Origin": "https://frontend.example"}
        )
        admin_response = client.get(
            "/live", headers={"Origin": "https://admin.example"}
        )
        rejected_response = client.get(
            "/live", headers={"Origin": "https://unlisted.example"}
        )

    assert frontend_response.headers["access-control-allow-origin"] == (
        "https://frontend.example"
    )
    assert admin_response.headers["access-control-allow-origin"] == (
        "https://admin.example"
    )
    assert "access-control-allow-origin" not in rejected_response.headers


def test_layer_3_cors_defaults_to_wildcard(monkeypatch) -> None:
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    default_app = create_app()

    with TestClient(default_app) as client:
        response = client.get(
            "/live", headers={"Origin": "https://frontend.example"}
        )

    assert response.headers["access-control-allow-origin"] == (
        "https://frontend.example"
    )


def test_layer_3_health_probes_report_live_and_ready() -> None:
    with TestClient(app) as client:
        live_response = client.get("/live")
        ready_response = client.get("/ready")

    assert live_response.status_code == 200
    assert live_response.json() == {"status": "alive"}
    assert ready_response.status_code == 200
    assert ready_response.json() == {"status": "ready"}


def test_layer_3_is_not_ready_when_model_registry_is_invalid(monkeypatch) -> None:
    monkeypatch.setattr(health_api, "BACKEND_REGISTRY", {})

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_layer_3_health_probes_have_no_health_prefix() -> None:
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 404
        assert client.get("/health/ready").status_code == 404


def test_evaluate_returns_frontend_shape_when_snapshot_is_false(monkeypatch) -> None:
    output = SimpleNamespace(
        field=[[0.1, 0.2]],
        samples_x=[12.0],
        samples_y=[34.0],
        uncertainty=[0.5],
        uncertainty_by_constraint={"parking": [0.6]},
        usage=[70],
        usage_by_constraint={"parking": [80]},
        usage_uncertainty=[0.1],
        usage_uncertainty_by_constraint={"parking": [0.2]},
        kpis={"critical constraint": {"name": "parking"}},
        x_max=100.0,
        y_max=200.0,
        capacity_mean=50.0,
        capacity_mean_by_constraint={"parking": 40.0},
        constraint_curves={"parking": [[1.0], [2.0]]},
    )
    output.to_snapshot = lambda: {
        "samples_x": output.samples_x,
        "samples_y": output.samples_y,
        "uncertainty": output.uncertainty,
        "uncertainty_by_constraint": output.uncertainty_by_constraint,
        "usage": output.usage,
        "usage_by_constraint": output.usage_by_constraint,
        "usage_uncertainty": output.usage_uncertainty,
        "usage_uncertainty_by_constraint": output.usage_uncertainty_by_constraint,
        "kpis": output.kpis,
        "x_max": output.x_max,
        "y_max": output.y_max,
        "capacity_mean": output.capacity_mean,
        "capacity_mean_by_constraint": output.capacity_mean_by_constraint,
        "constraint_curves": output.constraint_curves,
    }

    class FakeBackend:
        def evaluate(self, param_overrides):
            return output

        def arrange_data(self, data):
            return arrange_frontend_data(data)

    monkeypatch.setattr("src.api.routes.get_backend", lambda model_key: FakeBackend())

    with TestClient(app) as client:
        response = client.post(
            "/models/molveno/evaluate?as_snapshot=false",
            json={"param_overrides": {}},
        )

    assert response.status_code == 200
    assert response.json() == {
        "points": {
            "uncertainty": [
                {
                    "tourists": 12.0,
                    "excursionists": 34.0,
                    "index": 0.5,
                    "usage": 70,
                    "usage_uncertainty": 0.1,
                }
            ],
            "uncertainty_by_constraint": {
                "parking": [
                    {
                        "tourists": 12.0,
                        "excursionists": 34.0,
                        "index": 0.6,
                        "usage": 80,
                        "usage_uncertainty": 0.2,
                    }
                ]
            },
        },
        "kpis": {"critical constraint": {"name": "parking"}},
        "x_max": 100.0,
        "y_max": 200.0,
        "capacity_mean": 50.0,
        "capacity_mean_by_constraint": {"parking": 40.0},
        "constraint_curves": {"parking": [[1.0], [2.0]]},
    }


def test_evaluate_returns_snapshot_when_snapshot_is_true(monkeypatch) -> None:
    output = SimpleNamespace()

    class FakeBackend:
        def evaluate(self, param_overrides):
            return output

    monkeypatch.setattr("src.api.routes.get_backend", lambda model_key: FakeBackend())
    monkeypatch.setattr(
        "src.api.routes.EvaluateResponse.from_output",
        lambda data: {"field": [[0.1]]},
    )

    with TestClient(app) as client:
        response = client.post("/models/molveno/evaluate", json={})

    assert response.status_code == 200
    assert response.json() == {"field": [[0.1]]}


def test_get_schema_builds_full_metadata_from_backend_minimal_set(monkeypatch) -> None:
    """`get_schema` builds the full presentation metadata from the
    backend's minimal set (`_build_metadata`).

    Uses distinctive fake `mapper`/axis/field values so the assertions
    prove the values genuinely flow from `minimal`.
    """

    class FakeBackend:
        def schema(self):
            return {
                "metadata": {
                    "mapper": {
                        "parking": "FAKE-PARK",
                        "beach": "FAKE-BEACH",
                    },
                    "x_axis_name": "FAKE-X",
                    "y_axis_name": "FAKE-Y",
                    "x_field": "fake_x_field",
                    "y_field": "fake_y_field",
                },
                "indexes": [{"name": "parking", "kind": "scalar"}],
            }

    monkeypatch.setattr("src.api.routes.get_backend", lambda model_key: FakeBackend())

    with TestClient(app) as client:
        response = client.get("/models/molveno/schema")

    assert response.status_code == 200
    body = response.json()

    # Backend-owned entries taken verbatim; "default" is injected as a standard.
    assert body["metadata"]["mapper"] == {
        "default": "Tutti",
        "parking": "FAKE-PARK",
        "beach": "FAKE-BEACH",
    }
    assert body["metadata"]["plot_mapper"]["bidimensional"] == {
        "x": {"label": "FAKE-X", "field": "fake_x_field"},
        "y": {"label": "FAKE-Y", "field": "fake_y_field"},
    }

    # Derived mechanically from `mapper` (one per non-"default" key).
    assert body["metadata"]["kpi_mapper"]["constraint level parking"] == (
        "Sostenibilità FAKE-PARK"
    )
    assert body["metadata"]["kpi_mapper"]["constraint level beach"] == (
        "Sostenibilità FAKE-BEACH"
    )
    assert "constraint level default" not in body["metadata"]["kpi_mapper"]

    # Universal, model-independent constants.
    assert body["metadata"]["color_map"][0] == [0.0, "rgb(5, 102, 8)"]
    assert body["metadata"]["kpi_mapper"]["title"] == "Indici"
    assert body["metadata"]["kpi_mapper"]["critical constraint"] == "Vincolo Critico"
    assert body["metadata"]["plot_mapper"]["monodimensional"] == {
        "x": {"label": "Giorni (ordinati per utilizzo)"},
        "y": {"label": "Livello di utilizzo della destinazione", "field": "usage"},
    }

    assert body["indexes"][0]["name"] == "parking"
    assert body["indexes"][0]["kind"] == "scalar"


def test_index_diffs_returns_changes_for_a_batch_of_scenarios(monkeypatch) -> None:
    class FakeBackend:
        def schema(self):
            return {
                "indexes": [
                    OvertourismParameterMeta(
                        name="car mode share",
                        kind="scalar",
                        default=0.69,
                    ),
                ]
            }

    monkeypatch.setattr("src.api.routes.get_backend", lambda model_key: FakeBackend())

    with TestClient(app) as client:
        response = client.post(
            "/models/molveno/index-diffs",
            json={
                "param_overrides_by_scenario": {
                    "draft-1": {"car mode share": 0.8},
                    "draft-2": {"car mode share": 0.9},
                    "draft-3": {"car mode share": 0.69},
                }
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "index_diffs_by_scenario": {
            "draft-1": {"car mode share": "0.69 -> 0.8"},
            "draft-2": {"car mode share": "0.69 -> 0.9"},
            "draft-3": {},
        }
    }


def test_get_fazzon_schema_returns_fazzon_frontend_metadata() -> None:
    with TestClient(app) as client:
        response = client.get("/models/fazzon/schema")

    assert response.status_code == 200
    assert response.json()["metadata"] == {
        "mapper": {
            "default": "Tutti",
            "parking": "Parcheggi",
            "road": "Viabilità",
            "food": "Ristorazione",
        },
        "color_map": [
            [0.0, "rgb(5, 102, 8)"],
            [0.05, "rgb(100, 180, 90)"],
            [0.2, "rgb(180, 230, 170)"],
            [0.4, "rgb(230, 250, 225)"],
            [0.5, "yellow"],
            [0.6, "rgb(255, 242, 242)"],
            [0.8, "rgb(242, 204, 204)"],
            [0.95, "rgb(204, 76, 76)"],
            [1.0, "rgb(180, 4, 38)"],
        ],
        "kpi_mapper": {
            "title": "Indici",
            "area": "Area Totale",
            "sustainability_level": "Sostenibilità complessiva",
            "constraint level parking": "Sostenibilità Parcheggi",
            "constraint level road": "Sostenibilità Viabilità",
            "constraint level food": "Sostenibilità Ristorazione",
            "critical constraint": "Vincolo Critico",
            "dropouts mean": "Visitatori potenziali che non arrivano: media stagionale (persone/giorno)",
            "dropouts peak": "Visitatori potenziali che non arrivano: giorni di picco (persone/giorno)",
        },
        "plot_mapper": {
            "monodimensional": {
                "x": {"label": "Giorni (ordinati per utilizzo)"},
                "y": {
                    "label": "Livello di utilizzo della destinazione",
                    "field": "usage",
                },
            },
            "bidimensional": {
                "x": {
                    "label": "Visitatori arrivati in auto / giorno",
                    "field": "visitors_car",
                },
                "y": {
                    "label": "Visitatori arrivati in totale / giorno",
                    "field": "visitors_total",
                },
            },
        },
    }


def test_download_data_refreshes_backend(monkeypatch, tmp_path) -> None:
    from src.api import registry

    calls = []

    def fake_download(overwrite: bool = False):
        calls.append(overwrite)
        return tmp_path

    monkeypatch.setitem(registry.DATA_DOWNLOADERS, "fazzon", fake_download)
    before = registry.get_backend("fazzon")
    response = TestClient(app).post("/models/fazzon/download-data")

    assert response.status_code == 200
    assert response.json()["data_dir"] == str(tmp_path)
    assert calls == [True]
    assert registry.get_backend("fazzon") is not before


def test_download_data_failure_returns_502(monkeypatch) -> None:
    from src.api import registry

    def failing_download(overwrite: bool = False):
        raise RuntimeError("platform unreachable")

    monkeypatch.setitem(registry.DATA_DOWNLOADERS, "fazzon", failing_download)
    response = TestClient(app).post("/models/fazzon/download-data")

    assert response.status_code == 502
    assert "platform unreachable" in response.json()["detail"]


def test_download_data_unknown_or_without_data(monkeypatch) -> None:
    from src.api import registry

    client = TestClient(app)
    assert client.post("/models/nope/download-data").status_code == 404
    monkeypatch.delitem(registry.DATA_DOWNLOADERS, "molveno")
    assert client.post("/models/molveno/download-data").status_code == 404


def test_startup_fails_without_fazzon_data(monkeypatch, tmp_path) -> None:
    import pytest

    monkeypatch.setenv("FAZZON_DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("DT_OVERTURISM_STANDALONE_MODE", "true")
    with pytest.raises(FileNotFoundError), TestClient(create_app()):
        pass
