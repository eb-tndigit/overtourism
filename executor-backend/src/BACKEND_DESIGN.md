<!-- SPDX-License-Identifier: Apache-2.0 -->

# Overtourism Backend Architecture

This document describes the layered architecture of the `src`
package — how the computation layer for the Fazzon and Molveno overtourism
models is organized, why it is split the way it is, and how it fits together
with the rest of the system (application backend, production frontend).

**Location note**: everything this document describes (Layers 2–3 and 5)
lives under `overtourism/layer_3/` — e.g. `src.model.common`,
`src.api`, `src.dt_studio`. Layer 4
(application backend) lives outside `layer_3/` entirely, in
`overtourism/dt_manager/` plus its own API surface in `overtourism/backend/`
— see the Layer 4 note below.

**Status**: Layers 1–3, a REST API (Layer 5), and Streamlit dev/test tooling
(Layer 5) are implemented for Fazzon and Molveno. Layer 4 (application
backend) is implemented as well, in `overtourism/dt_manager/` plus its own
API surface in `overtourism/backend/` — documented at a high level here
(§Layer 4), with its own code as the source of truth for details. A
production UI (Layer 5) is implemented too, as a separate Angular
single-page application — see [Next steps](#next-steps).

---

## Layer architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  LAYER 5 — Frontend                                                      │
│  REST API · Streamlit (dev/test) · production UI (Angular SPA)           │
│  Entry point lives here. Presentation only.                              │
├──────────────────────────────────────────────────────────────────────────┤
│  LAYER 4 — Application / tool backend                                    │
│  overtourism.dt_manager + overtourism.backend                            │
│  Scenario catalogue · persistence · user storage · i18n                  │
├──────────────────────────────────────────────────────────────────────────┤
│  LAYER 3 — Computation backend  (model-specific, plain classes)          │
│  FazzonBackend · MolvenoBackend — no shared base class                   │
├──────────────────────────────────────────────────────────────────────────┤
│  LAYER 2 — layer_3.model.common                                          │
│  (overtourism-model-family code, not domain-agnostic)                    │
│  OvertourismParameterMeta · SustainabilityFieldOutput ·                  │
│  OvertourismEvaluationConfig · shared sustainability-field math          │
├──────────────────────────────────────────────────────────────────────────┤
│  LAYER 1 — civic_digital_twins library  (external, read-only)            │
│  Model/@define/@inputs/@outputs · ModelOutput · EvaluationConfig ·       │
│  ParameterMeta · build_scenario() · Scenario · CrossProductEnsemble ·    │
│  Evaluation                                                              │
└──────────────────────────────────────────────────────────────────────────┘
```

| Layer | Responsibility |
|---|---|
| 1 — `civic_digital_twins` | The external digital-twin modeling library. Read-only from this repo's point of view. |
| 2 — `src.model.common` | Code shared by the overtourism models specifically (currently Fazzon and Molveno) — array math and types that only make sense for this "2D presence-vs-presence sustainability field" model family. |
| 3 — Computation backend | One class per model (`FazzonBackend`, `MolvenoBackend`): builds the model, exposes its parameter schema, and evaluates scenarios. Frontend-agnostic — no Streamlit, FastAPI, or other UI/transport code. |
| 4 — Application backend | Implemented outside `layer_3/`: `overtourism.dt_manager` (scenario/problem/proposal/session managers, persistence stores) plus its own API surface in `overtourism.backend` (auth, `api/v2/*` routers) — not detailed further in this document; its own code is the source of truth. See [Next steps](#next-steps). |
| 5 — Frontend | Presentation only. A REST API (`src.api`) and Streamlit apps (in-process and HTTP-driven) for development and manual testing, plus a production UI (Angular SPA, separate repo) for end users. |

---

## Layer 1 — `civic_digital_twins` (external dependency)

Pinned in `pyproject.toml` as `civic-digital-twins>=0.12.0,<0.13` (any
0.12.x patch release, no 0.13+). Both `fazzon_model.py`
and `molveno_model.py` are written against the library's `@define`/`@inputs`/
`@outputs`/`compute()` model-definition API (`civic_digital_twins.dt_model.Model`
and friends) — every concern sub-model declares its inputs/outputs as typed
dataclasses and implements a single `compute()` method. `SustainabilityFieldOutput`
(§Layer 2) subclasses the library's `ModelOutput`.

This API surface only exists from `civic-digital-twins>=0.10.0` — earlier
releases (`<0.6.0`) instead exposed an `AbstractModel`-based API that has since
been removed. There is no published version that supports both; if any other
code in this repository needs to target the older API, it cannot share a
dependency pin with `src.model.*`.

### Parameter schema and scenario construction

`ParameterMeta`, `build_scenario()`, and the `ensemble_seed`/
`n_samples_per_combo` fields of `EvaluationConfig` are domain-agnostic
utilities — useful to any `civic_digital_twins` model, not just the
overtourism ones. They were originally staged in this repo
(`src.cdt_ext.runner_ext`) and have been upstreamed into the
library as of `civic-digital-twins` 0.12.0; the local copies were dropped and
everything now imports them from `civic_digital_twins.dt_model`.

### `ParameterMeta`

A typed schema for describing one model parameter — the minimum needed to
**validate or reconstruct** a submitted override, not to render a widget:

```python
@dataclass
class ParameterMeta:
    name: str  # matches index.name exactly; the boundary key
    kind: str  # "scalar" | "categorical" | "distribution"
    distribution_family: str | None = None
    distribution_fixed_params: dict[str, Any] | None = None
    support: list[str] = field(default_factory=list)
    default: float | None = None
    default_category: str | None = None
```

Presentation fields (`label`, `description`, `unit`, UI ranges, the display
labels of categorical values `support_labels`, ...) deliberately
do **not** belong here — `build_scenario()` never reads them, they only affect
widget rendering. A model-family that needs them extends via plain dataclass
inheritance — see `OvertourismParameterMeta` in `src.model.common`
(§Layer 2), the only subclass today.

`name` is kept on the object (not left to a dict key alone) so that any
`list[ParameterMeta]` view — e.g. a `Backend.parameter_schema()` return value,
or the FastAPI `/schema` JSON array (§Layer 5 — REST API) — is
self-describing. Such a list is always *derived* (`list(schema.values())`),
never hand-authored separately.

### `EvaluationConfig`

```python
@dataclass
class EvaluationConfig:
    ensemble_size: int
    ensemble_seed: int | None = None
    n_samples_per_combo: int = 1
```

Makes `CrossProductEnsemble`'s reproducibility knobs (`ensemble_seed`,
`n_samples_per_combo`) explicit, injectable config instead of hardcoded
constants. (Previously staged here as an `EnsembleEvaluationConfig`
subclass; deleted outright once the fields landed on the library's own
`EvaluationConfig`.)

### `build_scenario()`

```python
def build_scenario(
    model: Model | ModelVariant,
    param_overrides: Mapping[str, Any],
    index_map: Mapping[str, GenericIndex],     # index.name → Index object
    spec_map: Mapping[str, ParameterMeta],     # index.name → ParameterMeta
    parameter_axes: list[GenericIndex] | None = None,
) -> Scenario:
```

Resolves a string-keyed `param_overrides` dict (the shape a frontend submits)
into a `Scenario`, dispatching on `spec_map[name].kind`:

- **scalar** (`float`) — passed directly as the index override
- **distribution** (`(lo, hi)` tuple) — reconstructed as a frozen `scipy.stats`
  distribution via `ParameterMeta.distribution_family`/`.distribution_fixed_params`,
  using the convention `loc = lo`, `scale = hi - lo`. This convention belongs to
  the frontends that produce `(lo, hi)` pairs, not to `DistributionIndex` itself.
- **categorical** (`str`) — passed directly as the index override
- keys absent from `param_overrides`, or not present in `index_map`/`spec_map`,
  are left unoverridden — model defaults apply

`parameter_axes` is optional — a model with no presence-variable grid to sweep
passes `None`.

There is no `ModelBackend`/`ParametricModelBackend` base class: once
`ParameterMeta` and `build_scenario()` exist, evaluating from string overrides
is a one-line call (`evaluator.evaluate(build_scenario(...), config)`), so a
wrapping ABC would add no capability. Layer 3 backends (§Layer 3) are plain
concrete classes.

---

## Layer 2 — `src.model.common` (overtourism-model-family code)

**Location**: `overtourism/layer_3/model/common/sustainability_field.py`,
importable as `src.model.common.sustainability_field`.

Code shared by the overtourism models — currently Fazzon and Molveno. Explicitly
**not** a candidate for upstreaming into `civic_digital_twins`: everything
here (field math, output shape, config fields like `sample_seed`/
`target_presence_samples`/`category`/`step`) only means something for this
specific "2D presence-vs-presence sustainability field" model family, not for
`civic_digital_twins` models in general.

### `OvertourismParameterMeta`

`ParameterMeta` (§Layer 1) extended with presentation fields via plain dataclass
inheritance:

```python
@dataclass
class OvertourismParameterMeta(ParameterMeta):
    label: str = ""
    description: str = ""
    unit: str = ""
    category: str = ""
    step: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    default_range: tuple[float, float] | None = None
    support_labels: dict[str, str] = field(default_factory=dict)
```

`support_labels` maps categorical values to display labels (e.g.
`{"saturday": "sabato"}`, shared labels in `src.model.common.labels`): the
dashboards and the index-diff labels show the label, the API keeps the value;
a value without a label is shown as it is.

`build_scenario()` only touches the base `ParameterMeta` fields, so it works
unmodified against this (or any other) subclass.

### `OvertourismEvaluationConfig`

```python
@dataclass
class OvertourismEvaluationConfig(EvaluationConfig):
    sample_seed: int | None = None
    target_presence_samples: int = 2000
    confidence: float = 0.8
```

Adds the presence-sampling and confidence-interval parameters that configure
`sample_across()`-based scatter-overlay sampling and the sustainability-index
confidence interval — concepts specific to this model family.

### Shared sustainability-field math

Pure array functions (field + axes + presences + confidence in; tuple/dict
out), with no model-specific coupling, used by both `FazzonBackend` and
`MolvenoBackend`:

- `compute_sustainability_field` — builds the raw `field`/`field_elements`
  arrays, plus `usage_fields` (per-constraint weighted-average *usage
  value*, as opposed to `field_elements`'s probability of staying under
  capacity) and `capacity_distributions` (`{"loc": mean, "scale": std}` per
  constraint), from an `EvaluationResult` and a model's `constraints` list.
  Takes `constraints: Iterable[Any]` (structurally typed — `.name`/`.usage`/
  `.capacity` — rather than a shared `Constraint` class, since each model
  keeps its own) and the two axis indexes directly. `usage_fields` is
  weighted-averaged over the categorical ensemble the same way
  `field_elements` is (`tensordot` against `result.weights`) — the
  replacement for the older `EvaluationResult.marginalize()` call, which no
  longer exists in the pinned `civic_digital_twins` version.
- `compute_sustainable_area`
- `compute_sustainability_index_with_ci`
- `compute_sustainability_by_constraint`
- `compute_modal_lines` — per-constraint modal line via orthogonal regression
  (first principal component)
- `_usage_uncertainty_from_params` — per-sample exceedance probability from
  a constraint's capacity distribution (normal approximation). Handles
  `scale == 0` (a deterministic, non-distribution capacity — e.g. Fazzon's
  `road`/`food`) as the exact step-function limit rather than the
  `scipy.stats.norm(scale=0)` NaN that would otherwise reach the API
  response.
- `arrange_frontend_data` — reshapes a `SustainabilityFieldOutput.to_snapshot()`
  into the legacy points-based structure the current frontend consumes
  (per-sample dicts instead of parallel arrays). An explicit bridge, not a
  permanent shape — see §Layer 5 — REST API's `as_snapshot=false` note.

### `SustainabilityFieldOutput`

A `ModelOutput` subclass (Layer 1) defining the unified output schema for both
models — no per-model output subclass is needed since, with the math above
shared, neither model has a field to add beyond the common set:

```python
@dataclass(eq=False)
class SustainabilityFieldOutput(ModelOutput):
    field: np.ndarray  # shape (N_x, N_y), values in [0, 1]
    field_elements: dict[str, np.ndarray]  # per-constraint fields, same shape
    x_values: np.ndarray  # 1-D, shape (N_x,)
    y_values: np.ndarray  # 1-D, shape (N_y,)
    samples_x: list[float]  # presence samples, for scatter overlay
    samples_y: list[float]
    usage_fields: dict[str, np.ndarray]  # per-constraint usage grids
    capacity_distributions: dict[str, dict]  # {name: {"loc", "scale"}}
    confidence: float = 0.8

    # Derived, lazily via functools.cached_property:
    #   sustainable_area, sustainability_index, sustainability_by_constraint,
    #   modal_lines, x_max, y_max, uncertainty, uncertainty_by_constraint,
    #   usage, usage_by_constraint, capacity_mean, capacity_mean_by_constraint,
    #   usage_uncertainty, usage_uncertainty_by_constraint, kpis,
    #   constraint_curves
```

The `usage`/`uncertainty`/`capacity_mean`/`kpis` family and `x_max`/`y_max`/
`constraint_curves` were restored from the pre-reorg `MolvenoEvaluator`/
`MolvenoOutput` (removed early on as "duplicating `MolvenoBackend`'s logic")
after a Layer 4 integration surfaced that the removal had also dropped real,
non-duplicate capability those fields provided. `kpis` stays English-only —
locale translation was already a separate presentation-layer post-processing
step in the pre-reorg code, not part of this computation.

`kpis` reports **sustainability**, not overtourism: `sustainability_level`
(the overall `sustainability_index`), `critical constraint` (the constraint
with the lowest sustainability), and one `constraint level <name>` entry per
constraint (its `sustainability_by_constraint` value) — each as `{"level",
"confidence"}` percentages. (Earlier versions reported the complement,
`1 - sustainability`, under an `overtourism_level` key.)

A model can add its own KPIs (`SustainabilityFieldOutput.extra_kpis`), reported
in `kpis` next to the standard ones. A `kpis` entry has one of two formats,
told apart by its fields (not by its key):

- `{"level", "confidence"}` (plus `"name"` for `critical constraint`): a
  percentage with its confidence half-width — the sustainability KPIs above;
- `{"value", "share"}`: a quantity, with its share in percent (0–100;
  `share` optional) — e.g. Fazzon's `dropouts mean` and `dropouts peak`
  (potential visitors who do not arrive, per day, and their share of the
  potential visitors). The dashboards show them as `value (share%)`.

The display names of all KPIs are in the schema's `kpi_mapper` (model-specific
ones from the backend's `kpi_labels`).

`to_snapshot()` returns a JSON-serialisable dict (base64-encoded array
fields) including all of the above — see §Layer 5 — REST API for why the
live `/evaluate` response uses a different, plain-JSON shape instead.

Axis labels (`x_axis_name`/`y_axis_name`) are deliberately **not** fields
here: they're evaluation-invariant (constant per model, not per
`param_overrides`) and purely presentational, so they don't belong in a
computation result. They live on the backend instead (`X_AXIS_NAME`/
`Y_AXIS_NAME`, §Layer 3) — see §Layer 5 for how each frontend style sources
them.

---

## Layer 3 — Computation backends

**Location**: `overtourism/layer_3/model/fazzon/` and `overtourism/layer_3/model/molveno/`.

Each model package contains:

| File | Contents |
|---|---|
| `<model>_model.py` | The model definition itself — `civic_digital_twins` sub-models, the root `Model` class, `default_inputs()`. |
| `<model>_backend.py` | `FazzonBackend`/`MolvenoBackend` — the frontend-agnostic evaluation entry point. |
| `<model>_presence_stats.py` | Presence distributions by context, built from the statistics of the model's data-preparation pipeline (`<model>_data.py`: location, download, startup check). |
| `fazzon_scenarios.py` (Fazzon only) | Catalogue of named what-if scenarios, expressed against live `Index` objects. |

Each backend also carries `MAPPER`/`X_AXIS_NAME`/`Y_AXIS_NAME`/`X_FIELD`/
`Y_FIELD` class constants — the backend's own minimal, hardcoded presentation
metadata (same "hand-authored, single source of truth" pattern as `_schema`).
`MAPPER` omits the `"default"` key — that entry is a presentation-layer
standard added uniformly by the API (see §Layer 5 — REST API), not
model-specific content. These five constants are now the *only* per-model
presentation metadata `schema()` (below) returns; a former `schema_metadata.py`
file per model (subsystem display names, a color scale, KPI/plot label maps)
has been removed — diffing the two files field by field showed everything in
them beyond `mapper`/axis labels/axis fields was either identical across
both models or mechanically derivable from `mapper`, so
`src.api.routes._build_metadata` now generates the full
`/schema` presentation payload from this backend-owned minimal set alone
(§Layer 5 — REST API).

`FazzonBackend` and `MolvenoBackend` are plain, self-contained classes with no
shared base class (see the `ModelBackend` rationale in §Layer 1) and no
runtime dependency on each other. Each backend's constructor builds its model,
an `index.name → Index` map, and an `OvertourismEvaluationConfig` with fixed
seeds and sample counts — seeds are a compute-quality concern, so they live
here, not in any frontend.

Public contract (identical shape for both):

```python
class <Model>Backend:
    def __init__(self) -> None: ...

    @property
    def model(self) -> <Model>Model:
        """The live model instance — needed by frontends that resolve
        predefined what-if scenarios against live Index objects."""

    def parameter_schema(self) -> list[OvertourismParameterMeta]:
        """Ordered, self-describing parameter schema."""

    def schema(self) -> dict[str, Any]:
        """`{"metadata": {"mapper", "x_axis_name", "y_axis_name", "x_field",
        "y_field"}, "indexes": self.parameter_schema()}` — the backend's
        minimal presentation metadata; `src.api.routes
        ._build_metadata` expands it into the full `/schema` response
        (§Layer 5)."""

    def evaluate(self, param_overrides: dict[str, Any]) -> SustainabilityFieldOutput:
        """Evaluate the model under string-keyed parameter overrides."""

    def arrange_data(self, output: SustainabilityFieldOutput) -> dict[str, Any]:
        """`arrange_frontend_data(output)` (§Layer 2) — legacy frontend shape."""
```

`evaluate()` builds a `Scenario` via `build_scenario()` (§Layer 1), runs a
seeded `CrossProductEnsemble`/`Evaluation` over the model's two presence-variable
axes, and returns a `SustainabilityFieldOutput` computed via the shared field
math (§Layer 2). Grid-resolution parameters (axis max/sample-count) are
backend constructor arguments, not evaluation config — they are structural
choices about the axis grid, not per-evaluation quality knobs.

---

## Layer 5 — REST API

**Location**: `overtourism/layer_3/api/`, run with:

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8082 \
  --proxy-headers --log-config uvicorn_log_config.json
```

The service listens on one port, `8082`, for both the models API and the health probes. The container
image (`Dockerfile.model`) exposes only `8082` and starts uvicorn on it with `uvicorn_log_config.json`.

A REST layer over the Layer 3 backends. Depends only on
`src.model.*` — never on `src.dt_studio.*` — so it can be
deployed standalone. `FazzonBackend` and `MolvenoBackend` share an identical
contract by convention (§Layer 3), so one generic route set parameterized by
`{model_key}` covers both models instead of one router per model;
`src.api.registry` holds the `model_key -> Backend` mapping.

| Endpoint | Description |
|---|---|
| `GET /models` | Registry catalogue: `[{key, title}, ...]`. |
| `GET /models/{model_key}/schema` | `ModelSchema` — full presentation metadata built from the backend's minimal set, plus `indexes`. |
| `POST /models/{model_key}/evaluate?as_snapshot=true` | `{param_overrides: {...}}` → evaluation result. |
| `POST /models/{model_key}/download-data` | Downloads the model's data files from the platform again (Fazzon and Molveno; 404 for models without downloadable data) and drops the cached backend, so the next request uses the new data. `200 {message, data_dir}`; `502` if the download fails (the current files stay in place). Not authenticated: the wrapping layers must restrict it to administrators. |

Health probes (`src.api.health`) are served by the same app on port `8082`, at the root path. They have no
`/health` prefix: `/health/live` and `/health/ready` return `404`.

| Endpoint | Description |
|---|---|
| `GET /live` | `200 {"status": "alive"}` while the process answers. It checks no external dependency. |
| `GET /ready` | `200 {"status": "ready"}` when the model registry is consistent; otherwise `503 {"status": "not_ready"}`. |

Uvicorn opens port `8082` only after the startup below has completed, so both probes get a refused
connection while the model data downloads. In Kubernetes, point the liveness and readiness probes at
`8082` (`/live` and `/ready`) and give the startup time to finish, with a `startupProbe` or a liveness
`initialDelaySeconds` longer than the download; otherwise the liveness probe restarts the container during
the download.

Models with data files (`src.api.registry.DATA_STARTUP`: Fazzon, Molveno) are checked at startup (FastAPI
lifespan): unless `DT_OVERTURISM_STANDALONE_MODE=true` (the default) the files are downloaded from the platform
project `PROJECT_NAME`; missing files or a failed download stop the startup. See
`src.model.common.model_data` (shared by `fazzon_data` and `molveno_data`) and `docs/howto/backend.md`
(Model API data).

`GET /schema` no longer returns `backend.schema()` unchanged.
`src.api.routes._build_metadata()` takes the backend's
minimal metadata (`mapper`, `x_axis_name`, `y_axis_name`, `x_field`,
`y_field` — §Layer 3) and builds the full presentation payload from that
alone:

- injects the standard `"default"` mapper entry (`"Tutti"`) — a
  presentation convention, not model-specific content, so it doesn't belong
  on the backend;
- derives `kpi_mapper`'s per-constraint `"constraint level X"` entries
  mechanically, one per non-`"default"` `mapper` key, following the
  `"Sostenibilità {label}"` template (`"sustainability_level"` is labelled
  `"Sostenibilità complessiva"`), matching the sustainability values
  `SustainabilityFieldOutput.kpis` reports (§Layer 2);
- hardcodes the handful of presentation constants that never varied across
  models (`color_map`, `kpi_mapper`'s static keys, `plot_mapper.monodimensional`).

This replaced the former per-model `schema_metadata.py` files entirely
(§Layer 3): diffing them field by field showed no genuinely per-model
content left in them beyond what the backend already owns — see the Layer 4
note under [Next steps](#layer-4--application-backend) for more.

`/evaluate` returns `src.api.schemas.EvaluateResponse`
(`EvaluateResponse.from_output()`) by default, **not**
`SustainabilityFieldOutput.to_snapshot()` — a plain-JSON shape (array fields
flattened via `.tolist()`, no base64) covering `SustainabilityFieldOutput`'s
field set, including the derived usage/uncertainty/KPI set. `x_axis_name`/
`y_axis_name` are absent from both `EvaluateResponse` and
`SustainabilityFieldOutput` itself (§Layer 2) — they're evaluation-invariant
and purely presentational, so they live only on the backend (`X_AXIS_NAME`/
`Y_AXIS_NAME`, §Layer 3) and reach the wire via `/schema`'s
`metadata.plot_mapper.bidimensional.{x,y}.label` (via `_build_metadata`,
above). `HttpOvertourismAdapter` (§Layer 5 — Streamlit) reads axis labels —
and subsystem display names, from `metadata.mapper` — from there.

Returning `EvaluateResponse` instead of `to_snapshot()` verbatim was a
deliberate reversal of an earlier decision: that was reasonable when the
only known consumer was code in this repo (`HttpOvertourismAdapter`, willing
to decode base64); it stopped being reasonable once Layer 4 turned out to be
a real external consumer needing plain JSON matching the shape of the
model's previous (pre-reorg) evaluator output. `to_snapshot()` itself is
unchanged and keeps its original purpose — Layer 4 persistence/resume,
where base64 is the right tradeoff — `/evaluate` just no longer uses it by
default.

`?as_snapshot=false` instead returns `backend.arrange_data(output)` —
`arrange_frontend_data()`'s legacy points-based shape (§Layer 2), for the
current frontend to consume directly without a rewrite. An explicit,
acknowledged bridge, expected to be removed once that chain is streamlined;
because the two branches return incompatible shapes behind one query
parameter, the route's `response_model` is `Any`, so neither shape is
OpenAPI-validated today.

Layer 4 (`overtourism.dt_manager` + `overtourism.backend`) now exists (see
the Status note at the top of this document), but as an entirely separate
API surface (`overtourism.backend.api.v2.*`), not merged into these routes
— so `src.api` itself still has no `/scenarios` or
`/results/{id}` route.

---

## Layer 5 — Streamlit dev/test tooling

**Location**: `src/dt_studio/`, isolated from the computation backend
— nothing under `src.model.*` imports Streamlit, and the generic
shell (`dashboard/adapter.py`, `dashboard/app.py`) doesn't import
`src.model.*` either, in either direction (see below).

- `src/dt_studio/dashboard/` — the generic, model-agnostic Streamlit
  shell: `adapter.py` defines the `OvertourismAdapter` ABC, the `PlotData`/
  `ScenarioDef` data types, and a `ParameterSpec` `Protocol` describing the
  widget-metadata attributes a spec needs (`kind`, `label`, `min_value`, ...)
  — structural, not a concrete class, so this module never imports
  `src.model.common` just for a type hint. `app.py`'s `run_dashboard()`
  renders the sidebar widgets, field plot, KPI panel, and scenario selector
  against that `Protocol`, for any adapter.
- `src/dt_studio/fazzon_dashboard.py` / `molveno_dashboard.py` — the
  in-process, per-model entry points. Each defines a concrete
  `OvertourismAdapter` subclass wrapping the corresponding `Backend` directly
  (no HTTP involved — the fastest way to iterate on model/backend code) and
  returns real `OvertourismParameterMeta` instances from `parameter_specs()`,
  which satisfy `ParameterSpec` structurally without either module importing
  the other:

  ```bash
  uv run streamlit run src/dt_studio/fazzon_dashboard.py
  uv run streamlit run src/dt_studio/molveno_dashboard.py
  ```

  (from the `executor-backend` folder; the models need their data in
  `FAZZON_DATA_DIR` / `MOLVENO_DATA_DIR`, see `docs/howto/backend.md`).

- `src/dt_studio/dashboard/http_adapter.py` +
  `fazzon_api_dashboard.py` / `molveno_api_dashboard.py` — the same
  dashboard shell driven instead by the Layer 5 REST API over HTTP
  (`HttpOvertourismAdapter`, one generic class for both models). This process
  never imports `src.model.*` at all — `parameter_specs()` returns
  `HttpParameterSpec`, a local dataclass built from the API's JSON response
  that also satisfies `ParameterSpec` structurally, so the same `Protocol`
  is shared by both dashboard styles without either concrete type importing
  the other. Only `httpx`, `numpy`, and plain JSON are involved — this
  process exercises the real client/server boundary a production frontend
  would use, and (deliberately) has none of Layers 1–3's dependencies
  (`civic_digital_twins`, `scipy`) importable at all, matching the eventual
  container split (see the Status note at the top of this document).

  ```bash
  uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8082 \
    --proxy-headers --log-config uvicorn_log_config.json
  uv run streamlit run src/dt_studio/fazzon_api_dashboard.py
  uv run streamlit run src/dt_studio/molveno_api_dashboard.py
  ```

  Both dashboard styles are kept side by side: the in-process ones for fast
  backend iteration, the HTTP ones for testing the API contract itself. The
  HTTP-driven dashboards have no scenario selector yet —
  `predefined_scenarios()` returns `[]` since `src.api` has
  no `/scenarios` route (§Layer 5 — REST API).

`OvertourismAdapter.run(param_overrides)` calls `backend.evaluate(param_overrides)`
— directly, or via the REST API — and maps the result onto `PlotData`.
`SustainabilityFieldOutput` itself carries no axis-label or subsystem-display-
name fields (§Layer 2) — those are presentation, not computation — so both
adapter styles source them independently rather than reading them off the
evaluation output: the in-process adapters (`FazzonAdapter`/`MolvenoAdapter`)
read `self._backend.X_AXIS_NAME`/`.Y_AXIS_NAME`/`.MAPPER` directly.
`HttpOvertourismAdapter` caches one `GET /schema` call (`_schema`, a
`cached_property` also reused by `parameter_specs()`) and reads the
equivalent values from `metadata.plot_mapper.bidimensional.{x,y}.label` and
`metadata.mapper` — falling back to `""`/`{}` rather than raising if that
shape ever moves. Both populate `PlotData.constraint_labels` (subsystem
display name per constraint id) the same way, so `app.py`'s field-figure
legend/hover text and KPI tables show e.g. `"Parcheggi"` instead of the raw
constraint id `"parking"`, for either dashboard style.
`predefined_scenarios()` returns UI-facing `ScenarioDef` objects — scenario
descriptions are presentation content and intentionally do not travel
through the computation backend.

Neither dashboard style is the production frontend (see
[Next steps](#layer-5--production-ui)).

---

## Next steps

### Layer 4 — application backend

**Implemented** — `overtourism.dt_manager` (scenario/problem/proposal/session
managers, persistence stores) plus its own API surface in
`overtourism.backend` (auth, `api/v2/*` routers). Not built from the plan
originally sketched in this section, and not documented in detail here —
its own code is the source of truth. The table below is kept only as
historical context for what was originally scoped and why:

| Concern (as originally scoped) | Description |
|---|---|
| Scenario catalogue | Load named parameter configurations; validate overrides against `parameter_schema()`. |
| Result persistence | Store `SustainabilityFieldOutput.to_snapshot()` output keyed by scenario + params. |
| User scenario storage | Save/name/retrieve custom parameter sets across sessions. |
| i18n | Translate `OvertourismParameterMeta.label`/`.description` if needed. |

`src.api` (§Layer 5 — REST API) itself is still stateless and
has no `/scenarios`/`/results` route of its own — `dt_manager` was built as
its own separate API surface (`overtourism.backend.api.v2.*`) rather than
being merged into `layer_3.api`'s routes; the two are not (yet) unified.

Previously, a `schema_metadata.py` file per model — hardcoded presentation
metadata surfaced via `GET /schema` — lived inside `layer_3/` even though it
conceptually belongs to Layer 4 (frontend presentation/i18n content). It has
since been removed (§Layer 5 — REST API): diffing the two files field by
field showed nothing in them was genuinely per-model content beyond what the
backend already owns (`mapper`, axis labels/fields, §Layer 3) — everything
else was either identical across both models or mechanically derivable from
`mapper`. `src.api.routes._build_metadata` now generates the
full `/schema` presentation payload from the backend's minimal metadata
alone.

### Layer 5 — production UI

**Implemented** — a production UI for Fazzon and Molveno exists as an
Angular single-page application, `https://github.com/tn-aixpa/overtourism-frontend`
(not part of this repo). Not documented in detail here — its own repo is the
source of truth. It consumes the REST API implemented here (§Layer 5 — REST
API); the Streamlit dashboards (§Layer 5 — Streamlit dev/test tooling)
remain dev/test tooling only, not this production UI.

### Layer 5 — CLI (future)

Not designed yet, and no prior implementation to draw from. Would call
Layer 4 the same way any other frontend does — as `overtourism.backend`'s
HTTP API, not a plain library call, since Layer 4 was built as a separate
service rather than an in-process dependency (§Layer 4 — application
backend).
