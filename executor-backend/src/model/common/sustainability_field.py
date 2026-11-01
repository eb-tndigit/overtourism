# SPDX-License-Identifier: Apache-2.0
"""Shared config, output type, and field math for the overtourism model family.

See `overtourism/BACKEND_DESIGN.md` §4. `compute_sustainable_area`,
`compute_sustainability_index_with_ci`, `compute_sustainability_by_constraint`,
and `compute_modal_lines` were, before this module existed, near-identical
copies duplicated across `fazzon_model.py`, `molveno_model.py`, and
`portofino_model.py`. They are pure array math (field + axes + presences +
confidence in; tuple/dict out) with no model-specific coupling.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from dataclasses import field as dataclass_field
from numbers import Real
from typing import Any

import numpy as np
from civic_digital_twins.dt_model import (
    DistributionIndex,
    EvaluationConfig,
    EvaluationResult,
    ParameterMeta,
)
from civic_digital_twins.dt_model.simulation.runner import ModelOutput
from scipy import interpolate, ndimage
from scipy import stats as scipy_stats

__all__ = [
    "OvertourismEvaluationConfig",
    "OvertourismParameterMeta",
    "SustainabilityFieldOutput",
    "compute_modal_lines",
    "compute_sustainability_by_constraint",
    "compute_sustainability_field",
    "compute_sustainability_index_with_ci",
    "compute_sustainable_area",
    "get_index_diffs",
]


# ---------------------------------------------------------------------------
# OvertourismParameterMeta
# ---------------------------------------------------------------------------


@dataclass
class OvertourismParameterMeta(ParameterMeta):
    """`ParameterMeta` extended with presentation content for the dashboards.

    Built via plain dataclass inheritance (see the rationale on
    `civic_digital_twins.dt_model.ParameterMeta`). Superset of the fields
    `overtourism.dt_studio.dashboard.adapter.ParameterSpec` used to carry separately —
    that class is removed since it duplicated this one exactly.
    """

    label: str = ""
    description: str = ""
    unit: str = ""
    category: str = ""
    step: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    default_range: tuple[float, float] | None = None
    # Display label of each categorical value (``support``), e.g. ``{"monday": "lunedì"}``:
    # widgets and index-diff labels show the label and use the value. Values without a
    # label are shown as they are.
    support_labels: dict[str, str] = dataclass_field(default_factory=dict)


# ---------------------------------------------------------------------------
# OvertourismEvaluationConfig
# ---------------------------------------------------------------------------


@dataclass
class OvertourismEvaluationConfig(EvaluationConfig):
    """`EvaluationConfig` extended with presence-sampling/statistical parameters.

    Concepts — scatter-overlay sampling, a sustainability-index confidence
    interval — that don't exist outside this model family.
    """

    sample_seed: int | None = None
    target_presence_samples: int = 2000
    confidence: float = 0.8


# ---------------------------------------------------------------------------
# Shared field math
# ---------------------------------------------------------------------------


def _format_index_value(value: Any) -> str:
    """Format a scalar or categorical value for an index-diff label."""
    if isinstance(value, Real) and not isinstance(value, bool) and value == int(value):
        return str(int(value))
    return str(value)


def _distribution_range(value: Any) -> tuple[Any, Any] | None:
    """Return distribution endpoints from a schema range or loc/scale mapping."""
    if isinstance(value, Mapping):
        loc = value.get("loc")
        scale = value.get("scale")
        if loc is None or scale is None:
            return None
        return loc, loc + scale
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    ):
        return value[0], value[1]
    return None


def get_index_diffs(
    schema: Mapping[str, Any],
    param_overrides: Mapping[str, Any],
) -> dict[str, str]:
    """Return human-readable changes from Layer 3 defaults.

    The schema is the response from a Layer 3 ``/models/{territory}/schema``
    endpoint. Only parameters present in ``param_overrides`` are considered;
    unknown parameters and overrides without a usable base value are ignored.

    Values are formatted according to their schema kind:

    - categorical: ``"cat1 -> cat2"`` (with their ``support_labels``, if any);
    - distribution: ``"x-y -> x1-y1"``;
    - scalar: ``"x -> y"``.

    A categorical schema without ``default_category`` represents the
    unfiltered base scenario and is displayed as ``(tutte)``.
    """
    specs: dict[str, Mapping[str, Any]] = {}
    for raw_spec in schema.get("indexes", []):
        if isinstance(raw_spec, Mapping):
            spec = raw_spec
        elif is_dataclass(raw_spec):
            spec = asdict(raw_spec)
        else:
            continue

        name = spec.get("name")
        if name:
            specs[name] = spec

    diffs: dict[str, str] = {}

    for name, new_value in param_overrides.items():
        spec = specs.get(name)
        if spec is None:
            continue

        kind = spec.get("kind")
        if kind == "categorical":
            old_value = spec.get("default_category", "(tutte)")
            if old_value == new_value:
                continue
            labels = spec.get("support_labels") or {}
            old_label = labels.get(old_value, _format_index_value(old_value))
            new_label = labels.get(new_value, _format_index_value(new_value))
            diffs[name] = f"{old_label} -> {new_label}"
            continue

        if kind == "distribution":
            old_range = _distribution_range(spec.get("default_range"))
            new_range = _distribution_range(new_value)
            if old_range is None or new_range is None or old_range == new_range:
                continue
            old_label = "-".join(_format_index_value(value) for value in old_range)
            new_label = "-".join(_format_index_value(value) for value in new_range)
            diffs[name] = f"{old_label} -> {new_label}"
            continue

        if kind == "scalar":
            old_value = spec.get("default")
            if old_value is None or old_value == new_value:
                continue
            diffs[name] = (
                f"{_format_index_value(old_value)} -> {_format_index_value(new_value)}"
            )

    return diffs


def compute_sustainability_field(
    constraints: Any,
    result: EvaluationResult,
    x_param: Any,
    y_param: Any,
    scenario: Any = None,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, float]],
]:
    """Compute the sustainability field, field elements, usage fields, and capacity distributions.

    Shape is ``(N_x, N_y)`` matching the ``x_param``/``y_param`` parameter
    axes passed to the evaluation.

    Parameters
    ----------
    constraints : Iterable[Any]
        Objects exposing ``.name`` (str), ``.usage`` (Index), and
        ``.capacity`` (Index) — the model's ``Constraint`` list. Structurally
        typed rather than importing a shared ``Constraint`` class: each
        model keeps its own (Fazzon's, Molveno's, Portofino's), and none of
        that is model-specific from this function's point of view.
    result : EvaluationResult
        Evaluated ensemble result (usage values and weights).
    x_param, y_param : Any
        The two PARAMETER-axis indexes swept by ``result`` (used only to
        read the field shape from ``result.parameter_values``).
    scenario : Scenario, optional
        When provided, capacity distributions are resolved via
        ``scenario.effective_distribution(capacity_index)``, so that
        scenario overrides of distribution-backed capacities are correctly
        reflected in the field. When ``None`` the index's built-in default
        distribution is used.

    Returns
    -------
    field : np.ndarray
        Sustainability probability field.
    field_elements : dict[str, np.ndarray]
        Per-constraint probability fields (same shape as ``field``).
    usage_fields : dict[str, np.ndarray]
        Per-constraint weighted-average *usage value* (same shape as
        ``field``) — the expected resource consumption at each ``(x, y)``
        presence, as opposed to ``field_elements``'s probability of staying
        under capacity. Weighted-averaged over the categorical ensemble the
        same way ``field_elements`` is (``tensordot`` against
        ``result.weights``) — the equivalent of the older
        ``EvaluationResult.marginalize()`` call, which no longer exists in
        the pinned ``civic_digital_twins`` version.
    capacity_distributions : dict[str, dict[str, float]]
        ``{"loc": mean, "scale": std}`` per constraint.
    """
    field = np.ones(
        (
            result.parameter_values[x_param].size,
            result.parameter_values[y_param].size,
        )
    )
    field_elements: dict[str, np.ndarray] = {}
    usage_fields: dict[str, np.ndarray] = {}
    capacity_distributions: dict[str, dict[str, float]] = {}
    for c in constraints:
        usage = np.broadcast_to(result[c.usage], result.full_shape)
        usage_fields[c.name] = np.tensordot(usage, result.weights, axes=([-1], [0]))
        if isinstance(c.capacity, DistributionIndex):
            dist = (
                scenario.effective_distribution(c.capacity)
                if scenario is not None
                else c.capacity.frozen_distribution
            )
            mask = (1.0 - dist.cdf(usage)).astype(float)
            capacity_distributions[c.name] = {
                "loc": float(dist.mean()),
                "scale": float(dist.std()),
            }
        else:
            cap = np.broadcast_to(result[c.capacity], result.full_shape)
            mask = (usage <= cap).astype(float)
            capacity_distributions[c.name] = {
                "loc": float(np.mean(result[c.capacity])),
                "scale": 0.0,
            }
        field_elem = np.tensordot(mask, result.weights, axes=([-1], [0]))
        field_elements[c.name] = field_elem
        field *= field_elem
    return field, field_elements, usage_fields, capacity_distributions


def _usage_uncertainty_from_params(
    params: dict[str, float], usage: list[int]
) -> list[float]:
    """Compute per-sample usage uncertainty from capacity normal params.

    `params["scale"] == 0` means a deterministic (non-distribution)
    capacity — e.g. Fazzon's `road`/`food` constraints, which aren't
    distribution-backed. `scipy.stats.norm(scale=0).cdf()` is undefined
    (NaN) there, so it's handled as the exact step-function limit instead:
    0.0 (no exceedance) at or below `loc`, 1.0 (exceeded) above it — the
    same `usage <= capacity` convention `compute_sustainability_field` uses.
    """
    loc, scale = params["loc"], params["scale"]
    if scale <= 0:
        return [1.0 if u > loc else 0.0 for u in usage]
    capacity = scipy_stats.norm(loc=loc, scale=scale)
    capacity_mean = float(capacity.mean())
    y_max = max(1, int(max(max(usage, default=0), capacity_mean) * 1.2))
    cap_cdf = [float(capacity.cdf(y)) for y in range(y_max)]
    heatmap_y = np.linspace(0, y_max, len(cap_cdf))
    result = []
    for u in usage:
        idx = int(np.abs(heatmap_y - u).argmin())
        result.append(float(f"{cap_cdf[idx]:.4f}"))
    return result


def compute_sustainable_area(
    field: np.ndarray, x_values: np.ndarray, y_values: np.ndarray
) -> float:
    """Integral approximation of the sustainable area under the field."""
    return field.sum() * functools.reduce(
        lambda a, b: a * b,
        [axis.max() / (axis.size - 1) + 1 for axis in (x_values, y_values)],
    )


def compute_sustainability_index_with_ci(
    field: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    presences: list,
    confidence: float = 0.8,
) -> tuple[float, float]:
    """Return ``(sustainability_index, ci_half_width)`` over the sampled presences."""
    index = interpolate.interpn(
        (x_values, y_values),
        field,
        np.array(presences),
        bounds_error=False,
        fill_value=0.0,
    )
    m, se = np.mean(index), scipy_stats.sem(index)
    h = float(se * scipy_stats.t.ppf((1 + confidence) / 2.0, index.size - 1))
    return float(m), h


def compute_sustainability_by_constraint(
    field_elements: dict[str, np.ndarray],
    x_values: np.ndarray,
    y_values: np.ndarray,
    presences: list,
    confidence: float = 0.8,
) -> dict[str, tuple[float, float]]:
    """Return ``(index, ci_half_width)`` per constraint name."""
    result: dict[str, tuple[float, float]] = {}
    for name, fe in field_elements.items():
        index = interpolate.interpn(
            (x_values, y_values),
            fe,
            np.array(presences),
            bounds_error=False,
            fill_value=0.0,
        )
        m, se = np.mean(index), scipy_stats.sem(index)
        h = float(se * scipy_stats.t.ppf((1 + confidence) / 2.0, index.size - 1))
        result[name] = (float(m), h)
    return result


def compute_modal_lines(
    field_elements: dict[str, np.ndarray],
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute per-constraint modal lines via orthogonal regression (first PC)."""
    bounds = [x_values.max(), y_values.max()]
    modal_lines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, fe in field_elements.items():
        matrix = (fe <= 0.5) & (
            (ndimage.shift(fe, (0, 1)) > 0.5)
            | (ndimage.shift(fe, (0, -1)) > 0.5)
            | (ndimage.shift(fe, (1, 0)) > 0.5)
            | (ndimage.shift(fe, (-1, 0)) > 0.5)
        )
        yi, xi = np.nonzero(matrix)
        if len(yi) < 3:
            continue
        pts = np.stack([x_values[yi], y_values[xi]], axis=1)
        centroid = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
        direction = vt[0]
        t_lo, t_hi = -np.inf, np.inf
        for i, bound in enumerate(bounds):
            if abs(direction[i]) > 1e-10:
                ta = -centroid[i] / direction[i]
                tb = (bound - centroid[i]) / direction[i]
                t_lo = max(t_lo, min(ta, tb))
                t_hi = min(t_hi, max(ta, tb))
        if t_lo >= t_hi:
            continue
        p0 = centroid + t_lo * direction
        p1 = centroid + t_hi * direction
        modal_lines[name] = (np.array([p0[0], p1[0]]), np.array([p0[1], p1[1]]))
    return modal_lines


# ---------------------------------------------------------------------------
# SustainabilityFieldOutput
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class SustainabilityFieldOutput(ModelOutput):
    """Unified evaluation output for 2D sustainability-field models.

    Covers presence-vs-presence models (Fazzon, Molveno; Portofino as a
    future migration). Replaces per-model output classes directly — no per-model subclass is
    needed since, once the field math above is shared, no model has a field
    to add beyond this common set.

    Attributes
    ----------
    field : np.ndarray
        Shape ``(N_x, N_y)``. Each entry is the probability that all
        constraints are satisfied at the corresponding ``(x, y)`` presence.
    field_elements : dict[str, np.ndarray]
        Per-constraint field arrays, same shape as ``field``.
    x_values, y_values : np.ndarray
        1-D parameter-grid axes, shape ``(N_x,)`` / ``(N_y,)``.
    samples_x, samples_y : list[float]
        Raw presence samples for the scatter overlay.
    usage_fields : dict[str, np.ndarray]
        Per-constraint weighted-average usage value, same shape as
        ``field`` — see `compute_sustainability_field`.
    capacity_distributions : dict[str, dict[str, float]]
        ``{"loc": mean, "scale": std}`` per constraint.
    confidence : float
        Confidence level for the sustainability CI (default 0.8).
    extra_kpis : dict[str, dict[str, Any]]
        Model-specific KPIs, added to :attr:`kpis` as they are (e.g. Fazzon's
        potential visitors who cannot arrive).  Empty by default.
    """

    field: np.ndarray
    field_elements: dict[str, np.ndarray]
    x_values: np.ndarray
    y_values: np.ndarray
    samples_x: list[float]
    samples_y: list[float]
    usage_fields: dict[str, np.ndarray]
    capacity_distributions: dict[str, dict[str, float]]
    confidence: float = 0.8
    extra_kpis: dict[str, dict[str, Any]] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialise the base `ModelOutput`."""
        super().__init__()

    @functools.cached_property
    def _zip_samples(self) -> list[tuple[float, float]]:
        """Zip x/y presence samples into coordinate pairs."""
        return list(zip(self.samples_x, self.samples_y))

    @functools.cached_property
    def sustainable_area(self) -> float:
        """Integral approximation of the sustainable area under the field."""
        return compute_sustainable_area(self.field, self.x_values, self.y_values)

    @functools.cached_property
    def sustainability_index(self) -> tuple[float, float]:
        """Return ``(index, ci_half_width)`` over the sampled presences."""
        return compute_sustainability_index_with_ci(
            self.field, self.x_values, self.y_values, self._zip_samples, self.confidence
        )

    @functools.cached_property
    def sustainability_by_constraint(self) -> dict[str, tuple[float, float]]:
        """Return ``(index, ci_half_width)`` per constraint."""
        return compute_sustainability_by_constraint(
            self.field_elements,
            self.x_values,
            self.y_values,
            self._zip_samples,
            self.confidence,
        )

    @functools.cached_property
    def modal_lines(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per-constraint modal lines via orthogonal regression."""
        return compute_modal_lines(self.field_elements, self.x_values, self.y_values)

    @functools.cached_property
    def x_max(self) -> float:
        """Maximum value of the x-axis grid."""
        return float(self.x_values.max())

    @functools.cached_property
    def y_max(self) -> float:
        """Maximum value of the y-axis grid."""
        return float(self.y_values.max())

    @functools.cached_property
    def uncertainty(self) -> list[float]:
        """Per-sample overall sustainability field value, interpolated at the presence samples."""
        pts = np.array(self._zip_samples)
        vals = interpolate.interpn(
            (self.x_values, self.y_values),
            self.field,
            pts,
            bounds_error=False,
            fill_value=0.0,
        )
        return [float(f"{v:.4f}") for v in vals]

    @functools.cached_property
    def uncertainty_by_constraint(self) -> dict[str, list[float]]:
        """Per-sample per-constraint sustainability field values."""
        pts = np.array(self._zip_samples)
        result: dict[str, list[float]] = {}
        for name, fe in self.field_elements.items():
            vals = interpolate.interpn(
                (self.x_values, self.y_values),
                fe,
                pts,
                bounds_error=False,
                fill_value=0.0,
            )
            result[name] = [float(f"{v:.4f}") for v in vals]
        return result

    @functools.cached_property
    def usage_by_constraint(self) -> dict[str, list[int]]:
        """Per-sample per-constraint usage, interpolated from `usage_fields`."""
        pts = np.array(self._zip_samples)
        result: dict[str, list[int]] = {}
        for name, uf in self.usage_fields.items():
            vals = interpolate.interpn(
                (self.x_values, self.y_values),
                uf,
                pts,
                bounds_error=False,
                fill_value=0.0,
            )
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            result[name] = [max(0, int(v)) for v in vals]
        return result

    @functools.cached_property
    def usage(self) -> list[int]:
        """Per-sample aggregate normalised usage (0-100 scale) across all constraints."""
        pts = np.array(self._zip_samples)
        n = len(self._zip_samples)
        agg = np.ones(n)
        for name, uf in self.usage_fields.items():
            cap_loc = self.capacity_distributions[name]["loc"]
            u_vals = interpolate.interpn(
                (self.x_values, self.y_values),
                uf,
                pts,
                bounds_error=False,
                fill_value=0.0,
            )
            if np.isfinite(cap_loc) and cap_loc > 0:
                agg += u_vals / cap_loc
            else:
                agg += np.inf
        agg *= 100.0 / len(self.usage_fields)
        agg = np.nan_to_num(agg, nan=0.0, posinf=100.0, neginf=0.0)
        agg = np.clip(agg, 0.0, 100.0)
        return [int(u) for u in agg]

    @functools.cached_property
    def capacity_mean_by_constraint(self) -> dict[str, float]:
        """Mean capacity per constraint."""
        return {
            name: params["loc"] for name, params in self.capacity_distributions.items()
        }

    @functools.cached_property
    def capacity_mean(self) -> float:
        """Reference ceiling (always 100.0) for the normalised 0-100 `usage` scale — not an actual mean."""
        return 100.0

    @functools.cached_property
    def usage_uncertainty(self) -> list[float]:
        """Per-sample aggregate usage uncertainty via the aggregate capacity CDF."""
        variance = sum(
            (params["scale"] ** 2) / (params["loc"] ** 2)
            for params in self.capacity_distributions.values()
            if np.isfinite(params["loc"]) and params["loc"] > 0
        )
        n_c = len(self.capacity_distributions)
        agg_scale = (variance**0.5) * 100.0 / n_c
        return _usage_uncertainty_from_params(
            {"loc": 100.0, "scale": agg_scale}, self.usage
        )

    @functools.cached_property
    def usage_uncertainty_by_constraint(self) -> dict[str, list[float]]:
        """Per-sample per-constraint usage uncertainty."""
        return {
            name: _usage_uncertainty_from_params(params, self.usage_by_constraint[name])
            for name, params in self.capacity_distributions.items()
        }

    @functools.cached_property
    def kpis(self) -> dict[str, Any]:
        """KPI dict, in display order: the model-specific KPIs, the sustainability level, the
        per-constraint levels, the critical constraint. English-only — locale translation is a
        presentation-layer concern, not part of this computation.
        """
        idx, ci = self.sustainability_index
        sbc = self.sustainability_by_constraint
        kpis: dict[str, Any] = dict(self.extra_kpis)
        kpis["sustainability_level"] = {
            "level": round(idx * 100, 4),
            "confidence": round(ci * 100, 4),
        }
        for name, (c_mean, c_ci) in sbc.items():
            kpis["constraint level " + name] = {
                "level": round(c_mean * 100, 4),
                "confidence": round(c_ci * 100, 4),
            }
        critical_name = min(sbc, key=lambda k: sbc[k][0])
        c_mean, c_ci = sbc[critical_name]
        kpis["critical constraint"] = {
            "name": critical_name,
            "level": round(c_mean * 100, 4),
            "confidence": round(c_ci * 100, 4),
        }
        return kpis

    @functools.cached_property
    def constraint_curves(self) -> dict[str, list[list[float]]]:
        """Per-constraint modal line as `[x_coords, y_coords]` list pairs — same data as `modal_lines`, reshaped."""
        return {
            k: [list(x_coords), list(y_coords)]
            for k, (x_coords, y_coords) in self.modal_lines.items()
        }

    def to_snapshot(self) -> dict[str, Any]:
        """Serialise to a plain dict snapshot, including derived KPIs."""
        d = super().to_snapshot()
        d["sustainable_area"] = float(self.sustainable_area)
        idx, ci = self.sustainability_index
        d["sustainability_index"] = {"value": float(idx), "ci": float(ci)}
        d["sustainability_by_constraint"] = {
            k: {"value": float(v), "ci": float(c)}
            for k, (v, c) in self.sustainability_by_constraint.items()
        }
        d["modal_lines"] = {
            k: {"x": list(x_coords), "y": list(y_coords)}
            for k, (x_coords, y_coords) in self.modal_lines.items()
        }
        d["x_max"] = self.x_max
        d["y_max"] = self.y_max
        d["uncertainty"] = self.uncertainty
        d["uncertainty_by_constraint"] = self.uncertainty_by_constraint
        d["usage"] = self.usage
        d["usage_by_constraint"] = self.usage_by_constraint
        d["usage_uncertainty"] = self.usage_uncertainty
        d["usage_uncertainty_by_constraint"] = self.usage_uncertainty_by_constraint
        d["capacity_mean"] = self.capacity_mean
        d["capacity_mean_by_constraint"] = self.capacity_mean_by_constraint
        d["kpis"] = self.kpis
        d["constraint_curves"] = self.constraint_curves
        return d


def arrange_frontend_data(output: SustainabilityFieldOutput) -> dict[str, Any]:
    """Return the legacy frontend payload from a field evaluation output."""
    snapshot = output.to_snapshot()

    points = {
        "uncertainty": [],
        "uncertainty_by_constraint": {
            key: [] for key in snapshot["uncertainty_by_constraint"]
        },
    }
    for tourists, excursionists, index, usage, usage_uncertainty in zip(
        snapshot["samples_x"],
        snapshot["samples_y"],
        snapshot["uncertainty"],
        snapshot["usage"],
        snapshot["usage_uncertainty"],
    ):
        points["uncertainty"].append(
            {
                "tourists": tourists,
                "excursionists": excursionists,
                "index": index,
                "usage": usage,
                "usage_uncertainty": usage_uncertainty,
            }
        )

    for constraint, indices in snapshot["uncertainty_by_constraint"].items():
        for tourists, excursionists, index, usage, usage_uncertainty in zip(
            snapshot["samples_x"],
            snapshot["samples_y"],
            indices,
            snapshot["usage_by_constraint"][constraint],
            snapshot["usage_uncertainty_by_constraint"][constraint],
        ):
            points["uncertainty_by_constraint"][constraint].append(
                {
                    "tourists": tourists,
                    "excursionists": excursionists,
                    "index": index,
                    "usage": usage,
                    "usage_uncertainty": usage_uncertainty,
                }
            )

    return {
        "points": points,
        "kpis": snapshot["kpis"],
        "x_max": snapshot["x_max"],
        "y_max": snapshot["y_max"],
        "capacity_mean": snapshot["capacity_mean"],
        "capacity_mean_by_constraint": snapshot["capacity_mean_by_constraint"],
        "constraint_curves": snapshot["constraint_curves"],
    }
