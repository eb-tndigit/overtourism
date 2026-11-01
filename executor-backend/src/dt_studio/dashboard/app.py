# SPDX-License-Identifier: Apache-2.0
"""Generic Streamlit dashboard for overtourism digital twins.

The single public entry point is :func:`run_dashboard`.  Pass a concrete
:class:`~overtourism.dt_studio.dashboard.adapter.OvertourismAdapter` instance and the
function renders a fully interactive dashboard in the calling Streamlit app.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .adapter import OvertourismAdapter, ParameterSpec, PlotData, ScenarioDef

# ---------------------------------------------------------------------------
# Constraint colour palette
# ---------------------------------------------------------------------------

_CONSTRAINT_COLORS: dict[str, str] = {
    "parking": "#c0392b",
    "road": "#e07b39",
    "food": "#27ae60",
    "lakeside": "#2980b9",
}
_FALLBACK_COLORS = ["#8e44ad", "#16a085", "#d35400", "#2c3e50"]

# Number of distinct parameter combinations whose results are kept in the
# per-session cache.  Old entries are evicted LRU-style.
_RESULT_CACHE_SIZE = 50

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _param_hash(params: dict[str, Any]) -> str:
    """Return a stable MD5 hex digest of a parameter dict for change detection.

    Tuples are converted to lists before JSON encoding so that distribution
    range values ``(lo, hi)`` serialise correctly.
    """
    serialisable = {
        k: list(v) if isinstance(v, tuple) else v for k, v in params.items()
    }
    return hashlib.md5(json.dumps(serialisable, sort_keys=True).encode()).hexdigest()


def _default_value(spec: ParameterSpec) -> Any:
    """Return the appropriate session-state default value for a parameter spec."""
    if spec.kind == "scalar":
        return spec.default
    if spec.kind == "distribution":
        return spec.default_range
    # categorical
    return "(tutte)"


def _init_session(specs: list[ParameterSpec], scenarios: list[ScenarioDef]) -> None:
    """Initialise ``st.session_state`` on the first page load.

    Pre-loads all predefined scenarios into ``__saved_scenarios__`` as
    named parameter sets (no pre-computation — results are computed lazily
    from the cache when first needed).  Sets ``__view__ = "simulation"``.

    Calling this function a second time is a no-op.
    """
    if "__initialized__" in st.session_state:
        return
    for spec in specs:
        st.session_state[f"_p_{spec.name}"] = _default_value(spec)
    if scenarios:
        st.session_state["__active_scenario_key__"] = scenarios[0].key

    # Pre-load predefined scenarios as named saves (params only, no PlotData).
    # Use the scenario key as the internal name; category marks them as predefined.
    saved: dict[str, dict[str, Any]] = {}
    for s in scenarios:
        saved[s.key] = {
            "label": s.label,
            "params": s.params,
            "category": s.category,
            "predefined": True,
        }
    st.session_state["__saved_scenarios__"] = saved

    st.session_state["__view__"] = "simulation"
    st.session_state["__result_cache__"] = {}
    st.session_state["__cache_order__"] = []
    st.session_state["__initialized__"] = True


def _load_scenario_params(scenario: ScenarioDef, specs: list[ParameterSpec]) -> None:
    """Write a scenario's parameter values into ``st.session_state``."""
    for spec in specs:
        key = f"_p_{spec.name}"
        if spec.name in scenario.params:
            st.session_state[key] = scenario.params[spec.name]
        else:
            st.session_state[key] = _default_value(spec)


def _load_saved_params(saved: dict[str, Any], specs: list[ParameterSpec]) -> None:
    """Write a saved-scenario's parameter values into ``st.session_state``."""
    params = saved.get("params", {})
    for spec in specs:
        key = f"_p_{spec.name}"
        if spec.name in params:
            st.session_state[key] = params[spec.name]
        else:
            st.session_state[key] = _default_value(spec)


def _read_current_params(specs: list[ParameterSpec]) -> dict[str, Any]:
    """Read current widget values from ``st.session_state`` into a param dict.

    Categorical ``"(tutte)"`` values are omitted (treated as unpinned).
    """
    result: dict[str, Any] = {}
    for spec in specs:
        value = st.session_state.get(f"_p_{spec.name}")
        if spec.kind == "categorical" and value == "(tutte)":
            continue
        if value is not None:
            result[spec.name] = value
    return result


def _get_or_compute(
    adapter: OvertourismAdapter,
    params: dict[str, Any],
    cache: dict[str, PlotData],
    cache_order: list[str],
    spinner_msg: str = "Simulazione in corso…",
) -> PlotData:
    """Return cached PlotData for *params*, computing it if necessary."""
    h = _param_hash(params)
    if h not in cache:
        with st.spinner(spinner_msg):
            result = adapter.run(params)
        cache[h] = result
        cache_order.append(h)
        if len(cache_order) > _RESULT_CACHE_SIZE:
            evicted = cache_order.pop(0)
            cache.pop(evicted, None)
    return cache[h]


def _make_field_figure(data: PlotData, title: str | None = None) -> go.Figure:
    """Build a Plotly figure rendering the sustainability field.

    Uses a ``RdBu`` heatmap (red = unsustainable, blue = sustainable) with
    bilinear interpolation, per-constraint modal lines coloured from
    :data:`_CONSTRAINT_COLORS`, a translucent presence-sample scatter, an
    interactive colorbar, and hover tooltips.

    Parameters
    ----------
    data : PlotData
        Visualisation-ready data produced by the adapter's ``run()`` method.
    title : str, optional
        Optional figure title shown above the axes.

    Returns
    -------
    go.Figure
        Plotly figure ready to pass to ``st.plotly_chart``.
    """
    fig = go.Figure()

    # ── Heatmap (smooth gradient, bilinear interpolation) ──────────────────
    fig.add_trace(
        go.Heatmap(
            x=data.x_values,
            y=data.y_values,
            z=data.field.T,
            colorscale="RdBu",
            zmin=0.0,
            zmax=1.0,
            colorbar={
                "title": {"text": "P(sostenibile)", "side": "right"},
                "thickness": 14,
                "len": 0.85,
            },
            hovertemplate=(
                f"{data.x_label}: <b>%{{x:.0f}}</b><br>"
                f"{data.y_label}: <b>%{{y:.0f}}</b><br>"
                "P(tutto OK): <b>%{z:.2f}</b><extra></extra>"
            ),
            name="",
        )
    )

    # ── Modal lines (one per constraint, coloured by type) ─────────────────
    fallback_idx = 0
    for name, (x_coords, y_coords) in data.modal_lines.items():
        color = _CONSTRAINT_COLORS.get(name)
        if color is None:
            color = _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]
            fallback_idx += 1
        sust_val, sust_ci = data.sustainability_by_constraint.get(name, (0.0, 0.0))
        label = data.constraint_labels.get(name, name.capitalize())
        fig.add_trace(
            go.Scatter(
                x=list(x_coords),
                y=list(y_coords),
                mode="lines",
                name=f"{label}  {sust_val * 100:.0f}%",
                line={"color": color, "width": 3},
                hovertemplate=(
                    f"<b>{label} — frontiera</b><br>"
                    f"Sostenibilità: {sust_val * 100:.1f}% ± {sust_ci * 100:.1f}%<br>"
                    f"{data.x_label}: %{{x:.0f}}<br>"
                    f"{data.y_label}: %{{y:.0f}}<extra></extra>"
                ),
            )
        )

    # ── Presence-sample scatter ─────────────────────────────────────────────
    rng = np.random.default_rng(42)
    n_show = min(500, len(data.samples_x))
    idx = rng.choice(len(data.samples_x), size=n_show, replace=False)
    xs = np.asarray(data.samples_x)
    ys = np.asarray(data.samples_y)
    fig.add_trace(
        go.Scatter(
            x=xs[idx],
            y=ys[idx],
            mode="markers",
            name="Campioni (modello)",
            marker={
                "symbol": "circle",
                "color": "rgba(230,230,230,0.55)",
                "line": {"color": "rgba(80,80,80,0.6)", "width": 1},
                "size": 7,
            },
            hovertemplate=(
                f"<b>Campione</b><br>{data.x_label}: %{{x:.0f}}<br>{data.y_label}: %{{y:.0f}}<extra></extra>"
            ),
        )
    )

    # ── Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        title={"text": title, "font": {"size": 13}} if title else None,
        xaxis={
            "title": data.x_label,
            "range": [0, float(np.asarray(data.x_values).max())],
        },
        yaxis={
            "title": data.y_label,
            "range": [0, float(np.asarray(data.y_values).max())],
        },
        height=560,
        legend={
            "orientation": "v",
            "x": 1.18,
            "y": 1.0,
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "#cccccc",
            "borderwidth": 1,
        },
        margin={"r": 180, "t": 50 if title else 20, "l": 60, "b": 60},
        hoverlabel={"bgcolor": "white", "font_size": 13},
    )
    return fig


def _kpi_dataframe(named_outputs: list[tuple[str, PlotData]]) -> pd.DataFrame:
    """Build a comparison KPI DataFrame from a list of (label, PlotData) pairs.

    Rows are scenarios; columns are overall SI plus one column per constraint.
    """
    rows = []
    for label, data in named_outputs:
        idx_val, idx_ci = data.sustainability_index
        row: dict[str, str] = {
            "Scenario": label,
            "Complessivo": f"{idx_val * 100:.1f}% ±{idx_ci * 100:.1f}%",
        }
        for name, (v, c) in data.sustainability_by_constraint.items():
            label = data.constraint_labels.get(name, name.capitalize())
            row[label] = f"{v * 100:.1f}% ±{c * 100:.1f}%"
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Scenario")


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------


@st.dialog("Salva scenario corrente")
def _save_dialog(current_params: dict[str, Any]) -> None:
    """Dialog: ask for a name and save the current parameter set."""
    saved: dict[str, dict[str, Any]] = st.session_state["__saved_scenarios__"]
    name = st.text_input("Nome scenario", placeholder="es. test-navetta-alta")
    st.caption("I nomi degli scenari predefiniti non possono essere sovrascritti.")
    if st.button("Salva", type="primary", width="stretch"):
        name = name.strip()
        if not name:
            st.toast("⚠️ Il nome non può essere vuoto.")
        elif name in saved and saved[name].get("predefined"):
            st.toast("⚠️ Non puoi sovrascrivere uno scenario predefinito.")
        else:
            saved[name] = {
                "label": name,
                "params": current_params,
                "category": "Personale",
                "predefined": False,
            }
            st.session_state["__saved_scenarios__"] = saved
            st.toast(f"✅ Scenario «{name}» salvato.")
            st.rerun()


@st.dialog("Carica scenario salvato")
def _load_dialog(specs: list[ParameterSpec]) -> None:
    """Dialog: pick a saved scenario and restore its parameter values."""
    saved: dict[str, dict[str, Any]] = st.session_state["__saved_scenarios__"]
    if not saved:
        st.info("Nessuno scenario salvato.")
        return

    # Group by category for readability
    by_cat: dict[str, list[str]] = {}
    for key, meta in saved.items():
        cat = meta.get("category", "Altro")
        by_cat.setdefault(cat, []).append(key)

    options: list[str] = []
    display_labels: list[str] = []
    for cat, keys in by_cat.items():
        for k in keys:
            options.append(k)
            display_labels.append(f"[{cat}] {saved[k]['label']}")

    selected_display = st.selectbox("Seleziona scenario", display_labels)
    selected_key = (
        options[display_labels.index(selected_display)] if selected_display else None
    )

    if st.button("Carica", type="primary", width="stretch") and selected_key:
        _load_saved_params(saved[selected_key], specs)
        st.toast(f"✅ Parametri di «{saved[selected_key]['label']}» caricati.")
        st.rerun()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_dashboard(adapter: OvertourismAdapter) -> None:
    """Render the full overtourism dashboard for the given adapter.

    This is the single public entry point.  Call it at the module level of a
    Streamlit app script with a concrete adapter instance::

        from overtourism.dt_studio.dashboard.app import run_dashboard
        from my_package.my_adapter import MyAdapter

        run_dashboard(MyAdapter())

    The function renders:

    * A **sidebar** with a predefined-scenario selector and grouped parameter
      sliders/selectboxes.
    * A **main area** with Save / Load / Compare buttons, the sustainability
      field plot, and a KPI panel.
    * A **compare view** (activated by "Confronta") that shows two saved
      scenarios side by side with field plots and a KPI comparison table.
    """
    st.set_page_config(page_title=adapter.title, layout="wide")

    specs = adapter.parameter_specs()
    scenarios = adapter.predefined_scenarios()
    scenarios_by_key = {s.key: s for s in scenarios}

    _init_session(specs, scenarios)

    cache: dict[str, PlotData] = st.session_state["__result_cache__"]
    cache_order: list[str] = st.session_state["__cache_order__"]
    saved: dict[str, dict[str, Any]] = st.session_state["__saved_scenarios__"]
    view: str = st.session_state.get("__view__", "simulation")

    # ── Sidebar (parameters) ─────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Parametri")

        if scenarios:
            st.subheader("Scenari predefiniti")
            labels = [s.label for s in scenarios]
            active_key = st.session_state.get(
                "__active_scenario_key__", scenarios[0].key
            )
            active_label = scenarios_by_key.get(active_key, scenarios[0]).label
            selected_label = st.selectbox(
                "Scenario predefinito",
                labels,
                index=labels.index(active_label),
                label_visibility="collapsed",
            )
            selected_key = next(s.key for s in scenarios if s.label == selected_label)
            if st.button("Carica scenario", width="stretch"):
                _load_scenario_params(scenarios_by_key[selected_key], specs)
                st.session_state["__active_scenario_key__"] = selected_key
                st.rerun()

        st.divider()

        categories = list(dict.fromkeys(s.category for s in specs))
        for cat in categories:
            st.subheader(cat)
            for spec in (s for s in specs if s.category == cat):
                key = f"_p_{spec.name}"
                lbl = f"{spec.label} ({spec.unit})" if spec.unit else spec.label
                if spec.kind == "scalar":
                    st.slider(
                        lbl,
                        min_value=spec.min_value,
                        max_value=spec.max_value,
                        step=spec.step,
                        key=key,
                        help=spec.description,
                    )
                elif spec.kind == "categorical":
                    opts = ["(tutte)"] + spec.support
                    labels = spec.support_labels or {}
                    st.selectbox(
                        lbl,
                        opts,
                        key=key,
                        help=spec.description,
                        format_func=lambda v, labels=labels: labels.get(v, v),
                    )
                elif spec.kind == "distribution":
                    st.slider(
                        lbl,
                        min_value=spec.min_value,
                        max_value=spec.max_value,
                        step=spec.step,
                        key=key,
                        help=spec.description,
                    )

    # ── Page header + action buttons ─────────────────────────────
    st.title(adapter.title)

    current_params = _read_current_params(specs)

    _col_title_spacer, col_save, col_load, col_compare = st.columns([6, 1, 1, 1])
    with col_save:
        if st.button(
            "💾 Salva", width="stretch", help="Salva i parametri correnti con un nome"
        ):
            _save_dialog(current_params)
    with col_load:
        if st.button(
            "📂 Carica",
            width="stretch",
            help="Ripristina i parametri di uno scenario salvato",
        ):
            _load_dialog(specs)
    with col_compare:
        if view == "simulation":
            if st.button(
                "📊 Confronta",
                width="stretch",
                type="primary",
                help="Confronta due scenari salvati",
            ):
                st.session_state["__view__"] = "compare"
                st.rerun()
        else:
            if st.button("◀ Torna", width="stretch", type="primary"):
                st.session_state["__view__"] = "simulation"
                st.rerun()

    st.divider()

    # ── Compare view ─────────────────────────────────────────────
    if view == "compare":
        saved_names = list(saved.keys())
        display_names = [
            f"[{saved[k]['category']}] {saved[k]['label']}" for k in saved_names
        ]

        # Add "scenario corrente" as first option
        _CURRENT_KEY = "__current__"
        all_keys = [_CURRENT_KEY] + saved_names
        all_labels = ["▶ Scenario corrente"] + display_names

        col_a, col_b = st.columns(2)
        with col_a:
            sel_a_label = st.selectbox(
                "Scenario A", all_labels, index=0, key="__compare_a__"
            )
            sel_a_key = all_keys[all_labels.index(sel_a_label)]
        with col_b:
            default_b = min(1, len(all_keys) - 1)
            sel_b_label = st.selectbox(
                "Scenario B", all_labels, index=default_b, key="__compare_b__"
            )
            sel_b_key = all_keys[all_labels.index(sel_b_label)]

        def _resolve(key: str) -> tuple[str, dict[str, Any]]:
            if key == _CURRENT_KEY:
                return "Scenario corrente", current_params
            meta = saved[key]
            return meta["label"], meta["params"]

        label_a, params_a = _resolve(sel_a_key)
        label_b, params_b = _resolve(sel_b_key)

        data_a = _get_or_compute(
            adapter, params_a, cache, cache_order, f"Calcolo {label_a}…"
        )
        data_b = _get_or_compute(
            adapter, params_b, cache, cache_order, f"Calcolo {label_b}…"
        )

        # Side-by-side field plots
        col_fa, col_fb = st.columns(2)
        with col_fa:
            fig_a = _make_field_figure(data_a, title=label_a)
            st.plotly_chart(fig_a, width="stretch")
        with col_fb:
            fig_b = _make_field_figure(data_b, title=label_b)
            st.plotly_chart(fig_b, width="stretch")

        # KPI comparison table
        st.subheader("Confronto indicatori")
        df = _kpi_dataframe([(label_a, data_a), (label_b, data_b)])
        if not df.empty:
            st.dataframe(df, width="stretch")

        # Delta metric (overall SI difference)
        idx_a, _ = data_a.sustainability_index
        idx_b, _ = data_b.sustainability_index
        delta = idx_b - idx_a
        sign = "+" if delta >= 0 else ""
        st.metric(
            f"Δ Sostenibilità complessiva ({label_b} vs {label_a})",
            f"{sign}{delta * 100:.1f} pp",
            delta=f"{sign}{delta * 100:.1f} pp",
            delta_color="normal",
        )
        return  # no simulation view below in compare mode

    # ── Simulation view ──────────────────────────────────────────
    output = _get_or_compute(adapter, current_params, cache, cache_order)

    active_key = st.session_state.get("__active_scenario_key__")
    active_s = scenarios_by_key.get(active_key) if active_key else None
    if active_s and active_s.description:
        with st.expander(f"📋 {active_s.label}", expanded=False):
            st.markdown(active_s.description)

    col_plot, col_kpi = st.columns([3, 2])

    with col_plot:
        st.subheader("Campo di sostenibilità")
        fig = _make_field_figure(output)
        st.plotly_chart(fig, width="stretch")

    with col_kpi:
        st.subheader("Indicatori")
        for label, value in output.extra_kpis.items():
            st.metric(label, value)
        idx_val, idx_ci = output.sustainability_index
        st.metric(
            "Sostenibilità complessiva",
            f"{idx_val * 100:.1f}%",
            delta=f"CI ±{idx_ci * 100:.1f}%",
            delta_color="off",
        )
        st.markdown("**Per vincolo:**")
        rows = [
            {
                "Vincolo": output.constraint_labels.get(name, name.capitalize()),
                "Sost.": f"{v * 100:.1f}%",
                "CI": f"±{c * 100:.1f}%",
            }
            for name, (v, c) in output.sustainability_by_constraint.items()
        ]
        if rows:
            df = pd.DataFrame(rows).set_index("Vincolo")
            st.dataframe(df, width="stretch")
            critical = min(
                output.sustainability_by_constraint,
                key=lambda k: output.sustainability_by_constraint[k][0],
            )
            critical_label = output.constraint_labels.get(
                critical, critical.capitalize()
            )
            st.caption(f"⚠️ Vincolo critico: **{critical_label}**")
