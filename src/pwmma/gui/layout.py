# src/pwmma/gui/layout.py
"""Dash component tree for the pwmma GUI. Built per page load so saved defaults apply.

Desktop-EDA skeleton: title bar / toolbar / body (inputs left, results right) /
status bar. All theming lives in assets/style.css; the few inline styles here
are one-off height/spacing tweaks only.
"""
from __future__ import annotations

from dash import dcc, html

from . import defaults
from .callbacks import render_chain_header, render_chain_rows


def _field(label: str, component) -> html.Div:
    """A small column label stacked above its input."""
    return html.Div([html.Div(label, className="lbl"), component], className="field")


def _titlebar() -> html.Div:
    return html.Div([
        html.Div(className="logo"),
        html.Span("pwmma", className="title"),
        html.Span("Mode-Matching Analyzer", className="subtitle"),
    ], className="titlebar")


def _toolbar(d: dict) -> html.Div:
    return html.Div([
        html.Button("▶ Run", id="run-button", n_clicks=0, className="btn-primary"),
        html.Button("⏹ Stop", id="stop-button", n_clicks=0, disabled=True),
        html.Button("💾 Save as default", id="save-default", n_clicks=0),
        html.Div(className="toolbar-sep"),
        html.Span("precision", className="lbl"),
        dcc.Dropdown(id="precision", options=["single", "double"],
                     value=d.get("precision", "single"), clearable=False,
                     className="dd-small"),
        dcc.Checklist(id="use-gpu", options=[{"label": " GPU", "value": "gpu"}],
                      value=d.get("use_gpu", ["gpu"]), className="inline-check"),
        html.Div(className="toolbar-sep"),
        html.Span("compute", className="lbl"),
        dcc.Dropdown(id="compute-select",
                     options=[{"label": "Both", "value": "both"},
                              {"label": "S-parameters", "value": "spars"},
                              {"label": "Mode analysis", "value": "energy"}],
                     value=d.get("compute", "both"), clearable=False,
                     className="dd-small"),
        html.Span("sweep", className="lbl"),
        # One sampling strategy for the WHOLE run (spars and energy share the
        # grid): uniform = research line (f_n points, full S-matrix);
        # adaptive = production line (AAA/AFS, fundamental mode, ~40 solves).
        dcc.Dropdown(id="sweep-mode",
                     options=[{"label": "uniform", "value": "uniform"},
                              {"label": "adaptive (AAA)", "value": "adaptive"}],
                     value=d.get("sweep", "uniform"), clearable=False,
                     className="dd-small"),
    ], className="toolbar")


def _left_panel(d: dict) -> html.Div:
    return html.Div([
        html.Fieldset([
            html.Legend("Structure preview"),
            dcc.Graph(id="structure-preview", config={"displayModeBar": False},
                      style={"height": "216px"}),
        ], className="group"),

        html.Fieldset([
            html.Legend("Waveguide chain"),
            html.Div([
                render_chain_header(),
                html.Div(id="chain-rows", children=render_chain_rows(d["rows"])),
            ], className="chain-scroll"),
            html.Div([
                html.Button("+ add waveguide", id="add-wg", n_clicks=0,
                            className="btn-small"),
                dcc.Checklist(id="sym",
                              options=[{"label": " symmetric (sym)", "value": "sym"}],
                              value=d.get("sym", ["sym"]), className="inline-check"),
            ], className="row", style={"marginTop": "6px"}),
            html.Div(
                "a/r = rect width a or circle radius r · b = rect height (rect only) · "
                "l = length · N = modes · εr = permittivity · σ = wall conductivity. "
                "Geometry in mm, σ in S/m.",
                className="hint",
            ),
        ], className="group"),

        html.Fieldset([
            html.Legend("Solver settings"),
            html.Div([
                # One pipeline-wide "use about this many cores" budget: caps the
                # BLAS threads of the coupling-matrix build and the S-matrix sweep.
                _field("nproc (BLAS threads)",
                       dcc.Input(id="nproc", type="number",
                                 value=d.get("nproc", 8),
                                 className="num-input")),
            ], className="row"),
            html.Div([
                dcc.Checklist(
                    id="cm-cache-enable",
                    options=[{"label": " cache coupling matrices on disk", "value": "on"}],
                    value=d.get("cm_cache_enable", []), className="inline-check"),
                dcc.Input(id="cm-cache-dir", type="text",
                          value=d.get("cm_cache_dir", ".pwmma_cm_cache"),
                          placeholder="cache directory", className="grow-input"),
            ], className="row", style={"marginTop": "6px"}),
            html.Div([
                html.Button("Prune cache", id="prune-cache", n_clicks=0,
                            className="btn-small"),
                html.Span(id="prune-status", className="hint"),
            ], className="row", style={"marginTop": "6px"}),
        ], className="group"),

        html.Fieldset([
            html.Legend("Frequency sweep"),
            html.Div([
                _field("start", dcc.Input(id="f-start", type="number",
                                          value=d.get("f_start", 28.0),
                                          className="num-input")),
                html.Span("–", className="lbl range-dash"),
                _field("stop", dcc.Input(id="f-stop", type="number",
                                         value=d.get("f_stop", 34.0),
                                         className="num-input")),
                html.Span("GHz ×", className="lbl range-dash"),
                _field("points", dcc.Input(id="f-n", type="number",
                                           value=d.get("f_n", 61),
                                           className="num-input")),
            ], className="row"),
        ], className="group"),
    ], className="left-panel")


def _right_panel() -> html.Div:
    return html.Div([
        dcc.Tabs(id="result-tab", value="energy", className="eda-tabs", children=[
            dcc.Tab(label="S-parameters", value="spars",
                    className="eda-tab", selected_className="eda-tab--selected"),
            dcc.Tab(label="Energy coupling", value="energy",
                    className="eda-tab", selected_className="eda-tab--selected"),
        ]),
        html.Div([
            html.Div([
                html.Span("output mode(s)", className="lbl"),
                dcc.Dropdown(id="spars-out-modes", multi=True, value=[0],
                             options=[{"label": "0", "value": 0}], clearable=False,
                             className="dd-modes"),
                html.Span("input mode(s)", className="lbl"),
                dcc.Dropdown(id="spars-in-modes", multi=True, value=[0],
                             options=[{"label": "0", "value": 0}], clearable=False,
                             className="dd-modes"),
                html.Span("· |S11| reflection + |S21| transmission, dB",
                          className="hint"),
            ], id="spars-controls", className="energy-controls-row"),
            html.Div([
                html.Span("section", className="lbl"),
                dcc.Dropdown(id="section-select", placeholder="section",
                             className="dd-small"),
                dcc.RadioItems(id="plot-kind", options=["line", "heatmap"],
                               value="line", inline=True, className="inline-check"),
                html.Div(dcc.Slider(id="mode-threshold", min=0.0, max=0.2,
                                    step=0.01, value=0.04), className="slider-box"),
                dcc.Checklist(id="db-toggle", options=[{"label": " dB", "value": "db"}],
                              value=["db"], className="inline-check"),
                html.Span("mode", className="lbl"),
                dcc.Dropdown(id="mode-type-filter", options=["TE", "TM"],
                             placeholder="TE+TM", className="dd-small dd-xs"),
                dcc.Input(id="m-filter", type="number", placeholder="m", min=0,
                          debounce=True, className="num-xs"),
                dcc.Input(id="n-filter", type="number", placeholder="n", min=0,
                          debounce=True, className="num-xs"),
                html.Span("band", className="lbl"),
                dcc.Input(id="ef-lo", type="number", placeholder="min",
                          debounce=True, className="num-xs"),
                html.Span("–", className="lbl"),
                dcc.Input(id="ef-hi", type="number", placeholder="max",
                          debounce=True, className="num-xs"),
                html.Span("GHz", className="lbl"),
            ], id="energy-controls", className="energy-controls-row"),
            dcc.Graph(id="sparam-graph", className="result-graph", responsive=True),
            dcc.Graph(id="energy-graph", className="result-graph", responsive=True),
            html.Div(id="status", className="error-text"),
        ], className="plot-frame"),
    ], className="right-panel")


def _statusbar() -> html.Div:
    # Three blocks: fixed-width status word + LED (so the bar never shifts),
    # the bar itself, then the dynamic run info alongside the config summary.
    return html.Div([
        html.Div([
            html.Span(id="run-led", className="led led-idle"),
            html.Span("idle", id="run-status"),
        ], className="cell cell-status"),
        html.Div([
            html.Progress(id="run-progress", value="0", max="100",
                          className="eda-progress"),
        ], className="cell"),
        html.Div([
            html.Span(id="run-info", className="num run-info"),
            html.Span(id="config-summary", className="num"),
        ], className="cell grow"),
    ], className="statusbar")


def build_layout() -> html.Div:
    d = defaults.load_defaults()
    return html.Div([
        _titlebar(),
        _toolbar(d),
        html.Div([_left_panel(d), _right_panel()], className="app-body"),
        _statusbar(),
        dcc.Store(id="chain-store", data=d["rows"]),
        dcc.Store(id="result-store"),  # holds a cache token + meta, not the heavy arrays
        dcc.Store(id="save-tick"),  # server output for the "save as default" button
    ], id="app-root")
