# src/pwmma/gui/layout.py
"""Dash component tree for the pwmma GUI. Built per page load so saved defaults apply."""
from __future__ import annotations

from dash import dcc, html

from . import defaults
from .callbacks import CHAIN_COLUMNS, render_chain_rows

_LABEL = {"fontWeight": "bold", "fontSize": "13px"}
_HR = {"border": "none", "borderTop": "1px solid #ddd", "margin": "6px 0"}
_NUM = {"width": "64px"}
_ROW = {"display": "flex", "gap": "10px", "alignItems": "flex-end", "flexWrap": "wrap"}


def _field(label: str, component) -> html.Div:
    """A small column label stacked above its input."""
    return html.Div(
        [html.Div(label, style={"fontSize": "11px", "color": "#666"}), component],
        style={"display": "flex", "flexDirection": "column"},
    )


def _chain_header() -> html.Div:
    cells = [
        html.Div(name, style={"width": f"{w}px", "fontSize": "11px",
                              "fontWeight": "bold", "color": "#666"})
        for name, w in CHAIN_COLUMNS
    ]
    return html.Div(cells, style={"display": "flex", "gap": "4px", "padding": "2px 0"})


def build_layout() -> html.Div:
    d = defaults.load_defaults()
    left = html.Div([
        html.Label("Structure preview", style=_LABEL),
        dcc.Graph(id="structure-preview", config={"displayModeBar": False},
                  style={"height": "216px"}),
        html.Hr(style=_HR),

        html.Label("Waveguide chain", style=_LABEL),
        _chain_header(),
        html.Div(id="chain-rows", children=render_chain_rows(d["rows"]),
                 style={"height": "200px", "overflowY": "auto",
                        "border": "1px solid #eee", "padding": "2px"}),
        html.Div([
            html.Button("+ add waveguide", id="add-wg", n_clicks=0),
            dcc.Checklist(id="sym", options=[{"label": " symmetric (sym)", "value": "sym"}],
                          value=d.get("sym", ["sym"]), style={"display": "inline-block"}),
        ], style={"display": "flex", "gap": "12px", "alignItems": "center", "marginTop": "4px"}),
        html.Div(
            "a/r = rect width a or circle radius r · b = rect height (rect only) · "
            "l = length · N = modes · εr = permittivity · σ = wall conductivity. "
            "Geometry in mm, σ in S/m.",
            style={"fontSize": "11px", "color": "#888", "marginTop": "2px"},
        ),
        html.Hr(style=_HR),

        html.Label("Config", style=_LABEL),
        html.Div([
            _field("cm nproc", dcc.Input(id="cm-nproc", type="number",
                                         value=d.get("cm_nproc", 8), style=_NUM)),
            _field("sm nproc", dcc.Input(id="sm-nproc", type="number",
                                         value=d.get("sm_nproc", 8), style=_NUM)),
            _field("precision", dcc.Dropdown(id="precision", options=["single", "double"],
                                            value=d.get("precision", "single"), clearable=False,
                                            style={"width": "110px"})),
            dcc.Checklist(id="use-gpu", options=[{"label": " GPU", "value": "gpu"}],
                          value=d.get("use_gpu", ["gpu"]), style={"paddingBottom": "4px"}),
        ], style=_ROW),
        html.Div([
            dcc.Checklist(id="cm-cache-enable",
                          options=[{"label": " cache coupling matrices on disk", "value": "on"}],
                          value=d.get("cm_cache_enable", []), style={"display": "inline-block"}),
            dcc.Input(id="cm-cache-dir", type="text",
                      value=d.get("cm_cache_dir", ".pwmma_cm_cache"),
                      placeholder="cache directory", style={"width": "200px"}),
        ], style={"display": "flex", "gap": "8px", "alignItems": "center", "marginTop": "4px"}),
        html.Hr(style=_HR),

        html.Label("Frequency sweep", style=_LABEL),
        html.Div([
            html.Div([
                _field("start", dcc.Input(id="f-start", type="number",
                                          value=d.get("f_start", 28.0), style=_NUM)),
                _field("stop", dcc.Input(id="f-stop", type="number",
                                         value=d.get("f_stop", 34.0), style=_NUM)),
                _field("points", dcc.Input(id="f-n", type="number",
                                           value=d.get("f_n", 61), style=_NUM)),
                html.Span("GHz", style={"color": "#888", "fontSize": "12px",
                                        "paddingBottom": "4px"}),
            ], style={"display": "flex", "gap": "10px", "alignItems": "flex-end"}),
            html.Div([
                html.Button("▶ Run", id="run-button", n_clicks=0,
                            style={"padding": "10px 24px", "fontSize": "1.1em",
                                   "fontWeight": "bold", "cursor": "pointer"}),
                html.Div([
                    html.Button("save as default", id="save-default", n_clicks=0,
                                style={"fontSize": "11px", "cursor": "pointer"}),
                    html.Span(id="save-status", style={"fontSize": "11px", "color": "#2a7a2a"}),
                ], style={"display": "flex", "gap": "6px", "alignItems": "center",
                          "marginTop": "3px"}),
            ], style={"display": "flex", "flexDirection": "column", "alignItems": "flex-end"}),
        ], style={"display": "flex", "gap": "16px", "alignItems": "flex-end",
                  "justifyContent": "space-between", "flexWrap": "wrap"}),
        html.Hr(style=_HR),

        html.Progress(id="run-progress", value="0", max="100", style={"width": "100%"}),
        html.Div(id="run-status", children="⚪ idle",
                 style={"fontSize": "13px", "fontWeight": "500", "minHeight": "18px",
                        "marginTop": "2px"}),
    ], style={"flex": "0 0 35%", "minWidth": "480px", "display": "flex",
              "flexDirection": "column", "gap": "6px"})

    right = html.Div([
        dcc.Tabs(id="result-tab", value="energy", children=[
            dcc.Tab(label="S-parameters", value="spars"),
            dcc.Tab(label="Energy coupling", value="energy"),
        ]),
        html.Div([
            dcc.Dropdown(id="section-select", placeholder="section"),
            dcc.RadioItems(id="plot-kind", options=["line", "heatmap"], value="line", inline=True),
            dcc.Slider(id="mode-threshold", min=0.0, max=0.2, step=0.01, value=0.04),
            dcc.Checklist(id="db-toggle", options=[{"label": "dB", "value": "db"}], value=["db"]),
        ], id="energy-controls"),
        dcc.Graph(id="sparam-graph"),
        dcc.Graph(id="energy-graph"),
        html.Div(id="status", style={"whiteSpace": "pre-wrap", "color": "#a00"}),
    ], style={"flex": "1", "display": "flex", "flexDirection": "column", "gap": "8px"})

    return html.Div([
        html.H3("pwmma — Mode-Matching Analyzer"),
        dcc.Store(id="chain-store", data=d["rows"]),
        dcc.Store(id="result-store"),  # holds a cache token + meta, not the heavy arrays
        html.Div([left, right], style={"display": "flex", "gap": "16px"}),
    ], style={"fontFamily": "sans-serif", "padding": "12px"})
