# src/pwmma/gui/layout.py
"""Static Dash component tree for the pwmma GUI."""
from __future__ import annotations

from dash import dcc, html

from .callbacks import CHAIN_COLUMNS

_DEFAULT_ROWS = [
    {"kind": "rec", "a": 7.112, "b": 3.556, "l": 2.0, "N": 200, "er": "1", "sigma": "5.8e7"},
    {"kind": "cir", "r": 4.2, "l": 1.5, "N": 800, "er": "1", "sigma": "5.8e7"},
    {"kind": "cir", "r": 5.4, "l": 0.26, "N": 800, "er": "9.2", "sigma": "5.8e7"},
]

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
    left = html.Div([
        html.Label("Structure preview", style=_LABEL),
        dcc.Graph(id="structure-preview", config={"displayModeBar": False},
                  style={"height": "216px"}),
        html.Hr(style=_HR),

        html.Label("Waveguide chain", style=_LABEL),
        _chain_header(),
        html.Div(id="chain-rows", style={"height": "200px", "overflowY": "auto",
                                         "border": "1px solid #eee", "padding": "2px"}),
        html.Div([
            html.Button("+ add waveguide", id="add-wg", n_clicks=0),
            dcc.Checklist(id="sym", options=[{"label": " symmetric (sym)", "value": "sym"}],
                          value=["sym"], style={"display": "inline-block"}),
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
            _field("cm nproc", dcc.Input(id="cm-nproc", type="number", value=8, style=_NUM)),
            _field("sm nproc", dcc.Input(id="sm-nproc", type="number", value=8, style=_NUM)),
            _field("precision", dcc.Dropdown(id="precision", options=["single", "double"],
                                            value="single", clearable=False,
                                            style={"width": "110px"})),
            dcc.Checklist(id="use-gpu", options=[{"label": " GPU", "value": "gpu"}],
                          value=["gpu"], style={"paddingBottom": "4px"}),
        ], style=_ROW),
        html.Div([
            html.Span("Coupling-matrix cache", style={"fontSize": "11px", "color": "#888"}),
            dcc.Checklist(id="cm-cache-enable",
                          options=[{"label": " enable disk cache (coming soon)", "value": "on"}],
                          value=[], style={"display": "inline-block"}),
            dcc.Input(id="cm-cache-dir", type="text", placeholder="cache directory",
                      disabled=True, style={"width": "180px"}),
        ], style={"display": "flex", "gap": "8px", "alignItems": "center",
                  "marginTop": "4px", "opacity": "0.6"}),
        html.Hr(style=_HR),

        html.Label("Frequency sweep", style=_LABEL),
        html.Div([
            html.Div([
                _field("start", dcc.Input(id="f-start", type="number", value=28.0, style=_NUM)),
                _field("stop", dcc.Input(id="f-stop", type="number", value=34.0, style=_NUM)),
                _field("points", dcc.Input(id="f-n", type="number", value=61, style=_NUM)),
                html.Span("GHz", style={"color": "#888", "fontSize": "12px",
                                        "paddingBottom": "4px"}),
            ], style={"display": "flex", "gap": "10px", "alignItems": "flex-end"}),
            html.Button("▶ Run", id="run-button", n_clicks=0,
                        style={"padding": "10px 24px", "fontSize": "1.1em", "fontWeight": "bold",
                               "cursor": "pointer", "alignSelf": "flex-end"}),
        ], style={"display": "flex", "gap": "16px", "alignItems": "flex-end",
                  "justifyContent": "space-between", "flexWrap": "wrap"}),
        html.Hr(style=_HR),

        html.Div(id="cm-light", children="⚪ cm: idle",
                 style={"fontSize": "13px", "fontWeight": "bold", "marginBottom": "2px"}),
        html.Progress(id="run-progress", value="0", max="100", style={"width": "100%"}),
        html.Div(id="run-status", style={"fontSize": "12px", "color": "#555", "minHeight": "16px"}),
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
        dcc.Store(id="chain-store", data=_DEFAULT_ROWS),
        dcc.Store(id="result-store"),  # holds a cache token + meta, not the heavy arrays
        html.Div([left, right], style={"display": "flex", "gap": "16px"}),
    ], style={"fontFamily": "sans-serif", "padding": "12px"})
