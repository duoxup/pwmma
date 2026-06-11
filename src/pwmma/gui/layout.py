# src/pwmma/gui/layout.py
"""Static Dash component tree for the pwmma GUI."""
from __future__ import annotations

from dash import dcc, html

_DEFAULT_ROWS = [
    {"kind": "rec", "a": 7.112, "b": 3.556, "l": 2.0, "N": 200, "er": "1", "sigma": "5.8e7"},
    {"kind": "cir", "r": 4.2, "l": 1.5, "N": 800, "er": "1", "sigma": "5.8e7"},
    {"kind": "cir", "r": 5.4, "l": 0.26, "N": 800, "er": "9.2", "sigma": "5.8e7"},
]


def build_layout() -> html.Div:
    left = html.Div([
        html.Label("Structure preview"),
        dcc.Graph(id="structure-preview", config={"displayModeBar": False}),
        html.Hr(),
        html.Label("Waveguide chain"),
        html.Div(id="chain-rows"),
        html.Button("+ add waveguide", id="add-wg", n_clicks=0),
        dcc.Checklist(id="sym", options=[{"label": "symmetric (sym)", "value": "sym"}],
                      value=["sym"]),
        html.Hr(),
        html.Label("Frequency sweep (GHz)"),
        html.Div([
            dcc.Input(id="f-start", type="number", value=28.0, placeholder="start"),
            dcc.Input(id="f-stop", type="number", value=34.0, placeholder="stop"),
            dcc.Input(id="f-n", type="number", value=61, placeholder="N"),
        ]),
        html.Hr(),
        html.Label("Config"),
        html.Div([
            dcc.Input(id="cm-nproc", type="number", value=8),
            dcc.Input(id="sm-nproc", type="number", value=8),
            dcc.Checklist(id="use-gpu", options=[{"label": "GPU", "value": "gpu"}], value=["gpu"]),
            dcc.Dropdown(id="precision", options=["complex64", "complex128"], value="complex64",
                         clearable=False),
        ]),
        html.Hr(),
        html.Button("▶ Run", id="run-button", n_clicks=0),
        html.Progress(id="run-progress", value="0", max="100"),
    ], style={"flex": "0 0 40%", "display": "flex", "flexDirection": "column", "gap": "8px"})

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
