# src/pwmma/gui/callbacks.py
"""Thin Dash callbacks; all logic delegates to adapter/figures."""
from __future__ import annotations

from dash import ALL, Input, Output, State, ctx, dcc, html

from . import adapter, figures  # noqa: F401  (adapter used by callers that import this module)

_FIELDS = {"rec": ["a", "b", "l", "N", "er", "sigma"], "cir": ["r", "l", "N", "er", "sigma"]}


def render_chain_rows(rows: list[dict]) -> list:
    children = []
    for i, row in enumerate(rows):
        kind = str(row.get("kind", "rec")).lower()
        fields = [
            dcc.Dropdown(
                id={"role": "wg-kind", "i": i},
                options=["rec", "cir"],
                value=kind,
                clearable=False,
                style={"width": "80px"},
            ),
        ]
        for f in _FIELDS[kind]:
            fields.append(
                dcc.Input(
                    id={"role": "wg-field", "i": i, "f": f},
                    type="text",
                    value=str(row.get(f, "")),
                    placeholder=f,
                    style={"width": "60px"},
                )
            )
        fields.append(html.Button("✕", id={"role": "wg-del", "i": i}, n_clicks=0))
        children.append(html.Div(fields, style={"display": "flex", "gap": "4px"}))
    return children


def update_structure_preview(rows, sym_value):
    return figures.structure_preview_figure(rows or [], sym=bool(sym_value))


def register_callbacks(app):
    @app.callback(Output("chain-rows", "children"), Input("chain-store", "data"))
    def _render(rows):
        return render_chain_rows(rows or [])

    @app.callback(
        Output("structure-preview", "figure"),
        Input("chain-store", "data"),
        Input("sym", "value"),
    )
    def _preview(rows, sym_value):
        return update_structure_preview(rows, sym_value)

    @app.callback(
        Output("chain-store", "data"),
        Input("add-wg", "n_clicks"),
        Input({"role": "wg-del", "i": ALL}, "n_clicks"),
        Input({"role": "wg-kind", "i": ALL}, "value"),
        Input({"role": "wg-field", "i": ALL, "f": ALL}, "value"),
        State("chain-store", "data"),
        prevent_initial_call=True,
    )
    def _edit(add_clicks, del_clicks, kinds, field_values, rows):
        rows = list(rows or [])
        trig = ctx.triggered_id
        if trig == "add-wg":
            rows.append(
                {"kind": "cir", "r": 4.2, "l": 1.0, "N": 64, "er": "1", "sigma": "5.8e7"}
            )
            return rows
        if isinstance(trig, dict) and trig.get("role") == "wg-del":
            i = trig["i"]
            if 0 <= i < len(rows):
                rows.pop(i)
            return rows
        # field / kind edits: rebuild from the live input states
        for inp in ctx.inputs_list[3]:  # wg-field inputs
            cid = inp["id"]
            rows[cid["i"]][cid["f"]] = inp.get("value", "")
        for inp in ctx.inputs_list[2]:  # wg-kind inputs
            rows[inp["id"]["i"]]["kind"] = inp.get("value", "cir")
        return rows
