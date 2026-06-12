# src/pwmma/gui/callbacks.py
"""Thin Dash callbacks; all logic delegates to adapter/figures."""
from __future__ import annotations

from dash import ALL, Input, Output, State, ctx, dcc, html

from . import adapter, figures

# Column spec shared with layout's header so the rows line up like a table.
# (label, pixel width). 'a/r' holds rectangular width a or circular radius r;
# 'b' is rectangular-only (shown disabled for circular guides).
CHAIN_COLUMNS = [("Type", 72), ("a/r", 60), ("b", 60), ("l", 52),
                 ("N", 52), ("εr", 52), ("σ", 68)]
_TRAILING_FIELDS = [("l", 52), ("N", 52), ("er", 52), ("sigma", 68)]


def render_chain_rows(rows: list[dict]) -> list:
    children = []
    for i, row in enumerate(rows):
        kind = str(row.get("kind", "rec")).lower()
        first = "a" if kind == "rec" else "r"
        cells = [
            dcc.Dropdown(id={"role": "wg-kind", "i": i}, options=["rec", "cir"],
                         value=kind, clearable=False, style={"width": "72px"}),
            dcc.Input(id={"role": "wg-field", "i": i, "f": first}, type="text",
                      value=str(row.get(first, "")), style={"width": "60px"}),
        ]
        if kind == "rec":
            cells.append(dcc.Input(id={"role": "wg-field", "i": i, "f": "b"}, type="text",
                                   value=str(row.get("b", "")), style={"width": "60px"}))
        else:
            cells.append(dcc.Input(value="—", disabled=True, style={"width": "60px"}))
        for f, w in _TRAILING_FIELDS:
            cells.append(dcc.Input(id={"role": "wg-field", "i": i, "f": f}, type="text",
                                   value=str(row.get(f, "")), style={"width": f"{w}px"}))
        cells.append(html.Button("✕", id={"role": "wg-del", "i": i}, n_clicks=0,
                                 style={"width": "28px"}))
        children.append(html.Div(cells, style={"display": "flex", "gap": "4px",
                                               "alignItems": "center", "marginBottom": "3px"}))
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


def compute_payload(form: dict, progress_callback):
    """Pure compute step used by the background callback. Returns (payload, error_str)."""
    try:
        chain = adapter.parse_chain(form["rows"], form["sym"])
        freqs = adapter.parse_freqs(form["f_start"], form["f_stop"], form["f_n"])
        cfg = adapter.parse_config(
            {"nproc": form["cm_nproc"]},
            {"nproc": form["sm_nproc"], "use_gpu": form["use_gpu"], "precision": form["precision"]},
        )
    except adapter.GuiInputError as exc:
        return None, str(exc)
    try:
        n_phys = len(chain.wgs) + (len(chain.wgs) - 1 if chain.sym else 0)
        sections = list(range(1, n_phys - 1)) or None
        result = adapter.run_energy(chain, freqs, cfg, sections=sections,
                                    progress_callback=progress_callback)
        spars = adapter.run_spars(chain, freqs, cfg)
    except (NotImplementedError, ValueError) as exc:
        return None, f"computation failed: {exc}"
    return {
        "section_indices": list(result.section_indices),
        "result": result, "spars": spars,
    }, None


def register_run_callback(app):
    @app.callback(
        output=[Output("result-store", "data"), Output("status", "children")],
        inputs=Input("run-button", "n_clicks"),
        state=[State("chain-store", "data"), State("sym", "value"),
               State("f-start", "value"), State("f-stop", "value"), State("f-n", "value"),
               State("cm-nproc", "value"), State("sm-nproc", "value"),
               State("use-gpu", "value"), State("precision", "value")],
        background=True,
        running=[(Output("run-button", "disabled"), True, False)],
        progress=[Output("run-status", "children"),
                  Output("run-progress", "value"), Output("run-progress", "max")],
        prevent_initial_call=True,
    )
    def _run(set_progress, n_clicks, rows, sym, f_start, f_stop, f_n,
             cm_nproc, sm_nproc, use_gpu, precision):
        form = {"rows": rows, "sym": bool(sym), "f_start": f_start, "f_stop": f_stop,
                "f_n": f_n, "cm_nproc": cm_nproc, "sm_nproc": sm_nproc,
                "use_gpu": bool(use_gpu), "precision": precision}
        total = str(int(f_n) if f_n else 1)
        set_progress(("Preparing coupling matrices…", "0", total))

        def progress(done, tot):
            set_progress((f"Sweeping frequencies {done}/{tot}", str(done), str(tot)))

        payload, error = compute_payload(form, progress)
        if error is not None:
            return None, error
        token = f"run-{n_clicks}"
        app._pwmma_cache.set(token, payload)
        return {"token": token, "section_indices": payload["section_indices"]}, ""


def render_spars(payload):
    return figures.sparam_figure(payload["spars"], excitation_mode=0)


def render_energy(payload, *, section, kind, threshold, db):
    sec = payload["result"].get_section(int(section))
    if kind == "heatmap":
        return figures.energy_heatmap_figure(sec)
    return figures.energy_line_figure(sec, mode_threshold=float(threshold), dB=bool(db))


def tab_visibility(tab):
    """Styles for (sparam-graph, energy-graph, energy-controls) given the active tab."""
    show, hide = {"display": "block"}, {"display": "none"}
    if tab == "spars":
        return show, hide, hide
    return hide, show, show


def register_plot_callbacks(app):
    def _payload(token):
        return app._pwmma_cache.get(token) if token else None

    @app.callback(
        Output("sparam-graph", "style"), Output("energy-graph", "style"),
        Output("energy-controls", "style"), Input("result-tab", "value"),
    )
    def _toggle_tab(tab):
        return tab_visibility(tab)

    @app.callback(Output("section-select", "options"), Output("section-select", "value"),
                  Input("result-store", "data"))
    def _sections(data):
        if not data:
            return [], None
        idx = data["section_indices"]
        return [{"label": str(i), "value": i} for i in idx], (idx[len(idx) // 2] if idx else None)

    @app.callback(Output("sparam-graph", "figure"), Input("result-store", "data"))
    def _spars(data):
        p = _payload(data and data.get("token"))
        return figures.sparam_figure(p["spars"]) if p else figures.empty_figure("Run first")

    @app.callback(
        Output("energy-graph", "figure"),
        Input("result-store", "data"), Input("section-select", "value"),
        Input("plot-kind", "value"), Input("mode-threshold", "value"), Input("db-toggle", "value"),
    )
    def _energy(data, section, kind, threshold, db):
        p = _payload(data and data.get("token"))
        if not p or section is None:
            return figures.empty_figure("Run first")
        return render_energy(p, section=section, kind=kind, threshold=threshold, db=bool(db))
