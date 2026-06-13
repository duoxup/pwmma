# src/pwmma/gui/callbacks.py
"""Thin Dash callbacks; all logic delegates to adapter/figures."""
from __future__ import annotations

import dataclasses
import re

import numpy as np
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from . import adapter, defaults, figures

# Column spec (label, flex) shared by render_chain_header and render_chain_rows
# so the columns line up like a table. 'Type' has a fixed basis; the parameter
# fields flex to fill the block (equal width) with 'N' ~20% wider. 'a/r' holds
# rectangular width a or circular radius r; 'b' is rectangular-only (disabled
# for circular guides).
CHAIN_COLUMNS = [("Type", "0 0 72px"), ("a/r", "1"), ("b", "1"), ("l", "1"),
                 ("N", "1.2"), ("εr", "1"), ("σ", "1")]
_TRAILING_FIELDS = [("l", "1"), ("N", "1.2"), ("er", "1"), ("sigma", "1")]


def render_chain_header() -> html.Div:
    """Sticky grid-header row; shares CHAIN_COLUMNS flex values with the rows."""
    cells = [html.Div(name, style={"flex": fl}) for name, fl in CHAIN_COLUMNS]
    cells.append(html.Div(style={"flex": "0 0 26px"}))  # over the delete column
    return html.Div(cells, className="chain-head")


def render_chain_rows(rows: list[dict]) -> list:
    children = []
    for i, row in enumerate(rows):
        kind = str(row.get("kind", "rec")).lower()
        first = "a" if kind == "rec" else "r"

        def cell(f: str, fl: str, *, i=i, row=row):
            return dcc.Input(id={"role": "wg-field", "i": i, "f": f}, type="text",
                             debounce=True, value=str(row.get(f, "")), style={"flex": fl})

        cells = [
            html.Div(dcc.Dropdown(id={"role": "wg-kind", "i": i}, options=["rec", "cir"],
                                  value=kind, clearable=False, className="chain-kind"),
                     style={"flex": "0 0 72px"}),
            cell(first, "1"),
        ]
        if kind == "rec":
            cells.append(cell("b", "1"))
        else:
            cells.append(dcc.Input(value="—", disabled=True, style={"flex": "1"}))
        for f, fl in _TRAILING_FIELDS:
            cells.append(cell(f, fl))
        cells.append(html.Button("✕", id={"role": "wg-del", "i": i}, n_clicks=0,
                                 className="chain-del"))
        children.append(html.Div(cells, className="chain-row"))
    return children


def update_structure_preview(rows, sym_value):
    return figures.structure_preview_figure(rows or [], sym=bool(sym_value))


def register_callbacks(app):
    @app.callback(
        Output("structure-preview", "figure"),
        Input("chain-store", "data"),
        Input("sym", "value"),
    )
    def _preview(rows, sym_value):
        return update_structure_preview(rows, sym_value)

    @app.callback(
        Output("chain-store", "data"),
        Output("chain-rows", "children"),
        Input("add-wg", "n_clicks"),
        Input({"role": "wg-del", "i": ALL}, "n_clicks"),
        Input({"role": "wg-kind", "i": ALL}, "value"),
        Input({"role": "wg-field", "i": ALL, "f": ALL}, "value"),
        State("chain-store", "data"),
        prevent_initial_call=True,
    )
    def _edit(add_clicks, del_clicks, kinds, field_values, rows):
        rows = list(rows or [])
        # sync the live input values into the chain data
        for inp in ctx.inputs_list[3]:  # wg-field inputs
            cid = inp["id"]
            if cid["i"] < len(rows):
                rows[cid["i"]][cid["f"]] = inp.get("value", "")
        for inp in ctx.inputs_list[2]:  # wg-kind inputs
            if inp["id"]["i"] < len(rows):
                rows[inp["id"]["i"]]["kind"] = inp.get("value", "cir")
        trig = ctx.triggered_id
        if trig == "add-wg":
            rows.append({"kind": "cir", "r": 4.2, "l": 1.0, "N": 64, "er": "1", "sigma": "5.8e7"})
            return rows, render_chain_rows(rows)
        if isinstance(trig, dict) and trig.get("role") in ("wg-del", "wg-kind"):
            if trig["role"] == "wg-del":
                i = trig["i"]
                if 0 <= i < len(rows):
                    rows.pop(i)
            # row removed, or kind switched (different fields) -> rebuild that area
            return rows, render_chain_rows(rows)
        # a plain value edit: update data only, keep the inputs (no rebuild = no focus loss)
        return rows, no_update

    @app.callback(
        Output("save-tick", "data"),
        Input("save-default", "n_clicks"),
        State("chain-store", "data"), State("sym", "value"),
        State("f-start", "value"), State("f-stop", "value"), State("f-n", "value"),
        State("cm-nproc", "value"), State("sm-nproc", "value"),
        State("use-gpu", "value"), State("precision", "value"),
        State("cm-cache-enable", "value"), State("cm-cache-dir", "value"),
        State("compute-select", "value"),
        prevent_initial_call=True,
    )
    def _save_default(n, rows, sym, f_start, f_stop, f_n, cm_nproc, sm_nproc,
                      use_gpu, precision, cache_enable, cache_dir, compute):
        defaults.save_defaults({
            "rows": rows, "sym": list(sym or []),
            "f_start": f_start, "f_stop": f_stop, "f_n": f_n,
            "cm_nproc": cm_nproc, "sm_nproc": sm_nproc,
            "use_gpu": list(use_gpu or []), "precision": precision,
            "compute": compute,
            "cm_cache_enable": list(cache_enable or []), "cm_cache_dir": cache_dir,
        })
        return n

    @app.callback(
        Output("config-summary", "children"),
        Input("use-gpu", "value"), Input("precision", "value"),
        Input("cm-nproc", "value"), Input("sm-nproc", "value"),
    )
    def _config_summary(use_gpu, precision, cm_nproc, sm_nproc):
        device = "GPU" if use_gpu else "CPU"
        cm = "—" if cm_nproc is None else cm_nproc
        sm = "—" if sm_nproc is None else sm_nproc
        return f"{device} · {precision or '—'} · {cm}/{sm} proc"

    @app.callback(
        Output("prune-status", "children"),
        Input("prune-cache", "n_clicks"),
        State("cm-cache-dir", "value"),
        prevent_initial_call=True,
    )
    def _prune(n, cache_dir):
        # Removes coupling-matrix files that a larger cached matrix subsumes.
        # Best run when idle; a concurrent worker whose file vanishes recomputes.
        if not cache_dir:
            return "set a cache directory first"
        try:
            return format_prune_summary(adapter.prune_cache(cache_dir))
        except OSError as exc:
            return f"prune failed: {exc}"

    # transient button feedback: flash "saved ✓" then revert (browser-side timer)
    app.clientside_callback(
        """
        function(n) {
            if (n) {
                var b = document.getElementById('save-default');
                if (b) {
                    b.textContent = 'saved ✓';
                    setTimeout(function() { b.textContent = '💾 Save as default'; }, 1400);
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("save-default", "children"),
        Input("save-default", "n_clicks"),
        prevent_initial_call=True,
    )


def sweep_progress(done, total, f_start, f_stop, f_n):
    """Status-bar payload (text, bar value, bar max, LED class) for one sweep tick.

    ``done`` counts 1..2n across the energy and S-parameter sweeps, so the
    displayed frequency runs through the sweep twice.
    """
    text = f"sweeping {done}/{total}"
    try:
        n = int(f_n)
        idx = (int(done) - 1) % n
        f = float(f_start) + idx * (float(f_stop) - float(f_start)) / max(n - 1, 1)
        text += f" — {f:.3f} GHz"
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return text, str(done), str(total), "led led-sweep"


def compute_selection(mode):
    """``(do_spars, do_energy)`` for a ``compute-select`` value.

    ``None`` or any unknown value falls back to computing both, so older saved
    defaults and callers that omit the field keep the original behaviour.
    """
    do_spars = mode in ("both", "spars")
    do_energy = mode in ("both", "energy")
    if not (do_spars or do_energy):
        return True, True
    return do_spars, do_energy


def result_tab_for(mode):
    """Tab to activate after a run, or ``None`` to leave the user's choice.

    A single-result run jumps to the tab that actually has data; ``"both"``
    keeps whatever tab the user is on.
    """
    if mode == "spars":
        return "spars"
    if mode == "energy":
        return "energy"
    return None


def format_prune_summary(summary: dict) -> str:
    """Human-readable status line for a prune_coupling_matrix_cache() summary."""
    msg = (f"pruned {len(summary['removed'])} file(s), "
           f"freed {summary['freed_bytes'] / 1e6:.1f} MB · kept {summary['kept']}")
    if summary.get("temp_removed"):
        msg += f" · cleaned {len(summary['temp_removed'])} temp(s)"
    return msg


def compute_payload(form: dict, progress_callback, status_callback=None):
    """Pure compute step used by the background callback. Returns (payload, error_str).

    ``form["compute"]`` selects which sweeps run ("both" / "spars" / "energy");
    a missing value means both. Coupling matrices are computed once and shared by
    whichever sweeps run. ``status_callback(phase)`` (optional) is called with
    ``"cm"`` before the coupling-matrix computation and ``"sweep"`` before the
    frequency sweeps, so the UI can show the current phase. The skipped result
    type is stored as ``None``.
    """
    try:
        chain = adapter.parse_chain(form["rows"], form["sym"])
        freqs = adapter.parse_freqs(form["f_start"], form["f_stop"], form["f_n"])
        cfg = adapter.parse_config(
            {"nproc": form["cm_nproc"], "cache_dir": form.get("cm_cache_dir"),
             "cache_enabled": form.get("cm_cache_enabled")},
            {"nproc": form["sm_nproc"], "use_gpu": form["use_gpu"], "precision": form["precision"]},
        )
    except adapter.GuiInputError as exc:
        return None, str(exc)
    compute = form.get("compute", "both")
    do_spars, do_energy = compute_selection(compute)
    try:
        n_phys = len(chain.wgs) + (len(chain.wgs) - 1 if chain.sym else 0)
        sections = list(range(1, n_phys - 1)) or None
        n = len(freqs)
        total = (int(do_spars) + int(do_energy)) * n  # sweeps share one progress bar
        if status_callback:
            status_callback("cm")
        cms = adapter.compute_cms(chain, cfg)
        if status_callback:
            status_callback("sweep")
        done = 0
        result = None
        spars = None
        if do_energy:
            result = adapter.run_energy(
                chain, freqs, cfg, sections=sections, cms=cms,
                progress_callback=lambda d, _t: progress_callback(d, total))
            done = n
        if do_spars:
            spars = adapter.run_spars(
                chain, freqs, cfg, cms=cms,
                progress_callback=lambda d, _t: progress_callback(done + d, total))
    except (NotImplementedError, ValueError) as exc:
        return None, f"computation failed: {exc}"
    return {
        "section_indices": list(result.section_indices) if result is not None else [],
        "result": result, "spars": spars, "compute": compute,
    }, None


def register_run_callback(app):
    @app.callback(
        output=[Output("result-store", "data"), Output("status", "children")],
        inputs=Input("run-button", "n_clicks"),
        state=[State("chain-store", "data"), State("sym", "value"),
               State("f-start", "value"), State("f-stop", "value"), State("f-n", "value"),
               State("cm-nproc", "value"), State("sm-nproc", "value"),
               State("use-gpu", "value"), State("precision", "value"),
               State("cm-cache-enable", "value"), State("cm-cache-dir", "value"),
               State("compute-select", "value")],
        background=True,
        running=[(Output("run-button", "disabled"), True, False),
                 (Output("stop-button", "disabled"), False, True)],
        cancel=[Input("stop-button", "n_clicks")],
        progress=[Output("run-status", "children"),
                  Output("run-progress", "value"), Output("run-progress", "max"),
                  Output("run-led", "className")],
        prevent_initial_call=True,
    )
    def _run(set_progress, n_clicks, rows, sym, f_start, f_stop, f_n,
             cm_nproc, sm_nproc, use_gpu, precision, cm_cache_enable, cm_cache_dir,
             compute):
        form = {"rows": rows, "sym": bool(sym), "f_start": f_start, "f_stop": f_stop,
                "f_n": f_n, "cm_nproc": cm_nproc, "sm_nproc": sm_nproc,
                "use_gpu": bool(use_gpu), "precision": precision,
                "cm_cache_enabled": bool(cm_cache_enable), "cm_cache_dir": cm_cache_dir,
                "compute": compute}
        n_sweeps = sum(compute_selection(compute))
        bar_max = str(n_sweeps * int(f_n)) if f_n else "1"

        def status(phase):
            if phase == "cm":
                set_progress(("computing coupling matrices…", "0", bar_max, "led led-cm"))
            else:  # "sweep"
                set_progress(("sweeping frequencies…", "0", bar_max, "led led-sweep"))

        def progress(done, tot):
            set_progress(sweep_progress(done, tot, f_start, f_stop, f_n))

        payload, error = compute_payload(form, progress, status_callback=status)
        if error is not None:
            set_progress(("error — see message panel", "0", bar_max, "led led-error"))
            return None, error
        token = f"run-{n_clicks}"
        app._pwmma_cache.set(token, payload)
        set_progress(("done", bar_max, bar_max, "led led-done"))
        return {"token": token, "section_indices": payload["section_indices"],
                "compute": payload["compute"]}, ""

    # Stop kills the background job via `cancel` above; the job can no longer
    # report, so this regular callback resets the status-bar cells instead.
    @app.callback(
        Output("run-status", "children", allow_duplicate=True),
        Output("run-progress", "value", allow_duplicate=True),
        Output("run-led", "className", allow_duplicate=True),
        Input("stop-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def _stopped(n):
        return "stopped", "0", "led led-idle"


def render_spars(payload):
    if payload.get("spars") is None:
        return figures.empty_figure("S-parameters not computed for this run")
    return figures.sparam_figure(payload["spars"], excitation_mode=0)


# Matches both rectangular ("TE1,0") and circular ("TM0,1S") mode labels.
_MODE_LABEL_RE = re.compile(r"^(TE|TM)(\d+),(\d+)([CS])?$")


def mode_filter_mask(labels, *, mode_type=None, m=None, n=None):
    """Boolean mask over mode ``labels``; criteria left as None pass everything.

    With any criterion active, labels that don't parse as TE/TM mode names
    (the no-waveguide fallback of stringified ids) are excluded.
    """
    if mode_type is None and m is None and n is None:
        return np.ones(len(labels), dtype=bool)
    mask = np.zeros(len(labels), dtype=bool)
    for i, label in enumerate(labels):
        parsed = _MODE_LABEL_RE.match(label)
        if parsed is None:
            continue
        if mode_type is not None and parsed.group(1) != mode_type:
            continue
        if m is not None and int(parsed.group(2)) != int(m):
            continue
        if n is not None and int(parsed.group(3)) != int(n):
            continue
        mask[i] = True
    return mask


def band_slice(section, f_lo, f_hi):
    """Restrict a section to the [f_lo, f_hi] band (GHz; None = unbounded).

    Returns the section unchanged when no bound is active, None when no sweep
    point falls inside the band. Dominant-mode ranking on the sliced copy then
    only sees the selected band.
    """
    if f_lo is None and f_hi is None:
        return section
    keep = np.ones(len(section.freqs), dtype=bool)
    if f_lo is not None:
        keep &= section.freqs >= float(f_lo) * 1e9
    if f_hi is not None:
        keep &= section.freqs <= float(f_hi) * 1e9
    if not keep.any():
        return None
    if keep.all():
        return section
    per_freq = ["freqs", "modal_power", "propagating_mask", "evanescent_mask",
                "forward_left", "backward_left", "forward_right", "backward_right",
                "reflection_power", "total_reflected_power", "transmission_power",
                "power_balance"]
    return dataclasses.replace(section, **{f: getattr(section, f)[keep] for f in per_freq})


def render_energy(payload, *, section, kind, threshold, db,
                  mode_type=None, m=None, n=None, f_lo=None, f_hi=None):
    if payload.get("result") is None:
        return figures.empty_figure("Mode analysis not computed for this run")
    sec = band_slice(payload["result"].get_section(int(section)), f_lo, f_hi)
    if sec is None:
        return figures.empty_figure("no sweep points in the selected band")
    allowed = mode_filter_mask(sec.get_mode_labels(), mode_type=mode_type, m=m, n=n)
    if kind == "heatmap":
        return figures.energy_heatmap_figure(
            sec, mode_mask=None if allowed.all() else allowed)
    dominant = sec.dominant_mode_ids(threshold=float(threshold))
    return figures.energy_line_figure(
        sec, dB=bool(db), mode_ids=dominant[allowed[dominant]])


def tab_visibility(tab):
    """Styles for (sparam-graph, energy-graph, energy-controls) given the active tab.

    Graphs are display-toggled (the hidden one collapses). The energy controls
    use visibility instead, so their vertical space is reserved in both tabs and
    the graph below them keeps a stable position when switching tabs.
    """
    show, hide = {"display": "block"}, {"display": "none"}
    if tab == "spars":
        return show, hide, {"visibility": "hidden"}
    return hide, show, {"visibility": "visible"}


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
        return render_spars(p) if p else figures.empty_figure("Run first")

    @app.callback(
        Output("result-tab", "value", allow_duplicate=True),
        Input("result-store", "data"),
        prevent_initial_call=True,
    )
    def _switch_tab(data):
        # Jump to the tab that has data after a single-result run; leave the
        # user's choice for a "both" run.
        tab = result_tab_for(data.get("compute")) if data else None
        return tab if tab is not None else no_update

    @app.callback(
        Output("energy-graph", "figure"),
        Input("result-store", "data"), Input("section-select", "value"),
        Input("plot-kind", "value"), Input("mode-threshold", "value"), Input("db-toggle", "value"),
        Input("mode-type-filter", "value"), Input("m-filter", "value"),
        Input("n-filter", "value"), Input("ef-lo", "value"), Input("ef-hi", "value"),
    )
    def _energy(data, section, kind, threshold, db, mode_type, m, n, f_lo, f_hi):
        p = _payload(data and data.get("token"))
        if not p:
            return figures.empty_figure("Run first")
        if p.get("result") is not None and section is None:
            return figures.empty_figure("Run first")
        return render_energy(p, section=section, kind=kind, threshold=threshold, db=bool(db),
                             mode_type=mode_type, m=m, n=n, f_lo=f_lo, f_hi=f_hi)
