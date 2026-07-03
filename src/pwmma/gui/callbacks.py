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
        State("nproc", "value"),
        State("use-gpu", "value"), State("precision", "value"),
        State("cm-cache-enable", "value"), State("cm-cache-dir", "value"),
        State("compute-select", "value"), State("sweep-mode", "value"),
        prevent_initial_call=True,
    )
    def _save_default(n, rows, sym, f_start, f_stop, f_n, nproc,
                      use_gpu, precision, cache_enable, cache_dir, compute, sweep):
        defaults.save_defaults({
            "rows": rows, "sym": list(sym or []),
            "f_start": f_start, "f_stop": f_stop, "f_n": f_n,
            "nproc": nproc,
            "use_gpu": list(use_gpu or []), "precision": precision,
            "compute": compute, "sweep": sweep_mode_value(sweep),
            "cm_cache_enable": list(cache_enable or []), "cm_cache_dir": cache_dir,
        })
        return n

    @app.callback(
        Output("config-summary", "children"),
        Input("use-gpu", "value"), Input("precision", "value"),
        Input("nproc", "value"),
    )
    def _config_summary(use_gpu, precision, nproc):
        device = "GPU" if use_gpu else "CPU"
        n = "—" if nproc is None else nproc
        return f"{device} · {precision or '—'} · {n} BLAS threads"

    @app.callback(
        Output("compute-select", "options"),
        Output("compute-select", "value"),
        Output("f-n", "disabled"),
        Input("sweep-mode", "value"),
        State("compute-select", "value"),
    )
    def _sweep_compute_guard(sweep_check, compute):
        # compute=energy + sweep=adaptive cannot exist: the S11 fit loop IS the
        # sampler. Disable the option under adaptive and lift a stranded
        # energy-only selection to Both (compute_payload coerces defensively
        # too). "points" (f_n) is unused by adaptive -> grey it out as well.
        energy_off = sweep_mode_value(sweep_check) == "adaptive"
        options = [{"label": "Both", "value": "both"},
                   {"label": "S-parameters", "value": "spars"},
                   {"label": "Mode analysis", "value": "energy",
                    "disabled": energy_off}]
        value = "both" if (energy_off and compute == "energy") else no_update
        return options, value, energy_off

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


# Status-bar payloads are 6-tuples matching the background callback's progress
# outputs: (status word, bar value, bar max, LED class, info text, bar class).
# Cell 1 holds only the short fixed status word (its width never changes, so
# the bar stops jumping); the dynamic detail lives in cell 3's info text.
_BAR = "eda-progress"
_BAR_DIM = "eda-progress dim"


def sweep_progress(done, total, f_start, f_stop, f_n):
    """Statusbar payload for one uniform-sweep tick.

    ``done`` counts 1..2n across the energy and S-parameter sweeps, so the
    displayed frequency runs through the sweep twice.
    """
    info = f"{done}/{total}"
    try:
        n = int(f_n)
        idx = (int(done) - 1) % n
        f = float(f_start) + idx * (float(f_stop) - float(f_start)) / max(n - 1, 1)
        info = f"{f:.3f} GHz ({done}/{total})"
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return "sweeping", str(done), str(total), "led led-sweep", info, _BAR


def sweep_mode_value(checklist) -> str:
    """Internal sweep string for the adaptive checkbox ('adaptive' iff ticked).

    The form/payload/defaults keep the "uniform"/"adaptive" string so saved
    state stays widget-agnostic."""
    return "adaptive" if (checklist and "adaptive" in checklist) else "uniform"


def adaptive_progress(k, f_hz):
    """Statusbar payload for one adaptive solve: no meaningful total exists, so
    the bar is greyed (dim) and the live detail goes to the info cell."""
    try:
        info = f"solve {int(k)} · {float(f_hz) / 1e9:.3f} GHz"
    except (TypeError, ValueError):
        info = f"solve {k}"
    return "solving", "0", "1", "led led-sweep", info, _BAR_DIM


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
            {"nproc": form["nproc"], "use_gpu": form["use_gpu"],
             "precision": form["precision"],
             "cache_dir": form.get("cm_cache_dir"),
             "cache_enabled": form.get("cm_cache_enabled")},
        )
    except adapter.GuiInputError as exc:
        return None, str(exc)
    compute = form.get("compute", "both")
    sweep = form.get("sweep", "uniform")
    if sweep == "adaptive" and compute == "energy":
        # energy-only cannot drive the adaptive sampler (the S11 fit loop IS
        # the sampler) — the UI prevents this combo; coerce stale saved state.
        compute = "both"
    do_spars, do_energy = compute_selection(compute)
    try:
        n_phys = len(chain.wgs) + (len(chain.wgs) - 1 if chain.sym else 0)
        sections = list(range(1, n_phys - 1)) or None
        if status_callback:
            status_callback("cm")
        cms = adapter.compute_cms(chain, cfg)
        result = None
        spars = None
        spars_model = None
        energy_model = None
        if sweep == "adaptive":
            # spars first — the adaptive loop produces the run's sample grid;
            # energy then reuses that grid (one grid per run).
            solver = adapter.make_solver(chain, cfg, cms=cms)
            if status_callback:
                status_callback("solving")
            spars_model = adapter.run_adaptive_spars(
                solver, freqs[0], freqs[-1], progress_callback=progress_callback,
                cms=cms)
            if do_energy:
                if status_callback:
                    status_callback("sweep")
                grid = np.sort(spars_model["s11"]["F"])
                result = adapter.run_energy(
                    chain, grid, cfg, sections=sections, cms=cms,
                    progress_callback=lambda d, _t: progress_callback(d, len(grid)))
                # dense smoothed per-mode curves (post-processing, no solves);
                # reuse the "solving" phase word while the fits run
                if status_callback:
                    status_callback("solving")
                energy_model = adapter.run_energy_model(result, freqs[0], freqs[-1])
        else:
            solver = adapter.make_solver(chain, cfg, cms=cms) if do_spars else None
            if status_callback:
                status_callback("sweep")
            n = len(freqs)
            total = (int(do_spars) + int(do_energy)) * n  # sweeps share one bar
            done = 0
            if do_energy:
                result = adapter.run_energy(
                    chain, freqs, cfg, sections=sections, cms=cms,
                    progress_callback=lambda d, _t: progress_callback(d, total))
                done = n
            if do_spars:
                spars = adapter.run_spars_on(
                    solver, freqs,
                    progress_callback=lambda d, _t: progress_callback(done + d, total))
    except (NotImplementedError, ValueError) as exc:
        return None, f"computation failed: {exc}"
    if spars is not None:
        n_in = int(spars["s11"].shape[2])
        n_out = int(max(spars["s11"].shape[1], spars["s21"].shape[1]))
    elif spars_model is not None:
        n_in = n_out = 1                       # fundamental-mode line
    else:
        n_in = n_out = 0
    return {
        "section_indices": list(result.section_indices) if result is not None else [],
        "result": result, "spars": spars, "spars_model": spars_model,
        "energy_model": energy_model,
        "compute": compute, "sweep": sweep,
        # mode counts so the S-param panel can offer mode selectors without the
        # heavy arrays: n_in = excitation (port-1) modes, n_out = max over the
        # S11 (port-1) and S21 (port-2) response dimensions.
        "n_in": n_in,
        "n_out": n_out,
    }, None


def store_payload(cache, token, payload) -> bool:
    """Persist ``payload`` under ``token``; return whether it survived.

    The background worker and the UI process share results only through the
    diskcache, so the heavy arrays must round-trip through it. diskcache culls
    any value larger than its ``size_limit`` immediately on ``set`` (the value is
    gone, yet ``set`` still returns truthy) — which otherwise surfaces as a
    confusing 'Run first' placeholder. ``token in cache`` is a cheap key check
    (no deserialize) that detects the cull.
    """
    cache.set(token, payload)
    return token in cache


def payload_too_large_message(payload, cache) -> str:
    """Actionable error when a result exceeded the diskcache size_limit."""
    spars = payload.get("spars")
    detail = ""
    if spars is not None:
        mb = sum(np.asarray(v).nbytes for v in spars.values()) / 2**20
        detail = f" (S-parameter arrays ~{mb:.0f} MB)"
    limit_mb = getattr(cache, "size_limit", 0) / 2**20
    return (f"result too large to display{detail}: it exceeds the {limit_mb:.0f} MB "
            f"result cache. Reduce the mode count N (or the number of frequency "
            f"points) and run again.")


def register_run_callback(app):
    @app.callback(
        output=[Output("result-store", "data"), Output("status", "children")],
        inputs=Input("run-button", "n_clicks"),
        state=[State("chain-store", "data"), State("sym", "value"),
               State("f-start", "value"), State("f-stop", "value"), State("f-n", "value"),
               State("nproc", "value"),
               State("use-gpu", "value"), State("precision", "value"),
               State("cm-cache-enable", "value"), State("cm-cache-dir", "value"),
               State("compute-select", "value"), State("sweep-mode", "value")],
        background=True,
        running=[(Output("run-button", "disabled"), True, False),
                 (Output("stop-button", "disabled"), False, True)],
        cancel=[Input("stop-button", "n_clicks")],
        progress=[Output("run-status", "children"),
                  Output("run-progress", "value"), Output("run-progress", "max"),
                  Output("run-led", "className"),
                  Output("run-info", "children"),
                  Output("run-progress", "className")],
        prevent_initial_call=True,
    )
    def _run(set_progress, n_clicks, rows, sym, f_start, f_stop, f_n,
             nproc, use_gpu, precision, cm_cache_enable, cm_cache_dir,
             compute, sweep):
        sweep = sweep_mode_value(sweep)
        form = {"rows": rows, "sym": bool(sym), "f_start": f_start, "f_stop": f_stop,
                "f_n": f_n, "nproc": nproc,
                "use_gpu": bool(use_gpu), "precision": precision,
                "cm_cache_enabled": bool(cm_cache_enable), "cm_cache_dir": cm_cache_dir,
                "compute": compute, "sweep": sweep}
        adaptive = sweep == "adaptive"
        n_sweeps = sum(compute_selection(compute))
        bar_max = "1" if adaptive else (str(n_sweeps * int(f_n)) if f_n else "1")
        phase_box = {"phase": "sweep"}    # compute_payload reports phase changes

        def status(phase):
            phase_box["phase"] = phase
            if phase == "cm":
                set_progress(("cm", "0", bar_max, "led led-cm",
                              "computing coupling matrices…", _BAR))
            elif phase == "solving":
                set_progress(("solving", "0", "1", "led led-sweep",
                              "adaptive sampling…", _BAR_DIM))
            else:  # "sweep"
                set_progress(("sweeping", "0", bar_max, "led led-sweep", "", _BAR))

        def progress(done, tot):
            if phase_box["phase"] == "solving":
                set_progress(adaptive_progress(done, tot))   # tot carries f_hz
            elif adaptive:
                # energy phase of an adaptive run: known total, nonuniform grid
                set_progress(("sweeping", str(done), str(tot), "led led-sweep",
                              f"energy {done}/{tot}", _BAR))
            else:
                set_progress(sweep_progress(done, tot, f_start, f_stop, f_n))

        payload, error = compute_payload(form, progress, status_callback=status)
        if error is not None:
            set_progress(("error", "0", bar_max, "led led-error",
                          "see message panel", _BAR))
            return None, error
        token = f"run-{n_clicks}"
        if not store_payload(app._pwmma_cache, token, payload):
            set_progress(("error", "0", bar_max, "led led-error",
                          "see message panel", _BAR))
            return None, payload_too_large_message(payload, app._pwmma_cache)
        message = ""
        mp = payload.get("spars_model")
        if mp is not None and not mp.get("confident", True):
            message = ("warning: adaptive sweep did not converge within its solve "
                       "budget — curves are best-effort (try a uniform run to verify)")
        set_progress(("done", bar_max, bar_max, "led led-done", "", _BAR))
        return {"token": token, "section_indices": payload["section_indices"],
                "compute": payload["compute"], "sweep": payload["sweep"],
                "n_in": payload["n_in"], "n_out": payload["n_out"]}, message

    # Stop kills the background job via `cancel` above; the job can no longer
    # report, so this regular callback resets the status-bar cells instead.
    @app.callback(
        Output("run-status", "children", allow_duplicate=True),
        Output("run-progress", "value", allow_duplicate=True),
        Output("run-led", "className", allow_duplicate=True),
        Output("run-info", "children", allow_duplicate=True),
        Output("run-progress", "className", allow_duplicate=True),
        Input("stop-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def _stopped(n):
        return "stopped", "0", "led led-idle", "", _BAR


def render_spars(payload, out_modes=(0,), in_modes=(0,)):
    if payload.get("spars_model") is not None:      # adaptive line: model view
        return figures.sparam_model_figure(payload["spars_model"])
    if payload.get("spars") is None:
        return figures.empty_figure("S-parameters not computed for this run")
    return figures.sparam_figure(payload["spars"], out_modes=out_modes, in_modes=in_modes)


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
                "power_balance", "s11_complex"]
    return dataclasses.replace(section, **{
        f: getattr(section, f)[keep] for f in per_freq
        if getattr(section, f) is not None})


def energy_model_slice(model, f_lo, f_hi):
    """Band-restrict a smooth_section_energy dict (GHz bounds; None = unbounded)."""
    if model is None or (f_lo is None and f_hi is None):
        return model
    keep = np.ones(len(model["freqs"]), dtype=bool)
    if f_lo is not None:
        keep &= model["freqs"] >= float(f_lo) * 1e9
    if f_hi is not None:
        keep &= model["freqs"] <= float(f_hi) * 1e9
    if not keep.any():
        return None
    if keep.all():
        return model
    out = dict(model)
    for key in ("freqs", "modal_power", "propagating_mask", "evanescent_mask"):
        out[key] = model[key][keep]
    return out


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
    model = energy_model_slice(
        (payload.get("energy_model") or {}).get(int(section)), f_lo, f_hi)
    if model is not None:
        # threshold the dense curves for smoothed modes (teeth between samples
        # count); modes the model does not carry stay sampled-threshold based
        smooth_ids = model["mode_ids"][
            np.max(np.abs(model["modal_power"]), axis=0) > float(threshold)]
        extra = dominant[~np.isin(dominant, model["mode_ids"])]
        dominant = np.union1d(smooth_ids, extra).astype(int)
    return figures.energy_line_figure(
        sec, dB=bool(db), mode_ids=dominant[allowed[dominant]], model=model)


def tab_visibility(tab):
    """Styles for (sparam-graph, energy-graph, energy-controls, spars-controls).

    Each tab shows its own graph and its own controls row; the others collapse
    (display:none). A *visible* controls row must be restored to ``display:flex``
    (its CSS layout) — ``display:block`` would override the flex row and stack
    the controls vertically.
    """
    block, flex, hide = {"display": "block"}, {"display": "flex"}, {"display": "none"}
    if tab == "spars":
        return block, hide, hide, flex
    return hide, block, flex, hide


def register_plot_callbacks(app):
    def _payload(token):
        return app._pwmma_cache.get(token) if token else None

    @app.callback(
        Output("sparam-graph", "style"), Output("energy-graph", "style"),
        Output("energy-controls", "style"), Output("spars-controls", "style"),
        Input("result-tab", "value"),
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

    @app.callback(Output("spars-out-modes", "options"), Output("spars-in-modes", "options"),
                  Output("spars-out-modes", "disabled"), Output("spars-in-modes", "disabled"),
                  Input("result-store", "data"))
    def _spars_mode_options(data):
        # keep >= 1 option so the default [0] selection always stays valid
        n_out = max((data or {}).get("n_out", 0), 1)
        n_in = max((data or {}).get("n_in", 0), 1)
        # adaptive runs model the fundamental-mode pair only
        adaptive = (data or {}).get("sweep") == "adaptive"
        return ([{"label": str(i), "value": i} for i in range(n_out)],
                [{"label": str(j), "value": j} for j in range(n_in)],
                adaptive, adaptive)

    @app.callback(Output("sparam-graph", "figure"),
                  Input("result-store", "data"),
                  Input("spars-out-modes", "value"),
                  Input("spars-in-modes", "value"))
    def _spars(data, out_modes, in_modes):
        p = _payload(data and data.get("token"))
        return render_spars(p, out_modes, in_modes) if p else figures.empty_figure("Run first")

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
