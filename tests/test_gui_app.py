import importlib


def test_gui_package_imports_without_dash_in_core():
    # Importing the core must not require the GUI deps.
    import pwmma  # noqa: F401
    # The gui subpackage exists and is importable on its own.
    mod = importlib.import_module("pwmma.gui")
    assert mod is not None


def test_layout_has_core_component_ids():
    from pwmma.gui.layout import build_layout
    root = build_layout()
    ids = set()

    def walk(c):
        cid = getattr(c, "id", None)
        if isinstance(cid, str):
            ids.add(cid)
        children = getattr(c, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                walk(child)
        elif children is not None:
            walk(children)

    walk(root)
    for required in ["chain-store", "result-store", "run-button", "stop-button",
                     "save-default", "structure-preview", "sparam-graph", "energy-graph",
                     "status", "run-led", "run-status", "run-progress", "config-summary"]:
        assert required in ids, f"missing component id {required}"


def test_render_chain_rows_builds_inputs_per_row():
    from pwmma.gui.callbacks import render_chain_rows
    rows = [{"kind": "rec", "a": 7.0, "b": 3.0, "l": 2.0, "N": 24, "er": "1", "sigma": "5.8e7"}]
    children = render_chain_rows(rows)
    assert len(children) == 1


def test_preview_callback_returns_figure():
    import plotly.graph_objects as go

    from pwmma.gui.callbacks import update_structure_preview
    rows = [
        {"kind": "rec", "a": 7.0, "b": 3.0, "l": 2.0, "N": 24, "er": "1", "sigma": "5.8e7"},
        {"kind": "cir", "r": 4.2, "l": 1.5, "N": 64, "er": "1", "sigma": "5.8e7"},
    ]
    fig = update_structure_preview(rows, ["sym"])
    assert isinstance(fig, go.Figure)


def test_create_app_builds_dash_app():
    from pwmma.gui.app import create_app
    app = create_app()
    assert app.layout is not None
    # callbacks registered
    assert len(app.callback_map) > 0


def test_run_analysis_core_produces_result_payload():
    # The pure compute helper behind the background callback.
    from pwmma.gui.callbacks import compute_payload
    rows = [
        {"kind": "rec", "a": 7.112, "b": 3.556, "l": 2.0, "N": 24, "er": "1", "sigma": "5.8e7"},
        {"kind": "cir", "r": 4.2, "l": 1.5, "N": 64, "er": "1", "sigma": "5.8e7"},
        {"kind": "cir", "r": 5.4, "l": 0.26, "N": 96, "er": "9.2", "sigma": "5.8e7"},
    ]
    form = {"rows": rows, "sym": True, "f_start": 28.0, "f_stop": 34.0, "f_n": 3,
            "cm_nproc": 2, "sm_nproc": 2, "use_gpu": False, "precision": "complex64"}
    progress = []
    payload, error = compute_payload(form, lambda d, t: progress.append((d, t)))
    assert error is None
    assert payload["section_indices"] == [1, 2, 3]
    assert progress and progress[-1] == (6, 6)  # energy sweep (3) + S-param sweep (3)


def test_render_plots_from_payload():
    import plotly.graph_objects as go

    from pwmma.gui.callbacks import compute_payload, render_energy, render_spars
    rows = [
        {"kind": "rec", "a": 7.112, "b": 3.556, "l": 2.0, "N": 24, "er": "1", "sigma": "5.8e7"},
        {"kind": "cir", "r": 4.2, "l": 1.5, "N": 64, "er": "1", "sigma": "5.8e7"},
        {"kind": "cir", "r": 5.4, "l": 0.26, "N": 96, "er": "9.2", "sigma": "5.8e7"},
    ]
    form = {"rows": rows, "sym": True, "f_start": 28.0, "f_stop": 34.0, "f_n": 3,
            "cm_nproc": 2, "sm_nproc": 2, "use_gpu": False, "precision": "complex64"}
    payload, _ = compute_payload(form, lambda d, t: None)
    assert isinstance(render_spars(payload), go.Figure)
    assert isinstance(render_energy(payload, section=2, kind="heatmap",
                                    threshold=0.04, db=True), go.Figure)


def test_result_payload_survives_diskcache_roundtrip(tmp_path):
    import diskcache
    import plotly.graph_objects as go

    from pwmma.gui.callbacks import compute_payload, render_energy, render_spars
    rows = [
        {"kind": "rec", "a": 7.112, "b": 3.556, "l": 2.0, "N": 24, "er": "1", "sigma": "5.8e7"},
        {"kind": "cir", "r": 4.2, "l": 1.5, "N": 64, "er": "1", "sigma": "5.8e7"},
        {"kind": "cir", "r": 5.4, "l": 0.26, "N": 96, "er": "9.2", "sigma": "5.8e7"},
    ]
    form = {"rows": rows, "sym": True, "f_start": 28.0, "f_stop": 34.0, "f_n": 3,
            "cm_nproc": 2, "sm_nproc": 2, "use_gpu": False, "precision": "complex64"}
    payload, err = compute_payload(form, lambda d, t: None)
    assert err is None
    # The background worker and the main process communicate ONLY through the
    # diskcache, so the payload must survive a pickle round-trip and stay usable.
    cache = diskcache.Cache(str(tmp_path / "c"))
    try:
        cache.set("tok", payload)
        got = cache.get("tok")
    finally:
        cache.close()
    assert got is not None
    assert got["section_indices"] == payload["section_indices"]
    sec_id = got["section_indices"][0]
    assert isinstance(render_spars(got), go.Figure)
    assert isinstance(render_energy(got, section=sec_id, kind="line",
                                    threshold=0.04, db=True), go.Figure)


def test_defaults_save_load_roundtrip(tmp_path, monkeypatch):
    from pwmma.gui import defaults
    monkeypatch.setattr(defaults, "_PATH", str(tmp_path / "d.json"))

    # built-in defaults when no file exists
    d0 = defaults.load_defaults()
    assert "rows" in d0 and d0["precision"] == "single"

    # saved values override the built-ins, missing keys fall back to built-in
    defaults.save_defaults({"f_n": 123, "precision": "double"})
    d1 = defaults.load_defaults()
    assert d1["f_n"] == 123
    assert d1["precision"] == "double"
    assert "rows" in d1


def test_tab_visibility_switches_views():
    from pwmma.gui.callbacks import tab_visibility
    spars, energy_graph, energy_controls = tab_visibility("spars")
    assert spars["display"] == "block"
    assert energy_graph["display"] == "none"
    assert energy_controls["visibility"] == "hidden"
    spars, energy_graph, energy_controls = tab_visibility("energy")
    assert spars["display"] == "none"
    assert energy_graph["display"] == "block"
    assert energy_controls["visibility"] == "visible"


def test_assets_css_present_and_packaged():
    from pathlib import Path

    import pwmma.gui as gui
    css = Path(gui.__file__).parent / "assets" / "style.css"
    assert css.is_file()
    text = css.read_text(encoding="utf-8")
    # the design-system anchors the rest of the plan relies on
    for needle in [":root", ".titlebar", ".toolbar", ".statusbar", ".chain-row",
                   ".led-sweep", ".eda-tab", "--accent"]:
        assert needle in text, f"style.css missing {needle}"
    # the wheel must ship the assets dir (setuptools package-data)
    pyproject = Path(gui.__file__).parents[3] / "pyproject.toml"  # src layout: repo root
    assert '"pwmma.gui" = ["assets/*"]' in pyproject.read_text(encoding="utf-8")


def test_figures_use_shared_eda_theme():
    import numpy as np

    from pwmma.gui import figures

    fig = figures.empty_figure("x")
    assert fig.layout.plot_bgcolor == "#ffffff"

    spars = {"freqs": np.array([28e9, 29e9]),
             "s11": np.full((2, 1, 1), 0.5 + 0j), "s21": np.full((2, 1, 1), 0.5 + 0j)}
    fig2 = figures.sparam_figure(spars)
    assert fig2.layout.colorway[0] == "#1d5a9e"
    assert fig2.layout.xaxis.gridcolor == "#eef0f4"


def _synthetic_section():
    """A 5-point, 2-mode section: mode 0 always propagating, mode 1 cut off
    below the third frequency point (negative net contribution while cut off)."""
    import numpy as np

    from pwmma.analysis import SectionEnergyCoupling

    freqs = np.linspace(28e9, 34e9, 5)
    modal_power = np.array([
        [0.9, -0.06],
        [0.9, -0.05],
        [0.9, 0.30],
        [0.9, 0.32],
        [0.9, 0.35],
    ])
    prop = np.array([
        [True, False],
        [True, False],
        [True, True],
        [True, True],
        [True, True],
    ])
    z = np.zeros((5, 2), dtype=complex)
    zf = np.zeros(5)
    return SectionEnergyCoupling(
        section_index=1, freqs=freqs, mode_ids=np.arange(2),
        modal_power=modal_power, propagating_mask=prop, evanescent_mask=~prop,
        forward_left=z, backward_left=z, forward_right=z, backward_right=z,
        reflection_power=zf, total_reflected_power=zf, transmission_power=zf,
        power_balance=np.ones(5),
    )


def test_energy_line_shows_cutoff_segments_dotted():
    import numpy as np

    from pwmma.gui import figures

    fig = figures.energy_line_figure(_synthetic_section(), mode_threshold=0.04, dB=False)
    by_name = {t.name: t for t in fig.data}

    # cutoff (evanescent) segment: dotted twin in the same color as the solid trace
    solid, dotted = by_name["1"], by_name["1 (evan.)"]
    assert dotted.line.dash == "dot"
    assert dotted.line.color == solid.line.color
    solid_y = np.asarray(solid.y, dtype=float)
    dotted_y = np.asarray(dotted.y, dtype=float)
    assert np.isnan(solid_y[:2]).all() and np.isfinite(solid_y[2:]).all()
    assert np.isfinite(dotted_y[:2]).all() and np.isnan(dotted_y[2:]).all()

    # mode 0 never cuts off: no dotted twin
    assert "0 (evan.)" not in by_name

    # total evanescent contribution gets its own summary trace
    assert "Σ evan." in by_name


def test_sweep_progress_reports_current_frequency():
    from pwmma.gui.callbacks import sweep_progress

    text, value, vmax, led = sweep_progress(38, 122, 28.0, 34.0, 61)
    assert (value, vmax) == ("38", "122")
    assert led == "led led-sweep"
    assert "31.700 GHz" in text  # idx 37 of 28..34 GHz / 61 pts

    # the S-parameter sweep (second half) wraps back to the sweep start
    text2, *_ = sweep_progress(62, 122, 28.0, 34.0, 61)
    assert "28.000 GHz" in text2


def test_sweep_progress_tolerates_bad_inputs():
    from pwmma.gui.callbacks import sweep_progress

    text, value, vmax, led = sweep_progress(1, 2, None, None, None)
    assert text.startswith("sweeping 1/2")
    assert led == "led led-sweep"

    # degenerate sweep sizes must not raise (modulo/slope guards)
    text1, *_ = sweep_progress(1, 2, 28.0, 34.0, 1)
    assert "28.000 GHz" in text1
    text0, *_ = sweep_progress(1, 2, 28.0, 34.0, 0)
    assert text0.startswith("sweeping 1/2")
