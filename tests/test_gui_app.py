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

    walk(root)
    for required in ["chain-store", "result-store", "run-button", "structure-preview",
                     "sparam-graph", "energy-graph", "status"]:
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
    assert progress and progress[-1] == (3, 3)
