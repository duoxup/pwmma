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
