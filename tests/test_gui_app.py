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
