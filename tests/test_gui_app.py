import importlib


def test_gui_package_imports_without_dash_in_core():
    # Importing the core must not require the GUI deps.
    import pwmma  # noqa: F401
    # The gui subpackage exists and is importable on its own.
    mod = importlib.import_module("pwmma.gui")
    assert mod is not None
