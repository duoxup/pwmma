"""Persisted default GUI state (written by the 'save as default' button)."""
from __future__ import annotations

import json
import os

# Stable per-user location, independent of where pwmma-gui was launched.
_PATH = os.path.expanduser("~/.pwmma_gui_defaults.json")

BUILTIN_DEFAULTS = {
    "rows": [
        {"kind": "rec", "a": 7.112, "b": 3.556, "l": 2.0, "N": 200, "er": "1", "sigma": "5.8e7"},
        {"kind": "cir", "r": 4.2, "l": 1.5, "N": 800, "er": "1", "sigma": "5.8e7"},
        {"kind": "cir", "r": 5.4, "l": 0.26, "N": 800, "er": "9.2", "sigma": "5.8e7"},
    ],
    "sym": ["sym"],
    "f_start": 28.0, "f_stop": 34.0, "f_n": 61,
    "nproc": 8,
    "use_gpu": ["gpu"], "precision": "single",
    "compute": "both", "sweep": "uniform",
    "cm_cache_enable": [], "cm_cache_dir": ".pwmma_cm_cache",
}


def load_defaults() -> dict:
    """Return the saved default state merged over the built-in defaults."""
    state = dict(BUILTIN_DEFAULTS)
    try:
        with open(_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            # legacy migration: pre-unification files stored cm_nproc/sm_nproc
            # separately; carry the value over so a saved setting is not lost.
            if "nproc" not in saved and ("sm_nproc" in saved or "cm_nproc" in saved):
                saved["nproc"] = saved.get("sm_nproc", saved.get("cm_nproc"))
            state.update(saved)
    except (OSError, ValueError):
        pass
    return state


def save_defaults(state: dict) -> None:
    """Persist the given GUI state as the new default."""
    with open(_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)
