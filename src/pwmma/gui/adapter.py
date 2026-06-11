"""Pure, Dash-independent translation between GUI form state and the pwmma core."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from waveguides import WG, CirWG, RecWG

from ..config import CMConfig, Config, SMConfig
from ..inputs import Chain


class GuiInputError(ValueError):
    """Raised when form input is invalid. The message is shown to the user."""


def _num(value, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise GuiInputError(f"{name!r} must be a number, got {value!r}")


def _positive(value, name: str) -> float:
    v = _num(value, name)
    if v <= 0:
        raise GuiInputError(f"{name!r} must be positive, got {v}")
    return v


def _parse_er(value) -> complex:
    try:
        return complex(str(value))
    except ValueError:
        raise GuiInputError(
            f"'er' must be a real or complex number (e.g. 9.2 or 9.2-0.5j), got {value!r}"
        )


def parse_waveguide(row: dict) -> WG:
    """Convert one chain-row dict (mm units) into a RecWG/CirWG (SI metres)."""
    kind = str(row.get("kind", "")).lower()
    n = int(_num(row.get("N"), "N"))
    if n < 1:
        raise GuiInputError(f"'N' must be >= 1, got {n}")
    length = _positive(row.get("l"), "l") * 1e-3
    er = _parse_er(row.get("er", "1"))
    sigma = _positive(row.get("sigma", 5.8e7), "sigma")
    if kind == "rec":
        a = _positive(row.get("a"), "a") * 1e-3
        b = _positive(row.get("b"), "b") * 1e-3
        return RecWG(a=a, b=b, l=length, N=n, er=er, sigma=sigma)
    if kind == "cir":
        r = _positive(row.get("r"), "r") * 1e-3
        return CirWG(r=r, l=length, N=n, er=er, sigma=sigma)
    raise GuiInputError(f"unknown waveguide kind {kind!r} (expected 'rec' or 'cir')")


def parse_chain(rows: Sequence[dict], sym: bool) -> Chain:
    if len(rows) < 2:
        raise GuiInputError("a chain needs at least 2 waveguide segments")
    return Chain([parse_waveguide(r) for r in rows], sym=bool(sym))


def parse_freqs(start_ghz, stop_ghz, n_points) -> np.ndarray:
    start = _num(start_ghz, "start")
    stop = _num(stop_ghz, "stop")
    n = int(_num(n_points, "N points"))
    if n < 1:
        raise GuiInputError("frequency point count must be >= 1")
    if not start < stop:
        raise GuiInputError(f"'start' must be < 'stop' (got {start} >= {stop})")
    return np.linspace(start, stop, n) * 1e9


def parse_config(cm: dict, sm: dict) -> Config:
    precision = str(sm.get("precision", "complex64"))
    return Config(
        cmconf=CMConfig(nproc=int(cm.get("nproc", 8))),
        smconf=SMConfig(
            nproc=int(sm.get("nproc", 8)),
            use_gpu=bool(sm.get("use_gpu", True)),
            use_double_precision=(precision == "complex128"),
        ),
    )
