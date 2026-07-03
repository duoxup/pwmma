"""Pure, Dash-independent translation between GUI form state and the pwmma core."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from waveguides import WG, CirWG, RecWG

from .. import analyze_energy_coupling
from ..analysis import adaptive_seed_frequencies, smooth_section_energy
from ..config import Config
from ..coupling_matrix import get_coupling_matrix
from ..inputs import Chain
from ..io.numpy import prune_coupling_matrix_cache
from ..main import calc_spars_of_wgchain
from ..solver import ChainSolver
from ..spar_model import adaptive_spar_model


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


def parse_config(cfg: dict) -> Config:
    # Accept the GUI's "single"/"double" labels (and the raw numpy names).
    precision = str(cfg.get("precision", "single"))
    cache_dir = cfg.get("cache_dir") or None
    use_cache = bool(cfg.get("cache_enabled")) and cache_dir is not None
    return Config(
        nproc=int(cfg.get("nproc", 8)),
        use_gpu=bool(cfg.get("use_gpu", True)),
        use_double_precision=(precision in ("double", "complex128")),
        cm_cache_dir=cache_dir if use_cache else None,
        try_read_cm_from_cache=use_cache,
        save_cm_to_cache=use_cache,
    )


def compute_cms(chain, config):
    """Raw coupling matrices for the chain's transitions (one per junction)."""
    return [get_coupling_matrix(wgt, config) for wgt in chain.transitions]


def run_energy(chain, freqs, config, *, sections, excitation_mode=0,
               progress_callback: Callable[[int, int], None] | None = None, cms=None):
    return analyze_energy_coupling(
        chain, freqs, config, sections=sections, excitation_mode=excitation_mode,
        show_progress=False, progress_callback=progress_callback, cms=cms,
    )


def run_spars(chain, freqs, config,
              progress_callback: Callable[[int, int], None] | None = None, cms=None) -> dict:
    s11, s12, s21, s22 = calc_spars_of_wgchain(
        chain, freqs, config, show_progress=False, progress_callback=progress_callback, cms=cms,
    )
    return {"freqs": np.asarray(freqs), "s11": s11, "s12": s12, "s21": s21, "s22": s22}


def make_solver(chain, config, cms=None) -> ChainSolver:
    """One prepared solver session per Run (CMs computed/uploaded once)."""
    return ChainSolver(chain, config, cms=cms)


def run_spars_on(solver, freqs,
                 progress_callback: Callable[[int, int], None] | None = None) -> dict:
    """Uniform sweep on an existing solver session; same dict as run_spars."""
    s11, s12, s21, s22 = solver.sweep(np.asarray(freqs, dtype=float),
                                      progress_callback=progress_callback)
    return {"freqs": np.asarray(freqs), "s11": s11, "s12": s12, "s21": s21, "s22": s22}


class _SolveRecorder:
    """Forwards ``smatrix_at``, counting solves and keeping the fundamental-mode
    scalars of every solved matrix.

    ``adaptive_spar_model`` only ever sees this proxy: it drives the sampling
    from S11[0,0], while S21[0,0] is captured here for free from the same
    matrices (the teeth are chain resonances — poles are shared across all S
    entries, so fitting S21 on the S11-chosen points is physically sound).
    """

    def __init__(self, solver, on_solve=None):
        self._solver = solver
        self._on_solve = on_solve
        self.s21_at: dict[float, complex] = {}
        self.count = 0

    def smatrix_at(self, f):
        S = self._solver.smatrix_at(f)
        self.count += 1
        self.s21_at[float(f)] = complex(S[2][0, 0])
        if self._on_solve is not None:
            self._on_solve(self.count, float(f))
        return S


def run_adaptive_spars(solver, f0, f1,
                       progress_callback: Callable[[int, float], None] | None = None,
                       cms=None, n_uniform: int = 0) -> dict:
    """Adaptive fundamental-mode S-parameter samples (the production line).

    Plain-numpy payload (diskcache-safe); the UI refits with fit_spar_model on
    render (milliseconds). ``progress_callback(k, f)`` fires once per solve.

    The default is the bare AFS loop — the economy behavior the adaptive line
    is designed around. It can converge on a quiet S11 background without
    feeling narrow sub-tolerance teeth; that is an accepted trade-off, like
    modal truncation or quadrature order (no solver setting is perfectly
    faithful). Runs that must identify every fine tooth opt in by setting
    ``n_uniform`` > 0 (the GUI "seeds" field), which pre-samples a uniform
    exploration floor of that many points plus cutoff hotspot probes via
    ``adaptive_seed_frequencies`` (reachability-filtered when ``cms`` is
    given); ``max_solves`` then gets headroom above the seed count.
    """
    rec = _SolveRecorder(solver, on_solve=progress_callback)
    chain = getattr(solver, "chain", None)
    seed = None
    max_solves = 200
    if n_uniform and chain is not None:
        seed = adaptive_seed_frequencies(chain, float(f0), float(f1), cms=cms,
                                         n_uniform=int(n_uniform))
        max_solves = len(seed) + 200
    model = adaptive_spar_model(rec, float(f0), float(f1),     # S11[0,0] drives
                                seed=seed, max_solves=max_solves)
    F = np.asarray(model.F, dtype=float)
    return {
        "f0": float(f0), "f1": float(f1),
        "s11": {"F": F, "y": np.asarray(model.y, dtype=complex)},
        "s21": {"F": F.copy(),
                "y": np.array([rec.s21_at[float(f)] for f in F], dtype=complex)},
        "n_solves": int(model.n_solves), "confident": bool(model.confident),
    }


def run_energy_model(result, f0, f1, n_dense: int = 2001) -> dict:
    """Dense smoothed per-mode energy curves for an adaptive run, per section.

    Post-processing only (zero extra solves): shared-pole sibling fits of the
    modal coefficients off each section's lossless s11_complex samples, with
    the analytic factors rebuilt exactly on the dense grid. Modes the
    shared-pole basis cannot represent land in ``unsmoothed_mode_ids`` and are
    drawn from their samples by the figure layer. Plain-numpy, ~1 MB payload.
    """
    dense = np.linspace(float(f0), float(f1), int(n_dense))
    return {int(idx): smooth_section_energy(result.get_section(idx), dense)
            for idx in result.section_indices}


def prune_cache(cache_dir: str, *, dry_run: bool = False) -> dict:
    """Remove redundant coupling-matrix cache files; returns a summary dict."""
    return prune_coupling_matrix_cache(cache_dir, dry_run=dry_run)
