#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rational S-parameter sweep layer (AAA fit + adaptive frequency sampling).

S11(f) of an overmoded window is meromorphic: its MHz-wide trapped-mode teeth
are poles, which a barycentric rational (AAA) model recovers from samples
placed anywhere — no tooth-resolving uniform grid needed. This module owns
"cheaply get an accurate broadband rational S-parameter model"; design
objectives (band thresholds, margins, optimizer fitness) stay in user code.

Layering (see local/chainsolver-sparmodel-spec.md):

    fit_spar_model(F, y, ...)              samples -> SparModel (atomic)
    uniform_spar_model(solver, freqs, ...) solver.sweep + fit
    adaptive_spar_model(solver, f0, f1)    AFS loop {smatrix_at -> fit -> argmax}
    minus_NdB_band(model, ...)             thin band helper

The adaptive stop rule mirrors CST's interpolative broadband sweep: the
leave-newest-out AAA error on a dense candidate grid must stay below ``tol``
for ``n_checks`` consecutive iterations. ``n_checks >= 2`` is load-bearing:
with a single check the error can dip below tol before any sample has probed a
tooth, silently missing every tooth (validated on the 650 GHz baseline:
1/7 teeth at n_checks=1 vs 7/7 at n_checks=2).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import AAA

_BLOCKS = {"s11": 0, "s12": 1, "s21": 2, "s22": 3}


def _fit_aaa(F, y, *, max_terms, clean_up, rtol=None):
    """AAA fit of complex samples (sorted internally; input order untouched)."""
    F = np.asarray(F, dtype=float)
    y = np.asarray(y, dtype=complex)
    if F.ndim != 1 or F.shape != y.shape:
        raise ValueError(f"F and y must be 1-D and equal-length, got {F.shape} vs {y.shape}")
    if not (np.isfinite(F).all() and np.isfinite(y).all()):
        raise ValueError("non-finite values in the samples")
    order = np.argsort(F)
    return AAA(F[order], y[order], rtol=rtol, max_terms=max_terms, clean_up=clean_up)


class SparModel:
    """Rational (AAA) model of ONE scalar S-parameter curve.

    ``F``/``y`` keep the solver-call **insertion order** (the adaptive loop's
    leave-newest-out estimator and warm starts depend on it); use ``samples``
    for the frequency-sorted view. ``n_solves`` counts true solver calls
    consumed to build the model; ``confident`` is False when the adaptive loop
    exhausted ``max_solves`` without converging.
    """

    def __init__(self, F, y, curve, aaa, *, n_solves, confident):
        self.F = np.asarray(F, dtype=float)
        self.y = np.asarray(y, dtype=complex)
        self.curve = tuple(curve)
        self.n_solves = int(n_solves)
        self.confident = bool(confident)
        self._aaa = aaa

    def __call__(self, f):
        """Dense evaluation (vectorized, ~free)."""
        return self._aaa(f)

    def poles(self) -> np.ndarray:
        """Complex poles of the rational approximant."""
        return self._aaa.poles()

    @property
    def samples(self):
        """(F_sorted, y_sorted) — the samples in frequency order."""
        order = np.argsort(self.F)
        return self.F[order], self.y[order]

    @property
    def support_points(self) -> np.ndarray:
        """Barycentric support points (a subset of the fitted samples).

        scipy's AAA stores them as complex; our abscissae are frequencies, so
        the (identically zero) imaginary part is dropped.
        """
        return np.real(np.asarray(self._aaa.support_points)).astype(float)

    @property
    def weights(self) -> np.ndarray:
        """Barycentric weights; together with the support points they fix the poles."""
        return np.asarray(self._aaa.weights, dtype=complex)

    def sibling_fit(self, F, y, *, curve=("s11", 0, 0)) -> "SparModel":
        """Model another curve of the SAME structure, reusing this model's poles.

        All response functions of one chain share its resonance poles; only the
        residues differ. This keeps the parent's barycentric denominator
        (support points + weights) fixed and solves just the numerator by
        linear least squares over ALL of ``(F, y)`` — so the sibling cannot
        grow poles absent from the parent: its failure mode is a missed
        feature, never an invented one (unlike an independent AAA refit, which
        can plant spurious pole pairs between sparse samples).

        ``F`` is typically the parent's sample grid but may be any solved set.
        The returned model reports ``n_solves=0`` (no new solver calls) and
        inherits the parent's ``confident`` flag.
        """
        F = np.asarray(F, dtype=float)
        y = np.asarray(y, dtype=complex)
        if F.ndim != 1 or F.shape != y.shape:
            raise ValueError(f"F and y must be 1-D and equal-length, got {F.shape} vs {y.shape}")
        if not (np.isfinite(F).all() and np.isfinite(y).all()):
            raise ValueError("non-finite values in the samples")

        z = np.real(np.asarray(self._aaa.support_points)).astype(float)
        w = np.asarray(self._aaa.weights, dtype=complex)
        diff = F[:, None] - z[None, :]
        on_support = diff == 0.0

        # Unknowns are the sibling's support VALUES v_k (numerator = Σ wₖvₖ/(f−zₖ)).
        # Off-support samples give Σₖ wₖvₖ/(Fⱼ−zₖ) = yⱼ·D(Fⱼ); a sample sitting
        # on a support point pins vₖ = yⱼ directly (barycentric limit).
        A = np.zeros((F.size, z.size), dtype=complex)
        b = np.empty(F.size, dtype=complex)
        off = ~on_support.any(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            C = w[None, :] / diff
        A[off] = C[off]
        b[off] = y[off] * C[off].sum(axis=1)
        pin = ~off
        A[pin, on_support[pin].argmax(axis=1)] = 1.0
        b[pin] = y[pin]

        scale = np.max(np.abs(A), axis=1)
        scale[scale == 0.0] = 1.0
        v, *_ = np.linalg.lstsq(A / scale[:, None], b / scale, rcond=None)

        return SparModel(F, y, curve, _SiblingRational(self._aaa, v),
                         n_solves=0, confident=self.confident)


class _SiblingRational:
    """Barycentric rational sharing a parent AAA's support points and weights.

    Quacks like ``scipy.interpolate.AAA`` where SparModel needs it (``__call__``,
    ``poles()``, ``support_points``/``weights``), so sibling models compose —
    including ``sibling_fit`` chained off a sibling.
    """

    def __init__(self, parent, support_values):
        self.support_points = np.real(np.asarray(parent.support_points)).astype(float)
        self.weights = np.asarray(parent.weights, dtype=complex)
        self.support_values = np.asarray(support_values, dtype=complex)
        self._parent = parent

    def __call__(self, f):
        f = np.asarray(f, dtype=float)
        ff = np.atleast_1d(f)
        diff = ff[:, None] - self.support_points[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            C = 1.0 / diff
            r = (C @ (self.weights * self.support_values)) / (C @ self.weights)
        rows, cols = np.nonzero(diff == 0.0)
        r[rows] = self.support_values[cols]
        return r.reshape(f.shape)

    def poles(self) -> np.ndarray:
        return self._parent.poles()


def fit_spar_model(F, y, *, max_terms=50, clean_up=True, rtol=None,
                   curve=("s11", 0, 0)) -> SparModel:
    """Fit a SparModel to already-solved samples.

    ``max_terms=50`` and ``clean_up=True`` are deliberate defaults: uncapped
    AAA overfits single-precision solver noise into 80+ spurious poles and 10x
    fit time. Rule of thumb: cap ~ 2x the expected in-band pole count.
    """
    aaa = _fit_aaa(F, y, max_terms=max_terms, clean_up=clean_up, rtol=rtol)
    return SparModel(F, y, curve, aaa, n_solves=len(np.asarray(F)), confident=True)


def _curve_index(curve):
    block, i, j = curve
    try:
        return _BLOCKS[str(block).lower()], int(i), int(j)
    except KeyError:
        raise ValueError(f"curve block must be one of {sorted(_BLOCKS)}, got {block!r}")


def uniform_spar_model(solver, freqs, *, curve=("s11", 0, 0), **fit_kw) -> SparModel:
    """solver.sweep over ``freqs``, then fit the scalar ``curve``.

    ``curve=(block, i, j)`` selects ``S<block>[k, i, j]``. If you also need the
    full S-matrix stack, call ``solver.sweep`` yourself and pass the extracted
    scalar to :func:`fit_spar_model` — there is no keep-everything flag here.
    """
    blk, i, j = _curve_index(curve)
    freqs = np.asarray(freqs, dtype=float)
    S = solver.sweep(freqs)
    y = S[blk][:, i, j]
    return fit_spar_model(freqs, y, curve=curve, **fit_kw)


def adaptive_spar_model(solver, f0, f1, *,
                        curve=("s11", 0, 0),
                        tol=1e-2,
                        n_checks=2,
                        max_terms=50,
                        n_seed=5,
                        seed=None,
                        n_cand=4001,
                        max_solves=200) -> SparModel:
    """Adaptive-frequency-sampling rational model of one S-parameter curve.

    Solves at a small seed set, then repeatedly places the next sample at the
    dense-grid frequency where the AAA fit and its leave-newest-out variant
    disagree most, until they agree to ``tol`` (max |R - R'| in complex linear
    S units, like CST's 0.01) for ``n_checks`` consecutive iterations.

    ``seed`` may be an array of frequencies to solve, or a warm start
    ``(F_prev, y_prev)`` of already-solved samples (zero re-solves; the loop
    only adds points the model is uncertain about). If ``max_solves`` is
    exhausted the model is returned anyway with ``confident=False``.

    CAUTION: the stop rule measures self-consistency, not truth. On a chain
    whose S11 background is quiet, the default 5-point seed can converge
    before any sample has felt a narrow resonance and silently miss it. For
    real waveguide chains, seed with
    :func:`pwmma.adaptive_seed_frequencies` (uniform exploration floor +
    cutoff hotspot probes); ``max_solves`` then needs headroom above the
    seed count.
    """
    if n_checks < 1:
        raise ValueError(f"n_checks must be >= 1, got {n_checks}")
    blk, i, j = _curve_index(curve)

    n_solves = 0

    def solve(f):
        nonlocal n_solves
        n_solves += 1
        return complex(solver.smatrix_at(float(f))[blk][i, j])

    # X/Y are append-only: the leave-newest-out estimator needs insertion order.
    if seed is None:
        X = [float(f) for f in np.linspace(f0, f1, n_seed)]
        Y = [solve(f) for f in X]
    elif isinstance(seed, tuple) and len(seed) == 2:
        F_prev, y_prev = seed                       # warm start: zero re-solves
        X = [float(f) for f in np.asarray(F_prev, dtype=float)]
        Y = [complex(v) for v in np.asarray(y_prev)]
    else:
        X = [float(f) for f in np.asarray(seed, dtype=float)]
        Y = [solve(f) for f in X]

    cand = np.linspace(f0, f1, n_cand)
    excl = 2.0 * (f1 - f0) / (n_cand - 1)           # exclusion radius

    consec = 0
    while n_solves < max_solves:
        R = _fit_aaa(X, Y, max_terms=max_terms, clean_up=True)
        Rp = _fit_aaa(X[:-1], Y[:-1], max_terms=max_terms, clean_up=True)
        err = np.abs(R(cand) - Rp(cand))
        near = np.min(np.abs(cand[:, None] - np.asarray(X)[None, :]), axis=1) <= excl
        err[near] = 0.0

        # AAA on tiny sets is degenerate: no convergence verdict before then.
        if len(X) >= n_seed + 2:
            if err.max() < tol:
                consec += 1
                if consec >= n_checks:
                    break                            # converged
            else:
                consec = 0

        k = int(np.argmax(err))
        if near[k]:
            # err is identically zero outside exclusions (R == R' everywhere):
            # probing argmax would duplicate a support point. Probe the largest
            # sampling gap instead.
            k = int(np.argmax(np.min(np.abs(cand[:, None] - np.asarray(X)[None, :]), axis=1)))
        X.append(float(cand[k]))
        Y.append(solve(cand[k]))

    aaa = _fit_aaa(X, Y, max_terms=max_terms, clean_up=True)
    return SparModel(X, Y, curve, aaa,
                     n_solves=n_solves, confident=(consec >= n_checks))


def minus_NdB_band(model, ndb=-20.0, center=None, npts=30001):
    """(f_lo, f_hi) of the contiguous run with 20*log10|model| below ``ndb``
    that contains ``center``; None if the curve at ``center`` is not below.

    ``center`` is required — the model may hold several sub-``ndb`` windows and
    pwmma will not guess which one is meant. Margins/objectives are user code.
    """
    if center is None:
        raise ValueError("center is required (which sub-band do you mean?)")
    g = np.linspace(model.F.min(), model.F.max(), npts)
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.abs(model(g)))
    below = db < ndb
    k = int(np.argmin(np.abs(g - center)))
    if not below[k]:
        return None
    lo = k
    while lo > 0 and below[lo - 1]:
        lo -= 1
    hi = k
    while hi < npts - 1 and below[hi + 1]:
        hi += 1
    return float(g[lo]), float(g[hi])
