#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rational sweep layer (AAA/AFS): fit, uniform, adaptive, band helper.

All tests run against a fake solver whose S11 is an EXACT rational function —
a smooth rational background plus narrow Lorentzian "teeth" whose widths
(~1e-4 of the span) are far below any seed spacing. That is precisely the
regime the adaptive sampler exists for: a rational model is determined by
O(degree) samples placed anywhere, so AAA recovers sub-grid teeth that a
uniform grid of the same budget cannot see.
"""
from __future__ import annotations

import numpy as np
import pytest

from pwmma import SparModel, adaptive_spar_model, fit_spar_model, minus_NdB_band, uniform_spar_model

F0, F1 = 600e9, 716e9
SPAN = F1 - F0


class FakeSolver:
    """smatrix_at/sweep of a 1-port chain with an exact rational S11."""

    # (center, width, strength): three teeth much narrower than seed spacing
    TEETH = ((628e9, 1.0e-4 * SPAN, 0.40),
             (655e9, 0.8e-4 * SPAN, 0.35),
             (690e9, 1.2e-4 * SPAN, 0.45))
    BG_POLE = 640e9 - 2.5e11j          # far pole -> smooth rational background

    def __init__(self):
        self.calls = 0                  # smatrix_at invocations (solver cost)

    def s11(self, f):
        f = np.asarray(f, dtype=complex)
        y = 0.05 + 1.5e10 / (f - self.BG_POLE)
        for fc, w, a in self.TEETH:
            y = y + (a * w) / (f - fc - 1j * w)
        return y

    def s21(self, f):
        """A sibling curve: exactly the same poles, different residues."""
        f = np.asarray(f, dtype=complex)
        y = 0.9 - 0.7e10 / (f - self.BG_POLE)
        for (fc, w, _), a in zip(self.TEETH, (-0.25, 0.5, 0.15)):
            y = y + (a * w) / (f - fc - 1j * w)
        return y

    def true_poles(self):
        return np.array([fc + 1j * w for fc, w, _ in self.TEETH] + [self.BG_POLE])

    def smatrix_at(self, f):
        self.calls += 1
        v = np.array([[complex(self.s11(f))]])
        z = np.zeros_like(v)
        return (v, z, z, z)

    def sweep(self, freqs, **_kw):
        freqs = np.asarray(freqs, dtype=float)
        s11 = self.s11(freqs)[:, None, None]
        z = np.zeros_like(s11)
        return (s11, z, z, z)


def test_fit_round_trip_recovers_function_and_poles():
    solver = FakeSolver()
    grid = np.linspace(F0, F1, 32)          # ~8x the pole count (4 poles)
    model = fit_spar_model(grid, solver.s11(grid))
    dense = np.linspace(F0, F1, 3200)
    assert np.max(np.abs(model(dense) - solver.s11(dense))) < 1e-6
    poles = model.poles()
    for p in solver.true_poles():
        assert np.min(np.abs(poles - p)) < 1e-3 * SPAN


def test_adaptive_finds_sub_grid_teeth():
    solver = FakeSolver()
    model = adaptive_spar_model(solver, F0, F1)          # spec defaults
    dense = np.linspace(F0, F1, 20001)
    assert np.max(np.abs(model(dense) - solver.s11(dense))) < 1e-2   # = tol
    assert model.confident
    assert model.n_solves == solver.calls
    # teeth widths ~1e-4 span: a tooth-resolving uniform grid needs >~1e4
    # points; the adaptive model gets there in a tiny fraction of that.
    assert model.n_solves <= 120


def test_n_checks_monotone_solve_counts():
    m1 = adaptive_spar_model(FakeSolver(), F0, F1, n_checks=1)
    m2 = adaptive_spar_model(FakeSolver(), F0, F1, n_checks=2)
    assert m1.n_solves <= m2.n_solves
    dense = np.linspace(F0, F1, 20001)
    assert np.max(np.abs(m2(dense) - FakeSolver().s11(dense))) < 1e-2
    with pytest.raises(ValueError):
        adaptive_spar_model(FakeSolver(), F0, F1, n_checks=0)


def test_max_solves_exhaustion_returns_unconfident_model():
    model = adaptive_spar_model(FakeSolver(), F0, F1, max_solves=8)
    assert isinstance(model, SparModel)
    assert not model.confident
    assert model.n_solves <= 8


def test_warm_start_consumes_zero_resolves_for_seed_points():
    pre = FakeSolver()
    F_u = np.linspace(F0, F1, 15)
    y_u = pre.s11(F_u)                       # "previously solved" uniform grid

    solver = FakeSolver()                    # fresh counter
    model = adaptive_spar_model(solver, F0, F1, seed=(F_u, y_u))
    assert model.confident
    assert solver.calls == model.n_solves    # only the NEW points were solved
    assert model.n_solves < 60
    assert len(model.F) == 15 + model.n_solves


def test_uniform_equals_sweep_plus_fit():
    grid = np.linspace(F0, F1, 41)
    solver = FakeSolver()
    m_uni = uniform_spar_model(solver, grid)
    m_fit = fit_spar_model(grid, solver.s11(grid))
    dense = np.linspace(F0, F1, 5000)
    np.testing.assert_array_equal(m_uni(dense), m_fit(dense))
    assert m_uni.n_solves == len(grid)
    assert m_uni.confident


def test_minus_ndb_band():
    # |y| = |f - 658 GHz| / 100 GHz: below -20 dB (|y| < 0.1) on 648-668 GHz
    grid = np.linspace(F0, F1, 41)
    y = (grid - 658e9) / 100e9
    model = fit_spar_model(grid, y.astype(complex))
    band = minus_NdB_band(model, ndb=-20.0, center=658e9)
    assert band is not None
    f_lo, f_hi = band
    step = SPAN / 30000
    assert abs(f_lo - 648e9) < 2 * step
    assert abs(f_hi - 668e9) < 2 * step
    assert minus_NdB_band(model, ndb=-20.0, center=600e9) is None
    with pytest.raises(ValueError):
        minus_NdB_band(model, ndb=-20.0)


def test_fit_rejects_nonfinite():
    with pytest.raises(ValueError):
        fit_spar_model(np.array([1.0, 2.0, np.nan]), np.array([1, 2, 3], dtype=complex))


def test_sibling_fit_exact_grid_round_trip():
    # Clean regime: parent fitted on a generous uniform grid; the sibling LS
    # must recover the second exact-rational curve to fit-level accuracy.
    solver = FakeSolver()
    grid = np.linspace(F0, F1, 32)
    parent = fit_spar_model(grid, solver.s11(grid))
    sib = parent.sibling_fit(grid, solver.s21(grid))
    dense = np.linspace(F0, F1, 3200)
    assert np.max(np.abs(sib(dense) - solver.s21(dense))) < 1e-6
    np.testing.assert_array_equal(sib.poles(), parent.poles())   # shared, exactly
    assert sib.n_solves == 0
    assert sib.confident


def test_sibling_fit_on_adaptive_samples():
    # Production shape: samples chosen by the ADAPTIVE loop for S11, sibling
    # curve captured at the same points (the GUI's S21/a± path).
    solver = FakeSolver()
    parent = adaptive_spar_model(solver, F0, F1)
    F_s, _ = parent.samples
    sib = parent.sibling_fit(F_s, np.asarray(solver.s21(F_s)))
    dense = np.linspace(F0, F1, 20001)
    assert np.max(np.abs(sib(dense) - solver.s21(dense))) < 1e-2   # = parent tol
    np.testing.assert_array_equal(sib.poles(), parent.poles())
    # sibling interpolates/approximates its own samples, including support points
    assert np.max(np.abs(sib(F_s) - solver.s21(F_s))) < 1e-6


def test_sibling_fit_validates_input():
    solver = FakeSolver()
    grid = np.linspace(F0, F1, 32)
    parent = fit_spar_model(grid, solver.s11(grid))
    with pytest.raises(ValueError):
        parent.sibling_fit(grid, np.ones(5, dtype=complex))
    with pytest.raises(ValueError):
        parent.sibling_fit(np.array([1.0, np.nan]), np.array([1, 2], dtype=complex))
