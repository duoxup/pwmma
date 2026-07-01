#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ChainSolver: the per-frequency solver session.

Pins the Step-1 contract of the ChainSolver extraction: smatrix_at == a sweep
slice, calc_spars_of_wgchain is a pure shim, and all frequency-independent
state (coupling matrices) is hoisted to construction.
"""
from __future__ import annotations

import numpy as np
from waveguides import CirWG, RecWG

import pwmma
import pwmma.solver as solver_mod
from pwmma import ChainSolver

CFG = pwmma.Config(nproc=2, use_gpu=False)     # small, CPU-only, single precision
FREQS = np.array([28e9, 30e9, 33e9])


def _sym_chain():
    return pwmma.Chain([RecWG(a=7.112e-3, b=3.556e-3, l=2e-3, N=10),
                        CirWG(r=4.2e-3, l=1.5e-3, N=24)], sym=True)


def _nonsym_chain():
    # 3 guides -> 2 transitions, so the Redheffer cascade branch runs too
    return pwmma.Chain([RecWG(a=7.112e-3, b=3.556e-3, l=2e-3, N=10),
                        CirWG(r=4.2e-3, l=1.5e-3, N=24),
                        CirWG(r=5.4e-3, l=1.0e-3, N=32)])


def test_smatrix_at_matches_sweep_slice_sym_and_nonsym():
    for chain in (_sym_chain(), _nonsym_chain()):
        solver = ChainSolver(chain, CFG)
        swept = solver.sweep(FREQS)
        for k, f in enumerate(FREQS):
            single = solver.smatrix_at(f)
            for blk in range(4):
                np.testing.assert_allclose(single[blk], swept[blk][k], rtol=1e-6)


def test_shim_equals_solver_sweep():
    chain = _sym_chain()
    via_shim = pwmma.calc_spars_of_wgchain(chain, FREQS, CFG, show_progress=False)
    via_solver = ChainSolver(chain, CFG).sweep(FREQS)
    for blk in range(4):
        np.testing.assert_array_equal(via_shim[blk], via_solver[blk])


def test_coupling_matrices_hoisted_to_construction(monkeypatch):
    calls = []
    real = solver_mod.get_coupling_matrix

    def spy(wgt, cfg):
        calls.append(1)
        return real(wgt, cfg)

    monkeypatch.setattr(solver_mod, "get_coupling_matrix", spy)
    chain = _nonsym_chain()
    solver = ChainSolver(chain, CFG)
    assert len(calls) == len(chain.transitions)   # once per junction, at init
    solver.smatrix_at(30e9)
    solver.sweep(FREQS)
    assert len(calls) == len(chain.transitions)   # zero during solves


def test_cms_passthrough_skips_computation(monkeypatch):
    chain = _sym_chain()
    cms = [pwmma.get_coupling_matrix(wgt, CFG) for wgt in chain.transitions]
    ref = ChainSolver(chain, CFG).sweep(FREQS)

    monkeypatch.setattr(solver_mod, "get_coupling_matrix",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    solver = ChainSolver(chain, CFG, cms=cms)
    got = solver.sweep(FREQS)
    for blk in range(4):
        np.testing.assert_array_equal(got[blk], ref[blk])
