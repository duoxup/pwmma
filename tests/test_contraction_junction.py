#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for contraction-junction (flag_csc == 2) scattering.

``calc_scattering_matrix`` derives its blocks assuming the rows of the coupling
matrix index the small/aperture side. ``get_coupling_matrix`` returns a
contraction transition in chain orientation (rows = large side), so feeding it
straight to the solver is dimensionally consistent but physically wrong. These
tests pin the two physical invariants that the bug violated, one per call site:

* ``calc_spars_of_wgchain`` (``main.py``): reversing a single junction must
  mirror its S-matrix (reciprocity of one interface).
* ``analyze_energy_coupling`` (``analysis.py``): a symmetric chain and the
  explicit non-symmetric chain describing the *same* physical structure must
  give identical modal energy, even though only the explicit one solves
  contraction base transitions directly.
"""

from __future__ import annotations

import numpy as np
from waveguides import CirWG

import pwmma


def _config() -> pwmma.Config:
    """Double precision, CPU, caching off — for exact numerical comparisons."""
    return pwmma.Config(
        cmconf=pwmma.CMConfig(
            nproc=2, try_read_cm_from_cache=False, save_cm_to_cache=False
        ),
        smconf=pwmma.SMConfig(
            nproc=2, use_gpu=False, use_double_precision=True
        ),
    )


def test_contraction_junction_smatrix_mirrors_expansion() -> None:
    """A single junction taken in both orders must yield mirrored S-matrices.

    For chain ``[large, small]`` (a contraction, flag_csc == 2) versus
    ``[small, large]`` (the expansion, flag_csc == 1), reciprocity of the same
    physical interface requires the four blocks to swap:
        S11(contraction) == S22(expansion),  S22(contraction) == S11(expansion),
        S12(contraction) == S21(expansion),  S21(contraction) == S12(expansion).
    """
    large = CirWG(r=4.2e-3, l=2e-3, N=30)
    small = CirWG(r=3.0e-3, l=2e-3, N=20)
    freqs = np.array([40e9])
    config = _config()

    s_expand = pwmma.calc_spars_of_wgchain(
        pwmma.Chain([small, large]), freqs, config, show_progress=False
    )
    s_contract = pwmma.calc_spars_of_wgchain(
        pwmma.Chain([large, small]), freqs, config, show_progress=False
    )

    # s = (S11, S12, S21, S22); reversing the chain swaps the port labels.
    np.testing.assert_allclose(s_contract[0], s_expand[3], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(s_contract[3], s_expand[0], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(s_contract[1], s_expand[2], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(s_contract[2], s_expand[1], rtol=1e-6, atol=1e-9)


def test_energy_coupling_explicit_nonsym_matches_sym() -> None:
    """The energy path must treat contraction base transitions correctly.

    ``Chain([c1, c2, c3], sym=True)`` and the explicit
    ``Chain([c1, c2, c3, c2, c1], sym=False)`` describe the same physical
    structure. The symmetric form obtains its second half by reversing
    correctly-computed expansions; the explicit form solves the trailing
    ``c3 -> c2`` and ``c2 -> c1`` contraction transitions directly. Both must
    produce identical per-mode energy in every internal section.
    """
    c1 = CirWG(r=3.0e-3, l=2e-3, N=20)
    c2 = CirWG(r=4.2e-3, l=1.5e-3, N=30)
    c3 = CirWG(r=5.4e-3, l=1.0e-3, N=40)
    freqs = np.linspace(30.0, 34.0, 2) * 1e9
    config = _config()

    sym = pwmma.analyze_energy_coupling(
        pwmma.Chain([c1, c2, c3], sym=True),
        freqs, config, sections=[1, 2, 3], excitation_mode=0, show_progress=False,
    )
    explicit = pwmma.analyze_energy_coupling(
        pwmma.Chain([c1, c2, c3, c2, c1], sym=False),
        freqs, config, sections=[1, 2, 3], excitation_mode=0, show_progress=False,
    )

    for idx in (1, 2, 3):
        a = explicit.get_section(idx)
        b = sym.get_section(idx)
        np.testing.assert_allclose(a.modal_power, b.modal_power, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            a.total_reflected_power, b.total_reflected_power, rtol=1e-6, atol=1e-8
        )
