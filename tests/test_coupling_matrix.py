#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from waveguides import CirWG, RecWG

import pwmma


def test_get_coupling_matrix_is_orientation_consistent() -> None:
    """The coupling matrix must be transpose-consistent regardless of the order
    in which the two waveguides of a transition are given.

    Cross-section *size* selects which kernel is used; the *side* (left/right
    order) only decides whether the result is transposed. Giving the same
    physical junction in both orders must therefore yield matrices that are
    transposes of each other. The reversed order exercises the ``flag_csc == 2``
    swap branch of ``get_coupling_matrix``.
    """
    rwg = RecWG(a=7.112e-3, b=3.556e-3, N=24)  # small: fits inside cwg
    cwg = CirWG(r=4.2e-3, N=64)                # large
    config = pwmma.CMConfig(nproc=2)

    cm_small_to_large = pwmma.get_coupling_matrix(pwmma.Transition(rwg, cwg), config)  # flag_csc == 1
    cm_large_to_small = pwmma.get_coupling_matrix(pwmma.Transition(cwg, rwg), config)  # flag_csc == 2

    assert cm_small_to_large.shape == (24, 64)
    assert cm_large_to_small.shape == (64, 24)
    np.testing.assert_allclose(cm_large_to_small, cm_small_to_large.T)
