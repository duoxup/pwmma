#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Element-wise parity of the vectorized coupling-matrix kernels against the
serial scalar oracle (``cm._calc_coupling_matrix_scalar``).

The scalar per-element path is the reference by construction; each vectorized
family kernel must reproduce it to floating-point tolerance, INCLUDING the
removable-singularity / degenerate branches. Geometries are chosen so those
branches are actually present in the mode grid (asserted per test).
"""
from __future__ import annotations

import numpy as np
from waveguides import CirWG, RecWG

import pwmma
from pwmma.numerics import cm

_PARITY = dict(rtol=1e-10, atol=1e-12)


def _scalar(wgt):
    return cm._calc_coupling_matrix_scalar(wgt)


def _vec(wgt):
    """Run the registered vectorized kernel for this junction directly."""
    key = (wgt.wg1.cross_tag.lower(), wgt.wg2.cross_tag.lower())
    kernel = cm._VEC_KERNELS[key]
    return kernel(wgt.wg1, wgt.wg2,
                  cm._mode_attr_arrays(wgt.wg1), cm._mode_attr_arrays(wgt.wg2))


# ---- cm_rr : rec (small) -> rec (large) -------------------------------------

def test_cm_rr_vectorized_matches_scalar():
    # a2 = 2*a1, b2 = 2*b1 -> commensurate, so degenerate elements
    # (m2*a1 == m1*a2, i.e. m2 == 2*m1) appear in the grid.
    small = RecWG(a=3.556e-3, b=1.778e-3, N=24)
    large = RecWG(a=7.112e-3, b=3.556e-3, N=28)
    wgt = pwmma.Transition(small, large)

    m1 = cm._mode_attr_arrays(wgt.wg1)["mode_num1"][:, None]
    m2 = cm._mode_attr_arrays(wgt.wg2)["mode_num1"][None, :]
    assert np.any((m2 * small.a == m1 * large.a) & (m2 * m1 != 0)), \
        "test geometry has no rec-rec degenerate elements to exercise"

    vec = _vec(wgt)
    ref = _scalar(wgt)
    assert vec.shape == ref.shape
    assert np.isfinite(vec).all()
    np.testing.assert_allclose(vec, ref, **_PARITY)
