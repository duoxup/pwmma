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


# ---- cm_cc : cir (small) -> cir (large) -------------------------------------

# scalar 16-way dispatch, keyed (mode_type1, mode_type2, polar_dir1, polar_dir2),
# mirroring cm._wrapper_cir2cir. Used as the general-(d, theta) oracle.
def _cc_dispatch():
    from pwmma.numerics import cm_cc as CC
    return {
        (1, 1, 1, 1): CC.hh_cc, (1, 1, 1, 0): CC.hh_cs,
        (1, 1, 0, 1): CC.hh_sc, (1, 1, 0, 0): CC.hh_ss,
        (1, 0, 1, 1): CC.he_cc, (1, 0, 1, 0): CC.he_cs,
        (1, 0, 0, 1): CC.he_sc, (1, 0, 0, 0): CC.he_ss,
        (0, 1, 1, 1): CC.eh_cc, (0, 1, 1, 0): CC.eh_cs,
        (0, 1, 0, 1): CC.eh_sc, (0, 1, 0, 0): CC.eh_ss,
        (0, 0, 1, 1): CC.ee_cc, (0, 0, 1, 0): CC.ee_cs,
        (0, 0, 0, 1): CC.ee_sc, (0, 0, 0, 0): CC.ee_ss,
    }


def _cc_scalar_block(wg1, wg2, d, theta):
    disp = _cc_dispatch()
    mi1, mi2 = wg1.mode_info_list, wg2.mode_info_list
    r01, r02 = wg1.r, wg2.r
    out = np.zeros((len(mi1), len(mi2)))
    for i, a in enumerate(mi1):
        for j, b in enumerate(mi2):
            f = disp[(a.mode_type, b.mode_type, a.polar_dir, b.polar_dir)]
            nf = a.plus_dir * b.plus_dir * a.norm_constant * b.norm_constant
            out[i, j] = np.real(f(r01, r02, a.mode_num1, a.mode_num2,
                                  b.mode_num1, b.mode_num2,
                                  a.kc * r01, b.kc * r02, nf, d, theta))
    return out


def _coincident_cir_pair(N):
    """Small/large circular guides whose radius ratio makes a TE_{1,1} (small)
    and TE_{1,2} (large) share the same cutoff (kc coincidence), exercising the
    removable-singularity branch of the vectorized cores."""
    from scipy.special import jnp_zeros
    z = jnp_zeros(1, 2)                 # first two zeros of J'_1
    r01 = 4.0e-3
    r02 = r01 * z[1] / z[0]            # kc(TE_1,1; r01) == kc(TE_1,2; r02)
    return CirWG(r=r01, N=N), CirWG(r=r02, N=N)


def test_cm_cc_vectorized_matches_scalar_at_coincidence():
    small, large = _coincident_cir_pair(N=40)
    wgt = pwmma.Transition(small, large)

    m1, m2 = cm._mode_attr_arrays(wgt.wg1), cm._mode_attr_arrays(wgt.wg2)
    kc1 = m1["kc"][:, None]
    kc2 = m2["kc"][None, :]
    mt1 = m1["mode_type"][:, None]
    mt2 = m2["mode_type"][None, :]
    q1 = m1["mode_num1"][:, None]
    q2 = m2["mode_num1"][None, :]
    coinc = (np.isclose(kc1, kc2, rtol=1e-7, atol=0.0)
             & (mt1 == 1) & (mt2 == 1) & (q1 == q2))
    assert np.any(coinc), "test geometry has no TE-TE kc-coincident diagonal element"

    vec = _vec(wgt)                    # assembler path, d=0, theta=0
    ref = _scalar(wgt)
    assert vec.shape == ref.shape
    assert np.isfinite(vec).all()
    np.testing.assert_allclose(vec, ref, **_PARITY)


def test_cm_cc_vectorized_matches_scalar_general_d_theta():
    from pwmma.numerics import cm_cc
    small = CirWG(r=4.2e-3, N=32)
    large = CirWG(r=6.7e-3, N=36)
    wgt = pwmma.Transition(small, large)
    d, theta = 1.3e-3, 0.37          # non-concentric, rotated

    m1, m2 = cm._mode_attr_arrays(wgt.wg1), cm._mode_attr_arrays(wgt.wg2)
    vec = cm_cc.block_vectorized(wgt.wg1, wgt.wg2, m1, m2, d=d, theta=theta)
    ref = _cc_scalar_block(wgt.wg1, wgt.wg2, d, theta)
    assert np.isfinite(vec).all()
    np.testing.assert_allclose(vec, ref, **_PARITY)


# ---- cm_cr : cir (small) -> rec (large) -------------------------------------

def test_cm_cr_vectorized_matches_scalar():
    small = CirWG(r=3.0e-3, N=32)
    large = RecWG(a=10.0e-3, b=8.0e-3, N=36)
    wgt = pwmma.Transition(small, large)     # cir smaller -> flag_csc == 1 -> cm_cr
    assert (wgt.wg1.cross_tag.lower(), wgt.wg2.cross_tag.lower()) == ("cir", "rec")

    vec = _vec(wgt)
    ref = _scalar(wgt)
    assert vec.shape == ref.shape
    assert np.isfinite(vec).all()
    np.testing.assert_allclose(vec, ref, **_PARITY)


def test_cm_cr_lommel_confluent_limit():
    """Directly exercise the vectorized _lommel confluent branch: an exact
    kc_cir == kc_rect element must take the limit and match the scalar _lommel.
    (A physical geometry can rarely hit the 1e-12 coincidence threshold, since
    kc_cir comes from a Bessel-zero table and kc_rect from a2/b2 geometry.)"""
    from pwmma.numerics import cm_cr

    R = 3.0e-3
    q = np.array([[1], [2]])                       # (2, 1)
    kc_cir = np.array([[600.0], [850.0]])          # (2, 1)
    kc_rect = np.array([[600.0, 720.0, 900.0]])    # (1, 3); [0,0] coincides
    assert kc_rect[0, 0] == kc_cir[0, 0]           # exact coincidence present

    for cir_te in (True, False):
        vec = cm_cr._lommel_vec(q, kc_cir, kc_rect, R, cir_te=cir_te)
        ref = np.array([[cm_cr._lommel(int(q[i, 0]), float(kc_cir[i, 0]),
                                       float(kc_rect[0, j]), R, cir_te=cir_te)
                         for j in range(kc_rect.shape[1])]
                        for i in range(kc_cir.shape[0])])
        assert np.isfinite(vec).all()
        np.testing.assert_allclose(vec, ref, **_PARITY)


# ---- pool removal / inert config --------------------------------------------

def test_cmconfig_nproc_is_inert():
    """The computation is pool-free now; CMConfig.nproc must not affect results
    (it is kept only as an inert field pending GUI cleanup)."""
    small = RecWG(a=3.556e-3, b=1.778e-3, N=16)
    large = CirWG(r=4.2e-3, N=40)
    t = pwmma.Transition(small, large)
    cm1 = pwmma.get_coupling_matrix(t, pwmma.CMConfig(nproc=1))
    cm7 = pwmma.get_coupling_matrix(t, pwmma.CMConfig(nproc=7))
    np.testing.assert_array_equal(cm1, cm7)
