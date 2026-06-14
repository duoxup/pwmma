"""Coupling matrices must equal the physical transverse-E mode overlap
``∬ e_a·e_b`` over the smaller aperture — with the CORRECT sign.

This guards the cm_rc transcription-sign fix: cm_rc.py had a stray leading ``-``
in every function (not present in Wei Zhao's thesis eqs. 3-146/3-147/3-148),
which made the rec<->cir coupling matrix the negative of the physical overlap
while cm_rr/cm_cc matched it. The sign flips only the transmission-phase of
mixed rec<->cir junctions. See local/backend-todos.md item 5a.

Ground truth: brute-force quadrature of the field functions over the small
rectangular aperture (the circular field is zero outside its own aperture, so
a grid over the rectangle is the full overlap domain).
"""
import numpy as np
from waveguides import CirWG, RecWG

from pwmma.inputs import Transition
from pwmma.numerics.cm import calc_coupling_matrix

_trap = getattr(np, "trapezoid", np.trapz)


def _overlap_over_rect(rec, other, n=601):
    """∬_{rect aperture} e^rec_i · e^other_j dA, evaluated numerically."""
    xs = np.linspace(-rec.a / 2, rec.a / 2, n)
    ys = np.linspace(-rec.b / 2, rec.b / 2, n)
    X, Y = np.meshgrid(xs, ys)
    f1 = [rec.get_mode_efield_distribution_at_gridpoints(i, X, Y) for i in range(rec.N)]
    f2 = [other.get_mode_efield_distribution_at_gridpoints(j, X, Y) for j in range(other.N)]
    C = np.empty((rec.N, other.N))
    for i, a in enumerate(f1):
        for j, b in enumerate(f2):
            integ = np.real(a.Ex) * np.real(b.Ex) + np.real(a.Ey) * np.real(b.Ey)
            C[i, j] = _trap(_trap(integ, xs, axis=1), ys)
    return C


def test_cm_rc_matches_physical_overlap_with_correct_sign():
    rec = RecWG(a=3.0, b=1.3, l=1, N=6, er=1)   # small (rows)
    cir = CirWG(r=4.0, l=1, N=6, er=1)          # large (cols)
    analytic = calc_coupling_matrix(Transition(rec, cir))
    gt = _overlap_over_rect(rec, cir)
    nrm = np.linalg.norm(gt)
    # correct sign: cm_rc == +overlap (a stray global -1 would fail this)
    assert np.linalg.norm(analytic - gt) / nrm < 1e-2
    # sanity: the matrix is non-trivial, so the flipped sign is clearly wrong
    assert np.linalg.norm(analytic + gt) / nrm > 1.0


def test_cm_rr_matches_physical_overlap():
    small = RecWG(a=3.0, b=1.3, l=1, N=6, er=1)
    large = RecWG(a=7.0, b=4.0, l=1, N=6, er=1)
    analytic = calc_coupling_matrix(Transition(small, large))
    gt = _overlap_over_rect(small, large)
    assert np.linalg.norm(analytic - gt) / np.linalg.norm(gt) < 1e-2


def _overlap_over_disk(cir, other, nr=700, nphi=1100):
    """∬_{disk aperture} e^cir_i · e^other_j dA on a polar grid (rows=cir)."""
    rs = np.linspace(0.0, cir.r, nr)
    ph = np.linspace(0.0, 2 * np.pi, nphi)
    RHO, PHI = np.meshgrid(rs, ph)
    X, Y = RHO * np.cos(PHI), RHO * np.sin(PHI)
    f1 = [cir.get_mode_efield_distribution_at_gridpoints(i, X, Y) for i in range(cir.N)]
    f2 = [other.get_mode_efield_distribution_at_gridpoints(j, X, Y) for j in range(other.N)]
    C = np.empty((cir.N, other.N))
    for i, a in enumerate(f1):
        for j, b in enumerate(f2):
            integ = (np.real(a.Ex) * np.real(b.Ex) + np.real(a.Ey) * np.real(b.Ey)) * RHO
            C[i, j] = _trap(_trap(integ, rs, axis=1), ph)
    return C


def test_cm_cr_matches_physical_overlap_with_correct_sign():
    cir = CirWG(r=1.0, l=1, N=6, er=1)          # small (rows)
    rec = RecWG(a=5.0, b=4.0, l=1, N=6, er=1)    # large (cols)
    analytic = calc_coupling_matrix(Transition(cir, rec))
    gt = _overlap_over_disk(cir, rec)
    nrm = np.linalg.norm(gt)
    assert np.linalg.norm(analytic - gt) / nrm < 2e-2       # correct sign + magnitude
    assert np.linalg.norm(analytic + gt) / nrm > 1.0        # not the flipped sign
