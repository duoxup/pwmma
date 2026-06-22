"""Guards the removable-singularity patch in ``cm_cc.py`` and the precision of
the Bessel-zero tables behind it.

Background (see local/PATCH-cm_cc-removable-singularity.md):
``_hh_com``/``_ee_com`` divide by ``kc1**2 - kc2**2``.  When the two guides'
transverse cutoffs coincide (radius ratio = chi_dsk / chi_cir) that denominator
vanishes; the numerator vanishes too (0/0, removable), but the unpatched code
returned a spurious spike.  The patch returns the analytic limit there.

The unit tests pass the Bessel zeros to ``_*_com`` *explicitly*, so they are
independent of the lookup table.  ``test_bessel_zero_tables_match_scipy`` guards
the table precision that the real pipeline relies on to actually hit kc1 == kc2.
"""
import numpy as np
from scipy.special import jn_zeros, jnp_zeros, jv, jvp

from pwmma.numerics.cm_cc import _ee_com, _hh_com, ee_cc, hh_cc, phase_parity


def _hh_limit(q1, x1, norm):
    # §4 closed form for _hh_com at kc1 == kc2 (cir mode is TE: J'_q(x1) == 0).
    return phase_parity(q1) * np.pi * norm * (x1**2 - q1**2) * jv(q1, x1) ** 2 / 2


def _ee_limit(q1, x1, norm):
    # §4 closed form for _ee_com at kc1 == kc2 (cir mode is TM: J_q(x1) == 0).
    return phase_parity(q1) * np.pi * norm * x1**2 * jvp(q1, x1) ** 2 / 2


def test_hh_com_returns_analytic_limit_at_cutoff_coincidence():
    q1, norm, r01 = 1, 0.7, 1.0
    x1 = jnp_zeros(q1, 2)[0]          # cir TE1,1
    x2 = jnp_zeros(q1, 2)[1]          # dsk TE1,2
    r02 = x2 / x1 * r01              # ratio = rho  ->  kc1 == kc2
    val = _hh_com(r01, r02, q1, 0, q1, 0, x1, x2, norm)
    assert abs(val - _hh_limit(q1, x1, norm)) < 1e-9


def test_ee_com_returns_analytic_limit_at_cutoff_coincidence():
    q1, norm, r01 = 1, 0.7, 1.0
    x1 = jn_zeros(q1, 2)[0]           # cir TM1,1
    x2 = jn_zeros(q1, 2)[1]           # dsk TM1,2
    r02 = x2 / x1 * r01
    val = _ee_com(r01, r02, q1, 0, q1, 0, x1, x2, norm)
    assert abs(val - _ee_limit(q1, x1, norm)) < 1e-9


def test_hh_com_is_smooth_across_coincidence():
    """No spike: the value at rho equals the average of its neighbours."""
    q1, norm, r01 = 1, 0.7, 1.0
    x1 = jnp_zeros(q1, 2)[0]
    x2 = jnp_zeros(q1, 2)[1]
    rho = x2 / x1
    ratios = np.linspace(0.97 * rho, 1.03 * rho, 401)   # index 200 hits rho exactly
    vals = np.array([_hh_com(r01, r * r01, q1, 0, q1, 0, x1, x2, norm) for r in ratios])
    assert np.all(np.isfinite(vals))
    c = 200
    assert abs(ratios[c] - rho) < 1e-12                 # the exact-coincidence sample
    typ = np.median(np.abs(vals))
    assert abs(vals[c] - 0.5 * (vals[c - 1] + vals[c + 1])) < 1e-6 * typ


def test_public_hh_cc_ee_cc_inherit_the_limit_at_coincidence():
    """Patching the two _*_com helpers fixes the public wrappers automatically."""
    q1, norm, r01 = 1, 0.7, 1.0
    x1 = jnp_zeros(q1, 2)[0]
    x2 = jnp_zeros(q1, 2)[1]
    r02 = x2 / x1 * r01
    ang = jv(2 * q1, 0) * 1.0 + phase_parity(q1) * jv(0, 0) * 1.0   # d=0, theta=0
    h = hh_cc(r01, r02, q1, 0, q1, 0, x1, x2, norm, d=0, theta=0)
    assert np.isfinite(h)
    assert abs(h - _hh_limit(q1, x1, norm) * ang) < 1e-9

    y1 = jn_zeros(q1, 2)[0]
    y2 = jn_zeros(q1, 2)[1]
    r02e = y2 / y1 * r01
    e = ee_cc(r01, r02e, q1, 0, q1, 0, y1, y2, norm, d=0, theta=0)
    assert np.isfinite(e)
    assert abs(e - _ee_limit(q1, y1, norm) * ang) < 1e-9


def test_hh_com_unchanged_away_from_coincidence():
    """The patch must not perturb the ordinary (kc1 != kc2) formula."""
    q1, norm, r01, r02 = 1, 0.7, 1.0, 1.0
    x1, x2 = 3.0, 5.0                                   # kc1 = 3, kc2 = 5: not close
    plain = (phase_parity(q1) * np.pi / r01 / r02 * norm * x1**2 * x2
             * jv(q1, x1) * jvp(q1, x2 * r01 / r02)
             / ((x1 / r01) ** 2 - (x2 / r02) ** 2))
    assert _hh_com(r01, r02, q1, 0, q1, 0, x1, x2, norm) == plain


def test_bessel_zero_tables_match_scipy_to_full_precision():
    """Part 2 root cause: the shipped tables must equal SciPy's zeros."""
    from waveguides.bessel_zeros import get_tables

    A, B = get_tables()
    Q, R = 60, 50
    sj = np.array([jn_zeros(q, R) for q in range(Q)])
    sjp = np.array([jnp_zeros(q, R) for q in range(Q)])
    assert np.max(np.abs(np.asarray(A[:Q, :R]) - sj)) < 1e-12
    assert np.max(np.abs(np.asarray(B[:Q, :R]) - sjp)) < 1e-12
