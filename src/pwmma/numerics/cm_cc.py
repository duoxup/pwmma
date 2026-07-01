#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 20:04:45 2026

@author: duoxup
"""


import numpy as np
from scipy.special import jv, jvp

# The scalar 16 wrappers (hh_cc ... ee_ss) and cores (_hh_com/_he_com/_ee_com)
# below are the reference oracle; the production path is block_vectorized at the
# bottom of this file.

# Relative tolerance for the removable-singularity branch in _hh_com / _ee_com.
# When the two guides' transverse cutoffs coincide (kc1 == kc2) the 1/(kc1^2-kc2^2)
# factor is a removable 0/0 and is replaced by its analytic limit. With full-precision
# Bessel zeros the coincidence is exact, so a tight tolerance only catches floating
# cancellation near the pole. See local/PATCH-cm_cc-removable-singularity.md (§5.3).
_KC_RTOL = 1e-7

def A(q1, q2, theta=0):
    return np.cos((q1+q2) * theta)

def B(q1, q2, theta=0):
    return np.sin((q1+q2)  * theta)

def C(q1, q2, theta=0):
    return np.cos((q1-q2)  * theta)

def D(q1, q2, theta=0):
    return np.sin((q1-q2)  * theta)

def phase_parity(x: int):
    x_int = int(x)
    sgn = -1.0 if (x_int % 2) else 1.0
    return sgn
    
def _hh_com(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor):
    kc1 = x1 / r01
    kc2 = x2 / r02
    if np.isclose(kc1, kc2, rtol=_KC_RTOL, atol=0.0):
        # removable singularity kc1 == kc2: analytic limit (cir mode is TE,
        # J'_q(x1) == 0). See local/PATCH-cm_cc-removable-singularity.md §4.
        return phase_parity(q1) * np.pi * norm_factor * (x1**2 - q1**2) * jv(q1, x1)**2 / 2
    return  phase_parity(q1) * np.pi / r01 / r02 * norm_factor * x1**2 * x2 * \
        jv(q1, x1) * jvp(q1, x2 * r01 /r02) / ( kc1**2 - kc2**2)

def _he_com(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor):
    return phase_parity(q1) * q1 * np.pi * norm_factor * \
        jv(q1, x1) * jv(q1, x2 * r01 / r02)

def _ee_com(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor):
    kc1 = x1 / r01
    kc2 = x2 / r02
    if np.isclose(kc1, kc2, rtol=_KC_RTOL, atol=0.0):
        # removable singularity kc1 == kc2: analytic limit (cir mode is TM,
        # J_q(x1) == 0). The L'Hopital -1 cancels the (q1+1) parity -> q1.
        return phase_parity(q1) * np.pi * norm_factor * x1**2 * jvp(q1, x1)**2 / 2
    return phase_parity(q1 + 1) * np.pi * norm_factor * (x2/r02)**2 * x1 * \
        jvp(q1, x1) * jv(q1, x2 * r01 / r02) / ( kc1**2 - kc2**2)



def hh_cc(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor, d=0, theta=0):   #M2HCC
    return _hh_com(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor) * \
        (jv(q1 + q2, x1 / r02 * d) * A(q2, q1, theta) + \
         phase_parity(q2) * jv(q1 - q2, x1 / r02 * d) * C(q2, q1, theta))
    
def hh_sc(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor, d=0, theta=0):
    return _hh_com(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor) * \
        (jv(q1 + q2, x1 / r02 * d) * B(q2, q1, theta) + \
         phase_parity(q2) * jv(q1 - q2, x1 / r02 * d) * D(q2, q1, theta))
            
def hh_cs(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor, d=0, theta=0):
    return _hh_com(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor) * \
        (jv(q1 + q2, x1 / r02 * d) * B(q2, q1, theta) + \
         phase_parity(q2) * jv(q1 - q2, x1 / r02 * d) * -D(q2, q1, theta))

def hh_ss(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor, d=0, theta=0):
    return _hh_com(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor) * \
        (jv(q1 + q2, x1 / r02 * d) * -A(q2, q1, theta) + \
         phase_parity(q2) * jv(q1 - q2, x1 / r02 * d) * C(q2, q1, theta))
            
def eh_cc(*args, **kwargs):
    return 0.

def eh_sc(*args, **kwargs):
    return 0.

def eh_cs(*args, **kwargs):
    return 0.

def eh_ss(*args, **kwargs):
    return 0.



def he_sc(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor, d=0, theta=0):   #M2QCC
    return _he_com(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor) * \
        (jv(q1 + q2, x2 / r02 *d) * A(q2, q1, theta) - \
         phase_parity(q2) * jv(q1 - q2, x2 / r02 * d) * -C(q2, q1, theta))

def he_cc(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor, d=0, theta=0):
    return _he_com(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor) * \
        (jv(q1 + q2, x2 / r02 *d) * -B(q2, q1, theta) - \
         phase_parity(q2) * jv(q1 - q2, x2 / r02 * d) * D(q2, q1, theta))

def he_ss(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor, d=0, theta=0):
    return _he_com(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor) * \
        (jv(q1 + q2, x2 / r02 *d) * B(q2, q1, theta) - \
         phase_parity(q2) * jv(q1 - q2, x2 / r02 * d) * D(q2, q1, theta))

def he_cs(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor, d=0, theta=0):
    return _he_com(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor) * \
        (jv(q1 + q2, x2 / r02 *d) * A(q2, q1, theta) - \
         phase_parity(q2) * jv(q1 - q2, x2 / r02 * d) * A(q2, q1, theta)) 
        #!!!DX20260112 Here is A + A in Wei Zhao's thesis pp. 42, not A + C?
        #For now, there is no difference when theta=0
            

def ee_cc(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor, d=0, theta=0):   #M2ECC
    return _ee_com(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor) * \
        (jv(q1 + q2, x2 / r02 * d) * A(q2, q1, theta) + \
         phase_parity(q2) * jv(q1 - q2, x2 / r02 * d) * C(q2, q1, theta))

def ee_sc(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor, d=0, theta=0):
    return _ee_com(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor) * \
        (jv(q1 + q2, x2 / r02 * d) * B(q2, q1, theta) + \
         phase_parity(q2) * jv(q1 - q2, x2 / r02 * d) * D(q2, q1, theta))

def ee_cs(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor, d=0, theta=0):
    return _ee_com(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor) * \
        (jv(q1 + q2, x2 / r02 * d) * B(q2, q1, theta) + \
         phase_parity(q2) * jv(q1 - q2, x2 / r02 * d) * -D(q2, q1, theta))

def ee_ss(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor, d=0, theta=0):
    return _ee_com(r01, r02, q1, r1, q2, r2, x1, x2, norm_factor) * \
        (jv(q1 + q2, x2 / r02 * d) * -A(q2, q1, theta) + \
         phase_parity(q2) * jv(q1 - q2, x2 / r02 * d) * C(q2, q1, theta))


# %% Vectorized kernel (production path)

def _hh_com_vec(r01, r02, q1, x1, x2, norm):
    """Vectorized _hh_com over grids (q1, x1 are (N1,1); x2 is (1,N2); norm is
    (N1,N2)). Removable kc-coincidence singularity handled with a guarded
    denominator + np.where."""
    kc1, kc2 = x1 / r01, x2 / r02
    mask = np.isclose(kc1, kc2, rtol=_KC_RTOL, atol=0.0)
    par1 = np.where(q1 % 2 == 0, 1.0, -1.0)
    limit = par1 * np.pi * norm * (x1 ** 2 - q1 ** 2) * jv(q1, x1) ** 2 / 2
    denom = kc1 ** 2 - kc2 ** 2
    safe = np.where(mask, 1.0, denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        general = (par1 * np.pi / r01 / r02 * norm * x1 ** 2 * x2
                   * jv(q1, x1) * jvp(q1, x2 * r01 / r02) / safe)
    return np.where(mask, limit, general)


def _he_com_vec(r01, r02, q1, x1, x2, norm):
    par1 = np.where(q1 % 2 == 0, 1.0, -1.0)
    return par1 * q1 * np.pi * norm * jv(q1, x1) * jv(q1, x2 * r01 / r02)


def _ee_com_vec(r01, r02, q1, x1, x2, norm):
    """Vectorized _ee_com. Same guarded-denominator singularity handling."""
    kc1, kc2 = x1 / r01, x2 / r02
    mask = np.isclose(kc1, kc2, rtol=_KC_RTOL, atol=0.0)
    par1 = np.where(q1 % 2 == 0, 1.0, -1.0)          # phase_parity(q1)
    par1p1 = -par1                                     # phase_parity(q1 + 1)
    limit = par1 * np.pi * norm * x1 ** 2 * jvp(q1, x1) ** 2 / 2
    denom = kc1 ** 2 - kc2 ** 2
    safe = np.where(mask, 1.0, denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        general = (par1p1 * np.pi * norm * (x2 / r02) ** 2 * x1
                   * jvp(q1, x1) * jv(q1, x2 * r01 / r02) / safe)
    return np.where(mask, limit, general)


def block_vectorized(wg1, wg2, m1, m2, d=0.0, theta=0.0):
    """(N1, N2) real coupling block for a cir(small) -> cir(large) junction.

    Vectorized equivalent of the 16-way scalar dispatch. Dispatch is on the
    (mode_type1, mode_type2) pair (selecting the core hh/he/eh=0/ee) with the
    (polar_dir1, polar_dir2) pair folded in as vectorized angular factors.
    Written for general offset ``d`` and rotation ``theta`` (production uses the
    concentric d=0, theta=0, which the scalar wrappers also default to).
    """
    r01, r02 = wg1.r, wg2.r
    q1 = m1["mode_num1"][:, None]
    q2 = m2["mode_num1"][None, :]
    x1 = (m1["kc"] * r01)[:, None]
    x2 = (m2["kc"] * r02)[None, :]
    mt1 = m1["mode_type"][:, None]
    mt2 = m2["mode_type"][None, :]
    pd1 = m1["polar_dir"][:, None]
    pd2 = m2["polar_dir"][None, :]
    norm = (m1["plus_dir"] * m1["norm_constant"])[:, None] \
        * (m2["plus_dir"] * m2["norm_constant"])[None, :]

    hh_core = _hh_com_vec(r01, r02, q1, x1, x2, norm)
    he_core = _he_com_vec(r01, r02, q1, x1, x2, norm)
    ee_core = _ee_com_vec(r01, r02, q1, x1, x2, norm)

    par_q2 = np.where(q2 % 2 == 0, 1.0, -1.0)
    A_ = np.cos((q1 + q2) * theta)   # A(q2,q1,theta) = cos((q1+q2)theta)
    B_ = np.sin((q1 + q2) * theta)   # B(q2,q1,theta)
    C_ = np.cos((q2 - q1) * theta)   # C(q2,q1,theta)
    D_ = np.sin((q2 - q1) * theta)   # D(q2,q1,theta)
    jvp_x1 = jv(q1 + q2, x1 / r02 * d)   # hh uses x1 in the Bessel argument
    jvm_x1 = jv(q1 - q2, x1 / r02 * d)
    jvp_x2 = jv(q1 + q2, x2 / r02 * d)   # he/ee use x2
    jvm_x2 = jv(q1 - q2, x2 / r02 * d)

    def _sel(v11, v01, v10, v00):
        """Select by (polar_dir1, polar_dir2): cc/sc/cs/ss (c=1, s=0)."""
        out = np.where((pd1 == 1) & (pd2 == 1), v11, 0.0)
        out = np.where((pd1 == 0) & (pd2 == 1), v01, out)
        out = np.where((pd1 == 1) & (pd2 == 0), v10, out)
        out = np.where((pd1 == 0) & (pd2 == 0), v00, out)
        return out

    f1 = _sel(A_, B_, B_, -A_)       # hh/ee: coeff of jv(q1+q2, .)
    f2 = _sel(C_, D_, -D_, C_)       # hh/ee: coeff of parity(q2)*jv(q1-q2, .)
    hh_ang = jvp_x1 * f1 + par_q2 * jvm_x1 * f2
    ee_ang = jvp_x2 * f1 + par_q2 * jvm_x2 * f2
    he_t1 = _sel(-B_, A_, A_, B_)    # he (irregular): coeff of jv(q1+q2, .)
    he_t2 = _sel(-D_, C_, -A_, -D_)  # he: coeff of parity(q2)*jv(q1-q2, .)
    he_ang = jvp_x2 * he_t1 + par_q2 * jvm_x2 * he_t2

    out = np.zeros(np.broadcast_shapes(mt1.shape, mt2.shape), dtype=float)
    out = np.where((mt1 == 1) & (mt2 == 1), (hh_core * hh_ang).real, out)   # hh
    out = np.where((mt1 == 1) & (mt2 == 0), (he_core * he_ang).real, out)   # he
    out = np.where((mt1 == 0) & (mt2 == 0), (ee_core * ee_ang).real, out)   # ee
    # (mt1==0)&(mt2==1) -> eh == 0, already zero
    return out






