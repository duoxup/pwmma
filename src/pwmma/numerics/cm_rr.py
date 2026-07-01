#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 20:04:19 2026

@author: duoxup
"""

import numpy as np
from numpy import pi, cos, sin

# Scalar hh/he/eh/ee (and intcc*/intss*) below are the reference oracle; the
# production path is block_vectorized at the bottom of this file.

def phase_parity(x: int):
    x_int = int(x)
    sgn = -1.0 if (x_int % 2) else 1.0
    return sgn

def intcc1(a1, a2, m1, m2, x1, x2):
    if m2 == m1 and m1 == 0:
        return a1
    elif m2*a1 == m1*a2 and m2*m1 != 0:
        return a1/2 * cos(m2 * pi / a2 * x1)
    else:
        return m2 * a2 * a1**2 / (pi * ((m2 * a1)**2 - (m1 * a2)**2)) *\
            (phase_parity(m1) * sin(m2 * pi / a2 * x2) - sin(m2 * pi / a2 * x1))

def intcc2(b1, b2, n1, n2, y1, y2):
    if n2 == n1 and n1 == 0:
        return b1
    elif n2*b1 == n1*b2 and n2*n1 != 0:
        return b1/2 * cos(n2 * pi / b2 * y1)
    else:
        return n2 * b2 * b1**2 / (pi * ((n2 * b1)**2 - (n1 * b2)**2)) *\
            (phase_parity(n1) * sin(n2 * pi / b2 * y2) - sin(n2 * pi / b2 * y1))

def intss1(a1, a2, m1, m2, x1, x2):
    if m2*a1 == m1*a2:
        return a1/2 * cos(m2 * pi / a2 * x1)
    else:
        return m1 * a2**2 * a1 / (pi * ((m2 * a1)**2 - (m1 * a2)**2)) *\
            (phase_parity(m1) * sin(m2 * pi / a2 * x2) - sin(m2 * pi / a2 * x1))

def intss2(b1, b2, n1, n2, y1, y2):
    if n2*b1 == n1*b2:
        return b1/2 * cos(n2 * pi / b2 * y1)
    else:
        return n1 * b2**2 * b1 / (pi * ((n2 * b1)**2 - (n1 * b2)**2)) *\
            (phase_parity(n1) * sin(n2 * pi / b2 * y2) - sin(n2 * pi / b2 * y1))

def xys_concentric(a1, b1, a2, b2, x1, x2, y1, y2):
    x1 = a2/2 - a1/2 if x1 is None else x1
    x2 = a2/2 + a1/2 if x2 is None else x2
    y1 = b2/2 - b1/2 if y1 is None else y1
    y2 = b2/2 + b1/2 if y2 is None else y2
    return x1, x2, y1, y2
    

def hh(a1, b1, a2, b2, m1, n1, m2, n2, kc_mn1, kc_mn2, norm_factor,
       x1=None, x2=None, y1=None, y2=None):
    x1, x2, y1, y2 = xys_concentric(a1, b1, a2, b2, x1, x2, y1, y2)
    return norm_factor * kc_mn1**2 * intcc1(a1, a2, m1, m2, x1, x2) *\
        intcc2(b1, b2, n1, n2, y1, y2)

def eh(*args, **kwargs):
    return 0.

def he(a1, b1, a2, b2, m1, n1, m2, n2, kc_mn1, kc_mn2, norm_factor,
       x1=None, x2=None, y1=None, y2=None):
    x1, x2, y1, y2 = xys_concentric(a1, b1, a2, b2, x1, x2, y1, y2)
    return norm_factor * (m1 * pi / a1 * (phase_parity(n1) * sin(n2 * pi / b2 * y2) - sin(n2 * pi / b2 * y1)) * intss1(a1, a2, m1, m2, x1, x2) \
                          -n1 * pi / b1 * (phase_parity(m1) * sin(m2 * pi / a2 * x2) - sin(m2 * pi / a2 * x1)) * intss2(b1, b2, n1, n2, y1, y2))

def ee(a1, b1, a2, b2, m1, n1, m2, n2, kc_mn1, kc_mn2, norm_factor,
       x1=None, x2=None, y1=None, y2=None):
    x1, x2, y1, y2 = xys_concentric(a1, b1, a2, b2, x1, x2, y1, y2)
    return norm_factor * kc_mn2**2 * intss1(a1, a2, m1, m2, x1, x2) *\
        intss2(b1, b2, n1, n2, y1, y2)


# %% Vectorized kernel (production path)

def _intcc_vec(a1, a2, m1, m2, x1, x2):
    """Vectorized intcc over integer index grids m1, m2 (a1, a2, x1, x2 scalar).

    Three branches, matching the scalar intcc1/intcc2:
      m2==m1==0            -> a1
      m2*a1==m1*a2 (m!=0)  -> a1/2 * cos(m2*pi/a2*x1)
      else                 -> resonance form (denominator guarded).
    """
    par1 = np.where(m1 % 2 == 0, 1.0, -1.0)
    denom = (m2 * a1) ** 2 - (m1 * a2) ** 2
    safe = np.where(denom == 0.0, 1.0, denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        general = (m2 * a2 * a1 ** 2 / (pi * safe)
                   * (par1 * np.sin(m2 * pi / a2 * x2) - np.sin(m2 * pi / a2 * x1)))
    branch2 = a1 / 2 * np.cos(m2 * pi / a2 * x1)
    out = np.where((m2 * a1 == m1 * a2) & (m2 * m1 != 0), branch2, general)
    out = np.where((m2 == m1) & (m1 == 0), a1, out)
    return out


def _intss_vec(a1, a2, m1, m2, x1, x2):
    """Vectorized intss over integer index grids m1, m2. Two branches, matching
    the scalar intss1/intss2:
      m2*a1==m1*a2 -> a1/2 * cos(m2*pi/a2*x1)
      else         -> resonance form (denominator guarded).
    """
    par1 = np.where(m1 % 2 == 0, 1.0, -1.0)
    denom = (m2 * a1) ** 2 - (m1 * a2) ** 2
    safe = np.where(denom == 0.0, 1.0, denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        general = (m1 * a2 ** 2 * a1 / (pi * safe)
                   * (par1 * np.sin(m2 * pi / a2 * x2) - np.sin(m2 * pi / a2 * x1)))
    branch1 = a1 / 2 * np.cos(m2 * pi / a2 * x1)
    return np.where(m2 * a1 == m1 * a2, branch1, general)


def block_vectorized(wg1, wg2, m1, m2):
    """(N1, N2) real coupling block for a rec(small) -> rec(large) junction.

    Vectorized equivalent of the scalar hh/he/eh/ee dispatch. ``m1``/``m2`` are
    the per-mode attribute dicts from cm._mode_attr_arrays. No polar direction
    for rectangular modes; dispatch is purely on the (mode_type1, mode_type2)
    pair (TE=1, TM=0): (1,1)->hh, (1,0)->he, (0,1)->eh=0, (0,0)->ee.
    """
    a1, b1 = wg1.a, wg1.b
    a2, b2 = wg2.a, wg2.b
    x1, x2, y1, y2 = xys_concentric(a1, b1, a2, b2, None, None, None, None)

    mt1 = m1["mode_type"][:, None]
    mt2 = m2["mode_type"][None, :]
    mm1 = m1["mode_num1"][:, None]     # wg1 m-index
    nn1 = m1["mode_num2"][:, None]     # wg1 n-index
    mm2 = m2["mode_num1"][None, :]     # wg2 m-index
    nn2 = m2["mode_num2"][None, :]     # wg2 n-index
    kc1 = m1["kc"][:, None]
    kc2 = m2["kc"][None, :]
    norm = m1["norm_constant"][:, None] * m2["norm_constant"][None, :]

    intcc1 = _intcc_vec(a1, a2, mm1, mm2, x1, x2)
    intcc2 = _intcc_vec(b1, b2, nn1, nn2, y1, y2)
    intss1 = _intss_vec(a1, a2, mm1, mm2, x1, x2)
    intss2 = _intss_vec(b1, b2, nn1, nn2, y1, y2)

    hh_block = norm * kc1 ** 2 * intcc1 * intcc2
    ee_block = norm * kc2 ** 2 * intss1 * intss2

    par_n1 = np.where(nn1 % 2 == 0, 1.0, -1.0)
    par_m1 = np.where(mm1 % 2 == 0, 1.0, -1.0)
    he_block = norm * (
        mm1 * pi / a1 * (par_n1 * np.sin(nn2 * pi / b2 * y2) - np.sin(nn2 * pi / b2 * y1)) * intss1
        - nn1 * pi / b1 * (par_m1 * np.sin(mm2 * pi / a2 * x2) - np.sin(mm2 * pi / a2 * x1)) * intss2
    )

    te1 = mt1 == 1
    te2 = mt2 == 1
    out = np.zeros(np.broadcast_shapes(mt1.shape, mt2.shape), dtype=float)
    out = np.where(te1 & te2, hh_block, out)       # hh (TE, TE)
    out = np.where(te1 & ~te2, he_block, out)      # he (TE, TM)
    out = np.where(~te1 & ~te2, ee_block, out)     # ee (TM, TM)
    # (~te1 & te2) -> eh == 0, already zero
    return out