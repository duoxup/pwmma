#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 20:04:45 2026

@author: duoxup
"""


import numpy as np
from scipy.special import jv, jvp

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






