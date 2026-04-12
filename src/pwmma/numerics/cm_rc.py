#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 20:04:46 2026

Coupling matrix between small rec. and large cir. waveguides
@author: duoxup
"""

import numpy as np

def ImC(m, betad):
    if betad == m * np.pi:
        I = 1 / 2
    else:
        I = 1j * betad * ImM(m, betad)
    return I

def ImM(m, M):
    I = ((-1)**m * np.exp(-1j * M) - 1) / ((M)**2 - (m * np.pi)**2)
    return I

def ImS(m, betad):
    if betad == m * np.pi:
        I = -1j / 2
    else:
        I = m * np.pi * ImM(m, betad)
    return I

rtol_close = 1e-12
atol_close = 1e-15

rtol_close = 1e-21
atol_close = 1e-24

def imc_vec(m, betad, rtol=rtol_close, atol=atol_close):
    m_int = int(round(m))
    if not np.isclose(m, m_int):
        raise ValueError(f"m must be integer-like, got {m!r}")

    betad_arr = np.asarray(betad, dtype=float)

    # Special case: m == 0, match your original definition at betad==0
    if m_int == 0:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = 1j * betad_arr * ((np.exp(-1j * betad_arr) - 1.0) / (betad_arr * betad_arr))
        mask0 = np.isclose(betad_arr, 0.0, rtol=rtol, atol=atol)
        out = out.astype(complex, copy=False)
        out[mask0] = 0.5
        return out.item() if np.isscalar(betad) else out

    out = 1j * betad_arr * imm_vec_safe(m_int, betad_arr, rtol=rtol, atol=atol)
    return out.item() if np.isscalar(betad) else out

def ims_vec(m, betad, rtol=rtol_close, atol=atol_close):
    m_int = int(round(m))
    if not np.isclose(m, m_int):
        raise ValueError(f"m must be integer-like, got {m!r}")

    betad_arr = np.asarray(betad, dtype=float)

    # Special case: m == 0, match your original definition
    if m_int == 0:
        out = np.zeros_like(betad_arr, dtype=complex)
        mask0 = np.isclose(betad_arr, 0.0, rtol=rtol, atol=atol)
        out[mask0] = -0.5j
        return out.item() if np.isscalar(betad) else out

    mp = m_int * np.pi
    out = mp * imm_vec_safe(m_int, betad_arr, rtol=rtol, atol=atol)
    return out.item() if np.isscalar(betad) else out

def imm_vec_safe(m, M, rtol=rtol_close, atol=atol_close):
    m_int = int(round(m))
    if not np.isclose(m, m_int):
        raise ValueError(f"m must be integer-like, got {m!r}")

    mp = m_int * np.pi
    M = np.asarray(M, dtype=float)

    # NOTE: ImM for m=0 diverges at M->0; we keep it as-is and do NOT patch mp==0 here.
    sgn = -1.0 if (m_int % 2) else 1.0

    denom = M * M - mp * mp
    numer = sgn * np.exp(-1j * M) - 1.0

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = numer / denom

    # Removable singularity for mp != 0 at M = ±mp
    if mp != 0.0:
        mask = np.isclose(np.abs(M), mp, rtol=rtol, atol=atol)
        if np.any(mask):
            out = out.astype(complex, copy=False)
            out[mask] = -1j / (2.0 * mp)

    return out

def c_l(l, N):
    return np.cos(l * 2 * np.pi / N)

def s_l(l, N):
    return np.sin(l * 2 * np.pi / N)

def c_l_m(l, m, N):
    return np.cos(l * 2 * np.pi * m / N)

def s_l_m(l, m, N):
    return np.sin(l * 2 * np.pi * m / N)

def phase(l, N, kc_qr, x1, y1):
    return np.exp(-1j * kc_qr * ((c_l(l, N)) * x1 + s_l(l, N) * y1))

#%% Legacy functions

# def _hh_c(R, a, b, q, r, m, n, N, kc_mn, kc_qr, norm_factor):  #HqrmnC
#     x1 = -a / 2
#     y1 = -b / 2
#     aqrB = kc_qr * a
#     bqrB = kc_qr * b
#     Hc0 = 0
#     for l in range(N):
#         Cl = np.cos(l * 2 * np.pi / N)
#         Sl = np.sin(l * 2 * np.pi / N)
#         Hc00 = -np.cos(l * q * 2 * np.pi / N) * np.exp(-1j * kc_qr * (Cl * x1 + Sl * y1)) * a * b * ImC(m, Cl * aqrB) * ImC(n, Sl * bqrB)
#         Hc0 = Hc0 + Hc00
#     Hc = kc_mn**2 * norm_factor * (1j)**q / N * Hc0
#     return Hc

# def _hh_s(R, a, b, q, r, m, n, N, kc_mn, kc_qr, norm_factor):  #HqrmnC
#     x1 = -a / 2
#     y1 = -b / 2
#     aqrB = kc_qr * a
#     bqrB = kc_qr * b
#     Hc0 = 0
#     for l in range(N):
#         Cl = np.cos(l * 2 * np.pi / N)
#         Sl = np.sin(l * 2 * np.pi / N)
#         Hc00 = -np.sin(l * q * 2 * np.pi / N) * np.exp(-1j * kc_qr * (Cl * x1 + Sl * y1)) * a * b * ImC(m, Cl * aqrB) * ImC(n, Sl * bqrB)
#         Hc0 = Hc0 + Hc00
#     Hc = kc_mn**2 * norm_factor * (1j)**q / N * Hc0
#     return Hc

# def _eh_c(R, a, b, q, r, m, n, N, kc_mn, kc_qr, norm_factor, x1=None, y1=None):
#     x1 = -a / 2
#     y1 = -b / 2
#     kqrA = kc_qr
#     aqrA = kqrA * a
#     bqrA = kqrA * b
#     Qc0 = 0
#     for l in range(N):
#         Cl = np.cos(l * 2 * np.pi / N)
#         Sl = np.sin(l * 2 * np.pi / N)
#         Qc00 = -np.cos(l * q * 2 * np.pi / N) * np.exp(-1j * kqrA * (Cl * x1 + Sl * y1)) * a * b * (Cl * n * np.pi / b * ImC(m, Cl * aqrA) * ImS(n, Sl * bqrA) - Sl * m * np.pi / a * ImS(m, Cl * aqrA) * ImC(n, Sl * bqrA))
#         Qc0 = Qc0 + Qc00
#     Qc = kqrA * norm_factor * (1j)**(q + 1) / N * Qc0
#     return Qc


# def _eh_s(R, a, b, q, r, m, n, N, kc_mn, kc_qr, norm_factor, x1=None, y1=None):
#     x1 = -a / 2
#     y1 = -b / 2
#     kqrA = kc_qr
#     aqrA = kqrA * a
#     bqrA = kqrA * b
#     Qc0 = 0
#     for l in range(N):
#         Cl = np.cos(l * 2 * np.pi / N)
#         Sl = np.sin(l * 2 * np.pi / N)
#         Qc00 = -np.sin(l * q * 2 * np.pi / N) * np.exp(-1j * kqrA * (Cl * x1 + Sl * y1)) * a * b * (Cl * n * np.pi / b * ImC(m, Cl * aqrA) * ImS(n, Sl * bqrA) - Sl * m * np.pi / a * ImS(m, Cl * aqrA) * ImC(n, Sl * bqrA))
#         Qc0 = Qc0 + Qc00
#     Qc = kqrA * norm_factor * (1j)**(q + 1) / N * Qc0
#     return Qc



# def _ee_c(R, a, b, q, r, m, n, N, kc_mn, kc_qr, norm_factor, x1=None, y1=None):
#     x1 = -a / 2
#     y1 = -b / 2
#     kqrA = kc_qr
#     aqrA = kqrA * a
#     bqrA = kqrA * b
#     Ec0 = 0
#     for l in range(N):
#         Cl = np.cos(l * 2 * np.pi / N)
#         Sl = np.sin(l * 2 * np.pi / N)
#         Ec00 = -np.cos(l * q * 2 * np.pi / N) * np.exp(-1j * kqrA * (Cl * x1 + Sl * y1)) * a * b * (ImS(m, Cl * aqrA) * ImS(n, Sl * bqrA))
#         Ec0 = Ec0 + Ec00
#     Ec = (kqrA)**2 * norm_factor * (1j)**q / N * Ec0
#     return Ec

# def _ee_s(R, a, b, q, r, m, n, N, kc_mn, kc_qr, norm_factor, x1=None, y1=None):
#     x1 = -a / 2
#     y1 = -b / 2
#     kqrA = kc_qr
#     aqrA = kqrA * a
#     bqrA = kqrA * b
#     Ec0 = 0
#     for l in range(N):
#         Cl = np.cos(l * 2 * np.pi / N)
#         Sl = np.sin(l * 2 * np.pi / N)
#         Ec00 = -np.sin(l * q * 2 * np.pi / N) * np.exp(-1j * kqrA * (Cl * x1 + Sl * y1)) * a * b * (ImS(m, Cl * aqrA) * ImS(n, Sl * bqrA))
#         Ec0 = Ec0 + Ec00
#     Ec = (kqrA)**2 * norm_factor * (1j)**q / N * Ec0
#     return Ec



#%% New Functions

def hh_c(a, b, R, m, n, q, r, N, kc_mn, kc_qr, norm_factor, x1=None, y1=None):
    x1 = -a / 2 if x1 is None else x1
    y1 = -b / 2 if y1 is None else y1
    l = np.arange(N, dtype=float)
    inner = -c_l_m(l, q, N) * phase(l, N, kc_qr, x1, y1) * a * b * imc_vec(m, kc_qr*c_l(l, N)*a) * imc_vec(n, kc_qr*s_l(l, N)*b)
    return kc_mn**2 * norm_factor * (1j)**q / N * np.sum(inner)

def hh_s(a, b, R, m, n, q, r, N, kc_mn, kc_qr, norm_factor, x1=None, y1=None):
    x1 = -a / 2 if x1 is None else x1
    y1 = -b / 2 if y1 is None else y1
    kc_mn = np.sqrt((m * np.pi / a)**2 + (n * np.pi / b)**2)
    l = np.arange(N, dtype=float)
    inner = -s_l_m(l, q, N) * phase(l, N, kc_qr, x1, y1) * a * b * imc_vec(m, kc_qr*c_l(l, N)*a) * imc_vec(n, kc_qr*s_l(l, N)*b)
    return kc_mn**2 * norm_factor * (1j)**q / N * np.sum(inner)

def eh_c(*args, **kwargs):  
    return 0.

def eh_s(*args, **kwargs):
    return 0.

def he_c(a, b, R, m, n, q, r, N, kc_mn, kc_qr, norm_factor, x1=None, y1=None):  #QqrmnC
    x1 = -a / 2 if x1 is None else x1
    y1 = -b / 2 if y1 is None else y1
    l = np.arange(N, dtype=float)
    inner = -c_l_m(l, q, N) * phase(l, N, kc_qr, x1, y1) * a * b * \
        (c_l(l, N) * n * np.pi / b * imc_vec(m, kc_qr*c_l(l, N)*a) * ims_vec(n, kc_qr*s_l(l, N)*b) -\
         s_l(l, N) * m * np.pi / a * ims_vec(m, kc_qr*c_l(l, N)*a) * imc_vec(n, kc_qr*s_l(l, N)*b))
    return kc_qr * norm_factor * (1j)**(q+1) / N *np.sum(inner)

def he_s(a, b, R, m, n, q, r, N, kc_mn, kc_qr, norm_factor, x1=None, y1=None):
    x1 = -a / 2 if x1 is None else x1
    y1 = -b / 2 if y1 is None else y1
    l = np.arange(N, dtype=float)
    inner = -s_l_m(l, q, N) * phase(l, N, kc_qr, x1, y1) * a * b * \
        (c_l(l, N) * n * np.pi / b * imc_vec(m, kc_qr*c_l(l, N)*a) * ims_vec(n, kc_qr*s_l(l, N)*b) -\
         s_l(l, N) * m * np.pi / a * ims_vec(m, kc_qr*c_l(l, N)*a) * imc_vec(n, kc_qr*s_l(l, N)*b))
    return kc_qr * norm_factor * (1j)**(q+1) / N *np.sum(inner)

def ee_c(a, b, R, m, n, q, r, N, kc_mn, kc_qr, norm_factor, x1=None, y1=None):  #EqrmnC
    x1 = -a / 2 if x1 is None else x1
    y1 = -b / 2 if y1 is None else y1
    l = np.arange(N, dtype=float)
    inner = -c_l_m(l, q, N) * phase(l,N, kc_qr, x1, y1) * a * b * ims_vec(m, kc_qr*c_l(l, N)*a) * ims_vec(n, kc_qr*s_l(l, N)*b)
    return kc_qr**2 * norm_factor * (1j)**q / N * np.sum(inner)

def ee_s(a, b, R, m, n, q, r, N, kc_mn, kc_qr, norm_factor, x1=None, y1=None):
    x1 = -a / 2 if x1 is None else x1
    y1 = -b / 2 if y1 is None else y1
    l = np.arange(N, dtype=float)
    inner = -s_l_m(l, q, N) * phase(l,N, kc_qr, x1, y1) * a * b * ims_vec(m, kc_qr*c_l(l, N)*a) * ims_vec(n, kc_qr*s_l(l, N)*b)
    return kc_qr**2 * norm_factor * (1j)**q / N * np.sum(inner)

if __name__ == "__main__":
    import time
    st = time.time()
    print(eh_s(2, 3, 1.5, 1, 1, 1, 0, 1000, 1.0471975511965976, 1.8415, 1))
    t1 = time.time()
    # print(_eh_s(2, 3, 1.5, 1, 1, 1, 0, 1000, 1.0471975511965976, 1.8415, 1))
    t2 = time.time()
    
    print(f'Time cost: {t1-st:.4f}, {t2-t1:.4f}')