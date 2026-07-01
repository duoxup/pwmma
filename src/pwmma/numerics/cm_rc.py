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
# These transcribe Wei Zhao's thesis eqs. (3-146)/(3-147)/(3-148) for the
# large-circle / small-rect step. The summand carries NO leading sign there; an
# earlier stray leading '-' made the whole rec<->cir coupling matrix the
# negative of the physical mode overlap (cm_rr/cm_cc were already correct),
# which corrupted only the transmission phase of mixed rec<->cir junctions.
# Regression guard: tests/test_cm_ground_truth.py. Do NOT reintroduce the '-'.

def hh_c(a, b, R, m, n, q, r, N, kc_mn, kc_qr, norm_factor, x1=None, y1=None):
    x1 = -a / 2 if x1 is None else x1
    y1 = -b / 2 if y1 is None else y1
    l = np.arange(N, dtype=float)
    inner = c_l_m(l, q, N) * phase(l, N, kc_qr, x1, y1) * a * b * imc_vec(m, kc_qr*c_l(l, N)*a) * imc_vec(n, kc_qr*s_l(l, N)*b)
    return kc_mn**2 * norm_factor * (1j)**q / N * np.sum(inner)

def hh_s(a, b, R, m, n, q, r, N, kc_mn, kc_qr, norm_factor, x1=None, y1=None):
    x1 = -a / 2 if x1 is None else x1
    y1 = -b / 2 if y1 is None else y1
    kc_mn = np.sqrt((m * np.pi / a)**2 + (n * np.pi / b)**2)
    l = np.arange(N, dtype=float)
    inner = s_l_m(l, q, N) * phase(l, N, kc_qr, x1, y1) * a * b * imc_vec(m, kc_qr*c_l(l, N)*a) * imc_vec(n, kc_qr*s_l(l, N)*b)
    return kc_mn**2 * norm_factor * (1j)**q / N * np.sum(inner)

def eh_c(*args, **kwargs):  
    return 0.

def eh_s(*args, **kwargs):
    return 0.

def he_c(a, b, R, m, n, q, r, N, kc_mn, kc_qr, norm_factor, x1=None, y1=None):  #QqrmnC
    x1 = -a / 2 if x1 is None else x1
    y1 = -b / 2 if y1 is None else y1
    l = np.arange(N, dtype=float)
    inner = c_l_m(l, q, N) * phase(l, N, kc_qr, x1, y1) * a * b * \
        (c_l(l, N) * n * np.pi / b * imc_vec(m, kc_qr*c_l(l, N)*a) * ims_vec(n, kc_qr*s_l(l, N)*b) -\
         s_l(l, N) * m * np.pi / a * ims_vec(m, kc_qr*c_l(l, N)*a) * imc_vec(n, kc_qr*s_l(l, N)*b))
    return kc_qr * norm_factor * (1j)**(q+1) / N *np.sum(inner)

def he_s(a, b, R, m, n, q, r, N, kc_mn, kc_qr, norm_factor, x1=None, y1=None):
    x1 = -a / 2 if x1 is None else x1
    y1 = -b / 2 if y1 is None else y1
    l = np.arange(N, dtype=float)
    inner = s_l_m(l, q, N) * phase(l, N, kc_qr, x1, y1) * a * b * \
        (c_l(l, N) * n * np.pi / b * imc_vec(m, kc_qr*c_l(l, N)*a) * ims_vec(n, kc_qr*s_l(l, N)*b) -\
         s_l(l, N) * m * np.pi / a * ims_vec(m, kc_qr*c_l(l, N)*a) * imc_vec(n, kc_qr*s_l(l, N)*b))
    return kc_qr * norm_factor * (1j)**(q+1) / N *np.sum(inner)

def ee_c(a, b, R, m, n, q, r, N, kc_mn, kc_qr, norm_factor, x1=None, y1=None):  #EqrmnC
    x1 = -a / 2 if x1 is None else x1
    y1 = -b / 2 if y1 is None else y1
    l = np.arange(N, dtype=float)
    inner = c_l_m(l, q, N) * phase(l,N, kc_qr, x1, y1) * a * b * ims_vec(m, kc_qr*c_l(l, N)*a) * ims_vec(n, kc_qr*s_l(l, N)*b)
    return kc_qr**2 * norm_factor * (1j)**q / N * np.sum(inner)

def ee_s(a, b, R, m, n, q, r, N, kc_mn, kc_qr, norm_factor, x1=None, y1=None):
    x1 = -a / 2 if x1 is None else x1
    y1 = -b / 2 if y1 is None else y1
    l = np.arange(N, dtype=float)
    inner = s_l_m(l, q, N) * phase(l,N, kc_qr, x1, y1) * a * b * ims_vec(m, kc_qr*c_l(l, N)*a) * ims_vec(n, kc_qr*s_l(l, N)*b)
    return kc_qr**2 * norm_factor * (1j)**q / N * np.sum(inner)

#%% Vectorized kernel (production path; the scalar hh_*/he_*/ee_* above are the oracle)

# Azimuthal quadrature points. Matches the hardcoded 1000 the scalar dispatch
# (cm._wrapper_rec2cir) passes; do not change one without the other.
_N_QUAD = 1000

_I_POW = np.array([1.0, 1j, -1.0, -1j])   # exact i**k for integer arrays (k % 4)

# Byte budget for one j-chunk's ImC/ImS tables; keeps peak memory bounded
# regardless of mode counts.
_CHUNK_BYTES = 256 * 2**20


def block_vectorized(wg1, wg2, m1, m2, n_quad=_N_QUAD):
    """(N1, N2) real coupling block for a rec(small) -> cir(large) junction.

    Bilinear ("route A") evaluation of the azimuthal quadrature. The rect
    aperture integral is separable (an ImC/ImS m-part times an n-part coupled
    only through the plane-wave angle theta_l), so for one circular mode j the
    couplings to ALL rectangular (m, n) pairs form a bilinear sum

        S[j, m, n] = sum_l W[j, l] * G[j, m, l] * H[j, n, l]
                   = (G * W[:, None, :]) @ H^T        (batched zgemm over j)

    and each rec mode i just gathers S[j, m_i, n_i]. No (N1, N2, n_quad)
    tensor is built; memory peaks at the (n_vals, jc, n_quad) tables of one
    j-chunk. The per-point table values come from the same imc_vec/ims_vec the
    scalar oracle calls, so only the reduction order differs from the scalar
    path (float-noise level, not bit-identical).

    Dispatch: (TE_rec, TE_cir)->hh, (TE_rec, TM_cir)->he, (TM_rec, TM_cir)->ee,
    (TM_rec, TE_cir)->eh == 0; the circular polar_dir picks cos/sin in the
    angular factor (uniform across hh/he/ee, matching cm._wrapper_rec2cir).
    """
    a, b = wg1.a, wg1.b
    x1, y1 = -a / 2.0, -b / 2.0

    mm = m1["mode_num1"]
    nn = m1["mode_num2"]
    mt_rec = m1["mode_type"]
    kc_rec = m1["kc"]
    norm1 = m1["norm_constant"]
    q = m2["mode_num1"]
    kcq = m2["kc"]
    mt_cir = m2["mode_type"]
    pd_cir = m2["polar_dir"]
    nf2 = m2["plus_dir"] * m2["norm_constant"]

    n1, n2 = mm.size, q.size
    out = np.zeros((n1, n2), dtype=float)
    if n1 == 0 or n2 == 0:
        return out

    # distinct rect indices: the tables are built per VALUE, modes gather by index
    mvals = np.unique(mm)
    nvals = np.unique(nn)
    mi_idx = np.searchsorted(mvals, mm)
    ni_idx = np.searchsorted(nvals, nn)

    l = np.arange(n_quad, dtype=float)
    cl = np.cos(l * 2 * np.pi / n_quad)
    sl = np.sin(l * 2 * np.pi / n_quad)

    # j-chunk size from the table byte budget (4 tables of complex128)
    per_j = 2 * (mvals.size + nvals.size) * n_quad * 16
    jc_max = max(1, int(_CHUNK_BYTES // per_j))

    for j0 in range(0, n2, jc_max):
        js = slice(j0, min(j0 + jc_max, n2))
        kcs = kcq[js][:, None]                                  # (jc, 1)
        beta_a = kcs * (cl * a)[None, :]                        # (jc, nq)
        beta_b = kcs * (sl * b)[None, :]
        phase_ = np.exp(-1j * kcs * (cl * x1 + sl * y1)[None, :])
        lq = np.outer(q[js], l) * (2 * np.pi / n_quad)
        ang = np.where((pd_cir[js] == 1)[:, None], np.cos(lq), np.sin(lq))
        W = ang * phase_ * (a * b)                              # (jc, nq)

        # ImC/ImS tables, (jc, n_vals, nq) — same evaluations as the oracle
        Gc = np.stack([imc_vec(int(v), beta_a) for v in mvals], axis=1)
        Gs = np.stack([ims_vec(int(v), beta_a) for v in mvals], axis=1)
        Hc = np.stack([imc_vec(int(v), beta_b) for v in nvals], axis=1)
        Hs = np.stack([ims_vec(int(v), beta_b) for v in nvals], axis=1)

        # batched bilinear reductions over l
        S_hh = (Gc * W[:, None, :]) @ Hc.swapaxes(1, 2)         # (jc, n_m, n_n)
        S_ee = (Gs * W[:, None, :]) @ Hs.swapaxes(1, 2)
        T1 = (Gc * (W * cl[None, :])[:, None, :]) @ Hs.swapaxes(1, 2)
        T2 = (Gs * (W * sl[None, :])[:, None, :]) @ Hc.swapaxes(1, 2)
        S_he = T1 * (nvals * np.pi / b)[None, None, :] \
            - T2 * (mvals * np.pi / a)[None, :, None]

        # gather to (jc, N1) and apply per-type prefactors
        sel_hh = S_hh[:, mi_idx, ni_idx]
        sel_he = S_he[:, mi_idx, ni_idx]
        sel_ee = S_ee[:, mi_idx, ni_idx]
        ip = _I_POW[q[js] % 4]
        ip1 = _I_POW[(q[js] + 1) % 4]
        nfj = nf2[js]
        blk_hh = ((nfj * ip / n_quad)[:, None] * norm1[None, :]
                  * kc_rec[None, :] ** 2) * sel_hh
        blk_he = ((nfj * ip1 * kcq[js] / n_quad)[:, None] * norm1[None, :]) * sel_he
        blk_ee = ((nfj * ip * kcq[js] ** 2 / n_quad)[:, None] * norm1[None, :]) * sel_ee

        te_r = (mt_rec == 1)[None, :]
        te_c = (mt_cir[js] == 1)[:, None]
        res = np.where(te_r & te_c, blk_hh.real,
                       np.where(te_r & ~te_c, blk_he.real,
                                np.where(~te_r & ~te_c, blk_ee.real, 0.0)))
        out[:, js] = res.T
    return out


if __name__ == "__main__":
    import time
    st = time.time()
    print(eh_s(2, 3, 1.5, 1, 1, 1, 0, 1000, 1.0471975511965976, 1.8415, 1))
    t1 = time.time()
    # print(_eh_s(2, 3, 1.5, 1, 1, 1, 0, 1000, 1.0471975511965976, 1.8415, 1))
    t2 = time.time()

    print(f'Time cost: {t1-st:.4f}, {t2-t1:.4f}')