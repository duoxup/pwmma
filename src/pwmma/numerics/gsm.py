#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 17:35:17 2026

@author: duoxup
"""

import numpy as np


def _apply_left_port_propagation_to_SB(SB, p_int, cnp=np):
    """
    Shift SB's left port reference plane by a propagation segment with one-way factor P.

    If p_int is 1D: treat P = diag(p_int) (diagonal, per-mode propagation).
    If p_int is 2D: treat p_int itself as P (general matrix).

    Transformation:
      B11' = P B11 P
      B12' = P B12
      B21' = B21 P
      B22' = B22
    """
    B11, B12, B21, B22 = SB
    B11 = cnp.asarray(B11)
    B12 = cnp.asarray(B12)
    B21 = cnp.asarray(B21)
    B22 = cnp.asarray(B22)

    P = cnp.asarray(p_int)

    if P.ndim == 1:
        # diagonal P = diag(p)
        p = P
        B11p = (p[:, None] * B11) * p[None, :]
        B12p = p[:, None] * B12
        B21p = B21 * p[None, :]
        B22p = B22
    elif P.ndim == 2:
        # general matrix P
        B11p = P @ B11 @ P
        B12p = P @ B12
        B21p = B21 @ P
        B22p = B22
    else:
        raise ValueError("p_int must be a 1D vector or a 2D matrix.")

    return B11p, B12p, B21p, B22p



def calc_scattering_matrix_sl_elim_l(Csl, zs, zl, *,
                                     cnp, dtype, conjugate_output=True):
    """
    l-side elimination, ONE solve:
      Solve (I + K_l) * [S_ll, (1/2)S_ls, A^{-1}] = [K_l - I, G, I]
    then back out S_sl, S_ss via matrix products.

    Inputs:
      C_sl  : (Ns, Nl)
      zarr_s: (Ns,)
      zarr_l: (Nl,)

    Outputs:
      S_ss: (Nl, Nl)
      S_sl: (Nl, Ns)
      S_ls: (Ns, Nl)
      S_ll: (Ns, Ns)
    """
    ns, nl = Csl.shape

    Ds = cnp.sqrt(zs)          # Z_s^{1/2}
    Dl = cnp.sqrt(zl)          # Z_l^{1/2}
    Dl_inv = 1.0 / Dl          # Z_l^{-1/2}

    # C_ls = C_sl^T  (Nl, Ns) under current convention (no conjugation)
    Cls = Csl.T

    # G = Z_l^{-1/2} C_ls Z_s^{1/2}  (Nl x Ns)
    # H = Z_s^{1/2} C_sl Z_l^{-1/2}  (Ns x Nl)
    G = (Dl_inv[:, None] * Cls) * Ds[None, :]
    H = (Ds[:, None] * Csl) * Dl_inv[None, :]

    # K_l = G @ H (Nl x Nl)
    K = G @ H

    Il = cnp.eye(nl, dtype=dtype)
    Is = cnp.eye(ns, dtype=dtype)

    A = Il + K

    # One solve with a wide RHS:
    # RHS = [K - I, G, I]
    RHS = cnp.concatenate([K - Il, G, Il], axis=1)  # (Nl, 2Nl + Ns)
    SOL = cnp.linalg.solve(A, RHS)

    S_ll = SOL[:, :nl]
    XG = SOL[:, nl:nl + ns]          # A^{-1} G
    Ainv = SOL[:, nl + ns:]          # A^{-1}

    S_ls = 2.0 * XG
    S_sl = 2.0 * (H @ Ainv)
    S_ss = Is - 2.0 * (H @ XG)
    
    if conjugate_output:
        S_ss = S_ss.conj()
        S_sl = S_sl.conj()
        S_ls = S_ls.conj()
        S_ll = S_ll.conj()
        
    return S_ss, S_sl, S_ls, S_ll


def calc_scattering_matrix_sl_elim_s(Csl, zs, zl, *,
                                     cnp, dtype, conjugate_output=True):
    """
    s-side elimination, ONE solve:
      Solve (I + K_s) * [S_ss, (1/2)S_sl, B^{-1}] = [I - K_s, H, I]
    then back out S_ls, S_ll via matrix products.

    Inputs:
      C_sl  : (Ns, Nl)
      zarr_s: (Ns,)
      zarr_l: (Nl,)

    Outputs:
      S_ss: (Nl, Nl)
      S_sl: (Nl, Ns)
      S_ls: (Ns, Nl)
      S_ll: (Ns, Ns)
    """
    ns, nl = Csl.shape

    Ds = cnp.sqrt(zs)          # Z_s^{1/2}
    Dl = cnp.sqrt(zl)          # Z_l^{1/2}
    Dl_inv = 1.0 / Dl          # Z_l^{-1/2}

    Cls = Csl.T  # (Nl, Ns)

    G = (Dl_inv[:, None] * Cls) * Ds[None, :]
    H = (Ds[:, None] * Csl) * Dl_inv[None, :]

    # K_s = H @ G (Ns x Ns)
    K = H @ G

    Il = cnp.eye(nl, dtype=dtype)
    Is = cnp.eye(ns, dtype=dtype)

    B = Is + K

    # One solve with a wide RHS:
    # RHS = [I - K, H, I]
    RHS = cnp.concatenate([Is - K, H, Is], axis=1)  # (Ns, 2Ns + Nl)
    SOL = cnp.linalg.solve(B, RHS)

    S_ss = SOL[:, :ns]
    YH = SOL[:, ns:ns + nl]         # B^{-1} H
    Binv = SOL[:, ns + nl:]         # B^{-1}

    S_sl = 2.0 * YH
    S_ls = 2.0 * (G @ Binv)
    S_ll = -Il + 2.0 * (G @ YH)
    
    if conjugate_output:
        S_ss = S_ss.conj()
        S_sl = S_sl.conj()
        S_ls = S_ls.conj()
        S_ll = S_ll.conj()

    return S_ss, S_sl, S_ls, S_ll


# _elim_s has a higher speed than _elim_l because the small side usually has
# less number of modes
calc_scattering_matrix = calc_scattering_matrix_sl_elim_s

def apply_propagation_factors_to_smatrix(
    S_ss, S_sl, S_ls, S_ll,
    p_s, p_l, *,
    cnp
):
    """
    Apply one-way modal propagation factors to shift S-matrix reference planes.

    Given one-way factors:
      p_l[m] = exp(-gamma_l[m] * L_l)
      p_s[n] = exp(-gamma_s[n] * L_s)

    Block-wise transformation:
      S_ll' = D_l S_ll D_l
      S_ls' = D_l S_ls D_s
      S_sl' = D_s S_sl D_l
      S_ss' = D_s S_ss D_s
    where D_l = diag(p_l), D_s = diag(p_s).
    """
    
    # Apply diagonal scaling without forming diag matrices
    S_ll_p = (p_l[:, None] * S_ll) * p_l[None, :]
    S_ls_p = (p_l[:, None] * S_ls) * p_s[None, :]
    S_sl_p = (p_s[:, None] * S_sl) * p_l[None, :]
    S_ss_p = (p_s[:, None] * S_ss) * p_s[None, :]

    return S_ss_p, S_sl_p, S_ls_p, S_ll_p

def cascade_generalized_scattering_matrice(SA, SB, *, p_int=None, cnp=np):
    """
    Cascade two generalized scattering matrices using Redheffer star product,
    with an optional inserted waveguide propagation segment between them.

    Topology:
      [port-1] -- A -- (internal connection) -- [optional propagation] -- B -- [port-2]

    SA = (A11, A12, A21, A22)
    SB = (B11, B12, B21, B22)

    Optional:
      p_int:
        - None: direct connection
        - 1D array (n_int,): per-mode one-way propagation factor p (P=diag(p))
        - 2D array (n_int, n_int): general propagation matrix P

    Returns:
      S_total = (S11, S12, S21, S22)
    """
    A11, A12, A21, A22 = SA
    B11, B12, B21, B22 = SB

    A11 = cnp.asarray(A11)
    A12 = cnp.asarray(A12)
    A21 = cnp.asarray(A21)
    A22 = cnp.asarray(A22)

    B11 = cnp.asarray(B11)
    B12 = cnp.asarray(B12)
    B21 = cnp.asarray(B21)
    B22 = cnp.asarray(B22)

    # Apply inserted propagation by shifting SB's left port reference plane
    if p_int is not None:
        B11, B12, B21, B22 = _apply_left_port_propagation_to_SB((B11, B12, B21, B22), p_int, cnp=cnp)

    # Internal port dimension consistency check (lightweight)
    n_int = A22.shape[0]
    if B11.shape[0] != n_int:
        raise ValueError(f"Internal port size mismatch: A22 is {A22.shape}, B11 is {B11.shape}")

    I = cnp.eye(n_int, dtype=A22.dtype)

    # Redheffer star product (avoid explicit inverse; use solve)
    # S11 = A11 + A12 (I - B11 A22)^{-1} (B11 A21)
    X = cnp.linalg.solve(I - B11 @ A22, B11 @ A21)
    S11 = A11 + A12 @ X

    # S12 = A12 (I - B11 A22)^{-1} B12
    Y = cnp.linalg.solve(I - B11 @ A22, B12)
    S12 = A12 @ Y

    # S21 = B21 (I - A22 B11)^{-1} A21
    U = cnp.linalg.solve(I - A22 @ B11, A21)
    S21 = B21 @ U

    # S22 = B22 + B21 (I - A22 B11)^{-1} (A22 B12)
    V = cnp.linalg.solve(I - A22 @ B11, A22 @ B12)
    S22 = B22 + B21 @ V

    return S11, S12, S21, S22    
    
    
    
    
    
    
    
    
    
    
    