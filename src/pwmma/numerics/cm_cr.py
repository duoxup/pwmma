#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coupling matrix between a small circular and a large rectangular waveguide.

Canonical small->large orientation: wg1 = circle (small), wg2 = rectangle
(large); the overlap is integrated over the small CIRCULAR aperture. This is the
mirror of cm_rc (small rect inside large circle) and is needed whenever a circle
is smaller than a rectangle (a rect->cir contraction, or a cir->rect expansion).

Method (Wei Zhao thesis sec 3.4, eqs. 3-133..3-140; same result derived from the
field overlap): the large RECTANGULAR mode is a finite sum of <=4 plane waves
(all of magnitude k_c,rect); each plane wave, integrated against a circular mode
over the disk, reduces analytically by Jacobi-Anger (azimuth, picks the q-th
harmonic) and a Lommel integral (radius). So every matrix element is closed form
with no quadrature.

Conventions match `waveguides` exactly (rect/circle fields and norm constants),
so the result is the physical mode overlap (+GT). DO NOT add an overall sign
(that was the cm_rc bug). Guarded by tests/test_cm_ground_truth.py.

Sign/prefactor summary (cir = small/wg1, rect = large/wg2):
  hh  (TE_cir, TE_rect):  area integral,  prefactor k_c,cir^2
  ee  (TM_cir, TM_rect):  area integral,  prefactor k_c,rect^2
  he  (TE_cir, TM_rect):  rim line integral (no Lommel)
  eh  (TM_cir, TE_rect):  == 0  (TM_cir potential vanishes on the disk rim)
"""
from __future__ import annotations

import numpy as np
from scipy.special import jv, jvp

_I_POW = (1.0, 1j, -1.0, -1j)


def _ipow(k: int) -> complex:
    """Exact i**k for integer k (avoids float drift of (1j)**k)."""
    return _I_POW[int(k) % 4]


def _phi_mn(a, b, m, n):
    """Angle of the rectangular mode's spatial-frequency vector (m*pi/a, n*pi/b)."""
    return np.arctan2(n * np.pi / b, m * np.pi / a)


def _lommel(q, kc_cir, kc_rect, R, *, cir_te):
    r"""\int_0^R J_q(kc_cir r) J_q(kc_rect r) r dr  (the thesis `int bb`).

    Uses the rim zero of the circular mode: J'_q(kc_cir R)=0 (TE) or
    J_q(kc_cir R)=0 (TM). Falls back to the confluent alpha->beta limit if the
    two cutoffs happen to coincide.
    """
    a_arg = kc_cir * R
    b_arg = kc_rect * R
    denom = kc_rect**2 - kc_cir**2
    if abs(denom) < 1e-12 * max(kc_rect**2, 1.0):
        return (R**2 / 2.0) * (jvp(q, b_arg)**2 + (1.0 - (q / b_arg)**2) * jv(q, b_arg)**2)
    if cir_te:   # J'_q(kc_cir R) == 0
        return -R * kc_rect * jv(q, a_arg) * jvp(q, b_arg) / denom
    return R * kc_cir * jvp(q, a_arg) * jv(q, b_arg) / denom   # J_q(kc_cir R) == 0


# ---- area-integral angular sums over the 4 rectangular plane waves -----------
# cn = cos(n*pi/2), sn = sin(n*pi/2); im = i^m, imi = i^-m; sgnq = (-1)^q.
# Derived from cos*cos (TE rect) / sin*sin (TM rect) expanded into plane waves.

def _sigma_cos_te(q, m, n, phi):    # circle cos-variant, TE rect (Psi=cos*cos)
    cn = np.cos(n * np.pi / 2)
    return 2 * cn * np.cos(q * phi) * (_ipow(m) + (-1)**q * _ipow(-m))


def _sigma_sin_te(q, m, n, phi):    # circle sin-variant, TE rect
    sn = np.sin(n * np.pi / 2)
    return 2j * sn * np.sin(q * phi) * (_ipow(m) - (-1)**q * _ipow(-m))


def _sigma_cos_tm(q, m, n, phi):    # circle cos-variant, TM rect (Phi=sin*sin)
    sn = np.sin(n * np.pi / 2)
    return 2j * sn * np.cos(q * phi) * (_ipow(m) - (-1)**q * _ipow(-m))


def _sigma_sin_tm(q, m, n, phi):    # circle sin-variant, TM rect
    cn = np.cos(n * np.pi / 2)
    return 2 * cn * np.sin(q * phi) * (_ipow(m) + (-1)**q * _ipow(-m))


# ---- hh : TE_cir x TE_rect  (area integral, prefactor k_c,cir^2) -------------

def _hh(a, b, R, m, n, q, kc_cir, kc_rect, norm_factor, sigma):
    phi = _phi_mn(a, b, m, n)
    intbb = _lommel(q, kc_cir, kc_rect, R, cir_te=True)
    return kc_cir**2 * norm_factor * (np.pi / 2) * _ipow(q) * intbb * sigma(q, m, n, phi)


def hh_c(a, b, R, m, n, q, kc_cir, kc_rect, norm_factor):
    return _hh(a, b, R, m, n, q, kc_cir, kc_rect, norm_factor, _sigma_cos_te)


def hh_s(a, b, R, m, n, q, kc_cir, kc_rect, norm_factor):
    return _hh(a, b, R, m, n, q, kc_cir, kc_rect, norm_factor, _sigma_sin_te)


# ---- ee : TM_cir x TM_rect  (area integral, prefactor k_c,rect^2) ------------

def _ee(a, b, R, m, n, q, kc_cir, kc_rect, norm_factor, sigma):
    phi = _phi_mn(a, b, m, n)
    intbb = _lommel(q, kc_cir, kc_rect, R, cir_te=False)
    return kc_rect**2 * norm_factor * (-(np.pi / 2)) * _ipow(q) * intbb * sigma(q, m, n, phi)


def ee_c(a, b, R, m, n, q, kc_cir, kc_rect, norm_factor):
    return _ee(a, b, R, m, n, q, kc_cir, kc_rect, norm_factor, _sigma_cos_tm)


def ee_s(a, b, R, m, n, q, kc_cir, kc_rect, norm_factor):
    return _ee(a, b, R, m, n, q, kc_cir, kc_rect, norm_factor, _sigma_sin_tm)


# ---- he : TE_cir x TM_rect  (rim line integral; no radial Lommel) ------------
# C_he = +/- norm * q * J_q(kc_cir R) * (pi/2) i^q * J_q(kc_rect R) * Sigma'
#   cos-variant uses +Sigma'_S(tm) ; sin-variant uses -Sigma'_C(tm).

def he_c(a, b, R, m, n, q, kc_cir, kc_rect, norm_factor):
    phi = _phi_mn(a, b, m, n)
    pref = norm_factor * q * jv(q, kc_cir * R) * (np.pi / 2) * _ipow(q) * jv(q, kc_rect * R)
    return pref * _sigma_sin_tm(q, m, n, phi)


def he_s(a, b, R, m, n, q, kc_cir, kc_rect, norm_factor):
    phi = _phi_mn(a, b, m, n)
    pref = norm_factor * q * jv(q, kc_cir * R) * (np.pi / 2) * _ipow(q) * jv(q, kc_rect * R)
    return -pref * _sigma_cos_tm(q, m, n, phi)


# ---- eh : TM_cir x TE_rect  == 0 (TM_cir potential is zero on the disk rim) --

def eh_c(*args, **kwargs):
    return 0.0


def eh_s(*args, **kwargs):
    return 0.0
