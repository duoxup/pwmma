#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 20:07:29 2026

@author: duoxup
"""
from __future__ import annotations

from typing import Tuple, Optional, Union, Iterable
import numpy as np
from multiprocessing import Pool

from ..inputs import Transition

from . import cm_cc, cm_rc, cm_cr, cm_rr

WGTransition = Transition


def _wrapper_rec2rec(args):
    i, j, a1, b1, a2, b2, mi_1, mi_2 = args
    mode_type_1, mode_type_2 = mi_1.mode_type, mi_2.mode_type
    m1, n1 = mi_1.mode_num1, mi_1.mode_num2
    m2, n2 = mi_2.mode_num1, mi_2.mode_num2
    kc_mn1 = mi_1.kc
    kc_mn2 = mi_2.kc
    norm_factor = mi_1.norm_constant * mi_2.norm_constant
    
    args_calc = a1, b1, a2, b2, m1, n1, m2, n2, kc_mn1, kc_mn2, norm_factor
    match (mode_type_1, mode_type_2):
        case (1, 1):
            func = cm_rr.hh
        case (1, 0):
            func = cm_rr.he
        case (0, 1):
            func = cm_rr.eh
        case (0, 0):
            func = cm_rr.ee
        case _:
            raise ValueError(f'Unknown mode info found: ({mode_type_1}, {mode_type_2})')
    return i, j, func(*args_calc)

def _wrapper_rec2cir(args):    #small rec to large circular --> cm_rc
    i, j, a, b, R, mi_1, mi_2 = args
    mode_type_1, mode_type_2 = mi_1.mode_type, mi_2.mode_type
    mode_polar_dir_2 = mi_2.polar_dir
    m, n = mi_1.mode_num1, mi_1.mode_num2
    q, r = mi_2.mode_num1, mi_2.mode_num2
    kc_mn = mi_1.kc
    kc_qr = mi_2.kc
    norm_factor = mi_2.plus_dir * mi_1.norm_constant * mi_2.norm_constant
    
    args_calc = a, b, R, m, n, q, r, 1000, kc_mn, kc_qr, norm_factor
    match (mode_type_1, mode_type_2, mode_polar_dir_2):
        case (1, 1, 1):
            func = cm_rc.hh_c
        case (1, 1, 0):
            func = cm_rc.hh_s
        case (0, 0, 1):
            func = cm_rc.ee_c
        case (0, 0, 0):
            func = cm_rc.ee_s
        case (1, 0, 1):
            func = cm_rc.he_c
        case (1, 0, 0):
            func = cm_rc.he_s
        case (0, 1, 1):
            func = cm_rc.eh_c
        case (0, 1, 0):
            func = cm_rc.eh_s
        case _:
            raise ValueError(f'Unknown mode info found: ({mode_type_1}, {mode_type_2}, {mode_polar_dir_2})')    
    return i, j, func(*args_calc)


def _wrapper_cir2rec(args):    #small cir to large rec --> cm_cr
    i, j, R, a, b, mi_1, mi_2 = args            # mi_1 = cir (small), mi_2 = rec (large)
    type_cir, type_rec = mi_1.mode_type, mi_2.mode_type
    polar_cir = mi_1.polar_dir
    q = mi_1.mode_num1                          # circular azimuthal index
    m, n = mi_2.mode_num1, mi_2.mode_num2       # rectangular indices
    kc_cir = mi_1.kc
    kc_rec = mi_2.kc
    norm_factor = mi_1.plus_dir * mi_1.norm_constant * mi_2.norm_constant

    args_calc = a, b, R, m, n, q, kc_cir, kc_rec, norm_factor
    match (type_cir, type_rec, polar_cir):
        case (1, 1, 1):
            func = cm_cr.hh_c
        case (1, 1, 0):
            func = cm_cr.hh_s
        case (0, 0, 1):
            func = cm_cr.ee_c
        case (0, 0, 0):
            func = cm_cr.ee_s
        case (1, 0, 1):
            func = cm_cr.he_c
        case (1, 0, 0):
            func = cm_cr.he_s
        case (0, 1, 1):
            func = cm_cr.eh_c
        case (0, 1, 0):
            func = cm_cr.eh_s
        case _:
            raise ValueError(f'Unknown mode info found: ({type_cir}, {type_rec}, {polar_cir})')
    return i, j, func(*args_calc)

def _wrapper_cir2cir(args):    #wg1 is the smaller one
    i, j, r01, r02, mi_1, mi_2 = args
    mode_type_1, mode_type_2 = mi_1.mode_type, mi_2.mode_type
    mode_polar_dir_1, mode_polar_dir_2 = mi_1.polar_dir, mi_2.polar_dir
    q1, r1 = mi_1.mode_num1, mi_1.mode_num2
    q2, r2 = mi_2.mode_num1, mi_2.mode_num2
    x1 = mi_1.kc * r01
    x2 = mi_2.kc * r02
    norm_factor = mi_1.plus_dir * mi_2.plus_dir * mi_1.norm_constant * mi_2.norm_constant
    
    args_calc = r01, r02, q1, r1, q2, r2, x1, x2, norm_factor
    match (mode_type_1, mode_type_2, mode_polar_dir_1, mode_polar_dir_2):
        case (1, 1, 1, 1):
            func = cm_cc.hh_cc
        case (1, 1, 1, 0):
            func = cm_cc.hh_cs
        case (1, 1, 0, 1):
            func = cm_cc.hh_sc
        case (1, 1, 0, 0):
            func = cm_cc.hh_ss
        case (1, 0, 1, 1):
            func = cm_cc.he_cc
        case (1, 0, 1, 0):
            func = cm_cc.he_cs
        case (1, 0, 0, 1):
            func = cm_cc.he_sc
        case (1, 0, 0, 0):
            func = cm_cc.he_ss
        case (0, 1, 1, 1):
            func = cm_cc.eh_cc
        case (0, 1, 1, 0):
            func = cm_cc.eh_cs
        case (0, 1, 0, 1):
            func = cm_cc.eh_sc
        case (0, 1, 0, 0):
            func = cm_cc.eh_ss
        case (0, 0, 1, 1):
            func = cm_cc.ee_cc
        case (0, 0, 1, 0):
            func = cm_cc.ee_cs
        case (0, 0, 0, 1):
            func = cm_cc.ee_sc
        case (0, 0, 0, 0):
            func = cm_cc.ee_ss
        case _:
            raise ValueError(f'Unknown mode info found: ({mode_type_1}, {mode_type_2}, {mode_polar_dir_1}, {mode_polar_dir_2})')
    return i, j, func(*args_calc)
    # return i, j, (mode_type_1, mode_type_2, mode_polar_dir_1, mode_polar_dir_2)


def _args_mixer(wgt: WGTransition):
    wg1, wg2 = wgt.wg1, wgt.wg2
    mi_1 = np.asarray(wgt.wg1.mode_info_list, dtype=object)
    mi_2 = np.asarray(wgt.wg2.mode_info_list, dtype=object)
    match (wg1.cross_tag.lower(), wg2.cross_tag.lower()):
        case ('rec', 'rec'):
            out = np.fromfunction(
                np.vectorize(lambda i, j: (int(i), int(j), wg1.a, wg1.b, wg2.a, wg2.b,
                                           mi_1[int(i)], mi_2[int(j)]), otypes=[object]),
                (mi_1.size, mi_2.size),
                dtype=int
                )
        case ('rec', 'cir'):
            out = np.fromfunction(
                np.vectorize(lambda i, j: (int(i), int(j), wg1.a, wg1.b, wg2.r,
                                           mi_1[int(i)], mi_2[int(j)]), otypes=[object]),
                (mi_1.size, mi_2.size),
                dtype=int
                )
        case ('cir', 'rec'):
            out = np.fromfunction(
                np.vectorize(lambda i, j: (int(i), int(j), wg1.r, wg2.a, wg2.b,
                                           mi_1[int(i)], mi_2[int(j)]), otypes=[object]),
                (mi_1.size, mi_2.size),
                dtype=int
                )
        case ('cir', 'cir'):
            out = np.fromfunction(
                np.vectorize(lambda i, j: (int(i), int(j), wg1.r, wg2.r,
                                           mi_1[int(i)], mi_2[int(j)]), otypes=[object]),
                (mi_1.size, mi_2.size),
                dtype=int
                )
        case _:
            raise ValueError(f'Got unknown waveguide(s): {wg1.cross_tag} {wg2.cross_tag}')
    return out


ComplexLike = Union[complex, np.complexfloating]

def _results_to_matrix_auto_shape(
    results: Iterable[Tuple[int, int, ComplexLike]],
    *,
    fill_value: ComplexLike = np.nan + 1j * np.nan,
    dtype: np.dtype = np.complex128,
    check_duplicates: bool = False,
    allow_negative_index: bool = False,
) -> np.ndarray:
    """
    Convert an unordered iterable of (i, j, x) into a 2D complex matrix X,
    where X[i, j] = x, and the output shape is inferred automatically as:
        (max_i + 1, max_j + 1).

    Parameters
    ----------
    results:
        Iterable of (i, j, x). Typically from pool.imap_unordered.
    fill_value:
        Value used to initialize entries that are not present in results.
    dtype:
        Output dtype for X.
    check_duplicates:
        If True, raise ValueError when duplicate (i, j) pairs exist.
        If False, later entries overwrite earlier ones.
    allow_negative_index:
        If False (default), negative i/j will raise ValueError.
        If True, negative indices are allowed (NumPy-style), but note that
        auto-shape inference still uses max(i), max(j), so negatives can be unsafe.

    Returns
    -------
    X : np.ndarray
        2D array with inferred shape (max_i+1, max_j+1).

    Raises
    ------
    ValueError:
        If results is empty, or indices are invalid, or duplicates exist (optional).
    """
    # Materialize once (imap_unordered returns an iterator)
    results = list(results)
    if len(results) == 0:
        raise ValueError("`results` is empty; cannot infer matrix shape.")

    # Convert to object array to safely hold tuples then extract columns
    res = np.asarray(results, dtype=object)
    if res.ndim != 2 or res.shape[1] != 3:
        raise ValueError("`results` must be an iterable of 3-tuples: (i, j, x).")

    ii = res[:, 0].astype(np.int64, copy=False)
    jj = res[:, 1].astype(np.int64, copy=False)

    if not allow_negative_index:
        if (ii < 0).any() or (jj < 0).any():
            bad = np.where((ii < 0) | (jj < 0))[0][0]
            raise ValueError(f"Negative index found at results[{bad}] = {results[bad]}")

    # Infer shape
    nrows = int(ii.max()) + 1
    ncols = int(jj.max()) + 1
    if nrows <= 0 or ncols <= 0:
        raise ValueError(f"Inferred invalid shape: ({nrows}, {ncols}).")

    if check_duplicates:
        # Check duplicates by converting (i, j) to linear indices
        lin = ii * ncols + jj
        if np.unique(lin).size != lin.size:
            raise ValueError("Duplicate (i, j) pairs found in `results`.")

    # Extract x and assign
    xx = np.asarray(res[:, 2], dtype=dtype)

    X = np.empty((nrows, ncols), dtype=dtype)
    X[...] = fill_value
    X[ii, jj] = xx
    return X

def _dispatcher_calc(func_wrapper,
                     iterable: Iterable,
                     *,
                     pool: Optional[Pool] = None,
                     chunksize: Optional[int]=1):
    res = []
    if pool:
        for res_pool in pool.imap_unordered(func_wrapper, iterable, chunksize=chunksize):
            res.append(res_pool)
    else:
        for args in iterable:
            res.append(func_wrapper(args))
    return res


def _wrapper_selector(wgt: WGTransition):    #precondition wg1 < wg2
    wg1, wg2 = wgt.wg1, wgt.wg2
    match (wg1.cross_tag.lower(), wg2.cross_tag.lower()):
        case ('rec', 'rec'):
            wrapper = _wrapper_rec2rec
        case ('rec', 'cir'):
            wrapper = _wrapper_rec2cir
        case ('cir', 'rec'):
            wrapper = _wrapper_cir2rec
        case ('cir', 'cir'):
            wrapper = _wrapper_cir2cir
        case _:
            raise ValueError(f'No proper function for transitioon: {wg1.cross_tag} to {wg2.cross_tag}')            
    return wrapper






#%% API
def calc_coupling_matrix(wgt: WGTransition,
                        *,
                        pool: Optional[Pool] = None,
                        chunksize: int=64):

            
    func_wrapper = _wrapper_selector(wgt)
    iterable_args = _args_mixer(wgt).flatten()
    
    pool_res = _dispatcher_calc(func_wrapper, iterable_args, pool=pool, chunksize=chunksize)
    cm = _results_to_matrix_auto_shape(pool_res, dtype=complex)
    return cm.real
