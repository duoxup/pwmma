#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 21:18:05 2026

@author: duoxup
"""

from waveguides import WG
from ..inputs import Transition

WGTransition = Transition


def _dim_mm(value_m: float) -> str:
    """Format a length (metres) in millimetres, quantized to 1 nm.

    Coupling-matrix cache filenames are keyed on geometry, so they must be
    stable for the same physical dimension. Formatting the raw float would not
    be: an optimizer radius of 0.26 mm can arrive as 0.25999999999999995, whose
    ``repr`` is a different filename that misses the cache. Rounding to 1 nm
    (six decimals in mm) absorbs that noise while keeping genuinely different
    geometries distinct; trailing zeros are stripped for readability.
    """
    return f'{value_m * 1e3:.6f}'.rstrip('0').rstrip('.')


def wg_repr(wg: WG) -> str:
    ctag = wg.cross_tag.lower()
    match ctag:
        case 'rec':
            cdim = f'a{_dim_mm(wg.a)}_b{_dim_mm(wg.b)}'
        case 'cir':
            cdim = f'r{_dim_mm(wg.r)}'
        case _:
            raise ValueError(f'Unknown waveguide type \'{ctag}\'')
    mnum = f'n{wg.N}'
    return '_'.join([ctag, cdim, mnum])

def coupling_matrix_from_wgt(wgt: WGTransition, ext='.npy'):
    wg1, wg2 = wgt.wg1, wgt.wg2
    return '_&_'.join([wg_repr(wg1), wg_repr(wg2)])+ext
    
