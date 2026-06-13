#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 21:18:05 2026

@author: duoxup
"""

import re

from waveguides import WG
from ..inputs import Transition

WGTransition = Transition

# One side of a coupling-matrix filename: "<cross>_n<N>" (see wg_repr). The
# cross part never ends in "_n<digits>" (dims are numbers, tags are rec/cir), so
# the trailing "_n<N>" is unambiguous.
_WG_REPR_RE = re.compile(r'^(?P<cross>.+)_n(?P<n>\d+)$')


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


def wg_cross_repr(wg: WG) -> str:
    """Cross-section key (tag + quantized dims) without the mode count.

    Two waveguides with the same cross-section but different mode counts share
    this key, which is what lets the cache reuse a larger-N matrix for a
    smaller-N request.
    """
    ctag = wg.cross_tag.lower()
    match ctag:
        case 'rec':
            cdim = f'a{_dim_mm(wg.a)}_b{_dim_mm(wg.b)}'
        case 'cir':
            cdim = f'r{_dim_mm(wg.r)}'
        case _:
            raise ValueError(f'Unknown waveguide type \'{ctag}\'')
    return f'{ctag}_{cdim}'


def wg_repr(wg: WG) -> str:
    return f'{wg_cross_repr(wg)}_n{wg.N}'

def coupling_matrix_from_wgt(wgt: WGTransition, ext='.npy'):
    wg1, wg2 = wgt.wg1, wgt.wg2
    return '_&_'.join([wg_repr(wg1), wg_repr(wg2)])+ext


def parse_coupling_matrix_filename(fname: str):
    """Parse a coupling-matrix cache filename back to its keys.

    Returns ``(cross1, n1, cross2, n2)`` or ``None`` if *fname* is not a valid
    coupling-matrix filename (a temp file, a renamed/unrelated file, etc.).
    """
    if not fname.endswith('.npy'):
        return None
    stem = fname[:-len('.npy')]
    parts = stem.split('_&_')
    if len(parts) != 2:
        return None
    sides = []
    for part in parts:
        m = _WG_REPR_RE.match(part)
        if m is None:
            return None
        sides.append((m.group('cross'), int(m.group('n'))))
    (c1, n1), (c2, n2) = sides
    return c1, n1, c2, n2
    
