#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 21:18:05 2026

@author: duoxup
"""

from waveguides import WG
from ..inputs import Transition

WGTransition = Transition

def wg_repr(wg: WG) -> str:
    ctag = wg.cross_tag.lower()
    match ctag:
        case 'rec':
            cdim = f'a{wg.a*1e3}_b{wg.b*1e3}'
        case 'cir':
            cdim = f'r{wg.r*1e3}'
        case _:
            raise ValueError(f'Unknown waveguide type \'{ctag}\'')
    mnum = f'n{wg.N}'
    return '_'.join([ctag, cdim, mnum])

def coupling_matrix_from_wgt(wgt: WGTransition, ext='.npy'):
    wg1, wg2 = wgt.wg1, wgt.wg2
    return '_&_'.join([wg_repr(wg1), wg_repr(wg2)])+ext
    
