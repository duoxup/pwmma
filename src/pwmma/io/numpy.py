#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 21:13:06 2026

@author: duoxup
"""
from __future__ import annotations

import os
import numpy as np
from importlib import resources

from . import filenames
from ..inputs import Transition

WGTransition = Transition


def read_coupling_matrix_from_cache(wgt: WGTransition,
                                    cm_cache_dir: str):
    fname = filenames.coupling_matrix_from_wgt(wgt)
    path = os.path.join(cm_cache_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r")

def save_coupling_matrix_to_cache(cm: np.ndarray,
                                   wgt: WGTransition,
                                   cm_cache_dir: str):
    os.makedirs(cm_cache_dir, exist_ok=True)
    fname = filenames.coupling_matrix_from_wgt(wgt)
    np.save(os.path.join(cm_cache_dir, fname), cm)
    
def load_memmap_from_package(filename: str) -> np.ndarray:
    """
    Load a .npy file shipped inside the 'pwmma' package.
    Returns a numpy memmap-backed array (read-only).
    """
    # resources.files("pwmma") points to the top-level package directory
    ref = resources.files("pwmma").joinpath(filename)

    # as_file() gives a real filesystem Path even if the package is in a zip
    with resources.as_file(ref) as p:  # p: Path
        return np.load(p, mmap_mode="r")