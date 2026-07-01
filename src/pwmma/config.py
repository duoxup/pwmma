#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 20:26:35 2026

@author: duoxup
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Single flat configuration for the whole pipeline.

    Replaces the former CMConfig/SMConfig split: every computation stage is
    vectorized and in-process now, so one BLAS-thread budget covers the
    coupling-matrix build and the S-matrix sweep alike. The coupling-matrix
    disk cache stays (frequency-independent, reused across runs); S-matrix
    caching was never implemented and is intentionally dropped (its cache key
    would span frequency grid x chain x precision x N — not worth it).
    """
    # "Use about this many cores": BLAS-thread cap for the in-process
    # numerics (coupling matrices and the per-frequency GSM cascade).
    nproc: int = 8
    use_gpu: bool = True
    use_double_precision: bool = False
    # Coupling-matrix disk cache. cm_cache_dir must be set to enable caching.
    try_read_cm_from_cache: bool = False
    save_cm_to_cache: bool = False
    cm_cache_dir: Optional[str] = None
