#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 20:26:35 2026

@author: duoxup
"""

from dataclasses import dataclass
from typing import Optional, Union, Literal


@dataclass
class CMConfig:
    # Core budget for the coupling-matrix computation. All four junction
    # families are vectorized in-process; nproc caps the BLAS threads of the
    # rec->cir kernel's batched GEMMs (and would size the scalar-fallback pool
    # if a vectorized kernel were ever deregistered). Interim semantics —
    # final endgame (BLAS budget vs removal) pending review.
    nproc: int
    chunksize: Union[int, Literal['auto']] = 'auto'  # inert (computed internally)
    try_read_cm_from_cache: bool = False
    save_cm_to_cache: bool = False
    cm_cache_dir: Optional[str] = None  # must be set explicitly to enable caching

@dataclass
class SMConfig:
    # BLAS-thread budget for the (in-process, vectorized) S-matrix sweep. The
    # heavy_computation core dropped multiprocessing, so this no longer sizes a
    # pool; it caps OpenBLAS so nproc=1 does not saturate every core.
    nproc: int = 8
    use_gpu: bool = True
    use_double_precision: bool = False
    normalize: bool = True
    try_read_sm_from_cache: bool = False
    save_sm_to_cache: bool = False
    sm_cache_dir: Optional[str] = None  # must be set explicitly to enable caching
    
@dataclass
class Config:
    cmconf: CMConfig
    smconf: SMConfig