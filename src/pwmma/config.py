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
    nproc: int
    chunksize: Union[int, Literal['auto']] = 'auto'
    try_read_cm_from_cache: bool = False
    save_cm_to_cache: bool = False
    cm_cache_dir: Optional[str] = None  # must be set explicitly to enable caching

@dataclass
class SMConfig:
    nproc: int = 8
    chunksize: Union[int, Literal['auto']] = 'auto'
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