#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 22:21:39 2026

@author: duoxup
"""

import logging

import numpy as np
from multiprocessing import Pool

from .numerics.cm import calc_coupling_matrix
from .inputs import Transition
from .config import CMConfig
from .utils import judge_cross_section_containment
from .io.numpy import save_coupling_matrix_to_cache, read_coupling_matrix_from_cache

logger = logging.getLogger(__name__)


















def get_coupling_matrix(wgt: Transition,
                      config: CMConfig) -> np.ndarray:
    flag_csc = judge_cross_section_containment(wgt)
    match flag_csc:
        case 0:
            return np.eye(np.max([wgt.wg1.N, wgt.wg2.N]))[0:wgt.wg1.N, 0:wgt.wg2.N]
        case 1:
            wgt_s2l = wgt
        case 2:
            wgt_s2l = wgt.swap()
        case _:
            raise ValueError('Cannot judge cross-section containment.')
    wg1, wg2 = wgt.wg1, wgt.wg2
    logger.debug('Computing coupling matrix: %s(N=%d) -> %s(N=%d)',
                 wg1.cross_tag, wg1.N, wg2.cross_tag, wg2.N)

    cm = None
    if config.cm_cache_dir is not None and config.try_read_cm_from_cache:
        try:
            cm = read_coupling_matrix_from_cache(wgt_s2l, config.cm_cache_dir)
            logger.info('Coupling matrix loaded from cache: %s', config.cm_cache_dir)
        except FileNotFoundError:
            logger.info('Coupling matrix not found in cache, computing...')
            cm = None
    if cm is None:
        nproc = config.nproc
        if config.chunksize == 'auto':
            chunksize = 64 if wgt.wg1.cross_tag != wgt.wg2.cross_tag else 8192
        else:
            chunksize = config.chunksize
        with Pool(processes=nproc) as pool:
            cm = calc_coupling_matrix(wgt_s2l, pool=pool, chunksize=chunksize)
        logger.debug('Coupling matrix computed, shape=%s', cm.shape)
        if config.cm_cache_dir is not None and config.save_cm_to_cache:
            save_coupling_matrix_to_cache(cm, wgt_s2l, config.cm_cache_dir)
            logger.info('Coupling matrix saved to cache: %s', config.cm_cache_dir)
    
    if flag_csc == 2:
        cm = cm.T
    return cm
        
        
        