#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 17:20:32 2026

@author: duoxup
"""
from __future__ import annotations

import logging
from typing import Callable, Sequence, Tuple
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

import waveguides.heavy_computation as hc

from .inputs import Chain
from .config import Config
from .gpu import get_array_backend
from .coupling_matrix import get_coupling_matrix
from .numerics.gsm import (calc_scattering_matrix,
                           apply_propagation_factors_to_smatrix,
                           cascade_generalized_scattering_matrice)

logger = logging.getLogger(__name__)
    
    
#%% Main APIs
def calc_spars_of_wgchain(wgchain: 'Chain',
                          freqs: Sequence[float],
                          config: 'Config',
                          show_progress: bool = True,
                          progress_callback: Callable[[int, int], None] | None = None,
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cm_config, sm_config = config.cmconf, config.smconf
    cnp = get_array_backend(sm_config.use_gpu)
    dtype = cnp.complex64 if not sm_config.use_double_precision else cnp.complex128

    backend_name = 'CuPy (GPU)' if cnp is not np else 'NumPy (CPU)'
    precision = 'complex128' if sm_config.use_double_precision else 'complex64'
    logger.info('Backend: %s | Precision: %s | Frequencies: %d | Waveguides: %d',
                backend_name, precision, len(freqs), wgchain.n_wgs)
    with Pool(processes=sm_config.nproc) as pool:
        zarr_list = [cnp.asarray(hc.impedance_array(wg, freqs, \
                     pool=pool, chunksize=2048), dtype=dtype) \
                     for wg in wgchain.wgs]
        ps_list = [cnp.asarray(hc.propagation_factor_array(wg, freqs, \
                   pool=pool, chunksize=2048), dtype=dtype) \
                   for wg in wgchain.wgs]
            
    cms = [cnp.asarray(get_coupling_matrix(wgt, cm_config), dtype=dtype) \
           for wgt in wgchain.transitions]
        
    st11 = []; st12 = []; st21 = []; st22 = []
    for idx_f, f in enumerate(tqdm(freqs, disable=not show_progress)):
        counter = 0
        SA = None
        for idx_wgt, wgt in enumerate(wgchain.transitions):
            cm = cms[idx_wgt]
            zarr1 = zarr_list[idx_wgt][idx_f]
            zarr2 = zarr_list[idx_wgt+1][idx_f]
            s11, s12, s21, s22 = calc_scattering_matrix(cm, zarr1, zarr2,
                                                        cnp=cnp, dtype=dtype,
                                                        conjugate_output=True)
            if counter == 0:
                SA = (s11, s12, s21, s22)
            else:
                p_int = ps_list[idx_wgt][idx_f]
                SB = (s11, s12, s21, s22)
                SA = cascade_generalized_scattering_matrice(SA, SB, p_int=p_int, cnp=cnp)
            counter += 1

        if wgchain.sym:
            p_int = ps_list[-1][idx_f]
            p_l_r = ps_list[0][idx_f]
            SA = cascade_generalized_scattering_matrice(SA, reversed(SA), p_int=p_int, cnp=cnp)
            S = apply_propagation_factors_to_smatrix(SA[0], SA[1], SA[2], SA[3], p_l_r, p_l_r, cnp=cnp)
        else:
            p_l = ps_list[0][idx_f]
            p_r = ps_list[-1][idx_f]
            S = apply_propagation_factors_to_smatrix(SA[0], SA[1], SA[2], SA[3], p_l, p_r, cnp=cnp)

        st11.append(S[0])
        st12.append(S[1])
        st21.append(S[2])
        st22.append(S[3])

        if progress_callback is not None:
            progress_callback(idx_f + 1, len(freqs))

    st11 = np.stack([x.get() for x in st11]) if cnp is not np else np.stack(st11)
    st12 = np.stack([x.get() for x in st12]) if cnp is not np else np.stack(st12)
    st21 = np.stack([x.get() for x in st21]) if cnp is not np else np.stack(st21)
    st22 = np.stack([x.get() for x in st22]) if cnp is not np else np.stack(st22)
    
    return (st11, st12, st21, st22)
            
            