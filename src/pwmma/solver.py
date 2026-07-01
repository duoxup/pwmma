#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-frequency solver session for a waveguide chain.

``ChainSolver`` hoists everything frequency-independent (array backend, dtype,
coupling matrices on the compute device, contraction flags) at construction, so
a single-frequency S-matrix — the primitive an adaptive frequency sweep needs —
costs exactly one impedance/propagation evaluation plus one GSM cascade.
``calc_spars_of_wgchain`` (main.py) is a thin shim over ``sweep``.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import waveguides.heavy_computation as hc
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .config import Config
from .coupling_matrix import get_coupling_matrix
from .gpu import get_array_backend
from .inputs import Chain
from .numerics.gsm import (
    apply_propagation_factors_to_smatrix,
    calc_transition_scattering_matrix,
    cascade_generalized_scattering_matrice,
)
from .utils import judge_cross_section_containment

logger = logging.getLogger(__name__)


class ChainSolver:
    """One prepared solver session = Chain x Config.

    Hoists all frequency-independent state at construction:
      - array backend (numpy/cupy) and dtype from config
      - coupling matrices: computed via get_coupling_matrix(wgt, config)
        (or taken from ``cms=``), converted ONCE to backend arrays
      - is_contraction flags (judge_cross_section_containment(wgt) == 2)
    Frequency-dependent quantities (impedance, propagation factors) are
    computed per call.
    """

    def __init__(self, chain: Chain, config: Config,
                 cms: Sequence[np.ndarray] | None = None):
        self.chain = chain
        self.config = config
        cnp = get_array_backend(config.use_gpu)
        self._cnp = cnp
        self._dtype = cnp.complex64 if not config.use_double_precision else cnp.complex128

        backend_name = 'CuPy (GPU)' if cnp is not np else 'NumPy (CPU)'
        precision = 'complex128' if config.use_double_precision else 'complex64'
        logger.info('Backend: %s | Precision: %s | Waveguides: %d',
                    backend_name, precision, chain.n_wgs)

        if cms is None:
            self._cms = [cnp.asarray(get_coupling_matrix(wgt, config), dtype=self._dtype)
                         for wgt in chain.transitions]
        else:
            self._cms = [cnp.asarray(c, dtype=self._dtype) for c in cms]

        # A contraction (wg1 larger than wg2) needs orientation-aware assembly;
        # the flag only depends on geometry, so classify each transition once.
        self._is_contraction = [judge_cross_section_containment(wgt) == 2
                                for wgt in chain.transitions]

    # ---- shared per-frequency assembly (code motion from the old sweep loop) --

    def _assemble_at(self, zarr_at_f, ps_at_f):
        """Chain GSM at ONE frequency from per-waveguide impedance/propagation
        rows. Returns the 4 backend-device matrices (s11, s12, s21, s22)."""
        cnp, dtype = self._cnp, self._dtype
        chain = self.chain
        counter = 0
        SA = None
        for idx_wgt, wgt in enumerate(chain.transitions):
            cm = self._cms[idx_wgt]
            zarr1 = zarr_at_f[idx_wgt]
            zarr2 = zarr_at_f[idx_wgt + 1]
            s11, s12, s21, s22 = calc_transition_scattering_matrix(
                cm, zarr1, zarr2,
                is_contraction=self._is_contraction[idx_wgt],
                cnp=cnp, dtype=dtype, conjugate_output=True)
            if counter == 0:
                SA = (s11, s12, s21, s22)
            else:
                p_int = ps_at_f[idx_wgt]
                SB = (s11, s12, s21, s22)
                SA = cascade_generalized_scattering_matrice(SA, SB, p_int=p_int, cnp=cnp)
            counter += 1

        if chain.sym:
            p_int = ps_at_f[-1]
            p_l_r = ps_at_f[0]
            SA = cascade_generalized_scattering_matrice(SA, reversed(SA), p_int=p_int, cnp=cnp)
            S = apply_propagation_factors_to_smatrix(SA[0], SA[1], SA[2], SA[3], p_l_r, p_l_r, cnp=cnp)
        else:
            p_l = ps_at_f[0]
            p_r = ps_at_f[-1]
            S = apply_propagation_factors_to_smatrix(SA[0], SA[1], SA[2], SA[3], p_l, p_r, cnp=cnp)
        return S

    # ---- public API -----------------------------------------------------------

    def _smatrix_at_dev(self, f: float):
        """Backend-device (s11, s12, s21, s22) at ONE frequency (no host copy)."""
        cnp, dtype = self._cnp, self._dtype
        with threadpool_limits(limits=self.config.nproc):
            # scalar fs -> shape (1, N); take row 0
            zarr = [cnp.asarray(hc.impedance_array(wg, f), dtype=dtype)[0]
                    for wg in self.chain.wgs]
            ps = [cnp.asarray(hc.propagation_factor_array(wg, f), dtype=dtype)[0]
                  for wg in self.chain.wgs]
            return self._assemble_at(zarr, ps)

    def smatrix_at(self, f: float):
        """Full generalized S-matrix (s11, s12, s21, s22) at ONE frequency.

        Returns numpy arrays (converted from GPU if needed), each of shape
        (N_port, N_port) — identical to one frequency slice of ``sweep()``.
        """
        S = self._smatrix_at_dev(float(f))
        if self._cnp is not np:
            return tuple(x.get() for x in S)
        return tuple(S)

    def sweep(self, freqs, show_progress: bool = False, progress_callback=None):
        """Batch sweep: stacked (n_freq, N, N) numpy arrays x 4.

        Bit-identical behavior and return format to the historical
        ``calc_spars_of_wgchain`` (which is now a shim over this method).
        """
        cnp, dtype = self._cnp, self._dtype
        # heavy_computation is vectorized and pool-free; cap BLAS to
        # ``config.nproc`` so this in-process sweep uses ~nproc cores instead of
        # letting OpenBLAS saturate the machine.
        with threadpool_limits(limits=self.config.nproc):
            zarr_list = [cnp.asarray(hc.impedance_array(wg, freqs), dtype=dtype)
                         for wg in self.chain.wgs]
            ps_list = [cnp.asarray(hc.propagation_factor_array(wg, freqs), dtype=dtype)
                       for wg in self.chain.wgs]

        st11, st12, st21, st22 = [], [], [], []
        # The cascade runs serially in this process; cap its BLAS to ~nproc
        # cores so nproc=1 no longer saturates the machine.
        with threadpool_limits(limits=self.config.nproc):
            for idx_f, _f in enumerate(tqdm(freqs, disable=not show_progress)):
                S = self._assemble_at([z[idx_f] for z in zarr_list],
                                      [p[idx_f] for p in ps_list])
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
