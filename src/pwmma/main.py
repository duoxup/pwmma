#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 17:20:32 2026

@author: duoxup
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np

from .config import Config
from .inputs import Chain
from .solver import ChainSolver


#%% Main APIs
def calc_spars_of_wgchain(wgchain: 'Chain',
                          freqs: Sequence[float],
                          config: 'Config',
                          show_progress: bool = True,
                          progress_callback: Callable[[int, int], None] | None = None,
                          cms: Sequence[np.ndarray] | None = None,
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """S-parameters of a chain over a frequency sweep.

    Sugar over ``ChainSolver(wgchain, config, cms=cms).sweep(...)`` — construct
    a :class:`~pwmma.ChainSolver` directly when making repeated or
    single-frequency calls (e.g. adaptive sweeps), so the frequency-independent
    setup (coupling matrices, GPU upload) is paid once.
    """
    return ChainSolver(wgchain, config, cms=cms).sweep(
        freqs, show_progress=show_progress, progress_callback=progress_callback)
