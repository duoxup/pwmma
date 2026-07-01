#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thread-oversubscription regression tests.

``Config.nproc`` is a BLAS-thread budget: the S-matrix sweep runs in-process
(the vectorized ``heavy_computation`` core has no pool) and its linear algebra
goes to OpenBLAS, which defaults to *all* cores. The result is that ``nproc=1``
would still saturate the machine. ``nproc`` should instead mean "use about this
many cores", so the sweep must run with the BLAS thread pool capped to
``config.nproc``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from threadpoolctl import threadpool_info
from waveguides import CirWG

import pwmma
import pwmma.analysis as analysis_mod
import pwmma.main as main_mod

_HAS_THREAD_POOL = bool(threadpool_info())
_MULTICORE = (os.cpu_count() or 1) >= 2


@pytest.mark.skipif(
    not (_HAS_THREAD_POOL and _MULTICORE),
    reason="needs a controllable native thread pool on a multi-core host",
)
def test_serial_sweep_caps_blas_threads_to_nproc(monkeypatch) -> None:
    """The serial GSM cascade runs with BLAS capped to ``sm_config.nproc``.

    Sampled from inside the per-frequency loop via the real transition-assembly
    call. Without the cap OpenBLAS reports all cores here; with it the loop must
    report exactly ``nproc``.
    """
    nproc = 2
    sampled: dict[str, int] = {}
    real = main_mod.calc_transition_scattering_matrix

    def spy(*args, **kwargs):
        sampled.setdefault(
            "n", max((p["num_threads"] for p in threadpool_info()), default=0)
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(main_mod, "calc_transition_scattering_matrix", spy)

    small = CirWG(r=3.0e-3, l=2e-3, N=12)
    large = CirWG(r=4.2e-3, l=2e-3, N=16)
    config = pwmma.Config(
        nproc=nproc, use_gpu=False, use_double_precision=False,
        try_read_cm_from_cache=False, save_cm_to_cache=False,
    )

    pwmma.calc_spars_of_wgchain(
        pwmma.Chain([small, large]), np.array([40e9]), config, show_progress=False
    )

    assert sampled["n"] == nproc


@pytest.mark.skipif(
    not (_HAS_THREAD_POOL and _MULTICORE),
    reason="needs a controllable native thread pool on a multi-core host",
)
def test_energy_sweep_caps_blas_threads_to_nproc(monkeypatch) -> None:
    """The energy-coupling sweep shares the same serial-cascade cap as the
    S-parameter sweep."""
    nproc = 2
    sampled: dict[str, int] = {}
    real = analysis_mod.calc_transition_scattering_matrix

    def spy(*args, **kwargs):
        sampled.setdefault(
            "n", max((p["num_threads"] for p in threadpool_info()), default=0)
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(analysis_mod, "calc_transition_scattering_matrix", spy)

    c1 = CirWG(r=3.0e-3, l=2e-3, N=12)
    c2 = CirWG(r=4.2e-3, l=1.5e-3, N=16)
    c3 = CirWG(r=5.4e-3, l=1.0e-3, N=20)
    config = pwmma.Config(
        nproc=nproc, use_gpu=False, use_double_precision=False,
        try_read_cm_from_cache=False, save_cm_to_cache=False,
    )

    pwmma.analyze_energy_coupling(
        pwmma.Chain([c1, c2, c3], sym=True),
        np.array([40e9]), config, sections=[1], excitation_mode=0, show_progress=False,
    )

    assert sampled["n"] == nproc
