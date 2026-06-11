#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from waveguides import CirWG, RecWG

import pwmma


def _cpu_config() -> pwmma.Config:
    return pwmma.Config(
        cmconf=pwmma.CMConfig(nproc=2),
        smconf=pwmma.SMConfig(nproc=2, use_gpu=False),
    )


def test_calc_spars_progress_callback_runs_once_per_frequency() -> None:
    rwg = RecWG(a=7.112e-3, b=3.556e-3, l=2e-3, N=24)
    cwg = CirWG(r=4.2e-3, l=1.5e-3, N=64)
    chain = pwmma.Chain([rwg, cwg], sym=True)
    freqs = np.linspace(28.0, 34.0, 4) * 1e9

    calls: list[tuple[int, int]] = []
    pwmma.calc_spars_of_wgchain(
        chain, freqs, _cpu_config(), show_progress=False,
        progress_callback=lambda done, total: calls.append((done, total)),
    )

    assert calls == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_analyze_energy_coupling_progress_callback_runs_once_per_frequency() -> None:
    rwg = RecWG(a=7.112e-3, b=3.556e-3, l=2e-3, N=24)
    cwg = CirWG(r=4.2e-3, l=1.5e-3, N=64)
    dsk = CirWG(r=5.4e-3, l=0.26e-3, N=96, er=9.2)
    chain = pwmma.Chain([rwg, cwg, dsk], sym=True)
    freqs = np.linspace(28.0, 34.0, 3) * 1e9

    calls: list[tuple[int, int]] = []
    pwmma.analyze_energy_coupling(
        chain, freqs, _cpu_config(), sections=[2], show_progress=False,
        progress_callback=lambda done, total: calls.append((done, total)),
    )

    assert calls == [(1, 3), (2, 3), (3, 3)]
