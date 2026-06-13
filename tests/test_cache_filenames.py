#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coupling-matrix cache-filename quantization.

``wg_repr`` used to format raw floats (``f'r{wg.r*1e3}'``). Optimizer-produced
geometry rarely lands on an exactly-representable value, so two waveguides with
the *same* intended dimension (differing only by sub-nm float noise) produced
different cache filenames — duplicate files and missed hits. Dimensions are now
quantized to 1 nm, which absorbs the noise while keeping genuinely different
geometries distinct.
"""

from __future__ import annotations

from waveguides import CirWG, RecWG

from pwmma.io.filenames import wg_repr


def test_wg_repr_ignores_sub_nm_float_noise() -> None:
    clean = CirWG(r=0.26e-3, N=64)
    noisy = CirWG(r=0.26e-3 + 1e-18, N=64)  # different float, same value to < 1 nm
    assert clean.r != noisy.r  # sanity: the inputs really are different floats
    assert wg_repr(clean) == wg_repr(noisy)


def test_wg_repr_ignores_float_noise_on_rectangular_dims() -> None:
    clean = RecWG(a=7.112e-3, b=3.556e-3, N=24)
    noisy = RecWG(a=7.112e-3 + 1e-18, b=3.556e-3 - 1e-18, N=24)
    assert wg_repr(clean) == wg_repr(noisy)


def test_wg_repr_distinguishes_real_dimension_changes() -> None:
    # A 0.1 um change is a real geometry change and must stay a distinct key.
    base = CirWG(r=0.26e-3, N=64)
    larger = CirWG(r=0.2601e-3, N=64)
    assert wg_repr(base) != wg_repr(larger)


def test_wg_repr_still_separates_mode_count_and_shape() -> None:
    assert wg_repr(CirWG(r=4.2e-3, N=64)) != wg_repr(CirWG(r=4.2e-3, N=96))
    assert wg_repr(CirWG(r=4.2e-3, N=64)) != wg_repr(RecWG(a=4.2e-3, b=4.2e-3, N=64))
