#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np

import pwmma.gpu as gpu


def test_get_array_backend_falls_back_to_numpy_when_gpu_disabled() -> None:
    """use_gpu=False must return NumPy without probing the GPU at all."""
    assert gpu.get_array_backend(use_gpu=False) is np


def test_get_array_backend_matches_gpu_readiness() -> None:
    """use_gpu=True returns CuPy when a usable GPU is present, else NumPy."""
    xp = gpu.get_array_backend(use_gpu=True)
    if gpu.is_gpu_ready():
        import cupy as cp

        assert xp is cp
    else:
        assert xp is np


def test_gpu_detection_is_deferred_to_first_use() -> None:
    """Importing pwmma must not trigger GPU detection; it runs lazily on the
    first explicit request and is then cached for the rest of the process."""
    code = textwrap.dedent(
        """
        import pwmma.gpu as g
        at_import = g.is_gpu_ready.cache_info().currsize
        g.get_array_backend(use_gpu=False)      # CPU path must not probe the GPU
        after_cpu = g.is_gpu_ready.cache_info().currsize
        g.is_gpu_ready()                        # explicit -> evaluates and caches
        after_explicit = g.is_gpu_ready.cache_info().currsize
        print(at_import, after_cpu, after_explicit)
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    at_import, after_cpu, after_explicit = result.stdout.split()
    assert at_import == "0", "GPU detection must not run at import time"
    assert after_cpu == "0", "use_gpu=False must not trigger GPU detection"
    assert after_explicit == "1", "explicit is_gpu_ready() must evaluate and cache"
