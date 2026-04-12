#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 16:11:55 2026

@author: duoxup
"""

import logging
import os
import shutil
import subprocess
from typing import Sequence

logger = logging.getLogger(__name__)


def _cmd_ok(cmd: list[str]) -> bool:
    """Return True if command exists and runs with exit code 0."""
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            text=False,
        )
        return p.returncode == 0
    except Exception:
        return False
    

def detect_gpu_availability() -> dict[str, bool]:
    d: dict[str, bool] = {}

    # ---- Platform / WSL bridge ----
    d["is_wsl"] = False
    try:
        with open("/proc/version", "r", encoding="utf-8", errors="ignore") as f:
            d["is_wsl"] = ("microsoft" in f.read().lower())
    except Exception:
        d["is_wsl"] = False

    d["has_dev_dxg"] = os.path.exists("/dev/dxg")

    # ---- Driver / system-level visibility (NVIDIA) ----
    d["has_nvidia_smi"] = (shutil.which("nvidia-smi") is not None)
    d["nvidia_smi_works"] = d["has_nvidia_smi"] and _cmd_ok(["nvidia-smi", "-L"])

    # ---- PyTorch ----
    d["torch_installed"] = False
    d["torch_cuda_available"] = False
    try:
        import torch  # type: ignore
        d["torch_installed"] = True
        # This checks CUDA runtime + driver availability from PyTorch's perspective
        d["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        d["torch_installed"] = False
        d["torch_cuda_available"] = False

    # ---- TensorFlow ----
    d["tensorflow_installed"] = False
    d["tensorflow_gpu_available"] = False
    try:
        import tensorflow as tf  # type: ignore
        d["tensorflow_installed"] = True
        gpus = tf.config.list_physical_devices("GPU")
        d["tensorflow_gpu_available"] = bool(len(gpus) > 0)
    except Exception:
        d["tensorflow_installed"] = False
        d["tensorflow_gpu_available"] = False

    # ---- CuPy ----
    d["cupy_installed"] = False
    d["cupy_gpu_available"] = False
    try:
        import cupy as cp  # type: ignore
        d["cupy_installed"] = True
        # getDeviceCount() will raise if CUDA runtime/driver not usable
        try:
            n = cp.cuda.runtime.getDeviceCount()
            d["cupy_gpu_available"] = bool(n > 0)
        except Exception:
            d["cupy_gpu_available"] = False
    except Exception:
        d["cupy_installed"] = False
        d["cupy_gpu_available"] = False

    # ---- JAX ----
    d["jax_installed"] = False
    d["jax_gpu_available"] = False
    try:
        import jax  # type: ignore
        d["jax_installed"] = True
        try:
            # If GPU backend isn't present, JAX typically reports only CPU devices.
            d["jax_gpu_available"] = any(dev.platform == "gpu" for dev in jax.devices())
        except Exception:
            d["jax_gpu_available"] = False
    except Exception:
        d["jax_installed"] = False
        d["jax_gpu_available"] = False

    # ---- Numba CUDA ----
    d["numba_installed"] = False
    d["numba_cuda_available"] = False
    try:
        import numba  # type: ignore
        from numba import cuda  # type: ignore
        d["numba_installed"] = True
        try:
            d["numba_cuda_available"] = bool(cuda.is_available())
        except Exception:
            d["numba_cuda_available"] = False
    except Exception:
        d["numba_installed"] = False
        d["numba_cuda_available"] = False

    return d

def is_gpu_ready(keys: Sequence[str]=['cupy_gpu_available',
                                      'has_dev_dxg',
                                      'nvidia_smi_works',
                                      ]) -> bool:
    gpu_info = detect_gpu_availability()
    ready = all(gpu_info[key] for key in keys)
    if ready:
        logger.info('GPU is available and will be used (CuPy backend)')
    else:
        failed = [k for k in keys if not gpu_info[k]]
        logger.warning('GPU not available (failed checks: %s); falling back to CPU', failed)
    return ready

gpu_ready = is_gpu_ready()
