# pwmma

**Pillbox Window Mode-Matching Algorithm**

A Python library for computing scattering parameters (S-parameters) of waveguide junctions using the mode-matching method. Supports rectangular and circular waveguide geometries with optional GPU acceleration.

## Requirements

### Standard dependencies (PyPI)

- Python >= 3.10
- numpy, scipy, tqdm

### Private dependencies (not on PyPI)

The following packages must be installed manually before installing `pwmma`:

| Package | Purpose |
|---------|---------|
| `waveguides` | Rectangular / circular waveguide mode definitions |

### Optional: GPU acceleration

```bash
pip install cupy-cuda12x   # adjust cuda version to match your system
```

## Installation

```bash
pip install -e .
# or with dev tools
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from waveguides import RecWG, CirWG
import pwmma

# Define waveguides
rwg = RecWG(a=7.112e-3, b=3.556e-3, N=200)
cwg = CirWG(r=4.2e-3,              N=800)
dsk = CirWG(r=5.4e-3, er=9.2,      N=800)

# Build a symmetric chain: rwg -- cwg -- dsk -- cwg -- rwg
chain = pwmma.Chain([rwg, cwg, dsk], sym=True)

# Configure computation
config = pwmma.Config(
    cmconf=pwmma.CMConfig(nproc=8),
    smconf=pwmma.SMConfig(use_gpu=True, use_double_precision=False),
)

# Frequency sweep
freqs = np.linspace(600e9, 740e9, 281)

# Compute S-parameters
s11, s12, s21, s22 = pwmma.calc_spars_of_wgchain(chain, freqs, config)
# Each is a (n_freqs, N1, N1) array of complex mode coupling coefficients.
# s11[i] = S11 matrix at freqs[i], etc.
```

## Caching coupling matrices

Coupling matrix computation can be expensive. Enable caching to persist results across runs:

```python
config = pwmma.Config(
    cmconf=pwmma.CMConfig(
        nproc=8,
        cm_cache_dir='/path/to/cm_cache',   # created automatically if absent
        try_read_cm_from_cache=True,
        save_cm_to_cache=True,
    ),
    smconf=pwmma.SMConfig(),
)
```

## Logging

By default `pwmma` is completely silent. To enable internal status messages:

```python
import logging

logging.getLogger('pwmma').setLevel(logging.INFO)   # or DEBUG for more detail
logging.basicConfig(format='%(name)s [%(levelname)s] %(message)s')
```

| Level | Information shown |
|-------|------------------|
| `INFO` | Backend (GPU/CPU), precision, cache hit/miss |
| `DEBUG` | Waveguide types and mode counts, matrix shapes |
| `WARNING` | GPU requested but unavailable, fallback to CPU |

## API reference

### Primary workflow

| Symbol | Description |
|--------|-------------|
| `calc_spars_of_wgchain(chain, freqs, config, show_progress=True)` | Compute S-parameters for a waveguide chain across a frequency sweep |
| `get_coupling_matrix(transition, cmconfig)` | Compute (or load from cache) the coupling matrix for a single junction |

### Input structures

| Symbol | Description |
|--------|-------------|
| `Chain(wgs, sym=False)` | Sequence of waveguides; `sym=True` mirrors the chain about its centre |
| `Transition(wg1, wg2)` | A single waveguide junction |

### Configuration

| Symbol | Key fields |
|--------|-----------|
| `CMConfig(nproc, ...)` | Coupling matrix: parallelism, caching |
| `SMConfig(nproc, use_gpu, use_double_precision, ...)` | Scattering matrix: backend, precision |
| `Config(cmconf, smconf)` | Container for both configs |

### Mid-level numerical API (`pwmma.numerics`)

For research use when individual computation steps are needed:

| Symbol | Description |
|--------|-------------|
| `calc_coupling_matrix(transition, pool, chunksize)` | Raw coupling matrix (no caching) |
| `calc_scattering_matrix(cm, z1, z2, ...)` | Single-junction scattering matrix |
| `cascade_generalized_scattering_matrice(SA, SB, ...)` | Redheffer star product |
| `apply_propagation_factors_to_smatrix(...)` | Shift S-matrix reference planes |

### Utilities

| Symbol | Description |
|--------|-------------|
| `detect_gpu_availability()` | Returns a dict of GPU availability checks (CuPy, nvidia-smi, etc.) |
