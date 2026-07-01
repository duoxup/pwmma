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

## GUI

Install the GUI extra and launch the browser app:

```bash
pip install -e ".[gui]"
pwmma-gui
```

This opens an interactive analyzer to assemble a waveguide chain, configure the
computation, run it, and explore the S-parameter and energy-coupling results
(with a live structure preview). On WSL the local server is reachable from the
Windows browser at the printed `http://localhost:<port>` address.

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

# Analyze modal net-power contributions on internal sections
analysis = pwmma.analyze_energy_coupling(chain, freqs, config, sections=2)
section = analysis.sections[2]
# section.modal_power.shape == (n_freqs, chain.wgs[2].N) for non-symmetric chains.
# For symmetric chains, section indices refer to the expanded physical chain.
fig, axes = pwmma.plot_section_energy_coupling(section)
analysis.save_npz("energy_coupling.npz")
```

## Energy coupling analysis

`analyze_energy_coupling(...)` recovers the modal net-power contribution on one or more internal
waveguide sections of a chain. It is intended for inspecting which propagating and evanescent
modes participate in energy transfer inside a multi-section window structure.

### Section indexing

The `sections` argument always uses the expanded physical chain index.

For example:

```python
chain = pwmma.Chain([rwg, cwg, dsk], sym=True)
```

is analyzed as the physical chain:

```python
[rwg, cwg, dsk, cwg, rwg]
```

so the valid internal sections are:

- `1`: left circular guide
- `2`: dielectric window
- `3`: right circular guide

### Result objects

The analysis API returns a `ChainEnergyCouplingResult`, whose `.sections` dict contains one
`SectionEnergyCoupling` per requested section.

Each `SectionEnergyCoupling` stores:

- `freqs`: frequency array
- `mode_ids`: modal indices in that waveguide section
- `modal_power`: per-mode net-power contribution, shape `(n_freqs, n_modes)`
- `propagating_mask`, `evanescent_mask`: per-frequency modal type masks
- `forward_left`, `backward_left`, `forward_right`, `backward_right`: recovered modal wave coefficients
- `reflection_power`: reflected fundamental-mode power at the input port
- `power_balance`: `reflection + propagating + evanescent`

Mode names are not stored in the result file. If a `waveguides.WG` object is attached or passed
to plotting helpers, labels are reconstructed from `mode_ids`.

### Saving and loading results

Both result classes support lightweight NPZ serialization:

```python
analysis = pwmma.analyze_energy_coupling(chain, freqs, config, sections=[1, 2, 3])
analysis.save_npz("energy_coupling_chain.npz")

reloaded = pwmma.ChainEnergyCouplingResult.load_npz("energy_coupling_chain.npz")
window = reloaded.get_section(2)

window.save_npz("window_only.npz")
window2 = pwmma.SectionEnergyCoupling.load_npz("window_only.npz")
```

### Plotting helpers

Plot one section with modal detail:

```python
section = analysis.get_section(2)
fig, axes = pwmma.plot_section_energy_coupling(
    section,
    mode_threshold=0.04,
    title="Window Section Modal Energy Coupling",
)
pwmma.save_figure(fig, "window_section.png")
```

Compare multiple sections at a glance:

```python
fig, axes = pwmma.plot_chain_energy_overview(analysis, sections=[1, 2, 3])
pwmma.save_figure(fig, "energy_coupling_overview.png")
```

Useful convenience methods on `SectionEnergyCoupling` include:

- `dominant_mode_ids(threshold=...)`
- `max_power_balance_error()`
- `get_mode_labels(wg=...)`

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
| `analyze_energy_coupling(chain, freqs, config, sections=None, excitation_mode=0, show_progress=True)` | Recover per-mode net-power contributions on one or more internal waveguide sections |
| `plot_section_energy_coupling(section, ...)` | Plot propagating/evanescent modal contributions for one section |
| `plot_chain_energy_overview(result, ...)` | Compare total propagating/evanescent power and balance error across sections |
| `save_figure(fig, path, dpi=160)` | Save a matplotlib figure and create parent directories if needed |
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
| `calc_coupling_matrix(transition, nproc=1)` | Raw coupling matrix (no caching) |
| `calc_scattering_matrix(cm, z1, z2, ...)` | Single-junction scattering matrix |
| `cascade_generalized_scattering_matrice(SA, SB, ...)` | Redheffer star product |
| `apply_propagation_factors_to_smatrix(...)` | Shift S-matrix reference planes |

### Utilities

| Symbol | Description |
|--------|-------------|
| `detect_gpu_availability()` | Returns a dict of GPU availability checks (CuPy, nvidia-smi, etc.) |
