# Pillbox Window Modal Energy Coupling Analysis Workflow

This guide walks through a complete modal energy coupling analysis of a
pillbox (box-type) window using `pwmma`.

## Physical Model

A typical pillbox window consists of three waveguide sections:

```
rectangular ── circular ── dielectric disk ── circular ── rectangular
  (input)     (transition)    (εr > 1)       (transition)    (output)
```

Pass `sym=True` to `Chain` to auto-expand a symmetric structure — you only
need to define the first half.

## Step 1: Define the Waveguide Chain

```python
import numpy as np
from waveguides import CirWG, RecWG
import pwmma

# Ka-band WR-28 pillbox window
rwg = RecWG(a=7.112e-3, b=3.556e-3, l=2e-3, N=100)   # rectangular, 100 modes
cwg = CirWG(r=4.2e-3, l=1.5e-3, N=800)                # circular transition, 800 modes
dsk = CirWG(r=5.4e-3, l=0.26e-3, N=800, er=9.2)       # dielectric disk, εr=9.2

# sym=True expands to [rwg, cwg, dsk, cwg, rwg]
chain = pwmma.Chain([rwg, cwg, dsk], sym=True)
```

| Parameter | Description |
|-----------|-------------|
| `a`, `b`  | Rectangular waveguide width / height (m) |
| `r`       | Circular waveguide radius (m) |
| `l`       | Waveguide segment length (m) |
| `N`       | Modal truncation order |
| `er`      | Relative permittivity of the dielectric |

## Step 2: Define the Frequency Sweep

```python
freqs = np.linspace(28.0, 34.0, 61) * 1e9  # 28–34 GHz, 61 points
```

## Step 3: Configure Computation Parameters

```python
config = pwmma.Config(
    nproc=4,                       # BLAS-thread budget for the whole pipeline
    use_gpu=False,                 # set to True to use CuPy
    use_double_precision=False,    # False = complex64, True = complex128
    try_read_cm_from_cache=False,  # disable coupling-matrix cache on first run
    save_cm_to_cache=False,
)
```

## Step 4: Run the Energy Coupling Analysis

```python
result = pwmma.analyze_energy_coupling(
    chain,
    freqs,
    config,
    sections=2,          # analyze section 2 (the dielectric disk)
    excitation_mode=0,   # drive TE10 (mode 0 in the rectangular waveguide)
    show_progress=True,
)
```

Section indices use **physical** numbering.  For `[rwg, cwg, dsk]` with
`sym=True`, the expanded chain has 5 segments and index 2 is the dielectric
disk cross-section.

## Step 5: Inspect the Results

```python
section = result.sections[2]   # SectionEnergyCoupling object

print(f"Modes:      {len(section.mode_ids)}")
print(f"Freq points: {len(section.freqs)}")
print(f"Power balance error: {section.max_power_balance_error():.2e}")
```

### Key Attributes

| Attribute                  | Shape            | Description |
|----------------------------|------------------|-------------|
| `modal_power`              | `(n_freq, n_mode)` | Per-mode net power |
| `reflection_power`         | `(n_freq,)`        | \|S11[exc,exc]\|² (single-mode reflection) |
| `total_reflected_power`    | `(n_freq,)`        | Σ\|S11[:,exc]\|² (total reflection) |
| `total_propagating_power`  | `(n_freq,)`        | Summed propagating-mode power |
| `total_evanescent_power`   | `(n_freq,)`        | Summed evanescent-mode power |
| `power_balance`            | `(n_freq,)`        | Power conservation check (≈ 1 for lossless) |

## Step 6: Visualization

### 6a. Line Plot (Four Panels)

```python
fig, axes = pwmma.plot_section_energy_coupling(
    section,
    mode_threshold=0.04,   # modes below 4% peak are aggregated into "others"
    title="Window Section Modal Energy Coupling",
)
pwmma.save_figure(fig, "ka_window_line.png")
```

Four stacked panels:
1. **Prop. modes** — top-N propagating modes (individual traces) + remainder (gray)
2. **Evan. modes** — evanescent-mode contributions (same layout)
3. **Power summary** — total reflection, propagation, evanescent power, and sum (linear)
4. **dB** — \|S11[e,e]\|² and \|S21[e,e]\|² on a log scale

### 6b. Heatmap

```python
fig, ax = pwmma.plot_section_energy_heatmap(
    section,
    title="Ka-band Window — All Modes",
)
pwmma.save_figure(fig, "ka_window_heatmap.png")
```

- Y-axis: mode ID (0 at bottom), X-axis: frequency
- Red = propagating (positive), Blue = evanescent (negative), White = zero
- Gray dashed line marks the cutoff boundary

**Filtering modes:**

```python
# Show first 200 modes only (auto-sampled y-axis ticks)
fig, _ = pwmma.plot_section_energy_heatmap(section, max_modes=200)

# Explicit boolean mask (all ticks shown)
dominant = section.dominant_mode_ids(threshold=0.02)
mask = np.zeros(len(section.mode_ids), dtype=bool)
mask[dominant] = True
fig, _ = pwmma.plot_section_energy_heatmap(section, mode_mask=mask)

# Filter by mode family (e.g., all modes with first index = 1)
info = dsk.mode_info_array()
mask_m1 = info[:, 2] == 1   # column 2 = first mode index
fig, _ = pwmma.plot_section_energy_heatmap(section, mode_mask=mask_m1)
```

## Step 7: Save and Load

```python
# Persist the full chain result
result.save_npz("ka_window_result.npz")

# Persist a single section
section.save_npz("ka_window_section.npz")

# Reload later (no WG object needed)
section = pwmma.SectionEnergyCoupling.load_npz("ka_window_section.npz")
result = pwmma.ChainEnergyCouplingResult.load_npz("ka_window_result.npz")
```

Loaded results support all plotting and query methods — the waveguide objects
are not serialized, but the numerical data is self-contained.

## Mode Filtering Reference

### By Power Contribution

```python
dominant_ids = section.dominant_mode_ids(threshold=0.02)
```

Returns an `(n_dominant,)` array of mode IDs whose peak absolute power at
good-transmission frequencies (reflection < 0.1) exceeds `threshold`.

### By Mode Type

```python
info_mat = wg.mode_info_array()
# Columns: [kc, type(1=TE / 0=TM), modenum1, modenum2, fc, …]
```

Use column 2 (`modenum1`) and column 3 (`modenum2`) to select specific mode
families.  For TE modes, column 1 equals 1; for TM modes, column 1 equals 0.
