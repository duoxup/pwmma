# Modal Energy Coupling Analysis — Physical Picture

This document describes the physical model and mathematical formulation behind
the energy coupling analysis pipeline in `pwmma`.

## 1. Mode-Matching Method (MMM)

Consider a waveguide junction connecting two waveguide sections (possibly of
different cross-section).  The transverse electric and magnetic fields on each
side are expanded as a sum of eigenmodes:

$$E_t^{(i)} = \sum_m \left(a_m^{(i)+} + a_m^{(i)-}\right) e_m^{(i)}, \qquad H_t^{(i)} = \sum_m \left(a_m^{(i)+} - a_m^{(i)-}\right) h_m^{(i)}$$

where $a_m^{+}$ and $a_m^{-}$ are the forward- and backward-propagating mode
amplitudes, and $e_m$, $h_m$ are the normalised transverse mode fields.

Continuity of the tangential fields across the junction yields a linear system
relating the mode amplitudes on the two sides.  Solving this system gives the
**generalized scattering matrix** (GSM) of the junction:

$$\begin{bmatrix} a_1^{-} \\ a_2^{+} \end{bmatrix} = \begin{bmatrix} S_{11} & S_{12} \\ S_{21} & S_{22} \end{bmatrix} \begin{bmatrix} a_1^{+} \\ a_2^{-} \end{bmatrix}$$

## 2. Coupling Matrix

The scattering formulation begins with the **coupling matrix** $C_{ij}$:

$$C_{ij} = \int_{\text{aperture}} e_i^{(1)} \cdot e_j^{(2)} \, dS$$

This overlap integral quantifies how strongly mode $i$ on side 1 couples to
mode $j$ on side 2.  For circular-to-rectangular transitions and dielectric
disks (where the aperture is the full cross-section), this matrix captures
all inter-modal coupling.

From $C_{ij}$, the GSM is constructed via a linear solve that enforces the
power-normalised continuity conditions (see `numerics/gsm.py`).

## 3. Cascading via the Redheffer Star Product

For a chain of waveguide segments, individual GSMs are cascaded using the
Redheffer star product.  Between junctions, each waveguide segment is
represented by a diagonal propagation matrix $P$:

$$P_{mm} = e^{-\gamma_m L}$$

where $\gamma_m = \alpha_m + j\beta_m$ is the complex propagation constant
and $L$ is the segment length.  The sign convention uses $e^{+j\omega t}$,
so forward propagation multiplies by $e^{-j\beta L}$.

Cascading $S^A \star S^B$ with an intermediate propagation segment $P$ yields
the total GSM from the leftmost reference plane to the rightmost one.

## 4. Physical Meaning of the GSM Blocks

For a section excited from the left by mode $e$:

| Term | Physical Meaning |
|------|-----------------|
| $S_{11}[k,e]$ | Reflection coefficient: mode $e$ → mode $k$ on the left side |
| $S_{21}[k,e]$ | Transmission coefficient: mode $e$ → mode $k$ on the right side |

The squared magnitudes $|S_{11}[k,e]|^2$ and $|S_{21}[k,e]|^2$ represent
**power fractions** (under the normalisation used in `pwmma`, where modes
carry unit power when $|a|=1$).

## 5. Internal Field Reconstruction

To obtain the mode amplitudes at an *internal* cross-section (rather than at
the external ports), the chain is conceptually split into left and right
sub-chains at the section of interest, each represented by a cascaded GSM.

Let the left sub-chain (from the input reference plane to the section) have
GSM blocks $L_{11}, L_{12}, L_{21}, L_{22}$, and the right sub-chain (from the
section onward) have blocks $R_{11}, R_{12}, R_{21}, R_{22}$.  Denote the
section's own one-way propagation matrix as $P = \operatorname{diag}(e^{-\gamma_m l})$.

With excitation by mode $e$ from the left, the forward wave amplitude entering
the section from the left, $a_{\text{fwd\_left}}$, satisfies the linear system:

$$(I - L_{22} \tilde{R}_{11}) \, a_{\text{fwd\_left}} = L_{21}[:, e]$$

where $\tilde{R}_{11} = P \, R_{11} \, P$ folds the section's propagation into
the right-side reflection block.  Solving this system yields $a_{\text{fwd\_left}}$,
from which the remaining amplitudes follow directly:

$$
\begin{aligned}
a_{\text{fwd\_right}} &= P \, a_{\text{fwd\_left}} \\[2pt]
a_{\text{back\_right}} &= R_{11} \, a_{\text{fwd\_right}} \\[2pt]
a_{\text{back\_left}} &= P \, a_{\text{back\_right}}
\end{aligned}
$$

The matrix $(I - L_{22}\tilde{R}_{11})^{-1}$ captures the infinite series of
internal multiple reflections.  When $\|L_{22}\tilde{R}_{11}\| < 1$, the
Neumann expansion

$$(I - L_{22}\tilde{R}_{11})^{-1} = I + L_{22}\tilde{R}_{11} + (L_{22}\tilde{R}_{11})^2 + \cdots$$

converges, each term representing one additional round trip within the section.

This reconstruction is repeated at every frequency point and every internal
cross-section of interest, yielding the complete set of $a^{+}$ and $a^{-}$
needed for the per-mode power calculation in Section 6.

## 6. Modal Power Decomposition

With the amplitudes $a_{\text{fwd\_left}}$, $a_{\text{fwd\_right}}$, $a_{\text{back\_right}}$,
and $a_{\text{back\_left}}$ reconstructed in Section 5, the net power in each mode is
computed at the left reference plane of the segment.  The expression depends on
whether the mode is propagating or evanescent.

### 6.1 Propagating Modes

For modes above cutoff ($\beta$ real, $\alpha = 0$), a lossless segment preserves
the magnitude of each travelling wave: $|a_{\text{fwd\_right}}| = |a_{\text{fwd\_left}}|$.
The net real power flowing through mode $m$ is therefore

$$P_m = |a_{\text{fwd}, m}|^2 - |a_{\text{back\_right}, m}|^2$$

which reduces to the familiar $|a^{+}|^2 - |a^{-}|^2$ evaluated at either face.

### 6.2 Evanescent Modes

For modes below cutoff ($\gamma = \alpha$, purely real), the amplitudes decay
exponentially through the segment and the left- and right-face values differ.
The power is evaluated at the **left** reference plane.  Let

$$S_m = (a_{\text{fwd\_left}, m} + a_{\text{back\_left}, m})
        (a_{\text{fwd\_left}, m} - a_{\text{back\_left}, m})^{*}.$$

The mode impedance $Z_m$ is purely imaginary below cutoff, and its sign
distinguishes TE ($\operatorname{Im}(Z_m) > 0$) from TM
($\operatorname{Im}(Z_m) < 0$) modes.  The net real power contributed by
mode $m$ is

$$P_m = \operatorname{Im}\!\big(S_m \cdot \operatorname{sgn}(\operatorname{Im}(Z_m))\big).$$

This is equivalent to the full voltage--current derivation
$P_m = -\operatorname{Re}(V I^{*})$ with
$V = \sqrt{Z_m}(a_{\text{fwd\_left}} + a_{\text{back\_left}})$,
$I = (a_{\text{fwd\_left}} - a_{\text{back\_left}})/\sqrt{Z_m}$, but the
square-root impedance factors cancel algebraically, leaving only the sign
of $\operatorname{Im}(Z_m)$.

For a single evanescent mode in a lossless structure, the real part of $P_m$
vanishes — the mode carries no net real power across the section.  The imaginary
part, however, is non-zero and represents **reactive power** stored in the
near field, which oscillates between the junction and the surrounding space.
Although an individual evanescent mode transports no real power, its amplitude
is a direct indicator of mode conversion strength: energy coupled from the
excitation mode into an evanescent mode near a discontinuity may subsequently
couple into other propagating modes downstream or contribute to resonant
energy storage near the dielectric disk.

This behaviour is mathematically isomorphic to quantum-mechanical tunneling.
The waveguide dispersion $\beta = \sqrt{k_0^2 - k_c^2}$ becomes purely imaginary
below cutoff, just as the particle wave number
$k = \sqrt{2m(E - V)}/\hbar$ becomes imaginary inside a potential barrier.
In both cases, the field (or wavefunction) decays exponentially rather than
oscillating, carrying no net real flux yet coupling the two "allowed" regions
on either side.  For the pillbox window, the evanescent modes excited at the
dielectric disk surface play the role of the barrier modes: they mediate the
interaction between propagating modes across the junction, and their decay
length (analogous to the barrier width) determines the strength of that
coupling.

## 7. Power Balance

For a lossless junction excited by a single incident mode $e$, energy
conservation requires:

$$\underbrace{\sum_k |S_{11}[k,e]|^2}_{\text{total reflection}} \; + \; \underbrace{\sum_k |S_{21}[k,e]|^2}_{\text{total transmission}} = 1$$

In the `SectionEnergyCoupling` dataclass this is computed as:

$$P_{\text{balance}} = P_{\text{reflected}} + P_{\text{propagating}} + P_{\text{evanescent}}$$

where $P_{\text{reflected}}$ uses $\sum|S_{11}[:,e]|^2$ (all reflected modes,
not just the excitation mode) to correctly account for mode conversion at the
input:

| Field | Expression |
|-------|-----------|
| `reflection_power` | $|S_{11}[e,e]|^2$ |
| `total_reflected_power` | $\sum_k |S_{11}[k,e]|^2$ |
| `total_propagating_power` | $\sum_{m \in \text{prop}} P_m$ |
| `total_evanescent_power` | $\sum_{m \in \text{evan}} P_m$ |
| `power_balance` | $P_{\text{reflected}} + P_{\text{prop}} + P_{\text{evan}} \approx 1$ |

A deviation of `power_balance` from 1 indicates numerical truncation error
(insufficient modal expansion) or a violation of the lossless assumption.

**Note on cutoff excitation:** When the excitation mode itself is below
cutoff, power balance does not hold in the conventional sense because the
incident "power" in a cutoff mode is not well-defined in the
power-normalised S-parameter formulation.  This is physically expected.

## 8. Energy Coupling Analysis

The analysis pipeline implemented in `pwmma` goes beyond computing total
S-parameters.  For each frequency point and each mode, it computes:

1. **Forward/backward separation** — The forward and backward wave amplitudes
   $a^{+}$, $a^{-}$ are extracted from the internal GSM blocks for every
   mode at every cross-section.

2. **Per-mode net power** — $P_m$ is computed separately for propagating
   and evanescent modes using the correct physical expression for each case.

3. **Mode-by-mode accounting** — The contribution of every individual mode
   to the total reflected, transmitted, and stored power is quantified.
   This reveals *which* specific higher-order modes absorb energy from
   the excitation mode (mode conversion).

4. **Cross-section resolved** — By examining different physical sections
   along the chain, the spatial evolution of mode conversion can be traced.

## 9. Physical Interpretation of the Heatmap

In the heatmap visualisation:

- **Red (positive)** — The mode carries net power in the forward direction
  at that frequency.  Strong red in high-order modes indicates mode
  conversion.
- **Blue (negative)** — The mode carries net power in the backward
  direction.
- **White** — Zero net contribution.  The mode is either absent or fully
  decoupled.
- **Cutoff line** — Marks the boundary between propagating and evanescent
  regimes.  Its frequency dependence traces the dispersion of each mode.

Colour encodes only the direction of net power flow; whether a mode is
propagating or evanescent is read from the cutoff line, not from the
colour.

## 10. Application to Pillbox Windows

A pillbox window is a symmetric structure used to seal a vacuum envelope
while allowing RF power to pass through.  It typically consists of:

`rectangular WG → circular WG → dielectric disk → circular WG → rectangular WG`

The dielectric disk introduces strong reactive fields and can excite
higher-order modes at the disk surfaces.  The energy coupling analysis
quantifies:

- How much power is reflected into the excitation mode vs. converted to
  higher-order modes.
- How much energy is stored in evanescent modes near the disk (indicating
  resonant or near-resonant behaviour).
- Whether the modal expansion is converged (via power balance error).

