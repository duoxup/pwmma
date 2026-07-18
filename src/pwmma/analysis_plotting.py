#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting helpers for energy-coupling analysis results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from waveguides import WG

from .analysis import ChainEnergyCouplingResult, SectionEnergyCoupling


def _select_mode_ids(section: SectionEnergyCoupling, mode_threshold: float) -> np.ndarray:
    selected = section.dominant_mode_ids(threshold=mode_threshold)
    if selected.size == 0:
        return section.mode_ids[: min(10, len(section.mode_ids))]
    return selected


def _iter_mode_styles(n: int):
    """Yield ``(color, linestyle)`` pairs for ``n`` distinct mode lines."""
    cmap = plt.get_cmap("tab20")
    linestyles = ["-", "--", "-.", ":"]
    for i in range(n):
        yield cmap(i % 20), linestyles[(i // 20) % len(linestyles)]


def _rank_mode_ids(
    modal_power: np.ndarray,
    reflection_power: np.ndarray,
    threshold: float | None,
) -> np.ndarray:
    """Return mode indices sorted by descending peak |power| at good-transmission frequencies.

    When *threshold* is a float, only frequency points where
    ``reflection_power < threshold`` (e.g. 0.1 ≈ −10 dB return loss) are used
    to rank the modes.  When *threshold* is ``None`` all frequencies are used.
    """
    peak = np.max(np.abs(modal_power), axis=0)
    if threshold is not None:
        good = reflection_power < threshold
        if np.any(good):
            peak = np.max(np.abs(modal_power[good]), axis=0)
    return np.argsort(peak)[::-1]  # descending


def plot_section_energy_coupling(
    section: SectionEnergyCoupling,
    *,
    wg: WG | None = None,
    mode_threshold: float = 0.04,
    mode_ids: Sequence[int] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 12.0),
    max_plotted_modes: int = 24,
    good_transmission_threshold: float = 0.1,
    max_legend_entries: int | None = None,
    legend_ncol: int | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot the modal energy-coupling detail for one analyzed section.

    The figure contains four stacked panels:
    1. propagating-mode contributions (top ``max_plotted_modes`` individually),
    2. evanescent-mode contributions,
    3. total reflected power, total propagating power, total evanescent power,
       and their total,
    4. excitation-mode reflection and total propagating power in dB.

    Modes are ranked by peak absolute contribution at frequencies where the
    excitation-mode reflection is below *good_transmission_threshold* (default
    −10 dB).  Excess modes are aggregated into a single "others" trace.

    Parameters
    ----------
    section:
        Section result to visualize.
    wg:
        Optional waveguide object used to reconstruct mode labels. If omitted,
        ``section.wg`` is used.
    mode_threshold:
        Minimum peak absolute contribution for automatic mode selection when
        ``mode_ids`` is not given.
    mode_ids:
        Explicit mode ids to plot. If provided, ``mode_threshold`` is ignored.
    title:
        Optional figure title. Defaults to a title derived from ``section_index``.
    figsize:
        Matplotlib figure size.
    max_plotted_modes:
        Maximum number of individually styled mode lines.  Remaining modes are
        summed into a single "others" trace per panel.
    good_transmission_threshold:
        Reflection threshold for ranking frequencies (see notes in docstring).
        Set to ``None`` to rank over all frequencies.
    max_legend_entries:
        Maximum legend entries.  ``None`` = auto (``max_plotted_modes``
        displayed plus one for "others" if applicable).
    legend_ncol:
        Number of columns in the mode legend.  ``None`` = auto (up to 12 rows
        per column).

    Returns
    -------
    (Figure, ndarray)
        Matplotlib figure and axes array.
    """
    # --- mode selection ---
    if mode_ids is None:
        mode_ids_arr = _select_mode_ids(section, mode_threshold)
    else:
        mode_ids_arr = np.asarray(mode_ids, dtype=int)

    if len(mode_ids_arr) == 0:
        raise ValueError("No mode ids selected — nothing to plot.")

    freqs_ghz = section.freqs * 1e-9
    modal_power = section.modal_power[:, mode_ids_arr]
    prop_masks = section.propagating_mask[:, mode_ids_arr]
    evan_masks = section.evanescent_mask[:, mode_ids_arr]

    # --- rank & cap ---
    ranking = _rank_mode_ids(modal_power, section.reflection_power,
                             good_transmission_threshold)
    top_n = min(len(ranking), max_plotted_modes)
    top_idx = ranking[:top_n]
    rest_idx = ranking[top_n:]

    fig, axes = plt.subplots(nrows=4, sharex=True, figsize=figsize)

    # --- plot top-N modes with distinct (color, linestyle) pairs ---
    styles = list(_iter_mode_styles(top_n))
    lines1, lines2 = [], []
    for pos, mode_pos in enumerate(top_idx):
        color, ls = styles[pos]
        l1 = axes[0].plot(
            freqs_ghz,
            np.ma.MaskedArray(modal_power[:, mode_pos], ~prop_masks[:, mode_pos]),
            color=color, linestyle=ls, linewidth=1.2,
        )[0]
        l2 = axes[1].plot(
            freqs_ghz,
            np.ma.MaskedArray(modal_power[:, mode_pos], ~evan_masks[:, mode_pos]),
            color=color, linestyle=ls, linewidth=1.2,
        )[0]
        lines1.append(l1)
        lines2.append(l2)

    # --- "others" aggregation ---
    if len(rest_idx) > 0:
        others_prop = np.sum(
            np.where(prop_masks[:, rest_idx], modal_power[:, rest_idx], 0.0), axis=1
        )
        others_evan = np.sum(
            np.where(evan_masks[:, rest_idx], modal_power[:, rest_idx], 0.0), axis=1
        )
        axes[0].plot(freqs_ghz, others_prop, color="gray", linewidth=0.6)
        axes[1].plot(freqs_ghz, others_evan, color="gray", linewidth=0.6)

    # --- summary panel ---
    axes[2].plot(freqs_ghz, section.total_reflected_power)
    axes[2].plot(freqs_ghz, section.total_propagating_power)
    axes[2].plot(freqs_ghz, section.total_evanescent_power)
    axes[2].plot(freqs_ghz, section.total_power, "b--")

    # --- legend ---
    mode_labels = section.get_mode_labels(wg=wg, mode_ids=mode_ids_arr)
    handles = [lines1[pos] for pos in range(top_n)]
    labels = [mode_labels[int(top_idx[pos])] for pos in range(top_n)]
    if len(rest_idx) > 0:
        handles.append(axes[0].lines[-1])  # the "others" line just plotted
        labels.append("others")
    if max_legend_entries is None:
        max_legend_entries = len(handles)
    n_legend = min(len(handles), max_legend_entries)
    if legend_ncol is None:
        legend_ncol = max(1, int(np.ceil(n_legend / 12)))
    axes[0].legend(handles=handles[:n_legend], labels=labels[:n_legend], ncol=legend_ncol)
    axes[1].legend(handles=handles[:n_legend], labels=labels[:n_legend], ncol=legend_ncol)
    axes[2].legend(["reflection", "prop. modes", "evan. modes", "sum."])

    axes[0].set_ylabel("Prop. modes")
    axes[1].set_ylabel("Evan. modes")
    axes[2].set_ylabel("Power")
    axes[2].set_xlabel("Frequency (GHz)")
    axes[0].set_title(title or f"Section {section.section_index} Modal Energy Coupling")

    with np.errstate(divide="ignore", invalid="ignore"):
        refl_db = 10.0 * np.log10(section.reflection_power)
        trans_db = 10.0 * np.log10(section.total_propagating_power)
    refl_db = np.ma.masked_invalid(refl_db)
    trans_db = np.ma.masked_invalid(trans_db)
    axes[3].plot(freqs_ghz, refl_db, label="|S11[e,e]|²")
    axes[3].plot(freqs_ghz, trans_db, label="|S21[e,e]|²")
    axes[3].set_ylabel("dB")
    axes[3].set_xlabel("Frequency (GHz)")
    axes[3].legend()

    fig.tight_layout()
    return fig, axes


def plot_section_energy_heatmap(
    section: SectionEnergyCoupling,
    *,
    mode_mask: np.ndarray | None = None,
    max_modes: int | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Heatmap of per-mode energy coupling vs frequency for one analyzed section.

    A single colour-mapped grid with frequency on the x-axis and mode index
    on the y-axis (mode 0 at bottom, higher indices above).  Signed net modal
    power on a seismic scale centred at white (zero): red = net forward
    power, blue = net backward power.  Colour encodes only the direction of
    net power flow; whether a mode is propagating or evanescent is read from
    the gray dashed cutoff boundary line, not from the colour.

    Parameters
    ----------
    section:
        Section result to visualize.
    mode_mask:
        Optional boolean mask of shape ``(n_modes,)`` selecting which modes
        to display.  When given, every selected mode ID is shown on the
        y-axis (no tick sampling).  Modes are always in index order.
    max_modes:
        Show the first *max_modes* modes (by index).  Y-axis ticks are
        auto-sampled to ~10 labels.  Only used when *mode_mask* is ``None``.
    title:
        Optional figure title.
    figsize:
        Matplotlib figure size.

    Returns
    -------
    (Figure, Axes)
        Matplotlib figure and the single axes.
    """
    from matplotlib.colors import TwoSlopeNorm

    all_ids = section.mode_ids

    # --- mode selection ---
    if mode_mask is not None:
        mode_mask = np.asarray(mode_mask, dtype=bool)
        if mode_mask.shape != all_ids.shape:
            raise ValueError(
                f"mode_mask shape {mode_mask.shape} does not match "
                f"mode_ids shape {all_ids.shape}"
            )
        shown = np.where(mode_mask)[0]
        auto_ticks = False
    elif max_modes is not None:
        shown = np.arange(min(max_modes, len(all_ids)))
        auto_ticks = True
    else:
        shown = np.arange(len(all_ids))
        auto_ticks = True

    if len(shown) == 0:
        raise ValueError("No modes selected — nothing to plot.")

    n_shown = len(shown)
    freqs_ghz = section.freqs * 1e-9
    modal_power = section.modal_power[:, shown]
    prop_mask = section.propagating_mask[:, shown]
    evan_mask = section.evanescent_mask[:, shown]
    shown_ids = all_ids[shown]

    # --- signed net modal power; sign = transport direction ---
    data = np.where(prop_mask, modal_power, np.nan)
    data = np.where(evan_mask, modal_power, data)

    # --- symmetric colour scale, zero = white ---
    finite = data[np.isfinite(data)]
    vlim = np.max(np.abs(finite)) if len(finite) > 0 else 1.0
    norm = TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim)

    # --- y-axis ticks ---
    if auto_ticks:
        tick_step = max(1, n_shown // 10)
        tick_pos = np.arange(0, n_shown, tick_step)
    else:
        tick_pos = np.arange(n_shown)
    tick_lab = [str(shown_ids[int(p)]) for p in tick_pos]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(
        freqs_ghz, np.arange(n_shown), data.T,
        cmap="seismic", norm=norm, shading="nearest",
    )

    # --- cutoff boundary line ---
    boundary = np.full(len(freqs_ghz), np.nan)
    for i in range(len(freqs_ghz)):
        evan_idx = np.where(evan_mask[i, :])[0]
        if len(evan_idx) > 0:
            boundary[i] = evan_idx[0] - 0.5
    ax.plot(freqs_ghz, boundary, color="gray", linestyle="--", linewidth=1.5, label="cutoff")

    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lab, fontsize=7)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Mode")
    ax.set_title(title or f"Section {section.section_index} Modal Energy Coupling")
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label="Power", shrink=0.85)

    fig.tight_layout()
    return fig, ax


def plot_chain_energy_overview(
    result: ChainEnergyCouplingResult,
    *,
    sections: Sequence[int] | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot a compact multi-section overview for an analysis result.

    The three panels show:
    1. total propagating contribution per section,
    2. total evanescent contribution per section,
    3. absolute power-balance error per section.

    Parameters
    ----------
    result:
        Chain analysis result to summarize.
    sections:
        Optional subset of physical section indices to include. Defaults to all
        sections stored in ``result``.
    figsize:
        Matplotlib figure size.

    Returns
    -------
    (Figure, ndarray)
        Matplotlib figure and axes array.
    """
    section_indices = result.section_indices if sections is None else tuple(sections)
    freqs_ghz = result.freqs * 1e-9
    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=figsize)

    for section_idx in section_indices:
        section = result.sections[section_idx]
        label = f"section {section_idx}"
        axes[0].plot(freqs_ghz, section.total_propagating_power, label=label)
        axes[1].plot(freqs_ghz, section.total_evanescent_power, label=label)
        axes[2].plot(freqs_ghz, np.abs(section.power_balance - 1.0), label=label)

    axes[0].set_ylabel("Prop. power")
    axes[1].set_ylabel("Evan. power")
    axes[2].set_ylabel("|sum - 1|")
    axes[2].set_xlabel("Frequency (GHz)")
    axes[0].set_title("Energy Coupling Overview")
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    fig.tight_layout()
    return fig, axes


def save_figure(fig: plt.Figure, path: str | Path, *, dpi: int = 160) -> None:
    """Save a matplotlib figure and create parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
