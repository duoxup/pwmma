"""Pure Plotly figure builders for the pwmma GUI. No Dash imports."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..spar_model import fit_spar_model

_METAL = "#b3c7e6"
_DIELECTRIC = "#e6c9a8"
_OUTLINE = "#1d5a9e"        # structure outline (accent steel blue)
_OUTLINE_MIRROR = "#8a93a3"  # mirrored half of a symmetric chain

# Colors here (and the inline hexes in _theme) mirror the :root tokens
# in assets/style.css — keep the two in sync when changing the palette.
_COLORWAY = ["#1d5a9e", "#c0392b", "#27824f", "#b8860b", "#6d4fa3", "#16828c"]
_GRID = "#eef0f4"
_AXIS = "#c8ccd4"


def _theme(fig: go.Figure, *, axes: bool = True) -> go.Figure:
    """Shared EDA look: white plot area, light grid, palette colorway."""
    fig.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Segoe UI, system-ui, sans-serif", size=12, color="#41506b"),
        colorway=_COLORWAY,
    )
    if axes:
        fig.update_xaxes(gridcolor=_GRID, zerolinecolor="#d4d8df",
                         linecolor=_AXIS, mirror=True, ticks="outside", tickcolor=_AXIS)
        fig.update_yaxes(gridcolor=_GRID, zerolinecolor="#d4d8df",
                         linecolor=_AXIS, mirror=True, ticks="outside", tickcolor=_AXIS)
    return fig


def _aperture_mm(row: dict) -> float:
    if str(row.get("kind", "")).lower() == "rec":
        return float(row.get("a", 0) or 0)
    return 2.0 * float(row.get("r", 0) or 0)


def _expand(rows: list[dict], sym: bool) -> list[tuple[dict, bool]]:
    """Return (row, is_mirror) for the physical chain."""
    base = [(r, False) for r in rows]
    if sym and len(rows) >= 2:
        base += [(r, True) for r in reversed(rows[:-1])]
    return base


def structure_preview_figure(rows: list[dict], sym: bool) -> go.Figure:
    """Longitudinal schematic: width proportional to length l, height to aperture."""
    fig = go.Figure()
    segments = _expand(list(rows), sym)
    x = 0.0
    max_ap = 0.0
    for row, is_mirror in segments:
        try:
            length = max(float(row.get("l", 0) or 0), 1e-9)
            ap = _aperture_mm(row)
        except (TypeError, ValueError):
            continue
        max_ap = max(max_ap, ap)
        is_dielectric = abs(complex(str(row.get("er", "1") or "1"))) > 1.0001
        fill = _DIELECTRIC if is_dielectric else _METAL
        fig.add_shape(
            type="rect",
            x0=x,
            x1=x + length,
            y0=-ap / 2,
            y1=ap / 2,
            line=dict(color=_OUTLINE_MIRROR if is_mirror else _OUTLINE, dash="dash" if is_mirror else "solid"),
            fillcolor=fill,
            opacity=0.5 if is_mirror else 0.9,
            layer="below",
        )
        x += length
    if segments and x > 0:
        # An invisible trace pins the data extent; combined with equal-aspect
        # scaling, Plotly autoranges to fit the whole structure in the frame
        # (scaled down if needed) while preserving true proportions.
        fig.add_trace(go.Scatter(
            x=[0, x], y=[-max_ap / 2 * 1.1, max_ap / 2 * 1.1],
            mode="markers", marker=dict(opacity=0), hoverinfo="skip", showlegend=False,
        ))
    fig.update_layout(
        height=216,
        margin=dict(l=4, r=4, t=4, b=4),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        showlegend=False,
    )
    return _theme(fig, axes=False)


def sparam_figure(spars: dict, *, out_modes=(0,), in_modes=(0,)) -> go.Figure:
    """|S11| (reflection) and |S21| (transmission) in dB, one pair of curves for
    every (output i, input j) in the Cartesian product of the selected modes.

    ``out_modes`` index the response (matrix row), ``in_modes`` the excitation
    (matrix column); default (0, 0) is the dominant-mode view. Indices out of
    range for a block (S11 rows = port-1 modes, S21 rows = port-2 modes) are
    skipped, so unequal port mode counts are handled gracefully.
    """
    freqs_ghz = np.asarray(spars["freqs"]) * 1e-9
    s11 = np.asarray(spars["s11"])
    s21 = np.asarray(spars["s21"])
    out_modes = [int(i) for i in (out_modes or (0,))]
    in_modes = [int(j) for j in (in_modes or (0,))]
    n_in = s11.shape[2]
    fig = go.Figure()
    for j in in_modes:
        if not 0 <= j < n_in:
            continue
        for i in out_modes:
            for label, mat in (("S11", s11), ("S21", s21)):
                if 0 <= i < mat.shape[1]:
                    with np.errstate(divide="ignore"):
                        db = 20.0 * np.log10(np.abs(mat[:, i, j]))
                    # |S| == 0 (pervasive in cir<->cir junctions by azimuthal
                    # symmetry) -> -inf, which breaks Plotly's y autoscale and
                    # blanks the panel; NaN is skipped instead of plotted.
                    db = np.where(np.isfinite(db), db, np.nan)
                    fig.add_trace(go.Scatter(x=freqs_ghz, y=db, mode="lines",
                                             name=f"{label}[{i},{j}]"))
    fig.update_layout(
        autosize=True,
        margin=dict(l=50, r=10, t=30, b=40),
        xaxis_title="Frequency (GHz)",
        yaxis_title="Magnitude (dB)",
        legend=dict(orientation="h"),
    )
    return _theme(fig)


def sparam_model_figure(model_payload: dict) -> go.Figure:
    """Adaptive-sweep view: dense rational-model curves + true sample points.

    The payload stores plain sample arrays (diskcache-small); the AAA model is
    refit here on render (milliseconds). Sample markers show where the solver
    actually ran — the honesty layer of an interpolated sweep.
    """
    fig = go.Figure()
    dense = np.linspace(model_payload["f0"], model_payload["f1"], 2001)
    for k, (key, label) in enumerate((("s11", "S11[0,0]"), ("s21", "S21[0,0]"))):
        curve = model_payload.get(key)
        if curve is None:
            continue
        F = np.asarray(curve["F"], dtype=float)
        y = np.asarray(curve["y"], dtype=complex)
        model = fit_spar_model(F, y)
        with np.errstate(divide="ignore"):
            db = 20.0 * np.log10(np.abs(model(dense)))
            db_s = 20.0 * np.log10(np.abs(y))
        db = np.where(np.isfinite(db), db, np.nan)      # -inf breaks autoscale
        db_s = np.where(np.isfinite(db_s), db_s, np.nan)
        color = _COLORWAY[k % len(_COLORWAY)]
        fig.add_trace(go.Scatter(x=dense * 1e-9, y=db, mode="lines", name=label,
                                 legendgroup=label, line=dict(color=color)))
        fig.add_trace(go.Scatter(x=F * 1e-9, y=db_s, mode="markers",
                                 name=f"{label} samples", legendgroup=label,
                                 showlegend=False,
                                 marker=dict(color=color, size=6,
                                             symbol="circle-open")))
    fig.update_layout(
        autosize=True,
        margin=dict(l=50, r=10, t=30, b=40),
        xaxis_title="Frequency (GHz)",
        yaxis_title="Magnitude (dB)",
        legend=dict(orientation="h"),
    )
    return _theme(fig)


def energy_line_figure(section, *, mode_threshold: float = 0.04, dB: bool = True,
                       mode_ids=None) -> go.Figure:
    """Per-mode power for the dominant modes, plus totals.

    Each mode draws its propagating band solid and its cutoff (evanescent) band
    dotted in the same color, so the contribution below cutoff stays visible.
    Note: cutoff contributions can be negative, which a log axis (dB on) cannot
    render — switch dB off to see them.

    ``mode_ids`` overrides the threshold-based dominant-mode selection (used by
    the GUI mode filters). The Σ totals always sum over all modes.
    """
    freqs_ghz = section.freqs * 1e-9
    fig = go.Figure()
    if mode_ids is None:
        dominant = section.dominant_mode_ids(threshold=mode_threshold)
    else:
        dominant = np.asarray(mode_ids, dtype=int)
    labels = section.get_mode_labels(mode_ids=dominant)
    for i, (mode_id, label) in enumerate(zip(dominant, labels)):
        color = _COLORWAY[i % len(_COLORWAY)]
        prop_mask = section.propagating_mask[:, mode_id]
        evan_mask = section.evanescent_mask[:, mode_id]
        if prop_mask.any():
            y = np.where(prop_mask, section.modal_power[:, mode_id], np.nan)
            fig.add_trace(go.Scatter(x=freqs_ghz, y=y, mode="lines", name=label,
                                     legendgroup=label, line=dict(color=color)))
        if evan_mask.any():
            y = np.where(evan_mask, section.modal_power[:, mode_id], np.nan)
            fig.add_trace(go.Scatter(x=freqs_ghz, y=y, mode="lines",
                                     name=f"{label} (evan.)", legendgroup=label,
                                     showlegend=not prop_mask.any(),
                                     line=dict(color=color, dash="dot")))
    fig.add_trace(go.Scatter(x=freqs_ghz, y=section.total_propagating_power,
                             mode="lines", name="Σ prop.", line=dict(dash="dash", color="#1a1d21")))
    if section.evanescent_mask.any():
        fig.add_trace(go.Scatter(x=freqs_ghz, y=section.total_evanescent_power,
                                 mode="lines", name="Σ evan.",
                                 line=dict(dash="dot", color="#1a1d21")))
    fig.update_layout(
        autosize=True, margin=dict(l=50, r=10, t=30, b=40),
        xaxis_title="Frequency (GHz)", yaxis_title="Net modal power",
        legend=dict(orientation="h"),
    )
    if dB:
        fig.update_yaxes(type="log")
    return _theme(fig)


def energy_heatmap_figure(section, *, mode_mask=None, max_modes: int | None = None) -> go.Figure:
    """Per-mode power vs frequency; positive=propagating, negative=evanescent.

    A dashed gray line marks the cutoff boundary (modes above it are evanescent;
    valid because mode ids are sorted by cutoff frequency). ``mode_mask`` is a
    boolean per-mode filter; when given, every shown mode gets its own y tick.
    """
    n_total = section.modal_power.shape[1]
    if mode_mask is not None:
        shown = np.where(np.asarray(mode_mask, dtype=bool))[0]
    elif max_modes is not None:
        shown = np.arange(min(max_modes, n_total))
    else:
        shown = np.arange(n_total)
    if len(shown) == 0:
        return empty_figure("no modes match the filter")

    freqs_ghz = section.freqs * 1e-9
    prop = section.propagating_mask[:, shown]
    evan = section.evanescent_mask[:, shown]
    masked = np.where(prop | evan, section.modal_power[:, shown], np.nan)
    vlim = float(np.nanmax(np.abs(masked))) if np.isfinite(masked).any() else 1.0
    labels = section.get_mode_labels(mode_ids=shown)
    fig = go.Figure(go.Heatmap(
        x=freqs_ghz, y=np.arange(len(shown)), z=masked.T,
        colorscale="RdBu", zmid=0, zmin=-vlim, zmax=vlim,
        colorbar=dict(title="Power"),
        customdata=np.repeat(np.asarray(labels)[:, None], len(freqs_ghz), axis=1),
        hovertemplate="%{x:.3f} GHz · %{customdata}<br>power %{z:.4g}<extra></extra>",
    ))

    # cutoff boundary: first evanescent row at each frequency, drawn between cells
    boundary = np.full(len(freqs_ghz), np.nan)
    for i in range(len(freqs_ghz)):
        evan_idx = np.where(evan[i])[0]
        if len(evan_idx) > 0:
            boundary[i] = evan_idx[0] - 0.5
    fig.add_trace(go.Scatter(x=freqs_ghz, y=boundary, mode="lines", name="cutoff",
                             line=dict(color="#5a6678", dash="dash"), hoverinfo="skip"))

    step = 1 if mode_mask is not None else max(1, len(shown) // 12)
    tickvals = np.arange(0, len(shown), step)
    fig.update_layout(
        autosize=True, margin=dict(l=50, r=10, t=30, b=40),
        xaxis_title="Frequency (GHz)", yaxis_title="Mode",
        yaxis=dict(tickvals=tickvals, ticktext=[labels[int(i)] for i in tickvals]),
        legend=dict(orientation="h"),
    )
    return _theme(fig)


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16, color="#aab0bb"))
    fig.update_layout(autosize=True, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return _theme(fig, axes=False)
