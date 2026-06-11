"""Pure Plotly figure builders for the pwmma GUI. No Dash imports."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

_METAL = "#b3c7e6"
_DIELECTRIC = "#e6c9a8"


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
    for row, is_mirror in segments:
        try:
            length = max(float(row.get("l", 0) or 0), 1e-9)
            ap = _aperture_mm(row)
        except (TypeError, ValueError):
            continue
        is_dielectric = abs(complex(str(row.get("er", "1") or "1"))) > 1.0001
        fill = _DIELECTRIC if is_dielectric else _METAL
        fig.add_shape(
            type="rect",
            x0=x,
            x1=x + length,
            y0=-ap / 2,
            y1=ap / 2,
            line=dict(color="#888", dash="dash" if is_mirror else "solid"),
            fillcolor=fill,
            opacity=0.5 if is_mirror else 0.9,
            layer="below",
        )
        x += length
    fig.update_layout(
        height=120,
        margin=dict(l=4, r=4, t=4, b=4),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        showlegend=False,
        plot_bgcolor="white",
    )
    if segments:
        fig.update_xaxes(range=[-0.02 * x, 1.02 * x])
    return fig


def sparam_figure(spars: dict, *, excitation_mode: int = 0) -> go.Figure:
    """Plot |S11| and |S21| of the excitation mode (reflection/transmission) in dB."""
    freqs_ghz = np.asarray(spars["freqs"]) * 1e-9
    e = excitation_mode
    fig = go.Figure()
    for name, key in (("S11", "s11"), ("S21", "s21")):
        mat = np.asarray(spars[key])
        with np.errstate(divide="ignore"):
            db = 20.0 * np.log10(np.abs(mat[:, e, e]))
        fig.add_trace(go.Scatter(x=freqs_ghz, y=db, mode="lines", name=name))
    fig.update_layout(
        height=420,
        margin=dict(l=50, r=10, t=30, b=40),
        xaxis_title="Frequency (GHz)",
        yaxis_title="Magnitude (dB)",
        legend=dict(orientation="h"),
    )
    return fig


def energy_line_figure(section, *, mode_threshold: float = 0.04, dB: bool = True) -> go.Figure:
    """Per-mode propagating power for the dominant modes, plus totals."""
    freqs_ghz = section.freqs * 1e-9
    fig = go.Figure()
    dominant = section.dominant_mode_ids(threshold=mode_threshold)
    labels = section.get_mode_labels(mode_ids=dominant)
    for mode_id, label in zip(dominant, labels):
        y = np.where(section.propagating_mask[:, mode_id], section.modal_power[:, mode_id], np.nan)
        fig.add_trace(go.Scatter(x=freqs_ghz, y=y, mode="lines", name=label))
    fig.add_trace(go.Scatter(x=freqs_ghz, y=section.total_propagating_power,
                             mode="lines", name="Σ prop.", line=dict(dash="dash", color="black")))
    fig.update_layout(
        height=420, margin=dict(l=50, r=10, t=30, b=40),
        xaxis_title="Frequency (GHz)", yaxis_title="Net modal power",
        legend=dict(orientation="h"),
    )
    if dB:
        fig.update_yaxes(type="log")
    return fig


def energy_heatmap_figure(section, *, max_modes: int | None = None) -> go.Figure:
    """Per-mode power vs frequency; positive=propagating, negative=evanescent."""
    n = section.modal_power.shape[1] if max_modes is None else min(max_modes, section.modal_power.shape[1])
    data = section.modal_power[:, :n]
    masked = np.where(section.propagating_mask[:, :n] | section.evanescent_mask[:, :n], data, np.nan)
    vlim = float(np.nanmax(np.abs(masked))) if np.isfinite(masked).any() else 1.0
    fig = go.Figure(go.Heatmap(
        x=section.freqs * 1e-9, y=np.arange(n), z=masked.T,
        colorscale="RdBu", zmid=0, zmin=-vlim, zmax=vlim,
        colorbar=dict(title="Power"),
    ))
    fig.update_layout(
        height=420, margin=dict(l=50, r=10, t=30, b=40),
        xaxis_title="Frequency (GHz)", yaxis_title="Mode index",
    )
    return fig


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16, color="#999"))
    fig.update_layout(height=420, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig
