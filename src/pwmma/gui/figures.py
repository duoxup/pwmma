"""Pure Plotly figure builders for the pwmma GUI. No Dash imports."""
from __future__ import annotations

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
