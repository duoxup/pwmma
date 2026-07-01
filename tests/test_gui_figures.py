import numpy as np
import plotly.graph_objects as go
from waveguides import CirWG, RecWG

import pwmma
from pwmma.gui import figures as F

ROWS = [
    {"kind": "rec", "a": 7.112, "b": 3.556, "l": 2.0, "N": 24, "er": "1", "sigma": "5.8e7"},
    {"kind": "cir", "r": 4.2, "l": 1.5, "N": 64, "er": "1", "sigma": "5.8e7"},
    {"kind": "cir", "r": 5.4, "l": 0.26, "N": 96, "er": "9.2", "sigma": "5.8e7"},
]


def test_structure_preview_expands_sym_chain():
    fig = F.structure_preview_figure(ROWS, sym=True)
    assert isinstance(fig, go.Figure)
    # sym expands [rec, cir, disk] -> 5 physical segments -> 5 drawn bars (shapes)
    assert len(fig.layout.shapes) == 5


def test_structure_preview_no_sym():
    fig = F.structure_preview_figure(ROWS, sym=False)
    assert len(fig.layout.shapes) == 3


def test_structure_preview_empty_is_safe():
    fig = F.structure_preview_figure([], sym=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.shapes) == 0


def _fake_spars(n=3, modes=4):
    rng = np.random.default_rng(0)

    def m():
        return (rng.random((n, modes, modes)) + 1j * rng.random((n, modes, modes))) * 0.1

    return {"freqs": np.linspace(28e9, 34e9, n), "s11": m(), "s12": m(), "s21": m(), "s22": m()}


def test_sparam_figure_has_traces_in_db():
    fig = F.sparam_figure(_fake_spars())  # default (0,0)
    assert isinstance(fig, go.Figure)
    names = {t.name for t in fig.data}
    assert "S11[0,0]" in names and "S21[0,0]" in names
    assert "dB" in (fig.layout.yaxis.title.text or "")


def test_sparam_figure_zero_sparam_has_no_minus_inf():
    """|S| == 0 (pervasive in cir<->cir junctions by azimuthal symmetry) gives
    20*log10(0) == -inf, which breaks Plotly's y autoscale and blanks the panel.
    Such points must be dropped (NaN), never -inf; finite traces stay intact.
    """
    spars = _fake_spars()
    spars["s11"][:, 0, 0] = 0.0          # exact-zero dominant-mode reflection
    spars["s21"][:, 1, 0] = 0.0          # exact-zero off-diagonal transmission
    fig = F.sparam_figure(spars, out_modes=[0, 1], in_modes=[0])
    for t in fig.data:
        y = np.asarray(t.y, dtype=float)
        assert not np.isinf(y).any()     # no -inf anywhere
    s21_00 = next(t for t in fig.data if t.name == "S21[0,0]")
    assert np.all(np.isfinite(np.asarray(s21_00.y, dtype=float)))  # nonzero trace intact


def _small_section():
    rwg = RecWG(a=7.112e-3, b=3.556e-3, l=2e-3, N=24)
    cwg = CirWG(r=4.2e-3, l=1.5e-3, N=64)
    dsk = CirWG(r=5.4e-3, l=0.26e-3, N=96, er=9.2)
    chain = pwmma.Chain([rwg, cwg, dsk], sym=True)
    freqs = np.linspace(28e9, 34e9, 3)
    cfg = pwmma.Config(nproc=2, use_gpu=False)
    return pwmma.analyze_energy_coupling(chain, freqs, cfg, sections=[2],
                                         show_progress=False).get_section(2)


def test_energy_line_figure_builds():
    fig = F.energy_line_figure(_small_section(), mode_threshold=0.04, dB=True)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_energy_heatmap_figure_builds():
    fig = F.energy_heatmap_figure(_small_section(), max_modes=96)
    assert isinstance(fig, go.Figure)
    assert any(isinstance(t, go.Heatmap) for t in fig.data)
