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


def _small_section():
    rwg = RecWG(a=7.112e-3, b=3.556e-3, l=2e-3, N=24)
    cwg = CirWG(r=4.2e-3, l=1.5e-3, N=64)
    dsk = CirWG(r=5.4e-3, l=0.26e-3, N=96, er=9.2)
    chain = pwmma.Chain([rwg, cwg, dsk], sym=True)
    freqs = np.linspace(28e9, 34e9, 3)
    cfg = pwmma.Config(pwmma.CMConfig(nproc=2), pwmma.SMConfig(nproc=2, use_gpu=False))
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
