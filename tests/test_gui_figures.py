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
    # uniform sweeps draw the AAA-fitted dense curve + sample markers too
    s11 = next(t for t in fig.data if t.name == "S11[0,0]")
    assert len(s11.x) > len(_fake_spars()["freqs"])
    assert any(t.mode == "markers" for t in fig.data)


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
    # the nonzero curve's SAMPLES stay intact (the dense fit may legitimately
    # cross |S| = 0 between samples, which renders as NaN gaps, not -inf)
    s21_00 = next(t for t in fig.data if t.name == "S21[0,0] samples")
    assert np.all(np.isfinite(np.asarray(s21_00.y, dtype=float)))


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


def _fake_model_payload():
    """Adaptive-sweep payload: samples of a smooth rational curve."""
    F = np.linspace(600e9, 716e9, 12)
    y11 = 0.3 * np.exp(1j * (F - 600e9) / 40e9) + 0.1
    y21 = 0.9 - 0.2j * (F - 600e9) / 116e9
    return {"f0": 600e9, "f1": 716e9,
            "s11": {"F": F, "y": y11.astype(complex)},
            "s21": {"F": F, "y": y21.astype(complex)},
            "n_solves": 12, "confident": True}


def test_sparam_model_figure_has_curves_and_sample_markers():
    fig = F.sparam_model_figure(_fake_model_payload())
    assert isinstance(fig, go.Figure)
    lines = [t for t in fig.data if t.mode == "lines"]
    markers = [t for t in fig.data if t.mode == "markers"]
    assert {t.name for t in lines} == {"S11[0,0]", "S21[0,0]"}
    assert len(markers) == 2                     # sample overlay per curve
    assert len(lines[0].x) > 500                 # dense model evaluation
    for t in fig.data:
        y = np.asarray(t.y, dtype=float)
        assert not np.isinf(y).any()             # -inf guard inherited


def test_energy_line_figure_with_model_draws_dense_and_markers():
    sec = _small_section()
    dominant = sec.dominant_mode_ids(threshold=0.04)
    assert dominant.size >= 2
    smooth_id, raw_id = int(dominant[0]), int(dominant[1])
    n_dense = 50
    model = {
        "freqs": np.linspace(sec.freqs[0], sec.freqs[-1], n_dense),
        "mode_ids": np.array([smooth_id]),
        "modal_power": np.abs(np.random.default_rng(0).normal(0.5, 0.1, (n_dense, 1))),
        "propagating_mask": np.ones((n_dense, 1), dtype=bool),
        "evanescent_mask": np.zeros((n_dense, 1), dtype=bool),
        "unsmoothed_mode_ids": np.array([raw_id]),
    }
    fig = F.energy_line_figure(sec, mode_ids=[smooth_id, raw_id], model=model)
    by_name = {t.name: t for t in fig.data}
    labels = sec.get_mode_labels(mode_ids=[smooth_id, raw_id])
    dense_trace = by_name[labels[0]]
    assert len(dense_trace.x) == n_dense                     # rebuilt dense curve
    marker = by_name[f"{labels[0]} samples"]
    assert marker.mode == "markers" and len(marker.x) == len(sec.freqs)
    raw_trace = by_name[labels[1]]
    assert len(raw_trace.x) == len(sec.freqs)                # refused -> sampled
