import plotly.graph_objects as go

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
