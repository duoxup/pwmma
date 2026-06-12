import numpy as np
import pytest
from waveguides import CirWG, RecWG

from pwmma.gui import adapter as A


def test_parse_waveguide_rec_converts_mm_to_si():
    wg = A.parse_waveguide({"kind": "rec", "a": 7.112, "b": 3.556,
                            "l": 2.0, "N": 24, "er": "1", "sigma": "5.8e7"})
    assert isinstance(wg, RecWG)
    assert wg.a == pytest.approx(7.112e-3)
    assert wg.b == pytest.approx(3.556e-3)
    assert wg.l == pytest.approx(2.0e-3)
    assert wg.N == 24


def test_parse_waveguide_cir_complex_er():
    wg = A.parse_waveguide({"kind": "cir", "r": 5.4, "l": 0.26,
                            "N": 96, "er": "9.2-0.5j", "sigma": "5.8e7"})
    assert isinstance(wg, CirWG)
    assert wg.r == pytest.approx(5.4e-3)
    assert wg.er == pytest.approx(9.2 - 0.5j)


def test_parse_chain_builds_sym_chain():
    rows = [
        {"kind": "rec", "a": 7.112, "b": 3.556, "l": 2.0, "N": 24, "er": "1", "sigma": "5.8e7"},
        {"kind": "cir", "r": 4.2, "l": 1.5, "N": 64, "er": "1", "sigma": "5.8e7"},
    ]
    chain = A.parse_chain(rows, sym=True)
    assert chain.sym is True
    assert chain.n_wgs == 2


def test_parse_freqs_ghz_to_hz():
    freqs = A.parse_freqs(28.0, 34.0, 4)
    np.testing.assert_allclose(freqs, np.array([28.0, 30.0, 32.0, 34.0]) * 1e9)


def test_parse_config_maps_fields():
    cfg = A.parse_config({"nproc": 8}, {"nproc": 8, "use_gpu": False, "precision": "complex64"})
    assert cfg.cmconf.nproc == 8
    assert cfg.smconf.use_gpu is False
    assert cfg.smconf.use_double_precision is False


def test_empty_chain_raises():
    with pytest.raises(A.GuiInputError, match="at least 2"):
        A.parse_chain([{"kind": "rec", "a": 7, "b": 3, "l": 2, "N": 24, "er": "1", "sigma": "5.8e7"}], sym=False)


def test_bad_geometry_raises():
    with pytest.raises(A.GuiInputError, match="positive"):
        A.parse_waveguide({"kind": "cir", "r": -1, "l": 1, "N": 8, "er": "1", "sigma": "5.8e7"})


def test_bad_er_raises():
    with pytest.raises(A.GuiInputError, match="er"):
        A.parse_waveguide({"kind": "cir", "r": 4.2, "l": 1, "N": 8, "er": "abc", "sigma": "5.8e7"})


def test_bad_freq_range_raises():
    with pytest.raises(A.GuiInputError, match="start"):
        A.parse_freqs(34.0, 28.0, 4)


def test_run_energy_returns_result_and_reports_progress():
    rows = [
        {"kind": "rec", "a": 7.112, "b": 3.556, "l": 2.0, "N": 24, "er": "1", "sigma": "5.8e7"},
        {"kind": "cir", "r": 4.2, "l": 1.5, "N": 64, "er": "1", "sigma": "5.8e7"},
        {"kind": "cir", "r": 5.4, "l": 0.26, "N": 96, "er": "9.2", "sigma": "5.8e7"},
    ]
    chain = A.parse_chain(rows, sym=True)
    freqs = A.parse_freqs(28.0, 34.0, 3)
    cfg = A.parse_config({"nproc": 2}, {"nproc": 2, "use_gpu": False, "precision": "complex64"})
    seen = []
    result = A.run_energy(chain, freqs, cfg, sections=[2], excitation_mode=0,
                          progress_callback=lambda d, t: seen.append((d, t)))
    assert result.section_indices == (2,)
    assert result.get_section(2).modal_power.shape == (3, 96)
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_run_spars_returns_arrays():
    rows = [
        {"kind": "rec", "a": 7.112, "b": 3.556, "l": 2.0, "N": 24, "er": "1", "sigma": "5.8e7"},
        {"kind": "cir", "r": 4.2, "l": 1.5, "N": 64, "er": "1", "sigma": "5.8e7"},
    ]
    chain = A.parse_chain(rows, sym=True)
    freqs = A.parse_freqs(28.0, 34.0, 3)
    cfg = A.parse_config({"nproc": 2}, {"nproc": 2, "use_gpu": False, "precision": "complex64"})
    out = A.run_spars(chain, freqs, cfg)
    assert set(out) == {"s11", "s12", "s21", "s22", "freqs"}
    assert out["s11"].shape[0] == 3


def test_parse_config_enables_disk_cache():
    cfg = A.parse_config(
        {"nproc": 4, "cache_dir": "/tmp/cm", "cache_enabled": True},
        {"nproc": 4, "use_gpu": False, "precision": "single"},
    )
    assert cfg.cmconf.cm_cache_dir == "/tmp/cm"
    assert cfg.cmconf.try_read_cm_from_cache is True
    assert cfg.cmconf.save_cm_to_cache is True


def test_parse_config_no_cache_by_default():
    cfg = A.parse_config({"nproc": 4}, {"nproc": 4, "use_gpu": False, "precision": "single"})
    assert cfg.cmconf.cm_cache_dir is None
    assert cfg.cmconf.try_read_cm_from_cache is False


def test_compute_cms_disk_cache_roundtrip(tmp_path):
    rows = [
        {"kind": "rec", "a": 7.112, "b": 3.556, "l": 2.0, "N": 24, "er": "1", "sigma": "5.8e7"},
        {"kind": "cir", "r": 4.2, "l": 1.5, "N": 64, "er": "1", "sigma": "5.8e7"},
    ]
    chain = A.parse_chain(rows, sym=False)
    cdir = tmp_path / "cm"
    cfg = A.parse_config(
        {"nproc": 2, "cache_dir": str(cdir), "cache_enabled": True},
        {"nproc": 2, "use_gpu": False, "precision": "single"},
    )
    cms1 = A.compute_cms(chain, cfg)        # computes and saves to disk
    assert cdir.exists() and any(cdir.iterdir())
    cms2 = A.compute_cms(chain, cfg)        # reads back from cache
    np.testing.assert_allclose(np.asarray(cms1[0]), np.asarray(cms2[0]))
