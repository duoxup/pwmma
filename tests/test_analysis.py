#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from waveguides import CirWG, RecWG

import pwmma


matplotlib.use("Agg")


@pytest.fixture(scope="module")
def ka_window_analysis() -> pwmma.ChainEnergyCouplingResult:
    # Ka-band window geometry taken from the legacy analysis script, with
    # reduced modal truncation and a short frequency sweep to keep tests light.
    rwg = RecWG(a=7.112e-3, b=3.556e-3, l=2e-3, N=24)
    cwg = CirWG(r=4.2e-3, l=1.5e-3, N=64)
    dsk = CirWG(r=5.4e-3, l=0.26e-3, N=96, er=9.2)
    chain = pwmma.Chain([rwg, cwg, dsk], sym=True)
    freqs = np.linspace(28.0, 34.0, 3) * 1e9
    config = pwmma.Config(
        nproc=2,
        use_gpu=False,
        use_double_precision=False,
        try_read_cm_from_cache=False,
        save_cm_to_cache=False,
    )
    return pwmma.analyze_energy_coupling(
        chain,
        freqs,
        config,
        sections=[1, 2, 3],
        excitation_mode=0,
        show_progress=False,
    )


def test_analyze_energy_coupling_shapes_and_metadata(ka_window_analysis) -> None:
    result = ka_window_analysis

    assert result.section_indices == (1, 2, 3)
    assert result.indexing == "physical"
    assert result.excitation_mode == 0
    assert result.freqs.shape == (3,)

    left = result.get_section(1)
    window = result.get_section(2)
    right = result.get_section(3)

    assert left.modal_power.shape == (3, 64)
    assert window.modal_power.shape == (3, 96)
    assert right.modal_power.shape == (3, 64)

    assert np.array_equal(left.mode_ids, np.arange(64))
    assert np.array_equal(window.mode_ids, np.arange(96))

    assert left.propagating_mask.shape == left.modal_power.shape
    assert window.evanescent_mask.shape == window.modal_power.shape
    assert left.forward_left.shape == left.modal_power.shape
    assert window.backward_right.shape == window.modal_power.shape

    # total_reflected_power should equal or exceed excitation-mode reflection
    assert left.total_reflected_power.shape == (3,)
    assert window.total_reflected_power.shape == (3,)
    assert np.all(left.total_reflected_power >= left.reflection_power)
    assert np.all(window.total_reflected_power >= window.reflection_power)


def test_power_balance_and_symmetry(ka_window_analysis) -> None:
    result = ka_window_analysis
    left = result.get_section(1)
    right = result.get_section(3)
    window = result.get_section(2)

    assert left.max_power_balance_error() < 1e-3
    assert window.max_power_balance_error() < 1e-3
    assert result.max_power_balance_error() < 1e-3

    np.testing.assert_allclose(left.modal_power, right.modal_power, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        left.reflection_power,
        window.reflection_power,
        rtol=1e-7,
        atol=1e-7,
    )


def test_section_io_roundtrip(tmp_path: Path, ka_window_analysis) -> None:
    section = ka_window_analysis.get_section(2)
    output = tmp_path / "section_2.npz"

    section.save_npz(output)
    loaded = pwmma.SectionEnergyCoupling.load_npz(output)

    assert loaded.section_index == section.section_index
    assert loaded.excitation_mode == section.excitation_mode
    np.testing.assert_array_equal(loaded.mode_ids, section.mode_ids)
    np.testing.assert_allclose(loaded.modal_power, section.modal_power)
    np.testing.assert_allclose(loaded.total_reflected_power, section.total_reflected_power)
    np.testing.assert_allclose(loaded.power_balance, section.power_balance)


def test_chain_io_roundtrip(tmp_path: Path, ka_window_analysis) -> None:
    output = tmp_path / "chain_analysis.npz"

    ka_window_analysis.save_npz(output)
    loaded = pwmma.ChainEnergyCouplingResult.load_npz(output)

    assert loaded.section_indices == ka_window_analysis.section_indices
    assert loaded.indexing == "physical"
    np.testing.assert_allclose(loaded.freqs, ka_window_analysis.freqs)
    np.testing.assert_allclose(
        loaded.get_section(1).modal_power,
        ka_window_analysis.get_section(1).modal_power,
    )
    np.testing.assert_allclose(
        loaded.get_section(2).reflection_power,
        ka_window_analysis.get_section(2).reflection_power,
    )


def test_plotting_helpers(tmp_path: Path, ka_window_analysis) -> None:
    section = ka_window_analysis.get_section(2)

    fig1, axes1 = pwmma.plot_section_energy_coupling(section, mode_threshold=0.03)
    fig2, axes2 = pwmma.plot_chain_energy_overview(ka_window_analysis)

    out1 = tmp_path / "section_plot.png"
    out2 = tmp_path / "overview_plot.png"
    pwmma.save_figure(fig1, out1)
    pwmma.save_figure(fig2, out2)

    assert len(axes1) == 4
    assert len(axes2) == 3
    assert out1.exists()
    assert out2.exists()


def test_excitation_mode_nonzero(tmp_path: Path) -> None:
    """Analysis with a cut-off excitation_mode still produces correct shapes and data."""
    rwg = RecWG(a=7.112e-3, b=3.556e-3, l=2e-3, N=24)
    cwg = CirWG(r=4.2e-3, l=1.5e-3, N=64)
    dsk = CirWG(r=5.4e-3, l=0.26e-3, N=96, er=9.2)
    chain = pwmma.Chain([rwg, cwg, dsk], sym=True)
    freqs = np.linspace(28.0, 34.0, 3) * 1e9
    config = pwmma.Config(
        nproc=2, use_gpu=False,
    )
    result = pwmma.analyze_energy_coupling(
        chain, freqs, config, sections=[1], excitation_mode=1, show_progress=False,
    )
    section = result.get_section(1)
    assert section.modal_power.shape == (3, 64)
    assert np.all(section.total_reflected_power >= section.reflection_power)

    # Round-trip with non-zero excitation mode
    out = tmp_path / "exc_mode_1.npz"
    section.save_npz(out)
    loaded = pwmma.SectionEnergyCoupling.load_npz(out)
    assert loaded.excitation_mode == 1
    np.testing.assert_allclose(loaded.total_reflected_power, section.total_reflected_power)


def test_plotting_after_load(tmp_path: Path, ka_window_analysis) -> None:
    """Plotting helpers work on deserialised results without wg objects."""
    chain_out = tmp_path / "chain.npz"
    ka_window_analysis.save_npz(chain_out)
    loaded = pwmma.ChainEnergyCouplingResult.load_npz(chain_out)

    section_out = tmp_path / "section.npz"
    ka_window_analysis.get_section(2).save_npz(section_out)
    loaded_section = pwmma.SectionEnergyCoupling.load_npz(section_out)

    # Both plot-entry points should produce valid figures
    fig1, axes1 = pwmma.plot_section_energy_coupling(loaded_section)
    fig2, axes2 = pwmma.plot_chain_energy_overview(loaded)

    assert len(axes1) == 4
    assert len(axes2) == 3
    pwmma.save_figure(fig1, tmp_path / "loaded_section.png")
    pwmma.save_figure(fig2, tmp_path / "loaded_chain.png")


def test_modal_analysis_applies_lossless_assumption() -> None:
    """analyze_energy_coupling is a lossless theoretical tool: it strips loss
    from its inputs.

    Passing lossy waveguides (complex er, hence finite-conductivity-like
    behaviour) must yield exactly the same modal analysis as the corresponding
    lossless structure, because loss is removed internally (Re(er), perfectly
    conducting walls). This covers classification, per-mode power and balance.
    """

    def run(disk_er):
        rwg = RecWG(a=7.112e-3, b=3.556e-3, l=2e-3, N=24)
        cwg = CirWG(r=4.2e-3, l=1.5e-3, N=64)
        dsk = CirWG(r=5.4e-3, l=0.26e-3, N=96, er=disk_er)
        chain = pwmma.Chain([rwg, cwg, dsk], sym=True)
        freqs = np.linspace(28.0, 34.0, 3) * 1e9
        config = pwmma.Config(
            nproc=2, use_gpu=False,
        )
        return pwmma.analyze_energy_coupling(
            chain, freqs, config, sections=[2], show_progress=False,
        ).get_section(2)

    lossless = run(9.2)
    lossy = run(9.2 - 0.5j)

    # Sanity: the disk really has both propagating and evanescent modes here,
    # otherwise the comparison below would be vacuous.
    assert lossless.evanescent_mask.any() and lossless.propagating_mask.any()

    np.testing.assert_array_equal(lossless.propagating_mask, lossy.propagating_mask)
    np.testing.assert_array_equal(lossless.evanescent_mask, lossy.evanescent_mask)
    np.testing.assert_allclose(lossless.modal_power, lossy.modal_power)
    np.testing.assert_allclose(lossless.power_balance, lossy.power_balance)
    np.testing.assert_allclose(lossless.total_reflected_power, lossy.total_reflected_power)


@pytest.fixture(scope="module")
def ka_small():
    """Reduced-truncation Ka window chain with precomputed coupling matrices."""
    rwg = RecWG(a=7.112e-3, b=3.556e-3, l=2e-3, N=24)
    cwg = CirWG(r=4.2e-3, l=1.5e-3, N=64)
    dsk = CirWG(r=5.4e-3, l=0.26e-3, N=96, er=9.2)
    chain = pwmma.Chain([rwg, cwg, dsk], sym=True)
    config = pwmma.Config(nproc=2, use_gpu=False, use_double_precision=True)
    cms = [pwmma.get_coupling_matrix(t, config) for t in chain.transitions]
    return chain, config, cms


@pytest.fixture(scope="module")
def smoothing_case(ka_small):
    """Dense truth vs sparse-grid analysis of the reduced Ka window.

    The sparse grid is a strict subset of the dense one so the reconstruction
    is compared at exactly the frequencies the sparse analysis never saw.
    """
    chain, config, cms = ka_small
    dense = np.linspace(28.0, 34.0, 121) * 1e9
    sparse = dense[::4]                                    # 31 points
    truth = pwmma.analyze_energy_coupling(
        chain, dense, config, sections=[2], show_progress=False, cms=cms)
    coarse = pwmma.analyze_energy_coupling(
        chain, sparse, config, sections=[2], show_progress=False, cms=cms)
    return truth, coarse, dense


def test_s11_complex_matches_reflection_power(smoothing_case) -> None:
    truth, _, _ = smoothing_case
    sec = truth.get_section(2)
    np.testing.assert_allclose(
        np.abs(sec.s11_complex) ** 2, sec.reflection_power, rtol=1e-10)


def test_smooth_section_energy_matches_dense_truth(smoothing_case) -> None:
    truth, coarse, dense = smoothing_case
    sec = coarse.get_section(2)
    smooth = pwmma.smooth_section_energy(sec, dense)

    kept = smooth["mode_ids"]
    assert kept.size > 0
    assert smooth["modal_power"].shape == (dense.size, kept.size)

    # Mode 53 (TM1,3S) crosses cutoff at 29.6 GHz INSIDE the band: its branch
    # point is not representable in the shared-pole basis and the sparse
    # samples cannot resolve it either (an independent AAA fails just as
    # hard), so the honesty gate must route it to unsmoothed_mode_ids.
    assert 53 in smooth["unsmoothed_mode_ids"]
    assert 53 not in kept

    tsec = truth.get_section(2)
    # analytic factors are rebuilt exactly, not fitted
    np.testing.assert_array_equal(
        smooth["propagating_mask"], tsec.propagating_mask[:, kept])
    np.testing.assert_array_equal(
        smooth["evanescent_mask"], tsec.evanescent_mask[:, kept])

    err = np.max(np.abs(smooth["modal_power"] - tsec.modal_power[:, kept]))
    assert err < 1e-5, f"max dense-grid power error {err:.2e}"


def test_smooth_section_energy_power_floor_and_errors(smoothing_case) -> None:
    import dataclasses

    _, coarse, dense = smoothing_case
    sec = coarse.get_section(2)

    everything = pwmma.smooth_section_energy(sec, dense, power_floor=0.0)
    assert (everything["mode_ids"].size + everything["unsmoothed_mode_ids"].size
            == sec.mode_ids.size)

    strong_only = pwmma.smooth_section_energy(sec, dense, power_floor=0.5)
    assert strong_only["mode_ids"].size < everything["mode_ids"].size

    with pytest.raises(ValueError, match="section.wg"):
        pwmma.smooth_section_energy(dataclasses.replace(sec, wg=None), dense)
    stale = dataclasses.replace(sec, s11_complex=None)
    with pytest.raises(ValueError, match="s11_complex"):
        pwmma.smooth_section_energy(stale, dense)


def test_cutoff_probe_frequencies_invariants(ka_small) -> None:
    from waveguides.core import C_LIGHT

    from pwmma.analysis import _build_physical_layout, _lossless_wg

    chain, _, cms = ka_small
    f0, f1 = 28e9, 34e9
    spacing = (f1 - f0) / 2000.0

    probes = pwmma.cutoff_probe_frequencies(chain, f0, f1)
    assert probes.size > 0
    assert np.all((probes > f0) & (probes < f1))
    assert np.all(np.diff(probes) >= spacing * (1 - 1e-12))

    # never AT a cutoff: solving there is singular (beta = 0)
    cuts = []
    for wg in _build_physical_layout(chain)[0]:
        lw = _lossless_wg(wg)
        kc = np.array([m.kc for m in lw.mode_info_list])
        cuts.extend(kc * C_LIGHT / (2.0 * np.pi * np.sqrt(np.real(lw.er))))
    gap = np.min(np.abs(probes[:, None] - np.asarray(cuts)[None, :]), axis=1)
    assert gap.min() > 0.25 * spacing * (1 - 1e-12)

    # reachability filter only drops symmetry-forbidden (numerically zero)
    # families, so it reduces the set but never empties it
    filtered = pwmma.cutoff_probe_frequencies(chain, f0, f1, cms=cms)
    assert 0 < filtered.size <= probes.size


def test_adaptive_seed_frequencies_exploration_floor(ka_small) -> None:
    chain, _, cms = ka_small
    f0, f1 = 28e9, 34e9
    seeds = pwmma.adaptive_seed_frequencies(chain, f0, f1, cms=cms)
    floor = np.linspace(f0, f1, 129)
    assert np.all(np.isin(floor, seeds))          # uniform floor is present
    assert np.all(np.diff(seeds) > 0)             # sorted, unique
    assert seeds.size >= 129
