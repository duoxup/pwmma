#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coupling-matrix disk cache: submatrix slicing, atomic append-only writes,
and the manual prune of redundant files.

A cached matrix for the same cross-sections with at least as many modes on both
sides contains the requested one as a leading submatrix (coupling entries are
per-mode-pair integrals independent of N, and modes are cutoff-sorted), so it can
be sliced instead of recomputed. Writes are atomic and never delete; redundant
(subsumed) files are removed only by an explicit prune call.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from waveguides import CirWG, RecWG

import pwmma


def _cfg(cache_dir=None):
    return pwmma.CMConfig(
        nproc=2,
        cm_cache_dir=cache_dir,
        try_read_cm_from_cache=cache_dir is not None,
        save_cm_to_cache=cache_dir is not None,
    )


def _cm(small_n, large_n):
    """A computed coupling matrix (no cache) for a cir->cir expansion."""
    wgt = pwmma.Transition(CirWG(r=3.0e-3, N=small_n), CirWG(r=4.2e-3, N=large_n))
    return wgt, pwmma.get_coupling_matrix(wgt, _cfg())


def _npy(cache_dir):
    return sorted(f for f in os.listdir(cache_dir) if f.endswith(".npy"))


# --------------------------------------------------------------------------- #
# Filename helpers (pure)
# --------------------------------------------------------------------------- #

def test_wg_cross_repr_excludes_mode_count():
    from pwmma.io import filenames

    a = CirWG(r=4.2e-3, N=64)
    b = CirWG(r=4.2e-3, N=96)
    assert filenames.wg_cross_repr(a) == filenames.wg_cross_repr(b)
    assert filenames.wg_repr(a) != filenames.wg_repr(b)


def test_parse_coupling_matrix_filename_roundtrip():
    from pwmma.io import filenames

    wgt = pwmma.Transition(CirWG(r=4.2e-3, N=64), RecWG(a=7.112e-3, b=3.556e-3, N=24))
    fname = filenames.coupling_matrix_from_wgt(wgt)
    assert filenames.parse_coupling_matrix_filename(fname) == (
        filenames.wg_cross_repr(wgt.wg1), 64, filenames.wg_cross_repr(wgt.wg2), 24,
    )


def test_parse_rejects_non_cache_names():
    from pwmma.io import filenames

    assert filenames.parse_coupling_matrix_filename("tmpAB12CD") is None  # mkstemp temp
    assert filenames.parse_coupling_matrix_filename("notes.txt") is None
    assert filenames.parse_coupling_matrix_filename("cir_r4.2_n64.npy") is None  # no _&_


def test_dominated_filenames_pure():
    from pwmma.io.numpy import _dominated_filenames

    entries = [
        ("f10x15", "A", 10, "B", 15),
        ("f20x30", "A", 20, "B", 30),   # dominates f10x15
        ("f25x12", "A", 25, "B", 12),   # incomparable with f20x30
    ]
    assert set(_dominated_filenames(entries)) == {"f10x15"}

    # a different cross-section never dominates, equal shapes never delete both
    mixed = [("a", "A", 10, "B", 15), ("b", "C", 20, "D", 30)]
    assert _dominated_filenames(mixed) == []


# --------------------------------------------------------------------------- #
# Leading-submatrix assumption (numerical foundation of slicing)
# --------------------------------------------------------------------------- #

def test_larger_matrix_contains_smaller_as_leading_block():
    _, cm_small = _cm(10, 15)
    _, cm_big = _cm(20, 30)
    np.testing.assert_allclose(np.asarray(cm_big)[:10, :15], np.asarray(cm_small))


# --------------------------------------------------------------------------- #
# read / save primitives
# --------------------------------------------------------------------------- #

def test_save_then_read_exact(tmp_path):
    from pwmma.io.numpy import read_coupling_matrix_from_cache, save_coupling_matrix_to_cache

    wgt, cm = _cm(10, 15)
    save_coupling_matrix_to_cache(cm, wgt, str(tmp_path))
    got = read_coupling_matrix_from_cache(wgt, str(tmp_path))
    np.testing.assert_allclose(np.asarray(got), np.asarray(cm))


def test_read_slices_from_larger_cache(tmp_path):
    from pwmma.io.numpy import read_coupling_matrix_from_cache, save_coupling_matrix_to_cache

    big_wgt, cm_big = _cm(20, 30)
    save_coupling_matrix_to_cache(cm_big, big_wgt, str(tmp_path))

    small_wgt = pwmma.Transition(CirWG(r=3.0e-3, N=10), CirWG(r=4.2e-3, N=15))
    got = read_coupling_matrix_from_cache(small_wgt, str(tmp_path))
    assert got.shape == (10, 15)
    np.testing.assert_allclose(np.asarray(got), np.asarray(cm_big)[:10, :15])


def test_read_raises_when_shapes_incomparable(tmp_path):
    from pwmma.io.numpy import read_coupling_matrix_from_cache, save_coupling_matrix_to_cache

    cached = pwmma.Transition(CirWG(r=3.0e-3, N=10), CirWG(r=4.2e-3, N=30))
    save_coupling_matrix_to_cache(
        pwmma.get_coupling_matrix(cached, _cfg()), cached, str(tmp_path))

    # bigger on side 1, smaller on side 2 -> not contained
    req = pwmma.Transition(CirWG(r=3.0e-3, N=20), CirWG(r=4.2e-3, N=15))
    with pytest.raises(FileNotFoundError):
        read_coupling_matrix_from_cache(req, str(tmp_path))


def test_save_is_atomic_and_leaves_no_temp(tmp_path):
    from pwmma.io.numpy import save_coupling_matrix_to_cache

    wgt, cm = _cm(10, 15)
    save_coupling_matrix_to_cache(cm, wgt, str(tmp_path))
    files = os.listdir(tmp_path)
    assert files == _npy(tmp_path)  # every file is a finished .npy, no temp left
    assert len(files) == 1


def test_save_is_append_only(tmp_path):
    from pwmma.io.numpy import save_coupling_matrix_to_cache

    for sn, ln in [(10, 30), (20, 15)]:  # incomparable shapes
        wgt = pwmma.Transition(CirWG(r=3.0e-3, N=sn), CirWG(r=4.2e-3, N=ln))
        save_coupling_matrix_to_cache(pwmma.get_coupling_matrix(wgt, _cfg()), wgt, str(tmp_path))
    assert len(_npy(tmp_path)) == 2  # nothing deleted


# --------------------------------------------------------------------------- #
# prune
# --------------------------------------------------------------------------- #

def test_prune_removes_dominated_keeps_incomparable(tmp_path):
    from pwmma.io.numpy import prune_coupling_matrix_cache, save_coupling_matrix_to_cache

    shapes = [(10, 15), (20, 30), (25, 12)]  # (20,30) dominates (10,15); (25,12) incomparable
    for sn, ln in shapes:
        wgt = pwmma.Transition(CirWG(r=3.0e-3, N=sn), CirWG(r=4.2e-3, N=ln))
        save_coupling_matrix_to_cache(pwmma.get_coupling_matrix(wgt, _cfg()), wgt, str(tmp_path))

    summary = prune_coupling_matrix_cache(str(tmp_path))
    assert len(summary["removed"]) == 1
    assert summary["kept"] == 2
    assert len(_npy(tmp_path)) == 2

    # idempotent
    assert prune_coupling_matrix_cache(str(tmp_path))["removed"] == []


def test_prune_dry_run_reports_without_deleting(tmp_path):
    from pwmma.io.numpy import prune_coupling_matrix_cache, save_coupling_matrix_to_cache

    for sn, ln in [(10, 15), (20, 30)]:  # second dominates first
        wgt = pwmma.Transition(CirWG(r=3.0e-3, N=sn), CirWG(r=4.2e-3, N=ln))
        save_coupling_matrix_to_cache(pwmma.get_coupling_matrix(wgt, _cfg()), wgt, str(tmp_path))

    summary = prune_coupling_matrix_cache(str(tmp_path), dry_run=True)
    assert len(summary["removed"]) == 1
    assert len(_npy(tmp_path)) == 2  # nothing actually deleted


def test_prune_cleans_stray_temp_files(tmp_path):
    from pwmma.io.numpy import _TMP_PREFIX, prune_coupling_matrix_cache

    stray = tmp_path / f"{_TMP_PREFIX}orphan"
    stray.write_bytes(b"partial")
    unrelated = tmp_path / "keep_me.txt"
    unrelated.write_text("not ours")

    prune_coupling_matrix_cache(str(tmp_path))
    assert not stray.exists()       # our temp cleaned
    assert unrelated.exists()       # unrelated file untouched


def test_prune_missing_dir_is_noop():
    from pwmma.io.numpy import prune_coupling_matrix_cache

    summary = prune_coupling_matrix_cache(str("/no/such/dir/pwmma_xyz"))
    assert summary["removed"] == [] and summary["kept"] == 0


# --------------------------------------------------------------------------- #
# end-to-end through get_coupling_matrix
# --------------------------------------------------------------------------- #

def test_get_coupling_matrix_slices_without_saveback(tmp_path):
    cfg = _cfg(str(tmp_path))
    big = pwmma.Transition(CirWG(r=3.0e-3, N=20), CirWG(r=4.2e-3, N=30))
    pwmma.get_coupling_matrix(big, cfg)            # computes + saves
    before = _npy(tmp_path)

    small = pwmma.Transition(CirWG(r=3.0e-3, N=10), CirWG(r=4.2e-3, N=15))
    cm = pwmma.get_coupling_matrix(small, cfg)     # should slice, not save
    assert cm.shape == (10, 15)
    assert _npy(tmp_path) == before                # no new file written


def test_get_coupling_matrix_appends_when_not_contained(tmp_path):
    cfg = _cfg(str(tmp_path))
    small = pwmma.Transition(CirWG(r=3.0e-3, N=10), CirWG(r=4.2e-3, N=15))
    pwmma.get_coupling_matrix(small, cfg)
    bigger = pwmma.Transition(CirWG(r=3.0e-3, N=20), CirWG(r=4.2e-3, N=30))
    pwmma.get_coupling_matrix(bigger, cfg)         # not contained -> recompute + append
    assert len(_npy(tmp_path)) == 2                # append-only: small stays until prune


def test_prune_coupling_matrix_cache_is_public():
    assert hasattr(pwmma, "prune_coupling_matrix_cache")
