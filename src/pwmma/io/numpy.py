#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 21:13:06 2026

@author: duoxup
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from . import filenames
from ..inputs import Transition

WGTransition = Transition

# Prefix for the temp file used by atomic writes. It does not end in ".npy", so
# the cache scanner ignores it; the recognizable prefix lets prune clean strays
# left behind by an interrupted write without touching unrelated files.
_TMP_PREFIX = ".pwmma-cm-tmp-"


def read_coupling_matrix_from_cache(wgt: WGTransition,
                                    cm_cache_dir: str):
    """Return a cached coupling matrix usable for *wgt*.

    An exact (cross-section + mode-count) hit is returned directly. Otherwise a
    cached matrix for the same cross-sections with at least as many modes on
    both sides contains the request as a leading submatrix and is sliced to
    shape (the smallest such matrix is used). Raises ``FileNotFoundError`` when
    no compatible cache exists, so the caller recomputes.
    """
    want_c1 = filenames.wg_cross_repr(wgt.wg1)
    want_c2 = filenames.wg_cross_repr(wgt.wg2)
    n1, n2 = wgt.wg1.N, wgt.wg2.N

    # exact-match fast path: avoids scanning a possibly large directory
    exact = os.path.join(cm_cache_dir, filenames.coupling_matrix_from_wgt(wgt))
    if os.path.isfile(exact):
        return np.load(exact, mmap_mode="r")

    if not os.path.isdir(cm_cache_dir):
        raise FileNotFoundError(cm_cache_dir)

    # smallest (by area) cached matrix whose shape contains the requested one
    best = None  # (area, fname)
    for fname in os.listdir(cm_cache_dir):
        parsed = filenames.parse_coupling_matrix_filename(fname)
        if parsed is None:
            continue
        c1, cn1, c2, cn2 = parsed
        if c1 == want_c1 and c2 == want_c2 and cn1 >= n1 and cn2 >= n2:
            area = cn1 * cn2
            if best is None or area < best[0]:
                best = (area, fname)

    if best is None:
        raise FileNotFoundError(exact)

    # If another process pruned it between listing and load, np.load raises
    # FileNotFoundError -> the caller treats it as a miss and recomputes.
    arr = np.load(os.path.join(cm_cache_dir, best[1]), mmap_mode="r")
    return np.array(arr[:n1, :n2])  # materialise the sub-block, release the mmap


def save_coupling_matrix_to_cache(cm: np.ndarray,
                                   wgt: WGTransition,
                                   cm_cache_dir: str):
    """Atomically write *cm* to the cache (append-only; never deletes).

    The matrix is written to a uniquely named temp file in the same directory
    and ``os.replace``-d into its final name, so concurrent readers only ever
    see a complete file and two writers of the same name cannot corrupt it.
    Redundant (subsumed) files are removed separately, by
    :func:`prune_coupling_matrix_cache`.
    """
    os.makedirs(cm_cache_dir, exist_ok=True)
    final = os.path.join(cm_cache_dir, filenames.coupling_matrix_from_wgt(wgt))
    fd, tmp = tempfile.mkstemp(prefix=_TMP_PREFIX, dir=cm_cache_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, np.asarray(cm))
        os.replace(tmp, final)
    except BaseException:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        raise


def _dominated_filenames(entries):
    """Filenames in *entries* that another entry strictly contains.

    *entries* is an iterable of ``(fname, cross1, n1, cross2, n2)``. A file is
    dominated (and therefore redundant) when another file shares both
    cross-sections and has at least as many modes on each side, with a strictly
    larger shape — the larger matrix holds it as a leading submatrix.
    """
    items = list(entries)
    dominated = []
    for fname, c1, n1, c2, n2 in items:
        for other, oc1, on1, oc2, on2 in items:
            if other == fname:
                continue
            if (oc1 == c1 and oc2 == c2 and on1 >= n1 and on2 >= n2
                    and (on1, on2) != (n1, n2)):
                dominated.append(fname)
                break
    return dominated


def prune_coupling_matrix_cache(cm_cache_dir: str, *, dry_run: bool = False) -> dict:
    """Remove redundant coupling-matrix cache files from *cm_cache_dir*.

    Within each cross-section group, a file whose mode shape is contained by
    another's is removed (the larger matrix subsumes it). Stray temp files from
    interrupted writes are cleaned too. Returns a summary dict
    ``{"removed", "temp_removed", "kept", "freed_bytes"}``; with
    ``dry_run=True`` nothing is deleted but the summary still reports what would
    be.

    Safe to call while computations run: a reader whose chosen file is pruned
    mid-read simply recomputes. Best run when idle, though.
    """
    summary = {"removed": [], "temp_removed": [], "kept": 0, "freed_bytes": 0}
    if not os.path.isdir(cm_cache_dir):
        return summary

    entries = []
    temps = []
    for fname in os.listdir(cm_cache_dir):
        if not os.path.isfile(os.path.join(cm_cache_dir, fname)):
            continue
        parsed = filenames.parse_coupling_matrix_filename(fname)
        if parsed is not None:
            entries.append((fname, *parsed))
        elif fname.startswith(_TMP_PREFIX):
            temps.append(fname)

    dominated = set(_dominated_filenames(entries))
    summary["kept"] = len(entries) - len(dominated)

    def _drop(fname, bucket):
        path = os.path.join(cm_cache_dir, fname)
        try:
            summary["freed_bytes"] += os.path.getsize(path)
        except OSError:
            pass
        summary[bucket].append(fname)
        if not dry_run:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    for fname in sorted(dominated):
        _drop(fname, "removed")
    for fname in sorted(temps):
        _drop(fname, "temp_removed")
    return summary
