#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modal energy-coupling analysis utilities for waveguide chains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

import waveguides.heavy_computation as hc
from waveguides import WG
from waveguides.core import C_LIGHT

from .config import Config
from .coupling_matrix import get_coupling_matrix
from .gpu import get_array_backend
from .inputs import Chain
from .spar_model import SparModel, fit_spar_model
from .utils import judge_cross_section_containment
from .numerics.gsm import (
    apply_propagation_factors_to_smatrix,
    calc_transition_scattering_matrix,
    cascade_generalized_scattering_matrice,
)


logger = logging.getLogger(__name__)

PHYSICAL_INDEXING = "physical"


@dataclass(frozen=True)
class SectionEnergyCoupling:
    """
    Energy-coupling result for one internal waveguide section.

    All per-mode arrays use shape ``(n_freqs, n_modes)``. The section index uses
    the expanded physical-chain indexing convention. For symmetric chains this means
    indexing after expanding ``chain.wgs`` to ``chain.wgs + reversed(chain.wgs[:-1])``.

    Parameters
    ----------
    section_index:
        Physical-chain index of the analyzed internal section.
    freqs:
        Frequency array in Hz.
    mode_ids:
        Integer mode ids local to this waveguide section. Mode names are not stored;
        they can be reconstructed from ``wg`` or another compatible ``waveguides.WG``.
    modal_power:
        Net power contribution of each mode at each frequency.
    propagating_mask, evanescent_mask:
        Boolean masks indicating whether a mode is propagating or evanescent at each
        frequency point.
    forward_left, backward_left, forward_right, backward_right:
        Recovered modal wave coefficients on the left and right faces of the section.
    reflection_power:
        Reflected power of the input excitation mode at the chain input plane.
        This is the power reflected back into the same mode: ``|S11[e, e]|²``
        where ``e = excitation_mode``.
    total_reflected_power:
        Total reflected power at the chain input plane summed over all output
        modes: ``Σₖ |S11[k, excitation_mode]|²``.  Equal to or larger than
        ``reflection_power`` when mode conversion at the input is present.
    transmission_power:
        Transmitted power of the excitation mode at the chain output plane:
        ``|S21[e, e]|²`` where ``e = excitation_mode``.
    s11_complex:
        Complex ``S11[e, e]`` of the (lossless) analysis chain at each frequency.
        Unlike the main lossy S-parameter sweep, this shares its resonance poles
        with the recovered modal coefficients exactly, which is what
        :func:`smooth_section_energy` needs for its shared-pole fits.
    power_balance:
        ``total_reflected_power + sum(modal_power, axis=1)`` for this section.
        Equals 1 (within numerical precision) for a lossless structure.
    excitation_mode:
        Excited input-port mode index used in the analysis.
    wg:
        Optional waveguide object used only for reconstructing mode labels.
    """
    section_index: int
    freqs: np.ndarray
    mode_ids: np.ndarray
    modal_power: np.ndarray
    propagating_mask: np.ndarray
    evanescent_mask: np.ndarray
    forward_left: np.ndarray
    backward_left: np.ndarray
    forward_right: np.ndarray
    backward_right: np.ndarray
    reflection_power: np.ndarray
    total_reflected_power: np.ndarray
    transmission_power: np.ndarray
    power_balance: np.ndarray
    s11_complex: np.ndarray = field(default=None)
    excitation_mode: int = 0
    wg: WG | None = field(default=None, repr=False, compare=False)

    @property
    def propagating_power(self) -> np.ndarray:
        """Return ``modal_power`` with evanescent entries zeroed out."""
        return np.where(self.propagating_mask, self.modal_power, 0.0)

    @property
    def evanescent_power(self) -> np.ndarray:
        """Return ``modal_power`` with propagating entries zeroed out."""
        return np.where(self.evanescent_mask, self.modal_power, 0.0)

    @property
    def total_propagating_power(self) -> np.ndarray:
        """Return the total propagating-mode contribution at each frequency."""
        return np.sum(self.propagating_power, axis=1)

    @property
    def total_evanescent_power(self) -> np.ndarray:
        """Return the total evanescent-mode contribution at each frequency."""
        return np.sum(self.evanescent_power, axis=1)

    @property
    def total_power(self) -> np.ndarray:
        """Return ``total_reflected + propagating + evanescent`` at each frequency.
        This equals 1 for a lossless structure."""
        return self.total_reflected_power + self.total_propagating_power + self.total_evanescent_power

    def dominant_mode_ids(self, threshold: float = 0.04) -> np.ndarray:
        """
        Return mode ids whose peak absolute contribution exceeds ``threshold``.

        Parameters
        ----------
        threshold:
            Minimum value of ``max(abs(modal_power[:, mode]))`` required for a mode
            to be selected.
        """
        return self.mode_ids[np.max(np.abs(self.modal_power), axis=0) > threshold]

    def max_power_balance_error(self) -> float:
        """Return the maximum absolute deviation of ``power_balance`` from unity."""
        return float(np.max(np.abs(self.power_balance - 1.0)))

    def get_mode_labels(self, wg: WG | None = None, mode_ids: Sequence[int] | None = None) -> list[str]:
        """
        Reconstruct human-readable mode labels from a waveguide object.

        Parameters
        ----------
        wg:
            Optional waveguide instance providing ``mode_name_list``. If omitted,
            ``self.wg`` is used. If neither is available, stringified mode ids are
            returned instead.
        mode_ids:
            Optional subset of mode ids to label. Defaults to all ``self.mode_ids``.
        """
        waveguide = wg if wg is not None else self.wg
        if waveguide is None:
            ids = self.mode_ids if mode_ids is None else np.asarray(mode_ids)
            return [str(int(mode_id)) for mode_id in ids]
        labels = np.asarray(waveguide.mode_name_list)
        ids = self.mode_ids if mode_ids is None else np.asarray(mode_ids)
        return labels[ids].tolist()

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly copy of the stored numeric data."""
        return {
            "section_index": self.section_index,
            "freqs": self.freqs.copy(),
            "mode_ids": self.mode_ids.copy(),
            "modal_power": self.modal_power.copy(),
            "propagating_mask": self.propagating_mask.copy(),
            "evanescent_mask": self.evanescent_mask.copy(),
            "forward_left": self.forward_left.copy(),
            "backward_left": self.backward_left.copy(),
            "forward_right": self.forward_right.copy(),
            "backward_right": self.backward_right.copy(),
            "reflection_power": self.reflection_power.copy(),
            "total_reflected_power": self.total_reflected_power.copy(),
            "transmission_power": self.transmission_power.copy(),
            "power_balance": self.power_balance.copy(),
            "s11_complex": (self.s11_complex.copy() if self.s11_complex is not None
                            else np.full(self.reflection_power.shape, np.nan, dtype=complex)),
            "excitation_mode": self.excitation_mode,
        }

    def save_npz(self, path: str | Path) -> None:
        """Save this section result to a compressed ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.to_dict())

    @classmethod
    def load_npz(cls, path: str | Path, *, wg: WG | None = None) -> "SectionEnergyCoupling":
        """
        Load a section result saved by :meth:`save_npz`.

        Parameters
        ----------
        path:
            Input ``.npz`` file.
        wg:
            Optional waveguide object attached to the loaded result for later label
            reconstruction.
        """
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                section_index=int(data["section_index"]),
                freqs=np.asarray(data["freqs"]),
                mode_ids=np.asarray(data["mode_ids"]),
                modal_power=np.asarray(data["modal_power"]),
                propagating_mask=np.asarray(data["propagating_mask"]),
                evanescent_mask=np.asarray(data["evanescent_mask"]),
                forward_left=np.asarray(data["forward_left"]),
                backward_left=np.asarray(data["backward_left"]),
                forward_right=np.asarray(data["forward_right"]),
                backward_right=np.asarray(data["backward_right"]),
                reflection_power=np.asarray(data["reflection_power"]),
                total_reflected_power=np.asarray(
                    data.get("total_reflected_power",
                             data["reflection_power"].copy())
                ),
                transmission_power=np.asarray(
                    data.get("transmission_power",
                             np.full_like(data["reflection_power"], np.nan))
                ),
                power_balance=np.asarray(data["power_balance"]),
                s11_complex=np.asarray(
                    data.get("s11_complex",
                             np.full(data["reflection_power"].shape, np.nan)),
                    dtype=complex,
                ),
                excitation_mode=int(data["excitation_mode"]),
                wg=wg,
            )


@dataclass(frozen=True)
class ChainEnergyCouplingResult:
    """
    Energy-coupling result for one waveguide chain analysis run.

    Parameters
    ----------
    freqs:
        Frequency array in Hz.
    section_indices:
        Tuple of analyzed internal section indices. These use the physical-chain
        indexing convention.
    sections:
        Mapping from physical section index to :class:`SectionEnergyCoupling`.
    excitation_mode:
        Excited input-port mode index used in the analysis.
    indexing:
        Indexing convention string. The current implementation uses ``"physical"``.
    physical_wgs:
        Optional expanded physical waveguide tuple. This is not serialized by
        :meth:`save_npz`; it is only retained in memory for convenience.
    """
    freqs: np.ndarray
    section_indices: tuple[int, ...]
    sections: dict[int, SectionEnergyCoupling]
    excitation_mode: int = 0
    indexing: str = PHYSICAL_INDEXING
    physical_wgs: tuple[WG | None, ...] = field(default_factory=tuple, repr=False, compare=False)

    def get_section(self, section_index: int) -> SectionEnergyCoupling:
        """Return the per-section result for a physical-chain section index."""
        return self.sections[section_index]

    def max_power_balance_error(self) -> float:
        """Return the worst power-balance deviation among all stored sections."""
        return max(section.max_power_balance_error() for section in self.sections.values())

    def save_npz(self, path: str | Path) -> None:
        """
        Save the full chain analysis result to a compressed ``.npz`` file.

        The file stores only numeric result data and metadata such as frequency,
        excitation mode, section ids, and indexing convention. Waveguide objects are
        intentionally not serialized.

        ``reflection_power`` and ``total_reflected_power`` are stored once at the
        chain level (they are global properties of the input port), not per section.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        first_section = next(iter(self.sections.values()))
        arrays: dict[str, np.ndarray] = {
            "freqs": self.freqs,
            "section_indices": np.asarray(self.section_indices, dtype=int),
            "excitation_mode": np.asarray(self.excitation_mode, dtype=int),
            "indexing": np.asarray(self.indexing),
            "reflection_power": np.asarray(first_section.reflection_power),
            "total_reflected_power": np.asarray(first_section.total_reflected_power),
            "transmission_power": np.asarray(first_section.transmission_power),
            "s11_complex": np.asarray(
                first_section.s11_complex if first_section.s11_complex is not None
                else np.full(first_section.reflection_power.shape, np.nan, dtype=complex)),
        }
        _skip_keys = frozenset({"reflection_power", "total_reflected_power",
                                "transmission_power", "s11_complex"})
        for section_idx, section in self.sections.items():
            prefix = f"section_{section_idx}_"
            for key, value in section.to_dict().items():
                if key not in _skip_keys:
                    arrays[prefix + key] = np.asarray(value)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load_npz(
        cls,
        path: str | Path,
        *,
        physical_wgs: Sequence[WG | None] | None = None,
    ) -> "ChainEnergyCouplingResult":
        """
        Load a chain analysis result saved by :meth:`save_npz`.

        Parameters
        ----------
        path:
            Input ``.npz`` file.
        physical_wgs:
            Optional expanded physical waveguide sequence. When supplied, each loaded
            section result is re-attached to the matching waveguide object so plotting
            helpers can reconstruct mode labels.
        """
        with np.load(Path(path), allow_pickle=False) as data:
            section_indices = tuple(int(idx) for idx in np.asarray(data["section_indices"]))
            wg_lookup = {}
            if physical_wgs is not None:
                wg_lookup = {idx: physical_wgs[idx] for idx in range(len(physical_wgs))}

            # New format: chain-level reflection keys; old format: per-section
            if "reflection_power" in data:
                refl_global = np.asarray(data["reflection_power"])
                total_refl_global = np.asarray(data["total_reflected_power"])
                trans_global = np.asarray(data.get("transmission_power",
                                           np.full_like(refl_global, np.nan)))
                s11c_global = np.asarray(
                    data.get("s11_complex", np.full(refl_global.shape, np.nan)),
                    dtype=complex)
            else:
                refl_global = None
                total_refl_global = None
                trans_global = None
                s11c_global = None

            sections = {}
            for section_idx in section_indices:
                prefix = f"section_{section_idx}_"
                if refl_global is not None:
                    rp = refl_global
                    trp = total_refl_global
                    tp = trans_global
                    s11c = s11c_global
                else:
                    rp = np.asarray(data[prefix + "reflection_power"])
                    trp = np.asarray(
                        data.get(prefix + "total_reflected_power",
                                 data[prefix + "reflection_power"].copy())
                    )
                    tp = np.asarray(
                        data.get(prefix + "transmission_power",
                                 np.full_like(rp, np.nan))
                    )
                    s11c = np.asarray(
                        data.get(prefix + "s11_complex", np.full(rp.shape, np.nan)),
                        dtype=complex)
                sections[section_idx] = SectionEnergyCoupling(
                    section_index=int(data[prefix + "section_index"]),
                    freqs=np.asarray(data[prefix + "freqs"]),
                    mode_ids=np.asarray(data[prefix + "mode_ids"]),
                    modal_power=np.asarray(data[prefix + "modal_power"]),
                    propagating_mask=np.asarray(data[prefix + "propagating_mask"]),
                    evanescent_mask=np.asarray(data[prefix + "evanescent_mask"]),
                    forward_left=np.asarray(data[prefix + "forward_left"]),
                    backward_left=np.asarray(data[prefix + "backward_left"]),
                    forward_right=np.asarray(data[prefix + "forward_right"]),
                    backward_right=np.asarray(data[prefix + "backward_right"]),
                    reflection_power=rp,
                    total_reflected_power=trp,
                    transmission_power=tp,
                    power_balance=np.asarray(data[prefix + "power_balance"]),
                    s11_complex=s11c,
                    excitation_mode=int(data[prefix + "excitation_mode"]),
                    wg=wg_lookup.get(section_idx),
                )

            return cls(
                freqs=np.asarray(data["freqs"]),
                section_indices=section_indices,
                sections=sections,
                excitation_mode=int(data["excitation_mode"]),
                indexing=str(np.asarray(data["indexing"]).item()),
                physical_wgs=tuple(physical_wgs) if physical_wgs is not None else tuple(),
            )


TransitionSpec = tuple[int, bool]  # (base_transition_index, reverse_blocks)


def _build_physical_layout(chain: Chain) -> tuple[tuple[WG, ...], tuple[TransitionSpec, ...]]:
    if not chain.sym:
        physical_wgs = tuple(chain.wgs)
        transition_specs = tuple((idx, False) for idx in range(len(chain.transitions)))
        return physical_wgs, transition_specs

    physical_wgs = tuple(chain.wgs) + tuple(reversed(chain.wgs[:-1]))
    n_base_transitions = len(chain.transitions)
    transition_specs = tuple((idx, False) for idx in range(n_base_transitions)) + tuple(
        (idx, True) for idx in range(n_base_transitions - 1, -1, -1)
    )
    return physical_wgs, transition_specs


def _lossless_wg(wg: WG) -> WG:
    """Return a lossless copy of *wg* for theoretical modal analysis.

    The copy keeps the geometry and modal truncation but uses the real part of
    the permittivity and perfectly conducting walls (``sigma -> inf``). Modal
    analysis is therefore always performed under the lossless assumption,
    regardless of any loss carried by *wg* (the original, possibly lossy,
    waveguide is still used by the main S-parameter computation).
    """
    common = dict(l=wg.l, N=wg.N, er=float(np.real(wg.er)), sigma=float("inf"))
    if wg.cross_tag == "rec":
        return type(wg)(a=wg.a, b=wg.b, **common)
    if wg.cross_tag == "cir":
        return type(wg)(r=wg.r, **common)
    raise ValueError(f"Unsupported waveguide cross_tag for lossless copy: {wg.cross_tag!r}")


def _normalize_section_indices(
    n_wgs: int,
    sections: int | Sequence[int] | None,
) -> tuple[int, ...]:
    valid = set(range(1, n_wgs - 1))
    if not valid:
        raise ValueError("Energy coupling analysis requires at least one internal waveguide section.")

    if sections is None:
        return tuple(sorted(valid))

    if isinstance(sections, (int, np.integer)):
        requested = (int(sections),)
    else:
        requested = tuple(sections)

    if not requested:
        raise ValueError("'sections' cannot be empty.")

    unknown = [idx for idx in requested if idx not in valid]
    if unknown:
        raise ValueError(
            f"Section indices must be internal waveguide indices in [1, {n_wgs - 2}], got {unknown}."
        )

    return tuple(sorted(dict.fromkeys(requested)))


def _to_numpy(array, cnp):
    if cnp is np:
        return np.asarray(array)
    return array.get()


def _calc_total_scattering(prefix_last, p_left, p_right, *, cnp):
    return apply_propagation_factors_to_smatrix(
        prefix_last[0],
        prefix_last[1],
        prefix_last[2],
        prefix_last[3],
        p_left,
        p_right,
        cnp=cnp,
    )


def _solve_section_response(left_s, right_s, p_sec, excitation_mode, *, cnp):
    _, _, l21, l22 = left_s
    r11, _, _, _ = right_s

    p_sec = cnp.asarray(p_sec)
    r11_pp = (p_sec[:, None] * r11) * p_sec[None, :]
    mat = cnp.eye(p_sec.shape[0], dtype=l22.dtype) - l22 @ r11_pp

    rhs = l21[:, excitation_mode]
    forward_left = cnp.linalg.solve(mat, rhs)
    forward_right = p_sec * forward_left
    backward_right = r11 @ forward_right
    backward_left = p_sec * backward_right

    return forward_left, backward_left, forward_right, backward_right


def _propagating_mask(wg, freqs):
    """Return a boolean ``(n_freqs, N)`` mask of propagating modes.

    A mode is propagating when the operating frequency exceeds its cutoff
    frequency ``fc = kc · c / (2π·√(Re εr))``. The cutoff wavenumber ``kc`` is
    purely geometric and only ``Re(εr)`` enters, so the classification is
    independent of material loss: a nonzero ``Im(εr)`` adds attenuation but does
    not move the cutoff. This avoids misclassifying lossy evanescent modes,
    whose propagation factor acquires a small nonzero imaginary part.
    """
    kc = np.array([mode.kc for mode in wg.mode_info_list], dtype=float)
    fc = kc * C_LIGHT / (2.0 * np.pi * np.sqrt(np.real(wg.er)))
    return np.asarray(freqs, dtype=float)[:, None] > fc[None, :]


def _calc_modal_power(forward_left, backward_left, backward_right, z_sec, prop_mask, *, cnp):
    modal_power = cnp.zeros_like(z_sec.real)
    evan_mask = ~prop_mask

    modal_power[prop_mask] = (
        cnp.abs(forward_left[prop_mask]) ** 2 - cnp.abs(backward_right[prop_mask]) ** 2
    ).real

    # P = Im(S · sign(Im(Z))) where S = (a⁺ + a⁻)(a⁺ - a⁻)*.
    # sign(Im(Z)) encodes the TE (+1) vs TM (-1) impedance sign,
    # equivalent to √Z/√Z* = j in the full V·I* derivation.
    S = (forward_left[evan_mask] + backward_left[evan_mask]) * \
        (forward_left[evan_mask] - backward_left[evan_mask]).conj()
    sign_imz = cnp.sign(cnp.imag(z_sec[evan_mask]))
    modal_power[evan_mask] = cnp.imag(S * sign_imz)

    return modal_power, prop_mask, evan_mask


def analyze_energy_coupling(
    wgchain: Chain,
    freqs: Sequence[float],
    config: Config,
    *,
    sections: int | Sequence[int] | None = None,
    excitation_mode: int = 0,
    show_progress: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
    cms: Sequence[np.ndarray] | None = None,
) -> ChainEnergyCouplingResult:
    """
    Analyze modal net-power contributions on internal sections of a waveguide chain.

    This is a lossless theoretical tool: loss is stripped from the waveguides
    (the real part of the permittivity and perfectly conducting walls are used)
    before computing impedances and propagation factors. Passing lossy
    waveguides therefore yields the same result as the corresponding lossless
    structure. Use :func:`calc_spars_of_wgchain` when loss must be retained.

    Section indices always refer to the expanded physical chain. For a symmetric
    chain with ``Chain(..., sym=True)``, the physical chain is interpreted as
    ``chain.wgs + reversed(chain.wgs[:-1])``.

    Parameters
    ----------
    wgchain:
        Waveguide chain to analyze.
    freqs:
        Frequency array in Hz.
    config:
        Numerical configuration object.
    sections:
        Internal physical-chain section index or indices to analyze. If omitted, all
        internal sections are analyzed.
    excitation_mode:
        Input-port mode index used as the incident excitation.
    show_progress:
        Whether to display a tqdm progress bar during the frequency sweep.
    progress_callback:
        Optional callable invoked once per frequency point as
        ``progress_callback(done, total)``. Defaults to ``None``.
    cms:
        Optional precomputed coupling matrices (one per base transition). When
        given, they are used instead of being recomputed. Defaults to ``None``.

    Returns
    -------
    ChainEnergyCouplingResult
        Container holding one :class:`SectionEnergyCoupling` for each requested
        section.
    """
    physical_wgs, transition_specs = _build_physical_layout(wgchain)
    section_indices = _normalize_section_indices(len(physical_wgs), sections)
    # Modal analysis is a lossless theoretical tool: strip loss from the
    # waveguides used to compute impedances and propagation factors.
    lossless_wgs = tuple(_lossless_wg(wg) for wg in physical_wgs)

    cnp = get_array_backend(config.use_gpu)
    dtype = cnp.complex64 if not config.use_double_precision else cnp.complex128

    if len(freqs) == 0:
        raise ValueError(
            "'freqs' must contain at least one frequency point."
        )

    if excitation_mode < 0 or excitation_mode >= physical_wgs[0].N:
        raise ValueError(
            f"'excitation_mode' must be in [0, {physical_wgs[0].N - 1}], got {excitation_mode}."
        )

    base_transitions = tuple(wgchain.transitions)

    logger.info(
        "Energy coupling analysis | Frequencies: %d | Physical waveguides: %d | Sections: %s",
        len(freqs),
        len(physical_wgs),
        section_indices,
    )

    # heavy_computation is now vectorized and pool-free; cap BLAS to
    # ``config.nproc`` so this in-process sweep uses ~nproc cores instead of
    # letting OpenBLAS saturate the machine.
    with threadpool_limits(limits=config.nproc):
        zarr_list = [
            cnp.asarray(hc.impedance_array(wg, freqs), dtype=dtype)
            for wg in lossless_wgs
        ]
        ps_list = [
            cnp.asarray(hc.propagation_factor_array(wg, freqs), dtype=dtype)
            for wg in lossless_wgs
        ]

    if cms is None:
        cms = [cnp.asarray(get_coupling_matrix(wgt, config), dtype=dtype)
               for wgt in base_transitions]
    else:
        cms = [cnp.asarray(c, dtype=dtype) for c in cms]

    # A contraction (wg1 larger than wg2) needs orientation-aware assembly; this
    # is orthogonal to the sym ``reverse_blocks`` port swap, which only mirrors
    # already-correct base transitions onto the second half of a sym chain.
    base_is_contraction = [judge_cross_section_containment(wgt) == 2
                           for wgt in base_transitions]

    per_section = {
        idx: {
            "modal_power": [],
            "propagating_mask": [],
            "evanescent_mask": [],
            "forward_left": [],
            "backward_left": [],
            "forward_right": [],
            "backward_right": [],
            "reflection_power": [],
            "total_reflected_power": [],
            "transmission_power": [],
            "power_balance": [],
            "s11_complex": [],
        }
        for idx in section_indices
    }

    freqs_arr = np.asarray(freqs, dtype=float)
    section_prop_masks = {
        idx: _propagating_mask(lossless_wgs[idx], freqs_arr)
        for idx in section_indices
    }

    iterator: Iterable[tuple[int, float]] = enumerate(freqs)
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, total=len(freqs))

    with threadpool_limits(limits=config.nproc):
        for idx_f, _ in iterator:
            base_single_s = []
            for idx_t, cm in enumerate(cms):
                zarr1 = zarr_list[idx_t][idx_f]
                zarr2 = zarr_list[idx_t + 1][idx_f]
                base_single_s.append(
                    calc_transition_scattering_matrix(
                        cm,
                        zarr1,
                        zarr2,
                        is_contraction=base_is_contraction[idx_t],
                        cnp=cnp,
                        dtype=dtype,
                        conjugate_output=True,
                    )
                )

            single_s = []
            for base_idx, reverse_blocks in transition_specs:
                s_matrix = base_single_s[base_idx]
                single_s.append(tuple(reversed(s_matrix)) if reverse_blocks else s_matrix)

            prefix = [None] * len(single_s)
            prefix[0] = single_s[0]
            for idx_t in range(1, len(single_s)):
                prefix[idx_t] = cascade_generalized_scattering_matrice(
                    prefix[idx_t - 1],
                    single_s[idx_t],
                    p_int=ps_list[idx_t][idx_f],
                    cnp=cnp,
                )

            suffix = [None] * len(single_s)
            suffix[-1] = single_s[-1]
            for idx_t in range(len(single_s) - 2, -1, -1):
                suffix[idx_t] = cascade_generalized_scattering_matrice(
                    single_s[idx_t],
                    suffix[idx_t + 1],
                    p_int=ps_list[idx_t + 1][idx_f],
                    cnp=cnp,
                )

            total_s = _calc_total_scattering(
                prefix[-1],
                ps_list[0][idx_f],
                ps_list[-1][idx_f],
                cnp=cnp,
            )
            s11_full = total_s[0]
            s21_full = total_s[2]
            reflection_power = cnp.abs(s11_full[excitation_mode, excitation_mode]) ** 2
            total_reflected_power = cnp.sum(cnp.abs(s11_full[:, excitation_mode]) ** 2)
            transmission_power = cnp.abs(s21_full[excitation_mode, excitation_mode]) ** 2

            for section_idx in section_indices:
                left_s = prefix[section_idx - 1]
                right_s = suffix[section_idx]
                p_sec = ps_list[section_idx][idx_f]
                z_sec = zarr_list[section_idx][idx_f]
                prop_mask = cnp.asarray(section_prop_masks[section_idx][idx_f])

                (
                    forward_left,
                    backward_left,
                    forward_right,
                    backward_right,
                ) = _solve_section_response(
                    left_s,
                    right_s,
                    p_sec,
                    excitation_mode,
                    cnp=cnp,
                )
                modal_power, prop_mask, evan_mask = _calc_modal_power(
                    forward_left,
                    backward_left,
                    backward_right,
                    z_sec,
                    prop_mask,
                    cnp=cnp,
                )
                power_balance = total_reflected_power + cnp.sum(modal_power)

                bucket = per_section[section_idx]
                bucket["modal_power"].append(_to_numpy(modal_power, cnp))
                bucket["propagating_mask"].append(_to_numpy(prop_mask, cnp))
                bucket["evanescent_mask"].append(_to_numpy(evan_mask, cnp))
                bucket["forward_left"].append(_to_numpy(forward_left, cnp))
                bucket["backward_left"].append(_to_numpy(backward_left, cnp))
                bucket["forward_right"].append(_to_numpy(forward_right, cnp))
                bucket["backward_right"].append(_to_numpy(backward_right, cnp))
                bucket["reflection_power"].append(_to_numpy(reflection_power, cnp).item())
                bucket["total_reflected_power"].append(_to_numpy(total_reflected_power, cnp).item())
                bucket["transmission_power"].append(_to_numpy(transmission_power, cnp).item())
                bucket["power_balance"].append(_to_numpy(power_balance, cnp).item())
                bucket["s11_complex"].append(
                    complex(_to_numpy(s11_full[excitation_mode, excitation_mode], cnp).item()))

            if progress_callback is not None:
                progress_callback(idx_f + 1, len(freqs))

    result_sections = {}
    freqs_np = freqs_arr
    for section_idx in section_indices:
        data = per_section[section_idx]
        wg = physical_wgs[section_idx]
        result_sections[section_idx] = SectionEnergyCoupling(
            section_index=section_idx,
            freqs=freqs_np,
            mode_ids=np.arange(wg.N, dtype=int),
            modal_power=np.stack(data["modal_power"]),
            propagating_mask=np.stack(data["propagating_mask"]),
            evanescent_mask=np.stack(data["evanescent_mask"]),
            forward_left=np.stack(data["forward_left"]),
            backward_left=np.stack(data["backward_left"]),
            forward_right=np.stack(data["forward_right"]),
            backward_right=np.stack(data["backward_right"]),
            reflection_power=np.asarray(data["reflection_power"]),
            total_reflected_power=np.asarray(data["total_reflected_power"]),
            transmission_power=np.asarray(data["transmission_power"]),
            power_balance=np.asarray(data["power_balance"]),
            s11_complex=np.asarray(data["s11_complex"], dtype=complex),
            excitation_mode=excitation_mode,
            wg=wg,
        )

    return ChainEnergyCouplingResult(
        freqs=freqs_np,
        section_indices=section_indices,
        sections=result_sections,
        excitation_mode=excitation_mode,
        indexing=PHYSICAL_INDEXING,
        physical_wgs=physical_wgs,
    )


def cutoff_probe_frequencies(
    chain: Chain,
    f0: float,
    f1: float,
    *,
    cms: Sequence[np.ndarray] | None = None,
    excitation_mode: int = 0,
    offsets: Sequence[float] = (1e-3, 3e-3, 6e-3, 12e-3),
    reach_rtol: float = 1e-8,
    min_spacing: float | None = None,
) -> np.ndarray:
    """Cutoff-hotspot probe frequencies for adaptive S-parameter sampling.

    Just above an internal-section mode's cutoff the mode's group delay
    diverges, so resonances (near-cutoff trapped modes, and Fabry-Perot
    resonances of slowly-propagating modes) CLUSTER there. Probing a few
    relative ``offsets`` above every relevant in-band cutoff densifies the
    sampling in those hotspots. This is a heuristic densifier, NOT a
    guarantee — resonances of well-propagating modes can sit anywhere in the
    band; the exploration guarantee comes from the uniform floor that
    :func:`adaptive_seed_frequencies` adds around these probes.

    ``cms``: optional coupling matrices (one per base transition). When
    given, section modes are filtered by a reachability bound — the |CM|
    matrix-vector chain from the excited port mode. The reach distribution
    is bimodal (coupled modes ~1e-2..1, symmetry-forbidden ones ~1e-17 =
    numerical zeros, nothing in between), so the conservative default
    ``reach_rtol=1e-8`` only drops modes that provably cannot couple — on a
    heavily overmoded chain that is still most of them.

    A probe exactly AT a cutoff is never emitted — beta = 0 makes the
    junction assembly singular there (solving there returns non-finite
    values). ``min_spacing`` (default ``(f1-f0)/2000``, the AFS
    candidate-grid scale) deduplicates the probe set.
    """
    physical_wgs, transition_specs = _build_physical_layout(chain)
    if min_spacing is None:
        min_spacing = (f1 - f0) / 2000.0
    max_off = max(offsets)

    # reachability walk: |CM| matvecs from the excited port mode, renormalized
    # per junction. A heuristic upper bound — orientation of a square CM is
    # taken from the sym mirror flag; magnitudes only, no propagation physics.
    reach_per_section: list[np.ndarray | None] = [None] * len(physical_wgs)
    if cms is not None:
        v = np.zeros(physical_wgs[0].N)
        v[excitation_mode] = 1.0
        for j, (base_idx, reverse_blocks) in enumerate(transition_specs):
            M = np.abs(np.asarray(cms[base_idx]))
            n_next = physical_wgs[j + 1].N
            if M.shape == (n_next, v.size) and (M.shape[0] != M.shape[1] or not reverse_blocks):
                v = M @ v
            elif M.shape == (v.size, n_next):
                v = M.T @ v
            else:
                raise ValueError(
                    f"coupling matrix {base_idx} shape {M.shape} does not link "
                    f"sections of size {v.size} and {n_next}")
            v = v / max(v.max(), 1e-300)
            reach_per_section[j + 1] = v

    cutoffs_all: list[float] = []            # singularity-exclusion set
    probes: list[float] = []
    for idx, wg in enumerate(physical_wgs):
        lw = _lossless_wg(wg)
        kc = np.array([mode.kc for mode in lw.mode_info_list], dtype=float)
        fc = kc * C_LIGHT / (2.0 * np.pi * np.sqrt(np.real(lw.er)))
        in_reach = (fc > f0 * (1.0 - 2.0 * max_off)) & (fc < f1)
        cutoffs_all.extend(fc[in_reach])
        if idx == 0 or idx == len(physical_wgs) - 1:
            continue                          # ports host no trapped modes
        keep = in_reach
        if reach_per_section[idx] is not None:
            keep = keep & (reach_per_section[idx] > reach_rtol)
        for f_c in fc[keep]:
            probes.extend(f_c * (1.0 + np.asarray(offsets, dtype=float)))

    if not probes:
        return np.empty(0, dtype=float)
    probes_arr = np.asarray(sorted(probes), dtype=float)
    probes_arr = probes_arr[(probes_arr > f0) & (probes_arr < f1)]
    if cutoffs_all:
        near_cut = np.min(
            np.abs(probes_arr[:, None] - np.asarray(cutoffs_all)[None, :]), axis=1)
        probes_arr = probes_arr[near_cut > 0.25 * min_spacing]

    kept: list[float] = []
    for f in probes_arr:                      # greedy min-spacing dedupe
        if not kept or f - kept[-1] >= min_spacing:
            kept.append(float(f))
    return np.asarray(kept, dtype=float)


def adaptive_seed_frequencies(
    chain: Chain,
    f0: float,
    f1: float,
    *,
    cms: Sequence[np.ndarray] | None = None,
    n_uniform: int = 129,
    **probe_kw,
) -> np.ndarray:
    """Precision (fine-tooth) seed set for :func:`pwmma.adaptive_spar_model`.

    This is an OPT-IN preset, not the economy default: it multiplies the
    solve count severalfold, which is only worth paying when a run must
    identify every narrow tooth. The bare AFS stop rule measures model
    self-consistency, not truth, so on a window with a quiet S11 background
    the loop can converge before any sample has come within a resonance's
    influence radius (observed on the default Ka window: 8 solves, a narrow
    sub-relevance tooth at 29.17 GHz skipped — acceptable for design use,
    where such teeth rarely matter and CST-class solvers barely resolve them
    either). Two ingredients remove the luck when it does matter:

    - a **uniform exploration floor** of ``n_uniform`` points. With spacing
      ``d``, a tooth is reliably felt when its strength x width exceeds
      ~``tol * d/2``; wall loss floors realistic tooth widths, so the default
      129 catches every solidly-visible tooth of the validated window classes
      (Ka doublet: caught with 2x margin at 129, and even at 65).
    - :func:`cutoff_probe_frequencies` hotspot probes, densifying near
      in-band cutoffs where resonances cluster.

    Pass the result as ``seed=``; the validated AFS internals stay untouched
    and refine whatever the seeds light up.
    """
    probes = cutoff_probe_frequencies(chain, f0, f1, cms=cms, **probe_kw)
    return np.unique(np.concatenate([np.linspace(f0, f1, n_uniform), probes]))


def smooth_section_energy(
    section: SectionEnergyCoupling,
    freqs_dense: Sequence[float],
    *,
    base_model: SparModel | None = None,
    power_floor: float = 1e-3,
    rtol: float = 1e-2,
) -> dict[str, np.ndarray]:
    """Dense modal-power curves for one section, rebuilt from sparse samples.

    Under adaptive sampling the energy analysis runs on a sparse grid, so its
    curves alias the resonance teeth. This rebuilds them densely with ZERO
    extra solver calls, by splitting the power formula into its two natures:

    - the complex modal coefficients ``forward_left`` / ``backward_right`` are
      (near-)meromorphic with the chain's shared resonance poles — those are
      modeled by shared-pole :meth:`SparModel.sibling_fit`s off ``base_model``;
    - everything analytic (cutoff masks, ``sign(Im Z)``, propagation factors,
      and the power formula itself) is evaluated exactly on ``freqs_dense``.

    ``base_model`` defaults to an AAA fit of the section's own ``s11_complex``
    samples — the LOSSLESS analysis chain's reflection, whose poles are exactly
    those of the modal coefficients. (The main sweep's lossy S11 model is NOT a
    valid substitute: wall loss widens its poles by up to the wall-Q, which is
    comparable to or below the leakage-Q of trapped-mode teeth.)

    Modes whose sampled ``max |modal_power|`` stays below ``power_floor`` are
    dropped (they carry no visible curve); their ids are simply absent from the
    returned arrays.

    Per-mode honesty gate: a mode whose coefficients the shared-pole basis
    cannot even reproduce AT THE SAMPLES (relative residual > ``rtol``) goes to
    ``unsmoothed_mode_ids`` instead of getting a dense curve. The dominant
    cause is an in-band cutoff of the mode itself: the branch point is not
    meromorphic AND the sparse samples do not resolve it (an independent
    per-curve AAA fails just as hard there), so no rebuilt curve would be
    trustworthy — callers should show such modes as the plain sampled polyline.
    Healthy modes sit ~5 orders of magnitude below the default gate.

    Returns a plain-numpy dict: ``freqs``, ``mode_ids`` (smoothed modes),
    ``modal_power``, ``propagating_mask``, ``evanescent_mask`` (all dense,
    shaped ``(n_dense,)`` / ``(K,)`` / ``(n_dense, K)``) plus
    ``unsmoothed_mode_ids``.
    """
    if section.wg is None:
        raise ValueError("section.wg is required to rebuild the analytic factors")
    freqs_dense = np.asarray(freqs_dense, dtype=float)

    if base_model is None:
        s11c = section.s11_complex
        if s11c is None or not np.isfinite(np.asarray(s11c, dtype=complex)).all():
            raise ValueError(
                "section.s11_complex is missing (result predates it?); "
                "pass an explicit base_model instead"
            )
        base_model = fit_spar_model(section.freqs, s11c)

    peak = np.max(np.abs(section.modal_power), axis=0)
    candidates = np.asarray(section.mode_ids[peak >= power_floor], dtype=int)

    F = section.freqs
    fwd_cols, bwd_cols = [], []
    smoothed_ids, unsmoothed_ids = [], []
    for mode_id in candidates:
        y_f = np.asarray(section.forward_left[:, mode_id], dtype=complex)
        y_b = np.asarray(section.backward_right[:, mode_id], dtype=complex)
        sib_f = base_model.sibling_fit(F, y_f)
        sib_b = base_model.sibling_fit(F, y_b)
        residual = max(
            np.max(np.abs(sib_f(F) - y_f)) / max(np.max(np.abs(y_f)), 1e-300),
            np.max(np.abs(sib_b(F) - y_b)) / max(np.max(np.abs(y_b)), 1e-300),
        )
        if residual > rtol:
            unsmoothed_ids.append(int(mode_id))
            continue
        smoothed_ids.append(int(mode_id))
        fwd_cols.append(sib_f(freqs_dense))
        bwd_cols.append(sib_b(freqs_dense))

    kept = np.asarray(smoothed_ids, dtype=int)
    n_dense = freqs_dense.size
    forward_left = (np.stack(fwd_cols, axis=1) if fwd_cols
                    else np.empty((n_dense, 0), dtype=complex))
    backward_right = (np.stack(bwd_cols, axis=1) if bwd_cols
                      else np.empty((n_dense, 0), dtype=complex))

    wg = _lossless_wg(section.wg)
    prop_mask = _propagating_mask(wg, freqs_dense)[:, kept]
    z_dense = np.asarray(hc.impedance_array(wg, freqs_dense))[:, kept]
    p_dense = np.asarray(hc.propagation_factor_array(wg, freqs_dense))[:, kept]
    backward_left = p_dense * backward_right

    modal_power, prop_mask, evan_mask = _calc_modal_power(
        forward_left, backward_left, backward_right, z_dense, prop_mask, cnp=np,
    )
    return {
        "freqs": freqs_dense,
        "mode_ids": kept,
        "modal_power": modal_power,
        "propagating_mask": prop_mask,
        "evanescent_mask": evan_mask,
        "unsmoothed_mode_ids": np.asarray(unsmoothed_ids, dtype=int),
    }
