#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modal energy-coupling analysis utilities for waveguide chains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import waveguides.heavy_computation as hc
from waveguides import WG

from .config import Config
from .coupling_matrix import get_coupling_matrix
from .gpu import get_array_backend
from .inputs import Chain
from .numerics.gsm import (
    apply_propagation_factors_to_smatrix,
    calc_scattering_matrix,
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
        }
        _skip_keys = frozenset({"reflection_power", "total_reflected_power", "transmission_power"})
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
            else:
                refl_global = None
                total_refl_global = None
                trans_global = None

            sections = {}
            for section_idx in section_indices:
                prefix = f"section_{section_idx}_"
                if refl_global is not None:
                    rp = refl_global
                    trp = total_refl_global
                    tp = trans_global
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
    fc = kc * hc.C_LIGHT / (2.0 * np.pi * np.sqrt(np.real(wg.er)))
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

    cm_config, sm_config = config.cmconf, config.smconf
    cnp = get_array_backend(sm_config.use_gpu)
    dtype = cnp.complex64 if not sm_config.use_double_precision else cnp.complex128

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

    with Pool(processes=sm_config.nproc) as pool:
        zarr_list = [
            cnp.asarray(hc.impedance_array(wg, freqs, pool=pool, chunksize=2048), dtype=dtype)
            for wg in lossless_wgs
        ]
        ps_list = [
            cnp.asarray(
                hc.propagation_factor_array(wg, freqs, pool=pool, chunksize=2048),
                dtype=dtype,
            )
            for wg in lossless_wgs
        ]

    cms = [cnp.asarray(get_coupling_matrix(wgt, cm_config), dtype=dtype) for wgt in base_transitions]

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

    for idx_f, _ in iterator:
        base_single_s = []
        for idx_t, cm in enumerate(cms):
            zarr1 = zarr_list[idx_t][idx_f]
            zarr2 = zarr_list[idx_t + 1][idx_f]
            base_single_s.append(
                calc_scattering_matrix(
                    cm,
                    zarr1,
                    zarr2,
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
