import logging

from .analysis import (
    ChainEnergyCouplingResult,
    SectionEnergyCoupling,
    adaptive_seed_frequencies,
    analyze_energy_coupling,
    cutoff_probe_frequencies,
    smooth_section_energy,
)
from .analysis_plotting import (
    plot_chain_energy_overview,
    plot_section_energy_coupling,
    plot_section_energy_heatmap,
    save_figure,
)
from .config import Config
from .coupling_matrix import get_coupling_matrix
from .gpu import detect_gpu_availability
from .inputs import Chain, Transition
from .io.numpy import prune_coupling_matrix_cache
from .main import calc_spars_of_wgchain
from .solver import ChainSolver
from .spar_model import (
    SparModel,
    adaptive_spar_model,
    fit_spar_model,
    minus_NdB_band,
    uniform_spar_model,
)

# Library-level logger. By default does nothing (NullHandler).
# Users can enable output by configuring the 'pwmma' logger, e.g.:
#   import logging
#   logging.getLogger('pwmma').setLevel(logging.INFO)
#   logging.basicConfig()
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    # Primary workflow
    'calc_spars_of_wgchain',
    'analyze_energy_coupling',
    'smooth_section_energy',
    # Solver session (per-frequency core)
    'ChainSolver',
    # Rational sweep (AAA/AFS)
    'SparModel',
    'adaptive_seed_frequencies',
    'cutoff_probe_frequencies',
    'fit_spar_model',
    'uniform_spar_model',
    'adaptive_spar_model',
    'minus_NdB_band',
    'plot_section_energy_coupling',
    'plot_chain_energy_overview',
    'plot_section_energy_heatmap',
    'save_figure',
    # Input structures
    'Chain',
    'Transition',
    # Analysis outputs
    'SectionEnergyCoupling',
    'ChainEnergyCouplingResult',
    # Configuration
    'Config',
    # Mid-level: coupling matrix (with caching & config)
    'get_coupling_matrix',
    'prune_coupling_matrix_cache',
    # Utilities
    'detect_gpu_availability',
]
