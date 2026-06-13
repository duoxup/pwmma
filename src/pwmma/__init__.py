import logging

from .analysis import (
    ChainEnergyCouplingResult,
    SectionEnergyCoupling,
    analyze_energy_coupling,
)
from .analysis_plotting import (
    plot_chain_energy_overview,
    plot_section_energy_coupling,
    plot_section_energy_heatmap,
    save_figure,
)
from .main import calc_spars_of_wgchain
from .coupling_matrix import get_coupling_matrix
from .io.numpy import prune_coupling_matrix_cache
from .inputs import Chain, Transition
from .config import Config, CMConfig, SMConfig
from .gpu import detect_gpu_availability

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
    'CMConfig',
    'SMConfig',
    # Mid-level: coupling matrix (with caching & config)
    'get_coupling_matrix',
    'prune_coupling_matrix_cache',
    # Utilities
    'detect_gpu_availability',
]
