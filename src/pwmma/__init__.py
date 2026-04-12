import logging

from .main import calc_spars_of_wgchain
from .coupling_matrix import get_coupling_matrix
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
    # Input structures
    'Chain',
    'Transition',
    # Configuration
    'Config',
    'CMConfig',
    'SMConfig',
    # Mid-level: coupling matrix (with caching & config)
    'get_coupling_matrix',
    # Utilities
    'detect_gpu_availability',
]
