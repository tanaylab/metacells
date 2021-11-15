"""
Metacells single-cell RNA sequencing.
"""

__author__ = "Oren Ben-Kiki"
__email__ = "oren@ben-kiki.org"
__version__ = "0.7.0"

try:
    from .should_check_avx2 import SHOULD_CHECK_AVX2
except ModuleNotFoundError:
    SHOULD_CHECK_AVX2 = False

if SHOULD_CHECK_AVX2:
    from . import check_avx2

# pylint: disable=wrong-import-position

from . import pipeline as pl
from . import tools as tl
from . import utilities as ut
