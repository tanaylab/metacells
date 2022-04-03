"""
Metacells single-cell RNA sequencing.
"""

import sys

__author__ = "Oren Ben-Kiki"
__email__ = "oren@ben-kiki.org"
__version__ = "0.8.0"

if "sphinx" not in sys.argv[0]:
    from .should_check_avx2 import SHOULD_CHECK_AVX2

    if SHOULD_CHECK_AVX2:
        from . import check_avx2

# pylint: disable=wrong-import-position

from . import pipeline as pl
from . import tools as tl
from . import utilities as ut
