'''
Metacells single-cell RNA sequencing.
'''

from .version import version as __version__
__license__ = 'MIT'
__author__ = 'Oren Ben-Kiki'
__email__ = 'oren@ben-kiki.org'

from .should_check_avx2 import SHOULD_CHECK_AVX2
if SHOULD_CHECK_AVX2:
    from . import check_avx2

# pylint: disable=wrong-import-position

from . import pipeline as pl
from . import tools as tl
from . import utilities as ut
