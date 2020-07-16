'''
Metacells single-cell RNA sequencing.
'''

from . import preprocessing as pp
from . import utilities as ut

# from .version import __version__
__license__ = 'MIT'
__author__ = 'Oren Ben-Kiki'
__email__ = 'oren@ben-kiki.org'

import importlib_metadata  # type: ignore
try:
    __version__ = importlib_metadata.version(__name__)
except:  # pylint: disable=bare-except
    __version__ = 'unknown'
