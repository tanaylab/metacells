"""
Wrapper for the ``Metacells.jl`` package.

The plan is to incrementally migrate the metacells computations implementation to Julia (instead of a combination of
Python and C++) and provide wrappers for them here. For now this provides a small set of features.

All the functions included here are exported under ``metacells.jl``.
"""

from .julia_import import *  # isort: skip
from .anndata_format import *
