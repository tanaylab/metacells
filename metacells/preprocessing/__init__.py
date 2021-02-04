'''
Functions for pre-processing steps.

    "Any transformation of the data matrix that is not a *tool*. Other than *tools*, preprocessing
    steps usually don’t return an easily interpretable annotation, but perform a basic
    transformation on the data matrix."

    - Scanpy documentation

All the functions included here are exported under ``metacells.pp``.
'''

from .filter import *
from .group import *
