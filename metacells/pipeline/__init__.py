'''
Default metacell processing pipeline.

The functions here are thin wrappers which invoke a series of steps
(:py:mod:`metacells.preprocessing` and :py:mod:`metacells.tools`) to provide a complete pipeline for
computing metacells for your data.

This pipeline can be configured by tweaking the (many) parameters, but for any deeper customization
(adding and/or removing steps) just provide your own pipeline instead. You can use the
implementation here as a starting point.

All the functions included here are exported under ``metacells.pl``.
'''

from .clean_data import *
from .direct import *
from .direct_metacells import *
from .feature_data import *
from .result_metacells import *
