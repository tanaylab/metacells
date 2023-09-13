"""
Default metacell processing pipeline.

The functions here are thin wrappers which invoke a series of steps (:py:mod:`metacells.tools`) to
provide a complete pipeline for computing metacells for your data.

This pipeline can be configured by tweaking the (many) parameters, but for any deeper customization
(adding and/or removing steps) just provide your own pipeline instead. You can use the
implementation here as a starting point.

All the functions included here are exported under ``metacells.pl``.
"""

from .collect import *
from .direct import *
from .divide_and_conquer import *
from .exclude import *
from .mark import *
from .mcview import *
from .projection import *
from .related_genes import *
from .select import *
from .umap import *
