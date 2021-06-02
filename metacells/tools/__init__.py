'''
Functions for analysis tools.

Tools take as input some annotated data and either return some computed results, or write the
results as new annotations within the same data, or return a new annotated data object containing
the results. Tools are meant to be composable into a complete processing
:py:mod:`metacells.pipeline`, typically by having one tool create annotation with "agreen upon"
name(s) and another further processing them. While this is flexible and convenient, there is no
static typing or analysis to ensure that the input of the next tool was actually created by previous
tool(s).

All the functions included here are exported under ``metacells.tl``.
'''

from .apply import *
from .candidates import *
from .deviants import *
from .dissolve import *
from .distinct import *
from .downsample import *
from .filter import *
from .group import *
from .high import *
from .knn_graph import *
from .layout import *
from .mask import *
from .named import *
from .noisy_lonely import *
from .project import *
from .properly_sampled import *
from .quality import *
from .rare import *
from .similarity import *
