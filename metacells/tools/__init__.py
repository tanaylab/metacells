'''
Functions for analysis tools.

    "Any transformation of the data matrix that is not *preprocessing*. In contrast to a
    *preprocessing* function, a *tool* usually adds an easily interpretable annotation to the data
    matrix, which can then be visualized with a corresponding plotting function."

    - Scanpy documentation

All the functions included here are exported under ``metacells.tl``.
'''

from .candidates import *
from .collect import *
from .deviants import *
from .dissolve import *
from .downsample import *
from .excess import *
from .high import *
from .knn_graph import *
from .named import *
from .noisy_lonely import *
from .properly_sampled import *
from .rare import *
from .similarity import *
