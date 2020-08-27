'''
Functions for analysis tools.

    "Any transformation of the data matrix that is not *preprocessing*. In contrast to a
    *preprocessing* function, a *tool* usually adds an easily interpretable annotation to the data
    matrix, which can then be visualized with a corresponding plotting function."

    - Scanpy documentation

All the functions included here are exported under ``metacells.tl``.
'''

from .candidate_metacells import *
from .final_metacells import *
from .high_genes import *
from .knn_graph import *
from .lonely_genes import *
from .named_genes import *
from .outlier_cells import *
from .properly_sampled import *
from .rare_gene_modules import *
from .similarity import *
