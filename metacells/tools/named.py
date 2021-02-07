'''
Named
-----
'''

import logging
from re import Pattern
from typing import Collection, Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'find_named_genes',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def find_named_genes(
    adata: AnnData,
    names: Optional[Collection[str]] = None,
    patterns: Optional[Collection[Union[str, Pattern]]] = None,
    *,
    to: Optional[str] = None,
    invert: bool = False,
) -> Optional[ut.PandasSeries]:
    '''
    Find genes by their name.

    This creates a mask of all the genes whose name appears in ``names`` or matches any of the
    ``patterns``. If ``invert`` (default: {invert}), invert the resulting mask.

    If ``to`` (default: {to}) is specified, this is stored as a per-variable (gene) annotation with
    that name, and returns ``None``. This is useful to fill gene masks such as ``excluded_genes``
    (genes which should be excluded from the rest of the processing) and ``forbidden_genes`` (genes
    which must not be chosen as feature genes).

    Otherwise, it returns it as a pandas series (indexed by the variable, that is gene, names).
    '''
    level = ut.log_operation(LOG, adata, 'find_named_genes')

    if names is None:
        names_mask = np.zeros(adata.n_vars, dtype='bool')
    else:
        names_set = set(names)
        names_mask = np.array([name in names_set for name in adata.var_names])

    if patterns is None:
        patterns_mask = np.zeros(adata.n_vars, dtype='bool')
    else:
        patterns_mask = ut.patterns_matches(patterns, adata.var_names)

    genes_mask = names_mask | patterns_mask

    if invert:
        genes_mask = ~genes_mask

    if to is not None:
        ut.set_v_data(adata, to, genes_mask)
        return None

    ut.log_mask(LOG, level, 'named_genes', genes_mask)

    return pd.Series(genes_mask, index=adata.var_names)
