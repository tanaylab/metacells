'''
Mask genes by their name.
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
    name: Optional[str] = None,
    invert: bool = False,
) -> Optional[ut.PandasSeries]:
    '''
    Find genes by their name.

    This creates a mask of all the genes whose name appears in ``names`` or matches any of the
    ``patterns``. If ``invert`` (default: {invert}), invert the resulting mask.

    If ``name`` (default: {name}) is specified, this is stored as a per-variable (gene) annotation
    with that name, and returns ``None``. This is useful to fill gene masks such as
    ``excluded_genes`` (genes which should be excluded from the rest of the processing) and
    ``forbidden_genes`` (genes which must not be chosen as feature genes).

    Otherwise, it returns it as a pandas series (indexed by the variable, that is gene, names).
    '''
    LOG.debug('find_named_genes...')

    assert names is not None or patterns is not None

    if names is None:
        names_mask = None
    else:
        names_set = set(names)
        names_mask = np.array([name in names_set for name in adata.var_names])

    if patterns is None:
        patterns_mask = None
    else:
        patterns_mask = ut.patterns_matches(patterns, adata.var_names)

    if names_mask is None:
        assert patterns_mask is not None
        genes_mask = patterns_mask
    elif patterns_mask is None:
        genes_mask = names_mask
    else:
        genes_mask = names_mask & patterns_mask

    if invert:
        genes_mask = ~genes_mask

    if LOG.isEnabledFor(logging.INFO):
        LOG.info('find_named_genes: %s / %s',
                 np.sum(genes_mask), genes_mask.size)

    if name is None:
        return pd.Series(genes_mask, index=adata.var_names)

    adata.var[name] = genes_mask
    ut.safe_slicing_data(name, ut.ALWAYS_SAFE)
    return None
