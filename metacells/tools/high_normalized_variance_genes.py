'''
Find genes which have high normalized variance (high variance/mean compared to other genes with a
similar level of expression).
'''

import logging
from typing import Optional

import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'find_high_normalized_variance_genes',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def find_high_normalized_variance_genes(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    minimal_normalized_variance_of_genes: float = 0.1,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Find genes which have high normalized variance ``of`` some data (by default, the focus).

    The normalized variance measures the variance of the gene relative to genes with a similar
    expression level. See :py:func:`metacells.utilities.preparation.get_normalized_variance_per_var`
    for details.

    Genes with a high normalized variance are candidate for being "feature genes", that is, genes
    used to determine the similarity between cells to group them into metacells.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Variable (Gene) Annotations
        ``high_normalized_variance_genes``
            A boolean mask indicating whether each gene was found to have a high normalized
            variance.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Use :py:func:`metacells.utilities.preparation.get_normalized_variance_per_var` to get the
       normalized variance of each gene.

    2. Select the genes whose normalized variance is at least
       ``minimal_normalized_variance_of_genes`` (default: {minimal_normalized_variance_of_genes}).
    '''
    of, level = \
        ut.log_operation(LOG, adata, 'find_high_normalized_variance_genes', of)

    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        normalized_variance_of_genes = \
            ut.get_normalized_variance_per_var(adata).proper
        genes_mask = \
            normalized_variance_of_genes >= minimal_normalized_variance_of_genes

    if inplace:
        ut.set_v_data(adata, 'high_normalized_variance_genes',
                      genes_mask, ut.NEVER_SAFE)
        return None

    ut.log_mask(LOG, level, 'high_normalized_variance_genes', genes_mask)

    return pd.Series(genes_mask, index=adata.var_names)
