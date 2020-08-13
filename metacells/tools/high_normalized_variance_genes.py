'''
Find genes which have high normalized variance (high variance/mean compared to other genes with a
similar level of expression).
'''

from typing import Optional

import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'find_high_normalized_variance_genes',
]


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

    A :py:func:`metacells.utilities.preparation.prepare`-ed annotated ``adata``, where the
    observations are cells and the variables are genes.

    **Returns**

    Variable (Gene) Annotations
        ``high_normalized_variance_genes``
            A boolean mask indicating whether each gene was found to have a high normalized
            variance.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    If not ``intermediate`` (default: {intermediate}), this discards all the intermediate data used
    (e.g. sums). Otherwise, such data is kept for future reuse.

    **Computation Parameters**

    1. Use :py:func:`metacells.utilities.preparation.get_normalized_variance_per_var` to get the
       normalized variance of each gene.

    2. Select the genes whose normalized variance is at least
       ``minimal_normalized_variance_of_genes`` (default: {minimal_normalized_variance_of_genes}).
    '''
    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        normalized_variance_of_genes = \
            ut.get_normalized_variance_per_var(adata).proper
        genes_mask = \
            normalized_variance_of_genes >= minimal_normalized_variance_of_genes

    if inplace:
        adata.var['high_normalized_variance_genes'] = genes_mask
        ut.safe_slicing_data('high_normalized_variance_genes',
                             ut.SAFE_WHEN_SLICING_VAR)
        return None

    return pd.Series(genes_mask, index=adata.var_names)
