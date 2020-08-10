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
def find_high_normalized_variance_genes(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    minimal_normalized_variance_of_genes: float = 0.1,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[pd.Series]:
    '''
    Find genes which have high normalized variance (high variance/mean compared to other genes with
    a similar level of expression).

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Such genes are candidate for being "feature genes", that is, genes used to determine the
    similarity between cells to group them into metacells.

    **Input**

    A :py:func:`metacells.utilities.preparation.prepare`-ed annotated ``adata``, where the
    observations are cells and the variables are genes, containing the UMIs count in the ``of``
    (default: the focus) per-variable-per-observation data.

    **Returns**

    Variable (Gene) Annotations
        ``high_normalized_variance_genes``
            A boolean mask indicating whether each gene was found to have a high normalized
            variance.

    If ``inplace`` (default: {inplace}), these are written to ``adata`` and the function returns
    ``None``. Otherwise this is returned as a Pandas series (indexed by the variable names).

    If not ``intermediate`` (default: {intermediate}), this discards all the intermediate data used
    (e.g. sums). Otherwise, such data is kept for future reuse.

    **Computation Parameters**

    Given an annotated ``adata``, where the variables are cell RNA profiles and the observations are
    gene UMI counts, do the following:

    1. Use :py:func:`metacells.utilities.preparation.get_normalized_variance_per_var` to get the
       normalized variance of each gene.

    2. Select the genes whose normalized variance is at least
       ``minimal_normalized_variance_of_genes`` (default: {minimal_normalized_variance_of_genes}).
    '''
    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        normalized_variance_of_genes = \
            ut.unpandas(ut.get_normalized_variance_per_var(adata))
        genes_mask = normalized_variance_of_genes >= minimal_normalized_variance_of_genes

    if inplace:
        adata.var['high_normalized_variance_genes'] = genes_mask
        ut.safe_slicing_data('high_normalized_variance_genes',
                             ut.SAFE_WHEN_SLICING_VAR)
        return None

    return pd.Series(genes_mask, index=adata.var_names)
