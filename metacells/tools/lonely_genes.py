'''
Detect "Lonely" Genes
---------------------
'''

import logging
from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'find_lonely_genes',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def find_lonely_genes(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    max_similarity_of_genes: float = 0.15,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Detect "lonely" genes.

    Lonely genes have no (or low) correlations with any other gene. Such genes, especially if they
    are also "noisy" (have high variance) tend to throw off clustering algorithms. Since they are
    Since they are lonely, they don't contribute to meaningful clustering of the cells, and if they
    are also noisy, they will tend to be chosen as "feature" genes to cluster by.

    It is therefore useful to explicitly identify, in a pre-processing step, the few such genes, and
    exclude them from the rest of the analysis.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Variable (Gene) Annotations
        ``lonely_genes``
            A boolean mask indicating whether each gene was found to be a "lonely" gene.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Find genes which have a correlation of at most ``max_similarity_of_genes`` (default:
       {max_similarity_of_genes}) with at least one other gene.
    '''
    of, level = ut.log_operation(LOG, adata, 'find_noisy_lonely_genes',
                                 of, 'var_similarity')

    with ut.intermediate_step(adata, intermediate=intermediate):
        LOG.log(level, '  max_similarity_of_genes: %s',
                max_similarity_of_genes)

        similarity = ut.to_dense_matrix(ut.get_vv_data(adata, of), copy=True)

        assert similarity.shape[0] == similarity.shape[1]
        np.fill_diagonal(similarity, None)

        with ut.timed_step('np.nanmax'):
            ut.timed_parameters(results=similarity.shape[0],
                                elements=similarity.shape[1])
            if ut.matrix_layout(similarity) == 'column_major':
                similarity_of_genes = np.nanmax(similarity, axis=0)
            else:
                similarity_of_genes = np.nanmax(similarity, axis=1)

        mask = similarity_of_genes <= max_similarity_of_genes

    if inplace:
        ut.set_v_data(adata, 'lonely_genes', mask, ut.SAFE_WHEN_SLICING_VAR)
        return None

    ut.log_mask(LOG, level, 'lonely_genes', mask)

    return pd.Series(mask, index=adata.var_names)
