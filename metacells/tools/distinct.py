'''
Distincts
---------
'''

import logging
from typing import Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.extensions as xt  # type: ignore
import metacells.parameters as pr
import metacells.preprocessing as pp
import metacells.utilities as ut

__all__ = [
    'find_distinct_genes',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def find_distinct_genes(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    distinct_genes_count: int = pr.distinct_genes_count,
    consider_low_folds: bool = True,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[Tuple[ut.PandasFrame, ut.PandasFrame]]:
    '''
    Find for each observation (cell) the genes in which it is most distinct from the general
    population. This is typically applied to the metacells data rather than to the cells data.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are (mata)cells and the variables are genes.

    **Returns**

    Observation-Any (Cell) Annotations
        ``cell_distinct_gene_indices``
            For each cell, the indices of its top ``distinct_genes_count`` genes.
        ``cell_distinct_gene_folds``
            For each cell, the fold factor of its top ``distinct_genes_count``.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as two pandas frames (indexed by the observation and
    distinct gene rank).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Compute for each gene its fraction out of the total UMIs.

    2. Compute for each cell, for each gene, the fold factor (log 2 of the gene's fraction in the
       cell over the fraction in the overall data).

    3. Keep the ``distinct_genes_count`` (default: {distinct_genes_count}) top fold factors. If
       ``consider_low_folds`` (default: {consider_low_folds}), keep the top absolute fold factors,
       that is, also consider genes which have a much lower expression than in the overall data.
    '''
    of, _ = ut.log_operation(LOG, adata, 'find_distinct_genes', of)
    assert 0 < distinct_genes_count < adata.n_vars

    distinct_gene_indices = \
        np.empty((adata.n_obs, distinct_genes_count), dtype='int32')
    distinct_gene_folds = \
        np.empty((adata.n_obs, distinct_genes_count), dtype='float32')

    with ut.timed_step('.prepare'):
        with ut.focus_on(ut.get_vo_data, adata, of, layout='row_major',
                         intermediate=intermediate) as data:
            total_umis_of_cells = pp.get_per_obs(adata, ut.sum_per).proper
            total_umis_of_genes = pp.get_per_var(adata, ut.sum_per).proper

            fractions_in_data = total_umis_of_genes / \
                np.sum(total_umis_of_genes)
            fractions_in_data[fractions_in_data <= 0] = -1

            fold_in_cells = data / total_umis_of_cells[:, np.newaxis]
            fold_in_cells /= fractions_in_data
            fold_in_cells[fold_in_cells <= 0] = 1
            np.log2(fold_in_cells, out=fold_in_cells)

    with ut.timed_step('.top_distinct'):
        xt.top_distinct(distinct_gene_indices, distinct_gene_folds,
                        fold_in_cells, consider_low_folds)

    if inplace:
        ut.set_oa_data(adata, 'cell_distinct_gene_indices',
                       distinct_gene_indices)
        ut.set_oa_data(adata, 'cell_distinct_gene_folds', distinct_gene_folds)
        return None

    return pd.DataFrame(distinct_gene_indices, index=adata.obs_names), \
        pd.DataFrame(distinct_gene_folds, index=adata.obs_names)
