'''
Detect "Noisy Lonely" Genes
---------------------------
'''

import logging
from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.parameters as pr
import metacells.preprocessing as pp
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    'find_noisy_lonely_genes',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def find_noisy_lonely_genes(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    max_sampled_cells: int = pr.noisy_lonely_max_sampled_cells,
    downsample_cell_quantile: float = pr.noisy_lonely_downsample_cell_quantile,
    min_gene_fraction: float = pr.noisy_lonely_min_gene_fraction,
    min_gene_normalized_variance: float = pr.noisy_lonely_min_gene_normalized_variance,
    max_gene_similarity: float = pr.noisy_lonely_max_gene_similarity,
    random_seed: int = pr.random_seed,
    inplace: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Detect "noisy lonely" genes.

    Return the indices of genes which are "noisy" (have high variance compared to their mean) and
    also "lonely" (have low correlation with all other genes). Such genes should be excluded since
    they will never meaningfully help us compute groups, and will actively cause profiles to be
    considered outliers.

    Noisy genes have high expression and variance. Lonely genes have no (or low) correlations with
    any other gene. Noisy lonely genes tend to throw off clustering algorithms. In general, such
    algorithms try to group together cells with the same overall biological state. Since the genes
    are lonely, they don't contribute towards this goal. Since they are noisy, they actively hamper
    this, because they make cells which are otherwise similar appear different (just for this lonely
    gene).

    It is therefore useful to explicitly identify, in a pre-processing step, the (few) such genes,
    and exclude them from the rest of the analysis.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Variable (Gene) Annotations
        ``noisy_lonely_genes``
            A boolean mask indicating whether each gene was found to be a "noisy lonely" gene.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    **Computation Parameters**

    1. If we have more than ``max_sampled_cells`` (default: {max_sampled_cells}), pick this number
       of random cells from the data using the ``random_seed``.

    2. Invoke :py:func:`metacells.tools.downsample.downsample_cells` to downsample the surviving
       cells to the same total number of UMIs, using the ``downsample_cell_quantile`` (default:
       {downsample_cell_quantile}) and the ``random_seed`` (default: {random_seed}).

    3. Find "noisy" genes which have a mean fraction of at least ``min_gene_fraction`` (default:
       {min_gene_fraction}) and a relative variance of at least ``min_gene_normalized_variance``
       (default: ``min_gene_normalized_variance``).

    4. Cross-correlate the noisy genes.

    5. Find the noisy "lonely" genes whose maximal correlation is at most
       ``max_gene_similarity`` (default: {max_gene_similarity}) with all other genes.
    '''
    of, level = ut.log_operation(LOG, adata, 'find_noisy_lonely_genes',
                                 of, 'var_similarity')

    LOG.log(level, '  max_sampled_cells: %s', max_sampled_cells)
    if max_sampled_cells < adata.n_obs:
        np.random.seed(random_seed)
        cell_indices = \
            np.random.choice(np.arange(adata.n_obs),
                             size=max_sampled_cells, replace=False)
        bdata = ut.slice(adata, obs=cell_indices, name='SAMPLED', tmp=True)
    else:
        bdata = adata.copy()
        bdata.uns['__tmp__'] = True

    pp.track_base_indices(bdata, name='sampled_base_index')

    tl.downsample_cells(bdata,
                        downsample_cell_quantile=downsample_cell_quantile,
                        random_seed=random_seed,
                        infocus=True)

    tl.find_high_fraction_genes(bdata, min_gene_fraction=min_gene_fraction)
    tl.find_high_normalized_variance_genes(bdata,
                                           min_gene_normalized_variance=min_gene_normalized_variance)

    ndata = pp.filter_data(bdata, name='NOISY', tmp=True,
                           masks=['high_fraction_genes',
                                  'high_normalized_variance_genes'])

    noisy_lonely_genes_mask = np.full(adata.n_vars, False)

    if ndata is not None:
        LOG.log(level, '  max_gene_similarity: %s', max_gene_similarity)

        gene_gene_similarity_frame = \
            tl.compute_var_var_similarity(ndata, inplace=False)
        assert gene_gene_similarity_frame is not None
        gene_gene_similarity = \
            ut.to_dense_matrix(gene_gene_similarity_frame)
        np.fill_diagonal(gene_gene_similarity, -1)

        assert ut.matrix_layout(gene_gene_similarity) == 'row_major'
        max_similiraity_of_genes = ut.sum_per(gene_gene_similarity, per='row')

        lonely_genes_mask = max_similiraity_of_genes < max_gene_similarity
        base_index_of_genes = \
            ut.to_dense_vector(ut.get_v_data(ndata, 'var_sampled_base_index'))
        lonely_genes_indices = base_index_of_genes[lonely_genes_mask]

        noisy_lonely_genes_mask[lonely_genes_indices] = True

    LOG.debug('noisy lonely gene names: %s',
              sorted(list(adata.var_names[noisy_lonely_genes_mask])))

    if inplace:
        ut.set_v_data(adata, 'noisy_lonely_genes',
                      noisy_lonely_genes_mask, ut.SAFE_WHEN_SLICING_VAR)
        return None

    ut.log_mask(LOG, level, 'noisy_lonely_genes', noisy_lonely_genes_mask)

    return pd.Series(noisy_lonely_genes_mask, index=adata.var_names)
