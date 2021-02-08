'''
Noisy Lonely
------------
'''

import logging
from typing import Optional, Union

import numpy as np
from anndata import AnnData

import metacells.parameters as pr
import metacells.preprocessing as pp
import metacells.utilities as ut
from metacells.tools.downsample import downsample_cells
from metacells.tools.high import (find_high_fraction_genes,
                                  find_high_normalized_variance_genes)
from metacells.tools.similarity import compute_var_var_similarity

__all__ = [
    'find_noisy_lonely_genes',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def find_noisy_lonely_genes(
    adata: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
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
    considered "deviants".

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
    level = ut.log_operation(LOG, adata, 'find_noisy_lonely_genes', what)

    LOG.debug('  max_sampled_cells: %s', max_sampled_cells)
    if max_sampled_cells < adata.n_obs:
        np.random.seed(random_seed)
        cell_indices = \
            np.random.choice(np.arange(adata.n_obs),
                             size=max_sampled_cells, replace=False)
        bdata = ut.slice(adata, obs=cell_indices, name='.sampled', tmp=True)
    else:
        bdata = adata.copy()
        bdata.uns['__tmp__'] = True

    downsample_cells(bdata, what,
                     downsample_cell_quantile=downsample_cell_quantile,
                     random_seed=random_seed)

    find_high_fraction_genes(bdata, 'downsampled',
                             min_gene_fraction=min_gene_fraction)
    find_high_normalized_variance_genes(bdata, 'downsampled',
                                        min_gene_normalized_variance=min_gene_normalized_variance)

    results = pp.filter_data(bdata, name='noisy', tmp=True,
                             track_var='sampled_gene_index',
                             var_masks=['high_fraction_gene',
                                        'high_normalized_variance_gene'])
    assert results is not None
    ndata = results[0]

    noisy_lonely_genes_mask = np.full(adata.n_vars, False)

    if ndata is not None:
        LOG.debug('  max_gene_similarity: %s', max_gene_similarity)

        gene_gene_similarity_frame = \
            compute_var_var_similarity(ndata, 'downsampled', inplace=False)
        assert gene_gene_similarity_frame is not None
        gene_gene_similarity = \
            ut.to_numpy_matrix(gene_gene_similarity_frame, only_extract=True)
        np.fill_diagonal(gene_gene_similarity, -1)

        assert ut.matrix_layout(gene_gene_similarity) == 'row_major'
        max_similarity_of_genes = ut.max_per(gene_gene_similarity, per='row')

        lonely_genes_mask = max_similarity_of_genes < max_gene_similarity
        base_index_of_genes = ut.get_v_numpy(ndata, 'sampled_gene_index')
        lonely_genes_indices = base_index_of_genes[lonely_genes_mask]

        noisy_lonely_genes_mask[lonely_genes_indices] = True

    LOG.debug('  noisy lonely gene names: %s',
              sorted(list(adata.var_names[noisy_lonely_genes_mask])))

    if inplace:
        ut.set_v_data(adata, 'noisy_lonely_gene', noisy_lonely_genes_mask)
        return None

    ut.log_mask(LOG, level, 'noisy_lonely_genes', noisy_lonely_genes_mask)

    return ut.to_pandas_series(noisy_lonely_genes_mask, index=adata.var_names)
