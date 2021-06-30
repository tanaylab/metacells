'''
Noisy Lonely
------------
'''

from typing import Optional, Union

import numpy as np
from anndata import AnnData

import metacells.parameters as pr
import metacells.utilities as ut
from metacells.tools.downsample import downsample_cells
from metacells.tools.filter import filter_data
from metacells.tools.high import (find_high_normalized_variance_genes,
                                  find_high_total_genes)
from metacells.tools.similarity import compute_var_var_similarity

__all__ = [
    'find_noisy_lonely_genes',
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def find_noisy_lonely_genes(  # pylint: disable=too-many-statements
    adata: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    *,
    excluded_genes_mask: Optional[str] = None,
    max_sampled_cells: int = pr.noisy_lonely_max_sampled_cells,
    downsample_min_samples: int = pr.noisy_lonely_downsample_min_samples,
    downsample_min_cell_quantile: float = pr.noisy_lonely_downsample_max_cell_quantile,
    downsample_max_cell_quantile: float = pr.noisy_lonely_downsample_min_cell_quantile,
    min_gene_total: int = pr.noisy_lonely_min_gene_total,
    min_gene_normalized_variance: float = pr.noisy_lonely_min_gene_normalized_variance,
    max_gene_similarity: float = pr.noisy_lonely_max_gene_similarity,
    random_seed: int = pr.random_seed,
    inplace: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Detect "noisy lonely" genes based on ``what`` (default: {what}) data.

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

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    **Returns**

    Variable (Gene) Annotations
        ``noisy_lonely_genes``
            A boolean mask indicating whether each gene was found to be a "noisy lonely" gene.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    **Computation Parameters**

    1. If we have more than ``max_sampled_cells`` (default: {max_sampled_cells}), pick this number
       of random cells from the data using the ``random_seed``.

    2. If we were specified an ``excluded_genes_mask``, this is the name of a per-variable (gene)
       annotation containing a mask of excluded genes. Get rid of all these excluded genes.

    3. Invoke :py:func:`metacells.tools.downsample.downsample_cells` to downsample the cells to the
       same total number of UMIs, using the ``downsample_min_samples`` (default:
       {downsample_min_samples}), ``downsample_min_cell_quantile`` (default:
       {downsample_min_cell_quantile}), ``downsample_max_cell_quantile`` (default:
       {downsample_max_cell_quantile}) and the ``random_seed`` (default: {random_seed}).

    4. Find "noisy" genes which have a total number of UMIs of at least ``min_gene_total`` (default:
       {min_gene_total}) and a normalized variance of at least ``min_gene_normalized_variance``
       (default: ``min_gene_normalized_variance``).

    5. Cross-correlate the noisy genes.

    6. Find the noisy "lonely" genes whose maximal correlation is at most
       ``max_gene_similarity`` (default: {max_gene_similarity}) with all other genes.
    '''
    if max_sampled_cells < adata.n_obs:
        np.random.seed(random_seed)
        cell_indices = \
            np.random.choice(np.arange(adata.n_obs),
                             size=max_sampled_cells, replace=False)
        s_data = ut.slice(adata, obs=cell_indices,
                          name='.sampled', top_level=False)
    else:
        s_data = ut.copy_adata(adata, top_level=False)

    track_var: Optional[str] = 'sampled_gene_index'

    if excluded_genes_mask is not None:
        results = filter_data(s_data, name='included', top_level=False,
                              track_var=track_var,
                              var_masks=[f'~{excluded_genes_mask}'])
        track_var = None
        assert results is not None
        i_data = results[0]
        assert i_data is not None
    else:
        i_data = s_data

    downsample_cells(i_data, what,
                     downsample_min_samples=downsample_min_samples,
                     downsample_min_cell_quantile=downsample_min_cell_quantile,
                     downsample_max_cell_quantile=downsample_max_cell_quantile,
                     random_seed=random_seed)

    find_high_total_genes(i_data, 'downsampled', min_gene_total=min_gene_total)

    results = filter_data(i_data, name='high_total', top_level=False,
                          track_var=track_var,
                          var_masks=['high_total_gene'])
    track_var = None
    assert results is not None
    ht_data = results[0]

    noisy_lonely_genes_mask = np.full(adata.n_vars, False)

    if ht_data is not None:
        ht_genes_count = ht_data.shape[1]

        ht_gene_ht_gene_similarity_frame = \
            compute_var_var_similarity(ht_data, 'downsampled', inplace=False,
                                       reproducible=(random_seed != 0))
        assert ht_gene_ht_gene_similarity_frame is not None

        ht_gene_ht_gene_similarity_matrix = \
            ut.to_numpy_matrix(ht_gene_ht_gene_similarity_frame,
                               only_extract=True)
        ht_gene_ht_gene_similarity_matrix = \
            ut.to_layout(ht_gene_ht_gene_similarity_matrix,
                         layout='row_major', symmetric=True)
        np.fill_diagonal(ht_gene_ht_gene_similarity_matrix, -1)

        htv_mask_series = \
            find_high_normalized_variance_genes(ht_data, 'downsampled',
                                                min_gene_normalized_variance=min_gene_normalized_variance,
                                                inplace=False)
        assert htv_mask_series is not None
        htv_mask = ut.to_numpy_vector(htv_mask_series)

        htv_genes_count = np.sum(htv_mask)
        assert 0 < htv_genes_count <= ht_genes_count

        htv_gene_ht_gene_similarity_matrix = ht_gene_ht_gene_similarity_matrix[htv_mask, :]
        assert \
            ut.matrix_layout(htv_gene_ht_gene_similarity_matrix) == 'row_major'
        assert htv_gene_ht_gene_similarity_matrix.shape \
            == (htv_genes_count, ht_genes_count)

        max_similarity_of_htv_genes \
            = ut.max_per(htv_gene_ht_gene_similarity_matrix, per='row')
        htvl_mask = max_similarity_of_htv_genes <= max_gene_similarity
        htvl_genes_count = np.sum(htvl_mask)
        ut.log_calc('noisy_lonely_genes_count', htvl_genes_count)

        if htvl_genes_count > 0:
            base_index_of_ht_genes = \
                ut.get_v_numpy(ht_data, 'sampled_gene_index')
            base_index_of_htv_genes = base_index_of_ht_genes[htv_mask]
            base_index_of_htvl_genes = base_index_of_htv_genes[htvl_mask]

            noisy_lonely_genes_mask[base_index_of_htvl_genes] = True

            htvl_gene_ht_gene_similarity_matrix = \
                htv_gene_ht_gene_similarity_matrix[htvl_mask, :]
            htvl_gene_ht_gene_similarity_matrix = \
                ut.to_layout(htvl_gene_ht_gene_similarity_matrix,
                             layout='row_major')
            assert htvl_gene_ht_gene_similarity_matrix.shape \
                == (htvl_genes_count, ht_genes_count)

            if ut.logging_calc():
                i_gene_totals = ut.get_v_numpy(i_data, 'downsampled', sum=True)
                ht_mask = ut.get_v_numpy(i_data, 'high_total_gene')
                i_total = np.sum(i_gene_totals)
                htvl_gene_totals = i_gene_totals[ht_mask][htv_mask][htvl_mask]
                top_similarity_of_htvl_genes = \
                    ut.top_per(htvl_gene_ht_gene_similarity_matrix,
                               10, per='row')
                for htvl_index, gene_index in enumerate(base_index_of_htvl_genes):
                    gene_name = adata.var_names[gene_index]
                    gene_total = htvl_gene_totals[htvl_index]
                    gene_percent = 100 * gene_total / i_total
                    similar_ht_values = ut.to_numpy_vector(  #
                        top_similarity_of_htvl_genes[htvl_index, :])
                    assert len(similar_ht_values) == ht_genes_count
                    top_similar_ht_mask = similar_ht_values > 0
                    top_similar_ht_values = similar_ht_values[top_similar_ht_mask]
                    top_similar_ht_indices = base_index_of_ht_genes[top_similar_ht_mask]
                    top_similar_ht_names = adata.var_names[top_similar_ht_indices]
                    ut.log_calc(f'- {gene_name}',
                                f'total downsampled UMIs: {gene_total} '
                                + f'({gene_percent:.4g}%), correlated with: '
                                + ', '.join([f'{similar_gene_name}: {similar_gene_value:.4g}'
                                             for similar_gene_value, similar_gene_name
                                             in reversed(sorted(zip(top_similar_ht_values, top_similar_ht_names)))]))

    if ut.logging_calc():
        ut.log_calc('noisy_lonely_gene_names',
                    sorted(list(adata.var_names[noisy_lonely_genes_mask])))

    if inplace:
        ut.set_v_data(adata, 'noisy_lonely_gene', noisy_lonely_genes_mask)
        return None

    ut.log_return('noisy_lonely_genes', noisy_lonely_genes_mask)
    return ut.to_pandas_series(noisy_lonely_genes_mask, index=adata.var_names)
