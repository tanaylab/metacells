'''
Find cells which are outliers in the metacells they are belong to.
'''

import logging
from typing import List, Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy.sparse as sparse  # type: ignore
import scipy.stats as stats  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'find_outlier_cells',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def find_outlier_cells(  # pylint: disable=too-many-locals,too-many-statements
    adata: AnnData,
    *,
    of: Optional[str] = None,
    communities: Union[str, ut.Vector] = 'candidate_metacell',
    minimal_gene_fold_factor: float = 3.0,
    maximal_genes_fraction: float = 0.03,
    maximal_cells_fraction: float = 0.25,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Find cells which are outliers in the metacells they are belong to based ``of`` some data (by
    default, the focus).

    **Input**

    A :py:func:`metacells.utilities.preparation.prepare`-ed annotated ``adata``, where the
    observations are cells and the variables are genes.

    **Returns**

    Observation (Cell) Annotations
        ``outlier_cells``
            A boolean mask indicating whether each cell was found to be an outlier in the metacells
            it belongs to.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    Intuitively, we first select some fraction of the genes which were least predictable compared to
    the mean expression in the communities. We then mark as outliers some fraction of the cells
    whose expression of these genes was least predictable compared to the mean expression in the
    communities. Operationally:

    1. Compute for each community the mean fraction of the UMIs expressed by each gene.
       Scale this by each cell's total UMIs to compute the expected number of UMIs for each cell.
       Compute the fold factor log2((actual UMIs + 1) / (expected UMIs + 1)) for each gene for
       each cell.

    2. Ignore all fold factors less than the ``minimal_gene_fold_factor`` (default:
       {minimal_gene_fold_factor}). Count the number of genes which have a fold factor above this
       minimum in at least one cell. If the fraction of such genes is above
       ``maximal_genes_fraction`` (default: {maximal_genes_fraction}), then raise the minimal gene
       fold factor such that at most this fraction of genes remain.

    3. For each remaining gene, rank all the cells where it is expressed above the minimal fold
       factor. Give an artificial maximal rank to all cells with fold factor 0, that is, below the
       minimum.

    4. For each cell, compute the minimal rank it has in any of these genes. That is, if a cell has
       a rank of 1, it means that it has at least one gene whose expression fold factor is the worst
       (highest) across all cells (and is also above the minimum).

    5. Select as outliers all cells whose minimal rank is below the artificial maximal rank, that
       is, which contain at least one gene whose expression fold factor is high relative to the rest
       of the cells. If the fraction of such cells is higher than ``maximal_cells_fraction``
       (default: {maximal_cells_fraction}), reduce the maximal rank such that at most this fraction
       of cells are selected as outliers.
    '''
    assert minimal_gene_fold_factor > 0
    assert 0 < maximal_genes_fraction < 1
    assert 0 < maximal_cells_fraction < 1

    LOG.debug('find_outlier_cells...')

    with ut.focus_on(ut.get_vo_data, adata, of, layout='row_major',
                     intermediate=intermediate) as data:
        cells_count, genes_count = data.shape

        if isinstance(communities, str):
            community_of_cells = ut.get_o_data(adata, communities)
        else:
            community_of_cells = communities

        community_of_cells = ut.to_dense_vector(community_of_cells)
        assert community_of_cells.size == cells_count

        totals_of_cells = ut.get_per_obs(adata, ut.sum_per).proper
        assert totals_of_cells.size == cells_count

        communities_count = np.max(community_of_cells) + 1

        LOG.debug('  minimal_gene_fold_factor: %s', minimal_gene_fold_factor)

        with ut.timed_step('.fold_factors'):
            with ut.timed_step('.collect'):
                list_of_fold_factors: List[ut.CompressedMatrix] = []
                list_of_cell_index_of_rows: List[ut.DenseVector] = []

                ut.timed_parameters(communities=communities_count,
                                    cells=cells_count, genes=genes_count)
                remaining_cells_count = cells_count

                for community_index in range(communities_count):
                    community_cell_indices = \
                        np.where(community_of_cells == community_index)[0]

                    community_cells_count = community_cell_indices.size
                    assert community_cells_count > 0

                    list_of_cell_index_of_rows.append(community_cell_indices)
                    remaining_cells_count -= community_cells_count

                    totals_of_community_cells = totals_of_cells[community_cell_indices]

                    data_of_community = \
                        ut.to_dense_matrix(data[community_cell_indices, :],
                                           copy=True)

                    totals_of_community_genes = data_of_community.sum(axis=0)
                    assert totals_of_community_genes.size == genes_count

                    fractions_of_community_genes = \
                        totals_of_community_genes / \
                        np.sum(totals_of_community_genes)

                    expected_data_of_community = \
                        np.outer(totals_of_community_cells,
                                 fractions_of_community_genes)
                    assert expected_data_of_community.shape == data_of_community.shape

                    expected_data_of_community += 1
                    data_of_community += 1
                    data_of_community /= expected_data_of_community
                    np.log2(data_of_community, out=data_of_community)
                    data_of_community[data_of_community
                                      < minimal_gene_fold_factor] = 0

                    list_of_fold_factors.append(  #
                        sparse.csr_matrix(data_of_community))

                assert remaining_cells_count == 0

            with ut.timed_step('.construct'):
                cell_index_of_rows = np.concatenate(list_of_cell_index_of_rows)
                cell_row_of_indices = np.empty_like(cell_index_of_rows)
                cell_row_of_indices[cell_index_of_rows] = \
                    np.arange(cells_count)

                fold_factors = \
                    sparse.vstack(list_of_fold_factors, format='csr')
                fold_factors = fold_factors[cell_row_of_indices, :]
                fold_factors = ut.to_layout(fold_factors, 'column_major')

        if intermediate:
            adata.layers['fold_factors'] = fold_factors
            ut.safe_slicing_data('fold_factors', ut.NEVER_SAFE)

        with ut.timed_step('.filter_genes'):
            ut.timed_parameters(cells=cells_count, genes=genes_count,
                                fold_factors=fold_factors.nnz)
            maximal_fold_factors_of_genes = \
                ut.to_dense_vector(fold_factors.max(axis=0))
            assert maximal_fold_factors_of_genes.size == genes_count

            mask_of_outlier_genes = \
                maximal_fold_factors_of_genes >= minimal_gene_fold_factor
            outlier_genes_fraction = \
                np.sum(mask_of_outlier_genes) / genes_count

            LOG.debug('  outlier_genes_fraction: %s', outlier_genes_fraction)

            if maximal_genes_fraction is not None \
                    and outlier_genes_fraction > maximal_genes_fraction:
                quantile_gene_fold_factor = \
                    np.quantile(maximal_fold_factors_of_genes,
                                1 - maximal_genes_fraction)
                assert quantile_gene_fold_factor is not None

                LOG.debug('  maximal_genes_fraction: %s',
                          maximal_genes_fraction)
                LOG.debug('  quantile_gene_fold_factor: %s',
                          quantile_gene_fold_factor)

                if quantile_gene_fold_factor > minimal_gene_fold_factor:
                    minimal_gene_fold_factor = quantile_gene_fold_factor
                    mask_of_outlier_genes = \
                        maximal_fold_factors_of_genes >= minimal_gene_fold_factor

                    fold_factors[fold_factors < minimal_gene_fold_factor] = 0
                    with ut.timed_step('eliminate_zeros'):
                        ut.timed_parameters(before=fold_factors.nnz)
                        fold_factors.eliminate_zeros()
                        ut.timed_parameters(after=fold_factors.nnz)

            outlier_gene_indices = np.where(mask_of_outlier_genes)[0]
            outlier_genes_count = outlier_gene_indices.size
            LOG.debug('  outlier_genes_fraction: %s',
                      outlier_genes_count / genes_count)

        with ut.timed_step('.fold_ranks'):
            assert fold_factors.getformat() == 'csc'
            ut.timed_parameters(cells=cells_count,
                                outlier_genes=outlier_genes_count)

            outlier_genes_fold_ranks = \
                np.full((cells_count, outlier_genes_count),
                        cells_count, order='F')
            assert ut.matrix_layout(outlier_genes_fold_ranks) == 'column_major'

            for outlier_gene_index, gene_index in enumerate(outlier_gene_indices):
                gene_start_offset = fold_factors.indptr[gene_index]
                gene_stop_offset = fold_factors.indptr[gene_index + 1]

                gene_fold_factors = fold_factors.data[gene_start_offset:gene_stop_offset]
                gene_suspect_cell_indices = fold_factors.indices[gene_start_offset:gene_stop_offset]

                gene_fold_ranks = \
                    stats.rankdata(gene_fold_factors, method='min')
                gene_fold_ranks *= -1
                gene_fold_ranks += gene_fold_ranks.size + 1

                outlier_genes_fold_ranks[gene_suspect_cell_indices,
                                         outlier_gene_index] = gene_fold_ranks

        with ut.timed_step('.filter_cells'):
            minimal_fold_ranks_of_cells = outlier_genes_fold_ranks.min(axis=1)
            assert minimal_fold_ranks_of_cells.size == cells_count

            mask_of_outlier_cells = minimal_fold_ranks_of_cells < cells_count
            outliers_cells_count = sum(mask_of_outlier_cells)
            outlier_cells_fraction = outliers_cells_count / cells_count

            LOG.debug('  outlier_cells_fraction: %s', outlier_cells_fraction)

            if maximal_cells_fraction is not None \
                    and outlier_cells_fraction > maximal_cells_fraction:
                LOG.debug('  maximal_cells_fraction: %s',
                          maximal_cells_fraction)
                quantile_cells_fold_rank = \
                    np.quantile(minimal_fold_ranks_of_cells,
                                maximal_cells_fraction)
                assert quantile_cells_fold_rank is not None

                LOG.debug('  quantile_cells_fold_rank: %s',
                          quantile_cells_fold_rank)

                if quantile_cells_fold_rank < cells_count:
                    mask_of_outlier_cells = minimal_fold_ranks_of_cells < quantile_cells_fold_rank

                if LOG.isEnabledFor(logging.DEBUG):
                    LOG.debug('  maximal_cells_fraction: %s',
                              np.sum(mask_of_outlier_cells) / cells_count)

    if LOG.isEnabledFor(logging.INFO):
        LOG.info('find_outlier_cells: %s', np.sum(mask_of_outlier_cells))

    if inplace:
        adata.obs['outlier_cells'] = mask_of_outlier_cells
        ut.safe_slicing_data('outlier_cells', ut.NEVER_SAFE)
        return None

    return pd.Series(mask_of_outlier_cells, index=adata.obs_names)
