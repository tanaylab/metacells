'''
Outliers
--------
'''

import logging
from typing import List, Optional, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy.sparse as sparse  # type: ignore
import scipy.stats as stats  # type: ignore
from anndata import AnnData

import metacells.extensions as xt  # type: ignore
import metacells.parameters as pr
import metacells.preprocessing as pp
import metacells.utilities as ut

__all__ = [
    'find_outlier_cells',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def find_outlier_cells(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    candidate_metacells: Union[str, ut.Vector] = 'candidate_metacell',
    min_gene_fold_factor: float = pr.outliers_min_gene_fold_factor,
    max_gene_fraction: float = pr.outliers_max_gene_fraction,
    max_cell_fraction: float = pr.outliers_max_cell_fraction,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Find cells which are outliers in the metacells they are belong to based ``of`` some data (by
    default, the focus).

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

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
    the mean expression in the candidate metacells. We then mark as outliers some fraction of the
    cells whose expression of these genes was least predictable compared to the mean expression in
    the candidate metacells. Operationally:

    1. Compute for each candidate metacell the mean fraction of the UMIs expressed by each gene.
       Scale this by each cell's total UMIs to compute the expected number of UMIs for each cell.
       Compute the fold factor log2((actual UMIs + 1) / (expected UMIs + 1)) for each gene for each
       cell.

    2. Ignore all fold factors less than the ``min_gene_fold_factor`` (default:
       {min_gene_fold_factor}). Count the number of genes which have a fold factor above this
       minimum in at least one cell. If the fraction of such genes is above ``max_gene_fraction``
       (default: {max_gene_fraction}), then raise the minimal gene fold factor such that at most
       this fraction of genes remain.

    3. For each remaining gene, rank all the cells where it is expressed above the min fold
       factor. Give an artificial maximum rank to all cells with fold factor 0, that is, below the
       minimum.

    4. For each cell, compute the minimal rank it has in any of these genes. That is, if a cell has
       a rank of 1, it means that it has at least one gene whose expression fold factor is the worst
       (highest) across all cells (and is also above the minimum).

    5. Select as outliers all cells whose minimal rank is below the artificial maximum rank, that
       is, which contain at least one gene whose expression fold factor is high relative to the rest
       of the cells. If the fraction of such cells is higher than ``max_cell_fraction`` (default:
       {max_cell_fraction}), reduce the maximal rank such that at most this fraction of cells are
       selected as outliers.
    '''
    assert min_gene_fold_factor > 0
    assert 0 < max_gene_fraction < 1
    assert 0 < max_cell_fraction < 1

    of, level = ut.log_operation(LOG, adata, 'find_outlier_cells', of)

    with ut.focus_on(ut.get_vo_data, adata, of, layout='row_major',
                     intermediate=intermediate) as data:
        cells_count, genes_count = data.shape

        candidate_metacell_of_cells = \
            ut.get_vector_parameter_data(LOG, level, adata, candidate_metacells,
                                         per='o', name='candidate_metacells')
        assert candidate_metacell_of_cells is not None
        assert candidate_metacell_of_cells.size == cells_count

        totals_of_cells = pp.get_per_obs(adata, ut.sum_per).proper
        assert totals_of_cells.size == cells_count

        LOG.log(level, '  min_gene_fold_factor: %s', min_gene_fold_factor)

        list_of_fold_factors, list_of_cell_index_of_rows = \
            _collect_fold_factors(data=data,
                                  candidate_metacell_of_cells=candidate_metacell_of_cells,
                                  totals_of_cells=totals_of_cells,
                                  min_gene_fold_factor=min_gene_fold_factor)

        fold_factors = _construct_fold_factors(cells_count,
                                               list_of_fold_factors,
                                               list_of_cell_index_of_rows)

        outlier_gene_indices = \
            _filter_genes(cells_count=cells_count,
                          genes_count=genes_count,
                          fold_factors=fold_factors,
                          min_gene_fold_factor=min_gene_fold_factor,
                          max_gene_fraction=max_gene_fraction)

        if intermediate:
            ut.set_vo_data(adata, 'fold_factors', fold_factors, ut.NEVER_SAFE)

        outlier_genes_fold_ranks = \
            _fold_ranks(cells_count=cells_count,
                        fold_factors=fold_factors,
                        outlier_gene_indices=outlier_gene_indices)

        mask_of_outlier_cells = \
            _filter_cells(cells_count=cells_count,
                          outlier_genes_fold_ranks=outlier_genes_fold_ranks,
                          max_cell_fraction=max_cell_fraction)

    if inplace:
        ut.set_o_data(adata, 'outlier_cells',
                      mask_of_outlier_cells, ut.NEVER_SAFE)
        return None

    ut.log_mask(LOG, level, 'outlier_cells', mask_of_outlier_cells)

    return pd.Series(mask_of_outlier_cells, index=adata.obs_names)


@ut.timed_call('.collect_fold_factors')
def _collect_fold_factors(
    *,
    data: ut.ProperMatrix,
    candidate_metacell_of_cells: ut.DenseVector,
    totals_of_cells: ut.DenseVector,
    min_gene_fold_factor: float,
) -> Tuple[List[ut.CompressedMatrix], List[ut.DenseVector]]:
    list_of_fold_factors: List[ut.CompressedMatrix] = []
    list_of_cell_index_of_rows: List[ut.DenseVector] = []

    cells_count, genes_count = data.shape
    candidate_metacells_count = np.max(candidate_metacell_of_cells) + 1

    ut.timed_parameters(candidate_metacells=candidate_metacells_count,
                        cells=cells_count, genes=genes_count)
    remaining_cells_count = cells_count

    for candidate_metacell_index in range(candidate_metacells_count):
        candidate_metacell_cell_indices = \
            np.where(candidate_metacell_of_cells ==
                     candidate_metacell_index)[0]

        candidate_metacell_cells_count = candidate_metacell_cell_indices.size
        assert candidate_metacell_cells_count > 0

        list_of_cell_index_of_rows.append(candidate_metacell_cell_indices)
        remaining_cells_count -= candidate_metacell_cells_count

        totals_of_candidate_metacell_cells = \
            totals_of_cells[candidate_metacell_cell_indices]

        data_of_candidate_metacell = \
            data[candidate_metacell_cell_indices, :].copy()
        assert ut.matrix_layout(data_of_candidate_metacell) == 'row_major'

        totals_of_candidate_metacell_genes = \
            ut.sum_per(data_of_candidate_metacell, per='column')
        assert totals_of_candidate_metacell_genes.size == genes_count

        fractions_of_candidate_metacell_genes = ut.to_dense_vector(totals_of_candidate_metacell_genes
                                                                   / np.sum(totals_of_candidate_metacell_genes))

        extension_name = 'fold_factor_%s_t_%s_t_%s_t' \
            % (data_of_candidate_metacell.data.dtype,
               data_of_candidate_metacell.indices.dtype,
               data_of_candidate_metacell.indptr.dtype)
        extension = getattr(xt, extension_name)

        with ut.timed_step('extensions.fold_factor'):
            extension(data_of_candidate_metacell.data,
                      data_of_candidate_metacell.indices,
                      data_of_candidate_metacell.indptr,
                      data_of_candidate_metacell.shape[1],
                      min_gene_fold_factor,
                      totals_of_candidate_metacell_cells,
                      fractions_of_candidate_metacell_genes)

        ut.eliminate_zeros(data_of_candidate_metacell)

        list_of_fold_factors.append(data_of_candidate_metacell)

    assert remaining_cells_count == 0
    return list_of_fold_factors, list_of_cell_index_of_rows


@ ut.timed_call('.construct_fold_factors')
def _construct_fold_factors(
    cells_count: int,
    list_of_fold_factors: List[ut.CompressedMatrix],
    list_of_cell_index_of_rows: List[ut.DenseVector],
) -> ut.CompressedMatrix:
    cell_index_of_rows = np.concatenate(list_of_cell_index_of_rows)
    cell_row_of_indices = np.empty_like(cell_index_of_rows)
    cell_row_of_indices[cell_index_of_rows] = np.arange(cells_count)

    fold_factors = sparse.vstack(list_of_fold_factors, format='csr')
    fold_factors = fold_factors[cell_row_of_indices, :]
    fold_factors = ut.to_layout(fold_factors, 'column_major')

    return fold_factors


@ ut.timed_call('.filter_genes')
def _filter_genes(
    *,
    cells_count: int,
    genes_count: int,
    fold_factors: ut.CompressedMatrix,
    min_gene_fold_factor: float,
    max_gene_fraction: Optional[float] = None,
) -> ut.DenseVector:
    ut.timed_parameters(cells=cells_count, genes=genes_count,
                        fold_factors=fold_factors.nnz)
    max_fold_factors_of_genes = ut.max_per(fold_factors, per='column')
    assert max_fold_factors_of_genes.size == genes_count

    mask_of_outlier_genes = max_fold_factors_of_genes >= min_gene_fold_factor
    outlier_gene_fraction = np.sum(mask_of_outlier_genes) / genes_count

    LOG.debug('  outlier_gene_fraction: %s',
              ut.fraction_description(outlier_gene_fraction))

    if max_gene_fraction is not None \
            and outlier_gene_fraction > max_gene_fraction:
        quantile_gene_fold_factor = np.quantile(max_fold_factors_of_genes,
                                                1 - max_gene_fraction)
        assert quantile_gene_fold_factor is not None

        LOG.debug('  max_gene_fraction: %s',
                  ut.fraction_description(max_gene_fraction))
        LOG.debug('  quantile_gene_fold_factor: %s', quantile_gene_fold_factor)

        if quantile_gene_fold_factor > min_gene_fold_factor:
            min_gene_fold_factor = quantile_gene_fold_factor
            mask_of_outlier_genes = max_fold_factors_of_genes >= min_gene_fold_factor

            fold_factors.data[fold_factors.data < min_gene_fold_factor] = 0
            ut.eliminate_zeros(fold_factors)

    outlier_gene_indices = np.where(mask_of_outlier_genes)[0]
    LOG.debug('  outlier_genes: %s',
              ut.ratio_description(outlier_gene_indices.size,
                                   genes_count))

    return outlier_gene_indices


@ ut.timed_call('.fold_ranks')
def _fold_ranks(
    *,
    cells_count: int,
    fold_factors: ut.CompressedMatrix,
    outlier_gene_indices: ut.DenseVector,
) -> ut.DenseMatrix:
    assert fold_factors.getformat() == 'csc'

    outlier_genes_count = outlier_gene_indices.size

    ut.timed_parameters(cells=cells_count, outlier_genes=outlier_genes_count)

    outlier_genes_fold_ranks = \
        np.full((cells_count, outlier_genes_count), cells_count, order='F')
    assert ut.matrix_layout(outlier_genes_fold_ranks) == 'column_major'

    for outlier_gene_index, gene_index in enumerate(outlier_gene_indices):
        gene_start_offset = fold_factors.indptr[gene_index]
        gene_stop_offset = fold_factors.indptr[gene_index + 1]

        gene_fold_factors = fold_factors.data[gene_start_offset:gene_stop_offset]
        gene_suspect_cell_indices = fold_factors.indices[gene_start_offset:gene_stop_offset]

        gene_fold_ranks = stats.rankdata(gene_fold_factors, method='min')
        gene_fold_ranks *= -1
        gene_fold_ranks += gene_fold_ranks.size + 1

        outlier_genes_fold_ranks[gene_suspect_cell_indices,
                                 outlier_gene_index] = gene_fold_ranks

    return outlier_genes_fold_ranks


@ ut.timed_call('.filter_cells')
def _filter_cells(
    *,
    cells_count: int,
    outlier_genes_fold_ranks: ut.DenseMatrix,
    max_cell_fraction: Optional[float],
) -> ut.DenseVector:
    min_fold_ranks_of_cells = np.min(outlier_genes_fold_ranks, axis=1)
    assert min_fold_ranks_of_cells.size == cells_count

    mask_of_outlier_cells = min_fold_ranks_of_cells < cells_count
    outliers_cells_count = sum(mask_of_outlier_cells)
    outlier_cell_fraction = outliers_cells_count / cells_count

    LOG.debug('  outlier_cells: %s',
              ut.ratio_description(outliers_cells_count, cells_count))

    if max_cell_fraction is not None \
            and outlier_cell_fraction > max_cell_fraction:
        LOG.debug('  max_cell_fraction: %s',
                  ut.fraction_description(max_cell_fraction))
        quantile_cells_fold_rank = np.quantile(min_fold_ranks_of_cells,
                                               max_cell_fraction)
        assert quantile_cells_fold_rank is not None

        LOG.debug('  quantile_cells_fold_rank: %s', quantile_cells_fold_rank)

        if quantile_cells_fold_rank < cells_count:
            mask_of_outlier_cells = min_fold_ranks_of_cells < quantile_cells_fold_rank

        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug('  outlier_cells: %s',
                      ut.mask_description(mask_of_outlier_cells))

    return mask_of_outlier_cells
