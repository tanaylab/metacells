'''
Deviants
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
    'find_deviant_cells',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def find_deviant_cells(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    candidates: Union[str, ut.Vector] = 'candidate',
    min_gene_fold_factor: float = pr.deviants_min_gene_fold_factor,
    max_gene_fraction: Optional[float] = pr.deviants_max_gene_fraction,
    max_cell_fraction: Optional[float] = pr.deviants_max_cell_fraction,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[Tuple[ut.PandasSeries, ut.PandasSeries]]:
    '''
    Find cells which are have significantly different gene expression from the metacells they are
    belong to based ``of`` some data (by default, the focus).

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Observation (Cell) Annotations
        ``cell_deviant_votes``
            The number of genes that were the reason the cell was marked as deviant (if zero, the
            cell is not deviant).

    Variable (Gene) Annotations
        ``gene_deviant_votes``
            The number of cells each gene marked as deviant (if zero, the gene did not mark any cell
            as deviant).

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as two pandas series (indexed by the observation and
    variable names).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    Intuitively, we first select some fraction of the genes which were least predictable compared to
    the mean expression in the candidate metacells. We then mark as deviants some fraction of the
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

    5. Select as deviants all cells whose minimal rank is below the artificial maximum rank, that
       is, which contain at least one gene whose expression fold factor is high relative to the rest
       of the cells. If the fraction of such cells is higher than ``max_cell_fraction`` (default:
       {max_cell_fraction}), reduce the maximal rank such that at most this fraction of cells are
       selected as deviants.
    '''
    if max_gene_fraction is None:
        max_gene_fraction = 1

    if max_cell_fraction is None:
        max_cell_fraction = 1

    assert min_gene_fold_factor > 0
    assert 0 < max_gene_fraction < 1
    assert 0 < max_cell_fraction < 1

    of, level = ut.log_operation(LOG, adata, 'find_deviant_cells', of)

    with ut.focus_on(ut.get_vo_data, adata, of, layout='row_major',
                     intermediate=intermediate) as data:
        cells_count, genes_count = data.shape
        assert cells_count > 0

        candidate_of_cells = \
            ut.get_vector_parameter_data(LOG, adata, candidates,
                                         per='o', name='candidates')
        assert candidate_of_cells is not None
        assert candidate_of_cells.size == cells_count

        totals_of_cells = pp.get_per_obs(adata, ut.sum_per).dense
        assert totals_of_cells.size == cells_count

        LOG.debug('  min_gene_fold_factor: %s', min_gene_fold_factor)

        list_of_fold_factors, list_of_cell_index_of_rows = \
            _collect_fold_factors(data=data,
                                  candidate_of_cells=candidate_of_cells,
                                  totals_of_cells=totals_of_cells,
                                  min_gene_fold_factor=min_gene_fold_factor)

        fold_factors = _construct_fold_factors(cells_count,
                                               list_of_fold_factors,
                                               list_of_cell_index_of_rows)

        if fold_factors is None:
            votes_of_deviant_cells = np.zeros(adata.n_obs, dtype='int32')
            votes_of_deviant_genes = np.zeros(adata.n_vars, dtype='int32')

        else:
            deviant_gene_indices = \
                _filter_genes(cells_count=cells_count,
                              genes_count=genes_count,
                              fold_factors=fold_factors,
                              min_gene_fold_factor=min_gene_fold_factor,
                              max_gene_fraction=max_gene_fraction)

            if intermediate:
                ut.set_vo_data(adata, 'fold_factor', fold_factors)

            deviant_genes_fold_ranks = \
                _fold_ranks(cells_count=cells_count,
                            fold_factors=fold_factors,
                            deviant_gene_indices=deviant_gene_indices)

            votes_of_deviant_cells, votes_of_deviant_genes = \
                _filter_cells(cells_count=cells_count,
                              genes_count=genes_count,
                              deviant_genes_fold_ranks=deviant_genes_fold_ranks,
                              deviant_gene_indices=deviant_gene_indices,
                              max_cell_fraction=max_cell_fraction)

    if inplace:
        ut.set_o_data(adata, 'cell_deviant_votes', votes_of_deviant_cells,
                      log_value=ut.mask_description)
        ut.set_v_data(adata, 'gene_deviant_votes', votes_of_deviant_genes,
                      log_value=ut.mask_description)
        return None

    ut.log_mask(LOG, level, 'deviant_cells', votes_of_deviant_cells)
    ut.log_mask(LOG, level, 'deviant_genes', votes_of_deviant_genes)

    return pd.Series(votes_of_deviant_cells, index=adata.obs_names), \
        pd.Series(votes_of_deviant_genes, index=adata.var_names)


@ut.timed_call('.collect_fold_factors')
def _collect_fold_factors(
    *,
    data: ut.ProperMatrix,
    candidate_of_cells: ut.DenseVector,
    totals_of_cells: ut.DenseVector,
    min_gene_fold_factor: float,
) -> Tuple[List[ut.CompressedMatrix], List[ut.DenseVector]]:
    list_of_fold_factors: List[ut.CompressedMatrix] = []
    list_of_cell_index_of_rows: List[ut.DenseVector] = []

    cells_count, genes_count = data.shape
    candidates_count = np.max(candidate_of_cells) + 1

    ut.timed_parameters(candidates=candidates_count,
                        cells=cells_count, genes=genes_count)
    remaining_cells_count = cells_count

    for candidate_index in range(candidates_count):
        candidate_cell_indices = \
            np.where(candidate_of_cells == candidate_index)[0]

        candidate_cells_count = candidate_cell_indices.size
        assert candidate_cells_count > 0

        list_of_cell_index_of_rows.append(candidate_cell_indices)
        remaining_cells_count -= candidate_cells_count

        if candidate_cells_count < 2:
            compressed = \
                sparse.csr_matrix(([], [], [0] * (candidate_cells_count + 1)),
                                  shape=(candidate_cells_count, genes_count))
            list_of_fold_factors.append(compressed)
            assert compressed.has_sorted_indices
            assert compressed.has_canonical_format
            continue

        data_of_candidate: ut.ProperMatrix = \
            data[candidate_cell_indices, :].copy()
        assert ut.matrix_layout(data_of_candidate) == 'row_major'
        assert data_of_candidate.shape == (candidate_cells_count, genes_count)

        totals_of_candidate_cells = totals_of_cells[candidate_cell_indices]

        totals_of_candidate_genes = \
            ut.sum_per(ut.to_layout(data_of_candidate, 'column_major'),
                       per='column')
        assert totals_of_candidate_genes.size == genes_count

        fractions_of_candidate_genes = \
            ut.to_dense_vector(totals_of_candidate_genes
                               / np.sum(totals_of_candidate_genes))

        _, dense, compressed = ut.to_proper_matrices(data_of_candidate)

        if compressed is not None:
            if compressed.nnz == 0:
                list_of_fold_factors.append(compressed)
                continue

            extension_name = 'fold_factor_compressed_%s_t_%s_t_%s_t' \
                % (compressed.data.dtype,
                   compressed.indices.dtype,
                   compressed.indptr.dtype)
            extension = getattr(xt, extension_name)

            with ut.timed_step('extensions.fold_factor_compressed'):
                extension(compressed.data,
                          compressed.indices,
                          compressed.indptr,
                          min_gene_fold_factor,
                          totals_of_candidate_cells,
                          fractions_of_candidate_genes)

            ut.eliminate_zeros(compressed)

        else:
            assert dense is not None

            extension_name = 'fold_factor_dense_%s_t' % dense.dtype
            extension = getattr(xt, extension_name)

            with ut.timed_step('extensions.fold_factor_dense'):
                extension(dense,
                          min_gene_fold_factor,
                          totals_of_candidate_cells,
                          fractions_of_candidate_genes)

            compressed = sparse.csr_matrix(dense)
            assert compressed.has_sorted_indices
            assert compressed.has_canonical_format

        list_of_fold_factors.append(compressed)

    assert remaining_cells_count == 0
    return list_of_fold_factors, list_of_cell_index_of_rows


@ ut.timed_call('.construct_fold_factors')
def _construct_fold_factors(
    cells_count: int,
    list_of_fold_factors: List[ut.CompressedMatrix],
    list_of_cell_index_of_rows: List[ut.DenseVector],
) -> Optional[ut.CompressedMatrix]:
    cell_index_of_rows = np.concatenate(list_of_cell_index_of_rows)
    if cell_index_of_rows.size == 0:
        return None

    cell_row_of_indices = np.empty_like(cell_index_of_rows)
    cell_row_of_indices[cell_index_of_rows] = np.arange(cells_count)

    fold_factors = sparse.vstack(list_of_fold_factors, format='csr')
    fold_factors = fold_factors[cell_row_of_indices, :]
    if fold_factors.nnz == 0:
        return None

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

    mask_of_deviant_genes = max_fold_factors_of_genes >= min_gene_fold_factor
    deviant_gene_fraction = np.sum(mask_of_deviant_genes) / genes_count

    LOG.debug('  deviant_gene_fraction: %s',
              ut.fraction_description(deviant_gene_fraction))

    if max_gene_fraction is not None \
            and deviant_gene_fraction > max_gene_fraction:
        quantile_gene_fold_factor = np.quantile(max_fold_factors_of_genes,
                                                1 - max_gene_fraction)
        assert quantile_gene_fold_factor is not None

        LOG.debug('  max_gene_fraction: %s',
                  ut.fraction_description(max_gene_fraction))
        LOG.debug('  quantile_gene_fold_factor: %s', quantile_gene_fold_factor)

        if quantile_gene_fold_factor > min_gene_fold_factor:
            min_gene_fold_factor = quantile_gene_fold_factor
            mask_of_deviant_genes = max_fold_factors_of_genes >= min_gene_fold_factor

            fold_factors.data[fold_factors.data < min_gene_fold_factor] = 0
            ut.eliminate_zeros(fold_factors)

    deviant_gene_indices = np.where(mask_of_deviant_genes)[0]
    LOG.debug('  deviant_genes: %s',
              ut.ratio_description(deviant_gene_indices.size,
                                   genes_count))

    return deviant_gene_indices


@ ut.timed_call('.fold_ranks')
def _fold_ranks(
    *,
    cells_count: int,
    fold_factors: ut.CompressedMatrix,
    deviant_gene_indices: ut.DenseVector,
) -> ut.DenseMatrix:
    assert fold_factors.getformat() == 'csc'

    deviant_genes_count = deviant_gene_indices.size

    ut.timed_parameters(cells=cells_count, deviant_genes=deviant_genes_count)

    deviant_genes_fold_ranks = \
        np.full((cells_count, deviant_genes_count), cells_count, order='F')
    assert ut.matrix_layout(deviant_genes_fold_ranks) == 'column_major'

    for deviant_gene_index, gene_index in enumerate(deviant_gene_indices):
        gene_start_offset = fold_factors.indptr[gene_index]
        gene_stop_offset = fold_factors.indptr[gene_index + 1]

        gene_fold_factors = fold_factors.data[gene_start_offset:gene_stop_offset]
        gene_suspect_cell_indices = fold_factors.indices[gene_start_offset:gene_stop_offset]

        gene_fold_ranks = stats.rankdata(gene_fold_factors, method='min')
        gene_fold_ranks *= -1
        gene_fold_ranks += gene_fold_ranks.size + 1

        deviant_genes_fold_ranks[gene_suspect_cell_indices,
                                 deviant_gene_index] = gene_fold_ranks

    return deviant_genes_fold_ranks


@ ut.timed_call('.filter_cells')
def _filter_cells(
    *,
    cells_count: int,
    genes_count: int,
    deviant_genes_fold_ranks: ut.DenseMatrix,
    deviant_gene_indices: ut.DenseVector,
    max_cell_fraction: Optional[float],
) -> Tuple[ut.DenseVector, ut.DenseVector]:
    min_fold_ranks_of_cells = np.min(deviant_genes_fold_ranks, axis=1)
    assert min_fold_ranks_of_cells.size == cells_count

    threshold_cells_fold_rank = cells_count

    mask_of_deviant_cells = min_fold_ranks_of_cells < threshold_cells_fold_rank
    deviants_cells_count = sum(mask_of_deviant_cells)
    deviant_cell_fraction = deviants_cells_count / cells_count

    LOG.debug('  deviant_cell: %s',
              ut.ratio_description(deviants_cells_count, cells_count))

    if max_cell_fraction is not None \
            and deviant_cell_fraction > max_cell_fraction:
        LOG.debug('  max_cell_fraction: %s',
                  ut.fraction_description(max_cell_fraction))
        quantile_cells_fold_rank = np.quantile(min_fold_ranks_of_cells,
                                               max_cell_fraction)
        assert quantile_cells_fold_rank is not None

        LOG.debug('  quantile_cells_fold_rank: %s', quantile_cells_fold_rank)

        if quantile_cells_fold_rank < threshold_cells_fold_rank:
            threshold_cells_fold_rank = quantile_cells_fold_rank
            mask_of_deviant_cells = min_fold_ranks_of_cells < threshold_cells_fold_rank

        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug('  deviant_cell: %s',
                      ut.mask_description(mask_of_deviant_cells))

    deviant_votes = deviant_genes_fold_ranks < threshold_cells_fold_rank
    votes_of_deviant_cells = np.sum(deviant_votes, axis=1)
    assert votes_of_deviant_cells.size == cells_count
    votes_of_deviant_genes = np.sum(deviant_votes, axis=0)
    assert votes_of_deviant_genes.size == deviant_gene_indices.size

    votes_of_all_genes = np.zeros(genes_count, dtype='int32')
    votes_of_all_genes[deviant_gene_indices] = votes_of_deviant_genes

    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug('  deviant_genes: %s',
                  ut.ratio_description(np.sum(votes_of_all_genes > 0),
                                       genes_count))

    return votes_of_deviant_cells, votes_of_all_genes
