"""
Deviants
--------
"""

import sys
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore
from scipy import sparse  # type: ignore
from scipy import stats

import metacells.parameters as pr
import metacells.utilities as ut

if "sphinx" not in sys.argv[0]:
    import metacells.extensions as xt  # type: ignore

__all__ = [
    "find_deviant_cells",
]


@ut.logged(candidates=ut.groups_description)
@ut.timed_call()
@ut.expand_doc()
def find_deviant_cells(  # pylint: disable=too-many-statements
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    candidates: Union[str, ut.Vector] = "candidate",
    min_gene_fold_factor: float = pr.deviants_min_gene_fold_factor,
    abs_folds: bool = pr.deviants_abs_folds,
    max_gene_fraction: Optional[float] = pr.deviants_max_gene_fraction,
    max_cell_fraction: Optional[float] = pr.deviants_max_cell_fraction,
    inplace: bool = True,
) -> Optional[Tuple[ut.PandasSeries, ut.PandasSeries]]:
    """
    Find cells which are have significantly different gene expression from the metacells they are
    belong to based on ``what`` (default: {what}) data.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

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

    **Computation Parameters**

    Intuitively, we first select some fraction of the genes which were least predictable compared to
    the mean expression in the candidate metacells. We then mark as deviants some fraction of the
    cells whose expression of these genes was least predictable compared to the mean expression in
    the candidate metacells. Operationally:

    1. Compute for each candidate metacell the mean fraction of the UMIs expressed by each gene.
       Scale this by each cell's total UMIs to compute the expected number of UMIs for each cell.
       Compute the fold factor log2((actual UMIs + 1) / (expected UMIs + 1)) for each gene for each
       cell.

    2. Ignore all fold factors less than the ``min_gene_fold_factor`` (default: {min_gene_fold_factor}). If
       ``abs_folds`` (default: {abs_folds}), consider the absolute fold factors. Count the number of genes which have a
       fold factor above this minimum in at least one cell. If the fraction of such genes is above ``max_gene_fraction``
       (default: {max_gene_fraction}), then raise the minimal gene fold factor such that at most this fraction of genes
       remain.

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

    6. If the total fraction of deviants is below the ``max_cell_fraction``, repeats steps 1-5 to account for the mean
       expression in each candidate metacell having been modified due to the removal of deviant cells.
    """
    if max_gene_fraction is None:
        max_gene_fraction = 1

    if max_cell_fraction is None:
        max_cell_fraction = 1

    assert min_gene_fold_factor > 0
    assert 0 < max_gene_fraction < 1
    assert 0 < max_cell_fraction < 1

    cells_count, genes_count = adata.shape
    assert cells_count > 0

    candidate_of_cells = ut.get_o_numpy(adata, candidates, formatter=ut.groups_description).copy()

    totals_of_cells = ut.get_o_numpy(adata, what, sum=True)
    assert totals_of_cells.size == cells_count

    data = ut.get_vo_proper(adata, what, layout="row_major")

    full_indices_of_remaining_cells = np.arange(cells_count)
    full_votes_of_deviant_cells = np.zeros(cells_count, dtype="int32")
    votes_of_deviant_genes = np.zeros(genes_count, dtype="int32")
    max_deviant_cells_count = cells_count * max_cell_fraction

    while True:
        ut.log_calc("max_deviant_cells_count", max_deviant_cells_count)
        acceptable_cells_mask = np.zeros(cells_count, dtype="bool")
        list_of_fold_factors, list_of_cell_index_of_rows = _collect_fold_factors(
            data=data,
            candidate_of_cells=candidate_of_cells,
            totals_of_cells=totals_of_cells,
            min_gene_fold_factor=min_gene_fold_factor,
            abs_folds=abs_folds,
            acceptable_cells_mask=acceptable_cells_mask,
        )

        fold_factors = _construct_fold_factors(cells_count, list_of_fold_factors, list_of_cell_index_of_rows)
        ut.log_calc("fold_factors_nnz", fold_factors.nnz)
        if fold_factors.nnz == 0:
            break

        deviant_gene_indices = _filter_genes(
            cells_count=cells_count,
            genes_count=genes_count,
            fold_factors=fold_factors,
            min_gene_fold_factor=min_gene_fold_factor,
            max_gene_fraction=max_gene_fraction,
        )

        deviant_genes_fold_ranks = _fold_ranks(
            cells_count=cells_count, fold_factors=fold_factors, deviant_gene_indices=deviant_gene_indices
        )

        votes_of_deviant_cells, did_reach_max_deviant_cells = _filter_cells(
            cells_count=cells_count,
            deviant_genes_fold_ranks=deviant_genes_fold_ranks,
            deviant_gene_indices=deviant_gene_indices,
            votes_of_deviant_genes=votes_of_deviant_genes,
            max_deviant_cells_count=max_deviant_cells_count,
        )

        full_votes_of_deviant_cells[full_indices_of_remaining_cells] = votes_of_deviant_cells
        remaining_cells_mask = votes_of_deviant_cells == 0

        ut.log_calc("did_reach_max_deviant_cells", did_reach_max_deviant_cells)
        ut.log_calc("surviving_cells_mask", remaining_cells_mask)

        if did_reach_max_deviant_cells or np.all(remaining_cells_mask):
            break

        max_deviant_cells_count -= np.sum(~remaining_cells_mask)
        assert max_deviant_cells_count > 0

        ut.log_calc("acceptable_cells_mask", acceptable_cells_mask)
        remaining_cells_mask &= ~acceptable_cells_mask
        ut.log_calc("remaining_cells_mask", remaining_cells_mask)
        if not np.any(remaining_cells_mask):
            break

        data = data[remaining_cells_mask, :]
        candidate_of_cells = candidate_of_cells[remaining_cells_mask]
        totals_of_cells = totals_of_cells[remaining_cells_mask]
        full_indices_of_remaining_cells = full_indices_of_remaining_cells[remaining_cells_mask]
        cells_count = len(full_indices_of_remaining_cells)

    if inplace:
        ut.set_v_data(adata, "gene_deviant_votes", votes_of_deviant_genes, formatter=ut.mask_description)
        ut.set_o_data(adata, "cell_deviant_votes", full_votes_of_deviant_cells, formatter=ut.mask_description)
        return None

    ut.log_return("gene_deviant_votes", votes_of_deviant_genes, formatter=ut.mask_description)
    ut.log_return("cell_deviant_votes", full_votes_of_deviant_cells, formatter=ut.mask_description)

    return (
        ut.to_pandas_series(votes_of_deviant_cells, index=adata.obs_names),
        ut.to_pandas_series(votes_of_deviant_genes, index=adata.var_names),
    )


@ut.timed_call()
def _collect_fold_factors(  # pylint: disable=too-many-statements
    *,
    data: ut.ProperMatrix,
    candidate_of_cells: ut.NumpyVector,
    totals_of_cells: ut.NumpyVector,
    min_gene_fold_factor: float,
    abs_folds: bool,
    acceptable_cells_mask: ut.NumpyVector,
) -> Tuple[List[ut.CompressedMatrix], List[ut.NumpyVector]]:
    list_of_fold_factors: List[ut.CompressedMatrix] = []
    list_of_cell_index_of_rows: List[ut.NumpyVector] = []

    cells_count, genes_count = data.shape
    candidates_count = np.max(candidate_of_cells) + 1

    ut.timed_parameters(candidates=candidates_count, cells=cells_count, genes=genes_count)
    remaining_cells_count = cells_count

    for candidate_index in range(candidates_count):
        candidate_cell_indices = np.where(candidate_of_cells == candidate_index)[0]

        candidate_cells_count = candidate_cell_indices.size
        if candidate_cells_count == 0:
            continue

        list_of_cell_index_of_rows.append(candidate_cell_indices)
        remaining_cells_count -= candidate_cells_count

        if candidate_cells_count < 2:
            compressed = sparse.csr_matrix(
                ([], [], [0] * (candidate_cells_count + 1)), shape=(candidate_cells_count, genes_count)
            )
            list_of_fold_factors.append(compressed)
            assert compressed.has_sorted_indices
            assert compressed.has_canonical_format
            continue

        data_of_candidate: ut.ProperMatrix = data[candidate_cell_indices, :].copy()
        assert ut.is_layout(data_of_candidate, "row_major")
        assert data_of_candidate.shape == (candidate_cells_count, genes_count)

        totals_of_candidate_cells = totals_of_cells[candidate_cell_indices]

        totals_of_candidate_genes = ut.sum_per(ut.to_layout(data_of_candidate, "column_major"), per="column")
        assert totals_of_candidate_genes.size == genes_count

        fractions_of_candidate_genes = ut.to_numpy_vector(totals_of_candidate_genes / np.sum(totals_of_candidate_genes))

        _, dense, compressed = ut.to_proper_matrices(data_of_candidate)

        if compressed is not None:
            if compressed.nnz == 0:
                list_of_fold_factors.append(compressed)
                continue

            extension_name = "fold_factor_compressed_%s_t_%s_t_%s_t" % (  # pylint: disable=consider-using-f-string
                compressed.data.dtype,
                compressed.indices.dtype,
                compressed.indptr.dtype,
            )
            extension = getattr(xt, extension_name)

            with ut.timed_step("extensions.fold_factor_compressed"):
                extension(
                    compressed.data,
                    compressed.indices,
                    compressed.indptr,
                    min_gene_fold_factor,
                    abs_folds,
                    totals_of_candidate_cells,
                    fractions_of_candidate_genes,
                )

            ut.eliminate_zeros(compressed)

        else:
            assert dense is not None

            extension_name = f"fold_factor_dense_{dense.dtype}_t"
            extension = getattr(xt, extension_name)

            with ut.timed_step("extensions.fold_factor_dense"):
                extension(
                    dense,
                    min_gene_fold_factor,
                    abs_folds,
                    totals_of_candidate_cells,
                    fractions_of_candidate_genes,
                )

            compressed = sparse.csr_matrix(dense)
            assert compressed.has_sorted_indices
            assert compressed.has_canonical_format

        if compressed.nnz == 0:
            acceptable_cells_mask[candidate_cell_indices] = True
        list_of_fold_factors.append(compressed)

    if remaining_cells_count > 0:
        assert remaining_cells_count == np.sum(candidate_of_cells < 0)
        list_of_cell_index_of_rows.append(np.where(candidate_of_cells < 0)[0])
        compressed = sparse.csr_matrix(
            ([], [], [0] * (remaining_cells_count + 1)), shape=(remaining_cells_count, genes_count)
        )
        assert compressed.has_sorted_indices
        assert compressed.has_canonical_format
        list_of_fold_factors.append(compressed)

    return list_of_fold_factors, list_of_cell_index_of_rows


@ut.timed_call()
def _construct_fold_factors(
    cells_count: int,
    list_of_fold_factors: List[ut.CompressedMatrix],
    list_of_cell_index_of_rows: List[ut.NumpyVector],
) -> ut.CompressedMatrix:
    cell_index_of_rows = np.concatenate(list_of_cell_index_of_rows)
    if cell_index_of_rows.size == 0:
        return sparse.csr_matrix((0, 0))

    cell_row_of_indices = np.empty_like(cell_index_of_rows)
    cell_row_of_indices[cell_index_of_rows] = np.arange(cells_count)

    fold_factors = sparse.vstack(list_of_fold_factors, format="csr")
    fold_factors = fold_factors[cell_row_of_indices, :]
    if fold_factors.nnz > 0:
        fold_factors = ut.to_layout(fold_factors, "column_major")
    return fold_factors


@ut.timed_call()
def _filter_genes(
    *,
    cells_count: int,
    genes_count: int,
    fold_factors: ut.CompressedMatrix,
    min_gene_fold_factor: float,
    max_gene_fraction: Optional[float] = None,
) -> ut.NumpyVector:
    ut.timed_parameters(cells=cells_count, genes=genes_count, fold_factors=fold_factors.nnz)
    max_fold_factors_of_genes = ut.max_per(fold_factors, per="column")
    assert max_fold_factors_of_genes.size == genes_count

    mask_of_deviant_genes = max_fold_factors_of_genes >= min_gene_fold_factor
    deviant_gene_fraction = np.sum(mask_of_deviant_genes) / genes_count

    if max_gene_fraction is not None and deviant_gene_fraction > max_gene_fraction:
        if ut.logging_calc():
            ut.log_calc("candidate_deviant_genes", mask_of_deviant_genes)

        quantile_gene_fold_factor = np.quantile(max_fold_factors_of_genes, 1 - max_gene_fraction)
        assert quantile_gene_fold_factor is not None
        ut.log_calc("quantile_gene_fold_factor", quantile_gene_fold_factor)

        if quantile_gene_fold_factor > min_gene_fold_factor:
            min_gene_fold_factor = quantile_gene_fold_factor
            mask_of_deviant_genes = max_fold_factors_of_genes >= min_gene_fold_factor

            fold_factors.data[fold_factors.data < min_gene_fold_factor] = 0
            ut.eliminate_zeros(fold_factors)

    if ut.logging_calc():
        ut.log_calc("deviant_genes", mask_of_deviant_genes)

    deviant_gene_indices = np.where(mask_of_deviant_genes)[0]
    return deviant_gene_indices


@ut.timed_call()
def _fold_ranks(
    *,
    cells_count: int,
    fold_factors: ut.CompressedMatrix,
    deviant_gene_indices: ut.NumpyVector,
) -> ut.NumpyMatrix:
    assert fold_factors.getformat() == "csc"

    deviant_genes_count = deviant_gene_indices.size

    ut.timed_parameters(cells=cells_count, deviant_genes=deviant_genes_count)

    deviant_genes_fold_ranks = np.full((cells_count, deviant_genes_count), cells_count, order="F")
    assert ut.is_layout(deviant_genes_fold_ranks, "column_major")

    for deviant_gene_index, gene_index in enumerate(deviant_gene_indices):
        gene_start_offset = fold_factors.indptr[gene_index]
        gene_stop_offset = fold_factors.indptr[gene_index + 1]

        gene_fold_factors = fold_factors.data[gene_start_offset:gene_stop_offset]
        gene_suspect_cell_indices = fold_factors.indices[gene_start_offset:gene_stop_offset]

        gene_fold_ranks = stats.rankdata(gene_fold_factors, method="min")
        gene_fold_ranks *= -1
        gene_fold_ranks += gene_fold_ranks.size + 1

        deviant_genes_fold_ranks[gene_suspect_cell_indices, deviant_gene_index] = gene_fold_ranks

    return deviant_genes_fold_ranks


@ut.timed_call()
def _filter_cells(
    *,
    cells_count: int,
    deviant_genes_fold_ranks: ut.NumpyMatrix,
    deviant_gene_indices: ut.NumpyVector,
    votes_of_deviant_genes: ut.NumpyVector,
    max_deviant_cells_count: float,
) -> Tuple[ut.NumpyVector, bool]:
    min_fold_ranks_of_cells = np.min(deviant_genes_fold_ranks, axis=1)
    assert min_fold_ranks_of_cells.size == cells_count

    threshold_cells_fold_rank = cells_count

    mask_of_deviant_cells = min_fold_ranks_of_cells < threshold_cells_fold_rank
    deviant_cells_count = sum(mask_of_deviant_cells)

    ut.log_calc("deviant_cells", mask_of_deviant_cells)

    did_reach_max_deviant_cells = deviant_cells_count >= max_deviant_cells_count
    if did_reach_max_deviant_cells:
        quantile_cells_fold_rank = np.quantile(min_fold_ranks_of_cells, max_deviant_cells_count / cells_count)
        assert quantile_cells_fold_rank is not None

        ut.log_calc("quantile_cells_fold_rank", quantile_cells_fold_rank)

        if quantile_cells_fold_rank < threshold_cells_fold_rank:
            threshold_cells_fold_rank = quantile_cells_fold_rank

    threshold_cells_fold_rank = max(threshold_cells_fold_rank, 2)

    ut.log_calc("threshold_cells_fold_rank", threshold_cells_fold_rank)
    deviant_votes = deviant_genes_fold_ranks < threshold_cells_fold_rank

    votes_of_deviant_cells = ut.sum_per(ut.to_layout(deviant_votes, "row_major"), per="row")
    assert votes_of_deviant_cells.size == cells_count

    used_votes_of_deviant_genes = ut.sum_per(deviant_votes, per="column")
    assert used_votes_of_deviant_genes.size == deviant_gene_indices.size

    votes_of_deviant_genes[deviant_gene_indices] += used_votes_of_deviant_genes

    return votes_of_deviant_cells, did_reach_max_deviant_cells
