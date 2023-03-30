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
from scipy import sparse as sp  # type: ignore
from scipy import stats as st

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
def find_deviant_cells(  # pylint: disable=too-many-statements,too-many-branches
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    candidates: Union[str, ut.Vector] = "candidate",
    min_gene_fold_factor: float = pr.deviants_min_gene_fold_factor,
    min_compare_umis: int = pr.deviants_min_compare_umis,
    gap_skip_cells: int = pr.deviants_gap_skip_cells,
    min_noisy_gene_fold_factor: float = pr.deviants_min_noisy_gene_fold_factor,
    max_gene_fraction: float = pr.deviants_max_gene_fraction,
    max_cell_fraction: Optional[float] = pr.deviants_max_cell_fraction,
    max_gap_cells_count: int = pr.deviants_max_gap_cells_count,
    max_gap_cells_fraction: float = pr.deviants_max_gap_cells_fraction,
    cells_regularization_quantile: float = pr.deviant_cells_regularization_quantile,
    policy: str = pr.deviants_policy,
) -> ut.Vector:
    """
    Find cells which are have significantly different gene expression from the metacells they are
    belong to based on ``what`` (default: {what}) data.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    Obeys (ignores the genes of) the ``noisy_gene`` per-gene (variable) annotation, if any.

    The exact method depends on the ``policy`` (one of ``gaps`` or ``max``). By default we use the ``gaps`` policy as it
    gives a much lower fraction of deviants at a minor cost in the variance inside each metacell. The ``max`` policy
    provides the inverse trade-off, giving slightly more consistent metacells at the cost of a much higher fraction of
    deviants.

    **Returns**

    A boolean mask of all the cells which should be considered "deviant".

    **Gaps Computation Parameters**

    Intuitively, for each gene for each metacell we can look at the sorted expression level of the gene in all the
    metacell's cells. We look for a large gap between a few low-expressing or high-expressing cells and the rest of the
    cells. If we find such a gap, the few cells below or above it are considered to be deviants.

    1. For each gene in each cell of each metacell, compute the log (base 2) of the fraction of the gene's
       UMIs out of the total UMIs of the metacell, with a 1-UMI regularization factor.

    2. Sort the expression level of each gene in each metacell.

    3. Look for a gap of at least ``min_gene_fold_factor`` (default: {min_gene_fold_factor}), or for ``noisy_gene``,
       an additional ``min_noisy_gene_fold_factor`` (default: {min_noisy_gene_fold_factor}) between the sorted gene
       expressions. If ``gap_skip_cells`` (default: {gap_skip_cells}) is 0, look for a gap between consecutive sorted
       cell expression levels. If it is 1 or 2, skip this number of entries. Ignore gaps if the total number of UMIs
       of the gene in the two compared cells is less than ``min_compare_umis`` (default: {min_compare_umis}).

    4. Ignore gaps that cause more than ``max_gap_cells_fraction`` (default: {max_gap_cells_fraction}) and also
       more than ``max_gap_cells_count`` (default: {max_gap_cells_count}) to be separated. That is, a single gene
       can only mark as deviants  "a few" cells of the metacell.

    5. If any cells were marked as deviants, re-run the above, ignoring any cells previously marked as deviants.

    6. If the total number of cells is more than ``max_cell_fraction`` (default: {max_cell_fraction}) of the cells,
       increase ``min_gene_fold_factor`` by 0.15 (~x1.1) and try again from the top.

    **Max Computation Parameters**

    1. Compute for each candidate metacell the median fraction of the UMIs expressed by each gene.
       Scale this by each cell's total UMIs to compute the expected number of UMIs for each cell.
       Compute the fold factor log2((actual UMIs + 1) / (expected UMIs + 1)) for each gene for each
       cell.

    2. Compute the excess fold factor for each gene in each cell by subtracting ``min_gene_fold_factor`` (default:
       {min_gene_fold_factor}) from the above. For ``noisy_gene``, also subtract ``min_noisy_gene_fold_factor`` to the
       threshold.

    4. For each cell, consider the maximal gene excess fold factor. Consider all
       cells with a positive maximal threshold as deviants. If more than ``max_cell_fraction`` (default:
       {max_cell_fraction}) of the cells have a positive maximal excess fold factor, increase the threshold from 0 so
       that only this fraction are marked as deviants.
    """
    assert 0 <= gap_skip_cells <= 2
    gap_skip_cells += 1
    # assert policy in ("max", "gaps")
    assert policy in ("votes", "count", "max", "sum", "gaps")
    if policy == "votes":
        return _votes_deviant_cells(
            adata,
            what,
            candidates=candidates,
            min_gene_fold_factor=min_gene_fold_factor,
            max_gene_fraction=max_gene_fraction or 1.0,
            max_cell_fraction=max_cell_fraction or 1.0,
        )

    if policy == "gaps":
        return _gaps_deviant_cells(
            adata,
            what,
            candidates=candidates,
            min_gene_fold_factor=min_gene_fold_factor,
            gap_skip_cells=gap_skip_cells,
            min_noisy_gene_fold_factor=min_noisy_gene_fold_factor,
            max_gap_cells_count=max_gap_cells_count,
            max_gap_cells_fraction=max_gap_cells_fraction,
            max_cell_fraction=max_cell_fraction or 1.0,
            cells_regularization_quantile=cells_regularization_quantile,
            min_compare_umis=min_compare_umis,
        )

    cells_count, genes_count = adata.shape
    assert cells_count > 0

    candidate_of_cells = ut.get_o_numpy(adata, candidates, formatter=ut.groups_description).copy()
    candidates_count = np.max(candidate_of_cells) + 1

    total_umis_per_cell = ut.get_o_numpy(adata, what, sum=True)
    assert total_umis_per_cell.size == cells_count

    noisy_genes_mask: Optional[ut.NumpyVector] = None
    if ut.has_data(adata, "noisy_gene"):
        noisy_genes_mask = ut.get_v_numpy(adata, "noisy_gene")
        if not np.any(noisy_genes_mask):
            noisy_genes_mask = None

    umis_per_gene_per_cell = ut.get_vo_proper(adata, what, layout="row_major")

    fold_per_gene_per_cell = np.zeros(adata.shape, dtype="float32")
    for candidate_index in range(candidates_count):
        candidate_cell_indices = np.where(candidate_of_cells == candidate_index)[0]

        candidate_cells_count = candidate_cell_indices.size
        assert candidate_cells_count > 0

        total_umis_per_grouped = total_umis_per_cell[candidate_cell_indices]
        umis_per_gene_per_grouped = ut.to_numpy_matrix(umis_per_gene_per_cell[candidate_cell_indices, :], copy=True)
        assert ut.is_layout(umis_per_gene_per_grouped, "row_major")
        assert umis_per_gene_per_grouped.shape == (candidate_cells_count, genes_count)

        fraction_per_gene_per_grouped = ut.fraction_by(umis_per_gene_per_grouped, by="row", sums=total_umis_per_grouped)

        fraction_per_gene_per_grouped_by_columns = ut.to_layout(fraction_per_gene_per_grouped, "column_major")
        assert fraction_per_gene_per_grouped_by_columns.shape == (candidate_cells_count, genes_count)
        median_fraction_per_gene = ut.median_per(fraction_per_gene_per_grouped_by_columns, per="column")
        assert len(median_fraction_per_gene) == genes_count

        _, dense, compressed = ut.to_proper_matrices(fraction_per_gene_per_grouped)

        if compressed is not None:
            if compressed.nnz == 0:
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
                    total_umis_per_grouped,
                    median_fraction_per_gene,
                )

            fold_per_gene_per_cell[candidate_cell_indices, :] = compressed

        else:
            assert dense is not None

            extension_name = f"fold_factor_dense_{dense.dtype}_t"
            extension = getattr(xt, extension_name)

            with ut.timed_step("extensions.fold_factor_dense"):
                extension(
                    dense,
                    total_umis_per_grouped,
                    median_fraction_per_gene,
                )

            fold_per_gene_per_cell[candidate_cell_indices, :] = dense

    effective_fold_per_gene_per_cell = np.abs(fold_per_gene_per_cell)
    effective_fold_per_gene_per_cell -= min_gene_fold_factor
    if noisy_genes_mask is not None:
        effective_fold_per_gene_per_cell[:, noisy_genes_mask] -= min_noisy_gene_fold_factor
    effective_fold_per_gene_per_cell[effective_fold_per_gene_per_cell < 0] = 0

    if policy == "count":
        policy_per_cell = ut.sum_per(effective_fold_per_gene_per_cell > 0, per="row")
    elif policy == "sum":
        policy_per_cell = ut.sum_per(effective_fold_per_gene_per_cell, per="row")
    elif policy == "max":
        policy_per_cell = ut.max_per(effective_fold_per_gene_per_cell, per="row")
    else:
        assert False
    ut.log_calc(f"{policy}_per_gene_per_cell", policy_per_cell)

    if max_cell_fraction is None or max_cell_fraction == 1.0:
        policy_threshold = 0
    else:
        policy_threshold = np.quantile(policy_per_cell, 1.0 - max_cell_fraction)
    ut.log_calc(f"{policy}_threshold", policy_threshold)

    deviant_cells_mask = policy_per_cell > policy_threshold
    ut.log_calc("deviant_cells_mask", deviant_cells_mask)
    return deviant_cells_mask


def _gaps_deviant_cells(  # pylint: disable=too-many-statements
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    candidates: Union[str, ut.Vector],
    min_gene_fold_factor: float,
    gap_skip_cells: int,
    min_noisy_gene_fold_factor: float,
    max_cell_fraction: float,
    max_gap_cells_count: int,
    max_gap_cells_fraction: float,
    cells_regularization_quantile: float,
    min_compare_umis: int,
) -> ut.Vector:
    assert min_gene_fold_factor > 0
    assert 0 < max_cell_fraction < 1
    cells_count = adata.n_obs
    genes_count = adata.n_vars

    min_gap_per_gene = np.full(genes_count, min_gene_fold_factor, dtype="float32")
    if ut.has_data(adata, "noisy_gene"):
        noisy_genes_mask = ut.get_v_numpy(adata, "noisy_gene")
        min_gap_per_gene[noisy_genes_mask] += min_noisy_gene_fold_factor

    candidate_per_cell = ut.get_o_numpy(adata, candidates, formatter=ut.groups_description).astype("int32")
    candidates_count = np.max(candidate_per_cell) + 1

    umis_per_gene_per_cell = ut.to_numpy_matrix(ut.get_vo_proper(adata, what, layout="row_major"), copy=True).astype(
        "float32"
    )

    regularization_per_cell = np.empty(cells_count, dtype="float32")
    max_gap_per_cell = np.empty(cells_count, dtype="float32")
    new_deviant_per_cell = np.empty(cells_count, dtype="bool")
    deviant_per_cell = np.empty(cells_count, dtype="bool")

    total_umis_per_cell = ut.get_o_numpy(adata, what, sum=True).copy()
    fraction_per_gene_per_cell = umis_per_gene_per_cell / total_umis_per_cell[:, np.newaxis]

    for candidate_index in range(candidates_count):
        mask_per_cell = candidate_per_cell == candidate_index
        total_umis_per_cell_of_candidate = total_umis_per_cell[mask_per_cell]
        regularization_of_candidate = max(
            1.0, np.quantile(total_umis_per_cell_of_candidate, cells_regularization_quantile)
        )
        regularization_per_cell_of_candidate = np.maximum(total_umis_per_cell_of_candidate, regularization_of_candidate)
        regularization_per_cell[mask_per_cell] = 1.0 / regularization_per_cell_of_candidate
    ut.log_calc("regularization_per_cell", regularization_per_cell, formatter=ut.sizes_description)

    log_fraction_per_gene_per_cell = np.log2(fraction_per_gene_per_cell + regularization_per_cell[:, np.newaxis])

    outer_repeat = 0
    while True:
        outer_repeat += 1
        ut.log_calc(f"min_gene_fold_factor ({outer_repeat}.*)", min_gene_fold_factor)

        new_deviant_per_cell[:] = True
        deviant_per_cell[:] = False
        deviants_fraction = 0.0

        inner_repeat = 0
        while deviants_fraction <= max_cell_fraction:
            inner_repeat += 1
            max_gap_per_cell[:] = 0.0
            with ut.timed_step("extensions.compute_cell_gaps"):
                extension = getattr(xt, "compute_cell_gaps")
                extension(
                    umis_per_gene_per_cell,
                    fraction_per_gene_per_cell,
                    log_fraction_per_gene_per_cell,
                    candidate_per_cell,
                    deviant_per_cell,
                    new_deviant_per_cell,
                    min_gap_per_gene,
                    candidates_count,
                    gap_skip_cells,
                    max_gap_cells_count,
                    max_gap_cells_fraction,
                    float(min_compare_umis),
                    max_gap_per_cell,
                )
            ut.log_calc(
                f"max_gap_per_cell ({outer_repeat}.{inner_repeat})", max_gap_per_cell, formatter=ut.sizes_description
            )

            large_gap_count = np.sum(max_gap_per_cell > 0.0)
            ut.log_calc(
                f"large_gap_count ({outer_repeat}.{inner_repeat})",
                large_gap_count,
                formatter=lambda count: ut.ratio_description(adata.n_obs, "cell", count, "large-gap"),
            )
            remaining_cell_fraction = 1 - max(max_cell_fraction - deviants_fraction, 0.0)
            if large_gap_count / adata.n_obs <= remaining_cell_fraction:
                gap_threshold = 0.0
            else:
                gap_threshold = np.quantile(max_gap_per_cell, remaining_cell_fraction)
            ut.log_calc(f"gap_threshold ({outer_repeat}.{inner_repeat})", gap_threshold)

            new_deviant_per_cell = (max_gap_per_cell > gap_threshold) & ~deviant_per_cell
            ut.log_calc(f"new_deviant_per_cell ({outer_repeat}.{inner_repeat})", new_deviant_per_cell)
            new_deviants_count = np.sum(new_deviant_per_cell)
            if new_deviants_count == 0:
                break
            deviants_fraction += new_deviants_count / adata.n_obs
            deviant_per_cell = deviant_per_cell | new_deviant_per_cell
            ut.log_calc(f"deviants_fraction ({outer_repeat}.{inner_repeat})", deviants_fraction)
            ut.log_calc(f"deviant_per_cell ({outer_repeat}.{inner_repeat})", deviant_per_cell)

        if deviants_fraction <= max_cell_fraction:
            ut.log_calc(f"final min_gene_fold_factor ({outer_repeat}.*)", min_gene_fold_factor)
            return deviant_per_cell

        min_gap_per_gene += 1 / 8
        min_gene_fold_factor += 1 / 8


def _votes_deviant_cells(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    candidates: Union[str, ut.Vector],
    min_gene_fold_factor: float,
    max_gene_fraction: float,
    max_cell_fraction: float,
) -> ut.Vector:
    assert min_gene_fold_factor > 0
    assert 0 < max_gene_fraction < 1
    assert 0 < max_cell_fraction < 1

    cells_count, genes_count = adata.shape
    assert cells_count > 0

    candidate_of_cells = ut.get_o_numpy(adata, candidates, formatter=ut.groups_description).copy()

    totals_of_cells = ut.get_o_numpy(adata, what, sum=True)
    assert totals_of_cells.size == cells_count

    noisy_genes_mask: Optional[ut.NumpyVector] = None
    if ut.has_data(adata, "noisy_gene"):
        noisy_genes_mask = ut.get_v_numpy(adata, "noisy_gene")

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
            acceptable_cells_mask=acceptable_cells_mask,
            noisy_genes_mask=noisy_genes_mask,
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

    deviant_cells_mask = full_votes_of_deviant_cells > 0
    ut.log_return("deviant_cells", deviant_cells_mask)
    return deviant_cells_mask


@ut.timed_call()
def _collect_fold_factors(  # pylint: disable=too-many-statements
    *,
    data: ut.ProperMatrix,
    candidate_of_cells: ut.NumpyVector,
    totals_of_cells: ut.NumpyVector,
    min_gene_fold_factor: float,
    acceptable_cells_mask: ut.NumpyVector,
    noisy_genes_mask: Optional[ut.NumpyVector],
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
            compressed = sp.csr_matrix(
                ([], [], [0] * (candidate_cells_count + 1)), shape=(candidate_cells_count, genes_count)
            )
            list_of_fold_factors.append(compressed)
            assert compressed.has_sorted_indices
            assert compressed.has_canonical_format
            continue

        umis_of_candidate_by_row: ut.ProperMatrix = data[candidate_cell_indices, :]
        assert ut.is_layout(umis_of_candidate_by_row, "row_major")
        assert umis_of_candidate_by_row.shape == (candidate_cells_count, genes_count)

        totals_of_candidate_cells = totals_of_cells[candidate_cell_indices]

        fractions_of_candidate_by_row = ut.scale_by(
            umis_of_candidate_by_row, np.reciprocal(totals_of_candidate_cells), by="row"
        )

        fractions_of_candidate_by_column = ut.to_layout(fractions_of_candidate_by_row, "column_major")
        medians_of_candidate_genes = ut.median_per(fractions_of_candidate_by_column, per="column")

        _, dense, compressed = ut.to_proper_matrices(umis_of_candidate_by_row)

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
                    totals_of_candidate_cells,
                    medians_of_candidate_genes,
                )

            if noisy_genes_mask is not None:
                compressed[:, noisy_genes_mask] = 0.0

            compressed.data[(compressed.data < min_gene_fold_factor) & (compressed.data > -min_gene_fold_factor)] = 0.0

            ut.eliminate_zeros(compressed)

        else:
            assert dense is not None

            extension_name = f"fold_factor_dense_{dense.dtype}_t"
            extension = getattr(xt, extension_name)

            with ut.timed_step("extensions.fold_factor_dense"):
                extension(
                    dense,
                    totals_of_candidate_cells,
                    medians_of_candidate_genes,
                )

            if noisy_genes_mask is not None:
                dense[:, noisy_genes_mask] = 0.0

            dense[(dense < min_gene_fold_factor) & (dense > -min_gene_fold_factor)] = 0.0

            compressed = sp.csr_matrix(dense)
            assert compressed.has_sorted_indices
            assert compressed.has_canonical_format

        if compressed.nnz == 0:
            acceptable_cells_mask[candidate_cell_indices] = True
        list_of_fold_factors.append(compressed)

    if remaining_cells_count > 0:
        assert remaining_cells_count == np.sum(candidate_of_cells < 0)
        list_of_cell_index_of_rows.append(np.where(candidate_of_cells < 0)[0])
        compressed = sp.csr_matrix(
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
        return sp.csr_matrix((0, 0))

    cell_row_of_indices = np.empty_like(cell_index_of_rows)
    cell_row_of_indices[cell_index_of_rows] = np.arange(cells_count)

    fold_factors = sp.vstack(list_of_fold_factors, format="csr")
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

        gene_fold_ranks = st.rankdata(gene_fold_factors, method="min")
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
