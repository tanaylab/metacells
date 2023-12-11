"""
Deviants
--------
"""

import sys
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore
from scipy import sparse as sp  # type: ignore

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
def find_deviant_cells(
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
    gene_modules: Optional[Dict[str, List[str]]] = None,
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

    **Computation Parameters**

    0. If any ``gene_modules`` are specified, then compute the sum of the UMIs of the genes of each one and
       treat this as if it was an additional gene in the methods below. Note this new (unnamed) gene is never considered
       to be noisy.

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
    assert policy in ("max", "gaps")
    assert 0 <= gap_skip_cells <= 2
    gap_skip_cells += 1
    gene_module_indices = _compute_gene_module_indices(adata, gene_modules or {})

    noisy_gene_indices: Optional[ut.NumpyVector] = None
    if ut.has_data(adata, "noisy_gene"):
        noisy_genes_mask = ut.get_v_numpy(adata, "noisy_gene")
        noisy_gene_indices = np.where(noisy_genes_mask)[0]
        if len(noisy_gene_indices) == 0:
            noisy_gene_indices = None

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
            gene_module_indices=gene_module_indices,
            noisy_gene_indices=noisy_gene_indices,
        )

    return _max_deviant_cells(
        adata,
        what,
        candidates=candidates,
        min_gene_fold_factor=min_gene_fold_factor,
        min_noisy_gene_fold_factor=min_noisy_gene_fold_factor,
        max_cell_fraction=max_cell_fraction or 1.0,
        gene_module_indices=gene_module_indices,
        noisy_gene_indices=noisy_gene_indices,
    )


def _max_deviant_cells(  # pylint: disable=too-many-statements
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    candidates: Union[str, ut.Vector],
    min_gene_fold_factor: float,
    min_noisy_gene_fold_factor: float,
    max_cell_fraction: float,
    gene_module_indices: Dict[str, List[int]],
    noisy_gene_indices: Optional[ut.NumpyVector],
) -> ut.Vector:
    cells_count, genes_count = adata.shape
    assert cells_count > 0

    candidate_of_cells = ut.get_o_numpy(adata, candidates, formatter=ut.groups_description).copy()
    candidates_count = np.max(candidate_of_cells) + 1

    total_umis_per_cell = ut.get_o_numpy(adata, what, sum=True)
    assert total_umis_per_cell.size == cells_count

    umis_per_gene_per_cell = ut.get_vo_proper(adata, what, layout="row_major")
    umis_per_gene_per_cell = _extend_with_gene_modules(umis_per_gene_per_cell, gene_module_indices)

    fold_per_gene_per_cell = np.zeros(umis_per_gene_per_cell.shape, dtype="float32")
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
    if noisy_gene_indices is not None:
        effective_fold_per_gene_per_cell[:, noisy_gene_indices] -= min_noisy_gene_fold_factor
    effective_fold_per_gene_per_cell[effective_fold_per_gene_per_cell < 0] = 0

    max_per_cell = ut.max_per(effective_fold_per_gene_per_cell, per="row")
    ut.log_calc("max_per_gene_per_cell", max_per_cell)

    if max_cell_fraction is None or max_cell_fraction == 1.0:
        max_threshold = 0
    else:
        max_threshold = np.quantile(max_per_cell, 1.0 - max_cell_fraction)
    ut.log_calc("max_threshold", max_threshold)

    deviant_cells_mask = max_per_cell > max_threshold
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
    gene_module_indices: Dict[str, List[int]],
    noisy_gene_indices: Optional[ut.NumpyVector],
) -> ut.Vector:
    assert min_gene_fold_factor > 0
    assert 0 < max_cell_fraction < 1
    cells_count = adata.n_obs
    genes_count = adata.n_vars

    min_gap_per_gene = np.full(genes_count + len(gene_module_indices), min_gene_fold_factor, dtype="float32")
    if noisy_gene_indices is not None:
        min_gap_per_gene[noisy_gene_indices] += min_noisy_gene_fold_factor

    candidate_per_cell = ut.get_o_numpy(adata, candidates, formatter=ut.groups_description).astype("int32")
    candidates_count = np.max(candidate_per_cell) + 1

    umis_per_gene_per_cell = ut.to_numpy_matrix(ut.get_vo_proper(adata, what, layout="row_major"), copy=True).astype(
        "float32"
    )
    umis_per_gene_per_cell = ut.mustbe_numpy_matrix(
        _extend_with_gene_modules(umis_per_gene_per_cell, gene_module_indices)
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


def _compute_gene_module_indices(adata: AnnData, gene_modules: Dict[str, List[str]]) -> Dict[str, List[int]]:
    return {
        gene_module_name: _compute_gene_indices(adata, gene_names)
        for gene_module_name, gene_names in gene_modules.items()
    }


def _compute_gene_indices(adata: AnnData, gene_names: List[str]) -> List[int]:
    return [_compute_gene_index(adata, gene_name) for gene_name in gene_names]


def _compute_gene_index(adata: AnnData, gene_name: str) -> int:
    try:
        return adata.var_names.get_loc("gene_name")
    except KeyError:
        raise KeyError(  # pylint: disable=raise-missing-from
            f"Unknown gene: {gene_name} in AnnData: {ut.get_name(adata, 'unnamed')}"
        )


def _extend_with_gene_modules(
    umis_per_gene_per_cell: ut.ProperMatrix, gene_module_indices: Dict[str, List[int]]
) -> ut.ProperMatrix:
    gene_modules_count = len(gene_module_indices)
    if gene_modules_count == 0:
        return umis_per_gene_per_cell
    umis_per_gene_per_cell_in_columns = ut.to_layout(umis_per_gene_per_cell, layout="column_major")
    cells_count = umis_per_gene_per_cell.shape[0]
    umis_per_gene_per_module = np.empty((cells_count, gene_modules_count), dtype="float32")
    for module_index, gene_indices in enumerate(gene_module_indices.values()):
        umis_per_module_genes_per_cell = umis_per_gene_per_cell_in_columns[:, gene_indices]
        umis_per_gene_per_module[:, module_index] = ut.sum_per(umis_per_module_genes_per_cell, per="column")

    _, dense, compressed = ut.to_proper_matrices(umis_per_module_genes_per_cell)
    if compressed is not None:
        return sp.hstack(compressed, sp.csr_matrix(umis_per_module_genes_per_cell))
    assert dense is not None
    return np.hstack((dense, umis_per_gene_per_module))
