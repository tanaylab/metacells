"""
Dissolve
--------
"""

from math import floor
from typing import Optional
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    "dissolve_metacells",
]


@ut.logged()
@ut.expand_doc()
@ut.timed_call()
def dissolve_metacells(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    candidates: Union[str, ut.Vector] = "candidate",
    deviants: ut.Vector,
    target_metacell_size: int = pr.target_metacell_size,
    min_metacell_size: int = pr.min_metacell_size,
    target_metacell_umis: int = pr.target_metacell_umis,
    cell_umis: Optional[ut.NumpyVector] = pr.cell_umis,
    min_robust_size_factor: float = pr.dissolve_min_robust_size_factor,
    min_convincing_gene_fold_factor: Optional[float] = pr.dissolve_min_convincing_gene_fold_factor,
) -> None:
    """
    Dissolve too-small metacells based on ``what`` (default: {what}) data.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    **Returns**

    Sets the following in ``adata``:

    Observation (Cell) Annotations
        ``metacell``
            The integer index of the metacell each cell belongs to. The metacells are in no
            particular order. Cells with no metacell assignment are given a metacell index of
            ``-1``.

        ``dissolved``
           A boolean mask of the cells which were in a dissolved metacell.

    **Computation Parameters**

    0. If ``cell_umis`` is not specified, use the sum of the ``what`` data for each cell.

    1. Mark all ``deviants`` cells "outliers". This can be the name of a per-observation (cell) annotation, or an
       explicit boolean mask of cells, or a or ``None`` if there are no deviant cells to mark.

    2. Any metacell which has less cells than the ``min_metacell_size`` is dissolved into outlier cells.

    3. If ``min_convincing_gene_fold_factor`` is not ``None``, preserve everything else. Otherwise:

    4. We are trying to create metacells of size ``target_metacell_size`` cells and ``target_metacell_umis`` UMIs each.
       Compute the UMIs of the metacells by summing the ``cell_umis``.

    5. Using ``min_robust_size_factor`` (default: {min_robust_size_factor}), any
       metacell whose total size is at least ``target_metacell_size * min_robust_size_factor``
       or whose total UMIs are at least ``target_metacell_umis * min_robust_size_factor`` is preserved.

    6. Using ``min_convincing_gene_fold_factor``, preserve any remaining metacells which have at least one gene whose
       fold factor (log2((actual + 1) / (expected_by_overall_population + 1))) is at least this high.

    6. Dissolve the remaining metacells into outlier cells.
    """
    dissolved_of_cells = np.zeros(adata.n_obs, dtype="bool")

    candidate_of_cells = ut.get_o_numpy(adata, candidates, formatter=ut.groups_description)
    candidate_of_cells = np.copy(candidate_of_cells)

    deviant_of_cells = ut.maybe_o_numpy(adata, deviants, formatter=ut.mask_description)
    if deviant_of_cells is not None:
        deviant_of_cells = deviant_of_cells > 0

    if cell_umis is None:
        cell_umis = ut.get_o_numpy(adata, what, sum=True, formatter=ut.sizes_description).astype("float32")
    else:
        assert cell_umis.dtype == "float32"
    assert isinstance(cell_umis, ut.NumpyVector)

    if deviant_of_cells is not None:
        candidate_of_cells[deviant_of_cells > 0] = -1
    candidate_of_cells = ut.compress_indices(candidate_of_cells)
    candidates_count = np.max(candidate_of_cells) + 1

    data = ut.get_vo_proper(adata, what, layout="column_major")
    fraction_of_genes = ut.fraction_per(data, per="column")

    min_robust_size = int(floor(target_metacell_size * min_robust_size_factor))
    min_robust_umis = int(floor(target_metacell_umis * min_robust_size_factor))
    ut.log_calc("min_robust_size", min_robust_size)
    ut.log_calc("min_robust_umis", min_robust_umis)

    did_dissolve = False
    for candidate_index in range(candidates_count):
        candidate_cell_indices = np.where(candidate_of_cells == candidate_index)[0]
        if not _keep_candidate(
            candidate_index,
            data=data,
            cell_umis=cell_umis,
            fraction_of_genes=fraction_of_genes,
            min_metacell_size=min_metacell_size,
            min_robust_size=min_robust_size,
            min_robust_umis=min_robust_umis,
            min_convincing_gene_fold_factor=min_convincing_gene_fold_factor,
            candidates_count=candidates_count,
            candidate_cell_indices=candidate_cell_indices,
        ):
            dissolved_of_cells[candidate_cell_indices] = True
            candidate_of_cells[candidate_cell_indices] = -1
            did_dissolve = True

    if did_dissolve:
        metacell_of_cells = ut.compress_indices(candidate_of_cells)
    else:
        metacell_of_cells = candidate_of_cells

    ut.set_o_data(adata, "dissolved", dissolved_of_cells)
    ut.set_o_data(adata, "metacell", metacell_of_cells, formatter=ut.groups_description)


def _keep_candidate(
    candidate_index: int,
    *,
    data: ut.ProperMatrix,
    cell_umis: ut.NumpyVector,
    fraction_of_genes: ut.NumpyVector,
    min_metacell_size: int,
    min_robust_size: int,
    min_robust_umis: int,
    min_convincing_gene_fold_factor: Optional[float],
    candidates_count: int,
    candidate_cell_indices: ut.NumpyVector,
) -> bool:
    metacell_size = candidate_cell_indices.size
    metacell_umis = np.sum(cell_umis[candidate_cell_indices])

    if metacell_size < min_metacell_size:
        if ut.logging_calc():
            ut.log_calc(
                f'- candidate: {ut.progress_description(candidates_count, candidate_index, "candidate")} '
                f"size: {metacell_size} "
                f"umis: {metacell_umis:g} "
                f"has: less than minimal size"
            )
        return False

    if metacell_size >= min_robust_size:
        if ut.logging_calc():
            ut.log_calc(
                f'- candidate: {ut.progress_description(candidates_count, candidate_index, "candidate")} '
                f"size: {metacell_size} "
                f"umis: {metacell_umis:g} "
                f"has: robust size"
            )
        return True

    if metacell_umis >= min_robust_umis:
        if ut.logging_calc():
            ut.log_calc(
                f'- candidate: {ut.progress_description(candidates_count, candidate_index, "candidate")} '
                f"size: {metacell_size} "
                f"umis: {metacell_umis:g} "
                f"has: robust umis"
            )
        return True

    if min_convincing_gene_fold_factor is None:
        if ut.logging_calc():
            ut.log_calc(
                f'- candidate: {ut.progress_description(candidates_count, candidate_index, "candidate")} '
                f"size: {metacell_size} "
                f"umis: {metacell_umis:g} "
                f"has: minimal size"
            )
        return True

    genes_count = data.shape[1]
    candidate_data = data[candidate_cell_indices, :]
    candidate_data_of_genes = ut.to_numpy_vector(candidate_data.sum(axis=0))
    assert candidate_data_of_genes.size == genes_count
    candidate_total = np.sum(candidate_data_of_genes)
    candidate_expected_of_genes = fraction_of_genes * candidate_total
    candidate_expected_of_genes += 1
    candidate_data_of_genes += 1
    candidate_data_of_genes /= candidate_expected_of_genes
    np.log2(candidate_data_of_genes, out=candidate_data_of_genes)
    convincing_genes_mask = np.abs(candidate_data_of_genes) >= min_convincing_gene_fold_factor
    keep_candidate = bool(np.any(convincing_genes_mask))

    if ut.logging_calc():
        if keep_candidate:
            ut.log_calc(
                f'- candidate: {ut.progress_description(candidates_count, candidate_index, "candidate")} '
                f"size: {metacell_size} "
                f"umis: {metacell_umis:g} "
                f"has: convincing genes"
            )
        else:
            ut.log_calc(
                f'- candidate: {ut.progress_description(candidates_count, candidate_index, "candidate")} '
                f"size: {metacell_size} "
                f"umis: {metacell_umis:g} "
                f"has: no convincing genes"
            )

    return keep_candidate
