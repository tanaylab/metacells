'''
Dissolve
--------
'''

from typing import Optional, Union

import numpy as np
from anndata import AnnData

import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    'dissolve_metacells',
]


@ut.logged()
@ut.expand_doc()
@ut.timed_call()
def dissolve_metacells(
    adata: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    *,
    candidates: Union[str, ut.Vector] = 'candidate',
    deviants: Optional[Union[str, ut.Vector]] = 'cell_deviant_votes',
    target_metacell_size: float = pr.target_metacell_size,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.dissolve_cell_sizes,
    min_metacell_cells: int = pr.dissolve_min_metacell_cells,
    min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    inplace: bool = True,
) -> Optional[ut.PandasFrame]:
    '''
    Dissolve too-small metacells based on ``what`` (default: {what}) data.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    **Returns**

    Observation (Cell) Annotations
        ``metacell``
            The integer index of the metacell each cell belongs to. The metacells are in no
            particular order. Cells with no metacell assignment are given a metacell index of
            ``-1``.

        ``dissolved``
            A boolean mask of the cells which were in a dissolved metacell.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas data frame (indexed by the observation names).

    **Computation Parameters**

    1. Mark all cells with non-zero ``deviants`` (default: {deviants}) as "outliers". This can be
       the name of a per-observation (cell) annotation, or an explicit boolean mask of cells, or a
       or ``None`` if there are no deviant cells to mark.

    2. Any metacell which has less cells than the ``min_metacell_cells`` is dissolved.

    3. We are trying to create metacells of size ``target_metacell_size``. Compute the sizes of the
       resulting metacells by summing the ``cell_sizes`` (default: {cell_sizes}). If it is ``None``,
       each has a size of one. These parameters are typically identical to these passed to
       :py:func:`metacells.tools.candidates.compute_candidate_metacells`.

    4. If ``min_robust_size_factor` (default: {min_robust_size_factor}) is specified, then any
       metacell whose total size is at least ``target_metacell_size * min_robust_size_factor`` is
       preserved.

    5. If ``min_convincing_size_factor`` (default: {min_convincing_size_factor}) is
       specified, then any remaining metacells whose size is at least ``target_metacell_size *
       min_convincing_size_factor`` are preserved, given they contain at least one gene whose fold
       factor (log2((actual + 1) / (expected + 1))) is at least ``min_convincing_gene_fold_factor``
       (default: {min_convincing_gene_fold_factor}). That is, we only preserve these smaller
       metacells if there is at least one gene whose expression is significantly different from the
       mean of the population.

    6 . Any remaining metacell is dissolved into "outlier" cells.
    '''
    dissolved_of_cells = np.zeros(adata.n_obs, dtype='bool')

    candidate_of_cells = ut.get_o_numpy(adata, candidates,
                                        formatter=ut.groups_description)
    candidate_of_cells = np.copy(candidate_of_cells)

    deviant_of_cells = \
        ut.maybe_o_numpy(adata, deviants, formatter=ut.mask_description)
    if deviant_of_cells is not None:
        deviant_of_cells = deviant_of_cells > 0
    cell_sizes = \
        ut.maybe_o_numpy(adata, cell_sizes, formatter=ut.sizes_description)

    if deviant_of_cells is not None:
        candidate_of_cells[deviant_of_cells > 0] = -1
    candidate_of_cells = ut.compress_indices(candidate_of_cells)
    candidates_count = np.max(candidate_of_cells) + 1

    data = ut.get_vo_proper(adata, what, layout='column_major')
    fraction_of_genes = ut.fraction_per(data, per='column')

    if min_robust_size_factor is None:
        min_robust_size = None
    else:
        min_robust_size = target_metacell_size * min_robust_size_factor
    ut.log_calc('min_robust_size', min_robust_size)

    if min_convincing_size_factor is None:
        min_convincing_size = None
    else:
        min_convincing_size = \
            target_metacell_size * min_convincing_size_factor
    ut.log_calc('min_convincing_size', min_convincing_size)

    did_dissolve = False
    for candidate_index in range(candidates_count):
        candidate_cell_indices = \
            np.where(candidate_of_cells == candidate_index)[0]
        if not _keep_candidate(adata, candidate_index,
                               data=data,
                               cell_sizes=cell_sizes,
                               fraction_of_genes=fraction_of_genes,
                               min_metacell_cells=min_metacell_cells,
                               min_robust_size=min_robust_size,
                               min_convincing_size=min_convincing_size,
                               min_convincing_gene_fold_factor=min_convincing_gene_fold_factor,
                               candidates_count=candidates_count,
                               candidate_cell_indices=candidate_cell_indices):
            dissolved_of_cells[candidate_cell_indices] = True
            candidate_of_cells[candidate_cell_indices] = -1
            did_dissolve = True

    if did_dissolve:
        metacell_of_cells = ut.compress_indices(candidate_of_cells)
    else:
        metacell_of_cells = candidate_of_cells

    if inplace:
        ut.set_o_data(adata, 'dissolved', dissolved_of_cells,
                      formatter=ut.mask_description)

        ut.set_o_data(adata, 'metacell', metacell_of_cells,
                      formatter=ut.groups_description)
        return None

    ut.log_return('dissolved', dissolved_of_cells)
    ut.log_return('metacell', metacell_of_cells,
                  formatter=ut.groups_description)

    obs_frame = ut.to_pandas_frame(index=adata.obs_names)
    obs_frame['dissolved'] = dissolved_of_cells
    obs_frame['metacell'] = metacell_of_cells
    return obs_frame


def _keep_candidate(  # pylint: disable=too-many-branches
    adata: AnnData,
    candidate_index: int,
    *,
    data: ut.ProperMatrix,
    cell_sizes: Optional[ut.NumpyVector],
    fraction_of_genes: ut.NumpyVector,
    min_metacell_cells: int,
    min_robust_size: Optional[float],
    min_convincing_size: Optional[float],
    min_convincing_gene_fold_factor: float,
    candidates_count: int,
    candidate_cell_indices: ut.NumpyVector,
) -> bool:
    genes_count = data.shape[1]

    if cell_sizes is None:
        candidate_total_size = candidate_cell_indices.size
    else:
        candidate_total_size = np.sum(cell_sizes[candidate_cell_indices])

    if candidate_cell_indices.size < min_metacell_cells:
        if ut.logging_calc():
            ut.log_calc(f'- candidate: {ut.progress_description(candidates_count, candidate_index, "candidate")} '
                        f'cells: {candidate_cell_indices.size} '
                        f'size: {candidate_total_size:g} '
                        f'is: little')
        return False

    if min_robust_size is not None \
            and candidate_total_size >= min_robust_size:
        if ut.logging_calc():
            ut.log_calc(f'- candidate: {ut.progress_description(candidates_count, candidate_index, "candidate")} '
                        f'cells: {candidate_cell_indices.size} '
                        f'size: {candidate_total_size:g} '
                        f'is: robust')
        return True

    if min_convincing_size is None:
        if ut.logging_calc():
            ut.log_calc(f'- candidate: {ut.progress_description(candidates_count, candidate_index, "candidate")} '
                        f'cells: {candidate_cell_indices.size} '
                        f'size: {candidate_total_size:g} '
                        f'is: accepted')
        return True

    if candidate_total_size < min_convincing_size:
        if ut.logging_calc():
            ut.log_calc(f'- candidate: {ut.progress_description(candidates_count, candidate_index, "candidate")} '
                        f'cells: {candidate_cell_indices.size} '
                        f'size: {candidate_total_size:g} '
                        f'is: unconvincing')
        return False

    candidate_data = data[candidate_cell_indices, :]
    candidate_data_of_genes = ut.to_numpy_vector(candidate_data.sum(axis=0))
    assert candidate_data_of_genes.size == genes_count
    candidate_total = np.sum(candidate_data_of_genes)
    candidate_expected_of_genes = fraction_of_genes * candidate_total
    candidate_expected_of_genes += 1
    candidate_data_of_genes += 1
    candidate_data_of_genes /= candidate_expected_of_genes
    np.log2(candidate_data_of_genes, out=candidate_data_of_genes)
    convincing_genes_mask = \
        candidate_data_of_genes >= min_convincing_gene_fold_factor
    keep_candidate = bool(np.any(convincing_genes_mask))

    if ut.logging_calc():
        convincing_gene_indices = np.where(convincing_genes_mask)[0]
        if keep_candidate:
            ut.log_calc(f'- candidate: {ut.progress_description(candidates_count, candidate_index, "candidate")} '
                        f'cells: {candidate_cell_indices.size} '
                        f'size: {candidate_total_size:g} '
                        f'is: convincing because:')
            for fold_factor, name \
                in reversed(sorted(zip(candidate_data_of_genes[convincing_gene_indices],
                                       adata.var_names[convincing_gene_indices]))):
                ut.log_calc(f'    {name}: {ut.fold_description(fold_factor)}')
        else:
            ut.log_calc(f'- candidate: {ut.progress_description(candidates_count, candidate_index, "candidate")} '
                        f'cells: {candidate_cell_indices.size} '
                        f'size: {candidate_total_size:g} '
                        f'is: not convincing')

    return keep_candidate
