'''
Finalize Grouping of Cells to Metacells
---------------------------------------
'''

import logging
from typing import Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.preprocessing as pp
import metacells.utilities as ut

__all__ = [
    'finalize_metacells',
]


LOG = logging.getLogger(__name__)


@ut.expand_doc()
@ut.timed_call()
def finalize_metacells(  # pylint: disable=too-many-branches
    adata: AnnData,
    *,
    of: Optional[str] = None,
    candidate_metacells: Union[str, ut.Vector] = 'candidate_metacell',
    outliers: Optional[Union[str, ut.Vector]] = 'outlier_cells',
    target_metacell_size: int,
    cell_sizes: Optional[Union[str, ut.Vector]],
    min_robust_size_factor: Optional[float] = 0.5,
    min_convincing_size_factor: Optional[float] = 0.25,
    min_convincing_gene_fold_factor: float = 3.0,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Finalize the grouping of cells to metacells based ``of`` some data (by default, the focus).

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Observation (Cell) Annotations
        ``metacell``
            The integer index of the metacell each cell belongs to. The metacells are in no
            particular order. Cells with no metacell assignment are given a metacell index of
            ``-1``.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Mark all ``outliers`` (default: {outliers}) as such. This can be the name of a
       per-observation (cell) annotation, or an explicit boolean mask of cells, or a vector of
       outlier cell indices, or ``None`` if there are no outlier cells to mark.

    2. We are trying to create metacells of size ``target_metacell_size``. Compute the sizes of the
       resulting metacells by summing the ``cell_sizes``, which again can be the name of a
       per-observation (cell) annotation, or an explicit sizes vector. If it is ``None``, then each
       cell is given a size of one. These parameters are typically identical to these passed to
       :py:func:`metacells.tools.candidate_metacells.compute_candidate_metacells`.

    3. If ``min_robust_size_factor` (default: {min_robust_size_factor}) is specified, then any
       metacell whose total size is at least ``target_metacell_size * min_robust_size_factor`` is
       preserved.

    3. If ``min_convincing_size_factor`` (default: {min_convincing_size_factor}) is
       specified, then any remaining metacells whose size is at least ``target_metacell_size *
       min_convincing_size_factor`` are preserved, given they contain at least one gene whose fold
       factor (log2((actual + 1) / (expected + 1))) is at least ``min_convincing_gene_fold_factor``
       (default: {min_convincing_gene_fold_factor}). That is, we only preserve these smaller
       metacells if there is at least one gene whose expression is significantly different from the
       mean of the population.

    4 . Any remaining metacell is dissolved into outlier cells.
    '''
    of, level = ut.log_operation(LOG, adata, 'finalize_metacells', of)

    with ut.focus_on(ut.get_vo_data, adata, of,
                     layout='row_major', intermediate=intermediate) as data:

        candidate_of_cells = \
            ut.get_vector_parameter_data(LOG, level, adata, candidate_metacells,
                                         per='o', name='candidate_metacells')
        candidate_of_cells = np.copy(candidate_of_cells)
        assert candidate_of_cells is not None
        raw_candidates_count = np.max(candidate_of_cells) + 1
        LOG.debug('  candidates: %s', raw_candidates_count)

        outlier_cells = \
            ut.get_vector_parameter_data(LOG, level, adata, outliers,
                                         per='o', name='outliers')

        cell_sizes = \
            ut.get_vector_parameter_data(LOG, level, adata, cell_sizes,
                                         per='o', name='cell_sizes')

        if outlier_cells is not None:
            candidate_of_cells[outlier_cells] = -1
        candidate_of_cells = ut.compress_indices(candidate_of_cells)
        candidates_count = np.max(candidate_of_cells) + 1

        LOG.log(level, '  target_metacell_size: %d', target_metacell_size)
        fraction_of_genes = pp.get_fraction_per_var(adata).proper

        if min_robust_size_factor is None:
            min_robust_size = None
        else:
            min_robust_size = target_metacell_size * min_robust_size_factor
            LOG.log(level, '  min_robust_size: %d', min_robust_size)

        if min_convincing_size_factor is None:
            min_convincing_size = None
        else:
            min_convincing_size = \
                target_metacell_size * min_convincing_size_factor
            LOG.log(level, '  min_convincing_size: %d', min_convincing_size)
            LOG.log(level, '  min_convincing_gene_fold_factor: %s',
                    min_convincing_gene_fold_factor)

        if min_robust_size is not None and min_convincing_size is not None:
            did_dissolve = False

            for candidate_index in range(candidates_count):
                candidate_cell_indices = \
                    np.where(candidate_of_cells == candidate_index)[0]
                if not _keep_candidate(adata, candidate_index,
                                       data=data,
                                       cell_sizes=cell_sizes,
                                       fraction_of_genes=fraction_of_genes,
                                       min_robust_size=min_robust_size,
                                       min_convincing_size=min_convincing_size,
                                       min_convincing_gene_fold_factor=min_convincing_gene_fold_factor,
                                       candidates_count=candidates_count,
                                       candidate_cell_indices=candidate_cell_indices):
                    candidate_of_cells[candidate_cell_indices] = -1
                    did_dissolve = True

    if did_dissolve:
        metacell_of_cells = ut.compress_indices(candidate_of_cells)
    else:
        metacell_of_cells = candidate_of_cells
    metacells_count = np.max(candidate_of_cells) + 1

    if LOG.isEnabledFor(level):
        LOG.log(level, '  final metacells: %s',
                ut.ratio_description(metacells_count, raw_candidates_count))

    if inplace:
        ut.set_o_data(adata, 'metacell', metacell_of_cells, ut.NEVER_SAFE,
                      log_value=lambda:
                      ut.ratio_description(metacells_count,
                                           raw_candidates_count))
        return None

    return pd.Series(metacell_of_cells, adata.obs_names)


def _keep_candidate(
    adata: AnnData,
    candidate_index: int,
    *,
    data: ut.ProperMatrix,
    cell_sizes: Optional[ut.DenseVector],
    fraction_of_genes: ut.DenseVector,
    min_robust_size: Optional[float],
    min_convincing_size: Optional[float],
    min_convincing_gene_fold_factor: float,
    candidates_count: int,
    candidate_cell_indices: ut.DenseVector,
) -> bool:
    genes_count = data.shape[1]

    if cell_sizes is None:
        candidate_total_size = candidate_cell_indices.size
    else:
        candidate_total_size = np.sum(cell_sizes[candidate_cell_indices])

    if min_robust_size is not None \
            and candidate_total_size >= min_robust_size:
        LOG.debug('  - candidate: %s / %s cells: %s size: %d is: robust',
                  candidate_index, candidates_count,
                  candidate_cell_indices.size, candidate_total_size)
        return True

    if min_convincing_size is None:
        LOG.debug('  - candidate: %s / %s cells: %s size: %d is: accepted',
                  candidate_index, candidates_count,
                  candidate_cell_indices.size, candidate_total_size)
        return True

    if candidate_total_size < min_convincing_size:
        LOG.debug('  - candidate: %s / %s cells: %s size: %d is: unconvincing',
                  candidate_index, candidates_count,
                  candidate_cell_indices.size, candidate_total_size)
        return False

    candidate_data = data[candidate_cell_indices, :]
    candidate_data_of_genes = ut.to_dense_vector(candidate_data.sum(axis=0))
    assert candidate_data_of_genes.size == genes_count
    candidate_total = np.sum(candidate_data_of_genes)
    candidate_expected_of_genes = fraction_of_genes * candidate_total
    candidate_expected_of_genes += 1
    candidate_data_of_genes += 1
    candidate_data_of_genes /= candidate_expected_of_genes
    np.log2(candidate_data_of_genes, out=candidate_data_of_genes)
    convincing_genes_mask = \
        candidate_data_of_genes >= min_convincing_gene_fold_factor
    keep_candidate = np.any(convincing_genes_mask)

    if LOG.isEnabledFor(logging.DEBUG):
        convincing_gene_indices = \
            np.where(convincing_genes_mask)[0]
        if keep_candidate:
            LOG.debug('  - candidate: %s / %s cells: %s size: %d is: convincing because %s',
                      candidate_index, candidates_count,
                      candidate_cell_indices.size,
                      candidate_total_size,
                      ', '.join(['%s: %s' % (name, fold_factor)
                                 for name, fold_factor
                                 in zip(adata.var_names[convincing_gene_indices],
                                        candidate_data_of_genes[convincing_gene_indices])]))
        else:
            LOG.debug('  - candidate: %s / %s cells: %s size: %d is: not convincing',
                      candidate_index, candidates_count,
                      candidate_cell_indices.size, candidate_total_size)

    return keep_candidate
