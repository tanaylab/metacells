'''
Dissolve
--------
'''

import logging
from typing import Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.parameters as pr
import metacells.preprocessing as pp
import metacells.utilities as ut

__all__ = [
    'dissolve_metacells',
]


LOG = logging.getLogger(__name__)


@ut.expand_doc()
@ut.timed_call()
def dissolve_metacells(  # pylint: disable=too-many-branches,too-many-statements
    adata: AnnData,
    *,
    of: Optional[str] = None,
    to: str = 'metacell',
    candidates: Union[str, ut.Vector] = 'candidate',
    deviants: Optional[Union[str, ut.Vector]] = 'cell_deviant_votes',
    target_metacell_size: int = pr.target_metacell_size,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.dissolve_cell_sizes,
    min_metacell_cells: int = pr.dissolve_min_metacell_cells,
    min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasFrame]:
    '''
    Dissolve too-small metacells based ``of`` some data (by default, the focus).

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Observation (Cell) Annotations
        ``<to>`` (default: {to})
            The integer index of the metacell each cell belongs to. The metacells are in no
            particular order. Cells with no metacell assignment are given a metacell index of
            ``-1``.

        ``dissolved`` (if ``intermediate``)
            A boolean mask of the cells which were in a dissolved metacell.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas data frame (indexed by the observation names).

    If ``intermediate`` (default: {intermediate}), also keep all all the intermediate data (e.g.
    sums) for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Mark all cells with non-zero ``deviants`` (default: {deviants}) as "outliers". This can be
       the name of a per-observation (cell) annotation, or an explicit boolean mask of cells, or a
       or ``None`` if there are no deviant cells to mark.

    2. Any metacell which has less cells than the ``min_metacell_cells`` is dissolved.

    3. We are trying to create metacells of size ``target_metacell_size``. Compute the sizes of the
       resulting metacells by summing the ``cell_sizes`` (default: {cell_sizes}) If the cell sizes
       is a string that contains ``<of>``, it is expanded using the name of the ``of`` data. If it
       is ``None``, each has a size of one.
       These parameters are typically identical to these passed
       to :py:func:`metacells.tools.candidates.compute_candidate_metacells`.

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
    of, level = ut.log_operation(LOG, adata, 'dissolve_metacells', of)

    dissolved_of_cells = np.zeros(adata.n_obs, dtype='bool')

    with ut.focus_on(ut.get_vo_data, adata, of,
                     layout='row_major', intermediate=intermediate) as data:

        candidate_of_cells = \
            ut.get_vector_parameter_data(LOG, adata, candidates,
                                         per='o', name='candidates')
        candidate_of_cells = np.copy(candidate_of_cells)
        assert candidate_of_cells is not None
        raw_candidates_count = np.max(candidate_of_cells) + 1
        LOG.debug('  candidates: %s', raw_candidates_count)

        LOG.debug('  min_metacell_cells: %s', min_metacell_cells)

        deviant_of_cells = \
            ut.get_vector_parameter_data(LOG, adata, deviants,
                                         per='o', name='deviants')

        cell_sizes = \
            ut.get_vector_parameter_data(LOG, adata, cell_sizes,
                                         per='o', name='cell_sizes')

        if deviant_of_cells is not None:
            candidate_of_cells[deviant_of_cells > 0] = -1
        candidate_of_cells = ut.compress_indices(candidate_of_cells)
        candidates_count = np.max(candidate_of_cells) + 1

        LOG.debug('  target_metacell_size: %s', target_metacell_size)
        fraction_of_genes = pp.get_fraction_per_var(adata).dense

        if min_robust_size_factor is None:
            min_robust_size = None
        else:
            min_robust_size = target_metacell_size * min_robust_size_factor
        LOG.debug('  min_robust_size: %s', min_robust_size)

        if min_convincing_size_factor is None:
            min_convincing_size = None
        else:
            min_convincing_size = \
                target_metacell_size * min_convincing_size_factor
        LOG.debug('  min_convincing_size: %s', min_convincing_size)
        if min_convincing_size_factor is not None:
            LOG.debug('  min_convincing_gene_fold_factor: %s',
                      min_convincing_gene_fold_factor)

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

    if LOG.isEnabledFor(level):
        metacells_count = np.max(metacell_of_cells) + 1
    else:
        metacells_count = -1

    if inplace:
        if intermediate:
            ut.set_o_data(adata, 'dissolved', dissolved_of_cells,
                          log_value=ut.mask_description)

        ut.set_o_data(adata, to, metacell_of_cells,
                      log_value=ut.groups_description)
        return None

    if LOG.isEnabledFor(level):
        LOG.log(level, '  dissolved: %s',
                ut.mask_description(dissolved_of_cells))
        LOG.log(level, '  %s: %s', to,
                ut.ratio_description(metacells_count, raw_candidates_count))

    obs_frame = pd.DataFrame(index=adata.obs_names)
    obs_frame['dissolved'] = dissolved_of_cells
    obs_frame[to] = metacell_of_cells
    return obs_frame


def _keep_candidate(
    adata: AnnData,
    candidate_index: int,
    *,
    data: ut.ProperMatrix,
    cell_sizes: Optional[ut.DenseVector],
    fraction_of_genes: ut.DenseVector,
    min_metacell_cells: int,
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

    if candidate_cell_indices.size < min_metacell_cells:
        LOG.debug('  - candidate: %s / %s cells: %s size: %s is: little',
                  candidate_index, candidates_count,
                  candidate_cell_indices.size, candidate_total_size)
        return False

    if min_robust_size is not None \
            and candidate_total_size >= min_robust_size:
        LOG.debug('  - candidate: %s / %s cells: %s size: %s is: robust',
                  candidate_index, candidates_count,
                  candidate_cell_indices.size, candidate_total_size)
        return True

    if min_convincing_size is None:
        LOG.debug('  - candidate: %s / %s cells: %s size: %s is: accepted',
                  candidate_index, candidates_count,
                  candidate_cell_indices.size, candidate_total_size)
        return True

    if candidate_total_size < min_convincing_size:
        LOG.debug('  - candidate: %s / %s cells: %s size: %s is: unconvincing',
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
            LOG.debug('  - candidate: %s / %s cells: %s size: %s is: convincing because %s',
                      candidate_index, candidates_count,
                      candidate_cell_indices.size,
                      candidate_total_size,
                      ', '.join(['%s: %s' % (name, fold_factor)
                                 for name, fold_factor
                                 in zip(adata.var_names[convincing_gene_indices],
                                        candidate_data_of_genes[convincing_gene_indices])]))
        else:
            LOG.debug('  - candidate: %s / %s cells: %s size: %s is: not convincing',
                      candidate_index, candidates_count,
                      candidate_cell_indices.size, candidate_total_size)

    return keep_candidate
