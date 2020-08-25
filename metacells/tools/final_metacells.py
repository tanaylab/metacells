'''
Finalize the grouping of cells to metacells.
'''

import logging
from typing import Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'finalize_metacells',
]


LOG = logging.getLogger(__name__)


@ut.expand_doc()
@ut.timed_call()
def finalize_metacells(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    adata: AnnData,
    *,
    of: Optional[str] = None,
    communities: Union[str, ut.Vector] = 'candidate_metacell',
    outliers: Optional[Union[str, ut.Vector]] = 'outlier_cells',
    target_metacell_size: Optional[int],
    cell_sizes: Optional[Union[str, ut.Vector]],
    minimal_robust_size_factor: Optional[float] = 0.5,
    minimal_convincing_size_factor: Optional[float] = 0.25,
    minimal_convincing_gene_fold_factor: float = 3.0,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Finalize the grouping of cells to metacells based ``of`` some data (by default, the focus).

    **Input**

    A :py:func:`metacells.utilities.preparation.prepare`-ed annotated ``adata``, where the
    observations are cells and the variables are genes.

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
       per-observation (cell) annotation, or an explicit boolean mask of cells, or ``None`` if there
       are no outlier cells to mark.

    2. If a ``target_metacell_size`` is specified, compute the sizes of the resulting metacells
       by summing the ``cell_sizes``, which again can be the name of a per-observation (cell)
       annotation, or an explicit sizes vector. If it is ``None``, then each cell is given a size of
       one. These parameters are typically identical to these passed to
       ``metacells.tools.compute_candidate_metacells``.

    3. If both ``target_metacell_size`` and ``minimal_robust_size_factor`` (default:
       {minimal_robust_size_factor}) are specified, then any metacell whose total size is at least
       ``target_metacell_size * minimal_robust_size_factor`` is preserved.

    3. If both ``target_metacell_size`` and ``minimal_convincing_size_factor`` (default:
       {minimal_convincing_size_factor}) are specified, then any remaining metacells whose size is
       at least ``target_metacell_size * minimal_convincing_size_factor`` are preserved, given they
       contain at least one gene whose fold factor (log2((actual + 1) / (expected + 1))) is at least
       ``minimal_convincing_gene_fold_factor`` (default: {minimal_convincing_gene_fold_factor}).
       That is, we only preserve these smaller metacells if there is at least one gene whose
       expression is significantly different from the mean of the population.

    4 . Any remaining metacell is dissolved into outlier cells.
    '''
    LOG.debug('finalize_metacells...')
    with ut.focus_on(ut.get_vo_data, adata, of,
                     layout='row_major', intermediate=intermediate) as data:
        genes_count = data.shape[1]

        if isinstance(communities, str):
            communities = ut.get_o_data(adata, communities)

        if isinstance(outliers, str):
            outliers = ut.get_o_data(adata, outliers)

        metacells_of_cells = ut.to_dense_vector(communities, copy=True)
        if outliers is not None:
            metacells_of_cells[outliers] = -1
        metacells_of_cells = ut.compress_indices(metacells_of_cells)

        candidate_metacells_count = np.max(metacells_of_cells) + 1
        LOG.info('  candidate_metacells: %s', candidate_metacells_count)

        if target_metacell_size is not None:  # pylint: disable=too-many-nested-blocks
            LOG.debug('  target_metacell_size: %s', target_metacell_size)
            fraction_of_genes = ut.get_fraction_per_var(adata).proper

            if minimal_robust_size_factor is None:
                minimal_robust_size = None
            else:
                minimal_robust_size = \
                    target_metacell_size * minimal_robust_size_factor
                LOG.info('  minimal_robust_size: %s', minimal_robust_size)

            if minimal_convincing_size_factor is None:
                minimal_convincing_size = None
            else:
                minimal_convincing_size = \
                    target_metacell_size * minimal_convincing_size_factor
                LOG.info('  minimal_convincing_size: %s',
                         minimal_convincing_size)
                LOG.info('  minimal_convincing_gene_fold_factor: %s',
                         minimal_convincing_gene_fold_factor)

            if minimal_robust_size is not None and minimal_convincing_size is not None:
                for metacell_index in range(candidate_metacells_count):
                    metacell_cell_indices = \
                        np.where(metacells_of_cells == metacell_index)[0]
                    if cell_sizes is None:
                        metacell_total_size = metacell_cell_indices.size
                    else:
                        if isinstance(cell_sizes, str):
                            cell_sizes = ut.get_o_data(adata, cell_sizes)
                        cell_sizes = ut.to_dense_vector(cell_sizes)
                        metacell_total_size = \
                            np.sum(cell_sizes[metacell_cell_indices])

                    if minimal_robust_size is not None \
                            and metacell_total_size >= minimal_robust_size:
                        LOG.debug('  - metacell: %s size: %s is: robust',
                                  metacell_index, metacell_total_size)
                        continue

                    if minimal_convincing_size is None:
                        LOG.debug('  - metacell: %s size: %s is: accepted',
                                  metacell_index, metacell_total_size)
                        continue

                    if metacell_total_size < minimal_convincing_size:
                        LOG.debug('  - metacell: %s size: %s is: unconvincing',
                                  metacell_index, metacell_total_size)
                        keep_metacell = False
                    else:
                        metacell_data = data[metacell_cell_indices, :]
                        metacell_data_of_genes = \
                            ut.to_dense_vector(metacell_data.sum(axis=0))
                        assert metacell_data_of_genes.size == genes_count
                        metacell_total = np.sum(metacell_data_of_genes)
                        metacell_expected_of_genes = fraction_of_genes * metacell_total
                        metacell_expected_of_genes += 1
                        metacell_data_of_genes += 1
                        metacell_data_of_genes /= metacell_expected_of_genes
                        np.log2(metacell_data_of_genes,
                                out=metacell_data_of_genes)
                        convincing_genes_mask = \
                            metacell_data_of_genes >= minimal_convincing_gene_fold_factor
                        keep_metacell = np.any(convincing_genes_mask)

                        if LOG.isEnabledFor(logging.DEBUG):
                            convincing_gene_indices = \
                                np.where(convincing_genes_mask)[0]
                            if keep_metacell:
                                LOG.debug('  - metacell: %s size: %s is: convincing because %s',
                                          metacell_index, metacell_total_size,
                                          ', '.join(['%s: %s' % (name, fold_factor)
                                                     for name, fold_factor
                                                     in zip(adata.var_names[convincing_gene_indices],
                                                            metacell_data_of_genes[convincing_gene_indices])]))
                            else:
                                LOG.debug('  - metacell_index: %s size: %s is: not convincing',
                                          metacell_index, metacell_total_size)

                    if not keep_metacell:
                        metacells_of_cells[metacell_cell_indices] = -1

            metacells_of_cells = ut.compress_indices(metacells_of_cells)

    if LOG.isEnabledFor(logging.INFO):
        metacells_count = np.max(metacells_of_cells) + 1
        LOG.info('finalize_metacells: %s / %s',
                 metacells_count, candidate_metacells_count)

    if inplace:
        adata.obs['metacell'] = metacells_of_cells
        ut.safe_slicing_data('metacell', ut.NEVER_SAFE)
        return None

    return pd.Series(metacells_of_cells, adata.obs_names)
