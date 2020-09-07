'''
Divide and Conquer
------------------
'''

import logging
from re import Pattern
from typing import Collection, Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.parameters as pr
import metacells.preprocessing as pp
import metacells.utilities as ut
from metacells.pipeline.direct import compute_direct_metacells

__all__ = [
    'compute_divide_and_conquer_metacells',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def compute_divide_and_conquer_metacells(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    target_pile_size: int = pr.target_pile_size,
    pile_min_split_size_factor: float = pr.pile_min_split_size_factor,
    pile_max_merge_size_factor: Optional[float] = pr.pile_max_merge_size_factor,
    feature_downsample_cell_quantile: float = pr.feature_downsample_cell_quantile,
    feature_min_gene_fraction: float = pr.feature_min_gene_fraction,
    feature_min_gene_relative_variance: float = pr.feature_min_gene_relative_variance,
    forbidden_gene_names: Optional[Collection[str]] = None,
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    cells_similarity_log_data: bool = pr.cells_similarity_log_data,
    cells_similarity_log_normalization: float = pr.cells_similarity_log_normalization,
    cells_repeated_similarity: bool = pr.cells_repeated_similarity,
    knn_k: Optional[int] = pr.knn_k,
    knn_balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    knn_incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    knn_outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    candidates_partition_method: 'ut.PartitionMethod' = pr.candidates_partition_method,
    candidates_min_split_size_factor: Optional[float] = pr.candidates_min_split_size_factor,
    candidates_max_merge_size_factor: Optional[float] = pr.candidates_max_merge_size_factor,
    must_complete_cover: bool = False,
    outliers_min_gene_fold_factor: float = pr.outliers_min_gene_fold_factor,
    outliers_max_gene_fraction: float = pr.outliers_max_gene_fraction,
    outliers_max_cell_fraction: float = pr.outliers_max_cell_fraction,
    outliers_phase_2_max_cell_fraction: float = pr.outliers_phase_2_max_cell_fraction,
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    target_metacell_size: int = pr.target_metacell_size,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    random_seed: int = pr.random_seed,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Directly compute metacells.

    This is the body of the metacells methodology.

    **Input**

    The presumably "clean" :py:func:`metacells.utilities.annotation.setup` annotated ``adata``.

    All the computations will use the ``of`` data (by default, the focus).

    **Returns**

    Observation (Cell) Annotations
        ``metacell``
            The integer index of the metacell each cell belongs to. The metacells are in no
            particular order. Cells with no metacell assignment are given a metacell index of
            ``-1``.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the observation names).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. If the data is smaller than ``target_pile_size`` (default: {target_pile_size}) times the
       ``pile_min_split_size_factor`` (default: {pile_min_split_size_factor}), then just invoke
       :py:func:`metacells.pipeline.direct.compute_direct_metacells` and be done. See
       :py:func:`metacells.pipeline.direct.compute_direct_metacells` for details on the effect of
       the parameters.

    *Phase 1:*:

    2. Split the data into random piles of roughly the ``target_pile_size``, and invoke
       :py:func:`metacells.pipeline.direct.compute_direct_metacells` on each one.

    3. Collect the outlier cells from all the piles and recursively invoke
       :py:func:`compute_divide_and_conquer_metacells` on the result. Eventually this recursion
       will reduce the size to a single pile; in this case, set ``must_complete_cover`` to ``True``
       so that every cell will be assigned some phase-1 metacell.

    4. Sum the cells into phase-1 metacells, and invoke
       :py:func:`compute_divide_and_conquer_metacells` on the result, using the ``target_pile_size``
       (default: {target_pile_size}), ``pile_min_split_size_factor`` (default: {pile_min_split_size_factor})
       and ``pile_max_merge_size_factor`` (default: {pile_max_merge_size_factor})

       This will group the phase-1
       metacells to piles, each containing roughly ``target_pile_size`` cells.

    *Phase 2:*

    5. Split the data into the phase-1 piles, and invoke
       :py:func:`metacells.pipeline.direct.compute_direct_metacells` on each one. In this second
       phase, use the ``outliers_phase_2_max_cell_fraction`` (default:
       {outliers_phase_2_max_cell_fraction}). We are more aggressive weeding out outliers in the
       second phase to obtain higher quality metacells, which is possible since all the cells in
       each pile are known to be similar to each other.

    6. Collect the outlier cells from all the piles and recursively invoke
       :py:func:`compute_divide_and_conquer_metacells` on the result. Eventually this recursion will
       reduce the size to a single pile. As long as we are only on the (possibly recursive) second
       phase, we do not force every cell to belong to a metacell, so the final result may include
       outliers.
    '''
    level = ut.log_pipeline_step(LOG, adata,
                                 'compute_divide_and_conquer_metacells')

    LOG.log(level, '  target_pile_size: %s', target_pile_size)
    LOG.log(level, '  pile_min_split_size_factor: %s',
            pile_min_split_size_factor)

    pile_min_split_size = target_pile_size * pile_min_split_size_factor
    LOG.debug('  pile_min_split_size: %s', pile_min_split_size)
    if adata.n_obs < pile_min_split_size:
        return \
            compute_direct_metacells(adata, of,
                                     feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                     feature_min_gene_fraction=feature_min_gene_fraction,
                                     feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                     forbidden_gene_names=forbidden_gene_names,
                                     forbidden_gene_patterns=forbidden_gene_patterns,
                                     cells_similarity_log_data=cells_similarity_log_data,
                                     cells_similarity_log_normalization=cells_similarity_log_normalization,
                                     cells_repeated_similarity=cells_repeated_similarity,
                                     knn_k=knn_k,
                                     knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                                     knn_incoming_degree_factor=knn_incoming_degree_factor,
                                     knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                                     candidates_partition_method=candidates_partition_method,
                                     candidates_min_split_size_factor=candidates_min_split_size_factor,
                                     candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                                     must_complete_cover=must_complete_cover,
                                     outliers_min_gene_fold_factor=outliers_min_gene_fold_factor,
                                     outliers_max_gene_fraction=outliers_max_gene_fraction,
                                     outliers_max_cell_fraction=outliers_max_cell_fraction,
                                     dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                     dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                     dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                     target_metacell_size=target_metacell_size,
                                     cell_sizes=cell_sizes,
                                     random_seed=random_seed,
                                     inplace=inplace,
                                     intermediate=intermediate)

    pile_of_cells = ut.random_piles(adata.n_obs, target_pile_size)

    with ut.timed_step('.phase-1'):
        metacell_of_cells = \
            _compute_piled_metacells(adata, of,
                                     phase='phase-1',
                                     pile_of_cells=pile_of_cells,
                                     target_pile_size=target_pile_size,
                                     pile_min_split_size_factor=pile_min_split_size_factor,
                                     pile_max_merge_size_factor=pile_max_merge_size_factor,
                                     feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                     feature_min_gene_fraction=feature_min_gene_fraction,
                                     feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                     forbidden_gene_names=forbidden_gene_names,
                                     forbidden_gene_patterns=forbidden_gene_patterns,
                                     cells_similarity_log_data=cells_similarity_log_data,
                                     cells_similarity_log_normalization=cells_similarity_log_normalization,
                                     cells_repeated_similarity=cells_repeated_similarity,
                                     knn_k=knn_k,
                                     knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                                     knn_incoming_degree_factor=knn_incoming_degree_factor,
                                     knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                                     candidates_partition_method=candidates_partition_method,
                                     candidates_min_split_size_factor=candidates_min_split_size_factor,
                                     candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                                     must_complete_cover=must_complete_cover,
                                     outliers_min_gene_fold_factor=outliers_min_gene_fold_factor,
                                     outliers_max_gene_fraction=outliers_max_gene_fraction,
                                     outliers_max_cell_fraction=outliers_max_cell_fraction,
                                     dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                     dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                     dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                     target_metacell_size=target_metacell_size,
                                     cell_sizes=cell_sizes,
                                     random_seed=random_seed)

    with ut.timed_step('.recurse'):
        ut.log_operation(LOG, adata, 'recurse', of)
        mdata = pp.group_obs_data(adata, of=of, groups='metacell', tmp=True,
                                  intermediate=intermediate,
                                  name='%s.metacells'
                                  % (ut.get_name(adata) or 'clean'))
        if mdata is None:
            raise ValueError('Empty metacells data, giving up')
        assert mdata is not None
        pile_of_metacells = \
            compute_divide_and_conquer_metacells(mdata,
                                                 target_pile_size=target_pile_size,
                                                 pile_min_split_size_factor=pile_min_split_size_factor,
                                                 feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                                 feature_min_gene_fraction=feature_min_gene_fraction,
                                                 feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                                 forbidden_gene_names=forbidden_gene_names,
                                                 forbidden_gene_patterns=forbidden_gene_patterns,
                                                 cells_similarity_log_data=cells_similarity_log_data,
                                                 cells_similarity_log_normalization=cells_similarity_log_normalization,
                                                 cells_repeated_similarity=cells_repeated_similarity,
                                                 knn_k=knn_k,
                                                 knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                                                 knn_incoming_degree_factor=knn_incoming_degree_factor,
                                                 knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                                                 candidates_partition_method=candidates_partition_method,
                                                 candidates_min_split_size_factor=pile_min_split_size_factor,
                                                 candidates_max_merge_size_factor=pile_max_merge_size_factor,
                                                 must_complete_cover=True,
                                                 outliers_min_gene_fold_factor=outliers_min_gene_fold_factor,
                                                 outliers_max_gene_fraction=outliers_max_gene_fraction,
                                                 outliers_max_cell_fraction=outliers_max_cell_fraction,
                                                 dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                                 dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                                 dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                                 target_metacell_size=target_metacell_size,
                                                 cell_sizes='grouped',
                                                 random_seed=random_seed,
                                                 inplace=False,
                                                 intermediate=intermediate)
        assert pile_of_metacells is not None

    pile_of_cells = \
        ut.group_piles(metacell_of_cells.values,
                       pile_of_metacells.values)

    with ut.timed_step('.phase-2'):
        metacell_of_cells = \
            _compute_piled_metacells(adata, of,
                                     phase='phase-2',
                                     pile_of_cells=pile_of_cells,
                                     target_pile_size=target_pile_size,
                                     pile_min_split_size_factor=pile_min_split_size_factor,
                                     pile_max_merge_size_factor=pile_max_merge_size_factor,
                                     feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                     feature_min_gene_fraction=feature_min_gene_fraction,
                                     feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                     forbidden_gene_names=forbidden_gene_names,
                                     forbidden_gene_patterns=forbidden_gene_patterns,
                                     cells_similarity_log_data=cells_similarity_log_data,
                                     cells_similarity_log_normalization=cells_similarity_log_normalization,
                                     cells_repeated_similarity=cells_repeated_similarity,
                                     knn_k=knn_k,
                                     knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                                     knn_incoming_degree_factor=knn_incoming_degree_factor,
                                     knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                                     candidates_partition_method=candidates_partition_method,
                                     candidates_min_split_size_factor=candidates_min_split_size_factor,
                                     candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                                     must_complete_cover=must_complete_cover,
                                     outliers_min_gene_fold_factor=outliers_min_gene_fold_factor,
                                     outliers_max_gene_fraction=outliers_max_gene_fraction,
                                     outliers_max_cell_fraction=outliers_phase_2_max_cell_fraction,
                                     dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                     dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                     dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                     target_metacell_size=target_metacell_size,
                                     cell_sizes=cell_sizes,
                                     random_seed=random_seed)

    if inplace:
        ut.set_o_data(adata, 'metacell', metacell_of_cells,
                      log_value=lambda: str(np.max(metacell_of_cells) + 1))
        return None

    return pd.Series(metacell_of_cells, adata.obs_names)


def _compute_piled_metacells(
    adata: AnnData,
    of: Optional[str],
    *,
    phase: str,
    pile_of_cells: ut.DenseVector,
    target_pile_size: int,
    pile_min_split_size_factor: float,
    pile_max_merge_size_factor: Optional[float],
    feature_downsample_cell_quantile: float,
    feature_min_gene_fraction: float,
    feature_min_gene_relative_variance: float,
    forbidden_gene_names: Optional[Collection[str]],
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]],
    cells_similarity_log_data: bool,
    cells_similarity_log_normalization: float,
    cells_repeated_similarity: bool,
    knn_k: Optional[int],
    knn_balanced_ranks_factor: float,
    knn_incoming_degree_factor: float,
    knn_outgoing_degree_factor: float,
    candidates_partition_method: 'ut.PartitionMethod',
    candidates_min_split_size_factor: Optional[float],
    candidates_max_merge_size_factor: Optional[float],
    must_complete_cover: bool,
    outliers_min_gene_fold_factor: float,
    outliers_max_gene_fraction: float,
    outliers_max_cell_fraction: float,
    dissolve_min_robust_size_factor: Optional[float],
    dissolve_min_convincing_size_factor: Optional[float],
    dissolve_min_convincing_gene_fold_factor: float,
    target_metacell_size: int,
    cell_sizes: Optional[Union[str, ut.Vector]],
    random_seed: int,
) -> pd.Series:
    ut.log_operation(LOG, adata, phase, of)
    piles_count = np.max(pile_of_cells) + 1
    LOG.debug('  piles_count: %s', piles_count)
    assert piles_count > 1

    def _compute_pile_metacells(pile_index: int) -> pd.Series:
        pile_cells_mask = pile_of_cells[pile_of_cells] == pile_index
        assert np.sum(pile_cells_mask) > 0
        name = '%s.%s.pile-%s/%s' % (ut.get_name(adata)
                                     or 'clean', phase, pile_index, piles_count)
        pdata = ut.slice(adata, obs=pile_cells_mask, name=name, tmp=True)
        metacell_of_cells_of_pile = \
            compute_direct_metacells(pdata, of,
                                     feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                     feature_min_gene_fraction=feature_min_gene_fraction,
                                     feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                     forbidden_gene_names=forbidden_gene_names,
                                     forbidden_gene_patterns=forbidden_gene_patterns,
                                     cells_similarity_log_data=cells_similarity_log_data,
                                     cells_similarity_log_normalization=cells_similarity_log_normalization,
                                     cells_repeated_similarity=cells_repeated_similarity,
                                     knn_k=knn_k,
                                     knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                                     knn_incoming_degree_factor=knn_incoming_degree_factor,
                                     knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                                     candidates_partition_method=candidates_partition_method,
                                     candidates_min_split_size_factor=candidates_min_split_size_factor,
                                     candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                                     must_complete_cover=False,
                                     outliers_min_gene_fold_factor=outliers_min_gene_fold_factor,
                                     outliers_max_gene_fraction=outliers_max_gene_fraction,
                                     outliers_max_cell_fraction=outliers_max_cell_fraction,
                                     dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                     dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                     dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                     target_metacell_size=target_metacell_size,
                                     cell_sizes=cell_sizes,
                                     random_seed=random_seed,
                                     inplace=False)
        assert metacell_of_cells_of_pile is not None
        metacell_of_cells_of_pile.index = np.where(pile_cells_mask)
        return metacell_of_cells_of_pile

    with ut.timed_step('.piles'):
        metacell_of_cells_of_piles = \
            ut.parallel_map(_compute_pile_metacells, piles_count)

    metacells_count = 0
    metacell_of_cells = np.full(adata.n_obs, -1)
    for metacell_of_cells_of_pile in metacell_of_cells_of_piles:
        pile_metacells_count = np.max(metacell_of_cells_of_pile) + 1
        metacell_of_cells_of_pile[metacell_of_cells_of_pile >= 0] += \
            metacells_count
        metacells_count += pile_metacells_count
        metacell_of_cells[metacell_of_cells_of_pile.index] = \
            metacell_of_cells_of_pile.values

    with ut.timed_step('.outliers'):
        outlier_cells = metacell_of_cells < 0
        name = '%s.%s.outliers' % (ut.get_name(adata) or 'clean', phase)
        odata = ut.slice(adata, obs=outlier_cells, name=name, tmp=True)
        metacells_of_outlier_cells = \
            compute_divide_and_conquer_metacells(odata, of,
                                                 target_pile_size=target_pile_size,
                                                 pile_min_split_size_factor=pile_min_split_size_factor,
                                                 pile_max_merge_size_factor=pile_max_merge_size_factor,
                                                 feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                                 feature_min_gene_fraction=feature_min_gene_fraction,
                                                 feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                                 forbidden_gene_names=forbidden_gene_names,
                                                 forbidden_gene_patterns=forbidden_gene_patterns,
                                                 cells_similarity_log_data=cells_similarity_log_data,
                                                 cells_similarity_log_normalization=cells_similarity_log_normalization,
                                                 cells_repeated_similarity=cells_repeated_similarity,
                                                 knn_k=knn_k,
                                                 knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                                                 knn_incoming_degree_factor=knn_incoming_degree_factor,
                                                 knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                                                 candidates_partition_method=candidates_partition_method,
                                                 candidates_min_split_size_factor=candidates_min_split_size_factor,
                                                 candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                                                 must_complete_cover=must_complete_cover,
                                                 outliers_min_gene_fold_factor=outliers_min_gene_fold_factor,
                                                 outliers_max_gene_fraction=outliers_max_gene_fraction,
                                                 outliers_max_cell_fraction=outliers_max_cell_fraction,
                                                 dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                                 dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                                 dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                                 target_metacell_size=target_metacell_size,
                                                 cell_sizes=cell_sizes,
                                                 random_seed=random_seed,
                                                 inplace=False)

    assert metacells_of_outlier_cells is not None
    metacells_of_outlier_cells[metacells_of_outlier_cells.values >=
                               0] += metacells_count
    metacell_of_cells[outlier_cells] = metacells_of_outlier_cells.values

    return metacell_of_cells
