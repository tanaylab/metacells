'''
Divide and Conquer
------------------
'''

import logging
from re import Pattern
from typing import Collection, List, Optional, Tuple, Union

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
    deviants_min_gene_fold_factor: float = pr.deviants_min_gene_fold_factor,
    deviants_max_gene_fraction: float = pr.deviants_max_gene_fraction,
    deviants_max_cell_fraction: float = pr.deviants_max_cell_fraction,
    deviants_phase_2_max_cell_fraction: float = pr.deviants_phase_2_max_cell_fraction,
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    target_metacell_size: int = pr.target_metacell_size,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    random_seed: int = pr.random_seed,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[Tuple[ut.PandasFrame, ut.PandasFrame, ut.PandasFrame]]:
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

        ``cell_pile``
            The index of the (final) pile the cell was placed in when invoking
            :py:func:`metacells.pipeline.direct.compute_direct_metacells`.

        ``cell_directs``
            The total number of invocations of
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` the cell participated in.

        ``cell_outliers``
            The total number of times the cell was an outlier (not included in any metacell) across
            all the invocations of :py:func:`metacells.pipeline.direct.compute_direct_metacells` the
            cell participated in.

        ``cell_deviant_votes``
            The sum of the number of genes that were the reason the cell was marked as an deviant
            across all the invocations of
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` the cell participated in.

        ``cell_dissolves``
            The number of times the cell was in a dissolved metacell across all the invocations of
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` the cell participated in.

    Variable (Gene) Annotations
        ``feature_gene``
            A float value per gene which is 1.0 if it was selected as a feature and 0.0 if it
            wasn't. This isn't a simple boolean to be compatible with the divide-and-conquer
            algorithm.

        ``gene_deviant_votes``
            The sum of the number of cells each gene marked as an deviant
            across all the invocations of
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` the cell participated in.

    Variable-Any (Gene) Annotations
        ``gene_pile_feature``
            A sparse boolean 2D mask which is ``True`` for each feature gene for each (final) pile.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as three pandas frames (indexed by the observation names
    and the variable names).

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
       phase, use the ``deviants_phase_2_max_cell_fraction`` (default:
       {deviants_phase_2_max_cell_fraction}). We are more aggressive weeding out deviants in the
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
                                     deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                     deviants_max_gene_fraction=deviants_max_gene_fraction,
                                     deviants_max_cell_fraction=deviants_max_cell_fraction,
                                     dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                     dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                     dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                     target_metacell_size=target_metacell_size,
                                     cell_sizes=cell_sizes,
                                     random_seed=random_seed,
                                     inplace=inplace,
                                     intermediate=intermediate)

    phase_pile_of_cells = ut.random_piles(adata.n_obs, target_pile_size)

    feature_of_genes = np.zeros(adata.n_vars, dtype='int32')
    deviant_votes_of_genes = np.zeros(adata.n_vars, dtype='int32')
    directs_of_cells = np.zeros(adata.n_obs, dtype='int32')
    deviant_votes_of_cells = np.zeros(adata.n_obs, dtype='int32')
    dissolved_of_cells = np.zeros(adata.n_obs, dtype='int32')
    outliers_of_cells = np.zeros(adata.n_obs, dtype='int32')

    with ut.timed_step('.phase-1'):
        metacell_of_cells, _ = \
            _compute_piled_metacells(adata, of,
                                     phase='phase-1',
                                     feature_of_genes=feature_of_genes,
                                     deviant_votes_of_genes=deviant_votes_of_genes,
                                     pile_of_cells=None,
                                     directs_of_cells=directs_of_cells,
                                     deviant_votes_of_cells=deviant_votes_of_cells,
                                     dissolved_of_cells=dissolved_of_cells,
                                     outliers_of_cells=outliers_of_cells,
                                     phase_pile_of_cells=phase_pile_of_cells,
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
                                     deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                     deviants_max_gene_fraction=deviants_max_gene_fraction,
                                     deviants_max_cell_fraction=deviants_max_cell_fraction,
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
        results = \
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
                                                 deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                                 deviants_max_gene_fraction=deviants_max_gene_fraction,
                                                 deviants_max_cell_fraction=deviants_max_cell_fraction,
                                                 dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                                 dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                                 dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                                 target_metacell_size=target_metacell_size,
                                                 cell_sizes='grouped',
                                                 random_seed=random_seed,
                                                 inplace=False,
                                                 intermediate=intermediate)
        assert results is not None
        pile_of_metacells = ut.to_dense_vector(results[0]['metacell'])
        phase_pile_of_cells = \
            ut.group_piles(metacell_of_cells, pile_of_metacells)

    pile_of_cells = np.full(adata.n_obs, -1, dtype='int32')

    with ut.timed_step('.phase-2'):
        metacell_of_cells, feature_of_piles_of_genes = \
            _compute_piled_metacells(adata, of,
                                     phase='phase-2',
                                     feature_of_genes=feature_of_genes,
                                     deviant_votes_of_genes=deviant_votes_of_genes,
                                     pile_of_cells=pile_of_cells,
                                     directs_of_cells=directs_of_cells,
                                     deviant_votes_of_cells=deviant_votes_of_cells,
                                     dissolved_of_cells=dissolved_of_cells,
                                     outliers_of_cells=outliers_of_cells,
                                     phase_pile_of_cells=phase_pile_of_cells,
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
                                     deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                     deviants_max_gene_fraction=deviants_max_gene_fraction,
                                     deviants_max_cell_fraction=deviants_phase_2_max_cell_fraction,
                                     dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                     dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                     dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                     target_metacell_size=target_metacell_size,
                                     cell_sizes=cell_sizes,
                                     random_seed=random_seed)

    if inplace:
        ut.set_v_data(adata, 'feature_gene',
                      ut.to_dense_vector(feature_of_genes))

        ut.set_v_data(adata, 'gene_deviant_votes', deviant_votes_of_genes,
                      log_value=lambda:
                      ut.mask_description(deviant_votes_of_genes > 0))

        ut.set_o_data(adata, 'cell_pile', pile_of_cells)

        ut.set_o_data(adata, 'cell_directs', directs_of_cells)

        ut.set_o_data(adata, 'cell_deviant_votes', deviant_votes_of_cells,
                      log_value=lambda:
                      ut.mask_description(deviant_votes_of_cells > 0))

        ut.set_o_data(adata, 'cell_dissolves', dissolved_of_cells,
                      log_value=lambda:
                      ut.mask_description(dissolved_of_cells > 0))

        ut.set_o_data(adata, 'cell_outliers', outliers_of_cells,
                      log_value=lambda:
                      ut.mask_description(metacell_of_cells < 0))

        ut.set_o_data(adata, 'metacell', metacell_of_cells,
                      log_value=lambda: str(np.max(metacell_of_cells) + 1))

        ut.set_va_data(adata, 'gene_pile_feature',
                       feature_of_piles_of_genes.sparse.get_coo().tocsc())  # type: ignore

        return None

    var_frame = pd.DataFrame(index=adata.var_names)
    var_frame['feature_gene'] = ut.to_dense_vector(feature_of_genes)
    var_frame['gene_deviant_votes'] = deviant_votes_of_genes

    obs_frame = pd.DataFrame(index=adata.obs_names)
    obs_frame['cell_pile'] = pile_of_cells
    obs_frame['cell_directs'] = directs_of_cells
    obs_frame['cell_deviant_votes'] = deviant_votes_of_cells
    obs_frame['cell_dissolves'] = dissolved_of_cells
    obs_frame['cell_outliers'] = outliers_of_cells
    obs_frame['metacell'] = metacell_of_cells
    return obs_frame, var_frame, feature_of_piles_of_genes


def _compute_piled_metacells(
    adata: AnnData,
    of: Optional[str],
    *,
    phase: str,
    feature_of_genes: ut.DenseVector,
    deviant_votes_of_genes: ut.DenseVector,
    pile_of_cells: Optional[ut.DenseVector],
    directs_of_cells: ut.DenseVector,
    deviant_votes_of_cells: ut.DenseVector,
    dissolved_of_cells: ut.DenseVector,
    outliers_of_cells: ut.DenseVector,
    phase_pile_of_cells: ut.DenseVector,
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
    deviants_min_gene_fold_factor: float,
    deviants_max_gene_fraction: float,
    deviants_max_cell_fraction: float,
    dissolve_min_robust_size_factor: Optional[float],
    dissolve_min_convincing_size_factor: Optional[float],
    dissolve_min_convincing_gene_fold_factor: float,
    target_metacell_size: int,
    cell_sizes: Optional[Union[str, ut.Vector]],
    random_seed: int,
) -> Tuple[ut.DenseVector, ut.PandasFrame]:
    ut.log_operation(LOG, adata, phase, of)
    phase_piles_count = np.max(phase_pile_of_cells) + 1
    LOG.debug('  piles_count: %s', phase_piles_count)
    assert phase_piles_count > 1

    def _compute_pile_metacells(phase_pile_index: int) -> pd.Series:
        pile_cells_mask = phase_pile_of_cells[phase_pile_of_cells] == phase_pile_index
        assert np.sum(pile_cells_mask) > 0
        name = '%s.%s.pile-%s/%s' \
            % (ut.get_name(adata) or 'clean',
               phase, phase_pile_index, phase_piles_count)
        pdata = ut.slice(adata, obs=pile_cells_mask, name=name, tmp=True)
        results_of_pile = \
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
                                     deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                     deviants_max_gene_fraction=deviants_max_gene_fraction,
                                     deviants_max_cell_fraction=deviants_max_cell_fraction,
                                     dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                     dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                     dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                     target_metacell_size=target_metacell_size,
                                     cell_sizes=cell_sizes,
                                     random_seed=random_seed,
                                     inplace=False)
        assert results_of_pile is not None
        results_of_pile[0].index = np.where(pile_cells_mask)[0]
        results_of_pile[2].columns = [phase_pile_index]  # type: ignore
        return results_of_pile

    with ut.timed_step('.piles'):
        results_of_piles = \
            ut.parallel_map(_compute_pile_metacells, phase_piles_count)

    metacells_count = 0
    piles_count = 0
    metacell_of_cells = np.full(adata.n_obs, -1)

    feature_of_piles_of_genes_list: List[pd.Frame] = []
    for results_of_pile in results_of_piles:
        metacells_count, piles_count = \
            _collect_results(results_of_pile,
                             metacells_count=metacells_count,
                             piles_count=piles_count,
                             feature_of_genes=feature_of_genes,
                             deviant_votes_of_genes=deviant_votes_of_genes,
                             pile_of_cells=pile_of_cells,
                             directs_of_cells=directs_of_cells,
                             deviant_votes_of_cells=deviant_votes_of_cells,
                             dissolved_of_cells=dissolved_of_cells,
                             outliers_of_cells=outliers_of_cells,
                             metacell_of_cells=metacell_of_cells,
                             feature_of_piles_of_genes_list=feature_of_piles_of_genes_list)

    outlier_cells = metacell_of_cells < 0
    if np.any(outlier_cells):
        with ut.timed_step('.outliers'):
            name = '%s.%s.outliers' % (ut.get_name(adata) or 'clean', phase)
            odata = ut.slice(adata, obs=outlier_cells, name=name, tmp=True)
            results_of_outliers = \
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
                                                     deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                                     deviants_max_gene_fraction=deviants_max_gene_fraction,
                                                     deviants_max_cell_fraction=deviants_max_cell_fraction,
                                                     dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                                     dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                                     dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                                     target_metacell_size=target_metacell_size,
                                                     cell_sizes=cell_sizes,
                                                     random_seed=random_seed,
                                                     inplace=False)
            assert results_of_outliers is not None
            results_of_outliers[0].index = np.where(outlier_cells)[0]
            metacells_count, piles_count = \
                _collect_results(results_of_outliers,
                                 metacells_count=metacells_count,
                                 piles_count=piles_count,
                                 feature_of_genes=feature_of_genes,
                                 deviant_votes_of_genes=deviant_votes_of_genes,
                                 pile_of_cells=pile_of_cells,
                                 directs_of_cells=directs_of_cells,
                                 deviant_votes_of_cells=deviant_votes_of_cells,
                                 dissolved_of_cells=dissolved_of_cells,
                                 outliers_of_cells=outliers_of_cells,
                                 metacell_of_cells=metacell_of_cells,
                                 feature_of_piles_of_genes_list=feature_of_piles_of_genes_list)

    return metacell_of_cells, pd.concat(feature_of_piles_of_genes_list, axis=1)


def _collect_results(
    results: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    *,
    metacells_count: int,
    piles_count: int,
    feature_of_genes: ut.DenseVector,
    deviant_votes_of_genes: ut.DenseVector,
    pile_of_cells: Optional[ut.DenseVector],
    directs_of_cells: ut.DenseVector,
    deviant_votes_of_cells: ut.DenseVector,
    dissolved_of_cells: ut.DenseVector,
    outliers_of_cells: ut.DenseVector,
    metacell_of_cells: ut.DenseVector,
    feature_of_piles_of_genes_list: List[pd.DataFrame],
) -> Tuple[int, int]:
    cell_results, gene_results, feature_of_piles_of_genes = results

    directs_of_cells[cell_results.index] += 1
    deviant_votes_of_cells[cell_results.index] += cell_results['cell_deviant_votes']
    dissolved_of_cells[cell_results.index] += cell_results['cell_dissolves']
    outliers_of_cells[cell_results.index] += cell_results['cell_outliers']

    metacell_results = ut.to_dense_vector(cell_results['metacell'])
    results_metacells_count = np.max(metacell_results) + 1
    metacell_results[metacell_results >= 0] += metacells_count
    metacell_of_cells[cell_results.index] = metacell_results

    feature_of_genes += gene_results['feature_gene']
    deviant_votes_of_genes += gene_results['gene_deviant_votes']

    pile_results = ut.to_dense_vector(cell_results['cell_pile'])
    results_piles_count = np.max(pile_results) + 1
    assert results_piles_count == len(feature_of_piles_of_genes.columns)

    if pile_of_cells is not None:
        pile_results += piles_count
        pile_of_cells[cell_results.index] = pile_results

    feature_of_piles_of_genes.columns += piles_count
    feature_of_piles_of_genes_list.append(feature_of_piles_of_genes)

    return metacells_count + results_metacells_count, \
        piles_count + results_piles_count
