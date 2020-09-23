'''
Divide and Conquer
------------------
'''

import logging
from re import Pattern
from typing import Any, Callable, Collection, List, Optional, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.parameters as pr
import metacells.preprocessing as pp
import metacells.tools as tl
import metacells.utilities as ut

from .direct import compute_direct_metacells

__all__ = [
    'divide_and_conquer_pipeline',
    'compute_divide_and_conquer_metacells',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def divide_and_conquer_pipeline(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    rare_max_gene_cell_fraction: float = pr.rare_max_gene_cell_fraction,
    rare_min_gene_maximum: int = pr.rare_min_gene_maximum,
    rare_similarity_of: Optional[str] = None,
    rare_repeated_similarity: bool = pr.rare_repeated_similarity,
    rare_genes_cluster_method: str = pr.rare_genes_cluster_method,
    rare_min_size_of_modules: int = pr.rare_min_size_of_modules,
    rare_min_module_correlation: float = pr.rare_min_module_correlation,
    rare_min_cell_module_total: int = pr.rare_min_cell_module_total,
    feature_downsample_cell_quantile: float = pr.feature_downsample_cell_quantile,
    feature_min_gene_fraction: float = pr.feature_min_gene_fraction,
    feature_min_gene_relative_variance: float = pr.feature_min_gene_relative_variance,
    forbidden_gene_names: Optional[Collection[str]] = None,
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    cells_similarity_log_data: bool = pr.cells_similarity_log_data,
    cells_similarity_log_normalization: float = pr.cells_similarity_log_normalization,
    cells_repeated_similarity: bool = pr.cells_repeated_similarity,
    target_pile_size: int = pr.target_pile_size,
    pile_min_split_size_factor: float = pr.pile_min_split_size_factor,
    pile_min_robust_size_factor: float = pr.pile_min_robust_size_factor,
    pile_max_merge_size_factor: float = pr.pile_max_merge_size_factor,
    target_metacell_size: int = pr.target_metacell_size,
    knn_k: Optional[int] = pr.knn_k,
    knn_balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    knn_incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    knn_outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    candidates_partition_method: 'ut.PartitionMethod' = pr.candidates_partition_method,
    candidates_min_split_size_factor: Optional[float] = pr.candidates_min_split_size_factor,
    candidates_max_merge_size_factor: Optional[float] = pr.candidates_max_merge_size_factor,
    must_complete_cover: bool = False,
    max_outliers_levels: Optional[int] = pr.max_outliers_levels,
    deviants_min_gene_fold_factor: float = pr.deviants_min_gene_fold_factor,
    deviants_max_gene_fraction: Optional[float] = pr.deviants_max_gene_fraction,
    deviants_max_cell_fraction: Optional[float] = pr.deviants_max_cell_fraction,
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    random_seed: int = pr.random_seed,
    intermediate: bool = True,
) -> None:
    '''
    Complete pipeline using divide-and-conquer to compute the metacells for the whole data.

    .. note::

        This is applicable to "any size" data. If the data is "small" (O(10,000)), it will revert to
        using the direct metacell computation (but will still by default first look for rare gene
        modules). If the data is "large" (up to O(10,000,000)), this will be much faster and will
        require much less memory than using the direct approach. The current implementation is not
        optimized for "huge" data (O(1,000,000,000)) - it will work, and keep will use a limited
        amount of memory, but a faster implementation would distribute the computation across
        multiple servers.

    **Input**

    The presumably "clean" :py:func:`metacells.utilities.annotation.setup` annotated ``adata``.

    All the computations will use the ``of`` data (by default, the focus).

    **Returns**

    Sets the following annotations in ``adata``:

    Unstructured Annotations
        ``rare_gene_modules``
            An array of rare gene modules, where every entry is the array of the names of the genes
            of the module.

        ``pre_directs``, ``final_directs``
            The number of times we invoked
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` for computing the
            preliminary and final metacells. If we end up directly computing the metacells, the
            preliminary value will be zero, and the final value will be one. This can be used to
            normalize the ``pre_<name>`` and ``<name>`` properties below to fractions
            (probabilities).

    Variable (Gene) Annotations
        ``genes_rare_gene_module``
            The index of the rare gene module each gene belongs to, or ``-1`` in the common case it
            does not belong to any rare genes module.

        ``rare_genes``
            A boolean mask for the genes in any of the rare gene modules.

        ``pre_high_fraction_gene``, ``high_fraction_gene`` (if ``intermediate``)
            The number of times the gene was marked as having a high expression level when computing
            the preliminary and final metacells. This is zero for non-"clean" genes.

        ``pre_high_relative_variance_gene``, ``high_relative_variance_gene`` (if ``intermediate``)
            The number of times the gene was marked as having a high normalized variance relative to
            other genes with a similar expression level when when computing the preliminary and
            final metacells. This is zero for non-"clean" genes.

        ``forbidden`` (if ``intermediate``)
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name. This is ``False`` for non-"clean" genes.

        ``pre_feature`` (if ``intermediate``), ``feature``
            The number of times the gene was used as a feature when computing the preliminary and
            final metacells. If we end up directly computing the metacells, the preliminary value
            will be all-zero, and the final value will be one for feature genes, zero otherwise.

        ``pre_gene_deviant_votes``, ``gene_deviant_votes`` (if ``intermediate``)
            The total number of cells each gene marked as deviant (if zero, the gene did not mark
            any cell as deviant) when computing the preliminary and final metacells. This will be
            zero for non-"feature" genes.

    Observations (Cell) Annotations
        ``cells_rare_gene_module``
            The index of the rare gene module each cell expresses the most, or ``-1`` in the common
            case it does not express any rare genes module.

        ``rare_cells``
            A boolean mask for the (few) cells that express a rare gene module.

        ``pre_cell_directs``, ``final_cell_directs`` (if ``intermediate``)
            The number of times we invoked
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` to try and group this cell
            when computing the preliminary and final metacells. If we end up directly computing the
            metacells, the preliminary value will be zero. Otherwise this will be at least one, with
            higher values for cells which we tried to group again in the outlier pile(s).

        ``pre_pile``, ``pile`` (if ``intermediate``)
            The index of the pile the cell was in in the last invocation of
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` when computing the
            preliminary and final metacells. This is ``-1`` for non-"clean" cells. The preliminary
            value is likewise ``-1`` if we end up directly computing the metacells.

        ``pre_candidate``, ``candidate`` (if ``intermediate``)
            The index of the candidate metacell each cell was assigned to to in the last grouping
            attempt when computing the preliminary and final metacells. This is ``-1`` for
            non-"clean" cells.

        ``pre_cell_deviant_votes``, ``cell_deviant_votes`` (if ``intermediate``)
            The number of genes that were the reason the cell was marked as deviant in the last
            grouping attempt when computing the preliminary and final metacells (if zero, the cell
            is not deviant). This is zero for non-"clean" cells.

        ``pre_dissolved``, ``dissolved`` (if ``intermediate``)
            A boolean mask of the cells contained in a dissolved metacell in the last grouping
            attempt when computing the preliminary and final metacells. This is ``False`` for
            non-"clean" cells.

        ``pre_metacell`` (if ``intermediate``), ``metacell``
            The integer index of the preliminary and final metacell each cell belongs to. The
            metacells are in no particular order. This is ``-1`` for outlier cells and ``-2`` for
            non-"clean" cells.

        ``outlier`` (if ``intermediate``)
            A boolean mask of the cells not assigned to any final metacell. We assign every
            ("clean") cell to a preliminary metacell so there is no point in providing a
            ``pre_outlier`` mask (it would be identical to the ``clean_cell`` mask).

    If ``intermediate`` (default: {intermediate}), also keep all all the intermediate data (e.g.
    sums) for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.rare.find_rare_gene_modules` to isolate cells expressing
       rare gene modules, using the
       ``rare_max_gene_cell_fraction`` (default: {rare_max_gene_cell_fraction}),
       ``rare_min_gene_maximum`` (default: {rare_min_gene_maximum}),
       ``rare_similarity_of`` (default: {rare_similarity_of}),
       ``rare_repeated_similarity`` (default: {rare_repeated_similarity}),
       ``rare_genes_cluster_method`` (default: {rare_genes_cluster_method}),
       ``rare_min_size_of_modules`` (default: {rare_min_size_of_modules}),
       ``rare_min_module_correlation`` (default: {rare_min_module_correlation}),
       and
       ``rare_min_cell_module_total`` (default: {rare_min_cell_module_total}).

    2. For each detected rare gene module, collect all cells that express the module, and invoke
       :py:func:`compute_divide_and_conquer_metacells` to compute metacells for them,
       using the rest of the parameters.

    4. Collect all cells that either did not express any rare gene module, or did but were
       considered an outlier in the previous step, and invoke
       :py:func:`compute_divide_and_conquer_metacells` to compute metacells for them,
       using the rest of the parameters.

    4. Combine the results from the previous two steps.
    '''
    ut.log_pipeline_step(LOG, adata, 'compute_rare_candidates')

    final_results: List[AnnData] = []
    normal_cells_mask = np.full(adata.n_obs, True, dtype='bool')

    with ut.timed_step('.rare'):
        tl.find_rare_gene_modules(adata, of=of,
                                  max_gene_cell_fraction=rare_max_gene_cell_fraction,
                                  min_gene_maximum=rare_min_gene_maximum,
                                  similarity_of=rare_similarity_of,
                                  repeated_similarity=rare_repeated_similarity,
                                  genes_cluster_method=rare_genes_cluster_method,
                                  min_size_of_modules=rare_min_size_of_modules,
                                  min_module_correlation=rare_min_module_correlation,
                                  min_cell_module_total=rare_min_cell_module_total,
                                  intermediate=intermediate)

        rare_module_of_cells = \
            ut.to_dense_vector(ut.get_o_data(adata,
                                             'cells_rare_gene_module'))
        rare_modules_count = np.max(rare_module_of_cells) + 1

        for rare_module_index in range(rare_modules_count):
            rare_cells_mask = rare_module_of_cells == rare_module_index
            rare_cell_indices = np.where(rare_cells_mask)[0]
            assert len(rare_cell_indices) > 0
            rdata = ut.slice(adata, obs=rare_cell_indices,
                             track_obs='clean_cell_index',
                             tmp=True,
                             name='.rare-%s' % rare_module_index)
            compute_divide_and_conquer_metacells(rdata, of=of,
                                                 feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                                 feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                                 feature_min_gene_fraction=feature_min_gene_fraction,
                                                 forbidden_gene_names=forbidden_gene_names,
                                                 forbidden_gene_patterns=forbidden_gene_patterns,
                                                 cells_similarity_log_data=cells_similarity_log_data,
                                                 cells_similarity_log_normalization=cells_similarity_log_normalization,
                                                 cells_repeated_similarity=cells_repeated_similarity,
                                                 target_pile_size=target_pile_size,
                                                 pile_min_split_size_factor=pile_min_split_size_factor,
                                                 pile_min_robust_size_factor=pile_min_robust_size_factor,
                                                 pile_max_merge_size_factor=pile_max_merge_size_factor,
                                                 target_metacell_size=target_metacell_size,
                                                 knn_k=knn_k,
                                                 knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                                                 knn_incoming_degree_factor=knn_incoming_degree_factor,
                                                 knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                                                 candidates_partition_method=candidates_partition_method,
                                                 candidates_min_split_size_factor=candidates_min_split_size_factor,
                                                 candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                                                 must_complete_cover=must_complete_cover,
                                                 max_outliers_levels=max_outliers_levels,
                                                 deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                                 deviants_max_gene_fraction=deviants_max_gene_fraction,
                                                 deviants_max_cell_fraction=deviants_max_cell_fraction,
                                                 dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                                 dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                                 dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                                 cell_sizes=cell_sizes,
                                                 random_seed=random_seed,
                                                 intermediate=intermediate)
            outlier_of_rare_cells = ut.get_o_data(rdata, 'outlier')
            normal_cells_mask[rare_cell_indices] = outlier_of_rare_cells
            if not np.all(outlier_of_rare_cells):
                final_results.append(rdata)

    if np.all(normal_cells_mask):
        final_results = []
        cdata = adata
    else:
        cdata = ut.slice(adata, obs=normal_cells_mask, name='.common', tmp=True,
                         track_obs='clean_cell_index')

    compute_divide_and_conquer_metacells(cdata, of=of,
                                         feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                         feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                         feature_min_gene_fraction=feature_min_gene_fraction,
                                         forbidden_gene_names=forbidden_gene_names,
                                         forbidden_gene_patterns=forbidden_gene_patterns,
                                         cells_similarity_log_data=cells_similarity_log_data,
                                         cells_similarity_log_normalization=cells_similarity_log_normalization,
                                         cells_repeated_similarity=cells_repeated_similarity,
                                         target_pile_size=target_pile_size,
                                         pile_min_split_size_factor=pile_min_split_size_factor,
                                         pile_min_robust_size_factor=pile_min_robust_size_factor,
                                         pile_max_merge_size_factor=pile_max_merge_size_factor,
                                         target_metacell_size=target_metacell_size,
                                         knn_k=knn_k,
                                         knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                                         knn_incoming_degree_factor=knn_incoming_degree_factor,
                                         knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                                         candidates_partition_method=candidates_partition_method,
                                         candidates_min_split_size_factor=candidates_min_split_size_factor,
                                         candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                                         must_complete_cover=must_complete_cover,
                                         max_outliers_levels=max_outliers_levels,
                                         deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                         deviants_max_gene_fraction=deviants_max_gene_fraction,
                                         deviants_max_cell_fraction=deviants_max_cell_fraction,
                                         dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                         dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                         dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                         cell_sizes=cell_sizes,
                                         random_seed=random_seed,
                                         intermediate=intermediate)

    if len(final_results) > 0:
        final_results.append(cdata)
        _initialize_results(adata,
                            intermediate=intermediate,
                            pre_metacell=intermediate)
        _combine_results(adata, final_results, intermediate=intermediate)


def compute_divide_and_conquer_metacells(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    feature_downsample_cell_quantile: float = pr.feature_downsample_cell_quantile,
    feature_min_gene_fraction: float = pr.feature_min_gene_fraction,
    feature_min_gene_relative_variance: float = pr.feature_min_gene_relative_variance,
    forbidden_gene_names: Optional[Collection[str]] = None,
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    cells_similarity_log_data: bool = pr.cells_similarity_log_data,
    cells_similarity_log_normalization: float = pr.cells_similarity_log_normalization,
    cells_repeated_similarity: bool = pr.cells_repeated_similarity,
    target_pile_size: int = pr.target_pile_size,
    pile_min_split_size_factor: float = pr.pile_min_split_size_factor,
    pile_min_robust_size_factor: float = pr.pile_min_robust_size_factor,
    pile_max_merge_size_factor: float = pr.pile_max_merge_size_factor,
    target_metacell_size: int = pr.target_metacell_size,
    knn_k: Optional[int] = pr.knn_k,
    knn_balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    knn_incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    knn_outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    candidates_partition_method: 'ut.PartitionMethod' = pr.candidates_partition_method,
    candidates_min_split_size_factor: Optional[float] = pr.candidates_min_split_size_factor,
    candidates_max_merge_size_factor: Optional[float] = pr.candidates_max_merge_size_factor,
    must_complete_cover: bool = False,
    max_outliers_levels: Optional[int] = pr.max_outliers_levels,
    deviants_min_gene_fold_factor: float = pr.deviants_min_gene_fold_factor,
    deviants_max_gene_fraction: Optional[float] = pr.deviants_max_gene_fraction,
    deviants_max_cell_fraction: Optional[float] = pr.deviants_max_cell_fraction,
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    random_seed: int = pr.random_seed,
    intermediate: bool = True,
) -> None:
    '''
    Compute metacells using the divide-and-conquer method.

    This divides large data to smaller "piles" and directly computes metacells for each, then
    generates new piles of "similar" metacells and directly computes the final metacells from each
    such pile. Due to this divide-and-conquer approach, the total amount of memory required is
    bounded (by the pile size), and the amount of total CPU grows slowly with the problem size,
    allowing this method to be applied to very large data (millions of cells).

    .. todo::

        The divide-and-conquer implementation is restricted for using multiple CPUs on a single
        shared-memory machine. At some problem size (probably around a billion cells), it would make
        sense to modify it to support distributed execution of sub-problems on different servers.
        This would only be an implementation change, rather than an algorithmic change.

    **Input**

    The presumably "clean" :py:func:`metacells.utilities.annotation.setup` annotated ``adata``.

    All the computations will use the ``of`` data (by default, the focus).

    **Returns**

    Sets the following annotations in ``adata``:

    Structured Annotations
        ``pre_directs``, ``final_directs``
            The number of times we invoked
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` for computing the
            preliminary and final metacells. If we end up directly computing the metacells, the
            preliminary value will be zero, and the final value will be one. This can be used to
            normalize the ``pre_<name>`` and ``<name>`` properties below to fractions
            (probabilities).

    Variable (Gene) Annotations
        ``pre_high_fraction_gene``, ``high_fraction_gene`` (if ``intermediate``)
            The number of times the gene was marked as having a high expression level when computing
            the preliminary and final metacells. This is zero for non-"clean" genes.

        ``pre_high_relative_variance_gene``, ``high_relative_variance_gene`` (if ``intermediate``)
            The number of times the gene was marked as having a high normalized variance relative to
            other genes with a similar expression level when when computing the preliminary and
            final metacells. This is zero for non-"clean" genes.

        ``forbidden`` (if ``intermediate``)
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name. This is ``False`` for non-"clean" genes.

        ``pre_feature`` (if ``intermediate``), ``feature``
            The number of times the gene was used as a feature when computing the preliminary and
            final metacells. If we end up directly computing the metacells, the preliminary value
            will be all-zero, and the final value will be one for feature genes, zero otherwise.

        ``pre_gene_deviant_votes``, ``gene_deviant_votes`` (if ``intermediate``)
            The total number of cells each gene marked as deviant (if zero, the gene did not mark
            any cell as deviant) when computing the preliminary and final metacells. This will be
            zero for non-"feature" genes.

    Observations (Cell) Annotations
        ``pre_cell_directs``, ``final_cell_directs`` (if ``intermediate``)
            The number of times we invoked
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` to try and group this cell
            when computing the preliminary and final metacells. If we end up directly computing the
            metacells, the preliminary value will be zero. Otherwise this will be at least one, with
            higher values for cells which we tried to group again in the outlier pile(s).

        ``pre_pile``, ``pile`` (if ``intermediate``)
            The index of the pile the cell was in in the last invocation of
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` when computing the
            preliminary and final metacells. This is ``-1`` for non-"clean" cells. The preliminary
            value is likewise ``-1`` if we end up directly computing the metacells.

        ``pre_candidate``, ``candidate`` (if ``intermediate``)
            The index of the candidate metacell each cell was assigned to to in the last grouping
            attempt when computing the preliminary and final metacells. This is ``-1`` for
            non-"clean" cells.

        ``pre_cell_deviant_votes``, ``cell_deviant_votes`` (if ``intermediate``)
            The number of genes that were the reason the cell was marked as deviant in the last
            grouping attempt when computing the preliminary and final metacells (if zero, the cell
            is not deviant). This is zero for non-"clean" cells.

        ``pre_dissolved``, ``dissolved`` (if ``intermediate``)
            A boolean mask of the cells contained in a dissolved metacell in the last grouping
            attempt when computing the preliminary and final metacells. This is ``False`` for
            non-"clean" cells.

        ``pre_metacell`` (if ``intermediate``), ``metacell``
            The integer index of the preliminary and final metacell each cell belongs to. The
            metacells are in no particular order. This is ``-1`` for outlier cells and ``-2`` for
            non-"clean" cells.

        ``outlier`` (if ``intermediate``)
            A boolean mask of the cells not assigned to any final metacell. We assign every
            ("clean") cell to a preliminary metacell so there is no point in providing a
            ``pre_outlier`` mask (it would be identical to the ``clean_cell`` mask).

    If ``intermediate`` (default: {intermediate}), also keep all all the intermediate data (e.g.
    sums) for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. If the data is smaller than ``target_pile_size`` (default: {target_pile_size}) times the
       ``pile_min_split_size_factor`` (default: {pile_min_split_size_factor}), then just invoke
       :py:func:`metacells.pipeline.direct.compute_direct_metacells` using the parameters, patch the
       results to contain the expected annotations from a divide-and-conquer call, and return.
       Otherwise, perform the following steps.

    2. Group the cells randomly into equal-sized piles of roughly the ``target_pile_size`` using
       the ``random_seed`` (default: {random_seed}) to allow making this replicable.

    2. Compute preliminary metacells using the random piles and the parameters. Here we set
       ``must_complete_cover`` to ``True`` to ensure each cell is placed in some preliminary
       metacell so all cells will be in some pile in the next phase.

    3. Invoke :py:func:`metacells.preprocessing.group.group_obs_data` to sum the cells into
       preliminary metacells.

    4. Invoke :py:func:`compute_divide_and_conquer_metacells` using the parameters to group the
       preliminary metacells into piles. We use here ``pile_min_split_size_factor`` (default:
       {pile_min_split_size_factor}), ``pile_min_robust_size_factor`` (default:
       {pile_min_robust_size_factor}) and ``pile_max_merge_size_factor`` (default:
       {pile_max_merge_size_factor}), so control over the pile size (number of cells) is separate
       from control over the metacell size (number of UMIs).

    5. Compute the final metacells using the preliminary metacell piles. Since these piles contain
       "similar" cells, we aim for higher-quality metacells by using a more aggressive
       ``final_deviants_max_cell_fraction`` (default: {final_deviants_max_cell_fraction}), and using
       ``max_outliers_levels`` to prevent forcing outlier cells to be placed in low-quality
       metacells.
    '''
    ut.log_pipeline_step(LOG, adata, 'compute_divide_and_conquer_metacells')

    LOG.debug('  target_pile_size: %s', target_pile_size)
    LOG.debug('  pile_min_split_size_factor: %s',
              pile_min_split_size_factor)

    pile_min_split_size = target_pile_size * pile_min_split_size_factor
    LOG.debug('  pile_min_split_size: %s', pile_min_split_size)
    if adata.n_obs < pile_min_split_size:
        with ut.timed_step('.direct'):
            compute_direct_metacells(adata, of=of,
                                     feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                     feature_min_gene_fraction=feature_min_gene_fraction,
                                     feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                     forbidden_gene_names=forbidden_gene_names,
                                     forbidden_gene_patterns=forbidden_gene_patterns,
                                     cells_similarity_log_data=cells_similarity_log_data,
                                     cells_similarity_log_normalization=cells_similarity_log_normalization,
                                     cells_repeated_similarity=cells_repeated_similarity,
                                     target_metacell_size=target_metacell_size,
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
                                     cell_sizes=cell_sizes,
                                     random_seed=random_seed,
                                     intermediate=intermediate)
            _direct_results(adata, intermediate)
        return

    _initialize_results(adata,
                        intermediate=intermediate,
                        pre_metacell=True)

    LOG.debug('  random_seed: %s', random_seed)
    random_pile_of_cells = \
        ut.random_piles(adata.n_obs,
                        target_pile_size=target_pile_size,
                        random_seed=random_seed)

    with ut.timed_step('.preliminary_metacells'):
        _compute_piled_metacells(adata, of=of,
                                 phase='preliminary',
                                 pile_of_cells=random_pile_of_cells,
                                 feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                 feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                 feature_min_gene_fraction=feature_min_gene_fraction,
                                 forbidden_gene_names=forbidden_gene_names,
                                 forbidden_gene_patterns=forbidden_gene_patterns,
                                 cells_similarity_log_data=cells_similarity_log_data,
                                 cells_similarity_log_normalization=cells_similarity_log_normalization,
                                 cells_repeated_similarity=cells_repeated_similarity,
                                 target_pile_size=target_pile_size,
                                 pile_min_split_size_factor=pile_min_split_size_factor,
                                 pile_min_robust_size_factor=pile_min_robust_size_factor,
                                 pile_max_merge_size_factor=pile_max_merge_size_factor,
                                 target_metacell_size=target_metacell_size,
                                 knn_k=knn_k,
                                 knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                                 knn_incoming_degree_factor=knn_incoming_degree_factor,
                                 knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                                 candidates_partition_method=candidates_partition_method,
                                 candidates_min_split_size_factor=candidates_min_split_size_factor,
                                 candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                                 must_complete_cover=True,
                                 max_outliers_levels=None,
                                 deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                 deviants_max_gene_fraction=deviants_max_gene_fraction,
                                 deviants_max_cell_fraction=deviants_max_cell_fraction,
                                 dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                 dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                 dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                 cell_sizes=cell_sizes,
                                 random_seed=random_seed,
                                 intermediate=intermediate)

    with ut.timed_step('.metacell_piles'):
        ut.log_operation(LOG, adata, 'metacell_piles', of)

        mdata = pp.group_obs_data(adata, of=of, groups='pre_metacell',
                                  name='.preliminary_metacells', tmp=True)
        if mdata is None:
            raise ValueError('Empty metacells data, giving up')

        compute_divide_and_conquer_metacells(mdata,
                                             feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                             feature_min_gene_fraction=feature_min_gene_fraction,
                                             feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                             forbidden_gene_names=forbidden_gene_names,
                                             forbidden_gene_patterns=forbidden_gene_patterns,
                                             cells_similarity_log_data=cells_similarity_log_data,
                                             cells_similarity_log_normalization=cells_similarity_log_normalization,
                                             cells_repeated_similarity=cells_repeated_similarity,
                                             target_pile_size=target_pile_size,
                                             pile_min_split_size_factor=pile_min_split_size_factor,
                                             pile_min_robust_size_factor=pile_min_robust_size_factor,
                                             pile_max_merge_size_factor=pile_max_merge_size_factor,
                                             target_metacell_size=target_pile_size,
                                             knn_k=knn_k,
                                             knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                                             knn_incoming_degree_factor=knn_incoming_degree_factor,
                                             knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                                             candidates_partition_method=candidates_partition_method,
                                             candidates_min_split_size_factor=pile_min_split_size_factor,
                                             candidates_max_merge_size_factor=pile_max_merge_size_factor,
                                             must_complete_cover=True,
                                             max_outliers_levels=None,
                                             deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                             deviants_max_gene_fraction=deviants_max_gene_fraction,
                                             deviants_max_cell_fraction=deviants_max_cell_fraction,
                                             dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                             dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                             dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                             cell_sizes='grouped',
                                             random_seed=random_seed,
                                             intermediate=intermediate)
        preliminary_metacell_of_cells = \
            ut.to_dense_vector(ut.get_o_data(adata, 'pre_metacell'))

        pile_of_preliminary_metacells = \
            ut.to_dense_vector(ut.get_o_data(mdata, 'metacell'))

        preliminary_pile_of_cells = \
            ut.group_piles(preliminary_metacell_of_cells,
                           pile_of_preliminary_metacells)

    with ut.timed_step('.final_metacells'):
        _compute_piled_metacells(adata, of=of,
                                 phase='final',
                                 pile_of_cells=preliminary_pile_of_cells,
                                 feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                 feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                 feature_min_gene_fraction=feature_min_gene_fraction,
                                 forbidden_gene_names=forbidden_gene_names,
                                 forbidden_gene_patterns=forbidden_gene_patterns,
                                 cells_similarity_log_data=cells_similarity_log_data,
                                 cells_similarity_log_normalization=cells_similarity_log_normalization,
                                 cells_repeated_similarity=cells_repeated_similarity,
                                 target_pile_size=target_pile_size,
                                 pile_min_split_size_factor=pile_min_split_size_factor,
                                 pile_min_robust_size_factor=pile_min_robust_size_factor,
                                 pile_max_merge_size_factor=pile_max_merge_size_factor,
                                 target_metacell_size=target_metacell_size,
                                 knn_k=knn_k,
                                 knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                                 knn_incoming_degree_factor=knn_incoming_degree_factor,
                                 knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                                 candidates_partition_method=candidates_partition_method,
                                 candidates_min_split_size_factor=candidates_min_split_size_factor,
                                 candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                                 must_complete_cover=must_complete_cover,
                                 max_outliers_levels=max_outliers_levels,
                                 deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                 deviants_max_gene_fraction=deviants_max_gene_fraction,
                                 deviants_max_cell_fraction=deviants_max_cell_fraction,
                                 dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                 dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                 dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                 cell_sizes=cell_sizes,
                                 random_seed=random_seed,
                                 intermediate=intermediate)


def _compute_piled_metacells(
    cdata: AnnData,
    *,
    of: Optional[str],
    phase: str,
    pile_of_cells: ut.DenseVector,
    feature_downsample_cell_quantile: float,
    feature_min_gene_fraction: float,
    feature_min_gene_relative_variance: float,
    forbidden_gene_names: Optional[Collection[str]],
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]],
    cells_similarity_log_data: bool,
    cells_similarity_log_normalization: float,
    cells_repeated_similarity: bool,
    target_pile_size: int,
    pile_min_split_size_factor: float,
    pile_min_robust_size_factor: float,
    pile_max_merge_size_factor: float,
    target_metacell_size: int,
    knn_k: Optional[int],
    knn_balanced_ranks_factor: float,
    knn_incoming_degree_factor: float,
    knn_outgoing_degree_factor: float,
    candidates_partition_method: 'ut.PartitionMethod',
    candidates_min_split_size_factor: Optional[float],
    candidates_max_merge_size_factor: Optional[float],
    must_complete_cover: bool,
    max_outliers_levels: Optional[int],
    deviants_min_gene_fold_factor: float,
    deviants_max_gene_fraction: Optional[float],
    deviants_max_cell_fraction: Optional[float],
    dissolve_min_robust_size_factor: Optional[float],
    dissolve_min_convincing_size_factor: Optional[float],
    dissolve_min_convincing_gene_fold_factor: float,
    cell_sizes: Optional[Union[str, ut.Vector]],
    random_seed: int,
    intermediate: bool,
) -> None:
    ut.log_operation(LOG, cdata, phase + '_metacells', of)
    piles_count = np.max(pile_of_cells) + 1
    LOG.debug('  piles_count: %s', piles_count)
    assert piles_count > 1

    @ut.timed_call('compute_pile_metacells')
    def _compute_pile_metacells(pile_index: int) -> pd.Series:
        pile_cells_mask = pile_of_cells == pile_index
        assert np.any(pile_cells_mask)
        name = '.%s.pile-%s/%s' % (phase, pile_index, piles_count)
        pdata = ut.slice(cdata, obs=pile_cells_mask, name=name, tmp=True,
                         track_obs='complete_cell_index')
        compute_direct_metacells(pdata, of=of,
                                 feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                 feature_min_gene_fraction=feature_min_gene_fraction,
                                 feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                 forbidden_gene_names=forbidden_gene_names,
                                 forbidden_gene_patterns=forbidden_gene_patterns,
                                 cells_similarity_log_data=cells_similarity_log_data,
                                 cells_similarity_log_normalization=cells_similarity_log_normalization,
                                 cells_repeated_similarity=cells_repeated_similarity,
                                 target_metacell_size=target_metacell_size,
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
                                 cell_sizes=cell_sizes,
                                 random_seed=random_seed)
        return _pile_results(pdata, phase, intermediate)

    with ut.timed_step('.piles'):
        results_of_piles = \
            list(ut.parallel_map(_compute_pile_metacells, piles_count))

    piles_count = len(results_of_piles)
    metacells_count = \
        _collect_piles_results(cdata, results_of_piles, phase, intermediate)

    assert phase in ('preliminary', 'final')
    if phase == 'preliminary':
        metacell_of_cells = \
            ut.to_dense_vector(ut.get_o_data(cdata, 'pre_metacell'))
    else:
        metacell_of_cells = \
            ut.to_dense_vector(ut.get_o_data(cdata, 'metacell'))

    outlier_of_cells = metacell_of_cells < 0
    if not np.any(outlier_of_cells) \
        or (not must_complete_cover
            and max_outliers_levels is not None
            and max_outliers_levels <= 0):
        return

    with ut.timed_step('.outliers'):
        if max_outliers_levels is not None:
            max_outliers_levels = max_outliers_levels - 1
        name = '.%s.outliers' % phase
        odata = ut.slice(cdata, obs=outlier_of_cells, name=name, tmp=True,
                         track_obs='complete_cell_index')

        compute_divide_and_conquer_metacells(odata, of=of,
                                             feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                             feature_min_gene_fraction=feature_min_gene_fraction,
                                             feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                             forbidden_gene_names=forbidden_gene_names,
                                             forbidden_gene_patterns=forbidden_gene_patterns,
                                             cells_similarity_log_data=cells_similarity_log_data,
                                             cells_similarity_log_normalization=cells_similarity_log_normalization,
                                             cells_repeated_similarity=cells_repeated_similarity,
                                             target_pile_size=target_pile_size,
                                             pile_min_split_size_factor=pile_min_split_size_factor,
                                             pile_min_robust_size_factor=pile_min_robust_size_factor,
                                             pile_max_merge_size_factor=pile_max_merge_size_factor,
                                             target_metacell_size=target_metacell_size,
                                             knn_k=knn_k,
                                             knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                                             knn_incoming_degree_factor=knn_incoming_degree_factor,
                                             knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                                             candidates_partition_method=candidates_partition_method,
                                             candidates_min_split_size_factor=candidates_min_split_size_factor,
                                             candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                                             must_complete_cover=must_complete_cover,
                                             max_outliers_levels=max_outliers_levels,
                                             deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                             deviants_max_gene_fraction=deviants_max_gene_fraction,
                                             deviants_max_cell_fraction=deviants_max_cell_fraction,
                                             dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                             dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                             dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                             cell_sizes=cell_sizes,
                                             random_seed=random_seed)

        _collect_outliers_results(cdata, odata, phase,
                                  metacells_count, piles_count, intermediate)


# pylint: disable=cell-var-from-loop


@ut.timed_call('.direct_results')
def _direct_results(
    adata: AnnData,
    intermediate: bool
) -> None:
    ut.log_pipeline_step(LOG, adata, 'direct_results')
    ut.set_m_data(adata, 'pre_directs', 0)
    ut.set_m_data(adata, 'final_directs', 1)

    for name, always in (('pre_high_fraction_gene', False),
                         ('high_fraction_gene', False),
                         ('pre_high_relative_variance_gene', False),
                         ('high_relative_variance_gene', False),
                         ('pre_feature', False),
                         ('feature', True),
                         ('pre_gene_deviant_votes', False)):
        if not always and not intermediate:
            continue
        if name.startswith('pre_'):
            ut.set_v_data(adata, name,
                          np.zeros(adata.n_vars, dtype='int32'),
                          log_value=lambda _: '* <- 0')
        else:
            _modify_value(adata, ut.get_v_data, ut.set_v_data, name,
                          lambda mask:
                          ut.to_dense_vector(mask).astype('int32'),
                          log_value=ut.mask_description)

    if not intermediate:
        return

    for name, value, dtype in (('pre_cell_directs', 0, 'int32'),
                               ('final_cell_directs', 1, 'int32'),
                               ('pre_pile', -1, 'int32'),
                               ('pile', 0, 'int32'),
                               ('pre_candidate', -1, 'int32'),
                               ('pre_cell_deviant_votes', 0, 'int32'),
                               ('pre_dissolved', False, 'bool'),
                               ('pre_metacell', -1, 'int32')):
        ut.set_o_data(adata, name,
                      np.full(adata.n_obs, value, dtype=dtype),
                      log_value=lambda _: '* <- %s' % value)


def _pile_results(
    pdata: AnnData,
    phase: str,
    intermediate: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert phase in ('preliminary', 'final')
    is_final = phase == 'final'

    cells_frame = \
        pd.DataFrame(index=ut.to_dense_vector(ut.get_o_data(pdata,
                                                            'complete_cell_index')))
    cells_frame['metacell'] = \
        ut.to_dense_vector(ut.get_o_data(pdata, 'metacell'))

    genes_frame = pd.DataFrame(index=range(pdata.n_vars))
    genes_frame['feature'] = \
        ut.to_dense_vector(ut.get_v_data(pdata, 'feature'))

    if not intermediate:
        return cells_frame, genes_frame

    for name in ('high_fraction_gene',
                 'high_relative_variance_gene',
                 'forbidden',
                 'gene_deviant_votes'):
        genes_frame[name] = ut.to_dense_vector(ut.get_v_data(pdata, name))

    for name, always in (('candidate', True),
                         ('cell_deviant_votes', True),
                         ('dissolved', True),
                         ('outlier', False)):
        if always or is_final:
            cells_frame[name] = ut.to_dense_vector(ut.get_o_data(pdata, name))

    return cells_frame, genes_frame


def _collect_piles_results(
    cdata: AnnData,
    results_of_piles: List[Tuple[pd.DataFrame, pd.DataFrame]],
    phase: str,
    intermediate: bool,
) -> int:
    assert phase in ('preliminary', 'final')
    is_final = phase == 'final'

    metacells_count = 0

    for pile_index, (cells_frame, genes_frame) in enumerate(results_of_piles):
        LOG.debug('collect_pile_results %s/%s',
                  pile_index, len(results_of_piles))
        _collect_pile_results(cdata, is_final, cells_frame, genes_frame,
                              metacells_count, pile_index, intermediate)

    return metacells_count


def _collect_pile_results(
    cdata: AnnData,
    is_final: bool,
    cells_frame: pd.DataFrame,
    genes_frame: pd.DataFrame,
    metacells_count: int,
    pile_index: int,
    intermediate: bool,
) -> int:
    def _prefix(name: str, final_prefix: str = '') -> str:
        if is_final:
            return final_prefix + name
        return 'pre_' + name

    _modify_value(cdata, ut.get_m_data, ut.set_m_data, _prefix('directs', 'final_'),
                  lambda directs: directs + 1)

    for name, always in (('high_fraction_gene', False),
                         ('high_relative_variance_gene', False),
                         ('feature', is_final),
                         ('gene_deviant_votes', False)):
        if always or intermediate:
            _modify_value(cdata, ut.get_v_data, ut.set_v_data,
                          _prefix(name),
                          lambda value_of_genes: value_of_genes
                          + ut.to_dense_vector(genes_frame[name]).astype('int32'),
                          log_value=ut.mask_description)

    cell_indices = cells_frame.index.values

    pile_metacells = ut.to_dense_vector(cells_frame['metacell'])
    pile_metacells[pile_metacells >= 0] += metacells_count
    metacells_count = np.max(pile_metacells) + 1

    _modify_value_subset(cdata, ut.get_o_data, ut.set_o_data,
                         _prefix('metacell'), cell_indices,
                         lambda _: pile_metacells,
                         log_value=ut.groups_description)

    if not intermediate:
        return metacells_count

    _modify_value_subset(cdata, ut.get_o_data, ut.set_o_data,
                         _prefix('cell_directs', 'final_'), cell_indices,
                         lambda directs_of_cells: directs_of_cells + 1)

    _modify_value_subset(cdata, ut.get_o_data, ut.set_o_data,
                         _prefix('pile'), cell_indices,
                         lambda _: np.full(cell_indices.size, pile_index),
                         log_value=lambda _: '%s <- %s'
                         % (ut.ratio_description(len(cell_indices),
                                                 cdata.n_obs),
                            pile_index))

    for name, always, log_value \
            in (('candidate', True, ut.groups_description),
                ('cell_deviant_votes', True, ut.mask_description),
                ('dissolved', True, ut.mask_description),
                ('outlier', False, ut.mask_description)):
        if always or is_final:
            _modify_value_subset(cdata, ut.get_o_data, ut.set_o_data,
                                 _prefix(name), cell_indices,
                                 lambda _: cells_frame[name],
                                 log_value=log_value)

    return metacells_count


@ut.timed_call('.outliers_results')
def _collect_outliers_results(
    cdata: AnnData,
    odata: AnnData,
    phase: str,
    metacells_count: int,
    piles_count: int,
    intermediate: bool
) -> None:
    ut.log_pipeline_step(LOG, cdata, 'outlier_results')
    LOG.debug('  from: %s %s', ut.get_name(odata), odata.shape)

    assert phase in ('preliminary', 'final')
    is_final = phase == 'final'

    def _prefix(name: str, final_prefix: str = '') -> str:
        if is_final:
            return final_prefix + name
        return 'pre_' + name

    _modify_value(cdata, ut.get_m_data, ut.set_m_data, _prefix('directs', 'final_'),
                  lambda directs: directs
                  + ut.get_m_data(odata, 'pre_directs')
                  + ut.get_m_data(odata, 'final_directs'))

    for name, always in (('high_fraction_gene', False),
                         ('high_relative_variance_gene', False),
                         ('feature', is_final),
                         ('gene_deviant_votes', False)):
        if always or intermediate:
            _modify_value(cdata, ut.get_v_data, ut.set_v_data, _prefix(name),
                          lambda value_of_genes: value_of_genes
                          + ut.to_dense_vector(ut.get_v_data(odata,
                                                             'pre_' + name))
                          + ut.to_dense_vector(ut.get_v_data(odata,
                                                             name)).astype('int32'),
                          log_value=ut.mask_description)

    cell_indices = \
        ut.to_dense_vector(ut.get_o_data(odata, 'complete_cell_index'))

    outlier_metacells = \
        ut.to_dense_vector(ut.get_o_data(odata, 'metacell'), copy=True)
    outlier_metacells[outlier_metacells >= 0] += metacells_count

    _modify_value_subset(cdata, ut.get_o_data, ut.set_o_data,
                         _prefix('metacell'), cell_indices,
                         lambda _: outlier_metacells,
                         log_value=ut.groups_description)

    if not intermediate:
        return

    _modify_value_subset(cdata, ut.get_o_data, ut.set_o_data,
                         _prefix('cell_directs', 'final_'), cell_indices,
                         lambda directs_of_cells: directs_of_cells
                         + ut.to_dense_vector(ut.get_o_data(odata,
                                                            'pre_cell_directs'))
                         + ut.to_dense_vector(ut.get_o_data(odata,
                                                            'final_cell_directs')))

    outlier_piles = ut.to_dense_vector(ut.get_o_data(odata, 'pile'))

    _modify_value_subset(cdata, ut.get_o_data, ut.set_o_data,
                         _prefix('pile'), cell_indices,
                         lambda _: outlier_piles + piles_count,
                         log_value=ut.groups_description)

    for name, always, log_value \
            in (('candidate', True, ut.groups_description),
                ('cell_deviant_votes', True, ut.mask_description),
                ('dissolved', True, ut.mask_description),
                ('outlier', False, ut.mask_description)):
        if always or is_final:
            _modify_value_subset(cdata, ut.get_o_data, ut.set_o_data,
                                 _prefix(name), cell_indices,
                                 lambda _:
                                 ut.to_dense_vector(ut.get_o_data(odata,
                                                                  name)),
                                 log_value=log_value)


@ut.timed_call('.combine_results')
def _combine_results(
    adata: AnnData,
    results: List[AnnData],
    *,
    intermediate: bool
) -> None:
    ut.log_pipeline_step(LOG, adata, 'combine_results')

    counts = dict(pre_pile=0, pile=0,
                  pre_candidate=0, candidate=0,
                  pre_metacell=0, metacell=0)

    for rdata in results:
        if LOG.isEnabledFor(logging.DEBUG):
            result_name = ut.get_name(rdata)
            if result_name is None:
                LOG.debug('- collect metacells')
            else:
                LOG.debug('- collect metacells from: %s', result_name)

        for name in ('pre_directs', 'final_directs'):
            _modify_value(adata, ut.get_m_data, ut.set_m_data, name,
                          lambda directs: directs + ut.get_m_data(rdata, name))

        for name, always, is_bool \
                in (('pre_high_fraction_gene', False, False),
                    ('high_fraction_gene', False, False),
                    ('pre_high_relative_variance_gene', False, False),
                    ('high_relative_variance_gene', False, False),
                    ('forbidden', False, True),
                    ('pre_feature', False, False),
                    ('feature', True, False),
                    ('pre_gene_deviant_votes', False, False),
                    ('gene_deviant_votes', False, False)):
            if always or intermediate:
                if is_bool:
                    _modify_value(adata, ut.get_v_data, ut.set_v_data, name,
                                  lambda value_of_genes: value_of_genes
                                  | ut.to_dense_vector(ut.get_v_data(rdata,
                                                                     name)),
                                  log_value=ut.mask_description)
                else:
                    _modify_value(adata, ut.get_v_data, ut.set_v_data, name,
                                  lambda value_of_genes: value_of_genes
                                  + ut.to_dense_vector(ut.get_v_data(rdata,
                                                                     name)).astype('int32'),
                                  log_value=ut.mask_description)

        cell_indices = \
            ut.to_dense_vector(ut.get_o_data(rdata, 'clean_cell_index'))

        for name, always, log_value \
                in (('pre_cell_directs', False, None),
                    ('final_cell_directs', False, None),
                    ('pre_pile', False, ut.groups_description),
                    ('pile', False, ut.groups_description),
                    ('pre_candidate', False, ut.groups_description),
                    ('candidate', False, ut.groups_description),
                    ('pre_cell_deviant_votes', False, ut.mask_description),
                    ('cell_deviant_votes', False, ut.mask_description),
                    ('pre_dissolved', False, ut.mask_description),
                    ('dissolved', False, ut.mask_description),
                    ('pre_metacell', False, ut.groups_description),
                    ('metacell', True, ut.groups_description),
                    ('outlier', False, ut.mask_description)):
            if always or intermediate:
                def _set_cell_values(_: ut.DenseVector) -> ut.DenseVector:
                    value_of_result_cells = \
                        ut.to_dense_vector(ut.get_v_data(rdata, name))

                    count = counts.get(name)
                    if count is not None:
                        ut.unfreeze(value_of_result_cells)
                        value_of_result_cells[value_of_result_cells >= 0] += count
                        counts[name] = np.max(value_of_result_cells) + 1

                    return value_of_result_cells

                _modify_value_subset(adata, ut.get_v_data, ut.set_v_data,
                                     name, cell_indices,
                                     _set_cell_values, log_value=log_value)


@ut.timed_call('_initialize_results')
def _initialize_results(
    adata: AnnData,
    *,
    intermediate: bool,
    pre_metacell: bool,
) -> None:
    ut.log_pipeline_step(LOG, adata, 'initialize_results')

    ut.set_m_data(adata, 'pre_directs', 0)
    ut.set_m_data(adata, 'final_directs', 0)

    for name, always, value, dtype \
            in (('pre_high_fraction_gene', False, 0, 'int32'),
                ('high_fraction_gene', False, 0, 'int32'),
                ('pre_high_relative_variance_gene', False, 0, 'int32'),
                ('high_relative_variance_gene', False, 0, 'int32'),
                ('forbidden', False, False, 'bool'),
                ('pre_feature', False, 0, 'int32'),
                ('feature', True, 0, 'int32'),
                ('pre_gene_deviant_votes', False, 0, 'int32'),
                ('gene_deviant_votes', False, 0, 'int32')):
        if always or intermediate:
            ut.set_v_data(adata, name,
                          np.full(adata.n_vars, value, dtype=dtype),
                          log_value=lambda _: '* <- %s' % value)

    for name, always, value, dtype \
            in (('pre_cell_directs', False, 0, 'int32'),
                ('final_cell_directs', False, 0, 'int32'),
                ('pre_pile', False, -1, 'int32'),
                ('pile', False, -1, 'int32'),
                ('pre_candidate', False, -1, 'int32'),
                ('candidate', False, -1, 'int32'),
                ('pre_cell_deviant_votes', False, 0, 'int32'),
                ('cell_deviant_votes', False, 0, 'int32'),
                ('pre_dissolved', False, False, 'bool'),
                ('dissolved', False, False, 'bool'),
                ('pre_metacell', pre_metacell, -1, 'int32'),
                ('metacell', True, -1, 'int32'),
                ('outlier', False, False, 'bool')):
        if always or intermediate:
            ut.set_o_data(adata, name,
                          np.full(adata.n_obs, value, dtype=dtype),
                          log_value=lambda _: '* <- %s' % value)


LogValue = Optional[Callable[[Any], Optional[str]]]
try:
    from mypy_extensions import NamedArg
    Setter = \
        Callable[[AnnData, str, Any, NamedArg(LogValue, 'log_value')], None]
except ModuleNotFoundError:
    pass


def _modify_value(
    adata: AnnData,
    getter: Callable[[AnnData, str], Any],
    setter: 'Setter',
    name: str,
    modifier: Callable[[Any], Any],
    log_value: LogValue = None,
) -> None:
    old_value = getter(adata, name)
    new_value = modifier(old_value)
    assert new_value is not None
    setter(adata, name, new_value, log_value=log_value)


def _modify_value_subset(
    adata: AnnData,
    getter: Callable[[AnnData, str], ut.Vector],
    setter: 'Setter',
    name: str,
    indices: ut.DenseVector,
    modifier: Callable[[ut.DenseVector], ut.DenseVector],
    log_value: LogValue = None,
) -> None:
    old_value = ut.to_dense_vector(getter(adata, name))
    ut.unfreeze(old_value)
    old_value[indices] = modifier(old_value[indices])
    ut.freeze(old_value)
    setter(adata, name, old_value, log_value=log_value)
