'''
Direct Pipeline
---------------
'''

import logging
from re import Pattern
from typing import Collection, Optional, Union

from anndata import AnnData

import metacells.parameters as pr
import metacells.utilities as ut

from .clean import extract_clean_data
from .direct import compute_direct_metacells
from .feature import extract_feature_data
from .results import collect_result_metacells

__all__ = [
    'direct_pipeline',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def direct_pipeline(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    properly_sampled_min_cell_total: Optional[int] = pr.properly_sampled_min_cell_total,
    properly_sampled_max_cell_total: Optional[int] = pr.properly_sampled_max_cell_total,
    properly_sampled_min_gene_total: int = pr.properly_sampled_min_gene_total,
    noisy_lonely_max_sampled_cells: int = pr.noisy_lonely_max_sampled_cells,
    noisy_lonely_downsample_cell_quantile: float = pr.noisy_lonely_downsample_cell_quantile,
    noisy_lonely_min_gene_fraction: float = pr.noisy_lonely_min_gene_fraction,
    noisy_lonely_min_gene_normalized_variance: float = pr.noisy_lonely_min_gene_normalized_variance,
    noisy_lonely_max_gene_similarity: float = pr.noisy_lonely_max_gene_similarity,
    excluded_gene_names: Optional[Collection[str]] = None,
    excluded_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    feature_downsample_cell_quantile: float = pr.feature_downsample_cell_quantile,
    feature_min_gene_fraction: float = pr.feature_min_gene_fraction,
    feature_min_gene_relative_variance: float = pr.feature_min_gene_relative_variance,
    forbidden_gene_names: Optional[Collection[str]] = None,
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    cells_similarity_log_data: bool = pr.cells_similarity_log_data,
    cells_similarity_log_normalization: float = pr.cells_similarity_log_normalization,
    cells_repeated_similarity: bool = pr.cells_repeated_similarity,
    knn_k: Optional[int] = None,
    knn_balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    knn_incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    knn_outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    candidates_partition_method: 'ut.PartitionMethod' = pr.candidates_partition_method,
    candidates_min_split_size_factor: Optional[float] = pr.candidates_min_split_size_factor,
    candidates_max_merge_size_factor: Optional[float] = pr.candidates_max_merge_size_factor,
    outliers_min_gene_fold_factor: float = pr.outliers_min_gene_fold_factor,
    outliers_max_gene_fraction: float = pr.outliers_max_gene_fraction,
    outliers_max_cell_fraction: float = pr.outliers_max_cell_fraction,
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    target_metacell_size: int = pr.target_metacell_size,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    random_seed: int = pr.random_seed,
    results_name: str = 'METACELLS',
    results_tmp: bool = False,
    intermediate: bool = True,
) -> Optional[AnnData]:
    '''
    Complete pipeline directly computing the metacells on the whole data.

    If your data is reasonably sized (up to O(10,000) cells), don't want to bother with details, and
    maybe just tweak a few parameters, use this and forget about most anything else in the package.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    All the computations will use the ``of`` data (by default, the focus).

    **Returns**

    Annotated metacell data containing for each observation the sum ``of`` the data (by default,
    the focus) of the cells for each metacell.

    Also sets the following in the full data:

    Observations (Cell) Annotations
        ``metacell``
            The index of the metacell each cell belongs to. This is ``-1`` for outlier cells and
            ``-2`` for excluded cells.

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.pipeline.clean_data.extract_clean_data` to extract the "clean"
       data from the full input data, using the ``properly_sampled_min_cell_total`` (default:
       {properly_sampled_min_cell_total}), ``properly_sampled_max_cell_total`` (default:
       {properly_sampled_max_cell_total}), ``excluded_gene_names`` (default: {excluded_gene_names})
       and ``excluded_gene_patterns`` (default: {excluded_gene_patterns}).

    2. Invoke :py:func:`metacells.pipeline.feature_data.extract_feature_data` to extract "feature"
       data from the clean data, using the ``noisy_lonely_max_sampled_cells`` (default:
       {noisy_lonely_max_sampled_cells}), ``noisy_lonely_downsample_cell_quantile`` (default:
       {noisy_lonely_downsample_cell_quantile}), ``noisy_lonely_min_gene_fraction`` (default:
       {noisy_lonely_min_gene_fraction}), ``noisy_lonely_min_gene_normalized_variance``
       (default: {noisy_lonely_min_gene_normalized_variance}), ``forbidden_gene_names`` (default:
       {forbidden_gene_names}), ``forbidden_gene_patterns`` (default: {forbidden_gene_patterns}),
       and the ``random_seed`` (default: {random_seed}) to make this replicable.

    3. Invoke :py:func:`metacells.pipeline.direct_metacells.compute_direct_metacells` to directly
       compute metacells using the clean and feature data, using ``cells_similarity_log_data``
       (default: {cells_similarity_log_data}), ``cells_similarity_log_normalization`` (default:
       {cells_similarity_log_normalization}), ``cells_repeated_similarity`` (default:
       {cells_repeated_similarity}), ``knn_k`` (default: {knn_k}), ``knn_balanced_ranks_factor``
       (default: {knn_balanced_ranks_factor}), ``knn_incoming_degree_factor`` (default:
       {knn_incoming_degree_factor}), ``knn_outgoing_degree_factor`` (default:
       {knn_outgoing_degree_factor}), ``candidates_partition_method`` (default:
       {candidates_partition_method.__qualname__}), ``candidates_min_split_size_factor`` (default:
       {candidates_min_split_size_factor}), ``candidates_max_merge_size_factor`` (default:
       {candidates_max_merge_size_factor}), ``outliers_min_gene_fold_factor`` (default:
       {outliers_min_gene_fold_factor}), ``outliers_max_gene_fraction`` (default:
       {outliers_max_gene_fraction}), ``outliers_max_cell_fraction`` (default:
       {outliers_max_cell_fraction}), ``dissolve_min_robust_size_factor`` (default:
       {dissolve_min_robust_size_factor}), ``dissolve_min_convincing_size_factor`` (default:
       {dissolve_min_convincing_size_factor}), ``dissolve_min_convincing_gene_fold_factor``
       (default: {dissolve_min_convincing_gene_fold_factor}), ``target_metacell_size`` (default:
       {target_metacell_size}), ``cell_sizes`` (default: {cell_sizes}), and the ``random_seed``
       (default: {random_seed}) to make this replicable.

    4. Invoke :py:func:`metacells.pipeline.result_metacells.collect_result_metacells` to collect the
       result metacells, using the ``results_name`` (default: {results_name}) and ``results_tmp``
       (default: {results_tmp}).
    '''
    cdata = \
        extract_clean_data(adata, of,
                           properly_sampled_min_cell_total=properly_sampled_min_cell_total,
                           properly_sampled_max_cell_total=properly_sampled_max_cell_total,
                           properly_sampled_min_gene_total=properly_sampled_min_gene_total,
                           noisy_lonely_max_sampled_cells=noisy_lonely_max_sampled_cells,
                           noisy_lonely_downsample_cell_quantile=noisy_lonely_downsample_cell_quantile,
                           noisy_lonely_min_gene_fraction=noisy_lonely_min_gene_fraction,
                           noisy_lonely_min_gene_normalized_variance=noisy_lonely_min_gene_normalized_variance,
                           noisy_lonely_max_gene_similarity=noisy_lonely_max_gene_similarity,
                           excluded_gene_names=excluded_gene_names,
                           excluded_gene_patterns=excluded_gene_patterns,
                           intermediate=intermediate)

    if cdata is None:
        LOG.warning('! Empty clean data, giving up')
        return None

    fdata = \
        extract_feature_data(cdata, of,
                             downsample_cell_quantile=feature_downsample_cell_quantile,
                             min_gene_relative_variance=feature_min_gene_relative_variance,
                             min_gene_fraction=feature_min_gene_fraction,
                             forbidden_gene_names=forbidden_gene_names,
                             forbidden_gene_patterns=forbidden_gene_patterns,
                             random_seed=random_seed)

    if fdata is None:
        LOG.warning('! Empty feature data, giving up')
        return None

    compute_direct_metacells(cdata, fdata, of,
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
                             outliers_min_gene_fold_factor=outliers_min_gene_fold_factor,
                             outliers_max_gene_fraction=outliers_max_gene_fraction,
                             outliers_max_cell_fraction=outliers_max_cell_fraction,
                             dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                             dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                             dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                             target_metacell_size=target_metacell_size,
                             cell_sizes=cell_sizes,
                             random_seed=random_seed)

    mdata = collect_result_metacells(adata, cdata, of,
                                     name=results_name, tmp=results_tmp,
                                     intermediate=intermediate)

    if mdata is None:
        LOG.warning('! Empty metacells data, giving up')
        return None

    return mdata
