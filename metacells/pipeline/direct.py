'''
Direct Metacells
----------------

This directly computes the metacells on the whole data. Like any method that directly looks at the
whole data at once, the amount of CPU and memory needed becomes unreasonable when the data size
grows. At O(10,000) and definitely for anything above O(100,000) you are better off using the
divide-and-conquer method.
'''

import logging
from re import Pattern
from typing import Collection, Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.parameters as pr
import metacells.preprocessing as pp
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    'compute_direct_metacells',
    'extract_feature_data',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def compute_direct_metacells(
    adata: AnnData,
    of: Optional[str] = None,
    *,
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
    must_complete_cover: bool = False,
    outliers_min_gene_fold_factor: float = pr.outliers_min_gene_fold_factor,
    outliers_max_gene_fraction: float = pr.outliers_max_gene_fraction,
    outliers_max_cell_fraction: float = pr.outliers_max_cell_fraction,
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    target_metacell_size: int = pr.target_metacell_size,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    random_seed: int = 0,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Directly compute metacells.

    This is the heart of the metacells methodology.

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

    1. Invoke :py:func:`extract_feature_data` to extract "feature" data from the clean data, using
       the
       ``feature_downsample_cell_quantile`` (default: {feature_downsample_cell_quantile}),
       ``feature_min_gene_fraction`` (default: {feature_min_gene_fraction}),
       ``feature_min_gene_relative_variance (default: {feature_min_gene_relative_variance}),
       ``forbidden_gene_names`` (default: {forbidden_gene_names}),
       ``forbidden_gene_patterns`` (default: {forbidden_gene_patterns})
       and
       ``random_seed`` (default: {random_seed})
       to make this replicable.

    1. If ``cells_similarity_log_data`` (default: {cells_similarity_log_data}), invoke
       :py:func:`metacells.preprocessing.common.get_log_matrix` function to compute the log (base 2)
       of the data, using the
       ``cells_similarity_log_normalization`` (default: {cells_similarity_log_normalization}).

    2. Invoke :py:func:`metacells.tools.similarity.compute_obs_obs_similarity` to compute the
       similarity between each pair of cells, using the
       ``cells_repeated_similarity`` (default: {cells_repeated_similarity})
       to compensate for the sparseness of the data.

    3. Invoke :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph` to compute a
       K-Nearest-Neighbors graph, using the
       ``knn_balanced_ranks_factor`` (default: {knn_balanced_ranks_factor}),
       ``knn_incoming_degree_factor`` (default: {knn_incoming_degree_factor})
       and
       ``knn_outgoing_degree_factor`` (default: {knn_outgoing_degree_factor}).
       If ``knn_k`` (default: {knn_k}) is not specified, then it is
       chosen to be the mean number of cells required to reach the target metacell size.

    4. Invoke :py:func:`metacells.tools.candidates.compute_candidate_metacells` to compute
       the candidate metacells, using the
       ``candidates_partition_method`` (default: {candidates_partition_method.__qualname__}),
       ``candidates_min_split_size_factor`` (default: {candidates_min_split_size_factor}),
       ``candidates_max_merge_size_factor`` (default: {candidates_max_merge_size_factor})
       and
       ``random_seed`` (default: {random_seed})
       to make this replicable. This tries to build metacells of the
       ``target_metacell_size`` (default: {target_metacell_size})
       using the
       ``cell_sizes`` (default: {cell_sizes}).

    5. Unless ``must_complete_cover`` (default: {must_complete_cover}), invoke
       :py:func:`metacells.tools.outliers.find_outlier_cells` to remove outliers from the candidate
       metacells, using the
       ``outliers_min_gene_fold_factor`` (default: {outliers_min_gene_fold_factor}),
       ``outliers_max_gene_fraction`` (default: {outliers_max_gene_fraction})
       and
       ``outliers_max_cell_fraction`` (default: {outliers_max_cell_fraction}).

    6. Unless ``must_complete_cover`` (default: {must_complete_cover}), invoke
       :py:func:`metacells.tools.dissolve.dissolve_metacells` to dissolve small unconvincing
       metacells, using the same
       ``target_metacell_size`` (default: {target_metacell_size}),
       and
       ``cell_sizes`` (default: {cell_sizes}),
       and the
       ``dissolve_min_robust_size_factor`` (default: {dissolve_min_robust_size_factor}),
       ``dissolve_min_convincing_size_factor`` (default: {dissolve_min_convincing_size_factor}),
       and
       ``dissolve_min_convincing_gene_fold_factor`` (default: {dissolve_min_convincing_size_factor}).
    '''
    fdata = \
        extract_feature_data(adata, of,
                             downsample_cell_quantile=feature_downsample_cell_quantile,
                             min_gene_relative_variance=feature_min_gene_relative_variance,
                             min_gene_fraction=feature_min_gene_fraction,
                             forbidden_gene_names=forbidden_gene_names,
                             forbidden_gene_patterns=forbidden_gene_patterns,
                             random_seed=random_seed)

    if fdata is None:
        raise ValueError('Empty feature data, giving up')

    level = ut.log_pipeline_step(LOG, fdata, 'compute_direct_metacells')

    with ut.focus_on(ut.get_vo_data, fdata, of, intermediate=intermediate):
        if isinstance(cell_sizes, str):
            cell_sizes = cell_sizes.replace('<of>', ut.get_focus_name(fdata))

        cell_sizes = \
            ut.get_vector_parameter_data(LOG, level, adata, cell_sizes,
                                         indent='', per='o', name='cell_sizes')

        if cells_similarity_log_data:
            LOG.log(level, 'log of: %s base: 2 normalization: 1/%s',
                    ut.get_focus_name(fdata), 1/cells_similarity_log_normalization)
            of = pp.get_log_matrix(fdata, base=2,
                                   normalization=cells_similarity_log_normalization).name
        else:
            LOG.log(level, 'log: None')
            of = None

        tl.compute_obs_obs_similarity(fdata, of=of,
                                      repeated=cells_repeated_similarity)

        if knn_k is None:
            if cell_sizes is None:
                total_cell_sizes = fdata.n_obs
            else:
                total_cell_sizes = np.sum(cell_sizes)
            knn_k = round(total_cell_sizes / target_metacell_size)

        tl.compute_obs_obs_knn_graph(fdata,
                                     k=knn_k,
                                     balanced_ranks_factor=knn_balanced_ranks_factor,
                                     incoming_degree_factor=knn_incoming_degree_factor,
                                     outgoing_degree_factor=knn_outgoing_degree_factor)

        tl.compute_candidate_metacells(fdata,
                                       target_metacell_size=target_metacell_size,
                                       cell_sizes=cell_sizes,
                                       partition_method=candidates_partition_method,
                                       min_split_size_factor=candidates_min_split_size_factor,
                                       max_merge_size_factor=candidates_max_merge_size_factor,
                                       random_seed=random_seed)

        candidate_metacell_of_cells = \
            ut.to_dense_vector(ut.get_o_data(fdata, 'candidate_metacell'))

    if must_complete_cover:
        final_metacell_of_cells = candidate_metacell_of_cells
        assert np.min(final_metacell_of_cells) == 0
    else:
        with ut.intermediate_step(adata, intermediate=intermediate):
            ut.set_o_data(adata, 'candidate_metacell',
                          candidate_metacell_of_cells, ut.NEVER_SAFE)

            tl.find_outlier_cells(adata,
                                  min_gene_fold_factor=outliers_min_gene_fold_factor,
                                  max_gene_fraction=outliers_max_gene_fraction,
                                  max_cell_fraction=outliers_max_cell_fraction)

            tl.dissolve_metacells(adata,
                                  target_metacell_size=target_metacell_size,
                                  cell_sizes=cell_sizes,
                                  min_robust_size_factor=dissolve_min_robust_size_factor,
                                  min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                  min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor)

            final_metacell_of_cells = \
                ut.to_dense_vector(ut.get_o_data(adata, 'solid_metacell'))

    if inplace:
        ut.set_o_data(adata, 'metacell', final_metacell_of_cells,
                      log_value=lambda: str(np.max(final_metacell_of_cells) + 1))
        return None

    return pd.Series(final_metacell_of_cells, index=adata.obs_names)


@ut.timed_call()
@ut.expand_doc()
def extract_feature_data(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    name: Optional[str] = 'FEATURE',
    tmp: bool = False,
    downsample_cell_quantile: float = pr.feature_downsample_cell_quantile,
    min_gene_fraction: float = pr.feature_min_gene_fraction,
    min_gene_relative_variance: float = pr.feature_min_gene_relative_variance,
    forbidden_gene_names: Optional[Collection[str]] = None,
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    random_seed: int = 0,
    intermediate: bool = True,
) -> Optional[AnnData]:
    '''
    Extract a "feature" subset of the ``adata`` to compute metacells for.

    When computing metacells (or clustering cells in general), it makes sense to use a subset of the
    genes for computing cell-cell similarity, for both technical (e.g., too low an expression level)
    and biological (e.g., ignoring bookkeeping and cell cycle genes) reasons. The steps provided
    here are expected to be generically useful, but as always specific data sets may require custom
    feature selection steps on a case-by-case basis.

    **Input**

    A presumably "clean" :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where
    the observations are cells and the variables are genes.

    All the computations will use the ``of`` data (by default, the focus).

    **Returns**

    Annotated sliced data containing the "feature" subset of the original data. The focus of the
    data will be the (slice) ``of`` the (downsampled) input data. By default, the ``name`` of this
    data is ``FEATURE``.

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.downsample.downsample_cells` to downsample the cells to the
       same total number of UMIs, using the ``downsample_cell_quantile`` (default:
       {downsample_cell_quantile}) and the ``random_seed`` (default: {random_seed}).

    2. Invoke :py:func:`metacells.tools.high.find_high_fraction_genes` to select high-expression
       feature genes (based on the downsampled data), using ``min_gene_fraction``.

    3. Invoke :py:func:`metacells.tools.high.find_high_relative_variance_genes` to select
       high-variance feature genes (based on the downsampled data), using
       ``min_gene_relative_variance``.

    4. Invoke :py:func:`metacells.tools.named.find_named_genes` to forbid genes from being used as
       feature genes, based on their name. using the ``forbidden_gene_names`` (default:
       {forbidden_gene_names}) and ``forbidden_gene_patterns`` (default: {forbidden_gene_patterns}).
       This is stored in an intermediate per-variable (gene) ``forbidden_genes`` boolean mask.

    5. Invoke :py:func:`metacells.preprocessing.filter.filter_data` to slice just the selected
       "feature" genes using the ``name`` (default: {name}) and ``tmp`` (default: {tmp}).
    '''
    ut.log_pipeline_step(LOG, adata, 'extract_feature_data')

    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        tl.downsample_cells(adata,
                            downsample_cell_quantile=downsample_cell_quantile,
                            random_seed=random_seed,
                            infocus=True)

        tl.find_high_fraction_genes(adata,
                                    min_gene_fraction=min_gene_fraction)

        tl.find_high_relative_variance_genes(adata,
                                             min_gene_relative_variance=min_gene_relative_variance)

        if forbidden_gene_names is not None \
                or forbidden_gene_patterns is not None:
            tl.find_named_genes(adata,
                                to='forbidden_genes',
                                names=forbidden_gene_names,
                                patterns=forbidden_gene_patterns)

        fdata = pp.filter_data(adata, name=name, tmp=tmp,
                               masks=['high_fraction_genes',
                                      'high_relative_variance_genes',
                                      '~forbidden_genes'])

    if fdata is not None:
        ut.get_vo_data(fdata, ut.get_focus_name(adata), infocus=True)

    return fdata
