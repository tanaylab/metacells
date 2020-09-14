'''
Direct
------

This directly computes the metacells on the whole data. Like any method that directly looks at the
whole data at once, the amount of CPU and memory needed becomes unreasonable when the data size
grows. At O(10,000) and definitely for anything above O(100,000) you are better off using the
divide-and-conquer method.
'''

import logging
from re import Pattern
from typing import Collection, Optional, Union

import numpy as np  # type: ignore
from anndata import AnnData

import metacells.parameters as pr
import metacells.preprocessing as pp
import metacells.tools as tl
import metacells.utilities as ut

from .clean import extract_clean_data

__all__ = [
    'direct_pipeline',
    'compute_direct_metacells',
    'extract_feature_data',
    'collect_direct_metacells',
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
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    target_metacell_size: int = pr.target_metacell_size,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    random_seed: int = pr.random_seed,
    results_name: str = 'metacells',
    results_tmp: bool = False,
    intermediate: bool = True,
) -> Optional[AnnData]:
    '''
    Complete pipeline directly computing the metacells on the whole data.

    .. note::

        This is only applicable to "reasonably" sized data, (up to O(10,000) cells). Above this
        point, both CPU and memory costs grow (roughly quadratically). In addition, expensive
        ``leidenalg`` graph partitioning algorithm which takes a significant fraction of the overall
        processing time is inherently serial, so will not benefit from
        :py:func:`metacells.utilities.parallel.get_cpus_count`.

        You are therefore typically better off invoking TODO-divide-and-conquer-pipeline. If
        the data is small, it will will automatically revert to internally invoking the single
        :py:func:`direct_pipeline`. Only call :py:func:`direct_pipeline` if you have a strong reason
        to force using the direct method for larger data (e.g., benchmarking).

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    All the computations will use the ``of`` data (by default, the focus).

    **Returns**

    Annotated metacell data containing for each observation the sum ``of`` the data (by default,
    the focus) of the cells for each metacell, which contains the following annotations:

    Variable (Gene) Annotations
        ``excluded`` (if ``intermediate``)
            A mask of the genes which were excluded by name.

        ``clean_gene`` (if ``intermediate``)
            A boolean mask of the clean genes.

        ``forbidden`` (if ``intermediate``)
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name. This is ``False`` for non-"clean" genes.

        ``feature``
            A boolean mask of the "feature" genes. This is ``False`` for non-"clean" genes.

    Observations (Cell) Annotations
        ``grouped``
            The number of ("clean") cells grouped into each metacell.

        ``candidate`` (if ``intermediate``)
            The index of the candidate metacell each cell was assigned to to. This is ``-1`` for
            excluded cells.

    Sets the following in the full ``adata``:

    Variable (Gene) Annotations
        ``properly_sampled_gene`` (if ``intermediate``)
            A mask of the "properly sampled" genes.

        ``noisy_lonely_gene`` (if ``intermediate``)
            A mask of the "noisy lonely" genes.

        ``excluded`` (if ``intermediate``)
            A mask of the genes which were excluded by name.

        ``clean_gene`` (if ``intermediate``)
            A boolean mask of the clean genes.

        ``high_fraction_gene`` (if ``intermediate``)
            A boolean mask of genes with "high" expression level. This is ``False`` for non-"clean" genes.

        ``high_relative_variance_gene`` (if ``intermediate``)
            A boolean mask of genes with "high" normalized variance, relative to other genes with a
            similar expression level. This is ``False`` for non-"clean" genes.

        ``forbidden`` (if ``intermediate``)
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name. This is ``False`` for non-"clean" genes.

        ``feature``
            A boolean mask of the "feature" genes. This is ``False`` for non-"clean" genes.

        ``gene_deviant_votes`` (if ``intermediate``)
            The number of cells each gene marked as deviant (if zero, the gene did not mark any cell
            as deviant). This will be zero for non-"feature" genes.

    Observations (Cell) Annotations
        ``properly_sampled_cell``
            A mask of the "properly sampled" cells.

        ``clean_cell`` (if ``intermediate``)
            A boolean mask of the clean cells.

        ``candidate`` (if ``intermediate``)
            The index of the candidate metacell each cell was assigned to to. This is ``-1`` for
            non-"clean" cells.

        ``cell_deviant_votes`` (if ``intermediate``)
            The number of genes that were the reason the cell was marked as deviant (if zero, the
            cell is not deviant). This is zero for non-"clean" cells.

        ``dissolved`` (if ``intermediate``)
            A boolean mask of the cells contained in a dissolved metacell. This is ``False`` for
            non-"clean" cells.

        ``metacell``
            The integer index of the metacell each cell belongs to. The metacells are in no
            particular order. This is ``-1`` for outlier cells and ``-2`` for non-"clean" cells.





    Sets the following in the full ``adata``:

    Variable (Gene) Annotations
        ``high_fraction_gene`` (if ``intermediate``)
            A boolean mask of genes with "high" expression level.

        ``high_relative_variance_gene`` (if ``intermediate``)
            A boolean mask of genes with "high" normalized variance, relative to other genes with a
            similar expression level.

        ``forbidden`` (if ``intermediate``)
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name.

        ``feature``
            A boolean mask of the "feature" genes.

        ``gene_deviant_votes`` (if ``intermediate``)
            The number of cells each gene marked as deviant (if zero, the gene did not mark any cell
            as deviant). This will be zero for non-"feature" genes.

    Observation (Cell) Annotations
        ``candidate`` (if ``intermediate``)
            The index of the candidate metacell each cell was assigned to to. This is ``-1`` for
            non-"clean" cells.

        ``cell_deviant_votes`` (if ``intermediate``)
            The number of genes that were the reason the cell was marked as deviant (if zero, the
            cell is not deviant).

        ``dissolved`` (if ``intermediate``)
            A boolean mask of the cells contained in a dissolved metacell.

        ``metacell``
            The integer index of the metacell each cell belongs to. The metacells are in no
            particular order. Cells with no metacell assignment ("outliers") are given a metacell
            index of ``-1``.


    Observations (Cell) Annotations
        ``clean_cell`` (if ``intermediate``)
            A boolean mask of the "clean" cells.

        ``metacell``
            The index of the metacell each cell belongs to. This is ``-1`` for outlier cells and
            ``-2`` for excluded cells.

        ``candidate`` (if ``intermediate``)
            The index of the candidate metacell each cell was assigned to to. This is ``-1`` for
            excluded cells.

        ``cell_deviant_votes`` (if ``intermediate``)
            The number of genes that were the reason the cell was marked as deviant (if zero, the
            cell is not deviant).

        ``dissolved`` (if ``intermediate``)
            A boolean mask of the cells contained in a dissolved metacell.

    Variable (Gene) Annotations
        ``clean_gene``
            A boolean mask of the "clean" genes.

        ``feature``
            A boolean mask of the "feature" genes.

        ``gene_deviant_votes`` (if ``intermediate``)
            The number of cells each gene marked as deviant (if zero, the gene did not mark any cell
            as deviant).

    If ``intermediate`` (default: {intermediate}), also keep all all the intermediate data (e.g.
    sums) for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.pipeline.clean.extract_clean_data` to extract the "clean" data
       from the full input data, using the
       ``properly_sampled_min_cell_total`` (default: {properly_sampled_min_cell_total}),
       ``properly_sampled_max_cell_total`` (default: {properly_sampled_max_cell_total}),
       ``properly_sampled_min_gene_total`` (default: {properly_sampled_min_gene_total}),
       ``noisy_lonely_max_sampled_cells`` (default: {noisy_lonely_max_sampled_cells}),
       ``noisy_lonely_downsample_cell_quantile`` (default: {noisy_lonely_downsample_cell_quantile}),
       ``noisy_lonely_min_gene_fraction`` (default: {noisy_lonely_min_gene_fraction}),
       ``noisy_lonely_min_gene_normalized_variance`` (default: {noisy_lonely_min_gene_normalized_variance}),
       ``noisy_lonely_max_gene_similarity`` (default: {noisy_lonely_max_gene_similarity}),
       ``excluded_gene_names`` (default: {excluded_gene_names})
       and
       ``excluded_gene_patterns`` (default: {excluded_gene_patterns}).

    2. Invoke :py:func:`compute_direct_metacells` to directly compute
       metacells using the clean data, using the
       ``feature_downsample_cell_quantile`` (default: {feature_downsample_cell_quantile}),
       ``feature_min_gene_fraction`` (default: {feature_min_gene_fraction}),
       ``feature_min_gene_relative_variance (default: {feature_min_gene_relative_variance}),
       ``forbidden_gene_names`` (default: {forbidden_gene_names}),
       ``forbidden_gene_patterns`` (default: {forbidden_gene_patterns}),
       ``cells_similarity_log_data`` (default: {cells_similarity_log_data}),
       ``cells_similarity_log_normalization`` (default: {cells_similarity_log_normalization}),
       ``cells_repeated_similarity`` (default: {cells_repeated_similarity}),
       ``knn_k`` (default: {knn_k}),
       ``knn_balanced_ranks_factor`` (default: {knn_balanced_ranks_factor}),
       ``knn_incoming_degree_factor`` (default: {knn_incoming_degree_factor}),
       ``knn_outgoing_degree_factor`` (default: {knn_outgoing_degree_factor}),
       ``candidates_partition_method`` (default: {candidates_partition_method.__qualname__}),
       ``candidates_min_split_size_factor`` (default: {candidates_min_split_size_factor}),
       ``candidates_max_merge_size_factor`` (default: {candidates_max_merge_size_factor}),
       ``must_complete_cover`` (default: {must_complete_cover}),
       ``deviants_min_gene_fold_factor`` (default: {deviants_min_gene_fold_factor}),
       ``deviants_max_gene_fraction`` (default: {deviants_max_gene_fraction}),
       ``deviants_max_cell_fraction`` (default: {deviants_max_cell_fraction}),
       ``dissolve_min_robust_size_factor`` (default: {dissolve_min_robust_size_factor}),
       ``dissolve_min_convincing_size_factor`` (default: {dissolve_min_convincing_size_factor}),
       ``dissolve_min_convincing_gene_fold_factor`` (default: {dissolve_min_convincing_gene_fold_factor}),
       ``target_metacell_size`` (default: {target_metacell_size}),
       ``cell_sizes`` (default: {cell_sizes})
       and
       ``random_seed`` (default: {random_seed})
       to make this replicable.

    3. Invoke :py:func:`collect_direct_metacells` to collect the result metacells, using the
       ``results_name`` (default: {results_name})
       and
       ``results_tmp`` (default: {results_tmp}).
    '''
    cdata = \
        extract_clean_data(adata, of, tmp=True,
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
        raise ValueError('Empty clean data, giving up')

    compute_direct_metacells(cdata, of,
                             feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                             feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                             feature_min_gene_fraction=feature_min_gene_fraction,
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
                             intermediate=intermediate)

    mdata = collect_direct_metacells(adata, cdata, of,
                                     name=results_name, tmp=results_tmp,
                                     intermediate=intermediate)

    if mdata is None:
        raise ValueError('Empty metacells data, giving up')

    return mdata


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
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    target_metacell_size: int = pr.target_metacell_size,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    random_seed: int = pr.random_seed,
    intermediate: bool = True,
) -> None:
    '''
    Directly compute metacells.

    This is the heart of the metacells methodology.

    **Input**

    The presumably "clean" :py:func:`metacells.utilities.annotation.setup` annotated ``adata``.

    All the computations will use the ``of`` data (by default, the focus).

    **Returns**

    Sets the following annotations in ``adata``:

    Variable (Gene) Annotations
        ``high_fraction_gene`` (if ``intermediate``)
            A boolean mask of genes with "high" expression level.

        ``high_relative_variance_gene`` (if ``intermediate``)
            A boolean mask of genes with "high" normalized variance, relative to other genes with a
            similar expression level.

        ``forbidden`` (if ``intermediate``)
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name.

        ``feature``
            A boolean mask of the "feature" genes.

        ``gene_deviant_votes`` (if ``intermediate``)
            The number of cells each gene marked as deviant (if zero, the gene did not mark any cell
            as deviant). This will be zero for non-"feature" genes.

    Observation (Cell) Annotations
        ``candidate`` (if ``intermediate``)
            The index of the candidate metacell each cell was assigned to to. This is ``-1`` for
            non-"clean" cells.

        ``cell_deviant_votes`` (if ``intermediate``)
            The number of genes that were the reason the cell was marked as deviant (if zero, the
            cell is not deviant).

        ``dissolved`` (if ``intermediate``)
            A boolean mask of the cells contained in a dissolved metacell.

        ``metacell``
            The integer index of the metacell each cell belongs to. The metacells are in no
            particular order. Cells with no metacell assignment ("outliers") are given a metacell
            index of ``-1``.

    If ``intermediate`` (default: {intermediate}), also keep all all the intermediate data (e.g.
    sums) for future reuse. Otherwise, discard it.

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
       :py:func:`metacells.tools.deviants.find_deviant_cells` to remove deviants from the candidate
       metacells, using the
       ``deviants_min_gene_fold_factor`` (default: {deviants_min_gene_fold_factor}),
       ``deviants_max_gene_fraction`` (default: {deviants_max_gene_fraction})
       and
       ``deviants_max_cell_fraction`` (default: {deviants_max_cell_fraction}).

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
        extract_feature_data(adata, of, tmp=True,
                             downsample_cell_quantile=feature_downsample_cell_quantile,
                             min_gene_relative_variance=feature_min_gene_relative_variance,
                             min_gene_fraction=feature_min_gene_fraction,
                             forbidden_gene_names=forbidden_gene_names,
                             forbidden_gene_patterns=forbidden_gene_patterns,
                             random_seed=random_seed,
                             intermediate=intermediate)

    if fdata is None:
        raise ValueError('Empty feature data, giving up')

    level = ut.log_pipeline_step(LOG, fdata, 'compute_direct_metacells')

    with ut.focus_on(ut.get_vo_data, fdata, of, intermediate=intermediate):
        focus = ut.get_focus_name(fdata)
        if isinstance(cell_sizes, str):
            cell_sizes = cell_sizes.replace('<of>', focus)

        if cell_sizes == focus + '|sum_per_obs' and cell_sizes not in adata.obs:
            pp.get_per_obs(adata, ut.sum_per, of=focus)

        cell_sizes = \
            ut.get_vector_parameter_data(LOG, level, adata, cell_sizes,
                                         indent='', per='o', name='cell_sizes')

        if cells_similarity_log_data:
            LOG.log(level, 'log of: %s base: 2 normalization: 1/%s',
                    focus, 1/cells_similarity_log_normalization)
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

        candidate_of_cells = \
            ut.to_dense_vector(ut.get_o_data(fdata, 'candidate'))

        if intermediate:
            ut.set_o_data(adata, 'candidate', candidate_of_cells)

    if must_complete_cover:
        assert np.min(candidate_of_cells) == 0

        if intermediate:
            deviant_votes_of_genes = np.zeros(adata.n_vars, dtype='float32')
            deviant_votes_of_cells = np.zeros(adata.n_obs, dtype='float32')
            dissolved_of_cells = np.zeros(adata.n_obs, dtype='bool')

            assert deviant_votes_of_cells is not None
            assert dissolved_of_cells is not None

            ut.set_v_data(adata, 'gene_deviant_votes', deviant_votes_of_genes,
                          log_value=lambda:
                          ut.mask_description(deviant_votes_of_genes > 0))

            ut.set_o_data(adata, 'cell_deviant_votes', deviant_votes_of_cells,
                          log_value=lambda:
                          ut.mask_description(deviant_votes_of_cells > 0))

            ut.set_o_data(adata, 'dissolved', dissolved_of_cells,
                          log_value=lambda:
                          ut.mask_description(dissolved_of_cells))

        ut.set_o_data(adata, 'metacell', candidate_of_cells,
                      log_value=lambda:
                      ut.groups_description(candidate_of_cells))

    else:
        with ut.intermediate_step(adata, intermediate=intermediate, keep='metacell'):
            tl.find_deviant_cells(adata,
                                  candidates=candidate_of_cells,
                                  min_gene_fold_factor=deviants_min_gene_fold_factor,
                                  max_gene_fraction=deviants_max_gene_fraction,
                                  max_cell_fraction=deviants_max_cell_fraction)

            tl.dissolve_metacells(adata,
                                  candidates=candidate_of_cells,
                                  target_metacell_size=target_metacell_size,
                                  cell_sizes=cell_sizes,
                                  min_robust_size_factor=dissolve_min_robust_size_factor,
                                  min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                  min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor)


@ut.timed_call()
@ut.expand_doc()
def extract_feature_data(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    name: Optional[str] = 'feature',
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

    Returns annotated sliced data containing the "feature" subset of the original data. The focus of
    the data will be the (slice) ``of`` the (downsampled) input data. By default, the ``name`` of
    this data is {name}. If no features were selected, return ``None``.

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse, and set the ``feature`` per-variable boolean mask of ``adata``. Otherwise,
    discard all intermediate data.

    If ``intermediate``, this will also set the following annotations in the full ``adata``:

    Variable (Gene) Annotations
        ``high_fraction_gene``
            A boolean mask of genes with "high" expression level.

        ``high_relative_variance_gene``
            A boolean mask of genes with "high" normalized variance, relative to other genes with a
            similar expression level.

        ``forbidden``
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name.

        ``feature``
            A boolean mask of the "feature" genes.

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

    with ut.focus_on(ut.get_vo_data, adata, of,
                     intermediate=intermediate, keep='feature'):
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
                                to='forbidden',
                                names=forbidden_gene_names,
                                patterns=forbidden_gene_patterns)

        results = pp.filter_data(adata, name=name, tmp=tmp,
                                 mask_var='feature',
                                 masks=['high_fraction_gene',
                                        'high_relative_variance_gene',
                                        '~forbidden'])
        if results is None:
            raise ValueError('Empty feature data, giving up')

    fdata = results[0]

    ut.get_vo_data(fdata, ut.get_focus_name(adata), infocus=True)

    return fdata


@ut.timed_call()
def collect_direct_metacells(
    adata: AnnData,
    cdata: AnnData,
    of: Optional[str] = None,
    *,
    name: str = 'metacells',
    tmp: bool = False,
    intermediate: bool = True,
) -> Optional[AnnData]:
    '''
    Collect the result metacells directly computed using the clean data.

    **Input**

    The full :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the
    observations are cells and the variables are genes, and the clean ``cdata`` we have computed
    metacells for using :py:func:`metacells.pipeline.direct.compute_direct_metacells`.

    **Returns**

    This will pass all the computed per-observation (cell) and per-variable (gene) annotations from
    ``cdata`` back into the full ``adata``. Non-"clean" cells are given a ``metacell`` index of
    ``-1``, a ``candidate`` index of ``-1``, and otherwise the value in all the annotations of all
    the non-"clean" cells/genes is zero (``False`` for booleans).

    Returns annotated metacell data containing for each observation the sum ``of`` the data (by
    default, the focus) of the cells for each metacell. This will preserve any annotations of the
    data which are safe for slicing observations (cells); in particular, it will preserve most
    per-variable (gene) annotations such as ``feature``.

    The result will also have the following annotations:

    Variable (Gene) Annotations
        ``excluded`` (if ``intermediate``)
            A mask of the genes which were excluded by name.

        ``clean_gene`` (if ``intermediate``)
            A boolean mask of the clean genes.

        ``forbidden`` (if ``intermediate``)
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name. This is ``False`` for non-"clean" genes.

        ``feature``
            A boolean mask of the "feature" genes. This is ``False`` for non-"clean" genes.

    Observation (Cell) Annotations
        ``grouped``
            The number of ("clean") cells grouped into each metacell.

        ``candidate`` (if ``intermediate``)
            The index of the metacell when it was only a candidate. This allows mapping deviant
            cells to the metacell they were rejected from.

    If ``intermediate`` (default: {intermediate}), also keep all all the intermediate data (e.g.
    sums) for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Pass all the annotations from the clean ``cdata`` to the full ``adata``.

    2. Invoke :py:func:`metacells.preprocessing.group.group_obs_data` to sum the ``of`` data into a
       new metacells annotated data based on the computed ``metacell`` annotation, using the
       ``name`` (default: {name}) and ``tmp`` (default: {tmp}).

    3. If ``intermediate``, invoke :py:func:`metacells.preprocessing.group.group_obs_annotation` to
       fill the ``candidate`` annotation of the result.
    '''
    ut.log_pipeline_step(LOG, adata, 'collect_direct_metacells')

    if LOG.isEnabledFor(logging.DEBUG):
        clean_name = ut.get_name(cdata)
        if clean_name is not None:
            LOG.debug('  collect metacells from: %s', clean_name)

    clean_gene_indices = ut.get_v_data(cdata, 'full_gene_index')

    for per_gene_name, value_of_clean_genes in ut.annotation_items(cdata.var):
        if per_gene_name in adata.var \
                or per_gene_name == 'full_gene_index' \
                or '|' in per_gene_name:
            continue
        value_of_all_genes = \
            np.zeros(adata.n_vars, dtype=value_of_clean_genes.dtype)
        value_of_all_genes[clean_gene_indices] = value_of_clean_genes
        ut.set_v_data(adata, per_gene_name, value_of_all_genes)

    clean_cell_indices = ut.get_o_data(cdata, 'full_cell_index')

    for per_cell_name, value_of_clean_cells in ut.annotation_items(cdata.obs):
        if per_cell_name in adata.var \
                or per_cell_name == 'full_cell_index' \
                or '|' in per_cell_name:
            continue
        if per_cell_name == 'metacell':
            value_of_all_cells = \
                np.full(adata.n_obs, -2, dtype=value_of_clean_cells.dtype)
        elif per_cell_name == 'candidate':
            value_of_all_cells = \
                np.full(adata.n_obs, -1, dtype=value_of_clean_cells.dtype)
        else:
            value_of_all_cells = \
                np.zeros(adata.n_obs, dtype=value_of_clean_cells.dtype)
        value_of_all_cells[clean_cell_indices] = value_of_clean_cells
        ut.set_o_data(adata, per_cell_name, value_of_all_cells)

    mdata = \
        pp.group_obs_data(adata, of=of, groups='metacell', name=name, tmp=tmp)
    assert mdata is not None

    ut.set_v_data(mdata, 'feature',
                  ut.to_dense_vector(ut.get_v_data(adata, 'feature')))

    if intermediate:
        for per_gene_name in ('clean_gene', 'forbidden', 'excluded'):
            ut.set_v_data(mdata, per_gene_name,
                          ut.to_dense_vector(ut.get_v_data(adata, per_gene_name)))

        pp.group_obs_annotation(adata, mdata, groups='metacell',
                                name='candidate', method='unique')

    return mdata
