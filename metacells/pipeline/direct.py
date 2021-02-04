'''
Direct
------
'''

import logging
from re import Pattern
from typing import Collection, Optional, Union

import numpy as np  # type: ignore
from anndata import AnnData

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

from .feature import extract_feature_data

__all__ = [
    'compute_direct_metacells',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def compute_direct_metacells(  # pylint: disable=too-many-branches,too-many-statements
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
    cells_similarity_method: str = pr.cells_similarity_method,
    target_metacell_size: int = pr.target_metacell_size,
    knn_k: Optional[int] = pr.knn_k,
    knn_balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    knn_incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    knn_outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    candidates_partition_method: 'ut.PartitionMethod' = pr.candidates_partition_method,
    candidates_min_split_size_factor: Optional[float] = pr.candidates_min_split_size_factor,
    candidates_max_merge_size_factor: Optional[float] = pr.candidates_max_merge_size_factor,
    candidates_min_metacell_cells: int = pr.min_metacell_cells,
    must_complete_cover: bool = False,
    deviants_min_gene_fold_factor: float = pr.deviants_min_gene_fold_factor,
    deviants_max_gene_fraction: Optional[float] = pr.deviants_max_gene_fraction,
    deviants_max_cell_fraction: Optional[float] = pr.deviants_max_cell_fraction,
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    dissolve_min_metacell_cells: int = pr.dissolve_min_metacell_cells,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    random_seed: int = pr.random_seed,
    intermediate: bool = True,
) -> None:
    '''
    Directly compute metacells.

    This directly computes the metacells on the whole data. Like any method that directly looks at
    the whole data at once, the amount of CPU and memory needed becomes unreasonable when the data
    size grows. Above O(10,000) you are much better off using the divide-and-conquer method.

    .. note::

        The current implementation is naive in that it computes the full dense N^2 correlation
        matrix, and only then extracts the sparse graph out of it. We actually need two copies where
        each requires 4 bytes per entry, so for O(100,000) cells, we have storage of
        O(100,000,000,000). In addition, the implementation is (mostly) serial for both the
        correlation and graph clustering phases.

        It is possible to mitigate this by fusing the correlations phase and the graph generation
        phase, parallelizing the result, and also (somehow) parallelizing the graph clustering
        phase. This might increase the "reasonable" size for the direct approach to O(100,000).

        We have decided not to invest in this direction since it won't allow us to push the size to
        O(1,000,000) and above. Instead we provide the divide-and-conquer method, which easily
        scales to O(1,000,000) on a single multi-core server, and to "unlimited" size if we further
        enhance the implementation to use a distributed compute cluster of such servers.

    .. todo::

        Should :py:func:`compute_direct_metacells` avoid computing the graph and running
        ``leidenalg`` for a very small number of cells?

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

        ``forbidden_gene`` (if ``intermediate``)
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name.

        ``feature_gene``
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

        ``outlier`` (if ``intermediate``)
            A boolean mask of the cells contained in no metacell.

    If ``intermediate`` (default: {intermediate}), also keep all all the intermediate data (e.g.
    sums) for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.pipeline.feature.extract_feature_data` to extract "feature" data
       from the clean data, using the
       ``feature_downsample_cell_quantile`` (default: {feature_downsample_cell_quantile}),
       ``feature_min_gene_fraction`` (default: {feature_min_gene_fraction}),
       ``feature_min_gene_relative_variance (default: {feature_min_gene_relative_variance}),
       ``forbidden_gene_names`` (default: {forbidden_gene_names}),
       ``forbidden_gene_patterns`` (default: {forbidden_gene_patterns})
       and
       ``random_seed`` (default: {random_seed})
       to make this replicable.

    2. If ``cells_similarity_log_data`` (default: {cells_similarity_log_data}), invoke the
       :py:func:`metacells.utilities.computation.log_data` function to compute the log (base 2) of
       the data, using the ``cells_similarity_log_normalization`` (default:
       {cells_similarity_log_normalization}).

    3. Invoke :py:func:`metacells.tools.similarity.compute_obs_obs_similarity` to compute the
       similarity between each pair of cells, using the
       ``cells_similarity_method`` (default: {cells_similarity_method}).

    4. Invoke :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph` to compute a
       K-Nearest-Neighbors graph, using the
       ``knn_balanced_ranks_factor`` (default: {knn_balanced_ranks_factor}),
       ``knn_incoming_degree_factor`` (default: {knn_incoming_degree_factor})
       and
       ``knn_outgoing_degree_factor`` (default: {knn_outgoing_degree_factor}).
       If ``knn_k`` (default: {knn_k}) is not specified, then it is
       chosen to be the mean number of cells required to reach the target metacell size.

    5. Invoke :py:func:`metacells.tools.candidates.compute_candidate_metacells` to compute
       the candidate metacells, using the
       ``candidates_partition_method`` (default: {candidates_partition_method.__qualname__}),
       ``candidates_min_split_size_factor`` (default: {candidates_min_split_size_factor}),
       ``candidates_max_merge_size_factor`` (default: {candidates_max_merge_size_factor}),
       ``candidates_min_metacell_cells`` (default: {candidates_min_metacell_cells}),
       and
       ``random_seed`` (default: {random_seed})
       to make this replicable. This tries to build metacells of the
       ``target_metacell_size`` (default: {target_metacell_size})
       using the
       ``cell_sizes`` (default: {cell_sizes}).

    6. Unless ``must_complete_cover`` (default: {must_complete_cover}), invoke
       :py:func:`metacells.tools.deviants.find_deviant_cells` to remove deviants from the candidate
       metacells, using the
       ``deviants_min_gene_fold_factor`` (default: {deviants_min_gene_fold_factor}),
       ``deviants_max_gene_fraction`` (default: {deviants_max_gene_fraction})
       and
       ``deviants_max_cell_fraction`` (default: {deviants_max_cell_fraction}).

    7. Unless ``must_complete_cover`` (default: {must_complete_cover}), invoke
       :py:func:`metacells.tools.dissolve.dissolve_metacells` to dissolve small unconvincing
       metacells, using the same
       ``target_metacell_size`` (default: {target_metacell_size}),
       and
       ``cell_sizes`` (default: {cell_sizes}),
       and the
       ``dissolve_min_robust_size_factor`` (default: {dissolve_min_robust_size_factor}),
       ``dissolve_min_convincing_size_factor`` (default: {dissolve_min_convincing_size_factor}),
       ``dissolve_min_convincing_gene_fold_factor`` (default: {dissolve_min_convincing_size_factor})
       and
       ``dissolve_min_metacell_cells`` (default: ``dissolve_min_metacell_cells``).
    '''
    data = ut.get_vo_proper(adata, of, layout='row_major')
    total_per_cell = ut.sum_per(data, per='row')

    fdata = \
        extract_feature_data(adata, of=of, tmp=True,
                             downsample_cell_quantile=feature_downsample_cell_quantile,
                             min_gene_relative_variance=feature_min_gene_relative_variance,
                             min_gene_fraction=feature_min_gene_fraction,
                             forbidden_gene_names=forbidden_gene_names,
                             forbidden_gene_patterns=forbidden_gene_patterns,
                             random_seed=random_seed,
                             intermediate=intermediate)

    if fdata is None:
        raise ValueError('Empty feature data, giving up')

    ut.log_pipeline_step(LOG, fdata, 'compute_direct_metacells')

    with ut.focus_on(ut.get_vo_data, fdata, of, intermediate=intermediate):
        focus = ut.get_focus_name(fdata)
        if isinstance(cell_sizes, str):
            cell_sizes = cell_sizes.replace('<of>', focus)
            LOG.debug('  cell_sizes: %s', cell_sizes)

        if cell_sizes == focus + '|sum_per_obs' and cell_sizes not in adata.obs:
            cell_sizes = ut.sum_per(data, per='row')

        cell_sizes = \
            ut.get_vector_parameter_data(LOG, adata, cell_sizes,
                                         indent='', per='o', name='cell_sizes')

        data = ut.get_vo_proper(fdata, of)
        data = ut.fraction_by(data, sums=total_per_cell, by='row')
        if cells_similarity_log_data:
            LOG.debug('  log of: %s base: 2 normalization: 1/%s',
                      focus, 1/cells_similarity_log_normalization)
            data = ut.log_data(data, base=2,
                               normalization=cells_similarity_log_normalization)

        ut.set_vo_data(fdata, 'data', data)
        with ut.focus_on(ut.get_vo_data, fdata, 'data', intermediate=intermediate):
            tl.compute_obs_obs_similarity(fdata, of=of,
                                          method=cells_similarity_method)

            if knn_k is None:
                if cell_sizes is None:
                    total_cell_sizes = fdata.n_obs
                else:
                    total_cell_sizes = np.sum(cell_sizes)
                knn_k = round(total_cell_sizes / target_metacell_size)

            LOG.debug('  knn_k: %s', knn_k)
            if knn_k == 0:
                LOG.debug('  too small, try a single metacell')
                ut.set_o_data(fdata, 'candidate',
                              np.full(fdata.n_obs, 0, dtype='int32'),
                              log_value=lambda _: '* <- 0')
            else:
                tl.compute_obs_obs_knn_graph(fdata,
                                             k=knn_k,
                                             balanced_ranks_factor=knn_balanced_ranks_factor,
                                             incoming_degree_factor=knn_incoming_degree_factor,
                                             outgoing_degree_factor=knn_outgoing_degree_factor)

                tl.compute_candidate_metacells(fdata,
                                               target_metacell_size=target_metacell_size,
                                               must_complete_cover=must_complete_cover,
                                               cell_sizes=cell_sizes,
                                               partition_method=candidates_partition_method,
                                               min_split_size_factor=candidates_min_split_size_factor,
                                               max_merge_size_factor=candidates_max_merge_size_factor,
                                               min_metacell_cells=candidates_min_metacell_cells,
                                               random_seed=random_seed)

            candidate_of_cells = ut.get_o_dense(fdata, 'candidate')

            if intermediate:
                ut.set_o_data(adata, 'candidate', candidate_of_cells,
                              log_value=ut.groups_description)
                outgoing_weights = \
                    ut.get_oo_proper(fdata, 'obs_outgoing_weights')
                ut.set_oo_data(adata, 'obs_outgoing_weights', outgoing_weights)

    if must_complete_cover:
        assert np.min(candidate_of_cells) == 0

        if intermediate:
            deviant_votes_of_genes = np.zeros(adata.n_vars, dtype='float32')
            deviant_votes_of_cells = np.zeros(adata.n_obs, dtype='float32')
            dissolved_of_cells = np.zeros(adata.n_obs, dtype='bool')

            ut.set_v_data(adata, 'gene_deviant_votes', deviant_votes_of_genes,
                          log_value=ut.mask_description)

            ut.set_o_data(adata, 'cell_deviant_votes', deviant_votes_of_cells,
                          log_value=ut.mask_description)

            ut.set_o_data(adata, 'dissolved', dissolved_of_cells,
                          log_value=ut.mask_description)

        ut.set_o_data(adata, 'metacell', candidate_of_cells,
                      log_value=ut.groups_description)

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
                                  min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                  min_metacell_cells=dissolve_min_metacell_cells)

    if intermediate:
        metacell_of_cells = ut.get_o_dense(adata, 'metacell')

        outlier_of_cells = metacell_of_cells < 0
        ut.set_o_data(adata, 'outlier', outlier_of_cells,
                      log_value=ut.mask_description)
