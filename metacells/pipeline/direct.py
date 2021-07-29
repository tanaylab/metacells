'''
Direct
------
'''

from re import Pattern
from typing import Collection, Optional, Union

import numpy as np
from anndata import AnnData

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

from .collect import compute_effective_cell_sizes
from .feature import extract_feature_data

__all__ = [
    'compute_direct_metacells',
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_direct_metacells(  # pylint: disable=too-many-statements,too-many-branches
    adata: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    *,
    feature_downsample_min_samples: int = pr.feature_downsample_min_samples,
    feature_downsample_min_cell_quantile: float = pr.feature_downsample_min_cell_quantile,
    feature_downsample_max_cell_quantile: float = pr.feature_downsample_max_cell_quantile,
    feature_min_gene_total: int = pr.feature_min_gene_total,
    feature_min_gene_top3: int = pr.feature_min_gene_top3,
    feature_min_gene_relative_variance: float = pr.feature_min_gene_relative_variance,
    forbidden_gene_names: Optional[Collection[str]] = None,
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    cells_similarity_value_normalization: float = pr.cells_similarity_value_normalization,
    cells_similarity_log_data: bool = pr.cells_similarity_log_data,
    cells_similarity_method: str = pr.cells_similarity_method,
    target_metacell_size: float = pr.target_metacell_size,
    max_cell_size: Optional[float] = pr.max_cell_size,
    max_cell_size_factor: Optional[float] = pr.max_cell_size_factor,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    knn_k: Optional[int] = pr.knn_k,
    min_knn_k: Optional[int] = pr.min_knn_k,
    knn_balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    knn_incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    knn_outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    candidates_cell_seeds: Optional[Union[str, ut.Vector]] = None,
    min_seed_size_quantile: float = pr.min_seed_size_quantile,
    max_seed_size_quantile: float = pr.max_seed_size_quantile,
    candidates_cooldown_pass: float = pr.cooldown_pass,
    candidates_cooldown_node: float = pr.cooldown_node,
    candidates_cooldown_phase: float = pr.cooldown_phase,
    candidates_min_split_size_factor: Optional[float] = pr.candidates_min_split_size_factor,
    candidates_max_merge_size_factor: Optional[float] = pr.candidates_max_merge_size_factor,
    candidates_min_metacell_cells: Optional[int] = pr.min_metacell_cells,
    must_complete_cover: bool = False,
    deviants_min_gene_fold_factor: float = pr.deviants_min_gene_fold_factor,
    deviants_max_gene_fraction: Optional[float] = pr.deviants_max_gene_fraction,
    deviants_max_cell_fraction: Optional[float] = pr.deviants_max_cell_fraction,
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    dissolve_min_metacell_cells: int = pr.dissolve_min_metacell_cells,
    random_seed: int = pr.random_seed,
) -> AnnData:
    '''
    Directly compute metacells using ``what`` (default: {what}) data.

    This directly computes the metacells on the whole data. Like any method that directly looks at
    the whole data at once, the amount of CPU and memory needed becomes unreasonable when the data
    size grows. Above O(10,000) you are much better off using the divide-and-conquer method.

    .. note::

        The current implementation is naive in that it computes the full dense N^2 correlation
        matrix, and only then extracts the sparse graph out of it. We actually need two copies where
        each requires 4 bytes per entry, so for O(100,000) cells, we have storage of
        O(100,000,000,000). In addition, the implementation is serial for the graph clustering
        phases.

        It is possible to mitigate this by fusing the correlations phase and the graph generation
        phase, parallelizing the result, and also (somehow) parallelizing the graph clustering
        phase. This might increase the "reasonable" size for the direct approach to O(100,000).

        We have decided not to invest in this direction since it won't allow us to push the size to
        O(1,000,000) and above. Instead we provide the divide-and-conquer method, which easily
        scales to O(1,000,000) on a single multi-core server, and to "unlimited" size if we further
        enhance the implementation to use a distributed compute cluster of such servers.

    .. todo::

        Should :py:func:`compute_direct_metacells` avoid computing the graph and partition it for a
        very small number of cells?

    **Input**

    The presumably "clean" annotated ``adata``, where the observations are cells and the variables
    are genes, where ``what`` is a per-variable-per-observation matrix or the name of a
    per-variable-per-observation annotation containing such a matrix.

    **Returns**

    Sets the following annotations in ``adata``:

    Variable (Gene) Annotations
        ``high_total_gene``
            A boolean mask of genes with "high" expression level.

        ``high_relative_variance_gene``
            A boolean mask of genes with "high" normalized variance, relative to other genes with a
            similar expression level.

        ``forbidden_gene``
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name.

        ``feature_gene``
            A boolean mask of the "feature" genes.

        ``gene_deviant_votes``
            The number of cells each gene marked as deviant (if zero, the gene did not mark any cell
            as deviant). This will be zero for non-"feature" genes.

    Observation (Cell) Annotations
        ``seed``
            The index of the seed metacell each cell was assigned to to. This is ``-1`` for
            non-"clean" cells.

        ``candidate``
            The index of the candidate metacell each cell was assigned to to. This is ``-1`` for
            non-"clean" cells.

        ``cell_deviant_votes``
            The number of genes that were the reason the cell was marked as deviant (if zero, the
            cell is not deviant).

        ``dissolved``
            A boolean mask of the cells contained in a dissolved metacell.

        ``metacell``
            The integer index of the metacell each cell belongs to. The metacells are in no
            particular order. Cells with no metacell assignment ("outliers") are given a metacell
            index of ``-1``.

        ``outlier``
            A boolean mask of the cells contained in no metacell.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.pipeline.feature.extract_feature_data` to extract "feature" data
       from the clean data, using the
       ``feature_downsample_min_samples`` (default: {feature_downsample_min_samples}),
       ``feature_downsample_min_cell_quantile`` (default: {feature_downsample_min_cell_quantile}),
       ``feature_downsample_max_cell_quantile`` (default: {feature_downsample_max_cell_quantile}),
       ``feature_min_gene_total`` (default: {feature_min_gene_total}), ``feature_min_gene_top3``
       (default: {feature_min_gene_top3}), ``feature_min_gene_relative_variance (default:
       {feature_min_gene_relative_variance}), ``forbidden_gene_names`` (default:
       {forbidden_gene_names}), ``forbidden_gene_patterns`` (default: {forbidden_gene_patterns})
       and
       ``random_seed`` (default: {random_seed})
       to make this replicable.

    2. Compute the fractions of each variable in each cell, and add the
       ``cells_similarity_value_normalization`` (default: {cells_similarity_value_normalization}) to
       it.

    3. If ``cells_similarity_log_data`` (default: {cells_similarity_log_data}), invoke the
       :py:func:`metacells.utilities.computation.log_data` function to compute the log (base 2) of
       the data.

    4. Invoke :py:func:`metacells.tools.similarity.compute_obs_obs_similarity` to compute the
       similarity between each pair of cells, using the
       ``cells_similarity_method`` (default: {cells_similarity_method}).

    5. Invoke :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph` to compute a
       K-Nearest-Neighbors graph, using the
       ``knn_balanced_ranks_factor`` (default: {knn_balanced_ranks_factor}),
       ``knn_incoming_degree_factor`` (default: {knn_incoming_degree_factor})
       and
       ``knn_outgoing_degree_factor`` (default: {knn_outgoing_degree_factor}).
       If ``knn_k`` (default: {knn_k}) is not specified, then it is
       chosen to be the median number of cells required to reach the target metacell size,
       but at least ``min_knn_k`` (default: {min_knn_k}).

    6. Invoke :py:func:`metacells.tools.candidates.compute_candidate_metacells` to compute
       the candidate metacells, using the
       ``candidates_cell_seeds`` (default: {candidates_cell_seeds}),
       ``min_seed_size_quantile`` (default: {min_seed_size_quantile}),
       ``max_seed_size_quantile`` (default: {max_seed_size_quantile}),
       ``candidates_cooldown_pass`` (default: {candidates_cooldown_pass}),
       ``candidates_cooldown_node`` (default: {candidates_cooldown_node}),
       ``candidates_cooldown_phase`` (default: {candidates_cooldown_phase}),
       ``candidates_min_split_size_factor`` (default: {candidates_min_split_size_factor}),
       ``candidates_max_merge_size_factor`` (default: {candidates_max_merge_size_factor}),
       ``candidates_min_metacell_cells`` (default: {candidates_min_metacell_cells}),
       and
       ``random_seed`` (default: {random_seed})
       to make this replicable. This tries to build metacells of the
       ``target_metacell_size`` (default: {target_metacell_size})
       using the
       ``cell_sizes`` (default: {cell_sizes}).

    7. Unless ``must_complete_cover`` (default: {must_complete_cover}), invoke
       :py:func:`metacells.tools.deviants.find_deviant_cells` to remove deviants from the candidate
       metacells, using the
       ``deviants_min_gene_fold_factor`` (default: {deviants_min_gene_fold_factor}),
       ``deviants_max_gene_fraction`` (default: {deviants_max_gene_fraction})
       and
       ``deviants_max_cell_fraction`` (default: {deviants_max_cell_fraction}).

    8. Unless ``must_complete_cover`` (default: {must_complete_cover}), invoke
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
    fdata = \
        extract_feature_data(adata, what, top_level=False,
                             downsample_min_samples=feature_downsample_min_samples,
                             downsample_min_cell_quantile=feature_downsample_min_cell_quantile,
                             downsample_max_cell_quantile=feature_downsample_max_cell_quantile,
                             min_gene_relative_variance=feature_min_gene_relative_variance,
                             min_gene_total=feature_min_gene_total,
                             min_gene_top3=feature_min_gene_top3,
                             forbidden_gene_names=forbidden_gene_names,
                             forbidden_gene_patterns=forbidden_gene_patterns,
                             random_seed=random_seed)

    if fdata is None:
        raise ValueError('Empty feature data, giving up')

    effective_cell_sizes, max_cell_size, _cell_scale_factors = \
        compute_effective_cell_sizes(adata,
                                     max_cell_size=max_cell_size,
                                     max_cell_size_factor=max_cell_size_factor,
                                     cell_sizes=cell_sizes)
    ut.log_calc('effective_cell_sizes',
                effective_cell_sizes, formatter=ut.sizes_description)

    if max_cell_size is not None:
        if candidates_min_metacell_cells is not None:
            target_metacell_size = \
                max(target_metacell_size,
                    max_cell_size * candidates_min_metacell_cells)

        if dissolve_min_metacell_cells is not None:
            target_metacell_size = \
                max(target_metacell_size,
                    max_cell_size * dissolve_min_metacell_cells)

        if candidates_min_metacell_cells is not None \
                or dissolve_min_metacell_cells is not None:
            ut.log_calc('target_metacell_size', target_metacell_size)

    data = ut.get_vo_proper(fdata, 'downsampled', layout='row_major')
    data = ut.to_numpy_matrix(data, copy=True)

    if cells_similarity_value_normalization > 0:
        data += cells_similarity_value_normalization

    if cells_similarity_log_data:
        data = ut.log_data(data, base=2)

    if knn_k is None:
        if effective_cell_sizes is None:
            median_cell_size = 1.0
        else:
            median_cell_size = float(np.median(effective_cell_sizes))
        knn_k = int(round(target_metacell_size / median_cell_size))
        if min_knn_k is not None:
            knn_k = max(knn_k, min_knn_k)

    if knn_k == 0:
        ut.log_calc('knn_k: 0 (too small, try single metacell)')
        ut.set_o_data(fdata, 'candidate',
                      np.full(fdata.n_obs, 0, dtype='int32'),
                      formatter=lambda _: '* <- 0')
    elif knn_k >= fdata.n_obs:
        ut.log_calc(f'knn_k: {knn_k} (too large, try single metacell)')
        ut.set_o_data(fdata, 'candidate',
                      np.full(fdata.n_obs, 0, dtype='int32'),
                      formatter=lambda _: '* <- 0')

    else:
        ut.log_calc('knn_k', knn_k)

        tl.compute_obs_obs_similarity(fdata, data,
                                      method=cells_similarity_method,
                                      reproducible=(random_seed != 0))

        tl.compute_obs_obs_knn_graph(fdata,
                                     k=knn_k,
                                     balanced_ranks_factor=knn_balanced_ranks_factor,
                                     incoming_degree_factor=knn_incoming_degree_factor,
                                     outgoing_degree_factor=knn_outgoing_degree_factor)

        tl.compute_candidate_metacells(fdata,
                                       target_metacell_size=target_metacell_size,
                                       cell_sizes=effective_cell_sizes,
                                       cell_seeds=candidates_cell_seeds,
                                       min_seed_size_quantile=min_seed_size_quantile,
                                       max_seed_size_quantile=max_seed_size_quantile,
                                       cooldown_pass=candidates_cooldown_pass,
                                       cooldown_node=candidates_cooldown_node,
                                       cooldown_phase=candidates_cooldown_phase,
                                       min_split_size_factor=candidates_min_split_size_factor,
                                       max_merge_size_factor=candidates_max_merge_size_factor,
                                       min_metacell_cells=candidates_min_metacell_cells,
                                       random_seed=random_seed)

        ut.set_oo_data(adata, 'obs_similarity',
                       ut.get_oo_proper(fdata, 'obs_similarity'))

        ut.set_oo_data(adata, 'obs_outgoing_weights',
                       ut.get_oo_proper(fdata, 'obs_outgoing_weights'))

        seed_of_cells = \
            ut.get_o_numpy(fdata, 'seed', formatter=ut.groups_description)

        ut.set_o_data(adata, 'seed', seed_of_cells,
                      formatter=ut.groups_description)

    candidate_of_cells = \
        ut.get_o_numpy(fdata, 'candidate', formatter=ut.groups_description)

    ut.set_o_data(adata, 'candidate', candidate_of_cells,
                  formatter=ut.groups_description)

    if must_complete_cover:
        assert np.min(candidate_of_cells) == 0

        deviant_votes_of_genes = np.zeros(adata.n_vars, dtype='float32')
        deviant_votes_of_cells = np.zeros(adata.n_obs, dtype='float32')
        dissolved_of_cells = np.zeros(adata.n_obs, dtype='bool')

        ut.set_v_data(adata, 'gene_deviant_votes', deviant_votes_of_genes,
                      formatter=ut.mask_description)

        ut.set_o_data(adata, 'cell_deviant_votes', deviant_votes_of_cells,
                      formatter=ut.mask_description)

        ut.set_o_data(adata, 'dissolved', dissolved_of_cells,
                      formatter=ut.mask_description)

        ut.set_o_data(adata, 'metacell', candidate_of_cells,
                      formatter=ut.groups_description)

    else:
        tl.find_deviant_cells(adata,
                              candidates=candidate_of_cells,
                              min_gene_fold_factor=deviants_min_gene_fold_factor,
                              max_gene_fraction=deviants_max_gene_fraction,
                              max_cell_fraction=deviants_max_cell_fraction)

        tl.dissolve_metacells(adata,
                              candidates=candidate_of_cells,
                              target_metacell_size=target_metacell_size,
                              cell_sizes=effective_cell_sizes,
                              min_robust_size_factor=dissolve_min_robust_size_factor,
                              min_convincing_size_factor=dissolve_min_convincing_size_factor,
                              min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                              min_metacell_cells=dissolve_min_metacell_cells)

        metacell_of_cells = \
            ut.get_o_numpy(adata, 'metacell', formatter=ut.groups_description)

        outlier_of_cells = metacell_of_cells < 0
        ut.set_o_data(adata, 'outlier', outlier_of_cells,
                      formatter=ut.mask_description)

    return fdata
