'''
Divide and Conquer
------------------
'''

import gc
import logging
import os
import sys
from re import Pattern
from typing import (Any, Callable, Collection, Dict, List, NamedTuple,
                    Optional, Union)

import numpy as np
from anndata import AnnData

import metacells.parameters as pr
import metacells.preprocessing as pp
import metacells.tools as tl
import metacells.utilities as ut

from .direct import compute_direct_metacells

__all__ = [
    'set_max_parallel_piles',
    'get_max_parallel_piles',
    'divide_and_conquer_pipeline',
    'compute_divide_and_conquer_metacells',
]


LOG = logging.getLogger(__name__)


MAX_PARALLEL_PILES = 0


def set_max_parallel_piles(max_cpus: int) -> None:
    '''
    Set the (maximal) number of piles to compute in parallel.

    By default, we use all the available hardware threads. Override this by setting the
    ``METACELLS_MAX_PARALLEL_PILES`` environment variable or by invoking this function from the main
    thread.

    A value of ``-1`` uses half of the available threads to combat hyper-threading, ``0`` (the
    default) uses all the available threads, and otherwise the value is the number of threads to
    use.

    It may be useful to restrict the number of parallel piles to restrict the total amount of memory
    used by the application. While this does not increase the total amount of CPU used by the
    program, it will increase the total elapsed time due to the reduced parallelism.
    '''
    global MAX_PARALLEL_PILES
    MAX_PARALLEL_PILES = max_cpus


if not 'sphinx' in sys.argv[0]:
    set_max_parallel_piles(int(os.environ.get('METACELLS_MAX_PARALLEL_PILES',
                                              '0')))


def get_max_parallel_piles() -> int:
    '''
    Return the maximal number of piles to compute in parallel.
    '''
    global MAX_PARALLEL_PILES
    return MAX_PARALLEL_PILES


class ResultAnnotation(NamedTuple):
    '''
    A per-gene or per-cell annotation used to report results.
    '''

    #: The annotation name.
    name: str

    #: The default per-cell value before we collect results.
    default: Any

    #: The data type to use.
    dtype: str

    #: How to log the value.
    log_value: Optional[Callable[[Any], str]]


GENE_ANNOTATIONS = [
    ResultAnnotation(name='pre_high_fraction_gene',
                     default=0, dtype='int32', log_value=ut.mask_description),
    ResultAnnotation(name='high_fraction_gene',
                     default=0, dtype='int32', log_value=ut.mask_description),
    ResultAnnotation(name='pre_high_relative_variance_gene',
                     default=0, dtype='int32', log_value=ut.mask_description),
    ResultAnnotation(name='high_relative_variance_gene',
                     default=0, dtype='int32', log_value=ut.mask_description),
    ResultAnnotation(name='forbidden_gene',
                     default=False, dtype='bool', log_value=None),
    ResultAnnotation(name='pre_feature_gene',
                     default=0, dtype='int32', log_value=ut.mask_description),
    ResultAnnotation(name='feature_gene',
                     default=0, dtype='int32', log_value=ut.mask_description),
    ResultAnnotation(name='pre_gene_deviant_votes',
                     default=0, dtype='int32', log_value=ut.mask_description),
    ResultAnnotation(name='gene_deviant_votes',
                     default=0, dtype='int32', log_value=ut.mask_description),
]


CELL_ANNOTATIONS = [
    ResultAnnotation(name='pre_cell_directs',
                     default=0, dtype='int32', log_value=None),
    ResultAnnotation(name='cell_directs',
                     default=0, dtype='int32', log_value=None),
    ResultAnnotation(name='pre_pile',
                     default=-1, dtype='int32', log_value=ut.groups_description),
    ResultAnnotation(name='pile',
                     default=0, dtype='int32', log_value=ut.groups_description),
    ResultAnnotation(name='pre_candidate',
                     default=-1, dtype='int32', log_value=ut.groups_description),
    ResultAnnotation(name='candidate',
                     default=-1, dtype='int32', log_value=ut.groups_description),
    ResultAnnotation(name='pre_cell_deviant_votes',
                     default=0, dtype='int32', log_value=ut.mask_description),
    ResultAnnotation(name='cell_deviant_votes',
                     default=0, dtype='int32', log_value=ut.mask_description),
    ResultAnnotation(name='pre_dissolved',
                     default=False, dtype='bool', log_value=ut.mask_description),
    ResultAnnotation(name='dissolved',
                     default=False, dtype='bool', log_value=ut.mask_description),
    ResultAnnotation(name='pre_metacell',
                     default=-1, dtype='int32', log_value=ut.groups_description),
    ResultAnnotation(name='metacell',
                     default=-1, dtype='int32', log_value=ut.groups_description),
    ResultAnnotation(name='outlier',
                     default=True, dtype='bool', log_value=ut.mask_description),
]


class SubsetResults:
    '''
    Results of computing metacells for a subset of the cells.

    This data is the only thing that is transmitted from sub-processes computing metacells for
    piles. Everything else done in such sub-processes is lost when the
    :py:func:`metacells.utilities.parallel.parallel_map` is completed.
    '''

    def __init__(  #
        self,
        adata: AnnData,
        *,
        is_direct: bool,
        pre_target: Optional[str],
        final_target: str,
    ) -> None:
        '''
        Extract the results from ``adata`` which is a subset of the "complete" (clean) data.

        This assumes the data contains a ``complete_cell_index`` per-observation (cell) annotation
        mapping the subset cells back to the complete cell indices.
        '''
        assert pre_target in (None, 'preliminary')
        assert final_target in ('rare', 'preliminary', 'final')
        assert pre_target != final_target
        assert not (is_direct and pre_target is not None)

        #: Whether we are collecting annotations from the results of a
        #: :py:func:`metacells.pipeline.direct.compute_direct_metacells`, that is, whether there are
        #: no preliminary annotations in the data.
        self.is_direct = is_direct

        #: Where to collect the preliminary annotations (either ``None`` or ``preliminary``).
        self.pre_target = pre_target

        #: Where to collect the final annotations (either ``preliminary`` or ``rare``/``final``).
        self.final_target = final_target

        if is_direct:
            #: The number of :py:func:`metacells.pipeline.direct.compute_direct_metacells`
            #: invocations in the preliminary phase.
            self.pre_directs = 0

            #: The number of :py:func:`metacells.pipeline.direct.compute_direct_metacells`
            #: invocations in the final phase.
            self.directs = 1

        else:
            self.pre_directs = ut.get_m_data(adata, 'pre_directs')
            self.directs = ut.get_m_data(adata, 'directs')

        #: The per-gene data.
        #:
        #: This must cover all the genes of the "complete" (clean) data. It must contain a
        #: ``feature_gene`` column, and optionally the ``high_fraction_gene``,
        #: ``high_relative_variance_gene``, ``forbidden_gene`` and ``gene_deviant_votes`` columns.
        self.genes_frame = ut.to_pandas_frame(index=range(adata.n_vars))

        for gene_annotation in GENE_ANNOTATIONS:
            if self.pre_target is None \
                    and gene_annotation.name.startswith('pre_'):
                continue

            self.genes_frame[gene_annotation.name] = \
                ut.get_v_numpy(adata, gene_annotation.name)

        cell_indices = ut.get_o_numpy(adata, 'complete_cell_index')

        #: The per-cell data.
        #:
        #: The index contains the indices of the cells in the "complete" (clean) data. It must
        #: contain a ``metacell`` column, and optionally the ``candidate``, ``cell_deviant_votes``,
        #: ``dissolved`` and ``outlier`` columns.
        self.cells_frame = ut.to_pandas_frame(index=cell_indices)

        for cell_annotation in CELL_ANNOTATIONS:
            if (self.pre_target is None
                    and cell_annotation.name.startswith('pre_')) \
                or (self.is_direct
                    and cell_annotation.name in ('cell_directs', 'pile')):
                continue

            self.cells_frame[cell_annotation.name] = \
                ut.get_o_numpy(adata, cell_annotation.name)

    def collect(  # pylint: disable=too-many-branches,too-many-statements
        self,
        adata: AnnData,
        counts: Dict[str, int],
    ) -> None:
        '''
        Collect the results of the subset into the specified ``adata``.

        Use the ``counts`` to ensure all identifiers (piles, metacells) are unique in the end result.
        '''

        if self.pre_target is not None:
            assert self.pre_target == 'preliminary'
            _modify_value(adata, ut.get_m_data, ut.set_m_data, 'pre_directs',
                          lambda directs: directs + self.pre_directs)

        if self.final_target == 'preliminary':
            directs_target_name = 'pre_directs'
        else:
            directs_target_name = 'directs'
        _modify_value(adata, ut.get_m_data, ut.set_m_data, directs_target_name,
                      lambda directs: directs + self.directs)

        for gene_annotation in GENE_ANNOTATIONS:
            if gene_annotation.name.startswith('pre_'):
                if self.pre_target is None:
                    continue
                assert self.pre_target == 'preliminary'
                target_name = gene_annotation.name

            else:
                if self.final_target == 'preliminary' \
                        and gene_annotation.name != 'forbidden_gene':
                    target_name = 'pre_' + gene_annotation.name
                else:
                    target_name = gene_annotation.name

            if target_name is None:
                continue

            # pylint: disable=cell-var-from-loop
            new_genes_value = \
                ut.to_numpy_vector(self.genes_frame[gene_annotation.name])

            if gene_annotation.dtype == 'bool':
                _modify_value(adata, ut.get_v_numpy, ut.set_v_data, target_name,
                              lambda old_genes_value:
                              old_genes_value | new_genes_value,
                              log_value=gene_annotation.log_value)
            else:
                _modify_value(adata, ut.get_v_numpy, ut.set_v_data, target_name,
                              lambda old_genes_value: old_genes_value
                              + new_genes_value.astype('int32'),
                              log_value=gene_annotation.log_value)
            # pylint: enable=cell-var-from-loop

        cell_indices = self.cells_frame.index.values

        for cell_annotation in CELL_ANNOTATIONS:
            if cell_annotation.name.startswith('pre_'):
                if self.pre_target is None:
                    continue
                assert self.pre_target == 'preliminary'
                target_name = cell_annotation.name

            else:
                if self.final_target == 'preliminary':
                    if cell_annotation.name == 'outlier':
                        continue
                    target_name = 'pre_' + cell_annotation.name
                else:
                    target_name = cell_annotation.name

            if target_name is None:
                continue

            if 'cell_directs' in cell_annotation.name:
                # pylint: disable=cell-var-from-loop
                if self.is_direct:
                    assert cell_annotation.name == 'cell_directs'
                    new_value: Union[int, ut.NumpyVector] = 1
                else:
                    new_value = ut.to_numpy_vector(  #
                        self.cells_frame[cell_annotation.name])

                _modify_value_subset(adata, ut.get_o_numpy, ut.set_o_data,
                                     target_name, cell_indices,
                                     lambda old_cells_value:
                                     old_cells_value + new_value,
                                     log_value=gene_annotation.log_value)
                continue
                # pylint: enable=cell-var-from-loop

            # pylint: disable=cell-var-from-loop
            if self.is_direct and 'pile' in cell_annotation.name:
                new_cells_value = \
                    np.zeros(len(self.cells_frame.index), dtype='int32')
            else:
                new_cells_value = \
                    ut.to_numpy_vector(self.cells_frame[cell_annotation.name])

            count = counts.get(target_name)
            if count is not None:
                new_cells_value[new_cells_value >= 0] += count
                counts[target_name] = max(count, np.max(new_cells_value) + 1)

            _modify_value_subset(adata, ut.get_o_numpy, ut.set_o_data,
                                 target_name, cell_indices,
                                 lambda _: new_cells_value,
                                 log_value=gene_annotation.log_value)
            # pylint: enable=cell-var-from-loop


@ut.timed_call('.initialize_results')
def _initialize_results(
    adata: AnnData,
    *,
    pre_metacells: bool = False,
) -> None:
    ut.log_pipeline_step(LOG, adata, 'initialize_results')

    ut.set_m_data(adata, 'pre_directs', 0)
    ut.set_m_data(adata, 'directs', 0)

    # pylint: disable=cell-var-from-loop
    for gene_annotation in GENE_ANNOTATIONS:
        ut.set_v_data(adata,
                      gene_annotation.name,
                      np.full(adata.n_vars,
                              gene_annotation.default,
                              dtype=gene_annotation.dtype),
                      log_value=lambda _: '* <- %s' % gene_annotation.default)
    # pylint: enable=cell-var-from-loop

    # pylint: disable=cell-var-from-loop
    for cell_annotation in CELL_ANNOTATIONS:
        if not (pre_metacells and cell_annotation.name == 'pre_metacells'):
            continue
        ut.set_o_data(adata,
                      cell_annotation.name,
                      np.full(adata.n_obs,
                              cell_annotation.default,
                              dtype=cell_annotation.dtype),
                      log_value=lambda _: '* <- %s' % cell_annotation.default)
    # pylint: enable=cell-var-from-loop


@ut.timed_call('.direct_results')
def _patch_direct_results(
    adata: AnnData,
) -> None:
    ut.log_pipeline_step(LOG, adata, 'direct_results')

    ut.set_m_data(adata, 'pre_directs', 0)
    ut.set_m_data(adata, 'directs', 1)

    # pylint: disable=cell-var-from-loop
    for gene_annotation in GENE_ANNOTATIONS:
        if not ut.has_data(adata, gene_annotation.name):
            ut.set_v_data(adata,
                          gene_annotation.name,
                          np.full(adata.n_vars,
                                  gene_annotation.default,
                                  dtype=gene_annotation.dtype),
                          log_value=lambda _: '* <- %s' % gene_annotation.default)
            continue

        value = ut.get_v_numpy(adata, gene_annotation.name)
        if str(value.dtype) == gene_annotation.dtype:
            continue

        value = value.astype(gene_annotation.dtype)
        ut.set_v_data(adata,
                      gene_annotation.name,
                      value,
                      log_value=gene_annotation.log_value)
    # pylint: enable=cell-var-from-loop

    # pylint: disable=cell-var-from-loop
    for cell_annotation in CELL_ANNOTATIONS:
        if ut.has_data(adata, cell_annotation.name):
            continue
        if cell_annotation.name == 'final_cell_directs':
            default = 1
        else:
            default = cell_annotation.default
        ut.set_o_data(adata,
                      cell_annotation.name,
                      np.full(adata.n_obs, default,
                              dtype=cell_annotation.dtype),
                      log_value=lambda _: '* <- %s' % default)
    # pylint: enable=cell-var-from-loop


@ut.timed_call()
@ut.expand_doc()
def divide_and_conquer_pipeline(
    adata: AnnData,
    what: str = '__x__',
    *,
    rare_max_gene_cell_fraction: float = pr.rare_max_gene_cell_fraction,
    rare_min_gene_maximum: int = pr.rare_min_gene_maximum,
    rare_similarity_of: Optional[str] = None,
    rare_genes_similarity_method: str = pr.rare_genes_similarity_method,
    rare_genes_cluster_method: str = pr.rare_genes_cluster_method,
    rare_min_genes_of_modules: int = pr.rare_min_genes_of_modules,
    rare_min_cells_of_modules: int = pr.rare_min_cells_of_modules,
    rare_min_modules_size_factor: float = pr.rare_min_modules_size_factor,
    rare_min_module_correlation: float = pr.rare_min_module_correlation,
    rare_min_related_gene_fold_factor: float = pr.rare_min_related_gene_fold_factor,
    rare_min_cell_module_total: int = pr.rare_min_cell_module_total,
    rare_max_cells_of_random_pile: int = pr.rare_max_cells_of_random_pile,
    rare_dissolve_min_robust_size_factor: Optional[float] = pr.rare_dissolve_min_robust_size_factor,
    rare_dissolve_min_convincing_size_factor: Optional[float] = pr.rare_dissolve_min_convincing_size_factor,
    rare_dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    feature_downsample_cell_quantile: float = pr.feature_downsample_cell_quantile,
    feature_min_gene_fraction: float = pr.feature_min_gene_fraction,
    feature_min_gene_relative_variance: float = pr.feature_min_gene_relative_variance,
    forbidden_gene_names: Optional[Collection[str]] = None,
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    cells_similarity_log_data: bool = pr.cells_similarity_log_data,
    cells_similarity_log_normalization: float = pr.cells_similarity_log_normalization,
    cells_similarity_method: str = pr.cells_similarity_method,
    groups_similarity_method: str = pr.groups_similarity_method,
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
    candidates_min_metacell_cells: int = pr.candidates_min_metacell_cells,
    must_complete_cover: bool = False,
    final_max_outliers_levels: Optional[int] = pr.final_max_outliers_levels,
    deviants_min_gene_fold_factor: float = pr.deviants_min_gene_fold_factor,
    deviants_max_gene_fraction: Optional[float] = pr.deviants_max_gene_fraction,
    deviants_max_cell_fraction: Optional[float] = pr.deviants_max_cell_fraction,
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    dissolve_min_metacell_cells: int = pr.dissolve_min_metacell_cells,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    random_seed: int = pr.random_seed,
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

    The presumably "clean" annotated ``adata``.

    All the computations will use the ``of`` data (by default, the focus).

    **Returns**

    Sets the following annotations in ``adata``:

    Unstructured Annotations
        ``rare_gene_modules``
            An array of rare gene modules, where every entry is the array of the names of the genes
            of the module.

        ``pre_directs``, ``directs``
            The number of times we invoked
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` for computing the
            preliminary and final metacells. If we end up directly computing the metacells, the
            preliminary value will be zero, and the final value will be one. This can be used to
            normalize the ``pre_<name>`` and ``<name>`` properties below to fractions
            (probabilities).

    Variable (Gene) Annotations
        ``genes_rare_gene_modules``
            The indices of the rare gene module(s) each gene belongs to, where every non-zero bit
            indicates the gene participates in the matching rare gene module.

        ``rare_genes``
            A boolean mask for the genes in any of the rare gene modules.

        ``pre_high_fraction_gene``, ``high_fraction_gene``
            The number of times the gene was marked as having a high expression level when computing
            the preliminary and final metacells. This is zero for non-"clean" genes.

        ``pre_high_relative_variance_gene``, ``high_relative_variance_gene``
            The number of times the gene was marked as having a high normalized variance relative to
            other genes with a similar expression level when when computing the preliminary and
            final metacells. This is zero for non-"clean" genes.

        ``forbidden_gene``
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name. This is ``False`` for non-"clean" genes.

        ``pre_feature_gene``, ``feature_gene``
            The number of times the gene was used as a feature when computing the preliminary and
            final metacells. If we end up directly computing the metacells, the preliminary value
            will be all-zero, and the final value will be one for feature genes, zero otherwise.

        ``pre_gene_deviant_votes``, ``gene_deviant_votes``
            The total number of cells each gene marked as deviant (if zero, the gene did not mark
            any cell as deviant) when computing the preliminary and final metacells. This will be
            zero for non-"feature" genes.

    Observations (Cell) Annotations
        ``cells_rare_gene_module``
            The index of the rare gene module each cell expresses the most, or ``-1`` in the common
            case it does not express any rare genes module.

        ``rare_cells``
            A boolean mask for the (few) cells that express a rare gene module.

        ``pre_cell_directs``, ``cell_directs``
            The number of times we invoked
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` to try and group this cell
            when computing the preliminary and final metacells. If we end up directly computing the
            metacells, the preliminary value will be zero. Otherwise this will be at least one, with
            higher values for cells which we tried to group again in the outlier pile(s).

        ``pre_pile``, ``pile``
            The index of the pile the cell was in in the last invocation of
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` when computing the
            preliminary and final metacells. This is ``-1`` for non-"clean" cells. The preliminary
            value is likewise ``-1`` if we end up directly computing the metacells.

        ``pre_candidate``, ``candidate``
            The index of the candidate metacell each cell was assigned to to in the last grouping
            attempt when computing the preliminary and final metacells. This is ``-1`` for
            non-"clean" cells.

        ``pre_cell_deviant_votes``, ``cell_deviant_votes``
            The number of genes that were the reason the cell was marked as deviant in the last
            grouping attempt when computing the preliminary and final metacells (if zero, the cell
            is not deviant). This is zero for non-"clean" cells.

        ``pre_dissolved``, ``dissolved``
            A boolean mask of the cells contained in a dissolved metacell in the last grouping
            attempt when computing the preliminary and final metacells. This is ``False`` for
            non-"clean" cells.

        ``pre_metacell``, ``metacell``
            The integer index of the preliminary and final metacell each cell belongs to. The
            metacells are in no particular order. This is ``-1`` for outlier cells and ``-2`` for
            non-"clean" cells.

        ``outlier``
            A boolean mask of the cells not assigned to any final metacell. We assign every
            ("clean") cell to a preliminary metacell so there is no point in providing a
            ``pre_outlier`` mask (it would be identical to the ``clean_cell`` mask).

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.rare.find_rare_gene_modules` to isolate cells expressing
       rare gene modules, using the
       ``forbidden_gene_names``, ``forbidden_gene_patterns``,
       ``rare_max_gene_cell_fraction`` (default: {rare_max_gene_cell_fraction}),
       ``rare_min_gene_maximum`` (default: {rare_min_gene_maximum}),
       ``rare_similarity_of`` (default: {rare_similarity_of}),
       ``rare_genes_similarity_method`` (default: {rare_genes_similarity_method}),
       ``rare_genes_cluster_method`` (default: {rare_genes_cluster_method}),
       ``rare_min_genes_of_modules`` (default: {rare_min_genes_of_modules}),
       ``rare_min_cells_of_modules`` (default: {rare_min_cells_of_modules}),
       ``rare_min_modules_size_factor`` (default: {rare_min_modules_size_factor}),
       ``rare_min_module_correlation`` (default: {rare_min_module_correlation}),
       ``rare_min_related_gene_fold_factor`` (default: {rare_min_related_gene_fold_factor})
       ``rare_max_cells_of_random_pile`` (default: {rare_max_cells_of_random_pile})
       and
       ``rare_min_cell_module_total`` (default: {rare_min_cell_module_total}).

    2. For each detected rare gene module, collect all cells that express the module, and invoke
       :py:func:`metacells.pipeline.direct.compute_direct_metacells` to compute metacells for them,
       using the rest of the parameters. Here we allow using a lower
       ``rare_dissolve_min_robust_size_factor`` (default: {rare_dissolve_min_robust_size_factor}),
       ``rare_dissolve_min_convincing_size_factor`` (default:
       {rare_dissolve_min_convincing_size_factor}) and ``rare_dissolve_min_convincing_size_factor``
       (default: {rare_dissolve_min_convincing_size_factor}) because even "weak" metacells detected
       here are known be relatively "convincing" based on them being different from the overall
       population.

       .. todo::

            Technically we should invoke :py:func:`compute_divide_and_conquer_metacells` for each
            group of cells that express a rare gene module. However, even for very large data sets
            (O(1,000,000) cells) the number of such cells is expected to be small (O(1,000) cells)
            so directly computing metacells for them seems reasonable, and it was easier to (ab)use
            the "compute metacells in parallel for piles" here, rather than create a new mechanism.

    4. Collect all cells that either did not express any rare gene module, or did but were
       considered an outlier in the previous step, and invoke
       :py:func:`compute_divide_and_conquer_metacells` to compute metacells for them,
       using the rest of the parameters.

    4. Combine the results from the previous two steps.
    '''
    ut.log_pipeline_step(LOG, adata, 'compute_rare_candidates')

    counts = dict(pre_pile=0, pile=0,
                  pre_candidate=0, candidate=0,
                  pre_metacell=0, metacell=0)

    did_apply_subset = False
    normal_cells_mask = np.full(adata.n_obs, True, dtype='bool')

    with ut.timed_step('.rare'):
        tl.find_rare_gene_modules(adata, what,
                                  forbidden_gene_names=forbidden_gene_names,
                                  forbidden_gene_patterns=forbidden_gene_patterns,
                                  max_gene_cell_fraction=rare_max_gene_cell_fraction,
                                  min_gene_maximum=rare_min_gene_maximum,
                                  similarity_of=rare_similarity_of,
                                  genes_similarity_method=rare_genes_similarity_method,
                                  genes_cluster_method=rare_genes_cluster_method,
                                  min_genes_of_modules=rare_min_genes_of_modules,
                                  min_cells_of_modules=rare_min_cells_of_modules,
                                  target_metacell_size=target_metacell_size,
                                  min_modules_size_factor=rare_min_modules_size_factor,
                                  min_module_correlation=rare_min_module_correlation,
                                  min_related_gene_fold_factor=rare_min_related_gene_fold_factor,
                                  target_pile_size=target_pile_size,
                                  max_cells_of_random_pile=rare_max_cells_of_random_pile,
                                  min_cell_module_total=rare_min_cell_module_total)

        rare_module_of_cells = ut.get_o_numpy(adata, 'cells_rare_gene_module')
        rare_modules_count = np.max(rare_module_of_cells) + 1
        if rare_modules_count > 0:
            subset_results = \
                _run_parallel_piles(adata, what,
                                    phase='rare',
                                    piles_count=rare_modules_count,
                                    pile_of_cells=rare_module_of_cells,
                                    feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                    feature_min_gene_fraction=feature_min_gene_fraction,
                                    feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                    forbidden_gene_names=forbidden_gene_names,
                                    forbidden_gene_patterns=forbidden_gene_patterns,
                                    cells_similarity_log_data=cells_similarity_log_data,
                                    cells_similarity_log_normalization=cells_similarity_log_normalization,
                                    cells_similarity_method=cells_similarity_method,
                                    target_metacell_size=target_metacell_size,
                                    knn_k=knn_k,
                                    knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                                    knn_incoming_degree_factor=knn_incoming_degree_factor,
                                    knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                                    candidates_partition_method=candidates_partition_method,
                                    candidates_min_split_size_factor=candidates_min_split_size_factor,
                                    candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                                    candidates_min_metacell_cells=candidates_min_metacell_cells,
                                    must_complete_cover=False,
                                    deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                    deviants_max_gene_fraction=deviants_max_gene_fraction,
                                    deviants_max_cell_fraction=deviants_max_cell_fraction,
                                    dissolve_min_robust_size_factor=rare_dissolve_min_robust_size_factor,
                                    dissolve_min_convincing_size_factor=rare_dissolve_min_convincing_size_factor,
                                    dissolve_min_convincing_gene_fold_factor=rare_dissolve_min_convincing_gene_fold_factor,
                                    dissolve_min_metacell_cells=dissolve_min_metacell_cells,
                                    cell_sizes=cell_sizes,
                                    random_seed=random_seed)

            with ut.timed_step('.collect_piles'):
                for rare_index, rare_results in enumerate(subset_results):
                    mask = rare_results.cells_frame['metacell'] >= 0
                    if not np.any(mask):
                        LOG.debug('skip empty results for rare gene module %s/%s',
                                  rare_index, rare_modules_count)
                        continue

                    if not did_apply_subset:
                        did_apply_subset = True
                        _initialize_results(adata)

                    LOG.debug('collect results for rare gene module %s/%s',
                              rare_index, rare_modules_count)

                    cell_indices = rare_results.cells_frame.index.values[mask]
                    normal_cells_mask[cell_indices] = False

                    rare_results.collect(adata, counts)

    if did_apply_subset:
        cdata = ut.slice(adata, obs=normal_cells_mask, name='.common', tmp=True,
                         track_obs='complete_cell_index')
    else:
        cdata = adata

    compute_divide_and_conquer_metacells(cdata, what,
                                         feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                         feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                         feature_min_gene_fraction=feature_min_gene_fraction,
                                         forbidden_gene_names=forbidden_gene_names,
                                         forbidden_gene_patterns=forbidden_gene_patterns,
                                         cells_similarity_log_data=cells_similarity_log_data,
                                         cells_similarity_log_normalization=cells_similarity_log_normalization,
                                         cells_similarity_method=cells_similarity_method,
                                         groups_similarity_method=groups_similarity_method,
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
                                         candidates_min_metacell_cells=candidates_min_metacell_cells,
                                         must_complete_cover=must_complete_cover,
                                         final_max_outliers_levels=final_max_outliers_levels,
                                         deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                         deviants_max_gene_fraction=deviants_max_gene_fraction,
                                         deviants_max_cell_fraction=deviants_max_cell_fraction,
                                         dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                         dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                         dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                         dissolve_min_metacell_cells=dissolve_min_metacell_cells,
                                         cell_sizes=cell_sizes,
                                         random_seed=random_seed)

    if did_apply_subset:
        LOG.debug('collect results for common cells')
        common_results = SubsetResults(cdata,
                                       is_direct=False,
                                       pre_target='preliminary',
                                       final_target='final')
        common_results.collect(adata, counts)


def compute_divide_and_conquer_metacells(
    adata: AnnData,
    what: str = '__x__',
    *,
    feature_downsample_cell_quantile: float = pr.feature_downsample_cell_quantile,
    feature_min_gene_fraction: float = pr.feature_min_gene_fraction,
    feature_min_gene_relative_variance: float = pr.feature_min_gene_relative_variance,
    forbidden_gene_names: Optional[Collection[str]] = None,
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    cells_similarity_log_data: bool = pr.cells_similarity_log_data,
    cells_similarity_log_normalization: float = pr.cells_similarity_log_normalization,
    cells_similarity_method: str = pr.cells_similarity_method,
    groups_similarity_method: str = pr.groups_similarity_method,
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
    candidates_min_metacell_cells: int = pr.min_metacell_cells,
    must_complete_cover: bool = False,
    final_max_outliers_levels: Optional[int] = pr.final_max_outliers_levels,
    deviants_min_gene_fold_factor: float = pr.deviants_min_gene_fold_factor,
    deviants_max_gene_fraction: Optional[float] = pr.deviants_max_gene_fraction,
    deviants_max_cell_fraction: Optional[float] = pr.deviants_max_cell_fraction,
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    dissolve_min_metacell_cells: int = pr.dissolve_min_metacell_cells,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    random_seed: int = pr.random_seed,
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

    The presumably "clean" annotated ``adata``.

    All the computations will use the ``of`` data (by default, the focus).

    **Returns**

    Sets the following annotations in ``adata``:

    Structured Annotations
        ``pre_directs``, ``directs``
            The number of times we invoked
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` for computing the
            preliminary and final metacells. If we end up directly computing the metacells, the
            preliminary value will be zero, and the final value will be one. This can be used to
            normalize the ``pre_<name>`` and ``<name>`` properties below to fractions
            (probabilities).

    Variable (Gene) Annotations
        ``pre_high_fraction_gene``, ``high_fraction_gene``
            The number of times the gene was marked as having a high expression level when computing
            the preliminary and final metacells. This is zero for non-"clean" genes.

        ``pre_high_relative_variance_gene``, ``high_relative_variance_gene``
            The number of times the gene was marked as having a high normalized variance relative to
            other genes with a similar expression level when when computing the preliminary and
            final metacells. This is zero for non-"clean" genes.

        ``forbidden_gene``
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name. This is ``False`` for non-"clean" genes.

        ``pre_feature_gene``, ``feature_gene``
            The number of times the gene was used as a feature when computing the preliminary and
            final metacells. If we end up directly computing the metacells, the preliminary value
            will be all-zero, and the final value will be one for feature genes, zero otherwise.

        ``pre_gene_deviant_votes``, ``gene_deviant_votes``
            The total number of cells each gene marked as deviant (if zero, the gene did not mark
            any cell as deviant) when computing the preliminary and final metacells. This will be
            zero for non-"feature" genes.

    Observations (Cell) Annotations
        ``pre_cell_directs``, ``cell_directs``
            The number of times we invoked
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` to try and group this cell
            when computing the preliminary and final metacells. If we end up directly computing the
            metacells, the preliminary value will be zero. Otherwise this will be at least one, with
            higher values for cells which we tried to group again in the outlier pile(s).

        ``pre_pile``, ``pile``
            The index of the pile the cell was in in the last invocation of
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` when computing the
            preliminary and final metacells. This is ``-1`` for non-"clean" cells. The preliminary
            value is likewise ``-1`` if we end up directly computing the metacells.

        ``pre_candidate``, ``candidate``
            The index of the candidate metacell each cell was assigned to to in the last grouping
            attempt when computing the preliminary and final metacells. This is ``-1`` for
            non-"clean" cells.

        ``pre_cell_deviant_votes``, ``cell_deviant_votes``
            The number of genes that were the reason the cell was marked as deviant in the last
            grouping attempt when computing the preliminary and final metacells (if zero, the cell
            is not deviant). This is zero for non-"clean" cells.

        ``pre_dissolved``, ``dissolved``
            A boolean mask of the cells contained in a dissolved metacell in the last grouping
            attempt when computing the preliminary and final metacells. This is ``False`` for
            non-"clean" cells.

        ``pre_metacell``, ``metacell``
            The integer index of the preliminary and final metacell each cell belongs to. The
            metacells are in no particular order. This is ``-1`` for outlier cells and ``-2`` for
            non-"clean" cells.

        ``outlier``
            A boolean mask of the cells not assigned to any final metacell. We assign every
            ("clean") cell to a preliminary metacell so there is no point in providing a
            ``pre_outlier`` mask (it would be identical to the ``clean_cell`` mask).

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

    5. Compute the final metacells using the preliminary metacell piles, using using
       ``final_max_outliers_levels`` to prevent forcing outlier cells to be placed in low-quality
       metacells.
    '''
    ut.log_pipeline_step(LOG, adata, 'compute_divide_and_conquer_metacells')

    LOG.debug('  target_pile_size: %s', target_pile_size)
    LOG.debug('  random_seed: %s', random_seed)
    random_pile_of_cells = \
        ut.random_piles(adata.n_obs,
                        target_pile_size=target_pile_size,
                        random_seed=random_seed)
    piles_count = np.max(random_pile_of_cells) + 1

    if piles_count < 2:
        with ut.timed_step('.direct'):
            compute_direct_metacells(adata, what,
                                     feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                     feature_min_gene_fraction=feature_min_gene_fraction,
                                     feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                     forbidden_gene_names=forbidden_gene_names,
                                     forbidden_gene_patterns=forbidden_gene_patterns,
                                     cells_similarity_log_data=cells_similarity_log_data,
                                     cells_similarity_log_normalization=cells_similarity_log_normalization,
                                     cells_similarity_method=cells_similarity_method,
                                     target_metacell_size=target_metacell_size,
                                     knn_k=knn_k,
                                     knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                                     knn_incoming_degree_factor=knn_incoming_degree_factor,
                                     knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                                     candidates_partition_method=candidates_partition_method,
                                     candidates_min_split_size_factor=candidates_min_split_size_factor,
                                     candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                                     candidates_min_metacell_cells=candidates_min_metacell_cells,
                                     must_complete_cover=must_complete_cover,
                                     deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                     deviants_max_gene_fraction=deviants_max_gene_fraction,
                                     deviants_max_cell_fraction=deviants_max_cell_fraction,
                                     dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                     dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                     dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                     dissolve_min_metacell_cells=dissolve_min_metacell_cells,
                                     cell_sizes=cell_sizes,
                                     random_seed=random_seed)
            _patch_direct_results(adata)
        return

    _initialize_results(adata, pre_metacells=True)

    with ut.timed_step('.preliminary_metacells'):
        _compute_piled_metacells(adata, what,
                                 phase='preliminary',
                                 pile_of_cells=random_pile_of_cells,
                                 feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                 feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                 feature_min_gene_fraction=feature_min_gene_fraction,
                                 forbidden_gene_names=forbidden_gene_names,
                                 forbidden_gene_patterns=forbidden_gene_patterns,
                                 cells_similarity_log_data=cells_similarity_log_data,
                                 cells_similarity_log_normalization=cells_similarity_log_normalization,
                                 cells_similarity_method=cells_similarity_method,
                                 groups_similarity_method=groups_similarity_method,
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
                                 candidates_min_metacell_cells=candidates_min_metacell_cells,
                                 must_complete_cover=True,
                                 final_max_outliers_levels=None,
                                 deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                 deviants_max_gene_fraction=deviants_max_gene_fraction,
                                 deviants_max_cell_fraction=deviants_max_cell_fraction,
                                 dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                 dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                 dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                 dissolve_min_metacell_cells=dissolve_min_metacell_cells,
                                 cell_sizes=cell_sizes,
                                 random_seed=random_seed)

    with ut.timed_step('.metacell_piles'):
        ut.log_operation(LOG, adata, 'metacell_piles', what)

        mdata = pp.group_obs_data(adata, what, groups='pre_metacell',
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
                                             cells_similarity_method=groups_similarity_method,
                                             groups_similarity_method=groups_similarity_method,
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
                                             candidates_min_metacell_cells=candidates_min_metacell_cells,
                                             must_complete_cover=True,
                                             final_max_outliers_levels=None,
                                             deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                             deviants_max_gene_fraction=deviants_max_gene_fraction,
                                             deviants_max_cell_fraction=deviants_max_cell_fraction,
                                             dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                             dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                             dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                             dissolve_min_metacell_cells=dissolve_min_metacell_cells,
                                             cell_sizes='grouped',
                                             random_seed=random_seed)
        preliminary_metacell_of_cells = ut.get_o_numpy(adata, 'pre_metacell')

        pile_of_preliminary_metacells = ut.get_o_numpy(mdata, 'metacell')

        preliminary_pile_of_cells = \
            ut.group_piles(preliminary_metacell_of_cells,
                           pile_of_preliminary_metacells)

    with ut.timed_step('.final_metacells'):
        _compute_piled_metacells(adata, what,
                                 phase='final',
                                 pile_of_cells=preliminary_pile_of_cells,
                                 feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                 feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                 feature_min_gene_fraction=feature_min_gene_fraction,
                                 forbidden_gene_names=forbidden_gene_names,
                                 forbidden_gene_patterns=forbidden_gene_patterns,
                                 cells_similarity_log_data=cells_similarity_log_data,
                                 cells_similarity_log_normalization=cells_similarity_log_normalization,
                                 cells_similarity_method=cells_similarity_method,
                                 groups_similarity_method=groups_similarity_method,
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
                                 candidates_min_metacell_cells=candidates_min_metacell_cells,
                                 must_complete_cover=must_complete_cover,
                                 final_max_outliers_levels=final_max_outliers_levels,
                                 deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                 deviants_max_gene_fraction=deviants_max_gene_fraction,
                                 deviants_max_cell_fraction=deviants_max_cell_fraction,
                                 dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                 dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                 dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                 dissolve_min_metacell_cells=dissolve_min_metacell_cells,
                                 cell_sizes=cell_sizes,
                                 random_seed=random_seed)


def _compute_piled_metacells(
    adata: AnnData,
    what: str,
    *,
    phase: str,
    pile_of_cells: ut.NumpyVector,
    feature_downsample_cell_quantile: float,
    feature_min_gene_fraction: float,
    feature_min_gene_relative_variance: float,
    forbidden_gene_names: Optional[Collection[str]],
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]],
    cells_similarity_log_data: bool,
    cells_similarity_log_normalization: float,
    cells_similarity_method: str,
    groups_similarity_method: str,
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
    candidates_min_metacell_cells: int = pr.min_metacell_cells,
    must_complete_cover: bool,
    final_max_outliers_levels: Optional[int],
    deviants_min_gene_fold_factor: float,
    deviants_max_gene_fraction: Optional[float],
    deviants_max_cell_fraction: Optional[float],
    dissolve_min_robust_size_factor: Optional[float],
    dissolve_min_convincing_size_factor: Optional[float],
    dissolve_min_convincing_gene_fold_factor: float,
    dissolve_min_metacell_cells: int,
    cell_sizes: Optional[Union[str, ut.Vector]],
    random_seed: int,
) -> None:
    ut.log_operation(LOG, adata, phase + '_metacells', what)
    piles_count = np.max(pile_of_cells) + 1
    assert piles_count > 0
    LOG.debug('  piles_count: %s', piles_count)

    subset_results = \
        _run_parallel_piles(adata, what,
                            phase=phase,
                            piles_count=piles_count,
                            pile_of_cells=pile_of_cells,
                            feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                            feature_min_gene_fraction=feature_min_gene_fraction,
                            feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                            forbidden_gene_names=forbidden_gene_names,
                            forbidden_gene_patterns=forbidden_gene_patterns,
                            cells_similarity_log_data=cells_similarity_log_data,
                            cells_similarity_log_normalization=cells_similarity_log_normalization,
                            cells_similarity_method=cells_similarity_method,
                            target_metacell_size=target_metacell_size,
                            knn_k=knn_k,
                            knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                            knn_incoming_degree_factor=knn_incoming_degree_factor,
                            knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                            candidates_partition_method=candidates_partition_method,
                            candidates_min_split_size_factor=candidates_min_split_size_factor,
                            candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                            candidates_min_metacell_cells=candidates_min_metacell_cells,
                            must_complete_cover=must_complete_cover and piles_count == 1,
                            deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                            deviants_max_gene_fraction=deviants_max_gene_fraction,
                            deviants_max_cell_fraction=deviants_max_cell_fraction,
                            dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                            dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                            dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                            dissolve_min_metacell_cells=dissolve_min_metacell_cells,
                            cell_sizes=cell_sizes,
                            random_seed=random_seed)

    assert len(subset_results) == piles_count

    counts = dict(pre_pile=0, pile=0,
                  pre_candidate=0, candidate=0,
                  pre_metacell=0, metacell=0)

    with ut.timed_step('.collect_piles'):
        for pile_index, pile_results in enumerate(subset_results):
            LOG.debug('collect results of pile %s/%s', pile_index, piles_count)
            pile_results.collect(adata, counts)

    assert phase in ('preliminary', 'final')
    if phase == 'preliminary':
        metacell_annotation = 'pre_metacell'
    else:
        metacell_annotation = 'metacell'

    metacell_of_cells = ut.get_o_numpy(adata, metacell_annotation)

    outlier_of_cells = metacell_of_cells < 0
    if piles_count == 1 \
        or not np.any(outlier_of_cells) \
        or (not must_complete_cover
            and final_max_outliers_levels is not None
            and final_max_outliers_levels <= 0):
        return

    with ut.timed_step('.outliers'):
        if final_max_outliers_levels is not None:
            final_max_outliers_levels = final_max_outliers_levels - 1
        name = '.%s.outliers' % phase
        odata = ut.slice(adata, obs=outlier_of_cells, name=name, tmp=True,
                         track_obs='complete_cell_index')

        compute_divide_and_conquer_metacells(odata, what,
                                             feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                             feature_min_gene_fraction=feature_min_gene_fraction,
                                             feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                             forbidden_gene_names=forbidden_gene_names,
                                             forbidden_gene_patterns=forbidden_gene_patterns,
                                             cells_similarity_log_data=cells_similarity_log_data,
                                             cells_similarity_log_normalization=cells_similarity_log_normalization,
                                             cells_similarity_method=cells_similarity_method,
                                             groups_similarity_method=groups_similarity_method,
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
                                             candidates_min_metacell_cells=candidates_min_metacell_cells,
                                             must_complete_cover=must_complete_cover,
                                             final_max_outliers_levels=final_max_outliers_levels,
                                             deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                             deviants_max_gene_fraction=deviants_max_gene_fraction,
                                             deviants_max_cell_fraction=deviants_max_cell_fraction,
                                             dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                             dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                             dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                             dissolve_min_metacell_cells=dissolve_min_metacell_cells,
                                             cell_sizes=cell_sizes,
                                             random_seed=random_seed)

        LOG.debug('collect_outliers_results')
        outliers_results = SubsetResults(odata,
                                         is_direct=False,
                                         pre_target=None,
                                         final_target=phase)
        outliers_results.collect(adata, counts)


def _run_parallel_piles(
    adata: AnnData,
    what: str,
    *,
    phase: str,
    piles_count: int,
    pile_of_cells: ut.NumpyVector,
    feature_downsample_cell_quantile: float,
    feature_min_gene_fraction: float,
    feature_min_gene_relative_variance: float,
    forbidden_gene_names: Optional[Collection[str]],
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]],
    cells_similarity_log_data: bool,
    cells_similarity_log_normalization: float,
    cells_similarity_method: str,
    target_metacell_size: int,
    knn_k: Optional[int],
    knn_balanced_ranks_factor: float,
    knn_incoming_degree_factor: float,
    knn_outgoing_degree_factor: float,
    candidates_partition_method: 'ut.PartitionMethod',
    candidates_min_split_size_factor: Optional[float],
    candidates_max_merge_size_factor: Optional[float],
    candidates_min_metacell_cells: int,
    must_complete_cover: bool,
    deviants_min_gene_fold_factor: float,
    deviants_max_gene_fraction: Optional[float],
    deviants_max_cell_fraction: Optional[float],
    dissolve_min_robust_size_factor: Optional[float],
    dissolve_min_convincing_size_factor: Optional[float],
    dissolve_min_convincing_gene_fold_factor: float,
    dissolve_min_metacell_cells: int,
    cell_sizes: Optional[Union[str, ut.Vector]],
    random_seed: int,
) -> List[SubsetResults]:
    def _compute_pile_metacells(pile_index: int) -> SubsetResults:
        pile_cells_mask = pile_of_cells == pile_index
        assert np.any(pile_cells_mask)
        name = '.%s.pile-%s/%s' % (phase, pile_index, piles_count)
        pdata = ut.slice(adata, obs=pile_cells_mask, name=name, tmp=True,
                         track_obs='complete_cell_index')
        compute_direct_metacells(pdata, what,
                                 feature_downsample_cell_quantile=feature_downsample_cell_quantile,
                                 feature_min_gene_fraction=feature_min_gene_fraction,
                                 feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                                 forbidden_gene_names=forbidden_gene_names,
                                 forbidden_gene_patterns=forbidden_gene_patterns,
                                 cells_similarity_log_data=cells_similarity_log_data,
                                 cells_similarity_log_normalization=cells_similarity_log_normalization,
                                 cells_similarity_method=cells_similarity_method,
                                 target_metacell_size=target_metacell_size,
                                 knn_k=knn_k,
                                 knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                                 knn_incoming_degree_factor=knn_incoming_degree_factor,
                                 knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                                 candidates_partition_method=candidates_partition_method,
                                 candidates_min_split_size_factor=candidates_min_split_size_factor,
                                 candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                                 candidates_min_metacell_cells=candidates_min_metacell_cells,
                                 must_complete_cover=must_complete_cover,
                                 deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                                 deviants_max_gene_fraction=deviants_max_gene_fraction,
                                 deviants_max_cell_fraction=deviants_max_cell_fraction,
                                 dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                                 dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                                 dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                                 dissolve_min_metacell_cells=dissolve_min_metacell_cells,
                                 cell_sizes=cell_sizes,
                                 random_seed=random_seed)
        return SubsetResults(pdata, is_direct=True,
                             pre_target=None, final_target=phase)

    @ut.timed_call('compute_pile_metacells')
    def _return_pile_results(pile_index: int) -> SubsetResults:
        results = _compute_pile_metacells(pile_index)
        gc.collect()
        return results

    with ut.timed_step('.piles'):
        gc.collect()
        return list(ut.parallel_map(_return_pile_results, piles_count,
                                    max_cpus=get_max_parallel_piles()))


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
    getter: Callable[[AnnData, str], ut.NumpyVector],
    setter: 'Setter',
    name: str,
    indices: ut.NumpyVector,
    modifier: Callable[[ut.NumpyVector], ut.NumpyVector],
    log_value: LogValue = None,
) -> None:
    old_value = getter(adata, name)
    ut.unfreeze(old_value)
    old_value[indices] = modifier(old_value[indices])
    ut.freeze(old_value)
    setter(adata, name, old_value, log_value=log_value)
