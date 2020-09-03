'''
Direct Metacells
----------------

This directly computes the metacells on the whole data. Like any method that directly looks at the
whole data at once, the amount of CPU and memory needed becomes unreasonable when the data size
grows. At O(10,000) and definitely for anything above O(100,000) you are better off using the
divide-and-conquer method.
'''

import logging
from typing import Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.preprocessing as pp
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    'compute_direct_metacells',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def compute_direct_metacells(
    cdata: AnnData,
    fdata: AnnData,
    of: Optional[str] = None,
    *,
    log_data: bool = True,
    log_normalization: float = 1/7,
    repeated_similarity: bool = True,
    k: Optional[int] = None,
    balanced_ranks_factor: float = 4.0,
    incoming_degree_factor: float = 3.0,
    outgoing_degree_factor: float = 1.0,
    partition_method: 'ut.PartitionMethod' = ut.leiden_bounded_surprise,
    target_metacell_size: int = 160000,
    cell_sizes: Optional[Union[str, ut.Vector]] = '<of>|sum_per_obs',
    min_split_factor: Optional[float] = 2.0,
    max_merge_factor: Optional[float] = 0.25,
    random_seed: int = 0,
    min_gene_fold_factor: float = 3.0,
    max_genes_fraction: float = 0.03,
    max_cells_fraction: float = 0.25,
    min_robust_size_factor: Optional[float] = 0.5,
    min_convincing_size_factor: Optional[float] = 0.25,
    min_convincing_gene_fold_factor: float = 3.0,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Directly compute metacells.

    This is the heart of the metacells methodology.

    **Input**

    The "clean" :py:func:`metacells.utilities.annotation.setup` annotated ``cdata`` and the selected
    "feature" ``fdata``, where the observations are cells and the variables are genes.

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

    1. If ``log_data`` (default: {log_data}), invoke
       :py:func:`metacells.preprocessing.common.get_log_matrix` function to compute the log (base 2)
       of the data, using the ``log_normalization`` (default: {log_normalization}).

    2. Invoke :py:func:`metacells.tools.similarity.compute_obs_obs_similarity` to compute the
       similarity between each pair of cells, using the ``repeated_similarity`` (default:
       {repeated_similarity}) to compensate for the sparseness of the data.

    3. Invoke :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph` to compute a
       K-Nearest-Neighbors graph, using ``balanced_ranks_factor`` (default:
       {balanced_ranks_factor}), ``incoming_degree_factor`` (default: {incoming_degree_factor}) and
       ``outgoing_degree_factor`` (default: {outgoing_degree_factor}). If ``k`` (default: {k}) is
       not specified, then it is chosen to be the mean number of cells required to reach the target
       metacell size.

    4. Invoke :py:func:`metacells.tools.candidate_metacells.compute_candidate_metacells` to compute
       the candidate metacells, using the ``partition_method`` (default:
       {partition_method.__qualname__}), ``min_split_factor`` (default: {min_split_factor}),
       ``max_merge_factor`` (default: {max_merge_factor}) and the ``random_seed`` (default:
       {random_seed}). This tries to build metacells of the ``target_metacell_size`` (default:
       {target_metacell_size}) using the ``cell_sizes`` (default: {cell_sizes}). If the cell sizes
       is a string that contains ``<of>``, it is expanded using the name of the ``of`` data.

    5. Invoke :py:func:`metacells.tools.outlier_cells.find_outlier_cells` to remove outliers from
       the candidate metacells, using ``min_gene_fold_factor`` default: {min_gene_fold_factor}),
       ``max_genes_fraction`` (default: {max_genes_fraction}) and ``max_cells_fraction`` (default:
       {max_cells_fraction}).

    6. Invoke :py:func:`metacells.tools.final_metacells.finalize_metacells` to dissolve small
       unconvincing metacells, using the ``target_metacell_size`` and ``cell_sizes``, the
       ``min_robust_size_factor`` (default: {min_robust_size_factor}),
       ``min_convincing_size_factor`` (default: {min_convincing_size_factor}), and
       ``min_convincing_gene_fold_factor`` (default: {min_convincing_size_factor}).
    '''
    assert id(cdata) != id(fdata)

    level = ut.log_pipeline_step(LOG, fdata, 'compute_direct_metacells')

    with ut.focus_on(ut.get_vo_data, fdata, of, intermediate=intermediate):
        if isinstance(cell_sizes, str):
            cell_sizes = cell_sizes.replace('<of>', ut.get_focus_name(fdata))

        cell_sizes = \
            ut.get_vector_parameter_data(LOG, level, cdata, cell_sizes,
                                         indent='', per='o', name='cell_sizes')

        if log_data:
            LOG.log(level, 'log of: %s base: 2 normalization: %s',
                    ut.get_focus_name(fdata), log_normalization)
            of = pp.get_log_matrix(fdata, base=2,
                                   normalization=log_normalization).name
        else:
            LOG.log(level, 'log: None')
            of = None

        tl.compute_obs_obs_similarity(fdata, of=of,
                                      repeated=repeated_similarity)

        if k is None:
            if cell_sizes is None:
                total_cell_sizes = fdata.n_obs
            else:
                total_cell_sizes = np.sum(cell_sizes)
            k = round(total_cell_sizes / target_metacell_size)

        tl.compute_obs_obs_knn_graph(fdata,
                                     k=k,
                                     balanced_ranks_factor=balanced_ranks_factor,
                                     incoming_degree_factor=incoming_degree_factor,
                                     outgoing_degree_factor=outgoing_degree_factor)

        tl.compute_candidate_metacells(fdata,
                                       target_metacell_size=target_metacell_size,
                                       cell_sizes=cell_sizes,
                                       partition_method=partition_method,
                                       min_split_factor=min_split_factor,
                                       max_merge_factor=max_merge_factor,
                                       random_seed=random_seed)

        candidate_metacell_of_cells = \
            ut.to_dense_vector(ut.get_o_data(fdata, 'candidate_metacell'))

    with ut.intermediate_step(cdata, intermediate=intermediate):
        ut.set_o_data(cdata, 'candidate_metacell',
                      candidate_metacell_of_cells, ut.NEVER_SAFE)

        tl.find_outlier_cells(cdata,
                              min_gene_fold_factor=min_gene_fold_factor,
                              max_genes_fraction=max_genes_fraction,
                              max_cells_fraction=max_cells_fraction)

        tl.finalize_metacells(cdata,
                              target_metacell_size=target_metacell_size,
                              cell_sizes=cell_sizes,
                              min_robust_size_factor=min_robust_size_factor,
                              min_convincing_size_factor=min_convincing_size_factor,
                              min_convincing_gene_fold_factor=min_convincing_gene_fold_factor)

        metacell_of_cells = ut.get_o_data(cdata, 'metacell')

    if inplace:
        ut.set_o_data(cdata, 'metacell', metacell_of_cells)
        return None

    return pd.Series(metacell_of_cells, index=cdata.obs_names)
