'''
Direct Pipeline
---------------
'''

import logging
from re import Pattern
from typing import Collection, Optional, Union

from anndata import AnnData

import metacells.utilities as ut

from .clean_data import extract_clean_data
from .direct_metacells import compute_direct_metacells
from .feature_data import extract_feature_data
from .result_metacells import collect_result_metacells

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
    # Clean data:
    min_total_of_cells: Optional[int] = 800,
    max_total_of_cells: Optional[int] = None,
    min_total_of_genes: int = 1,
    excluded_gene_names: Optional[Collection[str]] = None,
    excluded_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    # Feature data:
    downsample_cell_quantile: float = 0.05,
    min_relative_variance_of_genes: float = 0.1,
    min_fraction_of_genes: float = 1e-5,
    forbidden_gene_names: Optional[Collection[str]] = None,
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    # Direct metacells:
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
    min_gene_fold_factor: float = 3.0,
    max_genes_fraction: float = 0.03,
    max_cells_fraction: float = 0.25,
    min_robust_size_factor: Optional[float] = 0.5,
    min_convincing_size_factor: Optional[float] = 0.25,
    min_convincing_gene_fold_factor: float = 3.0,
    # Results:
    name: str = 'METACELLS',
    tmp: bool = False,
    # General:
    random_seed: int = 0,
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
       data from the full input data, using the ``min_total_of_cells`` (default:
       {min_total_of_cells}), ``max_total_of_cells`` (default: {max_total_of_cells}),
       ``excluded_gene_names`` (default: {excluded_gene_names}) and ``excluded_gene_patterns``
       (default: {excluded_gene_patterns}).

    2. Invoke :py:func:`metacells.pipeline.feature_data.extract_feature_data` to extract "feature"
       data from the clean data, using the ``downsample_cell_quantile`` (default:
       {downsample_cell_quantile}), ``min_relative_variance_of_genes`` (default:
       {min_relative_variance_of_genes}), ``min_fraction_of_genes`` (default:
       {min_fraction_of_genes}), ``forbidden_gene_names`` (default: {forbidden_gene_names}),
       ``forbidden_gene_patterns`` (default: {forbidden_gene_patterns}), and the ``random_seed``
       (default: {random_seed}) to make this replicable.

    3. Invoke :py:func:`metacells.pipeline.direct_metacells.compute_direct_metacells` to directly
       compute metacells using the clean and feature data, using ``log_data`` (default: {log_data}),
       ``log_normalization`` (default: {log_normalization}), ``repeated_similarity`` (default:
       {repeated_similarity}), ``k`` (default: {k}), ``balanced_ranks_factor`` (default:
       {balanced_ranks_factor}), ``incoming_degree_factor`` (default: {incoming_degree_factor}),
       ``outgoing_degree_factor`` (default: {outgoing_degree_factor}), ``partition_method``
       (default: {partition_method.__qualname__}), ``target_metacell_size`` (default:
       {target_metacell_size}), ``cell_sizes`` (default: {cell_sizes}), ``min_split_factor``
       (default: {min_split_factor}), ``max_merge_factor`` (default: {max_merge_factor}),
       ``min_gene_fold_factor`` (default: {min_gene_fold_factor}), ``max_genes_fraction`` (default:
       {max_genes_fraction}), ``max_cells_fraction`` (default: {max_cells_fraction}),
       ``min_robust_size_factor`` (default: {min_robust_size_factor}),
       ``min_convincing_size_factor`` (default: {min_convincing_size_factor}),
       ``min_convincing_gene_fold_factor`` (default: {min_convincing_gene_fold_factor}), and the
       ``random_seed`` (default: {random_seed}) to make this replicable.

    4. Invoke :py:func:`metacells.pipeline.result_metacells.collect_result_metacells` to collect the
       result metacells, using the ``name`` (default: {name}) and ``tmp`` (default: {tmp}).
    '''
    cdata = \
        extract_clean_data(adata, of,
                           min_total_of_cells=min_total_of_cells,
                           max_total_of_cells=max_total_of_cells,
                           min_total_of_genes=min_total_of_genes,
                           excluded_gene_names=excluded_gene_names,
                           excluded_gene_patterns=excluded_gene_patterns,
                           intermediate=intermediate)

    if cdata is None:
        LOG.warning('! Empty clean data, giving up')
        return None

    fdata = \
        extract_feature_data(cdata, of,
                             downsample_cell_quantile=downsample_cell_quantile,
                             random_seed=random_seed,
                             min_relative_variance_of_genes=min_relative_variance_of_genes,
                             min_fraction_of_genes=min_fraction_of_genes,
                             forbidden_gene_names=forbidden_gene_names,
                             forbidden_gene_patterns=forbidden_gene_patterns)

    if fdata is None:
        LOG.warning('! Empty feature data, giving up')
        return None

    compute_direct_metacells(cdata, fdata, of,
                             log_data=log_data,
                             log_normalization=log_normalization,
                             repeated_similarity=repeated_similarity,
                             k=k,
                             balanced_ranks_factor=balanced_ranks_factor,
                             incoming_degree_factor=incoming_degree_factor,
                             outgoing_degree_factor=outgoing_degree_factor,
                             partition_method=partition_method,
                             target_metacell_size=target_metacell_size,
                             cell_sizes=cell_sizes,
                             min_split_factor=min_split_factor,
                             max_merge_factor=max_merge_factor,
                             random_seed=random_seed,
                             min_gene_fold_factor=min_gene_fold_factor,
                             max_genes_fraction=max_genes_fraction,
                             max_cells_fraction=max_cells_fraction,
                             min_robust_size_factor=min_robust_size_factor,
                             min_convincing_size_factor=min_convincing_size_factor,
                             min_convincing_gene_fold_factor=min_convincing_gene_fold_factor)

    mdata = collect_result_metacells(adata, cdata, of,
                                     name=name, tmp=tmp,
                                     intermediate=intermediate)

    if mdata is None:
        LOG.warning('! Empty metacells data, giving up')
        return None

    return mdata
