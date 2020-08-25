'''
Assign cells to (raw, candidate) metacells.
'''

from typing import Optional, Union

import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'compute_candidate_metacells',
]


@ut.timed_call()
@ut.expand_doc()
def compute_candidate_metacells(
    adata: AnnData,
    *,
    of: str = 'obs_outgoing_weights',
    partition_method: 'ut.PartitionMethod' = ut.leiden_bounded_surprise,
    target_metacell_size: Optional[int],
    cell_sizes: Optional[Union[str, ut.Vector]],
    minimal_split_factor: Optional[float] = 2.0,
    maximal_merge_factor: Optional[float] = 0.25,
    random_seed: int = 0,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Assign observations (cells) to (raw, candidate) metacells based ``of`` a weighted directed graph
    (by default, {of}).

    These candidate metacells typically go through additional vetting (e.g. outliers detection and
    dissolving too-small metacells) to obtain the final metacells.

    **Input**

    A :py:func:`metacells.utilities.preparation.prepare`-ed annotated ``adata``, where the
    observations are cells and the variables are genes.

    **Returns**

    Observation (Cell) Annotations
        ``candidate_metacell``
            The integer index of the (raw, candidate) metacell each cell belongs to. The metacells
            are in no particular order.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.utilities.partition.compute_communities`. The ``partition_method``
       (default: {partition_method.__qualname__}) and ``random_seed`` (default: {random_seed}) are
       passed to it as-is.

       Pass it the ``target_metacell_size`` as the ``target_comm_size``. Typically this desired
       metacell size is specified as some number of UMIs, and therefore ``cell_sizes`` is specified
       to be the name of a per-observation annotation, or an explicit vector, containing the total
       number of UMIs of the relevant genes per cell. If you want the target metacell size to be in
       term of the cells count, specify ``cell_sizes=None`` to get a default cell size of one.

       By default, this uses a ``minimal_split_factor`` of {minimal_split_factor} and a
       ``maximal_merge_factor`` of {maximal_merge_factor}, to ensure the candidate metacells fall
       within an acceptable range of sizes.
    '''
    with ut.intermediate_step(adata, intermediate=intermediate):
        edge_weights = ut.get_oo_data(adata, of or 'obs_outgoing_weights')
        if not isinstance(cell_sizes, str):
            node_sizes = cell_sizes
        else:
            node_sizes = ut.get_o_data(adata, cell_sizes)

    community_of_cells = \
        ut.compute_communities(edge_weights,
                               partition_method,
                               target_comm_size=target_metacell_size,
                               node_sizes=node_sizes,
                               minimal_split_factor=minimal_split_factor,
                               maximal_merge_factor=maximal_merge_factor,
                               random_seed=random_seed)

    if inplace:
        adata.obs['candidate_metacell'] = community_of_cells
        ut.safe_slicing_data('candidate_metacell', ut.NEVER_SAFE)
        return None

    return pd.Series(community_of_cells, index=adata.obs_names)
