'''
Candidates
----------
'''

from math import ceil, floor
from typing import Optional, Union

import numpy as np
from anndata import AnnData

import metacells.extensions as xt  # type: ignore
import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    'compute_candidate_metacells',
    'choose_seeds',
    'optimize_partitions',
    'score_partitions',
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_candidate_metacells(  # pylint: disable=too-many-statements
    adata: AnnData,
    what: Union[str, ut.Matrix] = 'obs_outgoing_weights',
    *,
    target_metacell_size: int,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.candidates_cell_sizes,
    cell_seeds: Optional[Union[str, ut.Vector]] = None,
    min_seed_size_quantile: float = pr.min_seed_size_quantile,
    max_seed_size_quantile: float = pr.max_seed_size_quantile,
    cooldown_step: float = pr.cooldown_step,
    cooldown_rate: float = pr.cooldown_rate,
    min_split_size_factor: Optional[float] = pr.candidates_min_split_size_factor,
    max_merge_size_factor: Optional[float] = pr.candidates_max_merge_size_factor,
    min_metacell_cells: Optional[int] = pr.candidates_min_metacell_cells,
    random_seed: int = 0,
    inplace: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Assign observations (cells) to (raw, candidate) metacells based on ``what`` data. (a weighted
    directed graph).

    These candidate metacells typically go through additional vetting (e.g. deviant detection and
    dissolving too-small metacells) to obtain the final metacells.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-observation-per-observation matrix where each row is the outgoing weights from
    each observation to the rest, or just the name of a per-observation-per-observation annotation
    containing such a matrix. Typically this matrix will be sparse for efficient processing.

    **Returns**

    Observation (Cell) Annotations
        ``candidate``
            The integer index of the (raw, candidate) metacell each cell belongs to. The metacells
            are in no particular order.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    **Computation Parameters**

    1. We are trying to build metacells of ``target_metacell_size``, using the ``cell_sizes``
       (default: {cell_sizes}) to assign a size for each node (cell). This can be a string name of a
       per-observation annotation or a vector of values.

       .. note::

            The cell sizes are converted to integer values, so if you have floating point sizes,
            make sure to scale them (and the target metacell size) so that the resulting integer
            values would make sense.

    2. We start with some an assignment of cells to ``cell_seeds`` (default: {cell_seeds}). If no
       seeds are provided, we use :py:func:`choose_seeds` using ``min_seed_size_quantile`` (default:
       {min_seed_size_quantile}) and ``max_seed_size_quantile`` (default: {max_seed_size_quantile})
       to compute them, picking a number of seeds such that the average metacell size would match
       the target.

    3. We optimize the seeds using :py:func:`optimize_partitions` to obtain initial communities by
       maximizing the "stability" of the solution (probability of starting at a random node and
       moving either forward or backward in the graph and staying within the same metacell, divided
       by the probability of staying in the metacell if the edges connected random nodes). We use a
       ``cooldown`` parameter of ``1 - cooldown_step / nodes_count``.

    4. If ``min_split_size_factor`` (default: {min_split_size_factor}) is specified, randomly split
       to two each community whose size is partition method on each community whose size is at least
       ``target_metacell_size * min_split_size_factor`` and re-optimize the solution (resulting in
       one additional metacell). Every time we re-optimize, we multiply the ``cooldown`` by the
       ``cooldown_rate`` (default: {cooldown_rate}).

    5. If ``max_merge_size_factor`` (default: {max_merge_size_factor}) or ``min_metacell_cells``
       (default: {min_metacell_cells}) are specified, make outliers of cells of a community whose
       size is at most ``target_metacell_size * max_merge_size_factor`` or contains less cells and
       re-optimize, which will assign these cells to other metacells (resulting on one less
       metacell). We again apply the ``cooldown_rate`` every time we re-optimize.

    6. Repeat the above steps until all metacells candidates are in the acceptable size range.
    '''
    edge_weights = ut.get_oo_proper(adata, what, layout='row_major')
    assert edge_weights.shape[0] == edge_weights.shape[1]
    assert 0.0 <= cooldown_rate <= 1.0

    size = edge_weights.shape[0]

    outgoing_edge_weights = ut.mustbe_compressed_matrix(edge_weights)

    assert ut.matrix_layout(outgoing_edge_weights) == 'row_major'
    incoming_edge_weights = \
        ut.mustbe_compressed_matrix(ut.to_layout(outgoing_edge_weights,
                                                 layout='column_major'))
    assert ut.matrix_layout(incoming_edge_weights) == 'column_major'

    assert outgoing_edge_weights.data.dtype == 'float32'
    assert outgoing_edge_weights.indices.dtype == 'int32'
    assert outgoing_edge_weights.indptr.dtype == 'int32'
    assert incoming_edge_weights.data.dtype == 'float32'
    assert incoming_edge_weights.indices.dtype == 'int32'
    assert incoming_edge_weights.indptr.dtype == 'int32'

    node_sizes = \
        ut.maybe_o_numpy(adata, cell_sizes, formatter=ut.sizes_description)
    if node_sizes is None:
        node_sizes = np.full(size, 1, dtype='int32')
    else:
        node_sizes = node_sizes.astype('int32')
    ut.log_calc('node_sizes', node_sizes, formatter=ut.sizes_description)

    assert target_metacell_size > 0
    max_metacell_size = None
    min_metacell_size = None

    if min_split_size_factor is not None:
        assert min_split_size_factor > 0
        max_metacell_size = \
            ceil(target_metacell_size * min_split_size_factor) - 1
    ut.log_calc('max_metacell_size', max_metacell_size)

    if max_merge_size_factor is not None:
        assert max_merge_size_factor > 0
        min_metacell_size = \
            floor(target_metacell_size * max_merge_size_factor) + 1
    ut.log_calc('min_metacell_size', min_metacell_size)

    target_metacell_cells = max(1.0 if min_metacell_cells is None else float(min_metacell_cells),
                                float(target_metacell_size / np.mean(node_sizes)))
    ut.log_calc('target_metacell_cells', target_metacell_cells)

    if min_split_size_factor is not None and max_merge_size_factor is not None:
        assert max_merge_size_factor < min_split_size_factor
        assert min_metacell_size is not None
        assert max_metacell_size is not None
        assert min_metacell_size <= max_metacell_size

    community_of_nodes = \
        ut.maybe_o_numpy(adata, cell_seeds, formatter=ut.groups_description)

    if community_of_nodes is None:
        target_seeds_count = ceil(size / target_metacell_cells)
        ut.log_calc('target_seeds_count', target_seeds_count)

        community_of_nodes = _choose_seeds(outgoing_edge_weights=outgoing_edge_weights,
                                           incoming_edge_weights=incoming_edge_weights,
                                           max_seeds_count=target_seeds_count,
                                           min_seed_size_quantile=min_seed_size_quantile,
                                           max_seed_size_quantile=max_seed_size_quantile,
                                           random_seed=random_seed)

    ut.set_o_data(adata, 'seed', community_of_nodes,
                  formatter=ut.groups_description)
    community_of_nodes = community_of_nodes.copy()

    np.random.seed(random_seed)
    cooldown = 1 - (cooldown_step / size)

    while True:
        _optimize_partitions(outgoing_edge_weights=outgoing_edge_weights,
                             incoming_edge_weights=incoming_edge_weights,
                             community_of_nodes=community_of_nodes,
                             random_seed=random_seed,
                             cooldown=cooldown)

        if not _patch_communities(community_of_nodes=community_of_nodes,
                                  node_sizes=node_sizes,
                                  min_metacell_size=min_metacell_size,
                                  min_metacell_cells=min_metacell_cells,
                                  max_metacell_size=max_metacell_size):
            break

        cooldown *= cooldown_rate

    ut.log_calc('communities', community_of_nodes,
                formatter=ut.groups_description)

    if inplace:
        ut.set_o_data(adata, 'candidate', community_of_nodes,
                      formatter=ut.groups_description)
        return None

    ut.log_return('candidate', community_of_nodes,
                  formatter=ut.groups_description)
    return ut.to_pandas_series(community_of_nodes, index=adata.obs_names)


@ut.logged()
@ut.timed_call()
def choose_seeds(
    *,
    edge_weights: ut.CompressedMatrix,
    max_seeds_count: int,
    min_seed_size_quantile: float,
    max_seed_size_quantile: float,
    random_seed: int,
) -> ut.NumpyVector:
    '''
    Choose initial assignment of cells to seeds based on the ``edge_weights``.

    Returns a vector assigning each node (cell) to a seed (initial community).

    **Computation Parameters**

    1. We compute for each candidate node the number of nodes it is connected to (by an outgoing
       edge).

    2. We pick as a seed a random node whose number of connected nodes ("seed size") quantile is at
       least ``min_seed_size_quantile`` and at most ``max_seed_size_quantile``. This ensures we pick
       seeds that aren't too small or too large to get a good coverage of the population with a low
       number of seeds.

    3. We assign each of the connected nodes to their seed, and discount them from the number of
       connected nodes of the remaining unassigned nodes.

    4. We repeat this until we reach the target number of seeds.
    '''
    outgoing_edge_weights = ut.mustbe_compressed_matrix(edge_weights)
    assert ut.matrix_layout(outgoing_edge_weights) == 'row_major'

    incoming_edge_weights = \
        ut.mustbe_compressed_matrix(ut.to_layout(outgoing_edge_weights,
                                                 layout='column_major'))
    assert ut.matrix_layout(incoming_edge_weights) == 'column_major'

    return _choose_seeds(outgoing_edge_weights=outgoing_edge_weights,
                         incoming_edge_weights=incoming_edge_weights,
                         max_seeds_count=max_seeds_count,
                         min_seed_size_quantile=min_seed_size_quantile,
                         max_seed_size_quantile=max_seed_size_quantile,
                         random_seed=random_seed)


@ut.timed_call()
def _choose_seeds(
    *,
    outgoing_edge_weights: ut.CompressedMatrix,
    incoming_edge_weights: ut.CompressedMatrix,
    max_seeds_count: int,
    min_seed_size_quantile: float,
    max_seed_size_quantile: float,
    random_seed: int,
) -> ut.NumpyVector:
    size = incoming_edge_weights.shape[0]
    seed_of_cells = np.full(size, -1, dtype='int32')

    xt.choose_seeds(outgoing_edge_weights.data,
                    outgoing_edge_weights.indices,
                    outgoing_edge_weights.indptr,
                    incoming_edge_weights.data,
                    incoming_edge_weights.indices,
                    incoming_edge_weights.indptr,
                    random_seed,
                    max_seeds_count,
                    min_seed_size_quantile,
                    max_seed_size_quantile,
                    seed_of_cells)

    ut.log_calc('seeds', seed_of_cells, formatter=ut.groups_description)
    return seed_of_cells


@ut.logged()
@ut.timed_call()
def optimize_partitions(
    *,
    edge_weights: ut.CompressedMatrix,
    community_of_nodes: ut.NumpyVector,
    cooldown: float,
    random_seed: int,
) -> None:
    '''
    Optimize partition to candidate metacells (communities) using the ``edge_weights``.

    This modifies the ``community_of_nodes`` in-place.

    The goal is to minimize the "stability" goal function which is defined to be the ratio between
    (1) the probability that, selecting a random node and either a random outgoing edge or a random
    incoming edge (biased by their weights), that the node connected to by that edge is in the same
    community (metacell) and (2) the probability that a random edge would lead to this same
    community (the fraction of its number of nodes out of the total).

    To maximize this, we repeatedly pass on a randomized permutation of the nodes, and for each
    node, move it to a random "better" community. When deciding if a community is better, we
    consider both (1) just the "local" product of the sum of the weights of incoming and outgoing edges
    between the node and the current and candidate communities and (2) the effect on the "global" goal
    function (considering the impact on this product for all other nodes connected to the current
    node).

    We define a notion of ``temperature`` (initially, the ``cooldown``) and we give a weight of
    ``temperature`` to the local score and (1 - ``temperature``) to the global score. When we move
    to the next node, we multiply the ``temperature`` by the ``cooldown``.

    This simulated-annealing-like behavior helps the algorithm to escape local maximums, although of
    course no claim is made of achieving the global maximum of the goal function.
    '''
    outgoing_edge_weights = ut.mustbe_compressed_matrix(edge_weights)
    assert ut.matrix_layout(outgoing_edge_weights) == 'row_major'

    incoming_edge_weights = \
        ut.mustbe_compressed_matrix(ut.to_layout(outgoing_edge_weights,
                                                 layout='column_major'))
    assert ut.matrix_layout(incoming_edge_weights) == 'column_major'
    _optimize_partitions(outgoing_edge_weights=outgoing_edge_weights,
                         incoming_edge_weights=incoming_edge_weights,
                         random_seed=random_seed,
                         cooldown=cooldown,
                         community_of_nodes=community_of_nodes)


@ut.timed_call()
def _optimize_partitions(
    *,
    outgoing_edge_weights: ut.CompressedMatrix,
    incoming_edge_weights: ut.CompressedMatrix,
    community_of_nodes: ut.NumpyVector,
    cooldown: float,
    random_seed: int,
) -> None:
    assert str(community_of_nodes.dtype) == 'int32'
    score = xt.optimize_partitions(outgoing_edge_weights.data,
                                   outgoing_edge_weights.indices,
                                   outgoing_edge_weights.indptr,
                                   incoming_edge_weights.data,
                                   incoming_edge_weights.indices,
                                   incoming_edge_weights.indptr,
                                   random_seed,
                                   cooldown,
                                   community_of_nodes)
    ut.log_calc('score', score)


@ut.logged()
@ut.timed_call()
def score_partitions(
    *,
    edge_weights: ut.CompressedMatrix,
    partition_of_nodes: ut.NumpyVector,
    with_orphans: bool = True,
) -> None:
    '''
    Compute the "stability" the "stability" goal function which is defined to be the ratio between
    (1) the probability that, selecting a random node and either a random outgoing edge or a random
    incoming edge (biased by their weights), that the node connected to by that edge is in the same
    community (metacell) and (2) the probability that a random edge would lead to this same
    community (the fraction of its number of nodes out of the total).

    If ``with_orphans`` is True (the default), outlier nodes are included in the computation. In
    general we add 1e-6 to the product of the incoming and outgoing weights so we can safely log it
    for efficient computation; thus orphans are given a very small (non-zero) weight so the overall
    score is not zeroed even when including them.
    '''
    assert str(partition_of_nodes.dtype) == 'int32'
    outgoing_edge_weights = ut.mustbe_compressed_matrix(edge_weights)
    assert ut.matrix_layout(outgoing_edge_weights) == 'row_major'

    incoming_edge_weights = \
        ut.mustbe_compressed_matrix(ut.to_layout(outgoing_edge_weights,
                                                 layout='column_major'))
    assert ut.matrix_layout(incoming_edge_weights) == 'column_major'

    with ut.unfrozen(partition_of_nodes):
        with ut.timed_step('.score'):
            score = xt.score_partitions(outgoing_edge_weights.data,
                                        outgoing_edge_weights.indices,
                                        outgoing_edge_weights.indptr,
                                        incoming_edge_weights.data,
                                        incoming_edge_weights.indices,
                                        incoming_edge_weights.indptr,
                                        partition_of_nodes,
                                        with_orphans)

    ut.log_calc('score', score)
    return score


def _patch_communities(
    *,
    community_of_nodes: ut.NumpyVector,
    node_sizes: ut.NumpyVector,
    min_metacell_size: Optional[float],
    min_metacell_cells: Optional[int],
    max_metacell_size: Optional[float]
) -> bool:
    if min_metacell_size is None \
            and min_metacell_cells is None \
            and max_metacell_size is None:
        return False

    communities_count = np.max(community_of_nodes) + 1
    assert communities_count > 0

    too_large_community_index: Optional[int] = None
    too_large_community_size = -1
    too_small_community_index: Optional[int] = None
    too_small_community_nodes_count = len(node_sizes) + 1

    for community_index in range(communities_count):
        community_mask = community_of_nodes == community_index
        community_nodes_count = np.sum(community_mask)

        if min_metacell_cells is not None and community_nodes_count < min_metacell_cells:
            ut.logger().debug('community: %s nodes: %s is too few',
                              community_index, community_nodes_count)
            if community_nodes_count < too_small_community_nodes_count:
                too_small_community_index = community_index
                too_small_community_nodes_count = community_nodes_count

        if min_metacell_size is None and max_metacell_size is None:
            continue

        community_size = np.sum(node_sizes[community_mask])

        if min_metacell_size is not None and community_size < min_metacell_size:
            ut.logger().debug('community: %s nodes: %s size: %s is too small',
                              community_index, community_nodes_count, community_size)
            if community_nodes_count < too_small_community_nodes_count:
                too_small_community_index = community_index
                too_small_community_nodes_count = community_nodes_count

        if max_metacell_size is not None and community_size > max_metacell_size:
            ut.logger().debug('community: %s nodes: %s size: %s is too large',
                              community_index, community_nodes_count, community_size)
            if community_size > too_large_community_size:
                too_large_community_index = community_index
                too_large_community_size = community_size

    if too_large_community_index is not None:
        community_node_indices = \
            np.where(community_of_nodes == too_large_community_index)[0]
        ut.log_calc('split too-large community', too_large_community_index)
        community_nodes_count = len(community_node_indices)
        split_node_indices = \
            np.random.permutation(community_node_indices)[:community_nodes_count
                                                          // 2]
        community_of_nodes[split_node_indices] = communities_count
        return True

    if too_small_community_index is not None:
        ut.log_calc('merge too-small community', too_small_community_index)
        community_of_nodes[community_of_nodes ==
                           too_small_community_index] = -1
        max_community_index = communities_count - 1
        if too_small_community_index != max_community_index:
            community_of_nodes[community_of_nodes ==
                               max_community_index] = too_small_community_index
        return True

    return False
