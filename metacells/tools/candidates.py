'''
Candidates
----------
'''

import sys
from math import ceil, floor
from typing import Optional, Set, Tuple, Union

import numpy as np
from anndata import AnnData

import metacells.parameters as pr
import metacells.utilities as ut

if not 'sphinx' in sys.argv[0]:
    import metacells.extensions as xt  # type: ignore

__all__ = [
    'compute_candidate_metacells',
    'choose_seeds',
    'optimize_partitions',
    'score_partitions',
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_candidate_metacells(  # pylint: disable=too-many-statements,too-many-branches
    adata: AnnData,
    what: Union[str, ut.Matrix] = 'obs_outgoing_weights',
    *,
    target_metacell_size: float,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.candidates_cell_sizes,
    cell_seeds: Optional[Union[str, ut.Vector]] = None,
    min_seed_size_quantile: float = pr.min_seed_size_quantile,
    max_seed_size_quantile: float = pr.max_seed_size_quantile,
    cooldown_pass: float = pr.cooldown_pass,
    cooldown_node: float = pr.cooldown_node,
    cooldown_phase: float = pr.cooldown_phase,
    min_split_size_factor: Optional[float] = pr.candidates_min_split_size_factor,
    max_merge_size_factor: Optional[float] = pr.candidates_max_merge_size_factor,
    min_metacell_cells: Optional[int] = pr.candidates_min_metacell_cells,
    max_split_min_cut_strength: Optional[float] = pr.max_split_min_cut_strength,
    min_cut_seed_cells: Optional[int] = pr.min_cut_seed_cells,
    must_complete_cover: bool = False,
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

    2. We start with some an assignment of cells to ``cell_seeds`` (default: {cell_seeds}). If no
       seeds are provided, we use :py:func:`choose_seeds` using ``min_seed_size_quantile`` (default:
       {min_seed_size_quantile}) and ``max_seed_size_quantile`` (default: {max_seed_size_quantile})
       to compute them, picking a number of seeds such that the average metacell size would match
       the target.

    3. We optimize the seeds using :py:func:`optimize_partitions` to obtain initial communities by
       maximizing the "stability" of the solution (probability of starting at a random node and
       moving either forward or backward in the graph and staying within the same metacell, divided
       by the probability of staying in the metacell if the edges connected random nodes). We pass
       it the ``cooldown_pass`` {cooldown_pass}) and ``cooldown_node`` (default: {cooldown_node}).

    4. If ``min_split_size_factor`` (default: {min_split_size_factor}) is specified, randomly split
       to two each community whose size is partition method on each community whose size is at least
       ``target_metacell_size * min_split_size_factor`` and re-optimize the solution (resulting in
       one additional metacell). Every time we re-optimize, we multiply 1 - ``cooldown_pass`` by
       1 - ``cooldown_phase`` (default: {cooldown_phase}).

    5. If ``max_split_min_cut_strength`` (default: {max_split_min_cut_strength}) is specified, and
       the minimal cut of a candidate is lower, split it into two. If one of the partitions is
       smaller than ``min_cut_seed_cells``, then mark the cells in it as outliers, or if
       ``must_complete_cover`` is ``True``, skip the cut altogether.

    5. If ``max_merge_size_factor`` (default: {max_merge_size_factor}) or ``min_metacell_cells``
       (default: {min_metacell_cells}) are specified, make outliers of cells of a community whose
       size is at most ``target_metacell_size * max_merge_size_factor`` or contains less cells and
       re-optimize, which will assign these cells to other metacells (resulting on one less
       metacell). We again apply the ``cooldown_phase`` every time we re-optimize.

    6. Repeat the above steps until all metacells candidates are in the acceptable size range.
    '''
    edge_weights = ut.get_oo_proper(adata, what, layout='row_major')
    assert edge_weights.shape[0] == edge_weights.shape[1]
    assert 0.0 < cooldown_pass < 1.0
    assert 0.0 <= cooldown_node <= 1.0
    assert 0.0 < cooldown_phase <= 1.0

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
        node_sizes = np.full(size, 1.0, dtype='float32')
    else:
        node_sizes = node_sizes.astype('float32')
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

    if community_of_nodes is not None:
        assert community_of_nodes.dtype == 'int32'
    else:
        target_seeds_count = ceil(size / target_metacell_cells)
        ut.log_calc('target_seeds_count', target_seeds_count)

        community_of_nodes = np.full(size, -1, dtype='int32')
        _choose_seeds(outgoing_edge_weights=outgoing_edge_weights,
                      incoming_edge_weights=incoming_edge_weights,
                      seed_of_cells=community_of_nodes,
                      max_seeds_count=target_seeds_count,
                      min_seed_size_quantile=min_seed_size_quantile,
                      max_seed_size_quantile=max_seed_size_quantile,
                      random_seed=random_seed)

    ut.set_o_data(adata, 'seed', community_of_nodes,
                  formatter=ut.groups_description)
    community_of_nodes = community_of_nodes.copy()

    np.random.seed(random_seed)

    cold_temperature = 1 - cooldown_pass

    old_score = 1e9
    old_communities = community_of_nodes
    old_small_nodes_count = len(community_of_nodes)
    atomic_candidates: Set[Tuple[int, ...]] = set()
    kept_communities_count = 0

    while True:
        cold_temperature, score = \
            _optimize_split_communities(  #
                outgoing_edge_weights=outgoing_edge_weights,
                incoming_edge_weights=incoming_edge_weights,
                community_of_nodes=community_of_nodes,
                node_sizes=node_sizes,
                target_metacell_size=target_metacell_size,
                max_metacell_size=max_metacell_size,
                max_split_min_cut_strength=max_split_min_cut_strength,
                min_cut_seed_cells=min_cut_seed_cells,
                must_complete_cover=must_complete_cover,
                min_seed_size_quantile=min_seed_size_quantile,
                max_seed_size_quantile=max_seed_size_quantile,
                random_seed=random_seed,
                cooldown_pass=cooldown_pass,
                cooldown_node=cooldown_node,
                cooldown_phase=cooldown_phase,
                kept_communities_count=kept_communities_count,
                cold_temperature=cold_temperature,
                atomic_candidates=atomic_candidates)

        small_communities, small_nodes_count = \
            _find_small_communities(community_of_nodes=community_of_nodes,
                                    node_sizes=node_sizes,
                                    min_metacell_size=min_metacell_size,
                                    min_metacell_cells=min_metacell_cells)

        small_communities_count = len(small_communities)
        if small_communities_count < 2:
            break

        if (old_small_nodes_count, old_score) <= (small_nodes_count, score):
            ut.logger().debug('is not better, revert')
            community_of_nodes = old_communities
            score = old_score
            ut.log_calc('communities', community_of_nodes,
                        formatter=ut.groups_description)
            ut.log_calc('score', score)
            break

        old_score = score
        old_communities = community_of_nodes.copy()
        old_small_nodes_count = small_nodes_count

        kept_communities_count = \
            _cancel_communities(community_of_nodes=community_of_nodes,
                                cancelled_communities=small_communities)

        _choose_seeds(outgoing_edge_weights=outgoing_edge_weights,
                      incoming_edge_weights=incoming_edge_weights,
                      seed_of_cells=community_of_nodes,
                      max_seeds_count=kept_communities_count + small_communities_count - 1,
                      min_seed_size_quantile=min_seed_size_quantile,
                      max_seed_size_quantile=max_seed_size_quantile,
                      random_seed=random_seed)

    if inplace:
        ut.set_o_data(adata, 'candidate', community_of_nodes,
                      formatter=ut.groups_description)
        return None

    if must_complete_cover:
        assert np.min(community_of_nodes) == 0
    else:
        community_of_nodes[community_of_nodes < 0] = -1

    ut.log_return('candidate', community_of_nodes,
                  formatter=ut.groups_description)
    return ut.to_pandas_series(community_of_nodes, index=adata.obs_names)


def _optimize_split_communities(
    outgoing_edge_weights: ut.CompressedMatrix,
    incoming_edge_weights: ut.CompressedMatrix,
    community_of_nodes: ut.NumpyVector,
    node_sizes: ut.NumpyVector,
    target_metacell_size: float,
    max_metacell_size: Optional[float],
    max_split_min_cut_strength: Optional[float],
    min_cut_seed_cells: Optional[int],
    must_complete_cover: bool,
    min_seed_size_quantile: float,
    max_seed_size_quantile: float,
    random_seed: int,
    cooldown_pass: float,
    cooldown_node: float,
    cooldown_phase: float,
    kept_communities_count: int,
    cold_temperature: float,
    atomic_candidates: Set[Tuple[int, ...]],
) -> Tuple[float, float]:
    while True:
        ut.log_calc('cold_temperature', cold_temperature)
        score = \
            _optimize_partitions(outgoing_edge_weights=outgoing_edge_weights,
                                 incoming_edge_weights=incoming_edge_weights,
                                 community_of_nodes=community_of_nodes,
                                 random_seed=random_seed,
                                 cooldown_pass=cooldown_pass,
                                 cooldown_node=cooldown_node,
                                 cold_communities_count=kept_communities_count,
                                 cold_temperature=cold_temperature)

        cold_temperature = cold_temperature * (1 - cooldown_phase)
        ut.log_calc('communities', community_of_nodes,
                    formatter=ut.groups_description)

        modified_communities_count, \
            new_communities_count, \
            kept_communities_count = \
            _split_communities(outgoing_edge_weights=outgoing_edge_weights,
                               community_of_nodes=community_of_nodes,
                               node_sizes=node_sizes,
                               target_metacell_size=target_metacell_size,
                               max_metacell_size=max_metacell_size,
                               max_split_min_cut_strength=max_split_min_cut_strength,
                               min_cut_seed_cells=min_cut_seed_cells,
                               must_complete_cover=must_complete_cover,
                               atomic_candidates=atomic_candidates)

        if new_communities_count > 0:
            assert modified_communities_count > 0
            _choose_seeds(outgoing_edge_weights=outgoing_edge_weights,
                          incoming_edge_weights=incoming_edge_weights,
                          seed_of_cells=community_of_nodes,
                          max_seeds_count=kept_communities_count + new_communities_count,
                          min_seed_size_quantile=min_seed_size_quantile,
                          max_seed_size_quantile=max_seed_size_quantile,
                          random_seed=random_seed)

        if modified_communities_count == 0:
            return cold_temperature, score


def _split_communities(  # pylint: disable=too-many-statements
    *,
    outgoing_edge_weights: ut.CompressedMatrix,
    community_of_nodes: ut.NumpyVector,
    node_sizes: ut.NumpyVector,
    target_metacell_size: float,
    max_metacell_size: Optional[float],
    max_split_min_cut_strength: Optional[float],
    min_cut_seed_cells: Optional[int],
    must_complete_cover: bool,
    atomic_candidates: Set[Tuple[int, ...]],
) -> Tuple[int, int, int]:
    communities_count = np.max(community_of_nodes) + 1
    assert communities_count > 0

    if max_metacell_size is None \
            and max_split_min_cut_strength is None:
        ut.logger().debug('no communities need to be split')
        return (0, 0, communities_count)

    cancelled_communities: Set[int] = set()
    split_communities_count = 0
    cut_communities_count = 0
    next_new_community_index = communities_count
    cancelled_total_size = 0

    did_not_add = False
    for community_index in range(communities_count):
        community_mask = community_of_nodes == community_index
        community_size = np.sum(node_sizes[community_mask])

        if max_metacell_size is not None and community_size > max_metacell_size:
            ut.logger().debug('community: %s nodes: %s size: %s is too large',
                              community_index, np.sum(community_mask), community_size)
            cancelled_total_size += community_size
            cancelled_communities.add(community_index)
            continue

        if max_split_min_cut_strength is None:
            continue

        community_indices = tuple(np.where(community_mask)[0])
        if community_indices in atomic_candidates:
            continue

        did_cut, did_split = \
            _min_cut_community(outgoing_edge_weights=outgoing_edge_weights,
                               community_of_nodes=community_of_nodes,
                               cut_community_index=community_index,
                               max_split_min_cut_strength=max_split_min_cut_strength,
                               min_cut_seed_cells=min_cut_seed_cells,
                               must_complete_cover=must_complete_cover,
                               new_community_index=next_new_community_index)
        if did_cut:
            cut_communities_count += 1
            ut.logger().debug('community: %s nodes: %s size: %s is too divided',
                              community_index, np.sum(community_mask), community_size)
            if did_split:
                split_communities_count += 1
                next_new_community_index += 1
        else:
            assert community_indices is not None
            atomic_candidates.add(community_indices)

    cancelled_communities_count = len(cancelled_communities)
    modified_communities_count = cut_communities_count + cancelled_communities_count
    if modified_communities_count == 0:
        ut.logger().debug('no communities were modified')
        return (0, 0, communities_count)

    ut.logger().debug('all communities: %s', communities_count)
    ut.logger().debug('cut communities: %s', cut_communities_count)
    ut.logger().debug('split communities: %s', split_communities_count)
    ut.logger().debug('cancelled communities: %s', cancelled_communities_count)

    if cancelled_communities_count > 0:
        new_communities_count = \
            max(ceil(cancelled_total_size / target_metacell_size),
                cancelled_communities_count + 1)
    else:
        new_communities_count = 0

    if did_not_add and new_communities_count == 0:
        new_communities_count = 1

    ut.logger().debug('reclaimed communities: %s', new_communities_count)

    kept_communities_count = \
        _cancel_communities(community_of_nodes, cancelled_communities)
    ut.logger().debug('kept communities: %s', kept_communities_count)
    assert kept_communities_count == \
        communities_count + split_communities_count - cancelled_communities_count

    return (modified_communities_count, new_communities_count, kept_communities_count)


def _min_cut_community(
    outgoing_edge_weights: ut.CompressedMatrix,
    community_of_nodes: ut.NumpyVector,
    cut_community_index: int,
    max_split_min_cut_strength: float,
    min_cut_seed_cells: Optional[int],
    must_complete_cover: bool,
    new_community_index: int
) -> Tuple[bool, bool]:
    community_mask = community_of_nodes == cut_community_index
    community_edge_weights = outgoing_edge_weights[community_mask,
                                                   :][:, community_mask]
    community_edge_weights += community_edge_weights.T

    cut, cut_strength = ut.min_cut(community_edge_weights)
    if cut_strength is None:
        return (False, False)

    ut.logger().debug('min cut community: %s partitions: %s + %s = %s strength: %s',
                      cut_community_index,
                      len(cut.partition[0]),
                      len(cut.partition[1]),
                      community_edge_weights.shape[0],
                      cut_strength)

    if cut_strength > max_split_min_cut_strength:
        return (False, False)

    community_indices = np.where(community_mask)[0]

    if min_cut_seed_cells is not None:
        if len(cut.partition[0]) < len(cut.partition[1]):
            small_partition = 0
        else:
            small_partition = 1

        if len(cut.partition[small_partition]) < min_cut_seed_cells:
            if must_complete_cover:
                return (False, False)
            small_community_indices = community_indices[cut.partition[small_partition]]
            community_of_nodes[small_community_indices] = -2
            ut.logger().debug('give up on small cut: %s',
                              len(cut.partition[small_partition]))
            return (True, False)

    second_partition_indices = community_indices[cut.partition[1]]
    community_of_nodes[second_partition_indices] = new_community_index
    return (True, True)


def _find_small_communities(
    *,
    community_of_nodes: ut.NumpyVector,
    node_sizes: ut.NumpyVector,
    min_metacell_size: Optional[float],
    min_metacell_cells: Optional[int],
) -> Tuple[Set[int], int]:
    communities_count = np.max(community_of_nodes) + 1
    assert communities_count > 0

    if min_metacell_size is None \
            and min_metacell_cells is None:
        ut.logger().debug('all communities are not too small')
        return (set(), communities_count)

    small_communities: Set[int] = set()
    small_nodes_count = 0

    for community_index in range(communities_count):
        community_mask = community_of_nodes == community_index
        community_nodes_count = np.sum(community_mask)
        community_size = np.sum(node_sizes[community_mask])

        if min_metacell_cells is not None and community_nodes_count < min_metacell_cells:
            ut.logger().debug('community: %s nodes: %s is too few',
                              community_index, community_nodes_count)
        elif min_metacell_size is not None and community_size < min_metacell_size:
            ut.logger().debug('community: %s nodes: %s size: %s is too small',
                              community_index, community_nodes_count, community_size)
        else:
            continue

        small_communities.add(community_index)
        small_nodes_count += community_nodes_count

    return (small_communities, small_nodes_count)


def _cancel_communities(
    community_of_nodes: ut.NumpyVector,
    cancelled_communities: Set[int]
) -> int:
    communities_count = np.max(community_of_nodes) + 1
    kept_communities_count = 0
    for community_index in range(communities_count):
        if community_index in cancelled_communities:
            community_of_nodes[community_of_nodes == community_index] = -1
            continue

        if community_index > kept_communities_count:
            community_of_nodes[community_of_nodes ==
                               community_index] = kept_communities_count

        kept_communities_count += 1

    assert kept_communities_count == communities_count - \
        len(cancelled_communities)
    return kept_communities_count


@ut.logged()
@ut.timed_call()
def choose_seeds(
    *,
    edge_weights: ut.CompressedMatrix,
    seed_of_cells: Optional[ut.NumpyVector] = None,
    max_seeds_count: int,
    min_seed_size_quantile: float = pr.min_seed_size_quantile,
    max_seed_size_quantile: float = pr.max_seed_size_quantile,
    random_seed: int,
) -> ut.NumpyVector:
    '''
    Choose initial assignment of cells to seeds based on the ``edge_weights``.

    Returns a vector assigning each node (cell) to a seed (initial community).

    If ``seed_of_cells`` is specified, it is expected to contain a vector of partial seeds. Only
    cells which have a negative seed will be assigned a new seed. New seeds will be created so that
    the total number of seeds will not exceed ``max_seeds_count``. The ``seed_of_cells`` will be
    modified in-place and returned.

    Otherwise, a new vector is created, initialized with ``-1`` (that is, no seed) for all nodes,
    filled as above, and returned.

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
    size = edge_weights.shape[0]

    outgoing_edge_weights = ut.mustbe_compressed_matrix(edge_weights)
    assert ut.matrix_layout(outgoing_edge_weights) == 'row_major'

    incoming_edge_weights = \
        ut.mustbe_compressed_matrix(ut.to_layout(outgoing_edge_weights,
                                                 layout='column_major'))
    assert ut.matrix_layout(incoming_edge_weights) == 'column_major'

    if seed_of_cells is None:
        seed_of_cells = np.full(size, -1, dtype='int32')
    else:
        assert seed_of_cells.dtype == 'int32'

    assert outgoing_edge_weights.shape \
        == incoming_edge_weights.shape \
        == (len(seed_of_cells), len(seed_of_cells))

    return _choose_seeds(outgoing_edge_weights=outgoing_edge_weights,
                         incoming_edge_weights=incoming_edge_weights,
                         seed_of_cells=seed_of_cells,
                         max_seeds_count=max_seeds_count,
                         min_seed_size_quantile=min_seed_size_quantile,
                         max_seed_size_quantile=max_seed_size_quantile,
                         random_seed=random_seed)


@ut.timed_call()
def _choose_seeds(
    *,
    outgoing_edge_weights: ut.CompressedMatrix,
    incoming_edge_weights: ut.CompressedMatrix,
    seed_of_cells: ut.NumpyVector,
    max_seeds_count: int,
    min_seed_size_quantile: float,
    max_seed_size_quantile: float,
    random_seed: int,
) -> ut.NumpyVector:
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
    cooldown_pass: float = pr.cooldown_pass,
    cooldown_node: float = pr.cooldown_node,
    random_seed: int,
) -> float:
    '''
    Optimize partition to candidate metacells (communities) using the ``edge_weights``.

    Returns the score of the optimized partition.

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

    We define a notion of ``temperature`` (initially, 1 - ``cooldown_pass``, default:
    {cooldown_pass}) and we give a weight of ``temperature`` to the local score and
    (1 - ``temperature``) to the global score. When we move to the next node, we multiply the
    temperature by 1 - ``cooldown_pass``. If we did not move the node, we multiply its temperature
    by ``cooldown_node`` (default: {cooldown_node}). We skip looking at nodes which are colder from
    the global temperature to accelerate the algorithm. If we don't move any node, we reduce the
    global temperature below that of any cold node; if there are no such nodes, we reduce it to zero
    to perform a final hill-climbing phase.

    This simulated-annealing-like behavior helps the algorithm to escape local maximums, although of
    course no claim is made of achieving the global maximum of the goal function.
    '''
    outgoing_edge_weights = ut.mustbe_compressed_matrix(edge_weights)
    assert ut.matrix_layout(outgoing_edge_weights) == 'row_major'

    incoming_edge_weights = \
        ut.mustbe_compressed_matrix(ut.to_layout(outgoing_edge_weights,
                                                 layout='column_major'))
    assert ut.matrix_layout(incoming_edge_weights) == 'column_major'
    return _optimize_partitions(outgoing_edge_weights=outgoing_edge_weights,
                                incoming_edge_weights=incoming_edge_weights,
                                random_seed=random_seed,
                                cooldown_pass=cooldown_pass,
                                cooldown_node=cooldown_node,
                                community_of_nodes=community_of_nodes,
                                cold_communities_count=0,
                                cold_temperature=cooldown_pass)


@ut.timed_call()
def _optimize_partitions(
    *,
    outgoing_edge_weights: ut.CompressedMatrix,
    incoming_edge_weights: ut.CompressedMatrix,
    community_of_nodes: ut.NumpyVector,
    cooldown_pass: float,
    cooldown_node: float,
    cold_communities_count: int,
    cold_temperature: float,
    random_seed: int,
) -> float:
    assert community_of_nodes.dtype == 'int32'
    score = xt.optimize_partitions(outgoing_edge_weights.data,
                                   outgoing_edge_weights.indices,
                                   outgoing_edge_weights.indptr,
                                   incoming_edge_weights.data,
                                   incoming_edge_weights.indices,
                                   incoming_edge_weights.indptr,
                                   random_seed,
                                   cooldown_pass,
                                   cooldown_node,
                                   community_of_nodes,
                                   cold_communities_count,
                                   cold_temperature)
    ut.log_calc('score', score)
    return score


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
