'''
Utilities for partitioning graphs.
'''

from dataclasses import dataclass
from math import ceil, floor
from typing import Callable, List, Optional, Tuple

import igraph as ig  # type: ignore
import leidenalg as la  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import metacells.utilities.computation as utc
import metacells.utilities.timing as utm
import metacells.utilities.typing as utt

__all__ = [
    'PartitionMethod',
    'compute_communities',
    'leiden_surprise',
    'leiden_bounded_surprise',
    'bin_pack',
]

try:
    from mypy_extensions import DefaultNamedArg, NamedArg

    #: A function for partitioning data.
    #:
    #: **Parameters:**
    #:
    #: ``edge_weights``
    #:    A square matrix of non-negative floating-point weights, where the entries in each row are
    #:    the weights of the edges from the node represented by the row to the rest of the nodes.
    #:
    #: ``node_sizes``
    #:    An optional vector of non-negative integer size for each node. If not specified, each node
    #:    has a size of one.
    #:
    #: ``target_comm_size``
    #:    An optional "ideal" community size to compute. May be ignored by the function.
    #:
    #: ``min_comm_size``
    #:    An optional minimal community size to compute. May be ignored by the function.
    #:
    #: ``max_comm_size``
    #:    An optional maximal community size to compute. May be ignored by the function.
    #:
    #: ``random_seed``
    #:    An optional random seed to ensure reproducible results. The default zero value will use the
    #:    current time.
    #:
    #: **Returns:**
    #:
    #: A numpy integer array holding the community index of each node, in no particular order. The
    #: community indices are expected to be consecutive, starting with 0. This is used instead of
    #: categorical data to make it easier to do further processing of the data.
    PartitionMethod = Callable[[NamedArg(utt.Matrix, 'edge_weights'),
                                DefaultNamedArg(Optional[utt.DenseVector],
                                                'node_sizes'),
                                DefaultNamedArg(Optional[int],
                                                'target_comm_size'),
                                DefaultNamedArg(Optional[int],
                                                'max_comm_size'),
                                DefaultNamedArg(Optional[int],
                                                'min_comm_size'),
                                DefaultNamedArg(int, 'random_seed')],
                               utt.DenseVector]

except ModuleNotFoundError:
    __all__.remove('PartitionMethod')


@dataclass
class Community:
    '''
    Metadata about a community.
    '''

    #: The index identifying the community.
    index: int

    #: The total size of the community (sum of the node sizes).
    size: int

    #: A boolean mask of all the nodes that belong to the community.
    mask: utt.DenseVector

    #: By how much (if at all) is the community smaller than the minimum allowed.
    too_small: int

    #: By how much (if at all) is the community larger than the maximal allowed.
    too_large: int

    #: Whether this community can't be split.
    monolithic: bool


@utm.timed_call()
def compute_communities(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    edge_weights: utt.Matrix,
    partition_method: 'PartitionMethod',
    *,
    target_comm_size: Optional[int] = None,
    node_sizes: Optional[utt.Vector] = None,
    minimal_split_factor: Optional[float] = None,
    maximal_merge_factor: Optional[float] = None,
    random_seed: int = 0,
) -> utt.DenseVector:
    '''
    Compute communities based on a weighted directed graph.

    **Input**

    A square matrix of non-negative floating-point ``edge_weights``, where the entries in each row
    are the weights of the edges from the node represented by the row to the rest of the nodes.

    **Returns**

    A numpy integer array holding the community index of each node, in no particular order. The
    community indices are expected to be consecutive, starting with 0. This is used instead of
    categorical data to make it easier to do further processing of the data.

    **Computation Parameters**

    1. If ``target_comm_size`` (default: {target_comm_size}) is specified, we are trying to obtain
       communities with this ideal total size. Use the ``node_sizes`` (default: {node_sizes}) to
       assign a size for each node. If not specified, each has a size of one.

       .. note::

            The node sizes are converted to integer values, so if you have floating point sizes,
            make sure to scale them (and the target size) so that the resulting integer values would
            make sense.

    2. Use the ``partition_method`` to compute initial communities. Several such possible methods
       are provided in this module, and you can also provide your own as long as it is compatible
       with :py:const:`metacells.utilities.partition.PartitionMethod` interface.

    3. If both ``target_comm_size`` and ``minimal_split_factor`` (default: {minimal_split_factor})
       are specified, re-run the partition method on each community whose size is at least
       ``target_comm_size
       * minimal_split_factor``, to split it to into smaller communities.

    4. If both ``target_comm_size`` and ``maximal_merge_factor`` (default: {minimal_merge_factor})
       are specified, condense each community whose size is at most ``target_comm_size
       * maximal_merge_factor`` into a single node (using the mean of the edge weights),
       and re-run the partition method on the resulting graph to merge these communities into large
       ones.

    5. Repeat the above steps until no further progress can be made.

    6. If a minimal community size was specified, arbitrarily combine the remaining smaller
       communities into groups no larger than the target community size.

    .. note::

        This doesn't guarantee that all communities would be in the size range we want, but comes as
        close as possible to it given the choice of partition method. Also, since we force merging
        of communities beyond what the partition method would have done on its own, not all the
        communities would have the same quality. Any too-low-quality groupings are expected to be
        corrected by removing outliers and/or by dissolving too-small communities.

    .. note::

        The partition method is given all the necessary hints so it can, if possible, generate
        better-sized communities to reduce or eliminate the need for the split and merge steps.
        However, most partition algorithms do not naturally allow for this level of control over the
        resulting communities.
    '''
    edge_weights = utc.to_layout(edge_weights, 'row_major')
    assert edge_weights.shape[0] == edge_weights.shape[1]
    size = edge_weights.shape[0]

    if node_sizes is not None:
        node_sizes = utt.to_dense_vector(node_sizes).astype('int')

    max_comm_size = None
    min_comm_size = None
    if target_comm_size is not None:
        assert target_comm_size > 0
        if minimal_split_factor is not None:
            assert minimal_split_factor > 0
            max_comm_size = ceil(target_comm_size * minimal_split_factor) - 1
        if maximal_merge_factor is not None:
            assert maximal_merge_factor > 0
            min_comm_size = floor(target_comm_size * maximal_merge_factor) + 1
        if minimal_split_factor is not None and maximal_merge_factor is not None:
            assert maximal_merge_factor < minimal_split_factor
            assert min_comm_size is not None
            assert max_comm_size is not None
            assert min_comm_size <= max_comm_size

    with utm.timed_step('.partition'):
        membership = partition_method(edge_weights=edge_weights,
                                      node_sizes=node_sizes,
                                      target_comm_size=target_comm_size,
                                      max_comm_size=max_comm_size,
                                      min_comm_size=min_comm_size,
                                      random_seed=random_seed)

    doing_splits = max_comm_size is not None
    doing_merges = min_comm_size is not None

    if not doing_splits and not doing_merges:
        return membership

    with utm.timed_step('.correction'):
        communities: List[Community] = []
        communities_count = 0
        too_small = 0
        too_large = 0

        def append_communities(count: int) -> None:
            nonlocal communities_count, too_small, too_large

            for index in range(count):
                index += communities_count
                mask = membership == index

                if node_sizes is None:
                    total_size = np.sum(mask)
                else:
                    total_size = np.sum(node_sizes[mask])

                if min_comm_size is not None and total_size < min_comm_size:
                    small = min_comm_size - total_size
                    too_small += small
                else:
                    small = 0

                if max_comm_size is not None and total_size > max_comm_size:
                    large = total_size - max_comm_size
                    too_large += large
                else:
                    large = 0

                communities.append(Community(index=index,
                                             size=total_size, mask=mask,
                                             too_small=small, too_large=large,
                                             monolithic=False))

            communities_count += count

        def _remove_community(position: int) -> None:
            nonlocal too_small, too_large
            community = communities[position]
            communities[position:position+1] = []
            too_small -= community.too_small
            too_large -= community.too_large
            assert too_small >= 0
            assert too_large >= 0

        append_communities(np.max(membership) + 1)

        @utm.timed_call('.split')
        def split_large_communities() -> bool:
            did_split = False
            position = 0
            while position < len(communities):
                split_community = communities[position]
                if split_community.too_large == 0 or split_community.monolithic:
                    position += 1
                    continue

                split_edge_weights = edge_weights[split_community.mask, :]
                split_edge_weights = \
                    utc.to_layout(split_edge_weights, 'column_major')
                split_edge_weights = split_edge_weights[:,
                                                        split_community.mask]
                split_node_sizes = None if node_sizes is None \
                    else node_sizes[split_community.mask]
                split_nodes_membership = partition_method(edge_weights=split_edge_weights,
                                                          node_sizes=split_node_sizes,
                                                          target_comm_size=target_comm_size,
                                                          max_comm_size=max_comm_size,
                                                          min_comm_size=min_comm_size,
                                                          random_seed=random_seed)

                split_communities_count = np.max(split_nodes_membership) + 1
                if split_communities_count == 1:
                    split_community.monolithic = True
                    position += 1
                    continue

                did_split = True

                _remove_community(position)

                split_nodes_membership += communities_count
                membership[split_community.mask] = split_nodes_membership
                append_communities(split_communities_count)

            return did_split

        @utm.timed_call('.merge')
        def merge_small_communities() -> bool:
            merged_nodes_mask = np.zeros(size, dtype='bool')
            location_of_nodes = np.full(size, -1)
            merged_communities: List[Community] = []
            position_of_merged_communities: List[int] = []
            size_of_merged_communities: List[int] = []

            for position, community in enumerate(communities):
                if community.too_small == 0:
                    continue
                merged_nodes_mask |= community.mask
                location_of_nodes[community.mask] = len(merged_communities)
                merged_communities.append(community)
                position_of_merged_communities.append(position)
                size_of_merged_communities.append(community.size)

            if len(merged_communities) < 2:
                return False

            edge_weights_of_merged_nodes = edge_weights[merged_nodes_mask, :]
            edge_weights_of_merged_nodes = \
                utc.to_layout(edge_weights_of_merged_nodes, 'column_major')
            edge_weights_of_merged_nodes = edge_weights_of_merged_nodes[:,
                                                                        merged_nodes_mask]
            edge_weights_of_merged_nodes = \
                utt.to_dense_matrix(edge_weights_of_merged_nodes)
            merge_frame = pd.DataFrame(edge_weights_of_merged_nodes)
            location_of_merged_nodes = location_of_nodes[merged_nodes_mask]
            merge_frame = \
                merge_frame.groupby(location_of_merged_nodes, axis=0).mean()
            merge_frame = \
                merge_frame.groupby(location_of_merged_nodes, axis=1).mean()
            merged_communities_edge_weights = utt.to_proper_matrix(merge_frame)
            np.fill_diagonal(merged_communities_edge_weights, 0)

            merged_communities_node_sizes = \
                np.array(size_of_merged_communities)

            merged_communities_membership = \
                partition_method(edge_weights=merged_communities_edge_weights,
                                 node_sizes=merged_communities_node_sizes,
                                 target_comm_size=target_comm_size,
                                 max_comm_size=max_comm_size,
                                 min_comm_size=min_comm_size,
                                 random_seed=random_seed)

            for merged_community, merged_index \
                    in zip(merged_communities, merged_communities_membership):
                membership[merged_community.mask] = \
                    merged_index + communities_count

            before_too_small = too_small
            for position in reversed(position_of_merged_communities):
                _remove_community(position)

            merged_communities_count = \
                np.max(merged_communities_membership) + 1
            append_communities(merged_communities_count)

            return too_small < before_too_small

        if doing_merges:
            while too_small > 0:
                if not merge_small_communities():
                    break

        penalty = too_small + too_large + 1
        while too_small + too_large < penalty:
            penalty = too_small + too_large
            if doing_splits:
                while too_large > 0:
                    if not split_large_communities():
                        break
                if doing_merges:
                    while too_small > 0:
                        if not merge_small_communities():
                            break

        if doing_merges:
            with utm.timed_step('.pack'):
                list_of_small_community_sizes: List[int] = []
                list_of_small_community_indices: List[int] = []
                for community in communities:
                    if community.too_small == 0:
                        continue
                    list_of_small_community_sizes.append(community.size)
                    list_of_small_community_indices.append(community.index)

                if len(list_of_small_community_sizes) > 0:
                    assert target_comm_size is not None
                    small_community_sizes = \
                        np.array(list_of_small_community_sizes)
                    small_community_bins = \
                        bin_pack(small_community_sizes, target_comm_size)
                    for community_index, community_bin \
                            in zip(list_of_small_community_indices, small_community_bins):
                        new_community_index = communities_count + community_bin + 1
                        membership[membership ==
                                   community_index] = new_community_index

        return utc.compress_indices(membership)


def leiden_surprise(
    *,
    edge_weights: utt.Matrix,
    node_sizes: Optional[utt.DenseVector] = None,
    target_comm_size: Optional[int] = None,  # pylint: disable=unused-argument
    max_comm_size: Optional[int] = None,  # pylint: disable=unused-argument
    min_comm_size: Optional[int] = None,  # pylint: disable=unused-argument
    random_seed: int = 0,
) -> utt.DenseVector:
    '''
    Use the Leiden algorithm from the ``leidenalg`` package to compute partitions,
    using the ``SurpriseVertexPartition`` goal function.

    This ignores all the size hints.
    '''
    return leiden_bounded_surprise(edge_weights=edge_weights,
                                   node_sizes=node_sizes,
                                   max_comm_size=None,
                                   random_seed=random_seed)


@utm.timed_call('leiden_surprise')
def leiden_bounded_surprise(
    *,
    edge_weights: utt.Matrix,
    node_sizes: Optional[utt.DenseVector] = None,
    target_comm_size: Optional[int] = None,  # pylint: disable=unused-argument
    max_comm_size: Optional[int] = None,
    min_comm_size: Optional[int] = None,  # pylint: disable=unused-argument
    random_seed: int = 0,
) -> utt.DenseVector:
    '''
    Use the Leiden algorithm from the ``leidenalg`` package to compute partitions,
    using the ``SurpriseVertexPartition`` goal function.

    This passes the ``max_comm_size`` to the Leiden algorithm so it does not create communities
    larger than this size.
    '''
    graph, weights_array = _build_igraph(edge_weights)
    utm.timed_parameters(size=edge_weights.shape[0], edges=weights_array.size)

    kwargs = dict(n_iterations=-1, weights=weights_array, seed=random_seed)
    if max_comm_size is not None:
        utm.timed_parameters(max_comm_size=max_comm_size)
        kwargs['max_comm_size'] = max_comm_size
    if node_sizes is not None:
        kwargs['node_sizes'] = [int(size) for size in node_sizes]

    partition = la.find_partition(graph, la.SurpriseVertexPartition, **kwargs)
    membership = np.array(partition.membership)
    return utc.compress_indices(membership)


def _build_igraph(edge_weights: utt.Matrix) -> Tuple[ig.Graph, utt.DenseVector]:
    edge_weights = utt.to_proper_matrix(edge_weights)
    assert edge_weights.shape[0] == edge_weights.shape[1]
    size = edge_weights.shape[0]

    sources, targets = edge_weights.nonzero()
    weights_array = \
        utt.to_dense_vector(edge_weights[sources, targets]).astype('float64')

    graph = ig.Graph(directed=True)
    graph.add_vertices(size)
    graph.add_edges(list(zip(sources, targets)))
    graph.es['weight'] = weights_array

    return graph, weights_array


@utm.timed_call()
def bin_pack(element_sizes: utt.Vector, max_bin_size: float) -> utt.DenseVector:
    '''
    Given a vector of ``element_sizes`` and the ``max_bin_size``, return a vector containing the bin
    number for each element.
    '''
    size_of_bins: List[float] = []
    element_sizes = utt.to_dense_vector(element_sizes)
    descending_size_indices = np.argsort(element_sizes)[::-1]
    bin_of_elements = np.empty(element_sizes.size, dtype='int')

    for element_index in descending_size_indices:
        element_size = element_sizes[element_index]

        assigned = False
        for bin_index in range(len(size_of_bins)):  # pylint: disable=consider-using-enumerate
            if size_of_bins[bin_index] + element_size <= max_bin_size:
                bin_of_elements[element_index] = bin_index
                size_of_bins[bin_index] += element_size
                assigned = True
                break

        if not assigned:
            bin_of_elements[element_index] = len(size_of_bins)
            size_of_bins.append(element_size)

    did_improve = True
    while did_improve:
        did_improve = False
        for element_index in descending_size_indices:
            element_size = element_sizes[element_index]
            current_bin_index = bin_of_elements[element_index]

            current_bin_space = max_bin_size - size_of_bins[current_bin_index]
            remove_loss = \
                (element_size + current_bin_space) ** 2 - current_bin_space ** 2

            for bin_index in range(len(size_of_bins)):  # pylint: disable=consider-using-enumerate
                if bin_index == current_bin_index:
                    continue
                bin_space = max_bin_size - size_of_bins[bin_index]
                if bin_space < element_size:
                    continue
                insert_gain = bin_space ** 2 - (bin_space - element_size) ** 2
                if insert_gain > remove_loss:
                    size_of_bins[current_bin_index] -= element_size
                    current_bin_index = bin_of_elements[element_index] = bin_index
                    size_of_bins[bin_index] += element_size
                    remove_loss = insert_gain
                    did_improve = True

    utm.timed_parameters(elements=element_sizes.size, bins=len(size_of_bins))
    return bin_of_elements
