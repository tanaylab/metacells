'''
Partition
---------

Utilities for computing "good" partitions of a graph (typically a KNN-graph). These are just
extremely thin adapters for functions provided by external packages, specifically the `leidenalg`
package (which had to be extended to support some features we need - now part of its standard
distribution). It should be "easy" to provide your own wrappers for other algorithms, if needed.

.. todo::
    A large part of the overall wall-clock-time is spent in computing graph partitions, so a
    parallel algorithm would be very useful. It is tricky to do this efficiently, especially if one
    would also like replicable results.
'''

from typing import Callable, Optional, Tuple

import igraph as ig  # type: ignore
import leidenalg as la  # type: ignore
import numpy as np

import metacells.utilities.computation as utc
import metacells.utilities.timing as utm
import metacells.utilities.typing as utt

__all__ = [
    'PartitionMethod',
    'leiden_surprise',
    'leiden_bounded_surprise',
    'leiden_surprise_quality',
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
    #: ``min_comm_nodes``
    #:    An optional minimal number of nodes in each community. May be ignored by the function.
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
    #: categorical data to make it easier to further process the data.
    PartitionMethod = Callable[[NamedArg(utt.ProperMatrix, 'edge_weights'),
                                DefaultNamedArg(Optional[utt.NumpyVector],
                                                'node_sizes'),
                                DefaultNamedArg(Optional[int],
                                                'target_comm_size'),
                                DefaultNamedArg(Optional[int],
                                                'max_comm_size'),
                                DefaultNamedArg(Optional[int],
                                                'min_comm_size'),
                                DefaultNamedArg(Optional[int],
                                                'min_comm_nodes'),
                                DefaultNamedArg(int, 'random_seed')],
                               utt.NumpyVector]

except ModuleNotFoundError:
    __all__.remove('PartitionMethod')


def leiden_surprise(
    *,
    edge_weights: utt.ProperMatrix,
    node_sizes: Optional[utt.NumpyVector] = None,
    target_comm_size: Optional[int] = None,  # pylint: disable=unused-argument
    max_comm_size: Optional[int] = None,  # pylint: disable=unused-argument
    min_comm_size: Optional[int] = None,  # pylint: disable=unused-argument
    min_comm_nodes: Optional[int] = None,  # pylint: disable=unused-argument
    random_seed: int = 0,
) -> utt.NumpyVector:
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
    edge_weights: utt.ProperMatrix,
    node_sizes: Optional[utt.NumpyVector] = None,
    target_comm_size: Optional[int] = None,  # pylint: disable=unused-argument
    max_comm_size: Optional[int] = None,
    min_comm_size: Optional[int] = None,  # pylint: disable=unused-argument
    min_comm_nodes: Optional[int] = None,  # pylint: disable=unused-argument
    random_seed: int = 0,
) -> utt.NumpyVector:
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


def leiden_surprise_quality(
    *,
    edge_weights: utt.ProperMatrix,
    partition_of_nodes: utt.NumpyVector
) -> float:
    '''
    Return the quality score for a partition of nodes using the the ``SurpriseVertexPartition`` goal
    function.
    '''
    mask = partition_of_nodes[partition_of_nodes >= 0]
    partition_of_nodes = partition_of_nodes[mask]
    edge_weights = edge_weights[mask, :][:, mask]

    graph, _weights_array = _build_igraph(edge_weights)

    partition = la.SurpriseVertexPartition(graph, partition_of_nodes)
    return partition.quality()


def _build_igraph(edge_weights: utt.Matrix) -> Tuple[ig.Graph, utt.NumpyVector]:
    edge_weights = utt.to_proper_matrix(edge_weights)
    assert edge_weights.shape[0] == edge_weights.shape[1]
    size = edge_weights.shape[0]

    sources, targets = edge_weights.nonzero()
    weights_array = \
        utt.to_numpy_vector(edge_weights[sources, targets]).astype('float64')

    graph = ig.Graph(directed=True)
    graph.add_vertices(size)
    graph.add_edges(list(zip(sources, targets)))
    graph.es['weight'] = weights_array

    return graph, weights_array
