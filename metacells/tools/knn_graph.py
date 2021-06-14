'''
K-Nearest-Neighbors Graph
-------------------------
'''

from typing import Optional, Union

import numpy as np
import scipy.sparse as sp  # type: ignore
from anndata import AnnData

import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    'compute_obs_obs_knn_graph',
    'compute_var_var_knn_graph',
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_obs_obs_knn_graph(
    adata: AnnData,
    what: Union[str, ut.Matrix] = 'obs_similarity',
    *,
    k: int,
    balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    inplace: bool = True,
) -> Optional[ut.PandasFrame]:
    '''
    Compute a directed  K-Nearest-Neighbors graph based on ``what`` (default: what) similarity data
    for each pair of observations (cells).

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-observation-per-observation matrix or the name of a
    per-observation-per-observation annotation containing such a matrix.

    **Returns**

    Observations-Pair Annotations
        ``obs_outgoing_weights``
            A sparse square matrix where each non-zero entry is the weight of an edge between a pair
            of cells or genes, where the sum of the weights of the outgoing edges for each element
            is 1 (there is always at least one such edge).

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas data frame (indexed by the observation names).

    **Computation Parameters**

    1. Use the ``obs_similarity`` and convert it to ranks (in descending order).
       This gives us a dense asymmetric ``<elements>_outgoing_ranks`` matrix.

    2. Convert the asymmetric outgoing ranks matrix into a symmetric ``obs_balanced_ranks`` matrix
       by element-wise multiplying it with its transpose and taking the square root. That is, for
       each edge to be high-balanced-rank, the geomean of its outgoing rank has to be high in both
       nodes it connects.

       .. note::

            This can drastically reduce the degree of the nodes, since to survive an edge needs to
            have been in the top ranks for both its nodes (as multiplying with zero drops the edge).
            This is why the ``balanced_ranks_factor`` needs to be large-ish.

    3. Keeping only balanced ranks of geomean of up to ``k * balanced_ranks_factor`` (default:
       {balanced_ranks_factor}). This does a preliminary pruning of low-quality edges.

    4. Prune the edges, keeping only the ``k * incoming_degree_factor`` (default: k *
       {incoming_degree_factor}) highest-ranked incoming edges for each node, and then only the ``k
       * outgoing_degree_factor`` (default: {outgoing_degree_factor}) highest-ranked outgoing edges
       for each node, while ensuring that the highest-balanced-ranked outgoing edge of each node is
       preserved. This gives us an asymmetric ``obs_pruned_ranks`` matrix, which has the structure
       we want, but not the correct edge weights yet.

       .. note::

            Balancing the ranks, and then pruning the incoming edges, ensures that "hub" nodes, that
            is nodes that many other nodes prefer to connect with, end up connected to a limited
            number of such "spoke" nodes.

    5. Normalize the outgoing edge weights by dividing them with the sum of their balanced ranks,
       such that the sum of the outgoing edge weights for each node is 1. Note that there is always
       at least one outgoing edge for each node. This gives us the ``obs_outgoing_weights`` for our
       directed K-Nearest-Neighbors graph.

       .. note::

            Ensuring each node has at least one outgoing edge allows us to always have at least one
            candidate grouping to add it to. This of course doesn't protect the node from being
            rejected by its group as deviant.
    '''
    return _compute_elements_knn_graph(adata, 'obs', what, k=k,
                                       balanced_ranks_factor=balanced_ranks_factor,
                                       incoming_degree_factor=incoming_degree_factor,
                                       outgoing_degree_factor=outgoing_degree_factor,
                                       inplace=inplace)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_var_var_knn_graph(
    adata: AnnData,
    what: Union[str, ut.Matrix] = 'var_similarity',
    *,
    k: int,
    balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    inplace: bool = True,
) -> Optional[ut.PandasFrame]:
    '''
    Compute a directed  K-Nearest-Neighbors graph based on ``what`` (default: what) similarity data
    for each pair of variables (genes).

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-variable matrix or the name of a per-variable-per-variable
    annotation containing such a matrix.

    **Returns**

    Variables-Pair Annotations
        ``var_outgoing_weights``
            A sparse square matrix where each non-zero entry is the weight of an edge between a pair
            of cells or genes, where the sum of the weights of the outgoing edges for each element
            is 1 (there is always at least one such edge).

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas data frame (indexed by the variable names).

    **Computation Parameters**

    1. Use the ``var_similarity`` and convert it to ranks (in descending order).
       This gives us a dense asymmetric ``<elements>_outgoing_ranks`` matrix.

    2. Convert the asymmetric outgoing ranks matrix into a symmetric ``var_balanced_ranks`` matrix
       by element-wise multiplying it with its transpose and taking the square root. That is, for
       each edge to be high-balanced-rank, the geomean of its outgoing rank has to be high in both
       nodes it connects.

       .. note::

            This can drastically reduce the degree of the nodes, since to survive an edge needs to
            have been in the top ranks for both its nodes (as multiplying with zero drops the edge).
            This is why the ``balanced_ranks_factor`` needs to be large-ish.

    3. Keeping only balanced ranks of up to ``k * k * balanced_ranks_factor`` (default:
       {balanced_ranks_factor}). This does a preliminary pruning of low-quality edges.

    4. Prune the edges, keeping only the ``k * incoming_degree_factor`` (default: k *
       {incoming_degree_factor}) highest-ranked incoming edges for each node, and then only the ``k
       * outgoing_degree_factor`` (default: {outgoing_degree_factor}) highest-ranked outgoing edges
       for each node, while ensuring that the highest-balanced-ranked outgoing edge of each node is
       preserved. This gives us an asymmetric ``var_pruned_ranks`` matrix, which has the structure
       we want, but not the correct edge weights yet.

       .. note::

            Balancing the ranks, and then pruning the incoming edges, ensures that "hub" nodes, that
            is nodes that many other nodes prefer to connect with, end up connected to a limited
            number of such "spoke" nodes.

    5. Normalize the outgoing edge weights by dividing them with the sum of their balanced ranks,
       such that the sum of the outgoing edge weights for each node is 1. Note that there is always
       at least one outgoing edge for each node. This gives us the ``var_outgoing_weights`` for our
       directed K-Nearest-Neighbors graph.

       .. note::

            Ensuring each node has at least one outgoing edge allows us to always have at least one
            candidate grouping to add it to. This of course doesn't protect the node from being
            rejected by its group as deviant.
    '''
    return _compute_elements_knn_graph(adata, 'var', what, k=k,
                                       balanced_ranks_factor=balanced_ranks_factor,
                                       incoming_degree_factor=incoming_degree_factor,
                                       outgoing_degree_factor=outgoing_degree_factor,
                                       inplace=inplace)


def _compute_elements_knn_graph(
    adata: AnnData,
    elements: str,
    what: Union[str, ut.Matrix] = '__x__',
    *,
    k: int,
    balanced_ranks_factor: float,
    incoming_degree_factor: float,
    outgoing_degree_factor: float,
    inplace: bool = True,
) -> Optional[ut.PandasFrame]:
    assert elements in ('obs', 'var')
    assert balanced_ranks_factor > 0.0
    assert incoming_degree_factor > 0.0
    assert outgoing_degree_factor > 0.0

    if elements == 'obs':
        get_data = ut.get_oo_proper
        set_data = ut.set_oo_data
    else:
        get_data = ut.get_vv_proper
        set_data = ut.set_vv_data

    def store_matrix(  #
        matrix: ut.CompressedMatrix,
        name: str,
        when: bool
    ) -> None:
        if when:
            name = elements + '_' + name
            set_data(adata, name, matrix,
                     formatter=lambda matrix:
                     ut.ratio_description(matrix.shape[0] * matrix.shape[1],
                                          'element', matrix.nnz, 'nonzero'))
        elif ut.logging_calc():
            ut.log_calc(f'{elements}_{name}',
                        ut.ratio_description(matrix.shape[0] * matrix.shape[1],
                                             'element', matrix.nnz, 'nonzero'))

    similarity = ut.to_proper_matrix(get_data(adata, what))
    similarity = ut.to_layout(similarity, 'row_major', symmetric=True)
    similarity = ut.to_numpy_matrix(similarity)

    ut.log_calc('similarity', similarity)

    outgoing_ranks = _rank_outgoing(similarity)

    balanced_ranks = _balance_ranks(outgoing_ranks, k, balanced_ranks_factor)
    store_matrix(balanced_ranks, 'balanced_ranks', True)

    pruned_ranks = _prune_ranks(balanced_ranks, k,
                                incoming_degree_factor,
                                outgoing_degree_factor)
    store_matrix(pruned_ranks, 'pruned_ranks', True)

    outgoing_weights = _weigh_edges(pruned_ranks)
    store_matrix(outgoing_weights, 'outgoing_weights', inplace)

    if inplace:
        return None

    if elements == 'obs':
        names = adata.obs_names
    else:
        names = adata.var_names

    return ut.to_pandas_frame(outgoing_weights, index=names, columns=names)


@ ut.timed_call()
def _rank_outgoing(similarity: ut.NumpyMatrix) -> ut.NumpyMatrix:
    size = similarity.shape[0]
    assert similarity.shape == (size, size)
    similarity = np.copy(similarity)

    min_similarity = ut.min_matrix(similarity)

    np.fill_diagonal(similarity, min_similarity - 1)

    assert ut.matrix_layout(similarity) == 'row_major'
    outgoing_ranks = ut.rank_matrix_by_layout(similarity, ascending=False)
    assert np.sum(np.diagonal(outgoing_ranks) == size) == size
    return outgoing_ranks


@ ut.timed_call()
def _balance_ranks(
    outgoing_ranks: ut.NumpyMatrix,
    k: int,
    balanced_ranks_factor: float
) -> ut.CompressedMatrix:
    size = outgoing_ranks.shape[0]

    with ut.timed_step('.multiply'):
        ut.timed_parameters(size=size)
        dense_balanced_ranks = outgoing_ranks
        assert np.sum(np.diagonal(dense_balanced_ranks) == size) == size
        dense_balanced_ranks *= outgoing_ranks.transpose()

    with ut.timed_step('.sqrt'):
        np.sqrt(dense_balanced_ranks, out=dense_balanced_ranks)

    max_rank = k * balanced_ranks_factor
    ut.log_calc('max_rank', max_rank)

    dense_balanced_ranks *= -1
    dense_balanced_ranks += 2**21

    with ut.timed_step('numpy.argmax'):
        ut.timed_parameters(size=size)
        max_index_of_each = \
            ut.to_numpy_vector(  #
                dense_balanced_ranks.argmax(axis=1))

    dense_balanced_ranks += max_rank + 1 - 2**21

    preserved_row_indices = np.arange(size)
    preserved_column_indices = max_index_of_each
    preserved_balanced_ranks = \
        ut.to_numpy_vector(dense_balanced_ranks[preserved_row_indices,
                                                preserved_column_indices])

    preserved_balanced_ranks[preserved_balanced_ranks < 1] = 1

    dense_balanced_ranks[dense_balanced_ranks < 0] = 0
    np.fill_diagonal(dense_balanced_ranks, 0)

    dense_balanced_ranks[preserved_row_indices, preserved_column_indices] = \
        preserved_balanced_ranks

    assert np.sum(np.diagonal(dense_balanced_ranks) == 0) == size
    sparse_balanced_ranks = sp.csr_matrix(dense_balanced_ranks)

    _assert_proper_compressed(sparse_balanced_ranks, 'csr')
    return sparse_balanced_ranks


@ ut.timed_call()
def _prune_ranks(
    balanced_ranks: ut.CompressedMatrix,
    k: int,
    incoming_degree_factor: float,
    outgoing_degree_factor: float
) -> ut.CompressedMatrix:
    size = balanced_ranks.shape[0]

    incoming_degree = int(round(k * incoming_degree_factor))
    incoming_degree = min(incoming_degree, size - 1)
    ut.log_calc('incoming_degree', incoming_degree)

    outgoing_degree = int(round(k * outgoing_degree_factor))
    outgoing_degree = min(outgoing_degree, size - 1)
    ut.log_calc('outgoing_degree', outgoing_degree)

    all_indices = np.arange(size)
    with ut.timed_step('numpy.argmax'):
        ut.timed_parameters(results=size, elements=balanced_ranks.nnz / size)
        max_index_of_each = ut.to_numpy_vector(balanced_ranks.argmax(axis=1))

    preserved_row_indices = all_indices
    preserved_column_indices = max_index_of_each
    preserved_balanced_ranks = \
        ut.to_numpy_vector(balanced_ranks[preserved_row_indices,
                                          preserved_column_indices])
    assert np.min(preserved_balanced_ranks) > 0
    preserved_matrix = \
        sp.coo_matrix((preserved_balanced_ranks,
                       (preserved_row_indices, preserved_column_indices)),
                      shape=balanced_ranks.shape)
    preserved_matrix.has_canonical_format = True

    pruned_ranks = \
        ut.mustbe_compressed_matrix(ut.to_layout(balanced_ranks,
                                                 'column_major',
                                                 symmetric=True))
    _assert_proper_compressed(pruned_ranks, 'csc')

    pruned_ranks = ut.prune_per(pruned_ranks, incoming_degree)
    _assert_proper_compressed(pruned_ranks, 'csc')

    pruned_ranks = \
        ut.mustbe_compressed_matrix(ut.to_layout(pruned_ranks, 'row_major'))
    _assert_proper_compressed(pruned_ranks, 'csr')

    pruned_ranks = ut.prune_per(pruned_ranks, outgoing_degree)
    _assert_proper_compressed(pruned_ranks, 'csr')

    with ut.timed_step('sparse.maximum'):
        ut.timed_parameters(collected=pruned_ranks.nnz,
                            preserved=preserved_matrix.nnz)
        pruned_ranks = pruned_ranks.maximum(preserved_matrix)
        pruned_ranks = pruned_ranks.maximum(preserved_matrix.transpose())

    ut.sort_compressed_indices(pruned_ranks)

    pruned_ranks = ut.mustbe_compressed_matrix(pruned_ranks)
    _assert_proper_compressed(pruned_ranks, 'csr')
    return pruned_ranks


@ut.timed_call()
def _weigh_edges(pruned_ranks: ut.CompressedMatrix) -> ut.CompressedMatrix:
    size = pruned_ranks.shape[0]

    total_ranks_per_row = ut.sum_per(pruned_ranks, per='row')

    ut.timed_parameters(size=size)
    scale_per_row = \
        np.reciprocal(total_ranks_per_row, out=total_ranks_per_row)
    edge_weights = pruned_ranks.multiply(scale_per_row[:, None])
    edge_weights = ut.to_layout(edge_weights, 'row_major')

    _assert_proper_compressed(edge_weights, 'csr')
    return edge_weights


def _assert_proper_compressed(matrix: ut.CompressedMatrix, layout: str) -> None:
    assert sp.issparse(matrix)
    assert ut.matrix_dtype(matrix) == 'float32'
    assert matrix.getformat() == layout
    assert matrix.has_sorted_indices
    assert matrix.has_canonical_format
