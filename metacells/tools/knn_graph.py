'''
Compute a K-Nearest-Neighbors graph based on similarity data.
'''

from typing import Callable, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy.sparse as sparse  # type: ignore
import scipy.stats as stats  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'compute_obs_obs_knn_graph',
    'compute_var_var_knn_graph',
]


@ut.timed_call()
@ut.expand_doc()
def compute_obs_obs_knn_graph(  # pylint: disable=too-many-locals
    adata: AnnData,
    of: Optional[str] = None,
    *,
    k: int,
    balanced_ranks_factor: float = 4.0,
    incoming_degree_factor: float = 3.0,
    outgoing_degree_factor: float = 1.0,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasFrame]:
    '''
    Compute a directed  K-Nearest-Neighbors graph based on similarity data for each pair of
    observations (cells).

    If ``of`` (default: {of}) is specified, this specific data is used. Otherwise,
    ``obs_similarity`` is used.

    **Input**

    A :py:func:`metacells.utilities.preparation.prepare`-ed annotated ``adata``, where the
    observations are cells and the variables are genes.

    **Returns**

    Observations-Pair Annotations
        ``obs_outgoing_weights``
            A sparse square matrix where each non-zero entry is the weight of an edge between a pair
            of cells or genes, where the sum of the weights of the outgoing edges for each element
            is 1 (there is always at least one such edge).

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas data frame (indexed by the observation names).

    If not ``intermediate`` (default: {intermediate}), this discards all the intermediate data used
    (e.g. sums). Otherwise, such data is kept for future reuse.

    **Computation Parameters**

    1. Use the ``obs_similarity`` to rank the edges, and keep the highest-ranked ``k *
       balanced_ranks_factor`` (default: k * {balanced_ranks_factor}) outgoing edges for each node.
       while ensuring the highest-ranked outgoing edge of each node is also used by the other node
       it is connected to. This gives us an asymmetric sparse ``<elements>_outgoing_ranks`` matrix.

    2. Convert the asymmetric outgoing ranks matrix into a symmetric ``obs_balanced_ranks`` matrix
       by element-wise multiplying it with its transpose. That is, for each edge to be
       high-balanced-rank, it has to be high-outgoing-rank for both the nodes it connects.

       .. note::

            This can drastically reduce the degree of the nodes, since to survive an edge needs to
            have been in the top ranks for both its nodes (as multiplying with zero drops the edge).
            This is why the ``balanced_ranks_factor`` needs to be large-ish. At the same time, we
            know at least the highest-ranked edge of each node will survive, so the minimum degree
            of a node using the balanced rank edges is 1.

    3. Prune the edges, keeping only the ``k * incoming_degree_factor`` (default: k *
       {incoming_degree_factor}) highest-ranked incoming edges for each node, and then only the ``k
       * outgoing_degree_factor`` (default: {outgoing_degree_factor}) highest-ranked outgoing edges
       for each node, while ensuring that the highest-balanced-ranked outgoing edge of each node is
       preserved. This gives us an asymmetric ``obs_pruned_ranks`` matrix, which has the structure
       we want, but not the correct edge weights yet.

       .. note::

            Balancing the ranks, and then pruning the incoming edges, ensures that "hub" nodes, that
            is nodes that many other nodes prefer to connect with, end up connected to a limited
            number of such "spoke" nodes.

    4. Normalize the outgoing edge weights by dividing them with the sum of their balanced ranks,
       such that the sum of the outgoing edge weights for each node is 1. Note that there is always
       at least one outgoing edge for each node. This gives us the ``obs_outgoing_weights`` for our
       directed K-Nearest-Neighbors graph.

       .. note::

            Ensuring each node has at least one outgoing edge allows us to always have at least one
            candidate grouping to add it to. This of course doesn't protect the node from being
            rejected by its group as an outlier.
    '''
    return _compute_elements_knn_graph(adata, 'obs', of, k=k,
                                       balanced_ranks_factor=balanced_ranks_factor,
                                       incoming_degree_factor=incoming_degree_factor,
                                       outgoing_degree_factor=outgoing_degree_factor,
                                       inplace=inplace, intermediate=intermediate)


@ut.timed_call()
@ut.expand_doc()
def compute_var_var_knn_graph(  # pylint: disable=too-many-locals
    adata: AnnData,
    of: Optional[str] = None,
    *,
    k: int,
    balanced_ranks_factor: float = 4.0,
    incoming_degree_factor: float = 3.0,
    outgoing_degree_factor: float = 1.0,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasFrame]:
    '''
    Compute a directed  K-Nearest-Neighbors graph based on similarity data for each pair of
    variables (genes).

    If ``of`` (default: {of}) is specified, this specific data is used. Otherwise,
    ``var_similarity`` is used.

    **Input**

    A :py:func:`metacells.utilities.preparation.prepare`-ed annotated ``adata``, where the
    observations are cells and the variables are genes.

    **Returns**

    Variables-Pair Annotations
        ``var_outgoing_weights``
            A sparse square matrix where each non-zero entry is the weight of an edge between a pair
            of cells or genes, where the sum of the weights of the outgoing edges for each element
            is 1 (there is always at least one such edge).

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas data frame (indexed by the variable names).

    If not ``intermediate`` (default: {intermediate}), this discards all the intermediate data used
    (e.g. sums). Otherwise, such data is kept for future reuse.

    **Computation Parameters**

    1. Use the ``var_similarity`` to rank the edges, and keep the highest-ranked ``k *
       balanced_ranks_factor`` (default: k * {balanced_ranks_factor}) outgoing edges for each node.
       while ensuring the highest-ranked outgoing edge of each node is also used by the other node
       it is connected to. This gives us an asymmetric sparse ``<elements>_outgoing_ranks`` matrix.

    2. Convert the asymmetric outgoing ranks matrix into a symmetric ``var_balanced_ranks`` matrix
       by element-wise multiplying it with its transpose. That is, for each edge to be
       high-balanced-rank, it has to be high-outgoing-rank for both the nodes it connects.

       .. note::

            This can drastically reduce the degree of the nodes, since to survive an edge needs to
            have been in the top ranks for both its nodes (as multiplying with zero drops the edge).
            This is why the ``balanced_ranks_factor`` needs to be large-ish. At the same time, we
            know at least the highest-ranked edge of each node will survive, so the minimum degree
            of a node using the balanced rank edges is 1.

    3. Prune the edges, keeping only the ``k * incoming_degree_factor`` (default: k *
       {incoming_degree_factor}) highest-ranked incoming edges for each node, and then only the ``k
       * outgoing_degree_factor`` (default: {outgoing_degree_factor}) highest-ranked outgoing edges
       for each node, while ensuring that the highest-balanced-ranked outgoing edge of each node is
       preserved. This gives us an asymmetric ``var_pruned_ranks`` matrix, which has the structure
       we want, but not the correct edge weights yet.

       .. note::

            Balancing the ranks, and then pruning the incoming edges, ensures that "hub" nodes, that
            is nodes that many other nodes prefer to connect with, end up connected to a limited
            number of such "spoke" nodes.

    4. Normalize the outgoing edge weights by dividing them with the sum of their balanced ranks,
       such that the sum of the outgoing edge weights for each node is 1. Note that there is always
       at least one outgoing edge for each node. This gives us the ``var_outgoing_weights`` for our
       directed K-Nearest-Neighbors graph.

       .. note::

            Ensuring each node has at least one outgoing edge allows us to always have at least one
            candidate grouping to add it to. This of course doesn't protect the node from being
            rejected by its group as an outlier.
    '''
    return _compute_elements_knn_graph(adata, 'var', of, k=k,
                                       balanced_ranks_factor=balanced_ranks_factor,
                                       incoming_degree_factor=incoming_degree_factor,
                                       outgoing_degree_factor=outgoing_degree_factor,
                                       inplace=inplace, intermediate=intermediate)


def _compute_elements_knn_graph(  # pylint: disable=too-many-locals
    adata: AnnData,
    elements: str,
    of: Optional[str] = None,
    *,
    k: int,
    balanced_ranks_factor: float = 4.0,
    incoming_degree_factor: float = 3.0,
    outgoing_degree_factor: float = 1.0,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasFrame]:
    assert elements in ('obs', 'var')
    assert balanced_ranks_factor > 0.0
    assert incoming_degree_factor > 0.0
    assert outgoing_degree_factor > 0.0

    if of is None:
        of = elements + '_similarity'

    if elements == 'obs':
        annotations = adata.obsp
        slicing_mask = ut.SAFE_WHEN_SLICING_OBS
    else:
        annotations = adata.varp
        slicing_mask = ut.SAFE_WHEN_SLICING_VAR

    def store_matrix(matrix: ut.CompressedMatrix, name: str, when: bool) -> None:
        if when:
            name = elements + '_' + name
            annotations[name] = matrix
            ut.safe_slicing_data(name, slicing_mask)

    similarity = ut.get_proper_matrix(adata, of)
    similarity = ut.to_layout(similarity, 'rows_major', symmetric=True)
    similarity = ut.DenseMatrix.be(similarity)

    with ut.timed_step('.outgoing_ranks'):
        outgoing_ranks = _rank_outgoing(similarity, k, balanced_ranks_factor)
        store_matrix(outgoing_ranks, 'outgoing_ranks', intermediate)

    with ut.timed_step('.balance_ranks'):
        balanced_ranks = _balance_ranks(outgoing_ranks)
        store_matrix(outgoing_ranks, 'balanced_ranks', intermediate)

    with ut.timed_step('.prune_ranks'):
        pruned_ranks = _prune_ranks(balanced_ranks, k,
                                    incoming_degree_factor,
                                    outgoing_degree_factor)
        store_matrix(outgoing_ranks, 'pruned_ranks', intermediate)

    with ut.timed_step('.weigh_edges'):
        outgoing_weights = _weigh_edges(pruned_ranks)
        store_matrix(outgoing_weights, 'outgoing_weights', inplace)

    if inplace:
        return None

    if elements == 'obs':
        names = adata.obs_names
    else:
        names = adata.var_names
    return pd.DataFrame(ut.to_dense_matrix(outgoing_weights), index=names, columns=names)


def _rank_outgoing(  # pylint: disable=too-many-locals
    similarity: ut.DenseMatrix,
    k: int,
    balanced_ranks_factor: float
) -> ut.CompressedMatrix:
    similarity = ut.to_dense_matrix(similarity, copy=True)
    size = similarity.shape[0]
    assert similarity.shape == (size, size)

    degree = int(round(k * balanced_ranks_factor))
    degree = min(degree, size - 1)

    with ut.timed_step('amin'):
        ut.timed_parameters(size=size * size)
        min_similarity = np.amin(similarity)

    np.fill_diagonal(similarity, min_similarity - 1)

    all_indices = np.arange(size)
    with ut.timed_step('argmax'):
        ut.timed_parameters(results=size, elements=size)
        max_index_of_each = similarity.argmax(axis=1)

    preserved_ranks = np.full(1, 2 * size, dtype='float32')
    preserved_row_indices = np.concatenate([all_indices, max_index_of_each])
    preserved_column_indices = np.concatenate([max_index_of_each, all_indices])
    preserved_matrix = \
        sparse.coo_matrix((preserved_ranks,
                           (preserved_row_indices, preserved_column_indices)),
                          shape=similarity.shape)
    preserved_matrix.has_canonical_format = True

    indptr = np.zeros(size + 1, dtype='int32')
    indices = np.zeros(degree * size, dtype='int32')
    ranks = np.zeros(degree * size, dtype='float32')

    def collect_outgoing_ranks(row_indices: range) -> None:
        for row_index in row_indices:
            row_similarities = similarity[row_index, :]
            indptr[row_index] = row_index * degree
            indptr[row_index + 1] = indptr[row_index] + degree
            indices_range = range(indptr[row_index], indptr[row_index + 1])
            if degree < size - 1:
                indices[indices_range] = \
                    np.argpartition(row_similarities, -degree)[-degree:]
                indices[indices_range].sort()
            else:
                indices[indices_range[0:row_index]] = range(0, row_index)
                indices[indices_range[row_index:]] = range(row_index + 1, size)
            ranks[indices_range] = \
                np.ArrayFloat32.be(stats.rankdata(row_similarities[indices[indices_range]],
                                                  method='ordinal'))

    with ut.timed_step('.collect'):
        ut.timed_parameters(size=size, keep=degree)
        ut.parallel_for(collect_outgoing_ranks, size)

    outgoing_ranks = sparse.csr_matrix((ranks, indices, indptr),
                                       shape=similarity.shape, dtype='float32')
    outgoing_ranks.has_sorted_indices = True
    outgoing_ranks.has_canonical_format = True

    with ut.timed_step('preserve'):
        ut.timed_parameters(collected=outgoing_ranks.nnz,
                            preserved=preserved_matrix.nnz)
        outgoing_ranks = outgoing_ranks.maximum(preserved_matrix)

    assert sparse.issparse(outgoing_ranks)
    assert outgoing_ranks.getformat() == 'csr'
    assert outgoing_ranks.has_sorted_indices
    assert outgoing_ranks.has_canonical_format

    return outgoing_ranks


def _balance_ranks(outgoing_ranks: ut.CompressedMatrix) -> ut.CompressedMatrix:
    size = outgoing_ranks.shape[0]

    transposed_ranks = \
        ut.CompressedMatrix.be(ut.to_layout(outgoing_ranks.transpose(),
                                            'row_major'))

    assert sparse.issparse(transposed_ranks)
    assert transposed_ranks.getformat() == 'csr'
    assert transposed_ranks.has_sorted_indices
    assert transposed_ranks.has_canonical_format

    with ut.timed_step('.multiply'):
        ut.timed_parameters(size=size, nnz=outgoing_ranks.nnz)
        balanced_ranks = \
            ut.CompressedMatrix.be(outgoing_ranks.multiply(transposed_ranks))

    assert sparse.issparse(balanced_ranks)
    assert balanced_ranks.getformat() == 'csr'
    assert balanced_ranks.has_sorted_indices
    assert balanced_ranks.has_canonical_format

    return balanced_ranks


def _prune_ranks(  # pylint: disable=too-many-locals,too-many-statements
    balanced_ranks: ut.CompressedMatrix,
    k: int,
    incoming_degree_factor: float,
    outgoing_degree_factor: float
) -> ut.CompressedMatrix:
    size = balanced_ranks.shape[0]

    incoming_degree = int(round(k * incoming_degree_factor))
    incoming_degree = min(incoming_degree, size - 1)

    outgoing_degree = int(round(k * outgoing_degree_factor))
    outgoing_degree = min(outgoing_degree, size - 1)

    maximal_degree = max(incoming_degree, outgoing_degree)

    all_indices = np.arange(size)
    with ut.timed_step('argmax'):
        ut.timed_parameters(results=size, elements=balanced_ranks.nnz / size)
        max_index_of_each = balanced_ranks.argmax(axis=1)

    preserved_row_indices = all_indices
    preserved_column_indices = max_index_of_each
    preserved_balanced_ranks = balanced_ranks[preserved_row_indices,
                                              preserved_column_indices]
    preserved_matrix = \
        sparse.coo_matrix((preserved_balanced_ranks,
                           (preserved_row_indices, preserved_column_indices)),
                          shape=balanced_ranks.shape)
    preserved_matrix.has_canonical_format = True

    degrees_array = np.empty(1 + size, dtype='int32')
    indices_matrix = np.empty((size, maximal_degree), dtype='int32')
    ranks_matrix = np.empty((size, maximal_degree), dtype='float32')

    pruned_ranks = \
        ut.CompressedMatrix.be(ut.to_layout(balanced_ranks,
                                            'column_major', symmetric=True))

    get_element = \
        lambda element_index: \
        ut.CompressedMatrix.be(pruned_ranks.getcol(element_index).transpose())
    pruned_degree = incoming_degree

    def collect_pruned(element_indices: range) -> None:
        for element_index in element_indices:
            indices_array = indices_matrix[element_index, :]
            ranks_array = ranks_matrix[element_index, :]

            element = get_element(element_index)
            assert element.has_sorted_indices

            element_non_zero_ranks = element.data
            element_non_zero_indices = element.indices
            element_degree = element_non_zero_indices.size
            assert element_non_zero_ranks.size == element_degree

            if element_degree <= pruned_degree:
                degrees_array[1 + element_index] = element_degree
                indices_array[:element_degree] = element_non_zero_indices
                ranks_array[:element_degree] = element_non_zero_ranks
            else:
                included_positions = \
                    np.argpartition(element_non_zero_ranks,
                                    -pruned_degree)[-pruned_degree:]
                included_positions.sort()

                degrees_array[1 + element_index] = pruned_degree
                indices_array[:pruned_degree] = element_non_zero_indices[included_positions]
                ranks_array[:pruned_degree] = element_non_zero_ranks[included_positions]

    def construct_pruned(matrix_constructor: Callable) -> ut.CompressedMatrix:
        degrees_array[0] = 0
        with ut.timed_step('cumsum'):
            ut.timed_parameters(size=size + 1)
            indptr = np.cumsum(degrees_array)

        ut.timed_parameters(size=size)
        indices = np.concatenate([indices_matrix[column_index,
                                                 range(degrees_array[column_index + 1])]
                                  for column_index in range(size)])
        ranks = np.concatenate([ranks_matrix[column_index,
                                             range(degrees_array[column_index + 1])]
                                for column_index in range(size)])

        return matrix_constructor((ranks, indices, indptr),
                                  shape=pruned_ranks.shape, dtype='float32')

    pruned_ranks = \
        ut.CompressedMatrix.be(ut.to_layout(pruned_ranks,
                                            'column_major', symmetric=True))

    assert pruned_ranks.getformat() == 'csc'
    assert pruned_ranks.has_sorted_indices
    assert pruned_ranks.has_canonical_format

    with ut.timed_step('.collect'):
        ut.timed_parameters(size=size, keep=pruned_degree)
        ut.parallel_for(collect_pruned, size)

    with ut.timed_step('.construct'):
        ut.timed_parameters(size=size)
        pruned_ranks = construct_pruned(sparse.csc_matrix)

    assert pruned_ranks.getformat() == 'csc'
    assert pruned_ranks.has_sorted_indices
    assert pruned_ranks.has_canonical_format

    pruned_ranks = \
        ut.CompressedMatrix.be(ut.to_layout(pruned_ranks, 'row_major'))

    assert pruned_ranks.getformat() == 'csr'
    assert pruned_ranks.has_sorted_indices
    assert pruned_ranks.has_canonical_format

    get_element = \
        lambda element_index: \
        ut.CompressedMatrix.be(pruned_ranks.getrow(element_index).transpose())
    pruned_degree = outgoing_degree

    with ut.timed_step('.collect'):
        ut.timed_parameters(size=size, keep=pruned_degree)
        ut.parallel_for(collect_pruned, size)

    with ut.timed_step('.construct'):
        ut.timed_parameters(size=size)
        pruned_ranks = construct_pruned(sparse.csr_matrix)

    assert pruned_ranks.getformat() == 'csr'
    assert pruned_ranks.has_sorted_indices
    assert pruned_ranks.has_canonical_format

    with ut.timed_step('preserve'):
        ut.timed_parameters(collected=pruned_ranks.nnz,
                            preserved=preserved_matrix.nnz)
        pruned_ranks = \
            ut.CompressedMatrix.be(pruned_ranks.maximum(preserved_matrix))

    assert pruned_ranks.getformat() == 'csr'
    assert pruned_ranks.has_sorted_indices
    assert pruned_ranks.has_canonical_format

    return pruned_ranks


def _weigh_edges(pruned_ranks: ut.CompressedMatrix) -> ut.CompressedMatrix:
    size = pruned_ranks.shape[0]

    total_ranks_per_row = ut.sum_per(pruned_ranks, per='row')
    with ut.timed_step('.scale'):
        ut.timed_parameters(size=size)
        scale_per_row = \
            np.reciprocal(total_ranks_per_row, out=total_ranks_per_row)
        edge_weights = \
            ut.CompressedMatrix.be(pruned_ranks.multiply(scale_per_row[:,
                                                                       None]))

    assert edge_weights.getformat() == 'csr'
    assert edge_weights.has_sorted_indices
    assert edge_weights.has_canonical_format

    return edge_weights
