'''
Utilities for performing efficient computations.
'''

from math import ceil
from typing import Any, Callable

import numpy as np  # type: ignore
from anndata import AnnData  # type: ignore
from scipy import sparse  # type: ignore

import metacells.utilities.timing as timed
from metacells.utilities.threading import parallel_for

MINIMAL_OPERATIONS_PER_BATCH = 200_000

__all__ = [
    'as_array',
    'sparse_corrcoef',

    'totals_of_cells',
    'totals_of_genes',
    'non_zero_genes_of_cells',
    'non_zero_cells_of_genes',
]


def as_array(data: Any) -> np.ndarray:
    '''
    Convert a matrix to an array.
    '''
    if sparse.issparse(data):
        data = data.todense()
    return np.ravel(data)


@timed.call
def sparse_corrcoef(matrix: Any) -> np.ndarray:
    '''
    Compute correlations between all rows of a sparse matrix.

    This should give the same results as ``numpy.corrcoef``, but faster.
    '''
    timed.parameters(*matrix.shape)
    size = matrix.shape[1]
    sum_of_rows = matrix.sum(axis=1)
    centering = sum_of_rows.dot(sum_of_rows.T) / size
    correlations = (matrix.dot(matrix.T) - centering) / (size - 1)
    diagonal = np.diag(correlations)
    correlations /= np.sqrt(np.outer(diagonal, diagonal))
    return correlations


@timed.call
def totals_of_cells(adata: AnnData) -> np.ndarray:
    '''
    Compute the total number of UMIs per cell.
    '''
    return _reduce_cells(adata, lambda data: data.sum(axis=1))


@timed.call
def totals_of_genes(adata: AnnData) -> np.ndarray:
    '''
    Compute the total number of UMIs per gene.
    '''
    return _reduce_genes(adata, lambda data: data.sum(axis=0))


@timed.call
def non_zero_genes_of_cells(adata: AnnData) -> np.ndarray:
    '''
    Compute the number of genes with non-zero UMIs per cell.
    '''
    if sparse.issparse(adata.X):
        return _reduce_cells(adata, lambda data: data.getnnz(axis=1))
    return _reduce_cells(adata, lambda data: np.count_nonzero(data, axis=1))


@timed.call
def non_zero_cells_of_genes(adata: AnnData) -> np.ndarray:
    '''
    Compute the number of cells each gene has non-zero UMIs in.
    '''
    if sparse.issparse(adata.X):
        return _reduce_genes(adata, lambda data: data.getnnz(axis=0))
    return _reduce_genes(adata, lambda data: np.count_nonzero(data, axis=0))


@timed.call
def max_umis_of_genes(adata: AnnData) -> np.ndarray:
    '''
    Compute the number of cells each gene has non-zero UMIs in.
    '''
    if sparse.issparse(adata.X):
        return _reduce_genes(adata, lambda data: data.max(axis=0))
    return _reduce_genes(adata, lambda data: np.amax(data, axis=0))


def _reduce_cells(
    adata: AnnData,
    reducer: Callable,
    *,
    operations_per_gene: int = 1,
) -> np.ndarray:
    cells_count = adata.X.shape[0]
    results = np.empty(cells_count)

    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr()
        genes_count = X.data.size / cells_count
    else:
        genes_count = adata.X.shape[1]

    def batch_reducer(indices: range) -> None:
        results[indices] = as_array(reducer(X[indices, :]))

    timed.parameters('non_zero_genes_per_cell', genes_count)
    minimal_invocations_per_batch = \
        _minimal_invocations_per_batch(genes_count, operations_per_gene)
    parallel_for(batch_reducer, cells_count,
                 minimal_invocations_per_batch=minimal_invocations_per_batch)

    return results


def _reduce_genes(
    adata: AnnData,
    reducer: Callable,
    *,
    operations_per_cell: int = 1,
) -> np.ndarray:
    genes_count = adata.X.shape[1]

    results = np.empty(genes_count)

    X = adata.X
    if sparse.issparse(X):
        X = X.tocsc()
        cells_count = X.data.size / genes_count
    else:
        cells_count = adata.X.shape[0]

    def batch_reducer(indices: range) -> None:
        results[indices] = as_array(reducer(X[:, indices]))

    timed.parameters('non_zero_cells_per_gene', cells_count)
    minimal_invocations_per_batch = \
        _minimal_invocations_per_batch(cells_count, operations_per_cell)
    parallel_for(batch_reducer, genes_count,
                 minimal_invocations_per_batch=minimal_invocations_per_batch)

    return results


def _minimal_invocations_per_batch(items_count: int, operations_per_item: int = 1) -> int:
    operations_per_invocation = items_count * operations_per_item
    return ceil(MINIMAL_OPERATIONS_PER_BATCH / operations_per_invocation)
