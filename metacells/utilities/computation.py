'''
Utilities for performing efficient computations.
'''

from typing import Any

import numpy as np  # type: ignore
from anndata import AnnData  # type: ignore
from scipy import sparse  # type: ignore

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


def sparse_corrcoef(matrix: Any) -> np.ndarray:
    '''
    Compute correlations between all rows of a sparse matrix.

    This should give the same results as ``numpy.corrcoef``, but faster.
    '''
    size = matrix.shape[1]
    sum_of_rows = matrix.sum(axis=1)
    centering = sum_of_rows.dot(sum_of_rows.T) / size
    correlations = (matrix.dot(matrix.T) - centering) / (size - 1)
    diagonal = np.diag(correlations)
    correlations /= np.sqrt(np.outer(diagonal, diagonal))
    return correlations


def totals_of_cells(adata: AnnData) -> np.ndarray:
    '''
    Compute the total number of UMIs per cell.
    '''
    return as_array(adata.X.sum(axis=1))


def totals_of_genes(adata: AnnData) -> np.ndarray:
    '''
    Compute the total number of UMIs per gene.
    '''
    return as_array(adata.X.sum(axis=0))


def non_zero_genes_of_cells(adata: AnnData) -> np.ndarray:
    '''
    Compute the number of genes with non-zero UMIs per cell.
    '''
    if sparse.issparse(adata.X):
        return as_array(adata.X.getnnz(axis=1))
    return as_array(np.count_nonzero(adata.X, axis=1))


def non_zero_cells_of_genes(adata: AnnData) -> np.ndarray:
    '''
    Compute the number of cells each gene has non-zero UMIs in.
    '''
    if sparse.issparse(adata.X):
        return as_array(adata.X.getnnz(axis=0))
    return as_array(np.count_nonzero(adata.X, axis=0))


def max_umis_of_genes(adata: AnnData) -> np.ndarray:
    '''
    Compute the number of cells each gene has non-zero UMIs in.
    '''
    if sparse.issparse(adata.X):
        return as_array(adata.X.max(axis=0))
    return as_array(np.amax(adata.X, axis=0))
