'''
Utilities for performing efficient computations.
'''

from typing import Callable, Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy import sparse  # type: ignore

import metacells.utilities.timing as timed
from metacells.utilities.threading import parallel_for

Matrix = Union[sparse.spmatrix, np.ndarray, pd.DataFrame]
Vector = Union[np.ndarray, pd.Series]

__all__ = [
    'as_array',
    'corrcoef',
    'log_matrix',
    'sum_matrix',
    'nnz_matrix',
    'max_matrix',
]


def as_array(data: Union[Matrix, Vector]) -> np.ndarray:
    '''
    Convert a matrix to an array.
    '''
    if sparse.issparse(data):
        data = data.todense()
    return np.ravel(data)


@timed.call()
def corrcoef(matrix: Matrix) -> np.ndarray:
    '''
    Compute correlations between all observations (rows, cells) containing variables (columns,
    genes).

    This should give the same results as ``numpy.corrcoef``, but faster for sparse matrices.

    .. note::

        To correlate between observations (cells), the expected layout is the transpose of the
        layout of ``X`` in ``AnnData``.
    '''
    if not sparse.issparse(matrix):
        return np.corrcoef(matrix)

    obs_count = matrix.shape[0]
    var_count = matrix.shape[1]
    timed.parameters(obs_count=obs_count, var_count=var_count)
    sum_of_rows = matrix.sum(axis=1)
    assert len(sum_of_rows) == obs_count
    centering = sum_of_rows.dot(sum_of_rows.T) / var_count
    correlations = (matrix.dot(matrix.T) - centering) / (var_count - 1)
    assert correlations.shape == (obs_count, obs_count)
    diagonal = np.diag(correlations)
    correlations /= np.sqrt(np.outer(diagonal, diagonal))
    return correlations


@timed.call()
def log_matrix(
    matrix: Matrix,
    *,
    base: Optional[float] = None,
    normalization: float = 1,
) -> np.ndarray:
    '''
    Compute the ``log2`` of some ``matrix``.

    The ``base`` is added to the count before ``log2`` is applied, to handle the common case of zero
    values in sparse data.

    The ``normalization`` (default: {normalization}) is added to the count before the log is
    applied, to handle the common case of sparse data.
    '''
    if sparse.issparse(matrix):
        matrix = matrix.todense()
    elif isinstance(matrix, pd.DataFrame):
        matrix = np.copy(matrix.values)
    else:
        assert isinstance(matrix, np.ndarray)
        matrix = np.copy(matrix)

    matrix += normalization
    if base == 2:
        np.log2(matrix, out=matrix)
    else:
        np.log(matrix, out=matrix)
        if base is not None:
            assert base > 0
            matrix /= np.log(base)

    return matrix


@timed.call()
def sum_matrix(matrix: Matrix, axis: int) -> np.ndarray:
    '''
    Compute the total per row (``axis`` = 1) or column (``axis`` = 0) of some ``matrix``.
    '''
    return _reduce_matrix(matrix, axis, lambda matrix: matrix.sum(axis=axis))


@timed.call()
def nnz_matrix(matrix: Matrix, axis: int) -> np.ndarray:
    '''
    Compute the number of non-zero elements per row (``axis`` = 1) or column (``axis`` = 0) of some
    ``matrix``.
    '''
    if sparse.issparse(matrix):
        return _reduce_matrix(matrix, axis, lambda matrix: matrix.getnnz(axis=axis))
    return _reduce_matrix(matrix, axis, lambda matrix: np.count_nonzero(matrix, axis=axis))


@timed.call()
def max_matrix(matrix: Matrix, axis: int) -> np.ndarray:
    '''
    Compute the maximal element per row (``axis`` = 1) or column (``axis`` = 0) of some ``matrix``.
    '''
    if sparse.issparse(matrix):
        return _reduce_matrix(matrix, axis, lambda matrix: matrix.max(axis=axis))
    return _reduce_matrix(matrix, axis, lambda matrix: np.amax(matrix, axis=axis))


def _reduce_matrix(
    matrix: Matrix,
    axis: int,
    reducer: Callable,
) -> np.ndarray:
    assert matrix.ndim == 2
    assert 0 <= axis <= 1

    elements_count = matrix.shape[axis]
    results_count = matrix.shape[1 - axis]
    results = np.empty(results_count)

    if sparse.issparse(matrix):
        if axis == 0:
            matrix = matrix.tocsc()
        else:
            matrix = matrix.tocsr()
        elements_count = len(matrix.indices) / results_count

    if axis == 0:
        def batch_reducer(indices: range) -> None:
            results[indices] = as_array(reducer(matrix[:, indices]))
    else:
        def batch_reducer(indices: range) -> None:
            results[indices] = as_array(reducer(matrix[indices, :]))

    timed.parameters(obs_count=results_count, var_count=elements_count)

    parallel_for(batch_reducer, results_count)

    return results
