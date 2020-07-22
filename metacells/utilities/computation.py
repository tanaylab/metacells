'''
Utilities for performing efficient computations.
'''

from math import ceil
from typing import Callable, Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy import sparse  # type: ignore

import metacells.utilities.timing as timed
from metacells.utilities.threading import parallel_for

Matrix = Union[sparse.spmatrix, np.ndarray, pd.DataFrame]
Vector = Union[np.ndarray, pd.Series]

MINIMAL_INSTRUCTIONS_PER_BATCH = 2_000_000

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
    data: Matrix,
    *,
    base: Optional[float] = None,
    normalization: float = 1,
) -> np.ndarray:
    '''
    Compute the ``log2`` of some data.

    The ``base`` is added to the count before ``log2`` is applied, to handle the common case of zero
    values in sparse data.

    The ``normalization`` (default: {normalization}) is added to the count before the log is
    applied, to handle the common case of sparse data.
    '''
    if sparse.issparse(data):
        matrix = data.todense()
    elif isinstance(data, pd.DataFrame):
        matrix = np.copy(data.values)
    else:
        assert isinstance(data, np.ndarray)
        matrix = np.copy(data)

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
def sum_matrix(data: Matrix, axis: int) -> np.ndarray:
    '''
    Compute the total per row (``axis`` = 1) or column (``axis`` = 0) of some ``data``.
    '''
    return _reduce_matrix(data, axis, lambda data: data.sum(axis=axis))


@timed.call()
def nnz_matrix(data: Matrix, axis: int) -> np.ndarray:
    '''
    Compute the number of non-zero elements per row (``axis`` = 1) or column (``axis`` = 0) of some
    ``data``.
    '''
    if sparse.issparse(data):
        return _reduce_matrix(data, axis, lambda data: data.getnnz(axis=axis))
    return _reduce_matrix(data, axis, lambda data: np.count_nonzero(data, axis=axis))


@timed.call()
def max_matrix(data: Matrix, axis: int) -> np.ndarray:
    '''
    Compute the maximal element per row (``axis`` = 1) or column (``axis`` = 0) of some ``data``.
    '''
    if sparse.issparse(data):
        return _reduce_matrix(data, axis, lambda data: data.max(axis=axis))
    return _reduce_matrix(data, axis, lambda data: np.amax(data, axis=axis))


def _reduce_matrix(
    data: Matrix,
    axis: int,
    reducer: Callable,
    *,
    instructions_per_element: int = 1,
) -> np.ndarray:
    assert data.ndim == 2
    assert 0 <= axis <= 1

    elements_count = data.shape[axis]
    results_count = data.shape[1 - axis]
    results = np.empty(results_count)

    if sparse.issparse(data):
        if axis == 0:
            data = data.tocsc()
        else:
            data = data.tocsr()
        mean_elements_per_result = len(data.indices) / results_count
    else:
        mean_elements_per_result = elements_count

    if axis == 0:
        def batch_reducer(indices: range) -> None:
            results[indices] = as_array(reducer(data[:, indices]))
    else:
        def batch_reducer(indices: range) -> None:
            results[indices] = as_array(reducer(data[indices, :]))

    mean_instructions_per_result = mean_elements_per_result * instructions_per_element
    minimal_invocations_per_batch = \
        ceil(MINIMAL_INSTRUCTIONS_PER_BATCH / mean_instructions_per_result)

    timed.parameters(results_count=results_count, elements_count=elements_count,
                     mean_elements_per_result=mean_elements_per_result)

    parallel_for(batch_reducer, results_count,
                 minimal_invocations_per_batch=minimal_invocations_per_batch)

    return results
