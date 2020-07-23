'''
Utilities for performing efficient computations.
'''

from typing import Callable, Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy import sparse  # type: ignore

import metacells.extensions as xt  # type: ignore
import metacells.utilities.documentation as utd
import metacells.utilities.timing as timed
from metacells.utilities.threading import parallel_for

__all__ = [
    'as_array',
    'corrcoef',
    'to_layout',
    'log_matrix',
    'sum_matrix',
    'nnz_matrix',
    'max_matrix',
    'downsample_array',
    'downsample_tmp_size',
]


Matrix = Union[sparse.spmatrix, np.ndarray, pd.DataFrame]
Vector = Union[np.ndarray, pd.Series]


DATA_TYPES = ['float32', 'float64', 'int32', 'int64', 'uint32', 'uint64']


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
def to_layout(matrix: Matrix, axis: int) -> Matrix:
    '''
    Re-layout a matrix for efficient axis slicing/processing.

    That is, for ``axis=0``, re-layout the matrix for efficient per-column (variable, gene)
    slicing/processing. For sparse matrices, this is ``csc`` format; for
    dense matrices, this is Fortran (column-major) format.

    Similarly, for ``axis=1``, re-layout the matrix for efficient per-row (observation, cell)
    slicing/processing. For sparse matrices, this is ``csr`` format; for dense matrices, this is C
    (row-major) format.

    If the matrix is already in the correct layout, it is returned as-is. Otherwise, a new copy is
    created. This is a costly operation as it needs to move a lot of data. However, it makes the
    following processing much more efficient, so it is typically a net performance gain overall.
    '''
    assert matrix.ndim == 2
    assert 0 <= axis <= 1

    if sparse.issparse(matrix):
        axis_format = ['csc', 'csr'][axis]
        if matrix.getformat() == axis_format:
            return matrix

        name = '.to' + axis_format
        with timed.step(name):
            return getattr(matrix, name[1:])()

    if axis == 0:
        if matrix.flags['F_CONTIGUOUS']:
            return matrix
        with timed.step('np.ravel'):
            return np.reshape(np.ravel(matrix, order='F'), matrix.shape, order='F')

    if matrix.flags['C_CONTIGUOUS']:
        return matrix
    with timed.step('np.ravel'):
        return np.reshape(np.ravel(matrix, order='C'), matrix.shape, order='C')


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

    results_count = matrix.shape[1 - axis]
    results = np.empty(results_count)

    if sparse.issparse(matrix):
        elements_count = len(matrix.indices) / results_count
    else:
        elements_count = matrix.shape[axis]

    if axis == 0:
        def batch_reducer(indices: range) -> None:
            results[indices] = as_array(reducer(matrix[:, indices]))
    else:
        def batch_reducer(indices: range) -> None:
            results[indices] = as_array(reducer(matrix[indices, :]))

    timed.parameters(obs_count=results_count, var_count=elements_count)

    parallel_for(batch_reducer, results_count)

    return results


@timed.call()
@utd.expand_doc(data_types=','.join(['``%s``' % data_type for data_type in DATA_TYPES]))
def downsample_array(
    data: np.ndarray,
    samples: int,
    *,
    tmp: Optional[np.ndarray] = None,
    output: Optional[np.ndarray] = None,
    random_seed: int = 0
) -> None:
    '''
    Downsample a vector of sample counters.

    **Input**

    * A Numpy array ``data`` containing non-negative integer sample counts.

    * A desired total number of ``samples``.

    * An optional temporary storage Numpy array minimize allocations for multiple invocations.

    * An optional Numpy array ``output`` to hold the results (otherwise, the input is overwritten).

    * An optional random seed to make the operation replicable.

    The arrays may have any of the data types: {data_types}.

    **Operation**

    If the total number of samples (sum of the data array) is not higher than the required number of
    samples, the output is identical to the input.

    Otherwise, treat the input as if it was a set where each index appeared its data count number of
    times. Randomly select the desired number of samples from this set (without repetition), and
    store in the output the number of times each index was chosen.

    **Motivation**

    Downsampling is an effective way to get the same number of samples in multiple observations (in
    particular, the same number of total UMIs in multiple cells), and serves as an alternative to
    normalization (e.g., working with UMI fractions instead of raw UMI counts).

    Downsampling is especially important when computing correlations between observations. When
    there is high variance between the total samples count in different observations (total UMI
    count in different cells), then normalization will return higher values when correlating
    observations with a higher sample count, which will result in an inflated estimation of their
    similarity to other observations. Downsampling avoids this effect.
    '''
    assert data.ndim == 1

    if tmp is None:
        tmp = np.empty(downsample_tmp_size(data.size), dtype='int32')
    else:
        tmp.resize(downsample_tmp_size(data.size))

    if output is None:
        output = data
    else:
        assert output.shape == data.shape

    function_name = \
        'downsample_%s_%s_%s' % (data.dtype, tmp.dtype, output.dtype)
    function = getattr(xt, function_name)
    function(data, tmp, output, samples, random_seed)


def downsample_tmp_size(size: int) -> int:
    '''
    Return the size of the temporary array needed to ``downsample`` data of the specified size.
    '''
    return xt.downsample_tmp_size(size)
