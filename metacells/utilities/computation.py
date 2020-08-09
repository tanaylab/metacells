'''
Utilities for performing efficient uniform computations.

Most of the functions defined here are thin wrappers around builtin Numpy functions. However, they
are provided with a uniform interface that works for both sparse and dense data. This allows safely
passing them to functions such as :py:func:`metacells.utilities.preparation.get_per_obs` etc.

All the functions here (optionally) collect timing information using
:py:mod:`metacells.utilities.timing`, to make it easier to locate the performance bottleneck of the
analysis pipeline.
'''

from typing import Callable, Optional
from warnings import warn

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy import sparse  # type: ignore

import metacells.extensions as xt  # type: ignore
import metacells.utilities.documentation as utd
import metacells.utilities.timing as timed
import metacells.utilities.typing as utt
from metacells.utilities.threading import SharedStorage, parallel_for

__all__ = [
    'to_layout',
    'relayout_compressed',

    'corrcoef',

    'log_matrix',

    'max_axis',
    'min_axis',
    'nnz_axis',
    'sum_axis',
    'sum_squared_axis',

    'bincount_array',

    'downsample_matrix',
    'downsample_array',
    'downsample_tmp_size',

    'sliding_window_function',
]


@ timed.call()
def to_layout(matrix: utt.Matrix, layout: str) -> utt.Matrix:  # pylint: disable=too-many-return-statements
    '''
    Re-layout a matrix for efficient axis slicing/processing.

    That is, if ``layout`` is ``column_major``, re-layout the matrix for efficient per-column
    (variable, gene) slicing/processing. For sparse matrices, this is ``csc`` format; for dense
    matrices, this is Fortran (column-major) format.

    Similarly, if ``layout`` is ``row_major``, re-layout the matrix for efficient per-row
    (observation, cell) slicing/processing. For sparse matrices, this is ``csr`` format; for dense
    matrices, this is C (row-major) format.

    If the matrix is already in the correct layout, it is returned as-is. Otherwise, a new copy is
    created. This is a costly operation as it needs to move a lot of data. However, it makes the
    following processing much more efficient, so it is typically a net performance gain overall.
    '''
    if utt.is_layout(matrix, layout):
        return matrix

    is_frozen = utt.frozen(matrix)

    if sparse.issparse(matrix):
        to_axis_format = utt.SPARSE_FAST_FORMAT[layout]
        from_axis_format = utt.SPARSE_SLOW_FORMAT[layout]
        if matrix.getformat() == from_axis_format:
            matrix = relayout_compressed(matrix)
        else:
            name = '.to' + to_axis_format
            with timed.step(name):
                timed.parameters(rows=matrix.shape[0], columns=matrix.shape[1])
                matrix = getattr(matrix, name[1:])()

    else:  # Dense.
        matrix = utt.unpandas(matrix)
        order = utt.DENSE_FAST_FLAG[layout][0]
        with timed.step('.ravel'):
            timed.parameters(rows=matrix.shape[0], columns=matrix.shape[1])
            matrix = np.reshape(np.ravel(matrix, order=order),
                                matrix.shape, order=order)

    if is_frozen:
        utt.freeze(matrix)

    return matrix


@ timed.call()
def relayout_compressed(matrix: sparse.spmatrix) -> sparse.spmatrix:
    '''
    Efficient parallel conversion of a CSR/CSC matrix to a CSC/CSR matrix.
    '''
    assert matrix.ndim == 2
    axis = ['csr', 'csc'].index(matrix.getformat())

    _output_elements_count = matrix_bands_count = matrix.shape[axis]
    output_bands_count = matrix_elements_count = matrix.shape[1 - axis]

    nnz_elements_of_output_bands = \
        bincount_array(matrix.indices, minlength=matrix_elements_count)

    output_indptr = np.empty(output_bands_count + 1, dtype=matrix.indptr.dtype)
    output_indptr[0:2] = 0
    with timed.step('cumsum'):
        timed.parameters(elements=nnz_elements_of_output_bands.size - 1)
        np.cumsum(nnz_elements_of_output_bands[:-1], out=output_indptr[2:])

    output_indices = np.empty(matrix.indices.size, dtype=matrix.indices.dtype)
    output_data = np.empty(matrix.data.size, dtype=matrix.data.dtype)

    extension_name = 'collect_compressed_%s_t_%s_t_%s_t' \
        % (matrix.data.dtype, matrix.indices.dtype, matrix.indptr.dtype)
    extension = getattr(xt, extension_name)

    def collect_compressed(matrix_band_indices: range) -> None:
        extension(matrix_band_indices.start, matrix_band_indices.stop,
                  matrix.data, matrix.indices, matrix.indptr,
                  output_data, output_indices, output_indptr[1:])

    with timed.step('.collect_compressed'):
        timed.parameters(results=output_bands_count,
                         elements=matrix_bands_count)
        parallel_for(collect_compressed, matrix_bands_count)

    assert output_indptr[-1] == matrix.indptr[-1]

    constructor = [sparse.csc_matrix, sparse.csr_matrix][axis]
    matrix = constructor((output_data, output_indices,
                          output_indptr), shape=matrix.shape)
    matrix.has_sorted_indices = True
    matrix.has_canonical_format = True
    return matrix


@timed.call()
def corrcoef(matrix: utt.Matrix, *, rowvar: bool = True) -> np.ndarray:
    '''
    Compute correlations between all observations (rows, cells) containing variables (columns,
    genes).

    .. todo::

        This always uses the dense implementation, which seems to be the best choice for correlating
        the less-sparse "feature" genes.
    '''
    matrix = utt.to_np_array(matrix)

    if rowvar:
        timed.parameters(results=matrix.shape[0], elements=matrix.shape[1])
    else:
        timed.parameters(results=matrix.shape[1], elements=matrix.shape[0])

    return np.corrcoef(matrix, rowvar=rowvar)


#: A function that derives one matrix from another of the same size.
def log_matrix(
    matrix: utt.Matrix,
    *,
    base: Optional[float] = None,
    normalization: float = 0,
) -> Callable[[utt.Matrix], utt.Matrix]:
    '''
    Return the log of the values in the ``matrix``.

    The ``base`` is added to the count before ``log`` is applied, to handle the common case of zero
    values in sparse data.

    The ``normalization`` (default: {normalization}) is added to the count before the log is
    applied, to handle the common case of sparse data.

    .. note::

        The result is always a dense matrix, as even for sparse data, the log is rarely zero.
    '''
    matrix = utt.to_np_array(matrix, copy=True)

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
def downsample_matrix(
    matrix: utt.Matrix,
    *,
    axis: int,
    samples: int,
    eliminate_zeros: bool = True,
    inplace: bool = False,
    random_seed: int = 0,
) -> utt.Matrix:
    '''
    Downsample the rows (``axis = 1``) or columns (``axis = 0``) of some ``matrix`` such that the
    sum of each one becomes ``samples``.

    If the matrix is sparse, if not ``eliminate_zeros``, then do not perform the final phase of
    eliminating leftover zero values from the compressed format. This means the result will not be
    in "canonical format" so some ``scipy`` sparse operations on it will be slower.

    If ``inplace``, modify the matrix, otherwise, return a modified copy.

    A ``random_seed`` can be provided to make the operation replicable.
    '''
    assert matrix.ndim == 2
    assert 0 <= axis <= 1

    assert samples > 0

    if not sparse.issparse(matrix):
        matrix = utt.unpandas(matrix)
        return _downsample_dense_matrix(matrix, axis, samples, inplace, random_seed)

    return _downsample_sparse_matrix(matrix, axis, samples, eliminate_zeros, inplace, random_seed)


def _downsample_sparse_matrix(
    matrix: sparse.spmatrix,
    axis: int,
    samples: int,
    eliminate_zeros: bool,
    inplace: bool,
    random_seed: int
) -> sparse.spmatrix:
    elements_count = matrix.shape[axis]
    results_count = matrix.shape[1 - axis]

    axis_format = ['csc', 'csr'][axis]
    if matrix.getformat() != axis_format:
        raise NotImplementedError('downsample axis: %s '
                                  'of a sparse matrix in the format: %s '
                                  'instead of the efficient format: %s'
                                  % (axis, matrix.getformat(), axis_format))

    if inplace:
        output = matrix
    else:
        constructor = [sparse.csc_matrix, sparse.csr_matrix][axis]
        output_data = np.empty(matrix.data.shape, dtype=matrix.data.dtype)
        output = constructor((output_data, matrix.indices, matrix.indptr))

    shared_storage = SharedStorage()
    max_tmp_size = downsample_tmp_size(elements_count)

    def downsample_sparse_vectors(indices: range) -> None:
        for index in indices:
            start_index = matrix.indptr[index]
            stop_index = matrix.indptr[index + 1]

            input_vector = matrix.data[start_index:stop_index]
            output_vector = output.data[start_index:stop_index]

            tmp = \
                shared_storage.get_private('tmp',
                                           make=lambda: np.empty(max_tmp_size,
                                                                 dtype=matrix.dtype))

            if random_seed != 0:
                index_seed = random_seed + index
            else:
                index_seed = 0

            _downsample_array(input_vector, samples, tmp,
                              output_vector, index_seed)

    with timed.step('.sparse'):
        timed.parameters(results=results_count,
                         elements=elements_count, samples=samples)
        parallel_for(downsample_sparse_vectors, results_count)

    output.has_sorted_indices = True
    if eliminate_zeros:
        with timed.step('.eliminate_zeros'):
            utt.unfreeze(output)
            output.eliminate_zeros()
            output.has_canonical_format = True

    return output


def _downsample_dense_matrix(  # pylint: disable=too-many-locals,too-many-statements
    matrix: np.ndarray,
    axis: int,
    samples: int,
    inplace: bool,
    random_seed: int
) -> np.ndarray:
    if inplace:
        output = matrix
    elif axis == 0:
        output = np.empty(matrix.shape, dtype=matrix.dtype, order='F')
    else:
        output = np.empty(matrix.shape, dtype=matrix.dtype, order='C')

    if axis == 0:
        sample_input_vector = matrix[:, 0]
        sample_output_vector = output[:, 0]
    else:
        sample_input_vector = matrix[0, :]
        sample_output_vector = output[0, :]

    input_is_contiguous = \
        sample_input_vector.flags.c_contiguous and sample_input_vector.flags.f_contiguous
    if not input_is_contiguous:
        downsampling_dense_input_matrix_of_inefficient_format = \
            'downsampling axis: %s ' \
            'of a dense input matrix with inefficient strides: %s' \
            % (axis, matrix.strides)
        warn(downsampling_dense_input_matrix_of_inefficient_format)

    output_is_contiguous = \
        sample_output_vector.flags.c_contiguous and sample_output_vector.flags.f_contiguous
    if not inplace and not output_is_contiguous:
        downsampling_dense_output_matrix_of_inefficient_format = \
            'downsampling axis: %s ' \
            'of a dense output matrix with inefficient strides: %s' \
            % (axis, output.strides)
        warn(downsampling_dense_output_matrix_of_inefficient_format)

    shared_storage = SharedStorage()

    elements_count = matrix.shape[axis]
    tmp_size = downsample_tmp_size(elements_count)

    def downsample_dense_vectors(indices: range) -> None:
        for index in indices:
            if axis == 0:
                input_vector = matrix[:, index]
            else:
                input_vector = matrix[index, :]
            input_array = utt.to_contiguous(input_vector)

            if not output_is_contiguous:
                output_vector = \
                    shared_storage.get_private('output',
                                               make=lambda: np.empty(elements_count,
                                                                     dtype=output.dtype))
            elif axis == 0:
                output_vector = output[:, index]
            else:
                output_vector = output[index, :]
            output_array = utt.to_1d_array(output_vector, copy=None)

            tmp = shared_storage.get_private('tmp',
                                             make=lambda: np.empty(tmp_size,
                                                                   dtype=matrix.dtype))

            if random_seed != 0:
                index_seed = random_seed + index
            else:
                index_seed = 0

            _downsample_array(input_array, samples, tmp,
                              output_array, index_seed)

            if output_is_contiguous:
                continue

            if axis == 0:
                output[:, index] = output_array
            else:
                output[index, :] = output_array

    results_count = matrix.shape[1 - axis]

    with timed.step('.dense'):
        timed.parameters(results=results_count,
                         elements=elements_count, samples=samples)
        parallel_for(downsample_dense_vectors, results_count)

    return output


@timed.call()
@utd.expand_doc(data_types=','.join(['``%s``' % data_type for data_type in utt.DATA_TYPES]))
def downsample_array(
    array: np.ndarray,
    samples: int,
    *,
    tmp: Optional[np.ndarray] = None,
    output: Optional[np.ndarray] = None,
    random_seed: int = 0
) -> None:
    '''
    Downsample a vector of sample counters.

    **Input**

    * A Numpy ``array`` containing non-negative integer sample counts.

    * A desired total number of ``samples``.

    * An optional temporary storage Numpy array minimize allocations for multiple invocations.

    * An optional Numpy array ``output`` to hold the results (otherwise, the input is overwritten).

    * An optional random seed to make the operation replicable.

    The arrays may have any of the data types: {data_types}.

    **Operation**

    If the total number of samples (sum of the array) is not higher than the required number of
    samples, the output is identical to the input.

    Otherwise, treat the input as if it was a set where each index appeared its value number of
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
    timed.parameters(elements=array.size, samples=samples)
    array = utt.unpandas(array)
    return _downsample_array(array, samples, tmp, output, random_seed)


def _downsample_array(
    array: np.ndarray,
    samples: int,
    tmp: Optional[np.ndarray] = None,
    output: Optional[np.ndarray] = None,
    random_seed: int = 0
) -> None:
    assert array.ndim == 1

    if tmp is None:
        tmp = np.empty(downsample_tmp_size(array.size), dtype='int32')
    else:
        assert tmp.size >= downsample_tmp_size(array.size)

    if output is None:
        output = array
    else:
        assert output.shape == array.shape

    extension_name = \
        'downsample_%s_t_%s_t_%s_t' % (array.dtype, tmp.dtype, output.dtype)
    extension = getattr(xt, extension_name)
    extension(array, tmp, output, samples, random_seed)


def downsample_tmp_size(size: int) -> int:
    '''
    Return the size of the temporary array needed to ``downsample`` an array of the specified size.
    '''
    return xt.downsample_tmp_size(size)


@timed.call()
def max_axis(matrix: utt.Matrix, *, axis: int) -> np.ndarray:
    '''
    Compute the maximal element per row (``axis`` = 1) or column (``axis`` = 0) of some ``matrix``.
    '''
    if sparse.issparse(matrix):
        return _reduce_matrix(matrix, axis, lambda matrix: matrix.max(axis=axis))
    return _reduce_matrix(matrix, axis, lambda matrix: np.amax(matrix, axis=axis))


@timed.call()
def min_axis(matrix: utt.Matrix, *, axis: int) -> np.ndarray:
    '''
    Compute the minimal element per row (``axis`` = 1) or column (``axis`` = 0) of some ``matrix``.
    '''
    if sparse.issparse(matrix):
        return _reduce_matrix(matrix, axis, lambda matrix: matrix.min(axis=axis))
    return _reduce_matrix(matrix, axis, lambda matrix: np.amin(matrix, axis=axis))


@timed.call()
def nnz_axis(matrix: utt.Matrix, *, axis: int) -> np.ndarray:
    '''
    Compute the number of non-zero elements per row (``axis`` = 1) or column (``axis`` = 0) of some
    ``matrix``.
    '''
    if sparse.issparse(matrix):
        return _reduce_matrix(matrix, axis, lambda matrix: matrix.getnnz(axis=axis))
    return _reduce_matrix(matrix, axis, lambda matrix: np.count_nonzero(matrix, axis=axis))


@timed.call()
def sum_axis(matrix: utt.Matrix, *, axis: int) -> np.ndarray:
    '''
    Compute the total of the values per row (``axis`` = 1) or column (``axis`` = 0) of some
    ``matrix``.
    '''
    return _reduce_matrix(matrix, axis, lambda matrix: matrix.sum(axis=axis))


@timed.call()
def sum_squared_axis(matrix: utt.Matrix, *, axis: int) -> np.ndarray:
    '''
    Compute the total of the squared values per row (``axis`` = 1) or column (``axis`` = 0) of some
    ``matrix``.

    .. todo::

        This implementation allocates and frees a complete copy of the matrix (to hold the squared
        values). An implementation that directly squares-and-adds the elements would be more
        efficient.
    '''
    if sparse.issparse(matrix):
        return _reduce_matrix(matrix, axis, lambda matrix: matrix.multiply(matrix).sum(axis=axis))

    return _reduce_matrix(matrix, axis, lambda matrix: np.square(matrix).sum(axis=axis))


def _reduce_matrix(
    matrix: utt.Matrix,
    axis: int,
    reducer: Callable,
) -> np.ndarray:
    assert matrix.ndim == 2
    assert 0 <= axis <= 1

    results_count = matrix.shape[1 - axis]

    if sparse.issparse(matrix):
        step = '.sparse'
        elements_count = matrix.nnz / results_count
        axis_format = ['csc', 'csr'][axis]
        if matrix.getformat() != axis_format:
            reducing_sparse_matrix_of_inefficient_format = \
                'reducing axis: %s ' \
                'of a sparse matrix in the format: %s ' \
                'instead of the efficient format: %s' \
                % (axis, matrix.getformat(), axis_format)
            warn(reducing_sparse_matrix_of_inefficient_format)
    else:
        step = '.dense'
        matrix = utt.unpandas(matrix)
        elements_count = matrix.shape[axis]
        axis_flag = [matrix.flags.f_contiguous,
                     matrix.flags.c_contiguous][axis]
        if not axis_flag:
            reducing_dense_matrix_of_inefficient_format = \
                'reducing axis: %s ' \
                'of a dense matrix with layout: %s ' \
                'instead of the efficient layout: %s' \
                % (axis,
                   'f_contiguous' if matrix.flags.f_contiguous
                   else 'c_contiguous' if matrix.flags.c_contiguous
                   else 'non-contiguous',
                   ['f_contiguous', 'c_contiguous'][axis])
            warn(reducing_dense_matrix_of_inefficient_format)

    with timed.step(step):
        timed.parameters(results=results_count, elements=elements_count)
        return utt.to_1d_array(reducer(matrix))


@timed.call()
def bincount_array(
    array: np.ndarray,
    *,
    minlength: int = 0,
) -> np.ndarray:
    '''
    Count the number of occurrences of each value in an ``array``.

    This is identical to Numpy's ``bincount``.

    .. todo::

        This isn't lightning-fast, and seems to be the builtin operation most likely to benefit from
        parallelization.
    '''
    array = utt.unpandas(array)
    result = np.bincount(array, minlength=minlength)
    timed.parameters(size=array.size, bins=result.size)
    return result


ROLLING_FUNCTIONS = {
    'mean': pd.core.window.Rolling.mean,
    'median': pd.core.window.Rolling.median,
    'std': pd.core.window.Rolling.std,
    'var': pd.core.window.Rolling.var,
}


@utd.expand_doc(functions=', '.join(['``%s``' % function for function in ROLLING_FUNCTIONS]))
def sliding_window_function(
    array: np.ndarray,
    *,
    function: str,
    window_size: int,
    order_by: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return an array of the same size as the input ``array``, where each entry is the result of
    applying the ``function`` (one of {functions}) to a sliding window of size ``window_size``
    centered on the matching array entry.

    If ``order_by`` is specified, the ``array`` is first sorted by this order, and the end result is
    unsorted back to the original order. That is, the sliding window centered at each position will
    contain the ``window_size`` of entries which have the nearest ``order_by`` values to the center
    entry.

    .. note::

        The window size must be an odd positive integer. If an even value is specified, it is
        automatically increased by one.
    """
    array = utt.unpandas(array)

    if window_size % 2 == 0:
        window_size += 1

    half_window_size = (window_size - 1) // 2

    if order_by is not None:
        assert array.size == order_by.size
        order_indices = np.argsort(order_by)
        reverse_order_indices = np.argsort(order_indices)
    else:
        reverse_order_indices = order_indices = np.arange(array.size)

    minimal_index = order_indices[0]
    maximal_index = order_indices[-1]

    extended_order_indices = np.concatenate([  #
        np.repeat(minimal_index, half_window_size),
        order_indices,
        np.repeat(maximal_index, half_window_size),
    ])

    extended_series = pd.Series(array[extended_order_indices])
    rolling_windows = extended_series.rolling(window_size)

    compute = ROLLING_FUNCTIONS[function]
    computed = compute(rolling_windows).values
    reordered = computed[window_size - 1:]
    assert reordered.size == array.size

    if order_by is not None:
        reordered = reordered[reverse_order_indices]

    return reordered
