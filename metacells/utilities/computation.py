'''
Utilities for performing efficient uniform computations.

Most of the functions defined here are thin wrappers around builtin numpy functions. However, they
are provided with a uniform interface that works for both sparse and dense data. This allows safely
passing them to functions such as :py:func:`metacells.utilities.preparation.get_per_obs` etc.

All the functions here (optionally) collect timing information using
:py:mod:`metacells.utilities.timing`, to make it easier to locate the performance bottleneck of the
analysis pipeline.
'''

import re
from re import Pattern
from typing import Callable, Collection, Optional, TypeVar, Union
from warnings import warn

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

import metacells.extensions as xt  # type: ignore
import metacells.utilities.documentation as utd
import metacells.utilities.timing as utm
import metacells.utilities.typing as utt
from metacells.utilities.threading import SharedStorage, parallel_for

__all__ = [
    'to_layout',
    'relayout_compressed',

    'corrcoef',

    'log_data',

    'max_per',
    'min_per',
    'nnz_per',
    'sum_per',
    'sum_squared_per',

    'bincount_vector',

    'downsample_matrix',
    'downsample_vector',
    'downsample_tmp_size',

    'sliding_window_function',
    'regex_matches_mask',
]


@utm.timed_call()
@utd.expand_doc()
def to_layout(  # pylint: disable=too-many-return-statements
    matrix: utt.Matrix,
    layout: str,
    *,
    symmetric: bool = False
) -> utt.ProperMatrix:
    '''
    Re-layout a matrix for efficient axis slicing/processing.

    That is, if ``layout`` is ``column_major``, re-layout the matrix for efficient per-column
    (variable, gene) slicing/processing. For sparse matrices, this is ``csc`` format; for dense
    matrices, this is Fortran (column-major) format.

    Similarly, if ``layout`` is ``row_major``, re-layout the matrix for efficient per-row
    (observation, cell) slicing/processing. For sparse matrices, this is ``csr`` format; for dense
    matrices, this is C (row-major) format.

    If the matrix is already in the correct layout, it is returned as-is.

    If the matrix is ``symmetric`` (default: {symmetric}), it must be square and is assumed to be
    equal to its own transpose. This allows converting it from one layout to another using the
    efficient (essentially zero-cost) transpose operation.

    Otherwise, a new copy is created. This is a costly operation as it needs to move all the data.
    However, it makes the following processing much more efficient, so it is typically a net
    performance gain overall.
    '''
    assert layout in utt.LAYOUT_OF_AXIS

    proper, dense, compressed = utt.to_proper_matrices(matrix, layout=layout)

    if utt.is_layout(proper, layout):
        return proper

    if symmetric:
        assert proper.shape[0] == proper.shape[1]
        proper = proper.transpose()
        assert utt.is_layout(proper, layout)
        return proper

    is_frozen = utt.frozen(proper)

    if dense is not None:
        order = utt.DENSE_FAST_FLAG[layout][0]
        with utm.timed_step('ravel'):
            utm.timed_parameters(rows=dense.shape[0], columns=dense.shape[1])
            result = np.reshape(np.ravel(dense, order=order),
                                dense.shape, order=order)

    else:
        assert compressed is not None
        to_format = utt.SPARSE_FAST_FORMAT[layout]
        from_format = utt.SPARSE_SLOW_FORMAT[layout]
        assert compressed.getformat() == from_format
        result = relayout_compressed(compressed)
        assert result.getformat() == to_format

    if is_frozen:
        utt.freeze(result)
    return result


@utm.timed_call()
def relayout_compressed(matrix: utt.CompressedMatrix) -> utt.CompressedMatrix:
    '''
    Efficient parallel conversion of a CSR/CSC ``matrix`` to a CSC/CSR matrix.
    '''
    compressed = utt.CompressedMatrix.be(matrix)

    axis = ('csr', 'csc').index(compressed.getformat())
    _output_elements_count = matrix_bands_count = compressed.shape[axis]
    output_bands_count = matrix_elements_count = compressed.shape[1 - axis]

    nnz_elements_of_output_bands = \
        bincount_vector(compressed.indices, minlength=matrix_elements_count)

    output_indptr = np.empty(output_bands_count + 1,
                             dtype=compressed.indptr.dtype)
    output_indptr[0:2] = 0
    with utm.timed_step('cumsum'):
        utm.timed_parameters(elements=nnz_elements_of_output_bands.size - 1)
        np.cumsum(nnz_elements_of_output_bands[:-1], out=output_indptr[2:])

    output_indices = np.empty(compressed.indices.size,
                              dtype=compressed.indices.dtype)
    output_data = np.empty(compressed.nnz, dtype=compressed.data.dtype)

    extension_name = 'collect_compressed_%s_t_%s_t_%s_t' \
        % (compressed.data.dtype, compressed.indices.dtype, compressed.indptr.dtype)
    extension = getattr(xt, extension_name)

    def collect_compressed(matrix_band_indices: range) -> None:
        extension(matrix_band_indices.start, matrix_band_indices.stop,
                  compressed.data, compressed.indices, compressed.indptr,
                  output_data, output_indices, output_indptr[1:])

    with utm.timed_step('.collect_compressed'):
        utm.timed_parameters(results=output_bands_count,
                             elements=matrix_bands_count)
        parallel_for(collect_compressed, matrix_bands_count)

    assert output_indptr[-1] == compressed.indptr[-1]

    constructor = (sp.csr_matrix, sp.csc_matrix)[1 - axis]
    compressed = constructor((output_data, output_indices,
                              output_indptr), shape=compressed.shape)
    compressed.has_sorted_indices = True
    compressed.has_canonical_format = True
    return compressed


@utm.timed_call()
@utd.expand_doc()
def corrcoef(
    matrix: utt.Matrix,
    *,
    per: Optional[str] = 'row',
) -> utt.DenseMatrix:
    '''
    Similar to for ``np.corrcoef``, but also works for a sparse ``matrix``.

    If ``per`` (default: {per}) is ``None``, the matrix must be square and is assumed to be
    symmetric, so the most efficient direction is used based on the matrix layout. Otherwise it must
    be one of ``row`` or ``column``.

    .. todo::

        This always uses the dense implementation. Need to investigate whether a sparse
        implementation may be faster.

    .. note::

        The result is always dense, as even for sparse data, the correlation is rarely exactly zero.
    '''
    dense = utt.to_dense_matrix(matrix)
    layout = utt.matrix_layout(dense)

    if per is None:
        assert dense.shape[0] == dense.shape[1]
        if layout is None:
            per = 'row'
        else:
            per = utt.PER_OF_AXIS[utt.LAYOUT_OF_AXIS.index(layout)]

    axis = utt.PER_OF_AXIS.index(per)
    fast_layout = utt.LAYOUT_OF_AXIS[axis]
    if layout != fast_layout:
        correlating_dense_input_matrix_of_inefficient_format = \
            'correlating %ss of a matrix with inefficient strides: %s' \
            % (per, dense.strides)
        warn(correlating_dense_input_matrix_of_inefficient_format)

    utm.timed_parameters(results=dense.shape[axis],
                         elements=dense.shape[1 - axis])
    return np.corrcoef(dense, rowvar=per == 'row')


@utd.expand_doc()
def log_data(
    shaped: utt.Shaped,
    *,
    base: Optional[float] = None,
    normalization: float = 0,
) -> utt.DenseShaped:
    '''
    Return the log of the values in the ``shaped`` data.

    If ``base`` is specified (default: {base}), use this base log. Otherwise, use the natural
    logarithm.

    The ``normalization`` (default: {normalization}) specifies how to deal with zeros in the data:

    * If it is zero, an input zeros will become an output ``NaN``.

    * If it is positive, it is added to the input before computing the log.

    * If it is negative, input zeros will become log(minimal positive value) + normalization,
      that is, the zeros will be given a value this much smaller than the minimal "real"
      log value.

    .. note::

        The result is always dense, as even for sparse data, the log is rarely zero.
    '''
    dense = utt.to_dense(shaped, copy=True)

    if normalization > 0:
        dense += normalization
    else:
        where = dense > 0
        if normalization < 0:
            dense[~where] = np.amin(dense[where])

    if base is None:
        log_function = np.log
        rebase = False
    elif base == 2:
        log_function = np.log2
        rebase = False
    elif base == 10:
        log_function = np.log10
        rebase = False
    else:
        assert base > 0
        log_function = np.log
        rebase = True

    if normalization == 0:
        log_function(dense, out=dense, where=where)
        dense[~where] = None
    else:
        log_function(dense, out=dense)

    if rebase:
        dense /= np.log(base)

    if normalization < 0:
        dense[~where] += normalization

    return dense


@utm.timed_call()
@utd.expand_doc()
def downsample_matrix(
    matrix: utt.Matrix,
    *,
    per: str,
    samples: int,
    eliminate_zeros: bool = True,
    inplace: bool = False,
    random_seed: int = 0,
) -> utt.Matrix:
    '''
    Downsample the data ``per`` (one of ``row`` and ``column``) such that the sum of each one
    becomes ``samples``.

    If the matrix is sparse, and ``eliminate_zeros`` (default: {eliminate_zeros}), then perform a
    final phase of eliminating leftover zero values from the compressed format. This means the
    result will be in "canonical format" so further ``scipy`` sparse operations on it will be
    faster.

    If ``inplace`` (default: {inplace}), modify the matrix, otherwise, return a modified copy.

    A non-zero ``random_seed`` (default: {random_seed}) can be provided to make the operation
    replicable.
    '''
    assert per in utt.PER_OF_AXIS
    assert samples > 0

    _, dense, compressed = utt.to_proper_matrices(matrix)

    if dense is not None:
        return _downsample_dense_matrix(dense, per, samples, inplace, random_seed)

    assert compressed is not None
    return _downsample_compressed_matrix(compressed, per, samples,
                                         eliminate_zeros, inplace, random_seed)


def _downsample_dense_matrix(  # pylint: disable=too-many-locals,too-many-statements
    matrix: utt.DenseMatrix,
    per: str,
    samples: int,
    inplace: bool,
    random_seed: int
) -> utt.DenseMatrix:
    if inplace:
        output = matrix
    elif per == 'row':
        output = np.empty(matrix.shape, dtype=matrix.dtype, order='C')
    else:
        output = np.empty(matrix.shape, dtype=matrix.dtype, order='F')

    if per == 'row':
        sample_input_vector = matrix[0, :]
        sample_output_vector = output[0, :]
    else:
        sample_input_vector = matrix[:, 0]
        sample_output_vector = output[:, 0]

    input_is_contiguous = \
        sample_input_vector.flags.c_contiguous and sample_input_vector.flags.f_contiguous
    if not input_is_contiguous:
        downsampling_dense_input_matrix_of_inefficient_format = \
            'downsampling per-%s ' \
            'of a matrix input matrix with inefficient strides: %s' \
            % (per, matrix.strides)
        warn(downsampling_dense_input_matrix_of_inefficient_format)

    output_is_contiguous = \
        sample_output_vector.flags.c_contiguous and sample_output_vector.flags.f_contiguous
    if not inplace and not output_is_contiguous:
        downsampling_dense_output_matrix_of_inefficient_format = \
            'downsampling per-%s ' \
            'of a matrix output matrix with inefficient strides: %s' \
            % (per, output.strides)
        warn(downsampling_dense_output_matrix_of_inefficient_format)

    shared_storage = SharedStorage()

    axis = utt.PER_OF_AXIS.index(per)
    results_count = matrix.shape[axis]
    elements_count = matrix.shape[1 - axis]
    tmp_size = downsample_tmp_size(elements_count)

    def downsample_dense_vectors(indices: range) -> None:
        for index in indices:
            if per == 'row':
                input_vector = matrix[index, :]
            else:
                input_vector = matrix[:, index]
            input_array = utt.to_contiguous(input_vector)

            if not output_is_contiguous:
                output_vector = \
                    shared_storage.get_private('output',
                                               make=lambda: np.empty(elements_count,
                                                                     dtype=output.dtype))
            elif per == 'row':
                output_vector = output[index, :]
            else:
                output_vector = output[:, index]
            output_array = utt.to_dense_vector(output_vector, copy=None)

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

            if per == 'row':
                output[index, :] = output_array
            else:
                output[:, index] = output_array

    with utm.timed_step('.dense'):
        utm.timed_parameters(results=results_count,
                             elements=elements_count, samples=samples)
        parallel_for(downsample_dense_vectors, results_count)

    return output


def _downsample_compressed_matrix(  # pylint: disable=too-many-locals
    matrix: utt.CompressedMatrix,
    per: str,
    samples: int,
    eliminate_zeros: bool,
    inplace: bool,
    random_seed: int
) -> utt.CompressedMatrix:
    axis = utt.PER_OF_AXIS.index(per)
    results_count = matrix.shape[axis]
    elements_count = matrix.shape[1 - axis]

    axis_format = ('csr', 'csc')[axis]
    if matrix.getformat() != axis_format:
        raise NotImplementedError('downsample per-%s '
                                  'of a sparse matrix in the format: %s '
                                  'instead of the efficient format: %s'
                                  % (per, matrix.getformat(), axis_format))

    if inplace:
        output = matrix
    else:
        constructor = (sp.csr_matrix, sp.csc_matrix)[axis]
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
                                                                 dtype=output.dtype))

            if random_seed != 0:
                index_seed = random_seed + index
            else:
                index_seed = 0

            _downsample_array(input_vector, samples, tmp,
                              output_vector, index_seed)

    with utm.timed_step('.sparse'):
        utm.timed_parameters(results=results_count,
                             elements=elements_count, samples=samples)
        parallel_for(downsample_sparse_vectors, results_count)

    output.has_sorted_indices = True
    if eliminate_zeros:
        with utm.timed_step('.eliminate_zeros'):
            utt.unfreeze(output)
            output.eliminate_zeros()
            output.has_canonical_format = True

    return output


@utm.timed_call()
@utd.expand_doc(data_types=','.join(['``%s``' % data_type for data_type in utt.CPP_DATA_TYPES]))
def downsample_vector(
    vector: utt.Vector,
    samples: int,
    *,
    tmp: Optional[np.ndarray] = None,
    output: Optional[np.ndarray] = None,
    random_seed: int = 0
) -> None:
    '''
    Downsample a vector of sample counters.

    **Input**

    * A numpy ``array`` containing non-negative integer sample counts.

    * A desired total number of ``samples``.

    * An optional ``tmp`` storage numpy array to minimize allocations for multiple invocations.

    * An optional numpy array ``output`` to hold the results (otherwise, the input is overwritten).

    * An optional non-zero ``random_seed`` (default: {random_seed}) to make the operation
      replicable.

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
    proper = utt.to_proper_vector(vector)
    utm.timed_parameters(elements=proper.size, samples=samples)
    return _downsample_array(proper, samples, tmp, output, random_seed)


def _downsample_array(
    array: utt.DenseVector,
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


@utm.timed_call()
def max_per(matrix: utt.Matrix, *, per: str) -> np.ndarray:
    '''
    Compute the maximal value ``per`` (``row`` or ``column``) of some ``matrix``.
    '''
    axis = utt.PER_OF_AXIS.index(per)

    sparse = utt.SparseMatrix.maybe(matrix)
    if sparse is not None:
        return _reduce_matrix(sparse, per, lambda sparse: sparse.max(axis=1 - axis))

    dense = utt.DenseMatrix.be(utt.to_proper_matrix(matrix))
    return _reduce_matrix(dense, per, lambda dense: np.amax(dense, axis=1 - axis))


@utm.timed_call()
def min_per(matrix: utt.Matrix, *, per: str) -> np.ndarray:
    '''
    Compute the minimal value ``per`` (``row`` or ``column``) of some ``matrix``.
    '''
    axis = utt.PER_OF_AXIS.index(per)

    sparse = utt.SparseMatrix.maybe(matrix)
    if sparse is not None:
        return _reduce_matrix(sparse, per, lambda sparse: sparse.min(axis=1 - axis))

    dense = utt.DenseMatrix.be(utt.to_proper_matrix(matrix))
    return _reduce_matrix(dense, per, lambda dense: np.amin(dense, axis=1 - axis))


@utm.timed_call()
def nnz_per(matrix: utt.Matrix, *, per: str) -> np.ndarray:
    '''
    Compute the number of non-zero values ``per`` (``row`` or ``column``) of some ``matrix``.
    '''
    axis = utt.PER_OF_AXIS.index(per)

    sparse = utt.SparseMatrix.maybe(matrix)
    if sparse is not None:
        return _reduce_matrix(sparse, per, lambda sparse: sparse.getnnz(axis=1 - axis))

    dense = utt.DenseMatrix.be(utt.to_proper_matrix(matrix))
    return _reduce_matrix(dense, per, lambda dense: np.count_nonzero(dense, axis=1 - axis))


@utm.timed_call()
def sum_per(matrix: utt.Matrix, *, per: str) -> np.ndarray:
    '''
    Compute the total of the values ``per`` (``row`` or ``column``) of some ``matrix``.
    '''
    axis = utt.PER_OF_AXIS.index(per)

    sparse = utt.SparseMatrix.maybe(matrix)
    if sparse is not None:
        return _reduce_matrix(sparse, per, lambda sparse: sparse.sum(axis=1 - axis))

    dense = utt.DenseMatrix.be(utt.to_proper_matrix(matrix))
    return _reduce_matrix(dense, per, lambda dense: np.sum(dense, axis=1 - axis))


@utm.timed_call()
def sum_squared_per(matrix: utt.Matrix, *, per: str) -> np.ndarray:
    '''
    Compute the total of the squared values ``per`` (``row`` or ``column``) of some ``matrix``.

    .. todo::

        This implementation allocates and frees a complete copy of the matrix (to hold the squared
        values). An implementation that directly squares-and-adds the elements would be more
        efficient.
    '''
    axis = utt.PER_OF_AXIS.index(per)

    sparse = utt.SparseMatrix.maybe(matrix)
    if sparse is not None:
        return \
            _reduce_matrix(sparse, per,
                           lambda sparse:
                           sparse.multiply(sparse).sum(axis=1 - axis))

    dense = utt.DenseMatrix.be(utt.to_proper_matrix(matrix))
    return _reduce_matrix(dense, per,
                          lambda dense: np.square(dense).sum(axis=1 - axis))


M = TypeVar('M', bound=utt.Matrix)


def _reduce_matrix(
    matrix: M,
    per: str,
    reducer: Callable[[M], utt.DenseVector],
) -> utt.DenseVector:
    assert matrix.ndim == 2
    axis = utt.PER_OF_AXIS.index(per)
    results_count = matrix.shape[1 - axis]

    if utt.SparseMatrix.am(matrix):
        timed_step = '.sparse'
        elements_count: float = matrix.shape[axis]
    else:
        _, dense, compressed = \
            utt.to_proper_matrices(matrix, layout=utt.LAYOUT_OF_AXIS[axis])

        if dense is not None:
            elements_count = dense.shape[axis]
            axis_flag = (dense.flags.c_contiguous,
                         dense.flags.f_contiguous)[axis]
            if axis_flag:
                timed_step = '.dense-efficient'
            else:
                timed_step = '.dense-inefficient'
                reducing_dense_matrix_of_inefficient_format = \
                    'reducing axis: %s ' \
                    'of a dense matrix with layout: %s ' \
                    'instead of the efficient layout: %s' \
                    % (axis,
                       'f_contiguous' if dense.flags.f_contiguous
                       else 'c_contiguous' if dense.flags.c_contiguous
                       else 'non-contiguous',
                       ['f_contiguous', 'c_contiguous'][axis])
                warn(reducing_dense_matrix_of_inefficient_format)

        else:
            assert compressed is not None
            elements_count = compressed.nnz / results_count
            axis_format = ('csr', 'csc')[axis]
            if compressed.getformat() == axis_format:
                timed_step = '.compressed-efficient'
            else:
                timed_step = '.compressed-inefficient'
                reducing_sparse_matrix_of_inefficient_format = \
                    'reducing axis: %s ' \
                    'of a sparse matrix in the format: %s ' \
                    'instead of the efficient format: %s' \
                    % (axis, compressed.getformat(), axis_format)
                warn(reducing_sparse_matrix_of_inefficient_format)

    with utm.timed_step(timed_step):
        utm.timed_parameters(results=results_count, elements=elements_count)
        return reducer(matrix)


@utm.timed_call()
def bincount_vector(
    vector: utt.Vector,
    *,
    minlength: int = 0,
) -> np.ndarray:
    '''
    Drop-in replacement for ``np.bincount``, which is timed and also works on pandas data.

    .. todo::

        This isn't lightning-fast, and seems to be the builtin operation most likely to benefit from
        parallelization.
    '''
    proper = utt.to_proper_vector(vector)
    array = utt.DenseVector.be(proper)
    result = np.bincount(array, minlength=minlength)
    utm.timed_parameters(size=array.size, bins=result.size)
    return result


ROLLING_FUNCTIONS = {
    'mean': pd.core.window.Rolling.mean,
    'median': pd.core.window.Rolling.median,
    'std': pd.core.window.Rolling.std,
    'var': pd.core.window.Rolling.var,
}


@utd.expand_doc(functions=', '.join(['``%s``' % function for function in ROLLING_FUNCTIONS]))
def sliding_window_function(  # pylint: disable=too-many-locals
    vector: utt.Vector,
    *,
    function: str,
    window_size: int,
    order_by: Optional[np.ndarray] = None,
) -> utt.DenseVector:
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
    proper = utt.to_proper_vector(vector)
    array = utt.DenseVector.be(proper)

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


def regex_matches_mask(
    pattern: Union[str, Pattern],
    strings: Collection[str]
) -> utt.DenseVector:
    '''
    Given a collection of ``strings``, return a numpy boolean mask specifying which of them match
    the given regular expression ``pattern``.
    '''
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    assert isinstance(pattern, Pattern)
    return np.array([bool(pattern.match(string)) for string in strings], dtype='bool')
