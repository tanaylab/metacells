'''
Computation
-----------

Most of the functions defined here are thin wrappers around builtin numpy or scipy functions.
However, they are provided with a uniform interface that works for both sparse and dense data. This
allows safely passing them to functions such as
:py:func:`metacells.preprocessing.common.get_per_obs` etc.

All the functions here (optionally) collect timing information using
:py:mod:`metacells.utilities.timing`, to make it easier to locate the performance bottleneck of the
analysis pipeline.
'''

import re
from re import Pattern
from typing import (Any, Callable, Collection, List, Optional, Tuple, TypeVar,
                    Union, overload)
from warnings import warn

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

import metacells.extensions as xt  # type: ignore
import metacells.utilities.documentation as utd
import metacells.utilities.timing as utm
import metacells.utilities.typing as utt

__all__ = [
    'to_layout',
    'relayout_compressed',
    'sort_compressed_indices',

    'corrcoef',
    'logistics',

    'log_data',

    'mean_per',
    'nanmean_per',
    'max_per',
    'nanmax_per',
    'min_per',
    'nanmin_per',
    'nnz_per',
    'sum_per',
    'sum_squared_per',
    'rank_per',
    'quantile_per',
    'nanquantile_per',

    'bincount_vector',
    'most_frequent',

    'downsample_matrix',
    'downsample_vector',

    'matrix_rows_auroc',

    'sliding_window_function',
    'patterns_matches',
    'compress_indices',
    'bin_pack',
    'bin_fill',
    'sum_groups',
    'shuffle_matrix',
]


@overload
def to_layout(
    matrix: utt.CompressedMatrix,
    layout: str,
    *,
    symmetric: bool = False
) -> utt.CompressedMatrix: ...


@overload
def to_layout(
    matrix: utt.DenseMatrix,
    layout: str,
    *,
    symmetric: bool = False
) -> utt.DenseMatrix: ...


@overload
def to_layout(
    matrix: utt.ImproperMatrix,
    layout: str,
    *,
    symmetric: bool = False
) -> utt.ProperMatrix: ...


@utm.timed_call()
@utd.expand_doc()
def to_layout(
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

    proper, dense, compressed = \
        utt.to_proper_matrices(matrix, default_layout=layout)

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
        with utm.timed_step('numpy.ravel'):
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

    if compressed.nnz < 2:
        if axis == 0:
            compressed = compressed.tocsc()
        else:
            compressed = compressed.tocsr()
        compressed.sort_indices()

    else:
        _output_elements_count = matrix_bands_count = compressed.shape[axis]
        output_bands_count = matrix_elements_count = compressed.shape[1 - axis]

        nnz_elements_of_output_bands = \
            bincount_vector(compressed.indices,
                            minlength=matrix_elements_count)
        output_indptr = \
            np.empty(output_bands_count + 1, dtype=compressed.indptr.dtype)
        output_indptr[0:2] = 0
        with utm.timed_step('numpy.cumsum'):
            utm.timed_parameters(elements=nnz_elements_of_output_bands.size
                                 - 1)
            np.cumsum(nnz_elements_of_output_bands[:-1], out=output_indptr[2:])

        output_indices = \
            np.empty(compressed.nnz, dtype=compressed.indices.dtype)
        output_data = np.empty(compressed.nnz, dtype=compressed.data.dtype)

        extension_name = 'collect_compressed_%s_t_%s_t_%s_t' \
            % (compressed.data.dtype,
               compressed.indices.dtype,
               compressed.indptr.dtype)
        extension = getattr(xt, extension_name)

        assert matrix_bands_count == compressed.indptr.size - 1

        with utm.timed_step('extensions.collect_compressed'):
            utm.timed_parameters(results=output_bands_count,
                                 elements=matrix_bands_count)
            assert compressed.indptr[-1] == compressed.data.size
            extension(compressed.data, compressed.indices, compressed.indptr,
                      output_data, output_indices, output_indptr[1:])

        assert output_indptr[-1] == compressed.indptr[-1]

        constructor = (sp.csr_matrix, sp.csc_matrix)[1 - axis]
        compressed = constructor((output_data,
                                  output_indices,
                                  output_indptr),
                                 shape=compressed.shape)

        sort_compressed_indices(compressed, force=True)

    compressed.has_canonical_format = True

    return compressed


def sort_compressed_indices(matrix: utt.CompressedMatrix, force: bool = False) -> None:
    '''
    Efficient parallel sort of indices in a CSR/CSC ``matrix``.

    This will skip sorting a matrix that is marked as sorted, unless ``force`` is specified.
    '''
    assert matrix.getformat() in ('csr', 'csc')
    if matrix.has_sorted_indices and not force:
        return

    matrix_bands_count = matrix.indptr.size - 1
    extension_name = 'sort_compressed_indices_%s_t_%s_t_%s_t' \
        % (matrix.data.dtype, matrix.indices.dtype, matrix.indptr.dtype)
    extension = getattr(xt, extension_name)

    with utm.timed_step('extensions.sort_compressed_indices'):
        utm.timed_parameters(results=matrix_bands_count,
                             elements=matrix.nnz / matrix_bands_count)
        extension(matrix.data, matrix.indices, matrix.indptr, matrix.shape[1])

    matrix.has_sorted_indices = True


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

        The :py:func:`corrcoef` always uses the dense implementation. Possibly a sparse
        implementation might be faster.

    .. todo::

        The results of ``corrcoef`` are not replicable between runs if the matrix contains
        non-integer values, because it uses numpy's ``matmul`` function, which produces slightly
        different results depending on the number of CPUs used to compute the answer. There doesn't
        seem to be a ``matmul_det`` variant which produces reproducible results.

    .. note::

        The result is always dense, as even for sparse data, the correlation is rarely exactly zero.

    .. note::

        This replicates the implementation from numpy, since the numpy implementation insists on
        making a copy of the matrix, and doing the computations in double precision, which more than
        doubles our memory usage, and slows down the computation with no significant benefit to our
        final results.

    .. note::

        We save making copy of the matrix by subtracting the averages in-place and then adding them
        back. This means that the matrix will be slightly perturbed, so computation done on it
        before and after invoking this will give slightly different results.
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

    is_frozen = utt.frozen(dense)
    if is_frozen:
        utt.unfreeze(dense)

    # Replication of numpy code:
    X = dense if per == 'row' else dense.T
    row_averages = np.average(X, axis=1)
    X -= row_averages[:, None]
    result = np.matmul(X, X.T)
    result *= 1 / X.shape[1]
    X += row_averages[:, None]
    diagonal = np.diag(result)
    stddev = np.sqrt(diagonal)
    result /= stddev[:, None]
    result /= stddev[None, :]
    np.clip(result, -1, 1, out=result)

    if is_frozen:
        utt.freeze(dense)
    return result


@utm.timed_call()
@utd.expand_doc()
def logistics(
    matrix: utt.Matrix,
    *,
    location: float = 0.8,
    scale: float = 5,
    per: Optional[str] = 'row',
) -> utt.DenseMatrix:
    '''
    Compute a similarity matrix, similar to for ``np.corrcoef``, but uses the logistics function.

    This computes, for each pair of vectors, the mean of ``1/(1+exp(-scale*(abs(x-y)-location)))``
    of each of the elements. Typically this is applied to the log of the raw data.

    If ``per`` (default: {per}) is ``None``, the matrix must be square and is assumed to be
    symmetric, so the most efficient direction is used based on the matrix layout. Otherwise it must
    be one of ``row`` or ``column``.

    .. todo::

        The :py:func:`logistics` always uses the dense implementation. Possibly a sparse
        implementation might be faster.

    .. note::

        The result is always dense, as even for sparse data, the result is rarely exactly zero.
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
            'logistics of %ss of a matrix with inefficient strides: %s' \
            % (per, dense.strides)
        warn(correlating_dense_input_matrix_of_inefficient_format)

    utm.timed_parameters(results=dense.shape[axis],
                         elements=dense.shape[1 - axis])

    if per == 'column':
        dense = dense.T

    extension_name = 'logistics_dense_matrix_%s_t' % dense.dtype
    result = np.empty((dense.shape[0], dense.shape[0]), dtype='float32')
    extension = getattr(xt, extension_name)
    extension(dense, result, location, scale)
    return result


@utm.timed_call()
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
            dense[~where] = np.min(dense[where])

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


def _downsample_dense_matrix(
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

    assert output.shape == matrix.shape

    axis = utt.PER_OF_AXIS.index(per)
    results_count = matrix.shape[axis]
    elements_count = matrix.shape[1 - axis]

    if per == 'row':
        sample_input_vector = matrix[0, :]
        sample_output_vector = output[0, :]
    else:
        sample_input_vector = matrix[:, 0]
        sample_output_vector = output[:, 0]

    input_is_contiguous = \
        sample_input_vector.flags.c_contiguous and sample_input_vector.flags.f_contiguous
    if not input_is_contiguous:
        raise \
            NotImplementedError('downsampling per-%s '
                                'of a matrix input matrix '
                                'with inefficient strides: %s'
                                % (per, matrix.strides))

    output_is_contiguous = \
        sample_output_vector.flags.c_contiguous and sample_output_vector.flags.f_contiguous
    if not inplace and not output_is_contiguous:
        raise NotImplementedError('downsampling per-%s '
                                  'of a matrix output matrix '
                                  'with inefficient strides: %s'
                                  % (per, output.strides))

    if per == 'column':
        matrix = np.transpose(matrix)
        output = np.transpose(output)

    assert results_count == matrix.shape[0]
    assert elements_count == matrix.shape[1]

    if results_count == 1:
        input_array = utt.to_dense_vector(matrix, copy=None)
        output_array = utt.to_dense_vector(output, copy=None)
        extension_name = 'downsample_array_%s_t_%s_t' \
            % (input_array.dtype, output_array.dtype)
        extension = getattr(xt, extension_name)
        with utm.timed_step('extensions.downsample_array'):
            utm.timed_parameters(elements=input_array.size, samples=samples)
            extension(input_array, output_array, samples, random_seed)
    else:
        extension_name = \
            'downsample_matrix_%s_t_%s_t' % (matrix.dtype, output.dtype)
        extension = getattr(xt, extension_name)
        with utm.timed_step('extensions.downsample_dense_matrix'):
            utm.timed_parameters(results=results_count,
                                 elements=elements_count, samples=samples)
            extension(matrix, output, samples, random_seed)

    if per == 'column':
        output = np.transpose(output)

    assert output.shape == matrix.shape
    return output


def _downsample_compressed_matrix(
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
        assert not utt.frozen(matrix)
        output = matrix
    else:
        constructor = (sp.csr_matrix, sp.csc_matrix)[axis]
        output_data = np.empty_like(matrix.data)
        output = constructor((output_data,
                              np.copy(matrix.indices),
                              np.copy(matrix.indptr)),
                             shape=matrix.shape)
        output.has_sorted_indices = matrix.has_sorted_indices
        output.has_canonical_format = matrix.has_canonical_format

    extension_name = 'downsample_compressed_%s_t_%s_t_%s_t' \
        % (matrix.data.dtype, matrix.indptr.dtype, output.data.dtype)
    extension = getattr(xt, extension_name)

    assert results_count == matrix.indptr.size - 1

    with utm.timed_step('extensions.downsample_sparse_matrix'):
        utm.timed_parameters(results=results_count,
                             elements=elements_count, samples=samples)
        extension(matrix.data, matrix.indptr,
                  output.data, samples, random_seed)

    if eliminate_zeros:
        with utm.timed_step('sparse.eliminate_zeros'):
            utm.timed_parameters(before=output.nnz)
            output.eliminate_zeros()
            utm.timed_parameters(after=output.nnz)

    return output


@utm.timed_call()
@utd.expand_doc(data_types=','.join(['``%s``' % data_type for data_type in utt.CPP_DATA_TYPES]))
def downsample_vector(
    vector: utt.Vector,
    samples: int,
    *,
    output: Optional[utt.DenseVector] = None,
    random_seed: int = 0
) -> None:
    '''
    Downsample a vector of sample counters.

    **Input**

    * A numpy ``array`` containing non-negative integer sample counts.

    * A desired total number of ``samples``.

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
    '''

    if output is None:
        array = utt.to_dense_vector(vector, copy=None)
        output = array
    else:
        array = utt.to_dense_vector(vector)
        assert output.shape == array.shape

    extension_name = 'downsample_array_%s_t_%s_t' % (array.dtype, output.dtype)
    extension = getattr(xt, extension_name)

    with utm.timed_step('extensions.downsample_array'):
        utm.timed_parameters(elements=array.size, samples=samples)
        extension(array, output, samples, random_seed)


@utm.timed_call()
def matrix_rows_auroc(
    matrix: utt.Matrix,
    columns_subset: utt.DenseVector,
    columns_scale: Optional[utt.DenseVector] = None,
) -> utt.Vector:
    '''
    Given a matrix and a subset of the columns, return a vector containing for each row the area
    under the receiver operating characteristic (AUROC) for the row, that is, the probability that a
    random column in the subset would have a higher value in this row than a random column outside
    the subset.

    If ``columns_scale`` is specified, the data is divided by this scale before computing the AUROC.
    '''
    proper, dense, compressed = utt.to_proper_matrices(matrix)
    rows_count, columns_count = proper.shape

    columns_subset = utt.to_dense_vector(columns_subset)
    assert columns_subset.size == columns_count

    if columns_scale is None:
        columns_scale = np.full(columns_count, 1.0, dtype='float32')
    else:
        columns_scale = columns_scale.astype('float32')
        assert columns_scale.size == columns_count

    rows_auroc = np.empty(rows_count, dtype='float64')

    if columns_subset.dtype != 'bool':
        mask: utt.DenseVector = np.full(columns_count, False)
        mask[columns_subset] = True
        columns_subset = mask

    if dense is not None:
        extension_name = 'auroc_dense_matrix_%s_t' % matrix.dtype
        extension = getattr(xt, extension_name)
        extension(dense, columns_subset, columns_scale, rows_auroc)
    else:
        assert compressed is not None
        assert compressed.has_sorted_indices
        extension_name = \
            'auroc_compressed_matrix_%s_t_%s_t_%s_t' \
            % (compressed.data.dtype,
               compressed.indices.dtype,
               compressed.indptr.dtype)
        extension = getattr(xt, extension_name)
        extension(compressed.data, compressed.indices, compressed.indptr,
                  columns_count, columns_subset, columns_scale, rows_auroc)

    return rows_auroc


@utm.timed_call()
def mean_per(matrix: utt.Matrix, *, per: str) -> utt.DenseVector:
    '''
    Compute the mean value ``per`` (``row`` or ``column``) of some ``matrix``.
    '''
    axis = utt.PER_OF_AXIS.index(per)

    sparse = utt.SparseMatrix.maybe(matrix)
    if sparse is not None:
        return _reduce_matrix(sparse, per, lambda sparse: sparse.mean(axis=1 - axis))

    dense = utt.DenseMatrix.be(utt.to_proper_matrix(matrix))
    return _reduce_matrix(dense, per, lambda dense: np.mean(dense, axis=1 - axis))


@utm.timed_call()
def nanmean_per(matrix: utt.DenseMatrix, *, per: str) -> utt.DenseVector:
    '''
    Compute the mean value ``per`` (``row`` or ``column``) of some ``matrix``,
    ignoring ``None`` values, if any.
    '''
    axis = utt.PER_OF_AXIS.index(per)

    dense = utt.DenseMatrix.be(utt.to_proper_matrix(matrix))
    return _reduce_matrix(dense, per, lambda dense: np.nanmean(dense, axis=1 - axis))


@utm.timed_call()
def max_per(matrix: utt.Matrix, *, per: str) -> utt.DenseVector:
    '''
    Compute the maximal value ``per`` (``row`` or ``column``) of some ``matrix``.
    '''
    axis = utt.PER_OF_AXIS.index(per)

    sparse = utt.SparseMatrix.maybe(matrix)
    if sparse is not None:
        return _reduce_matrix(sparse, per, lambda sparse: sparse.max(axis=1 - axis))

    dense = utt.DenseMatrix.be(utt.to_proper_matrix(matrix))
    return _reduce_matrix(dense, per, lambda dense: np.max(dense, axis=1 - axis))


@utm.timed_call()
def nanmax_per(matrix: utt.DenseMatrix, *, per: str) -> utt.DenseVector:
    '''
    Compute the maximal value ``per`` (``row`` or ``column``) of some ``matrix``,
    ignoring ``None`` values, if any.
    '''
    axis = utt.PER_OF_AXIS.index(per)

    dense = utt.DenseMatrix.be(utt.to_proper_matrix(matrix))
    return _reduce_matrix(dense, per, lambda dense: np.nanmax(dense, axis=1 - axis))


@utm.timed_call()
def min_per(matrix: utt.Matrix, *, per: str) -> utt.DenseVector:
    '''
    Compute the minimal value ``per`` (``row`` or ``column``) of some ``matrix``.
    '''
    axis = utt.PER_OF_AXIS.index(per)

    sparse = utt.SparseMatrix.maybe(matrix)
    if sparse is not None:
        return _reduce_matrix(sparse, per, lambda sparse: sparse.min(axis=1 - axis))

    dense = utt.DenseMatrix.be(utt.to_proper_matrix(matrix))
    return _reduce_matrix(dense, per, lambda dense: np.min(dense, axis=1 - axis))


@utm.timed_call()
def nanmin_per(matrix: utt.DenseMatrix, *, per: str) -> utt.DenseVector:
    '''
    Compute the minimal value ``per`` (``row`` or ``column``) of some ``matrix``,
    ignoring ``None`` values, if any.
    '''
    axis = utt.PER_OF_AXIS.index(per)

    sparse = utt.SparseMatrix.maybe(matrix)
    if sparse is not None:
        return _reduce_matrix(sparse, per, lambda sparse: sparse.nanmin(axis=1 - axis))

    dense = utt.DenseMatrix.be(utt.to_proper_matrix(matrix))
    return _reduce_matrix(dense, per, lambda dense: np.nanmin(dense, axis=1 - axis))


@utm.timed_call()
def nnz_per(matrix: utt.Matrix, *, per: str) -> utt.DenseVector:
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
def sum_per(matrix: utt.Matrix, *, per: str) -> utt.DenseVector:
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
def sum_squared_per(matrix: utt.Matrix, *, per: str) -> utt.DenseVector:
    '''
    Compute the total of the squared values ``per`` (``row`` or ``column``) of some ``matrix``.

    .. todo::

        The :py:func:`sum_squared_per` implementation allocates and frees a complete copy of the
        matrix (to hold the squared values). An implementation that directly squares-and-adds the
        elements would be more efficient.
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


@utm.timed_call()
def rank_per(matrix: utt.DenseMatrix, rank: int, *, per: Optional[str]) -> utt.DenseVector:
    '''
    Get the ``rank`` element ``per`` (``row`` or ``column``) of some ``matrix``.

    If ``per`` (default: {per}) is ``None``, the matrix must be square and is assumed to be
    symmetric, so the most efficient direction is used based on the matrix layout. Otherwise it must
    be one of ``row`` or ``column``, and the matrix must be in the appropriate layout (row-major for
    ranking data in each row, column-major for ranking data in each column).
    '''
    if per is None:
        layout = utt.matrix_layout(matrix)
        assert layout is not None
        per = layout[:-6]

    assert per in utt.PER_OF_AXIS
    assert utt.matrix_layout(matrix) == per + '_major'

    if per == 'column':
        matrix = matrix.transpose()

    output = np.empty(matrix.shape[0], dtype=matrix.dtype)
    extension_name = 'rank_matrix_%s_t' % matrix.dtype
    extension = getattr(xt, extension_name)
    extension(matrix, output, rank)
    return output


@utm.timed_call()
def quantile_per(matrix: utt.DenseMatrix, quantile: float, *, per: Optional[str]) -> utt.DenseVector:
    '''
    Get the ``quantile`` element ``per`` (``row`` or ``column``) of some ``matrix``.

    If ``per`` (default: {per}) is ``None``, the matrix must be square and is assumed to be
    symmetric, so the most efficient direction is used based on the matrix layout. Otherwise it must
    be one of ``row`` or ``column``, and the matrix must be in the appropriate layout (row-major for
    ranking data in each row, column-major for ranking data in each column).
    '''
    assert 0 <= quantile <= 1

    if per is None:
        layout = utt.matrix_layout(matrix)
        assert layout is not None
        per = layout[:-6]

    axis = 1 - utt.PER_OF_AXIS.index(per)

    assert utt.matrix_layout(matrix) == per + '_major'
    return np.nanquantile(matrix, quantile, axis)


@utm.timed_call()
def nanquantile_per(matrix: utt.DenseMatrix, quantile: float, *, per: Optional[str]) -> utt.DenseVector:
    '''
    Get the ``quantile`` element ``per`` (``row`` or ``column``) of some ``matrix``, ignoring
    ``None`` values.

    If ``per`` (default: {per}) is ``None``, the matrix must be square and is assumed to be
    symmetric, so the most efficient direction is used based on the matrix layout. Otherwise it must
    be one of ``row`` or ``column``, and the matrix must be in the appropriate layout (row-major for
    ranking data in each row, column-major for ranking data in each column).
    '''
    assert 0 <= quantile <= 1

    if per is None:
        layout = utt.matrix_layout(matrix)
        assert layout is not None
        per = layout[:-6]

    axis = 1 - utt.PER_OF_AXIS.index(per)

    assert utt.is_layout(matrix, per + '_major')
    return np.nanquantile(matrix, quantile, axis)


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
        _, dense, compressed = utt.to_proper_matrices(matrix,
                                                      default_layout=utt.LAYOUT_OF_AXIS[axis])

        if dense is not None:
            elements_count = dense.shape[axis]
            axis_flag = (dense.flags.c_contiguous,
                         dense.flags.f_contiguous)[axis]
            if axis_flag:
                timed_step = '.dense-efficient'
            else:
                timed_step = '.dense-inefficient'
                reducing_dense_matrix_of_inefficient_format = 'reducing axis: %s ' \
                    'of a dense matrix with layout: %s ' \
                    'instead of the efficient layout: %s' \
                    % (axis,
                       'f_contiguous' if dense.flags.f_contiguous
                       else 'c_contiguous' if dense.flags.c_contiguous
                       else 'non-contiguous',
                       ['c_contiguous', 'f_contiguous'][axis])
                warn(reducing_dense_matrix_of_inefficient_format)

        else:
            assert compressed is not None
            elements_count = compressed.nnz / results_count
            axis_format = ('csr', 'csc')[axis]
            if compressed.getformat() == axis_format:
                timed_step = '.compressed-efficient'
            else:
                timed_step = '.compressed-inefficient'
                reducing_sparse_matrix_of_inefficient_format = 'reducing axis: %s ' \
                    'of a sparse matrix in the format: %s ' \
                    'instead of the efficient format: %s' \
                    % (axis, compressed.getformat(), axis_format)
                warn(reducing_sparse_matrix_of_inefficient_format)

    with utm.timed_step(timed_step):
        utm.timed_parameters(results=results_count, elements=elements_count)
        return utt.to_dense_vector(reducer(matrix))


@utm.timed_call()
def bincount_vector(
    vector: utt.Vector,
    *,
    minlength: int = 0,
) -> utt.DenseVector:
    '''
    Drop-in replacement for ``np.bincount``, which is timed and also works on pandas data.
    '''
    proper = utt.to_proper_vector(vector)
    array = utt.DenseVector.be(proper)
    result = np.bincount(array, minlength=minlength)
    utm.timed_parameters(size=array.size, bins=result.size)
    return result


@utm.timed_call()
def most_frequent(
    vector: utt.Vector
) -> Any:
    '''
    Return the most frequent value in a vactor.

    This only
    '''
    unique, positions = \
        np.unique(utt.to_proper_vector(vector), return_inverse=True)
    counts = np.bincount(positions)
    maxpos = np.argmax(counts)
    return unique[maxpos]


ROLLING_FUNCTIONS = {
    'mean': pd.core.window.Rolling.mean,
    'median': pd.core.window.Rolling.median,
    'std': pd.core.window.Rolling.std,
    'var': pd.core.window.Rolling.var,
}


@utd.expand_doc(functions=', '.join(['``%s``' % function for function in ROLLING_FUNCTIONS]))
@utm.timed_call()
def sliding_window_function(
    vector: utt.Vector,
    *,
    function: str,
    window_size: int,
    order_by: Optional[utt.DenseVector] = None,
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

    utm.timed_parameters(function=function,
                         size=array.size,
                         window=window_size)

    half_window_size = (window_size - 1) // 2

    if order_by is not None:
        assert array.size == order_by.size
        order_indices = np.argsort(order_by)
        reverse_order_indices = np.argsort(order_indices)
    else:
        reverse_order_indices = order_indices = np.arange(array.size)

    min_index = order_indices[0]
    max_index = order_indices[-1]

    extended_order_indices = np.concatenate([  #
        np.repeat(min_index, half_window_size),
        order_indices,
        np.repeat(max_index, half_window_size),
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


@utm.timed_call()
def patterns_matches(
    patterns: Union[str, Pattern, Collection[Union[str, Pattern]]],
    strings: Collection[str],
    invert: bool = False,
) -> utt.DenseVector:
    '''
    Given a collection of ``strings``, return a numpy boolean mask specifying which of them match
    the given regular expression ``patterns``.

    If ``invert`` (default: {invert}), invert the mask.
    '''
    if isinstance(patterns, (str, Pattern)):
        patterns = [patterns]

    utm.timed_parameters(patterns=len(patterns), strings=len(strings))

    pattern: Pattern = \
        re.compile('|'.join([alternative if isinstance(alternative, str)
                             else alternative.pattern
                             for alternative in patterns]))

    mask = np.array([bool(pattern.match(string)) for string in strings])

    if invert:
        mask = ~mask

    return mask


@utm.timed_call()
def compress_indices(indices: utt.Vector) -> utt.DenseVector:
    '''
    Given a vector of ``indices`` per element, return a vector where the indices are consecutive.

    If the indices contain ``-1``, then it is preserved as ``-1`` in the result.
    '''
    unique, consecutive = np.unique(indices, return_inverse=True)
    consecutive += min(unique[0], 0)
    return consecutive


@utm.timed_call()
def bin_pack(element_sizes: utt.Vector, max_bin_size: float) -> utt.DenseVector:
    '''
    Given a vector of ``element_sizes`` return a vector containing the bin number for each element,
    such that the total size of each bin is up to and as close to the ``max_bin_size``.

    This uses the first-fit decreasing algorithm for finding an initial solution and then moves
    elements around to minimize the l2 norm of the wasted space in each bin.
    '''
    size_of_bins: List[float] = []
    element_sizes = utt.to_dense_vector(element_sizes)
    descending_size_indices = np.argsort(element_sizes)[::-1]
    bin_of_elements = np.empty(element_sizes.size, dtype='int')

    for element_index in descending_size_indices:
        element_size = element_sizes[element_index]

        assigned = False
        for bin_index in range(len(size_of_bins)):  # pylint: disable=consider-using-enumerate
            if size_of_bins[bin_index] + element_size <= max_bin_size:
                bin_of_elements[element_index] = bin_index
                size_of_bins[bin_index] += element_size
                assigned = True
                break

        if not assigned:
            bin_of_elements[element_index] = len(size_of_bins)
            size_of_bins.append(element_size)

    did_improve = True
    while did_improve:
        did_improve = False
        for element_index in descending_size_indices:
            element_size = element_sizes[element_index]
            if element_size > max_bin_size:
                continue

            current_bin_index = bin_of_elements[element_index]

            current_bin_space = max_bin_size - size_of_bins[current_bin_index]
            assert current_bin_space >= 0
            remove_loss = \
                (element_size + current_bin_space) ** 2 - current_bin_space ** 2

            for bin_index in range(len(size_of_bins)):  # pylint: disable=consider-using-enumerate
                if bin_index == current_bin_index:
                    continue
                bin_space = max_bin_size - size_of_bins[bin_index]
                if bin_space < element_size:
                    continue
                insert_gain = bin_space ** 2 - (bin_space - element_size) ** 2
                if insert_gain > remove_loss:
                    size_of_bins[current_bin_index] -= element_size
                    current_bin_index = bin_of_elements[element_index] = bin_index
                    size_of_bins[bin_index] += element_size
                    remove_loss = insert_gain
                    did_improve = True

    utm.timed_parameters(elements=element_sizes.size, bins=len(size_of_bins))
    return bin_of_elements


@utm.timed_call()
def bin_fill(  # pylint: disable=too-many-statements,too-many-branches
    element_sizes:
    utt.Vector, min_bin_size: float
) -> utt.DenseVector:
    '''
    Given a vector of ``element_sizes`` return a vector containing the bin number for each element,
    such that the total size of each bin is at most and as close to the ``min_bin_size``.

    This uses the first-fit decreasing algorithm for finding an initial solution and then moves
    elements around to minimize the l2 norm of the wasted space in each bin.
    '''
    total_size = np.sum(utt.to_dense_vector(element_sizes))
    assert min_bin_size > 0

    size_of_bins = [0.0]
    max_bins_count = int(total_size // min_bin_size) + 1
    while min(size_of_bins) < min_bin_size:
        max_bins_count -= 1
        if max_bins_count < 2:
            return np.zeros(element_sizes.size, dtype='int')

        size_of_bins = []
        element_sizes = utt.to_dense_vector(element_sizes)
        descending_size_indices = np.argsort(element_sizes)[::-1]
        bin_of_elements = np.empty(element_sizes.size, dtype='int')

        for element_index in descending_size_indices:
            element_size = element_sizes[element_index]

            assigned = False
            for bin_index in range(len(size_of_bins)):  # pylint: disable=consider-using-enumerate
                if size_of_bins[bin_index] + element_size <= min_bin_size:
                    bin_of_elements[element_index] = bin_index
                    size_of_bins[bin_index] += element_size
                    assigned = True
                    break

            if assigned:
                continue

            if len(size_of_bins) < max_bins_count:
                bin_of_elements[element_index] = len(size_of_bins)
                size_of_bins.append(element_size)
                continue

            best_bin_index = -1
            best_bin_waste = -1
            for bin_index in range(len(size_of_bins)):  # pylint: disable=consider-using-enumerate
                bin_waste = size_of_bins[bin_index] - \
                    min_bin_size + element_size
                assert bin_waste > 0
                if best_bin_index < 0 or bin_waste < best_bin_waste:
                    best_bin_index = bin_index
                    best_bin_waste = bin_waste

            assert best_bin_index >= 0
            assert best_bin_waste > 0
            bin_of_elements[element_index] = best_bin_index
            size_of_bins[best_bin_index] += element_size

    did_improve = True
    while did_improve:
        did_improve = False
        for element_index in descending_size_indices:
            element_size = element_sizes[element_index]
            current_bin_index = bin_of_elements[element_index]
            current_bin_waste = size_of_bins[current_bin_index] - min_bin_size
            assert current_bin_waste >= 0

            if current_bin_waste < element_size:
                continue

            remove_gain = \
                current_bin_waste ** 2 - \
                (current_bin_waste - element_size) ** 2

            for bin_index in range(len(size_of_bins)):  # pylint: disable=consider-using-enumerate
                if bin_index == current_bin_index:
                    continue
                bin_waste = size_of_bins[bin_index] - min_bin_size
                insert_loss = (bin_waste + element_size) ** 2 - bin_waste ** 2
                if insert_loss < remove_gain:
                    size_of_bins[current_bin_index] -= element_size
                    current_bin_index = bin_of_elements[element_index] = bin_index
                    size_of_bins[bin_index] += element_size
                    remove_gain = insert_loss
                    did_improve = True

    utm.timed_parameters(elements=element_sizes.size, bins=len(size_of_bins))
    return bin_of_elements


@utm.timed_call()
def sum_groups(
    matrix: utt.Matrix,
    groups: utt.Vector,
    *,
    per: str
) -> Optional[Tuple[utt.Matrix, utt.Vector]]:
    '''
    Given a ``matrix``, and a vector of ``groups`` ``per`` row or column, return a matrix with a row
    or column per group, containing the sum of the groups rows or columns, and a vector of sizes
    (the number of summed rows or columns) per group.

    Negative group indices are ignored and their data is not included in the result. If there are no
    non-negative group indices, returns ``None``.
    '''

    assert per in utt.PER_OF_AXIS

    groups = utt.to_dense_vector(groups)
    groups_count = np.max(groups) + 1

    if groups_count == 0:
        return None

    efficient_layout = per + '_major'

    sparse = utt.SparseMatrix.maybe(matrix)
    if sparse is not None:
        if utt.matrix_layout(matrix) == efficient_layout:
            timed_step = '.compressed-efficient'
        else:
            timed_step = '.compressed-inefficient'
            grouping_sparse_matrix_of_inefficient_format = 'grouping %ss ' \
                'of a sparse matrix in the format: %s ' \
                'instead of the efficient format: cs%s' \
                % (per, sparse.getformat(), per[0])
            warn(grouping_sparse_matrix_of_inefficient_format)
    else:
        matrix = utt.to_dense_matrix(matrix)
        if utt.matrix_layout(matrix) == efficient_layout:
            timed_step = '.dense-efficient'
        else:
            timed_step = '.dense-inefficient'
            grouping_dense_matrix_of_inefficient_format = 'grouping %ss ' \
                'of a dense matrix with inefficient strides: %s' \
                % (per, matrix.strides)
            warn(grouping_dense_matrix_of_inefficient_format)

    if per == 'column':
        matrix = matrix.transpose()

    group_sizes = np.zeros(groups_count, dtype='int32')

    with utm.timed_step(timed_step):
        utm.timed_parameters(groups=groups_count, entities=matrix.shape[0],
                             elements=matrix.shape[1])
        results = np.empty((groups_count, matrix.shape[1]), matrix.dtype)

        for group_index in range(groups_count):
            group_mask = groups == group_index
            group_size = np.sum(group_mask)
            assert group_size > 0
            group_sizes[group_index] = group_size
            group_matrix = matrix[group_mask, :]
            results[group_index, :] = \
                utt.to_dense_vector(group_matrix.sum(axis=0))

    if per == 'column':
        results = results.transpose()
    return results, group_sizes


@utm.timed_call()
def shuffle_matrix(
    matrix: utt.Matrix,
    *,
    per: str,
    random_seed: int = 0,
) -> None:
    '''
    Shuffle (in-place) the ``matrix`` data ``per`` column or row.

    The matrix must be in the appropriate layout (row-major for shuffling data in each row,
    column-major for shuffling data in each column).

    A non-zero ``random_seed`` (default: {random_seed}) can be provided to make the operation
    replicable.
    '''
    assert per in utt.PER_OF_AXIS
    axis = utt.PER_OF_AXIS.index(per)
    layout = utt.LAYOUT_OF_AXIS[axis]

    _, dense, compressed = \
        utt.to_proper_matrices(matrix, default_layout=layout)

    if compressed is not None:
        assert utt.matrix_layout(compressed) == layout
        extension_name = 'shuffle_compressed_%s_t_%s_t_%s_t' \
            % (compressed.data.dtype,
               compressed.indices.dtype,
               compressed.indptr.dtype)
        extension = getattr(xt, extension_name)
        extension(compressed.data, compressed.indices, compressed.indptr,
                  compressed.shape[1 - axis], random_seed)

    else:
        assert dense is not None
        assert utt.matrix_layout(dense) == layout
        if per == 'column':
            dense = dense.transpose()
        extension_name = 'shuffle_matrix_%s_t' % dense.dtype
        extension = getattr(xt, extension_name)
        extension(dense, random_seed)
