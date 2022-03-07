"""
Computation
-----------

Most of the functions defined here are thin wrappers around builtin numpy or scipy functions, with
others wrapping C++ extensions provided as part of the metacells package itself.

The key distinction of the functions here is that they provide a uniform interface for all the
supported :py:const:`metacells.utilities.typing.Matrix` and
:py:const:`metacells.utilities.typing.Vector` types, which makes them safe to use in our code
without worrying about the exact data type used. In theory, Python duck-typing should have provided
this out of the box, but it seems that without explicit types and interfaces, the interfaces of the
different types diverge to the point where this just doesn't work.

All the functions here (optionally) also allow collecting timing information using
:py:mod:`metacells.utilities.timing`, to make it easier to locate the performance bottleneck of the
analysis pipeline.
"""

import re
import sys
from math import ceil
from math import floor
from math import sqrt
from re import Pattern
from typing import Any
from typing import Callable
from typing import Collection
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import overload
from warnings import catch_warnings
from warnings import filterwarnings
from warnings import warn

import cvxpy
import igraph as ig  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

import metacells.utilities.documentation as utd
import metacells.utilities.timing as utm
import metacells.utilities.typing as utt

if "sphinx" not in sys.argv[0]:
    import metacells.extensions as xt  # type: ignore

__all__ = [
    "allow_inefficient_layout",
    "to_layout",
    "sort_compressed_indices",
    "corrcoef",
    "cross_corrcoef_rows",
    "pairs_corrcoef_rows",
    "logistics",
    "cross_logistics_rows",
    "pairs_logistics_rows",
    "log_data",
    "mean_per",
    "nanmean_per",
    "max_per",
    "nanmax_per",
    "min_per",
    "nanmin_per",
    "nnz_per",
    "sum_per",
    "sum_squared_per",
    "rank_per",
    "top_per",
    "prune_per",
    "quantile_per",
    "nanquantile_per",
    "scale_by",
    "fraction_by",
    "fraction_per",
    "variance_per",
    "normalized_variance_per",
    "relative_variance_per",
    "sum_matrix",
    "nnz_matrix",
    "mean_matrix",
    "max_matrix",
    "min_matrix",
    "nanmean_matrix",
    "nanmax_matrix",
    "nanmin_matrix",
    "rank_matrix_by_layout",
    "bincount_vector",
    "most_frequent",
    "highest_weight",
    "fraction_of_grouped",
    "downsample_matrix",
    "downsample_vector",
    "matrix_rows_folds_and_aurocs",
    "sliding_window_function",
    "patterns_matches",
    "compress_indices",
    "bin_pack",
    "bin_fill",
    "sum_groups",
    "shuffle_matrix",
    "cover_diameter",
    "cover_coordinates",
    "random_piles",
    "group_piles",
    "represent",
    "min_cut",
    "sparsify_matrix",
]


ALLOW_INEFFICIENT_LAYOUT: bool = True


def allow_inefficient_layout(allow: bool) -> bool:
    """
    Specify whether to allow processing using an inefficient layout.

    Returns the previous setting.

    This is ``True`` by default, which merely warns when an inefficient layout is used. Otherwise,
    processing an inefficient layout is treated as an error (raises an exception).
    """
    global ALLOW_INEFFICIENT_LAYOUT
    prev_allow = ALLOW_INEFFICIENT_LAYOUT
    ALLOW_INEFFICIENT_LAYOUT = allow
    return prev_allow


@overload
def to_layout(matrix: utt.CompressedMatrix, layout: str, *, symmetric: bool = False) -> utt.CompressedMatrix:
    ...


@overload
def to_layout(matrix: utt.NumpyMatrix, layout: str, *, symmetric: bool = False) -> utt.NumpyMatrix:
    ...


@overload
def to_layout(matrix: utt.ImproperMatrix, layout: str, *, symmetric: bool = False) -> utt.ProperMatrix:
    ...


@utm.timed_call()
@utd.expand_doc()
def to_layout(matrix: utt.Matrix, layout: str, *, symmetric: bool = False) -> utt.ProperMatrix:
    """
    Return the ``matrix`` in a specific ``layout`` for efficient processing.

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

    Otherwise, a new copy is created in the proper layout. This is a costly operation as it needs to move all the data
    elements to their proper place. This uses a C++ extension to deal with compressed data (the builtin implementation
    is much slower). Even so this operation is costly; still, it makes the following processing **much** more efficient,
    so it is typically a net performance gain overall.
    """
    assert layout in utt.LAYOUT_OF_AXIS

    proper, dense, compressed = utt.to_proper_matrices(matrix, default_layout=layout)

    utm.timed_parameters(rows=proper.shape[0], columns=proper.shape[1])

    if utt.is_layout(proper, layout):
        utm.timed_parameters(method="none")
        return proper

    if symmetric:
        assert proper.shape[0] == proper.shape[1]
        proper = proper.transpose()
        assert utt.is_layout(proper, layout)
        utm.timed_parameters(method="transpose")
        return proper

    is_frozen = utt.frozen(proper)
    result: utt.ProperMatrix

    if dense is not None:
        utm.timed_parameters(method="reshape")
        order = utt.DENSE_FAST_FLAG[layout][0]
        with utm.timed_step("numpy.ravel"):
            utm.timed_parameters(rows=dense.shape[0], columns=dense.shape[1])
            result = np.reshape(np.ravel(dense, order=order), dense.shape, order=order)  # type: ignore

    else:
        utm.timed_parameters(method="compressed")
        assert compressed is not None
        to_format = utt.SPARSE_FAST_FORMAT[layout]
        from_format = utt.SPARSE_SLOW_FORMAT[layout]
        assert compressed.getformat() == from_format
        compressed = _relayout_compressed(compressed)
        assert compressed.getformat() == to_format
        result = compressed

    if is_frozen:
        utt.freeze(result)

    assert result.shape == matrix.shape
    return result


@utm.timed_call()
def _relayout_compressed(compressed: utt.CompressedMatrix) -> utt.CompressedMatrix:
    """
    Efficient parallel conversion of a CSR/CSC ``matrix`` to a CSC/CSR matrix.
    """
    axis = ("csr", "csc").index(compressed.getformat())

    if compressed.nnz < 2:
        if axis == 0:
            compressed = compressed.tocsc()
        else:
            compressed = compressed.tocsr()
        utt.sort_indices(compressed)

    else:
        matrix_bands_count = compressed.shape[axis]
        output_bands_count = matrix_elements_count = compressed.shape[1 - axis]

        nnz_elements_of_output_bands = bincount_vector(compressed.indices, minlength=matrix_elements_count)
        output_indptr = np.empty(output_bands_count + 1, dtype=compressed.indptr.dtype)
        output_indptr[0:2] = 0
        with utm.timed_step("numpy.cumsum"):
            utm.timed_parameters(elements=nnz_elements_of_output_bands.size - 1)
            np.cumsum(nnz_elements_of_output_bands[:-1], out=output_indptr[2:])

        output_indices = np.empty(compressed.nnz, dtype=compressed.indices.dtype)
        output_data = np.empty(compressed.nnz, dtype=compressed.data.dtype)

        extension_name = "collect_compressed_%s_t_%s_t_%s_t" % (  # pylint: disable=consider-using-f-string
            compressed.data.dtype,
            compressed.indices.dtype,
            compressed.indptr.dtype,
        )
        extension = getattr(xt, extension_name)

        assert matrix_bands_count == compressed.indptr.size - 1

        with utm.timed_step("extensions.collect_compressed"):
            utm.timed_parameters(results=output_bands_count, elements=matrix_bands_count)
            assert compressed.indptr[-1] == compressed.data.size
            extension(
                compressed.data, compressed.indices, compressed.indptr, output_data, output_indices, output_indptr[1:]
            )

        assert output_indptr[-1] == compressed.indptr[-1]

        constructor = (sp.csr_matrix, sp.csc_matrix)[1 - axis]
        compressed = constructor((output_data, output_indices, output_indptr), shape=compressed.shape)

        sort_compressed_indices(compressed, force=True)

    compressed.has_canonical_format = True

    return compressed


def sort_compressed_indices(matrix: utt.CompressedMatrix, force: bool = False) -> None:
    """
    Efficient parallel sort of indices in a CSR/CSC ``matrix``.

    This will skip sorting a matrix that is marked as sorted, unless ``force`` is specified.
    """
    assert matrix.getformat() in ("csr", "csc")
    if matrix.has_sorted_indices and not force:
        return

    matrix_bands_count = matrix.indptr.size - 1
    extension_name = "sort_compressed_indices_%s_t_%s_t_%s_t" % (  # pylint: disable=consider-using-f-string
        matrix.data.dtype,
        matrix.indices.dtype,
        matrix.indptr.dtype,
    )
    extension = getattr(xt, extension_name)

    with utm.timed_step("extensions.sort_compressed_indices"):
        utm.timed_parameters(results=matrix_bands_count, elements=matrix.nnz / matrix_bands_count)
        extension(matrix.data, matrix.indices, matrix.indptr, matrix.shape[1])

    matrix.has_sorted_indices = True


def _ensure_per(matrix: utt.Matrix, per: Optional[str]) -> str:
    if per is not None:
        assert per in ("row", "column")
        return per

    assert matrix.shape[0] == matrix.shape[1]
    layout = utt.matrix_layout(matrix)
    assert layout is not None
    return layout[:-6]


def _ensure_layout_for(operation: str, matrix: utt.Matrix, per: Optional[str], allow_inefficient: bool = True) -> None:
    if utt.is_layout(matrix, f"{per}_major"):
        return

    if not ALLOW_INEFFICIENT_LAYOUT:
        allow_inefficient = False

    layout = utt.matrix_layout(matrix)
    if layout is not None:
        operating_on_matrix_of_wrong_layout = f"{operation} of {per}s of a matrix with {layout} layout"
        if not allow_inefficient:
            raise NotImplementedError(operating_on_matrix_of_wrong_layout)
        warn(operating_on_matrix_of_wrong_layout)
        return

    sparse = utt.maybe_sparse_matrix(matrix)
    if sparse is not None:
        operating_on_sparse_matrix_of_inefficient_format = (
            f"{operation} of {per}s of a sparse matrix with inefficient format: {sparse.getformat()}"
        )
        if not allow_inefficient:
            raise NotImplementedError(operating_on_sparse_matrix_of_inefficient_format)  #
        warn(operating_on_sparse_matrix_of_inefficient_format)
        return

    dense = utt.to_numpy_matrix(matrix, only_extract=True)
    operating_on_numpy_matrix_of_inefficient_strides = (
        f"{operation} of {per}s of a numpy matrix with inefficient strides: {dense.strides}"
    )
    if not allow_inefficient:
        raise NotImplementedError(operating_on_numpy_matrix_of_inefficient_strides)  #
    warn(operating_on_numpy_matrix_of_inefficient_strides)


def _ensure_per_for(operation: str, matrix: utt.Matrix, per: Optional[str]) -> str:
    per = _ensure_per(matrix, per)
    _ensure_layout_for(operation, matrix, per)
    return per


def _get_dense_for(operation: str, matrix: utt.Matrix, per: Optional[str]) -> Tuple[str, utt.NumpyMatrix]:
    per = _ensure_per(matrix, per)
    dense = utt.to_numpy_matrix(matrix, default_layout=f"{per}_major")
    _ensure_layout_for(operation, dense, per)
    return per, dense


@utm.timed_call()
def corrcoef(
    matrix: utt.Matrix,
    *,
    per: Optional[str],
    reproducible: bool,
) -> utt.NumpyMatrix:
    """
    Similar to for ``numpy.corrcoef``, but also works for a sparse ``matrix``, and can be
    ``reproducible`` regardless of the number of cores used (at the cost of some slowdown). It only
    works for matrices with a float or double element data type.

    If ``reproducible``, a slower (still parallel) but reproducible algorithm will be used.

    Unlike ``numpy.corrcoef``, if given a row with identical values, instead of complaining about
    division by zero, this will report a zero correlation. This makes sense for the intended usage
    of computing similarities between cells/genes - an all-zero row has no data so we declare it to
    be "not similar" to anything else.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).

    .. note::

        The result is always dense, as even for sparse data, the correlation is rarely exactly zero.
    """
    per, dense = _get_dense_for("corrcoef", matrix, per)

    if not reproducible or str(dense.dtype) not in ("float", "double", "float32", "float64"):
        return _corrcoef_fast(dense, per)

    return _corrcoef_reproducible(dense, per)


@utm.timed_call(".irreproducible")
def _corrcoef_fast(
    dense: utt.NumpyMatrix,
    per: str,
) -> utt.NumpyMatrix:
    axis = utt.PER_OF_AXIS.index(per)

    utm.timed_parameters(results=dense.shape[axis], elements=dense.shape[1 - axis])

    with utt.unfrozen(dense):
        # Replication of numpy code:
        X = dense if per == "row" else dense.T
        row_averages = np.average(X, axis=1)
        X -= row_averages[:, None]
        result = np.matmul(X, X.T)
        result *= 1 / X.shape[1]
        X += row_averages[:, None]
        diagonal = np.diag(result)
        stddev = np.sqrt(diagonal)
        stddev[stddev == 0] = 1
        result /= stddev[:, None]
        result /= stddev[None, :]
        np.clip(result, -1, 1, out=result)
        np.fill_diagonal(result, 1.0)

    return result


@utm.timed_call(".reproducible")
def _corrcoef_reproducible(
    dense: utt.NumpyMatrix,
    per: str,
) -> utt.NumpyMatrix:
    if per == "column":
        dense = dense.T
    utm.timed_parameters(results=dense.shape[0], elements=dense.shape[1])
    extension_name = f"correlate_dense_{dense.dtype}_t"
    result = np.empty((dense.shape[0], dense.shape[0]), dtype="float32")
    extension = getattr(xt, extension_name)
    extension(dense, result)
    return result


@utm.timed_call()
def cross_corrcoef_rows(
    first_matrix: utt.NumpyMatrix,
    second_matrix: utt.NumpyMatrix,
    *,
    reproducible: bool,  # pylint: disable=unused-argument
) -> utt.NumpyMatrix:
    """
    Similar to for ``numpy.corrcoef``, but computes the correlations between each row of the
    ``first_matrix`` and each row of the ``second_matrix``. The result matrix contains one row per
    row of the first matrix and one column per row of the second matrix. Both matrices must be
    dense, in row-major layout, have the same (float or double) element data type, and contain the
    same number of columns.

    If ``reproducible``, a slower (still parallel) but reproducible algorithm will be used.

    Unlike ``numpy.corrcoef``, if given a row with identical values, instead of complaining about
    division by zero, this will report a zero correlation. This makes sense for the intended usage
    of computing similarities between cells/genes - an all-zero row has no data so we declare it to
    be "not similar" to anything else.

    .. note::

        This only works for floating-point matrices.

    .. todo::

        Implement a fast algorithm for the non-reproducible case.
    """
    first_matrix = utt.mustbe_numpy_matrix(first_matrix)
    second_matrix = utt.mustbe_numpy_matrix(second_matrix)
    assert utt.is_layout(first_matrix, "row_major")
    assert utt.is_layout(second_matrix, "row_major")
    assert first_matrix.shape[1] == second_matrix.shape[1]
    assert first_matrix.dtype == second_matrix.dtype

    workaround = first_matrix.shape[0] == 1
    if workaround:
        first_matrix = np.concatenate([first_matrix, first_matrix])

    extension_name = f"cross_correlate_dense_{first_matrix.dtype}_t"
    result = np.empty((first_matrix.shape[0], second_matrix.shape[0]), dtype="float32")
    extension = getattr(xt, extension_name)
    extension(first_matrix, second_matrix, result)

    if workaround:
        result = result[0:1, :]

    return result


@utm.timed_call()
def pairs_corrcoef_rows(
    first_matrix: utt.NumpyMatrix,
    second_matrix: utt.NumpyMatrix,
    *,
    reproducible: bool,  # pylint: disable=unused-argument
) -> utt.NumpyVector:
    """
    Similar to for ``numpy.corrcoef``, but computes the correlations between each row of the
    ``first_matrix`` and each matching row of the ``second_matrix``. Both matrices must be dense, in
    row-major layout, have the same (float or double) element data type, and the same shape.

    If ``reproducible``, a slower (still parallel) but reproducible algorithm will be used.

    Unlike ``numpy.corrcoef``, if given a row with identical values, instead of complaining about
    division by zero, this will report a zero correlation. This makes sense for the intended usage
    of computing similarities between cells/genes - an all-zero row has no data so we declare it to
    be "not similar" to anything else.

    .. note::

        This only works for floating-point matrices.

    .. todo::

        Implement a fast algorithm for the non-reproducible case.
    """
    first_matrix = utt.mustbe_numpy_matrix(first_matrix)
    second_matrix = utt.mustbe_numpy_matrix(second_matrix)
    assert utt.is_layout(first_matrix, "row_major")
    assert utt.is_layout(second_matrix, "row_major")
    assert first_matrix.shape == second_matrix.shape
    assert first_matrix.dtype == second_matrix.dtype

    extension_name = f"pairs_correlate_dense_{first_matrix.dtype}_t"
    result = np.empty(first_matrix.shape[0], dtype="float32")
    extension = getattr(xt, extension_name)
    extension(first_matrix, second_matrix, result)
    return result


@utm.timed_call()
def logistics(matrix: utt.NumpyMatrix, *, location: float, slope: float, per: Optional[str]) -> utt.NumpyMatrix:
    """
    Compute a matrix of distances between each pair of rows in a dense (float or double) matrix
    using the logistics function.

    The raw value of the logistics distance between a pair of vectors ``x`` and ``y`` is the mean of
    ``1/(1+exp(-slope*(abs(x[i]-y[i])-location)))``. This has a minimum of
    ``1/(1+exp(slope*location))`` for identical vectors and an (asymptotic) maximum of 1. We
    normalize this to a range between 0 and 1, to be useful as a distance measure (with a zero
    distance between identical vectors).

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).

    .. todo::

        This always uses the dense implementation. Possibly a sparse implementation might be faster.

    .. note::

        The result is always dense, as even for sparse data, the result is rarely exactly zero.
    """
    per, dense = _get_dense_for("logistics", matrix, per)
    axis = utt.PER_OF_AXIS.index(per)

    utm.timed_parameters(results=dense.shape[axis], elements=dense.shape[1 - axis])

    if per == "column":
        dense = dense.T

    extension_name = f"logistics_dense_{dense.dtype}_t"
    result = np.empty((dense.shape[0], dense.shape[0]), dtype="float32")
    extension = getattr(xt, extension_name)
    extension(dense, result, location, slope)
    return result


@utm.timed_call()
def cross_logistics_rows(
    first_matrix: utt.NumpyMatrix,
    second_matrix: utt.NumpyMatrix,
    *,
    location: float,
    slope: float,
) -> utt.NumpyMatrix:
    """
    Similar to for :py:func:`logistics`, but computes the distances between each row of the
    ``first_matrix`` and each row of the ``second_matrix``. The result matrix contains one row per
    row of the first matrix and one column per row of the second matrix. Both matrices must be
    dense, in row-major layout, have the same (float or double) element data type, and contain the
    same number of columns.
    """
    first_matrix = utt.mustbe_numpy_matrix(first_matrix)
    second_matrix = utt.mustbe_numpy_matrix(second_matrix)
    assert utt.is_layout(first_matrix, "row_major")
    assert utt.is_layout(second_matrix, "row_major")
    assert first_matrix.shape[1] == second_matrix.shape[1]
    assert first_matrix.dtype == second_matrix.dtype

    extension_name = f"cross_logistics_dense_{first_matrix.dtype}_t"
    result = np.empty((first_matrix.shape[0], second_matrix.shape[0]), dtype="float32")
    extension = getattr(xt, extension_name)
    extension(first_matrix, second_matrix, location, slope, result)
    return result


@utm.timed_call()
def pairs_logistics_rows(
    first_matrix: utt.NumpyMatrix,
    second_matrix: utt.NumpyMatrix,
    *,
    location: float,
    slope: float,
) -> utt.NumpyVector:
    """
    Similar to for :py:func:`logistics`, but computes the distances between each row of the
    ``first_matrix`` and each matching row of the ``second_matrix``. Both matrices must be dense, in
    row-major layout, have the same (float or double) element data type, and the same shape.
    """
    first_matrix = utt.mustbe_numpy_matrix(first_matrix)
    second_matrix = utt.mustbe_numpy_matrix(second_matrix)
    assert utt.is_layout(first_matrix, "row_major")
    assert utt.is_layout(second_matrix, "row_major")
    assert first_matrix.shape == second_matrix.shape
    assert first_matrix.dtype == second_matrix.dtype

    extension_name = f"pairs_logistics_dense_{first_matrix.dtype}_t"
    result = np.empty(first_matrix.shape[0], dtype="float32")
    extension = getattr(xt, extension_name)
    extension(first_matrix, second_matrix, location, slope, result)
    return result


S = TypeVar("S", bound="utt.Shaped")


@utm.timed_call()
@utd.expand_doc()
def log_data(  # pylint: disable=too-many-branches
    shaped: S,
    *,
    base: Optional[float] = None,
    normalization: float = 0,
) -> S:
    """
    Return the log of the values in the ``shaped`` data.

    If ``base`` is specified (default: {base}), use this base log. Otherwise, use the natural
    logarithm.

    The ``normalization`` (default: {normalization}) specifies how to deal with zeros in the data:

    * If it is zero, an input zero will become an output ``NaN``.

    * If it is positive, it is added to the input before computing the log.

    * If it is negative, input zeros will become log(minimal positive value) + normalization,
      that is, the zeros will be given a value this much smaller than the minimal "real" log value.

    .. note::

        The result is always dense, as even for sparse data, the log is rarely zero.
    """
    dense: np.ndarray
    if shaped.ndim == 1:
        dense = utt.to_numpy_vector(shaped, copy=True)
    else:
        dense = utt.to_numpy_matrix(shaped, copy=True)  # type: ignore

    if normalization > 0:
        dense += normalization
    else:
        where = dense > 0
        if normalization < 0:
            dense[~where] = np.min(dense[where])

    if base is None:
        log_function = np.log
    elif base == 2:
        log_function = np.log2
        base = None
    elif base == 10:
        log_function = np.log10
        base = None
    else:
        assert base > 0
        log_function = np.log

    if normalization == 0:
        log_function(dense, out=dense, where=where)
        dense[~where] = None
    else:
        log_function(dense, out=dense)

    if base is not None:
        dense /= np.log(base)

    if normalization < 0:
        dense[~where] += normalization

    return dense  # type: ignore


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
) -> utt.ProperMatrix:
    """
    Downsample the data ``per`` (one of ``row`` and ``column``) such that the sum of each one
    becomes ``samples``.

    If the matrix is sparse, and ``eliminate_zeros`` (default: {eliminate_zeros}), then perform a
    final phase of eliminating leftover zero values from the compressed format. This means the
    result will be in "canonical format" so further ``scipy`` sparse operations on it will be
    faster.

    If ``inplace`` (default: {inplace}), modify the matrix in-place, otherwise, return a modified
    copy.

    A non-zero ``random_seed`` (default: {random_seed}) will make the operation replicable.
    """
    assert per in ("row", "column")
    assert samples > 0

    _, dense, compressed = utt.to_proper_matrices(matrix)

    if dense is not None:
        return _downsample_dense_matrix(dense, per, samples, inplace, random_seed)

    assert compressed is not None
    return _downsample_compressed_matrix(compressed, per, samples, eliminate_zeros, inplace, random_seed)


def _downsample_dense_matrix(
    matrix: utt.NumpyMatrix, per: str, samples: int, inplace: bool, random_seed: int
) -> utt.NumpyMatrix:
    if inplace:
        output = matrix
    elif per == "row":
        output = np.empty(matrix.shape, dtype=matrix.dtype, order="C")
    else:
        output = np.empty(matrix.shape, dtype=matrix.dtype, order="F")

    assert output.shape == matrix.shape

    axis = utt.PER_OF_AXIS.index(per)
    results_count = matrix.shape[axis]
    elements_count = matrix.shape[1 - axis]

    if per == "row":
        sample_input_vector = matrix[0, :]
        sample_output_vector = output[0, :]
    else:
        sample_input_vector = matrix[:, 0]
        sample_output_vector = output[:, 0]

    input_is_contiguous = sample_input_vector.flags.c_contiguous and sample_input_vector.flags.f_contiguous
    if not input_is_contiguous:
        raise NotImplementedError(
            f"downsampling per-{per} of a matrix input matrix with inefficient strides: {matrix.strides}"
        )

    output_is_contiguous = sample_output_vector.flags.c_contiguous and sample_output_vector.flags.f_contiguous
    if not inplace and not output_is_contiguous:
        raise NotImplementedError(
            f"downsampling per-{per} of a matrix output matrix with inefficient strides: {output.strides}"
        )

    if per == "column":
        matrix = np.transpose(matrix)
        output = np.transpose(output)

    assert results_count == matrix.shape[0]
    assert elements_count == matrix.shape[1]

    if results_count == 1:
        input_array = utt.to_numpy_vector(matrix)
        output_array = utt.to_numpy_vector(output)
        extension_name = f"downsample_array_{input_array.dtype}_t_{output_array.dtype}_t"
        extension = getattr(xt, extension_name)
        with utm.timed_step("extensions.downsample_array"):
            utm.timed_parameters(elements=input_array.size, samples=samples)
            extension(input_array, output_array, samples, random_seed)
    else:
        extension_name = f"downsample_dense_{matrix.dtype}_t_{output.dtype}_t"
        extension = getattr(xt, extension_name)
        with utm.timed_step("extensions.downsample_dense_matrix"):
            utm.timed_parameters(results=results_count, elements=elements_count, samples=samples)
            extension(matrix, output, samples, random_seed)

    if per == "column":
        output = np.transpose(output)

    assert output.shape == matrix.shape
    return output


def _downsample_compressed_matrix(
    matrix: utt.CompressedMatrix, per: str, samples: int, eliminate_zeros: bool, inplace: bool, random_seed: int
) -> utt.CompressedMatrix:
    axis = utt.PER_OF_AXIS.index(per)
    results_count = matrix.shape[axis]
    elements_count = matrix.shape[1 - axis]

    axis_format = ("csr", "csc")[axis]
    if matrix.getformat() != axis_format:
        raise NotImplementedError(
            f"downsample per-{per} "
            f"of a sparse matrix in the format: {matrix.getformat()} "
            f"instead of the efficient format: {axis_format}"
        )

    if inplace:
        assert not utt.frozen(matrix)
        output = matrix
    else:
        constructor = (sp.csr_matrix, sp.csc_matrix)[axis]
        output_data = np.empty_like(matrix.data)
        output = constructor((output_data, np.copy(matrix.indices), np.copy(matrix.indptr)), shape=matrix.shape)
        output.has_sorted_indices = matrix.has_sorted_indices
        output.has_canonical_format = matrix.has_canonical_format

    extension_name = "downsample_compressed_%s_t_%s_t_%s_t" % (  # pylint: disable=consider-using-f-string
        matrix.data.dtype,
        matrix.indptr.dtype,
        output.data.dtype,
    )
    extension = getattr(xt, extension_name)

    assert results_count == matrix.indptr.size - 1

    with utm.timed_step("extensions.downsample_sparse_matrix"):
        utm.timed_parameters(results=results_count, elements=elements_count, samples=samples)
        extension(matrix.data, matrix.indptr, output.data, samples, random_seed)

    if eliminate_zeros:
        utt.eliminate_zeros(output)

    return output


@utm.timed_call()
@utd.expand_doc(data_types=",".join([f"``{data_type}``" for data_type in utt.CPP_DATA_TYPES]))
def downsample_vector(
    vector: utt.Vector, samples: int, *, output: Optional[utt.NumpyVector] = None, random_seed: int = 0
) -> None:
    """
    Downsample a vector of sample counters.

    **Input**

    * A numpy ``vector`` containing non-negative integer sample counts.

    * A desired total number of ``samples``.

    * An optional numpy array ``output`` to hold the results (otherwise, the input is overwritten).

    * A ``random_seed`` (default: {random_seed}) which, if non-zero, will make the operation
      replicable.

    The arrays may have any of the data types: {data_types}.

    **Operation**

    If the total number of samples (sum of the array) is not higher than the required number of
    samples, the output is identical to the input.

    Otherwise, treat the input as if it was a set where each index appeared its value number of
    times. Randomly select the desired number of samples from this set (without repetition), and
    store in the output the number of times each index was chosen.
    """

    array = utt.to_numpy_vector(vector)
    if output is None:
        assert id(array) == id(vector)
        output = array
    else:
        assert output.shape == array.shape

    extension_name = f"downsample_array_{array.dtype}_t_{output.dtype}_t"
    extension = getattr(xt, extension_name)

    with utm.timed_step("extensions.downsample_array"):
        utm.timed_parameters(elements=array.size, samples=samples)
        extension(array, output, samples, random_seed)


@utm.timed_call()
def matrix_rows_folds_and_aurocs(
    matrix: utt.Matrix,
    *,
    columns_subset: utt.NumpyVector,
    columns_scale: Optional[utt.NumpyVector] = None,
    normalization: float,
) -> Tuple[utt.NumpyVector, utt.NumpyVector]:
    """
    Given a matrix and a subset of the columns, return two vectors. The first contains, for each
    row, the mean column value in the subset divided by the mean column value outside the subset.
    The second contains for each row the area under the receiver operating characteristic (AUROC)
    for the row, that is, the probability that a random column in the subset would have a higher
    value in this row than a random column outside the subset.

    If ``columns_scale`` is specified, the data is divided by this scale before computing the AUROC.
    """
    proper, dense, compressed = utt.to_proper_matrices(matrix)
    rows_count, columns_count = proper.shape

    if columns_scale is None:
        columns_scale = np.full(columns_count, 1.0, dtype="float32")
    else:
        columns_scale = columns_scale.astype("float32")
        assert columns_scale.size == columns_count

    rows_folds = np.empty(rows_count, dtype="float64")
    rows_auroc = np.empty(rows_count, dtype="float64")

    columns_subset = utt.to_numpy_vector(columns_subset)
    if columns_subset.dtype == "bool":
        assert columns_subset.size == columns_count
    else:
        mask: utt.NumpyVector = np.full(columns_count, False)
        mask[columns_subset] = True
        columns_subset = mask

    if dense is not None:
        extension_name = f"auroc_dense_matrix_{dense.dtype}_t"
        extension = getattr(xt, extension_name)
        extension(dense, columns_subset, columns_scale, normalization, rows_folds, rows_auroc)
    else:
        assert compressed is not None
        assert compressed.has_sorted_indices
        extension_name = "auroc_compressed_matrix_%s_t_%s_t_%s_t" % (  # pylint: disable=consider-using-f-string
            compressed.data.dtype,
            compressed.indices.dtype,
            compressed.indptr.dtype,
        )
        extension = getattr(xt, extension_name)
        extension(
            compressed.data,
            compressed.indices,
            compressed.indptr,
            columns_count,
            columns_subset,
            columns_scale,
            normalization,
            rows_folds,
            rows_auroc,
        )

    return (rows_folds, rows_auroc)


@utm.timed_call()
def mean_per(matrix: utt.Matrix, *, per: Optional[str]) -> utt.NumpyVector:
    """
    Compute the mean value ``per`` (``row`` or ``column``) of some ``matrix``.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).
    """
    per = _ensure_per_for("mean", matrix, per)
    axis = 1 - utt.PER_OF_AXIS.index(per)
    return sum_per(matrix, per=per) / matrix.shape[axis]


@utm.timed_call()
def nanmean_per(matrix: utt.Matrix, *, per: Optional[str]) -> utt.NumpyVector:
    """
    Compute the mean value ``per`` (``row`` or ``column``) of some ``matrix``, ignoring ``None``
    values, if any.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).
    """
    per, dense = _get_dense_for("nanmean", matrix, per)
    axis = utt.PER_OF_AXIS.index(per)

    with catch_warnings():
        filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        return _reduce_matrix("nanmean", dense, per, lambda dense: np.nanmean(dense, axis=1 - axis))


@utm.timed_call()
def max_per(matrix: utt.Matrix, *, per: Optional[str]) -> utt.NumpyVector:
    """
    Compute the maximal value ``per`` (``row`` or ``column``) of some ``matrix``.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).
    """
    per = _ensure_per_for("max", matrix, per)
    axis = utt.PER_OF_AXIS.index(per)

    sparse = utt.maybe_sparse_matrix(matrix)
    if sparse is not None:
        return _reduce_matrix("max", sparse, per, lambda sparse: sparse.max(axis=1 - axis))

    dense = utt.to_numpy_matrix(matrix, only_extract=True)
    return _reduce_matrix("max", dense, per, lambda dense: np.max(dense, axis=1 - axis))


@utm.timed_call()
def nanmax_per(matrix: utt.Matrix, *, per: Optional[str]) -> utt.NumpyVector:
    """
    Compute the maximal value ``per`` (``row`` or ``column``) of some ``matrix``, ignoring ``None``
    values, if any.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).
    """
    per, dense = _get_dense_for("nanmax", matrix, per)
    axis = utt.PER_OF_AXIS.index(per)

    with catch_warnings():
        filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        return _reduce_matrix("nanmax", dense, per, lambda dense: np.nanmax(dense, axis=1 - axis))


@utm.timed_call()
def min_per(matrix: utt.Matrix, *, per: Optional[str]) -> utt.NumpyVector:
    """
    Compute the minimal value ``per`` (``row`` or ``column``) of some ``matrix``.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).
    """
    per = _ensure_per_for("nanmax", matrix, per)
    axis = utt.PER_OF_AXIS.index(per)

    sparse = utt.maybe_sparse_matrix(matrix)
    if sparse is not None:
        return _reduce_matrix("min", sparse, per, lambda sparse: sparse.min(axis=1 - axis))

    dense = utt.to_numpy_matrix(matrix, only_extract=True)
    return _reduce_matrix("min", dense, per, lambda dense: np.min(dense, axis=1 - axis))


@utm.timed_call()
def nanmin_per(matrix: utt.Matrix, *, per: Optional[str]) -> utt.NumpyVector:
    """
    Compute the minimal value ``per`` (``row`` or ``column``) of some ``matrix``,
    ignoring ``None`` values, if any.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).
    """
    per, dense = _get_dense_for("nanmin", matrix, per)
    axis = utt.PER_OF_AXIS.index(per)

    with catch_warnings():
        filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        return _reduce_matrix("nanmean", dense, per, lambda dense: np.nanmin(dense, axis=1 - axis))


@utm.timed_call()
def nnz_per(matrix: utt.Matrix, *, per: Optional[str]) -> utt.NumpyVector:
    """
    Compute the number of non-zero values ``per`` (``row`` or ``column``) of some ``matrix``.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).

    .. note::

        If given a sparse matrix, this returns the number of **structural** non-zeros, that is, the
        number of entries we actually store data for, even if this data is zero. Use
        :py:func:`metacells.utilities.typing.eliminate_zeros` if you suspect the sparse matrix of
        containing structural zero data values.
    """
    per = _ensure_per_for("nnz", matrix, per)
    axis = utt.PER_OF_AXIS.index(per)

    sparse = utt.maybe_sparse_matrix(matrix)
    if sparse is not None:
        return _reduce_matrix("nnz", sparse, per, lambda sparse: sparse.getnnz(axis=1 - axis))

    dense = utt.to_numpy_matrix(matrix, only_extract=True)
    return _reduce_matrix(
        "nnz", dense, per, lambda dense: utt.mustbe_numpy_vector(np.count_nonzero(dense, axis=1 - axis))
    )


@utm.timed_call()
def sum_per(matrix: utt.Matrix, *, per: Optional[str]) -> utt.NumpyVector:
    """
    Compute the total of the values ``per`` (``row`` or ``column``) of some ``matrix``.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).
    """
    per = _ensure_per_for("sum", matrix, per)
    axis = utt.PER_OF_AXIS.index(per)

    sparse = utt.maybe_sparse_matrix(matrix)
    if sparse is not None:
        return _reduce_matrix("sum", sparse, per, lambda sparse: sparse.sum(axis=1 - axis))

    dense = utt.to_numpy_matrix(matrix, only_extract=True)
    return _reduce_matrix("sum", dense, per, lambda dense: utt.mustbe_numpy_vector(np.sum(dense, axis=1 - axis)))


@utm.timed_call()
def sum_squared_per(matrix: utt.Matrix, *, per: Optional[str]) -> utt.NumpyVector:
    """
    Compute the total of the squared values ``per`` (``row`` or ``column``) of some ``matrix``.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).

    .. todo::

        The :py:func:`sum_squared_per` implementation allocates and frees a complete copy of the
        matrix (to hold the squared values). An implementation that directly squares-and-adds the
        elements would be much more efficient and consume much less memory.
    """
    per = _ensure_per_for("sum_squared", matrix, per)
    axis = utt.PER_OF_AXIS.index(per)

    sparse = utt.maybe_sparse_matrix(matrix)
    if sparse is not None:
        return _reduce_matrix("sum_squared", sparse, per, lambda sparse: sparse.multiply(sparse).sum(axis=1 - axis))

    dense = utt.to_numpy_matrix(matrix, only_extract=True)
    return _reduce_matrix("sum_squared", dense, per, lambda dense: np.square(dense).sum(axis=1 - axis))


@utm.timed_call()
def rank_per(matrix: utt.Matrix, rank: int, *, per: Optional[str]) -> utt.NumpyVector:
    """
    Get the ``rank`` element ``per`` (``row`` or ``column``) of some ``matrix``.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).

    .. todo::

        This always uses the dense implementation. A sparse implementation should be faster.
    """
    per, dense = _get_dense_for("ranking", matrix, per)

    if per == "column":
        dense = dense.transpose()

    output = np.empty(dense.shape[0], dtype=dense.dtype)
    extension_name = f"rank_rows_{dense.dtype}_t"
    extension = getattr(xt, extension_name)
    extension(dense, output, rank)
    return output


@utd.expand_doc()
@utm.timed_call()
def top_per(matrix: utt.Matrix, top: int, *, per: Optional[str], ranks: bool = False) -> utt.CompressedMatrix:
    """
    Get the ``top`` elements ``per`` (``row`` or ``column``) of some ``matrix``, as a compressed
    ``per``-major matrix.

    If ``ranks`` (default: {ranks}), then fill the result with the rank of each element; Otherwise,
    just keep the original value.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).

    .. todo::

        This always uses the dense implementation. A sparse implementation should be faster.
    """
    per, dense = _get_dense_for("ranking", matrix, per)

    if per == "column":
        dense = dense.transpose()

    size = dense.shape[0]
    assert top < dense.shape[1]

    indptr = np.arange(size + 1, dtype="int64")
    indptr *= top

    if size == 1:
        with utm.timed_step(".argpartition"):
            dense_vector = utt.to_numpy_vector(dense)
            partition_indices = np.argpartition(dense_vector, len(dense_vector) - top)
            indices = partition_indices[-top:].astype("int32")
            np.sort(indices)
            data = dense_vector[indices].astype("float32")

    else:
        indices = np.empty(top * size, dtype="int32")
        data = np.empty(top * size, dtype=dense.dtype)

        with utm.timed_step("extensions.collect_top"):
            utm.timed_parameters(size=size, keep=top, dtype=str(dense.dtype))
            extension_name = f"collect_top_{dense.dtype}_t"
            extension = getattr(xt, extension_name)
            extension(top, dense, indices, data, ranks)

    top_data = sp.csr_matrix((data, indices, indptr), shape=dense.shape)
    top_data.has_sorted_indices = True
    top_data.has_canonical_format = True

    if per == "column":
        top_data = top_data.transpose()

    return top_data


@utm.timed_call()
def prune_per(compressed: utt.CompressedMatrix, top: int) -> utt.CompressedMatrix:
    """
    Keep just the ``top`` elements of some ``compressed`` matrix, per row for CSR and per column for
    CSC.
    """
    layout = utt.matrix_layout(compressed)
    if layout == "row_major":
        size = compressed.shape[0]
    else:
        assert layout == "column_major"
        size = compressed.shape[1]

    data_array = np.empty(size * top, dtype=compressed.data.dtype)
    indices_array = np.empty(size * top, dtype=compressed.indices.dtype)
    indptr_array = np.empty(1 + size, dtype=compressed.indptr.dtype)

    with utm.timed_step("extensions.collect_pruned"):
        utm.timed_parameters(size=size, nnz=compressed.nnz, keep=top)
        extension_name = "collect_pruned_%s_t_%s_t_%s_t" % (  # pylint: disable=consider-using-f-string
            compressed.data.dtype,
            compressed.indices.dtype,
            compressed.indptr.dtype,
        )
        extension = getattr(xt, extension_name)
        extension(top, compressed.data, compressed.indices, compressed.indptr, data_array, indices_array, indptr_array)

    if layout == "row_major":
        constructor = sp.csr_matrix
    else:
        constructor = sp.csc_matrix

    pruned = constructor(
        (data_array[: indptr_array[-1]], indices_array[: indptr_array[-1]], indptr_array), shape=compressed.shape
    )
    pruned.has_sorted_indices = True
    pruned.has_canonical_format = True

    return pruned


@utm.timed_call()
def quantile_per(matrix: utt.Matrix, quantile: float, *, per: Optional[str]) -> utt.NumpyVector:
    """
    Get the ``quantile`` element ``per`` (``row`` or ``column``) of some ``matrix``.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).

    .. todo::

        This always uses the dense implementation. Possibly a sparse implementation might be faster.
    """
    per, dense = _get_dense_for("quantile", matrix, per)
    axis = 1 - utt.PER_OF_AXIS.index(per)

    assert 0 <= quantile <= 1

    return np.nanquantile(dense, quantile, axis)


@utm.timed_call()
def nanquantile_per(matrix: utt.Matrix, quantile: float, *, per: Optional[str]) -> utt.NumpyVector:
    """
    Get the ``quantile`` element ``per`` (``row`` or ``column``) of some ``matrix``, ignoring
    ``None`` values.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).

    .. todo::

        This always uses the dense implementation. A sparse implementation should be more efficient.
    """
    per, dense = _get_dense_for("nanquantile", matrix, per)
    axis = 1 - utt.PER_OF_AXIS.index(per)

    assert 0 <= quantile <= 1

    return np.nanquantile(dense, quantile, axis)


@utm.timed_call()
def scale_by(matrix: utt.Matrix, scale: utt.Vector, *, by: str) -> utt.ProperMatrix:
    """
    Return a ``matrix`` where each ``by`` (``row`` or ``column``) is scaled by the matching
    value of the ``vector``.
    """
    axis = utt.PER_OF_AXIS.index(by)
    assert len(scale) == matrix.shape[axis]
    scale = utt.to_numpy_vector(scale)

    _, dense, compressed = utt.to_proper_matrices(matrix, default_layout=f"{by}_major")

    if compressed is not None:
        scale_matrix = sp.spdiags(scale, 0, len(scale), len(scale))
        if by == "row":
            result = utt.mustbe_compressed_matrix(scale_matrix * matrix)
        else:
            result = utt.mustbe_compressed_matrix(matrix * scale_matrix)
        utt.sum_duplicates(result)
        utt.eliminate_zeros(result)
        assert utt.matrix_layout(result) == utt.matrix_layout(compressed)
        return result

    assert dense is not None
    if by == "row":
        return dense * scale[:, None]
    return dense * scale[None, :]


@utm.timed_call()
def fraction_by(matrix: utt.Matrix, *, sums: Optional[utt.Vector] = None, by: str) -> utt.ProperMatrix:
    """
    Return a matrix containing, in each entry, the fraction of the original data out of the
    total ``by`` (``row`` or ``column``).

    That is, the sum of ``by`` in the result will be 1. However, if ``sums`` is specified, it is
    used instead of the sum of each ``by``, so the sum of the results may be different.

    .. note::

        This assumes all the data values are non-negative.
    """
    proper = utt.to_proper_matrix(matrix, default_layout=f"{by}_major")
    if sums is None:
        sums = sum_per(proper, per=by)
    else:
        sums = utt.to_numpy_vector(sums)

    zeros_mask = sums == 0
    scale = np.reciprocal(sums, where=~zeros_mask)
    scale[zeros_mask] = 0
    return scale_by(proper, scale, by=by)


@utm.timed_call()
def fraction_per(matrix: utt.Matrix, *, per: Optional[str]) -> utt.NumpyVector:
    """
    Get the fraction ``per`` (``row`` or ``column``) out of the total of some ``matrix``.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).
    """
    total_per_element = sum_per(matrix, per=per)
    return total_per_element / total_per_element.sum()


@utm.timed_call()
def variance_per(matrix: utt.Matrix, *, per: Optional[str]) -> utt.NumpyVector:
    """
    Get the variance ``per`` (``row`` or ``column``) of some ``matrix``.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).
    """
    per = _ensure_per_for("variance", matrix, per)
    sum_per_element = sum_per(matrix, per=per)
    sum_squared_per_element = sum_squared_per(matrix, per=per)
    axis = 1 - utt.PER_OF_AXIS.index(per)
    size = matrix.shape[axis]
    result = np.square(sum_per_element).astype(float)
    result /= -size
    result += sum_squared_per_element
    result /= size
    return result


@utm.timed_call()
def normalized_variance_per(matrix: utt.Matrix, *, per: Optional[str], zero_value: float = 1.0) -> utt.NumpyVector:
    """
    Get the normalized variance (variance / mean) ``per`` (``row`` or ``column``) of some ``matrix``.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).

    If all the values are zero, writes the ``zero_value`` (default: {zero_value}) into the result.
    """
    variance_per_element = variance_per(matrix, per=per)
    mean_per_element = mean_per(matrix, per=per)
    zeros_mask = mean_per_element == 0
    result = np.reciprocal(mean_per_element, where=~zeros_mask)
    result[zeros_mask] = 0
    result *= variance_per_element
    result[zeros_mask] = zero_value
    return result


@utm.timed_call()
def relative_variance_per(
    matrix: utt.Matrix,
    *,
    per: Optional[str],
    window_size: int,
) -> utt.NumpyVector:
    """
    Return the (log2(normalized_variance) - median(log2(normalized_variance_of_similar)) of the
    values ``per`` (``row`` or ``column``) of some ``matrix``.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).
    """
    normalized_variance_per_element = normalized_variance_per(matrix, per=per)
    np.log2(normalized_variance_per_element, out=normalized_variance_per_element)
    mean_per_element = mean_per(matrix, per=per)
    median_variance_per_element = sliding_window_function(
        normalized_variance_per_element, function="median", window_size=window_size, order_by=mean_per_element
    )
    return normalized_variance_per_element - median_variance_per_element


@utm.timed_call()
def sum_matrix(matrix: utt.Matrix) -> Any:
    """
    Compute the sum of all the values in a ``matrix``.
    """
    return np.sum(matrix)  # type: ignore


@utm.timed_call()
def nnz_matrix(matrix: utt.Matrix) -> Any:
    """
    Compute the number of non-zero entries in a ``matrix``.

    .. note::

        If given a sparse matrix, this returns the number of **structural** non-zeros, that is, the
        number of entries we actually store data for, even if this data is zero. Use
        :py:func:`metacells.utilities.typing.eliminate_zeros` if you suspect the sparse matrix of
        containing structural zero data values.
    """
    sparse = utt.maybe_sparse_matrix(matrix)
    if sparse is not None:
        return sparse.nnz
    return np.sum(matrix != 0)


@utm.timed_call()
def mean_matrix(matrix: utt.Matrix) -> Any:
    """
    Compute the mean of all the values in a ``matrix``.
    """
    return np.mean(matrix)  # type: ignore


@utm.timed_call()
def max_matrix(matrix: utt.Matrix) -> Any:
    """
    Compute the maximum of all the values in a ``matrix``.
    """
    return np.max(matrix)


@utm.timed_call()
def min_matrix(matrix: utt.Matrix) -> Any:
    """
    Compute the minimum of all the values in a ``matrix``.
    """
    return np.min(matrix)


@utm.timed_call()
def nanmean_matrix(matrix: utt.Matrix) -> Any:
    """
    Compute the mean of all the non-NaN values in a ``matrix``.
    """
    with catch_warnings():
        filterwarnings("ignore", r"All-NaN (slice|axis) encountered")

        _, dense, compressed = utt.to_proper_matrices(matrix)
        if compressed is not None:
            return np.nansum(compressed.data) / (compressed.shape[0] * compressed.shape[1])
        assert dense is not None
        return np.nanmean(dense)


@utm.timed_call()
def nanmax_matrix(matrix: utt.Matrix) -> Any:
    """
    Compute the maximum of all the non-NaN values in a ``matrix``.
    """
    with catch_warnings():
        filterwarnings("ignore", r"All-NaN (slice|axis) encountered")

        _, dense, compressed = utt.to_proper_matrices(matrix)
        if compressed is not None:
            return np.nanmax(dense)
        assert dense is not None
        return np.nanmax(dense)


@utm.timed_call()
def nanmin_matrix(matrix: utt.Matrix) -> Any:
    """
    Compute the minimum of all the non-NaN values in a ``matrix``.
    """
    with catch_warnings():
        filterwarnings("ignore", r"All-NaN (slice|axis) encountered")

        _, dense, compressed = utt.to_proper_matrices(matrix)
        if compressed is not None:
            return np.nanmin(dense)
        assert dense is not None
        return np.nanmin(dense)


@utm.timed_call()
def rank_matrix_by_layout(matrix: utt.NumpyMatrix, ascending: bool) -> Any:
    """
    Replace each element of the matrix with its rank (in row for ``row_major``,
    in column for ``column_major``).

    If ``ascending`` then rank 1 is the minimal element. Otherwise, rank 1 is the maximal element.
    """
    layout = utt.matrix_layout(matrix)
    assert layout is not None
    extension_name = f"rank_matrix_{matrix.dtype}_t"
    extension = getattr(xt, extension_name)
    extension(matrix, ascending)
    return matrix


M = TypeVar("M", bound=utt.Matrix)


def _reduce_matrix(
    _name: str,
    matrix: M,
    per: str,
    reducer: Callable[[M], utt.NumpyVector],
) -> utt.NumpyVector:
    assert matrix.ndim == 2
    axis = utt.PER_OF_AXIS.index(per)
    results_count = matrix.shape[1 - axis]

    compressed = utt.maybe_compressed_matrix(matrix)
    if compressed is None:
        timed_step = ".sparse"
        elements_count: float = matrix.shape[axis]
    else:
        _, dense, compressed = utt.to_proper_matrices(matrix, default_layout=utt.LAYOUT_OF_AXIS[axis])

        if dense is not None:
            elements_count = dense.shape[axis]
            axis_flag = (dense.flags.c_contiguous, dense.flags.f_contiguous)[axis]
            if axis_flag:
                timed_step = ".dense-efficient"
            else:
                timed_step = ".dense-inefficient"

        else:
            assert compressed is not None
            elements_count = compressed.nnz / results_count
            axis_format = ("csr", "csc")[axis]
            if compressed.getformat() == axis_format:
                timed_step = ".compressed-efficient"
            else:
                timed_step = ".compressed-inefficient"

    with utm.timed_step(timed_step):
        utm.timed_parameters(results=results_count, elements=elements_count)
        return utt.to_numpy_vector(reducer(matrix))


@utm.timed_call()
def bincount_vector(
    vector: utt.Vector,
    *,
    minlength: int = 0,
) -> utt.NumpyVector:
    """
    Drop-in replacement for ``numpy.bincount``, which is timed and works for any ``vector`` data.
    """
    dense = utt.to_numpy_vector(vector)
    result = np.bincount(dense, minlength=minlength)
    utm.timed_parameters(size=dense.size, bins=result.size)
    return result


@utm.timed_call()
def most_frequent(vector: utt.Vector) -> Any:
    """
    Return the most frequent value in a ``vector``.
    """
    unique, positions = np.unique(utt.to_numpy_vector(vector), return_inverse=True)
    counts = np.bincount(positions)
    maxpos = np.argmax(counts)
    return unique[maxpos]


@utm.timed_call()
def highest_weight(weights: utt.Vector, vector: utt.Vector) -> Any:
    """
    Return the value with the highest total ``weight`` in a ``vector``.
    """
    unique, positions = np.unique(utt.to_numpy_vector(vector), return_inverse=True)
    counts = np.bincount(positions, weights=weights)
    maxpos = np.argmax(counts)
    return unique[maxpos]


@utm.timed_call()
def fraction_of_grouped(value: Any) -> Callable[[utt.Vector], Any]:
    """
    Return a function, that takes a vector and returns the fraction of elements of the vector which
    are equal to a specific ``value``.
    """

    def compute(vector: utt.Vector) -> Any:
        return np.sum(utt.to_numpy_vector(vector) == value) / len(vector)

    return compute


ROLLING_FUNCTIONS = {
    "mean": pd.core.window.Rolling.mean,
    "median": pd.core.window.Rolling.median,
    "std": pd.core.window.Rolling.std,
    "var": pd.core.window.Rolling.var,
}

FULL_FUNCTIONS = {
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "var": np.var,
}


@utd.expand_doc(functions=", ".join([f"``{function}``" for function in ROLLING_FUNCTIONS]))
@utm.timed_call()
def sliding_window_function(
    vector: utt.Vector,
    *,
    function: str,
    window_size: int,
    order_by: Optional[utt.NumpyVector] = None,
) -> utt.NumpyVector:
    """
    Return a vector of the same size as the input ``vector``, where each entry is the result of
    applying the ``function`` (one of {functions}) to a sliding window of size ``window_size``
    centered on the entry.

    If ``order_by`` is specified, the ``vector`` is first sorted by this order, and the end result
    is unsorted back to the original order. That is, the sliding window centered at each position
    will contain the ``window_size`` of entries which have the nearest ``order_by`` values to the
    center entry.

    .. note::

        The window size should be an odd positive integer. If an even value is specified, it is
        automatically increased by one.
    """
    dense = utt.to_numpy_vector(vector)

    if window_size % 2 == 0:
        window_size += 1
    half_window_size = (window_size - 1) // 2

    utm.timed_parameters(function=function, size=dense.size, window=window_size)

    if window_size >= len(vector):
        value = FULL_FUNCTIONS[function](vector)
        return np.full(len(vector), value)

    if order_by is not None:
        assert dense.size == order_by.size
        order_indices = np.argsort(order_by)
        reverse_order_indices = np.argsort(order_indices)
    else:
        reverse_order_indices = order_indices = np.arange(dense.size)

    extended_order_indices = np.concatenate(
        [  #
            order_indices[np.arange(window_size - half_window_size, window_size)],
            order_indices,
            order_indices[np.arange(len(vector) - window_size, len(vector) - window_size + half_window_size)],
        ]
    )

    extended_series = utt.to_pandas_series(dense[extended_order_indices])
    rolling_windows = extended_series.rolling(window_size)  # type: ignore

    compute = ROLLING_FUNCTIONS[function]
    computed = compute(rolling_windows).values
    reordered = computed[window_size - 1 :]
    assert reordered.size == dense.size

    if order_by is not None:
        reordered = reordered[reverse_order_indices]

    return reordered


@utm.timed_call()
def patterns_matches(
    patterns: Union[str, Pattern, Collection[Union[str, Pattern]]],
    strings: Collection[str],
    invert: bool = False,
) -> utt.NumpyVector:
    """
    Given a collection of (case-insensitive) ``strings``, return a boolean mask specifying which of
    them match the given regular expression ``patterns``.

    If ``invert`` (default: {invert}), invert the mask.
    """
    if isinstance(patterns, (str, Pattern)):
        patterns = [patterns]

    utm.timed_parameters(patterns=len(patterns), strings=len(strings))

    unified_pattern = (
        "("
        + ")|(".join([alternative if isinstance(alternative, str) else alternative.pattern for alternative in patterns])
        + ")"
    )
    pattern: Pattern = re.compile(unified_pattern, re.IGNORECASE)

    mask = np.array([bool(pattern.match(string)) for string in strings])

    if invert:
        mask = ~mask

    return mask


@utm.timed_call()
def compress_indices(indices: utt.Vector) -> utt.NumpyVector:
    """
    Given a vector of group ``indices`` per element, return a vector where the group indices are
    consecutive.

    If the group indices contain ``-1`` ("outliers"), then it is preserved as ``-1`` in the result.
    """
    unique, consecutive = np.unique(indices, return_inverse=True)
    consecutive += min(unique[0], 0)
    return consecutive


@utm.timed_call()
def bin_pack(element_sizes: utt.Vector, max_bin_size: float) -> utt.NumpyVector:
    """
    Given a vector of ``element_sizes`` return a vector containing the bin number for each element,
    such that the total size of each bin is at most, and as close to as possible, to the
    ``max_bin_size``.

    This uses the first-fit decreasing algorithm for finding an initial solution and then moves
    elements around to minimize the l2 norm of the wasted space in each bin.
    """
    size_of_bins: List[float] = []
    element_sizes = utt.to_numpy_vector(element_sizes)
    descending_size_indices = np.argsort(element_sizes)[::-1]
    bin_of_elements = np.empty(element_sizes.size, dtype="int")

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
            remove_loss = (element_size + current_bin_space) ** 2 - current_bin_space ** 2

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
    element_sizes: utt.Vector, min_bin_size: float
) -> utt.NumpyVector:
    """
    Given a vector of ``element_sizes`` return a vector containing the bin number for each element,
    such that the total size of each bin is at least, and as close to as possible, to the
    ``min_bin_size``.

    This uses the first-fit decreasing algorithm for finding an initial solution and then moves
    elements around to minimize the l2 norm of the wasted space in each bin.
    """
    element_sizes = utt.to_numpy_vector(element_sizes)
    total_size = np.sum(element_sizes)
    assert min_bin_size > 0

    size_of_bins = [0.0]
    max_bins_count = int(total_size // min_bin_size) + 1
    while min(size_of_bins) < min_bin_size:
        max_bins_count -= 1
        if max_bins_count < 2:
            return np.zeros(element_sizes.size, dtype="int")

        size_of_bins = []
        descending_size_indices = np.argsort(element_sizes)[::-1]
        bin_of_elements = np.empty(element_sizes.size, dtype="int")

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
                bin_waste = size_of_bins[bin_index] - min_bin_size + element_size
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

            remove_gain = current_bin_waste ** 2 - (current_bin_waste - element_size) ** 2

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
    matrix: utt.ProperMatrix, groups: utt.Vector, *, per: Optional[str]
) -> Optional[Tuple[utt.Matrix, utt.NumpyVector]]:
    """
    Given a ``matrix``, and a vector of ``groups`` ``per`` column or row, return a matrix with a
    column or row ``per`` group, containing the sum of the groups columns or rows, and a vector of
    sizes (the number of summed columns or rows) ``per`` group.

    Negative group indices ("outliers") are ignored and their data is not included in the result. If
    there are no non-negative group indices, returns ``None``.

    If ``per`` is ``None``, the matrix must be square and is assumed to be symmetric, so the most
    efficient direction is used based on the matrix layout. Otherwise it must be one of ``row`` or
    ``column``, and the matrix must be in the appropriate layout (``row_major`` operating on rows,
    ``column_major`` for operating on columns).
    """
    per = _ensure_per_for("sum_groups", matrix, per)

    groups = utt.to_numpy_vector(groups)
    groups_count = np.max(groups) + 1

    if groups_count == 0:
        return None

    efficient_layout = per + "_major"

    sparse = utt.maybe_sparse_matrix(matrix)
    if sparse is not None:
        if utt.is_layout(matrix, efficient_layout):
            timed_step = ".compressed-efficient"
        else:
            timed_step = ".compressed-inefficient"
    else:
        matrix = utt.to_numpy_matrix(matrix, only_extract=True)
        if utt.is_layout(matrix, efficient_layout):
            timed_step = ".dense-efficient"
        else:
            timed_step = ".dense-inefficient"

    if per == "column":
        matrix = matrix.transpose()

    group_sizes = np.zeros(groups_count, dtype="int32")

    with utm.timed_step(timed_step):
        utm.timed_parameters(groups=groups_count, entities=matrix.shape[0], elements=matrix.shape[1])
        results = np.empty((groups_count, matrix.shape[1]), dtype=utt.shaped_dtype(matrix))

        for group_index in range(groups_count):
            group_mask = groups == group_index
            group_size = np.sum(group_mask)
            assert group_size > 0
            group_sizes[group_index] = group_size
            group_matrix = matrix[group_mask, :]
            results[group_index, :] = utt.to_numpy_vector(group_matrix.sum(axis=0))

    if per == "column":
        results = results.transpose()
    return results, group_sizes


@utm.timed_call()
def shuffle_matrix(
    matrix: utt.Matrix,
    *,
    per: str,
    random_seed: int = 0,
) -> None:
    """
    Shuffle (in-place) the ``matrix`` data ``per`` column or row.

    The matrix must be in the appropriate layout (``row_major`` for shuffling data in each row,
    ``column_major`` for shuffling data in each column).

    A non-zero ``random_seed`` (default: {random_seed}) will make the operation replicable.
    """
    _ensure_layout_for("shuffle", matrix, per, allow_inefficient=False)
    axis = utt.PER_OF_AXIS.index(per)

    _, dense, compressed = utt.to_proper_matrices(matrix)

    if compressed is not None:
        extension_name = "shuffle_compressed_%s_t_%s_t_%s_t" % (  # pylint: disable=consider-using-f-string
            compressed.data.dtype,
            compressed.indices.dtype,
            compressed.indptr.dtype,
        )
        extension = getattr(xt, extension_name)
        extension(compressed.data, compressed.indices, compressed.indptr, compressed.shape[1 - axis], random_seed)

    else:
        assert dense is not None
        if per == "column":
            dense = dense.transpose()
        extension_name = f"shuffle_dense_{dense.dtype}_t"
        extension = getattr(xt, extension_name)
        extension(dense, random_seed)


@utm.timed_call()
def cover_diameter(*, points_count: int, area: float, cover_fraction: float) -> float:
    """
    Return the diameter to give to each point so that the total area of ``points_count``
    will be a ``cover_fraction`` of the total ``area``.
    """
    return xt.cover_diameter(points_count, area, cover_fraction)


@utm.timed_call()
@utd.expand_doc()
def cover_coordinates(
    x_coordinates: utt.Vector,
    y_coordinates: utt.Vector,
    *,
    cover_fraction: float = 1 / 3,
    noise_fraction: float = 1.0,
    random_seed: int = 0,
) -> Tuple[utt.NumpyVector, utt.NumpyVector]:
    """
    Given x/y coordinates of points, move them so that the total area covered by them is
    ``cover_fraction`` (default: {cover_fraction}) of the total area of their bounding box, assuming
    each has the diameter of their minimal distance. The points are jiggled around by the
    ``noise_fraction`` of their minimal distance using the ``random_seed`` (default: {random_seed}).

    Returns new x/y coordinates vectors.
    """
    x_coordinates = utt.to_numpy_vector(x_coordinates)
    y_coordinates = utt.to_numpy_vector(y_coordinates)

    points_count = len(x_coordinates)
    assert x_coordinates.dtype == y_coordinates.dtype
    x_coordinates = utt.to_numpy_vector(x_coordinates, copy=True)
    y_coordinates = utt.to_numpy_vector(y_coordinates, copy=True)
    spaced_x_coordinates = np.full(points_count, -0.1, dtype=x_coordinates.dtype)
    spaced_y_coordinates = np.full(points_count, -0.2, dtype=y_coordinates.dtype)

    extension_name = f"cover_coordinates_{x_coordinates.dtype}_t"
    extension = getattr(xt, extension_name)
    extension(
        x_coordinates,
        y_coordinates,
        spaced_x_coordinates,
        spaced_y_coordinates,
        cover_fraction,
        noise_fraction,
        random_seed,
    )

    return spaced_x_coordinates, spaced_y_coordinates


@utm.timed_call()
def random_piles(
    elements_count: int,
    target_pile_size: int,
    *,
    random_seed: int = 0,
) -> utt.NumpyVector:
    """
    Split ``elements_count`` elements into piles of a size roughly equal to ``target_pile_size``.

    Return a vector specifying the pile index of each element.

    Specify a non-zero ``random_seed`` to make this replicable.
    """
    assert target_pile_size > 0
    piles_count = elements_count / target_pile_size

    few_piles_count = max(floor(piles_count), 1)
    many_piles_count = ceil(piles_count)

    if few_piles_count == many_piles_count:
        piles_count = few_piles_count

    else:
        few_piles_size = elements_count / few_piles_count
        many_piles_size = elements_count / many_piles_count

        few_piles_factor = few_piles_size / target_pile_size
        many_piles_factor = target_pile_size / many_piles_size

        assert few_piles_factor >= 1
        assert many_piles_factor >= 1

        if few_piles_factor < many_piles_factor:
            piles_count = few_piles_count
        else:
            piles_count = many_piles_count

    pile_of_elements_list: List[utt.NumpyVector] = []

    minimal_pile_size = floor(elements_count / piles_count)
    extra_elements = elements_count - minimal_pile_size * piles_count
    assert 0 <= extra_elements < piles_count

    if extra_elements > 0:
        pile_of_elements_list.append(np.arange(extra_elements))
    for pile_index in range(piles_count):
        pile_of_elements_list.append(np.full(minimal_pile_size, pile_index))

    pile_of_elements = np.concatenate(pile_of_elements_list)
    assert pile_of_elements.size == elements_count

    np.random.seed(random_seed)
    return np.random.permutation(pile_of_elements)


@utm.timed_call()
def group_piles(
    group_of_elements: utt.NumpyVector,
    group_of_groups: utt.NumpyVector,
) -> utt.NumpyVector:
    """
    Group some elements into piles after first grouping them, and then grouping the groups.

    Given the ``group_of_elements`` and for each such group, its larger ``group_of_groups``, compute
    the pile index of each element to be the group of the group it belongs to, and return a vector
    of the pile index of each element.

    .. note::

        Neither the ``group_of_elements`` nor the ``group_of_groups`` may contain "iutliers", that
        is, they must assign an valid group index to each element and group.
    """
    assert np.min(group_of_elements) == 0
    assert np.min(group_of_groups) == 0
    group_of_group_of_elements = group_of_groups[group_of_elements]
    return group_of_group_of_elements


@utm.timed_call()
def represent(
    goal: utt.NumpyVector,
    basis: utt.NumpyMatrix,
) -> Optional[Tuple[float, utt.NumpyVector]]:
    """
    Represent a ``goal`` vector as a weighted average of the row vectors of some ``basis`` matrix.

    This computes a non-negative weight for each matrix row, such that the sum of weights is 1,
    minimizing the distance (L2 norm) between the goal vector and the weighted average of the basis
    vectors. This is a convex problem quadratic subject to a linear constraint, so ``cvxpy`` solves
    it efficiently.

    The return value is a tuple with the score of the weights vector, and the weights vector itself.
    """
    rows_count, columns_count = basis.shape
    assert columns_count == len(goal)

    variables = cvxpy.Variable(rows_count, nonneg=True)
    constraints = [cvxpy.sum(variables) == 1]
    objective = cvxpy.norm(goal - variables @ basis, 2)

    problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
    try:
        result = problem.solve(solver="SCS")
    except cvxpy.error.SolverError:
        try:
            result = problem.solve(solver="ECOS")
        except cvxpy.error.SolverError:
            result = problem.solve(solver="QSQP")

    if result is None:
        return None
    return (float(result), utt.to_numpy_vector(variables.value))


@utm.timed_call()
def min_cut(  # pylint: disable=too-many-branches,too-many-statements
    weights: utt.Matrix,
) -> Tuple[ig.Cut, Optional[float]]:
    """
    Find the minimal cut that will split an undirected graph (with a symmetrical ``weights``
    matrix).

    Returns the ``igraph.Cut`` object describing the cut, and the scale-invariant strength of the
    cut edges. This strength is the ratio between the mean weight of an edge connecting a random
    node in each partition and the mean weight of an edge connecting two random nodes inside a
    random partition. If either of the partitions contains no edges (e.g. contains a single node),
    the strength will be ``None``.
    """
    assert weights.shape[0] == weights.shape[1]
    nodes_count = weights.shape[0]
    assert nodes_count > 1
    edges: List[Tuple[int, int]] = []
    weight_of_edges: List[float] = []

    _, dense, compressed = utt.to_proper_matrices(weights)
    if dense is not None:
        for source in range(nodes_count):
            for target in range(source):
                weight = dense[source, target]
                if weight > 0:
                    edges.append((source, target))
                    weight_of_edges.append(weight)
    else:
        assert compressed is not None
        for source in range(nodes_count):
            offsets = range(compressed.indptr[source], compressed.indptr[source + 1])
            for target, weight in zip(compressed.indices[offsets], compressed.data[offsets]):
                if target >= source:
                    break
                edges.append((source, target))
                weight_of_edges.append(weight)

    graph = ig.Graph(n=nodes_count, edges=edges)
    cut = graph.mincut(capacity=weight_of_edges)

    is_second = np.zeros(nodes_count, dtype="bool")
    is_second[cut.partition[1]] = True
    first_size = len(cut.partition[0])
    second_size = len(cut.partition[1])
    assert first_size + second_size == nodes_count

    cut_total_weight = 0.0
    first_total_weight = 0.0
    second_total_weight = 0.0
    for ((source, target), weight) in zip(edges, weight_of_edges):
        if is_second[source]:
            if is_second[target]:
                second_total_weight += weight
            else:
                cut_total_weight += weight
        else:
            if is_second[target]:
                cut_total_weight += weight
            else:
                first_total_weight += weight

    if cut_total_weight == 0:
        return (cut, 0.0)

    if first_total_weight > 0 and second_total_weight > 0:
        cut_edges_count = first_size * second_size
        first_edges_count = (first_size * (first_size - 1)) / 2
        second_edges_count = (second_size * (second_size - 1)) / 2
        total_edges_count = (nodes_count * (nodes_count - 1)) / 2
        assert cut_edges_count + first_edges_count + second_edges_count == total_edges_count

        cut_mean_weight = cut_total_weight / cut_edges_count
        first_mean_weight = first_total_weight / first_edges_count
        second_mean_weight = second_total_weight / second_edges_count
        inner_mean_weight = sqrt(first_mean_weight * second_mean_weight)

        cut_strength: Optional[float] = cut_mean_weight / inner_mean_weight
    else:
        cut_strength = None

    return (cut, cut_strength)


@utm.timed_call()
def sparsify_matrix(
    full: utt.ProperMatrix,
    min_column_max_value: float,
    min_entry_value: float,
    abs_values: bool,
) -> utt.CompressedMatrix:
    """
    Given a full matrix, return a sparse matrix such that all non-zero entries are at least ``min_entry_value``, and
    columns that have no value above ``min_column_max_value`` are set to all-zero. If ``abs_values`` consider the
    absolute values when comparing to the thresholds.
    """
    assert 0 <= min_entry_value <= min_column_max_value

    if abs_values:
        comparable = np.abs(full)  # type: ignore
    else:
        comparable = full

    max_per_column = max_per(comparable, per="column")
    low_columns = max_per_column < min_column_max_value

    sparse_comparable = utt.maybe_compressed_matrix(comparable)
    if sparse_comparable is None:
        low_entries = comparable < min_entry_value
    else:
        low_entries = ~utt.to_numpy_matrix(comparable >= min_entry_value)

    lil = sp.lil_matrix(full)
    lil[:, low_columns] = 0
    lil[low_entries] = 0

    return sp.csr_matrix(lil)
