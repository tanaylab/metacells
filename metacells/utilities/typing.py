'''
Typing
------

The code has to deal with many different alternative data types for what is essentially two basic
data types: 2D matrices and 1D vectors.

Specifically, we have pandas data frames and series, Scipy sparse matrices, and numpy
multi-dimensional arrays (not to mention the deprecated numpy matrix type).

Python has the great ability to "duck type", so in an ideal world, we could just pretend these are
just two data types and be done. In practice, this is hopelessly broken.

First, even operations that exists for all data types sometimes have different interfaces
(as in, ``np.foo(...)`` vs. ``matrix.foo``).

Second, operating on sparse and dense data often requires completely different code paths.

This makes it very easy to write code that works today and breaks tomorrow when someone passes a
pandas series to a function that expects a numpy array and it just *almost* works correctly (and god
help the poor soul that mixes up a numpy matrix with a 2d array).

"Eternal vigilance is the cost of freedom" - the solution here is to define a bunch of fake types,
which are almost entirely for the benefit of the ``mypy`` type checker (with some run-time
assertions as well).

This not only makes the code intent explicit ("explicit is better than implicit") but also allows us
to leverage ``mypy`` to catch errors such as applying a numpy operation on a sparse matrix, etc.

To put some order in this chaos, the following concepts are used:

* :py:const:`Shaped` is any 1d or 2d data in any format we can work with. :py:const:`Matrix` is
  any 2d data, and :py:const:`Vector` is any 1d data.

* For 2D data, we allow multiple data types that we can't directly operate on:
  most :py:class:`SparseMatrix` layouts, :py:class:`PandasFrame` and ``np.matrix`` have
  strange quirks when it comes to directly operating on them and should be avoided, while CSR and
  CSC :py:class:`CompressedMatrix` sparse matrices and properly-laid-out 2D numpy arrays
  :py:const:`NumpyMatrix` are in general well-behaved. We therefore introduce the concept of
  :py:const:`ImproperMatrix` and :py:const:`ProperMatrix` types, and provide functions that
  manipulate whether the "proper" data is in row-major or column-major order.

* For 1D data, we just distinguish between :py:class:`PandasSeries` and 1D numpy
  :py:const:`NumpyVector` arrays, as these are the only types we allow. In theory we could have also
  allowed for sparse vectors but mercifully these are very uncommon so we can just ignore them.
'''

from contextlib import contextmanager
from typing import (Any, Collection, Iterator, Optional, Sized, Tuple, TypeVar,
                    Union)

import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore
from typing_extensions import Protocol

import metacells.utilities.documentation as utd
import metacells.utilities.timing as utm

__all__ = [
    'CPP_DATA_TYPES',

    'Shaped',
    'ProperShaped',
    'ImproperShaped',

    'Matrix',
    'ProperMatrix',
    'NumpyMatrix',
    'CompressedMatrix',
    'ImproperMatrix',
    'SparseMatrix',
    'PandasFrame',

    'Vector',
    'NumpyVector',
    'ImproperVector',
    'PandasSeries',
    'PandasCategorical',

    'is_1d',
    'is_2d',

    'maybe_numpy_vector',
    'maybe_numpy_matrix',
    'maybe_sparse_matrix',
    'maybe_compressed_matrix',
    'maybe_pandas_frame',
    'maybe_pandas_series',

    'mustbe_numpy_vector',
    'mustbe_numpy_matrix',
    'mustbe_sparse_matrix',
    'mustbe_compressed_matrix',
    'mustbe_pandas_frame',
    'mustbe_pandas_series',

    'to_proper_matrix',
    'to_proper_matrices',
    'to_pandas_series',
    'to_pandas_frame',

    'frozen',
    'freeze',
    'unfreeze',
    'unfrozen',

    'to_numpy_matrix',
    'to_numpy_vector',

    'DENSE_FAST_FLAG',
    'SPARSE_FAST_FORMAT',
    'SPARSE_SLOW_FORMAT',
    'LAYOUT_OF_AXIS',
    'PER_OF_AXIS',

    'matrix_layout',

    'is_layout',
    'is_contiguous',
    'to_contiguous',

    'is_canonical',
    'eliminate_zeros',
    'sort_indices',
    'sum_duplicates',

    'shaped_checksum',
]


#: Numpy 2-dimensional data.
#:
#: .. note::
#:
#:    This is not to be confused with :py:const:`np.matrix` which must not be used, but is returned
#:    by the occasional function and would wreak havoc on the semantics of some operations unless
#:    immediately concerted to a proper ``NumpyMatrix``, which is a simple 2-dimensional
#:    ``ndarray``.
NumpyMatrix = np.ndarray  # Should be: Annotated[np.ndarray, NDim(2)]

#: Numpy 1-dimensional data.
NumpyVector = np.ndarray  # Should be: Annotated[np.ndarray, NDim(1)]


# pylint: disable=missing-function-docstring


S = TypeVar('S', bound='ShapedProtocol')


class ShapedProtocol(Protocol):
    '''
    A ``mypy`` type for any shaped (1- or 2-dimensional, proper or improper) data.
    '''
    ndim: int
    shape: Union[Tuple[int, int], Tuple[int]]

    def dtype(self) -> np.dtype: ...

    def __getitem__(self, key: Any) -> Any: ...

    def __setitem__(self, key: Any, value: Any) -> Any: ...

    def transpose(self: S) -> S: ...


SP = TypeVar('SP', bound='SparseMatrix')


class SparseMatrix(ShapedProtocol, Protocol):
    '''
    A ``mypy`` type for sparse 2-dimensional data.

    Should have been ``SparseMatrix = sp.base.spmatrix``.
    '''
    shape: Tuple[int, int]
    nnz: int

    def getformat(self) -> str: ...

    def toarray(self) -> NumpyMatrix: ...

    def multiply(self: SP, other: ShapedProtocol) -> SP: ...

    def getcol(self: SP, index: int) -> SP: ...

    def getrow(self: SP, index: int) -> SP: ...

    def sum(self, *, axis: int) -> NumpyVector: ...

    def max(self, *, axis: int) -> NumpyVector: ...

    def nanmax(self, *, axis: int) -> NumpyVector: ...

    def min(self, *, axis: int) -> NumpyVector: ...

    def nanmin(self, *, axis: int) -> NumpyVector: ...

    def getnnz(self, *, axis: int) -> NumpyVector: ...

    def argmax(self, *, axis: int) -> NumpyVector: ...

    def maximum(self: SP, other: SP) -> SP: ...

    def tocsr(self) -> 'CompressedMatrix': ...

    def tocsc(self) -> 'CompressedMatrix': ...

    def nonzero(self) -> Tuple[np.ndarray, np.ndarray]: ...

    def mean(self: SP, *, axis: int) -> 'np.ndarray': ...

    def copy(self: SP) -> SP: ...


class CompressedMatrix(SparseMatrix, Protocol):
    '''
    A ``mypy`` type for sparse CSR/CSC 2-dimensional data.

    Should have been ``CompressedMatrix = sp.compressed._cs_matrix``.
    '''
    indices: np.ndarray
    indptr: np.ndarray
    data: np.ndarray
    has_sorted_indices: bool
    has_canonical_format: bool

    def sum_duplicates(self) -> None: ...

    def eliminate_zeros(self) -> None: ...

    def sort_indices(self) -> None: ...


class PandasIndex(ShapedProtocol, Sized, Protocol):
    '''
    A ``mypy`` type for a pandas index.
    '''
    values: NumpyVector


class PandasFrame(ShapedProtocol, Protocol):
    '''
    A ``mypy`` type for pandas 2-dimensional data.

    Should have been ``PandasFrame = pd.DataFrame``.
    '''
    shape: Tuple[int, int]
    values: NumpyMatrix
    index: PandasIndex
    columns: PandasIndex

    def __delitem__(self, key: Any) -> None: ...

    def __constraints__(self, key: Any) -> bool: ...


class PandasSeries(ShapedProtocol, Sized, Protocol):
    '''
    A ``mypy`` type for pandas 1-dimensional data.

    Should have been ``PandasSeries = pd.Series``.
    '''
    size: int
    shape: Tuple[int]
    values: NumpyVector
    index: PandasIndex


class PandasCategorical(ShapedProtocol, Sized):
    '''
    A ``mypy`` type for pandas 1-dimensional categorical data.
    '''
    size: int
    shape: Tuple[int]
    codes: NumpyVector


# pylint: enable=missing-function-docstring


#: The data types supported by the C++ extensions code.
CPP_DATA_TYPES = ['float32', 'float64', 'int32', 'int64', 'uint32', 'uint64']

#: A ``mypy`` type for "proper" 2-dimensional data.
#:
#: "Proper" data allows for direct processing without having
#: to mess with its formatting.
ProperMatrix = Union[NumpyMatrix, CompressedMatrix]

#: A ``mypy`` type for "improper" 2-dimensional data.
#:
#: "Improper" data contains "proper" data somewhere inside it.
ImproperMatrix = Union[PandasFrame, SparseMatrix]

#: A ``mypy`` type for any 2-dimensional data.
Matrix = Union[ProperMatrix, ImproperMatrix]

#: An "improper" 1-dimensional data.
ImproperVector = Union[Collection[int],
                       Collection[float], PandasSeries, PandasCategorical]

#: A ``mypy`` type for any 1-dimensional data.
#:
#: .. todo::
#:
#:    Is there any need for ``SparseVector``?
Vector = Union[NumpyVector, ImproperVector]

#: A "proper" 1- or 2-dimensional data.
ProperShaped = Union[ProperMatrix, NumpyVector]

#: An "improper" 1- or 2- dimensional data.
ImproperShaped = Union[ImproperMatrix, ImproperVector]

#: Shaped data of any of the types we can deal with.
Shaped = Union[ProperShaped, ImproperShaped]

#: Pandas data in various types.
PandasShaped = Union[PandasFrame, PandasSeries, PandasCategorical]


def is_1d(shaped: Shaped) -> bool:
    '''
    Test whether the ``shaped`` is 1-dimensional.
    '''
    return not hasattr(shaped, 'ndim') or getattr(shaped, 'ndim') == 1


def is_2d(shaped: Shaped) -> bool:
    '''
    Test whether the ``shaped`` is 2-dimensional.
    '''
    return hasattr(shaped, 'ndim') and getattr(shaped, 'ndim') == 2


def maybe_numpy_vector(shaped: Any) -> Optional[NumpyVector]:
    '''
    Test whether ``shaped`` is a :py:const:`NumpyVector`.
    '''
    if isinstance(shaped, np.ndarray) and shaped.ndim == 1:
        return shaped
    return None


def maybe_numpy_matrix(shaped: Any) -> Optional[NumpyMatrix]:
    '''
    Test whether ``shaped`` is a :py:const:`NumpyMatrix`.

    .. note::

        This looks for a 2-d ``np.ndarray`` which is **not** an ``np.matrix``.
    '''
    if isinstance(shaped, np.ndarray) and shaped.ndim == 2 \
            and not isinstance(shaped, np.matrix):
        return shaped
    return None


def maybe_sparse_matrix(shaped: Any) -> Optional[SparseMatrix]:
    '''
    Test whether ``shaped`` is a :py:const:`SparseMatrix`.

    .. note::

        This will succeed for a :py:const:`CompressedMatrix`.
    '''
    if isinstance(shaped, sp.base.spmatrix):
        return shaped
    return None


def maybe_compressed_matrix(shaped: Any) -> Optional[CompressedMatrix]:
    '''
    Test whether ``shaped`` is a :py:const:`CompressedMatrix`.
    '''
    if isinstance(shaped,
                  sp.compressed._cs_matrix):  # pylint: disable=protected-access
        return shaped
    return None


def maybe_pandas_series(shaped: Any) -> Optional[PandasSeries]:
    '''
    Test whether ``shaped`` is a :py:const:`PandasSeries`.
    '''
    if isinstance(shaped, pd.Series):
        return shaped
    return None


def maybe_pandas_frame(shaped: Any) -> Optional[PandasFrame]:
    '''
    Test whether ``shaped`` is a :py:const:`PandasFrame`.
    '''
    if isinstance(shaped, pd.DataFrame):
        return shaped
    return None


def mustbe_numpy_vector(shaped: Any) -> NumpyVector:
    '''
    Test whether ``shaped`` is a :py:const:`NumpyVector`.
    '''
    assert isinstance(shaped, np.ndarray) and shaped.ndim == 1
    return shaped


def mustbe_numpy_matrix(shaped: Any) -> NumpyMatrix:
    '''
    Assert that ``shaped`` is a :py:const:`NumpyMatrix`.

    .. note::

        This looks for a 2-d ``np.ndarray`` which is **not** an ``np.matrix``.
    '''
    assert isinstance(shaped, np.ndarray) and shaped.ndim == 2 \
        and not isinstance(shaped, np.matrix)
    return shaped


def mustbe_sparse_matrix(shaped: Any) -> SparseMatrix:
    '''
    Assert that ``shaped`` is a :py:const:`SparseMatrix`.

    .. note::

        This will succeed for a :py:const:`CompressedMatrix`.
    '''
    assert isinstance(shaped, sp.base.spmatrix)
    return shaped


def mustbe_compressed_matrix(shaped: Any) -> CompressedMatrix:
    '''
    Assert that ``shaped`` is a :py:const:`CompressedMatrix`.
    '''
    assert isinstance(shaped,
                      sp.compressed._cs_matrix)  # pylint: disable=protected-access
    return shaped


def mustbe_pandas_series(shaped: Any) -> PandasSeries:
    '''
    Assert that ``shaped`` is a :py:const:`PandasSeries`.
    '''
    assert isinstance(shaped, pd.Series)
    return shaped


def mustbe_pandas_frame(shaped: Any) -> PandasFrame:
    '''
    Assert that ``shaped`` is a :py:const:`PandasFrame`.
    '''
    assert isinstance(shaped, pd.DataFrame)
    return shaped


@utd.expand_doc()
def to_proper_matrix(
    matrix: Matrix,
    *,
    default_layout: str = 'row_major'
) -> ProperMatrix:
    '''
    Given some 2D ``matrix``, return in in a :py:const:`ProperMatrix` format.

    If the data is in some strange sparse format, use ``default_layout`` (default: {default_layout})
    to decide whether to return it in ``row_major`` (CSR) or ``column_major`` (CSC) layout.
    Otherwise, this is ignored.
    '''
    if matrix.ndim != 2:
        raise \
            ValueError(f'data is {matrix.ndim}-dimensional, '
                       'expected 2-dimensional')

    if default_layout not in LAYOUT_OF_AXIS:
        raise ValueError(f'invalid default layout: {default_layout}')

    frame = maybe_pandas_frame(matrix)
    if frame is not None:
        matrix = frame.values
        if isinstance(matrix, pd.core.arrays.categorical.Categorical):
            matrix = np.array(matrix)

    compressed = maybe_compressed_matrix(matrix)
    if compressed is not None:
        return compressed

    sparse = maybe_sparse_matrix(matrix)
    if sparse is not None:
        if default_layout == 'column_major':
            with utm.timed_step('matrix.tocsc'):
                utm.timed_parameters(results=sparse.shape[1],
                                     elements=sparse.nnz / sparse.shape[1])
                return sparse.tocsc()

        with utm.timed_step('matrix.tocsr'):
            utm.timed_parameters(results=sparse.shape[0],
                                 elements=sparse.nnz / sparse.shape[0])
            return sparse.tocsr()

    dense = maybe_numpy_matrix(matrix)
    if dense is None:
        dense = np.asarray(matrix)

    return dense


def to_proper_matrices(
    matrix: Matrix,
    *,
    default_layout: str = 'row_major'
) -> Tuple[ProperMatrix, Optional[NumpyMatrix], Optional[CompressedMatrix]]:
    '''
    Given some matrix, return the properly-formatted data within it using
    :py:func:`to_proper_matrix`, which may be either dense or compressed (sparse).
    '''
    proper = to_proper_matrix(matrix, default_layout=default_layout)
    dense = maybe_numpy_matrix(proper)
    compressed = maybe_compressed_matrix(proper)
    assert (dense is None) or (compressed is None)
    assert (dense is None) != (compressed is None)
    return (proper, dense, compressed)


def to_pandas_series(
    vector: Optional[Vector] = None,
    *,
    index: Optional[Vector] = None,
) -> PandasSeries:
    '''
    Construct a pandas series from a vector.
    '''
    if vector is None:
        return pd.Series(index=index)

    return pd.Series(to_numpy_vector(vector), index=index)


def to_pandas_frame(
    matrix: Optional[Matrix] = None,
    *,
    index: Optional[Vector] = None,
    columns: Optional[Vector] = None
) -> PandasFrame:
    '''
    Construct a pandas frame from a matrix.
    '''
    if matrix is None:
        return pd.DataFrame(index=index, columns=columns)

    sparse = maybe_sparse_matrix(matrix)
    if sparse is not None:
        return pd.DataFrame.from_spmatrix(sparse, index=index, columns=columns)

    return pd.DataFrame(to_numpy_matrix(matrix, only_extract=True),
                        index=index, columns=columns)


def frozen(shaped: Union[ProperShaped, PandasShaped]) -> bool:
    '''
    Test whether the ``shaped`` data is protected against future modification.
    '''
    compressed = maybe_compressed_matrix(shaped)
    if compressed is not None:
        assert compressed.indices.flags.writeable \
            == compressed.indptr.flags.writeable \
            == compressed.data.flags.writeable
        return not compressed.data.flags.writeable

    if isinstance(shaped, (pd.DataFrame, pd.Series)):
        shaped = shaped.values

    if isinstance(shaped, pd.core.arrays.categorical.Categorical):
        shaped = shaped.codes

    if isinstance(shaped, np.ndarray):
        return not shaped.flags.writeable

    raise NotImplementedError('frozen of %s' % shaped.__class__)


def freeze(shaped: Union[ProperShaped, PandasShaped]) -> None:
    '''
    Protect the ``shaped`` data against future modification.
    '''
    compressed = maybe_compressed_matrix(shaped)
    if compressed is not None:
        compressed.indices.setflags(write=False)
        compressed.indptr.setflags(write=False)
        compressed.data.setflags(write=False)
        return

    if isinstance(shaped, (pd.DataFrame, pd.Series)):
        shaped = shaped.values

    if isinstance(shaped, pd.core.arrays.categorical.Categorical):
        shaped = shaped.codes

    if isinstance(shaped, np.ndarray):
        shaped.setflags(write=False)
        return

    raise NotImplementedError('freeze of %s' % shaped.__class__)


def unfreeze(shaped: Union[ProperShaped, PandasShaped]) -> None:
    '''
    Permit future modification of some ``shaped`` data.
    '''
    compressed = maybe_compressed_matrix(shaped)
    if compressed is not None:
        compressed.indices.setflags(write=True)
        compressed.indptr.setflags(write=True)
        compressed.data.setflags(write=True)
        return

    if isinstance(shaped, (pd.DataFrame, pd.Series)):
        shaped = shaped.values

    if isinstance(shaped, pd.core.arrays.categorical.Categorical):
        shaped = shaped.codes

    if isinstance(shaped, np.ndarray):
        shaped.setflags(write=True)
        return

    raise NotImplementedError('unfreeze of %s' % shaped.__class__)


@contextmanager
def unfrozen(proper: ProperShaped) -> Iterator[None]:
    '''
    Execute some in-place modification, temporarily unfreezing the ``proper`` shaped data.
    '''
    is_frozen = frozen(proper)
    if is_frozen:
        unfreeze(proper)

    try:
        yield

    finally:
        if is_frozen:
            freeze(proper)


@utd.expand_doc()
def to_numpy_matrix(
    matrix: Matrix,
    *,
    default_layout: str = 'row_major',
    copy: Optional[bool] = False,
    only_extract: bool = False,
) -> NumpyMatrix:
    '''
    Convert some (possibly improper) ``matrix`` data to a dense 2-dimensional numpy array.

    If ``copy`` (default: {copy}), a copy of the data is returned even if it is already a numpy
    array. If ``copy`` is ``None``, always returns the data without copying (fails on sparse data).

    If ``only_extract`` (default: {only_extract}) then this is only expected to extract the dense
    data out of some wrapper (typically pandas data).
    '''
    sparse = maybe_sparse_matrix(matrix)
    if sparse is not None:
        assert copy is not None
        assert not only_extract
        with utm.timed_step('sparse.toarray'):
            utm.timed_parameters(results=sparse.shape[0],
                                 elements=sparse.shape[1])
            return sparse.toarray()

    dense = to_proper_matrix(matrix, default_layout=default_layout)

    if copy and id(dense) == id(matrix):
        dense = np.copy(dense)

    return mustbe_numpy_matrix(dense)


@utd.expand_doc()
def to_numpy_vector(
    shaped: Shaped,
    *,
    copy: Optional[bool] = False,
    only_extract: bool = False,
) -> NumpyVector:
    '''
    Convert some (possibly improper) data to a dense 1-dimensional numpy array.

    This will convert a matrix where one of the dimensions has size one to a flat array.

    If ``copy`` (default: {copy}), a copy of the data is returned even if it is already a 1d numpy
    array. If ``copy`` is ``None``, always returns the data without copying (fails on sparse data).

    If ``only_extract`` (default: {only_extract}) then this is only expected to extract the dense
    data out of some wrapper (typically pandas data).
    '''
    if not hasattr(shaped, 'ndim'):
        dense = np.array(shaped)

    elif shaped.ndim == 1:  # type: ignore
        series = maybe_pandas_series(shaped)
        if series is not None:
            shaped = series.values

        if isinstance(shaped, pd.core.arrays.categorical.Categorical):
            shaped = np.array(shaped)

        dense = shaped  # type: ignore

    else:
        assert shaped.ndim == 2  # type: ignore
        assert shaped.shape[0] == 1 or shaped.shape[1] == 1  # type: ignore
        dense = to_numpy_matrix(shaped, copy=copy,  # type: ignore
                                only_extract=only_extract)
        dense = np.reshape(dense, -1)

    if copy and id(dense) == id(shaped):
        dense = np.copy(dense)

    return mustbe_numpy_vector(dense)


#: Which flag indicates efficient 2D dense matrix layout.
DENSE_FAST_FLAG = dict(column_major='F_CONTIGUOUS', row_major='C_CONTIGUOUS')

#: Which format indicates efficient 2D sparse matrix layout.
SPARSE_FAST_FORMAT = dict(column_major='csc', row_major='csr')

#: Which format indicates inefficient 2D sparse matrix layout.
SPARSE_SLOW_FORMAT = dict(column_major='csr', row_major='csc')

#: The layout by the ``axis`` parameter.
LAYOUT_OF_AXIS = ('row_major', 'column_major')

#: When reducing data, get results ``per`` row or column (by the ``axis`` parameter).
PER_OF_AXIS = ('row', 'column')


def is_layout(matrix: Matrix, layout: Optional[str]) -> bool:
    '''
    Test whether the ``layout`` of the ``matrix`` is one of ``column_major`` or ``row_major``.
    '''
    if layout is None:
        return True

    sparse = maybe_sparse_matrix(matrix)
    if sparse is not None:
        return sparse.getformat() == SPARSE_FAST_FORMAT[layout]

    dense = to_numpy_matrix(matrix, only_extract=True)
    return dense.flags[DENSE_FAST_FLAG[layout]]


def matrix_layout(matrix: Matrix) -> Optional[str]:
    '''
    Return which layout the ``matrix`` is optimized for (``row_major`` or ``column_major``).

    .. note::

        The matrix may be in neither layout, in which case this returns ``None``.
    '''
    sparse = maybe_sparse_matrix(matrix)
    if sparse is not None:
        for layout, sparse_format in SPARSE_FAST_FORMAT.items():
            if sparse.getformat() == sparse_format:
                return layout
        return None

    dense = to_numpy_matrix(matrix, only_extract=True)
    for layout, flag in DENSE_FAST_FLAG.items():
        if dense.flags[flag]:
            return layout

    return None


def is_contiguous(vector: Vector) -> bool:
    '''
    Return whether the ``vector`` is contiguous in memory.

    This is only ``True`` for a dense vector.
    '''
    dense = to_numpy_vector(vector)
    return dense.flags.c_contiguous and dense.flags.f_contiguous


def to_contiguous(vector: Vector, *, copy: bool = False) -> NumpyVector:
    '''
    Return the ``vector`` in contiguous (dense) format.

    If ``copy`` (default: {copy}), a copy of the vector is returned even if it is already a
    contiguous numpy array.
    '''
    dense = to_numpy_vector(vector)
    if copy or not dense.flags.c_contiguous or not dense.flags.f_contiguous:
        dense = np.copy(dense)
    return dense


def is_canonical(shaped: Shaped) -> bool:  # pylint: disable=too-many-return-statements
    '''
    Return whether the data is in proper canonical format.

    For numpy matrices or vectors, this means the data is contiguous
    (for matrices, in either row-major or column-major order).

    For sparse matrices, it means the data is in COO format, or compressed (in CSC or CSR format)
    with sorted indices and no duplicates.

    In general, we'd like all the data stored in the annotated data to be canonical.
    '''
    if not hasattr(shaped, 'ndim'):
        return False

    if shaped.ndim == 1:  # type: ignore
        return is_contiguous(shaped)  # type: ignore

    matrix: Matrix = shaped  # type: ignore
    sparse = maybe_sparse_matrix(matrix)
    if sparse is None:
        return matrix_layout(matrix) is not None

    if sparse.getformat() not in ('coo', 'csr', 'csc'):
        return False

    if hasattr(sparse, 'has_canonical_format') \
            and not getattr(sparse, 'has_canonical_format'):
        return False

    if hasattr(sparse, 'has_sorted_indices') \
            and not getattr(sparse, 'has_sorted_indices'):
        return False

    compressed = maybe_compressed_matrix(matrix)
    if compressed is not None:
        return compressed.indptr[-1] == compressed.data.size \
            and is_canonical(compressed.indptr) \
            and is_canonical(compressed.data) \
            and is_canonical(compressed.indices)

    return True


@utm.timed_call('sparse.eliminate_zeros')
def eliminate_zeros(compressed: CompressedMatrix) -> None:
    '''
    Eliminate zeros in a compressed matrix.
    '''
    with unfrozen(compressed):
        utm.timed_parameters(before=compressed.nnz)
        compressed.eliminate_zeros()
        utm.timed_parameters(after=compressed.nnz)


@utm.timed_call('sparse.sort_indices')
def sort_indices(compressed: CompressedMatrix) -> None:
    '''
    Ensure the indices are sorted in each row/column.
    '''
    with unfrozen(compressed):
        utm.timed_parameters(before=compressed.nnz)
        compressed.sort_indices()
        utm.timed_parameters(after=compressed.nnz)


@utm.timed_call('sparse.sum_duplicates')
def sum_duplicates(compressed: CompressedMatrix) -> None:
    '''
    Eliminate duplicates in a compressed matrix.
    '''
    with unfrozen(compressed):
        utm.timed_parameters(before=compressed.nnz)
        compressed.sum_duplicates()
        utm.timed_parameters(after=compressed.nnz)


def shaped_checksum(shaped: Shaped) -> float:
    '''
    Return a checksum of the contents of ``shaped`` data (for debugging reproducibility).
    '''
    if is_1d(shaped):
        values = to_numpy_vector(shaped)
    else:
        values = to_numpy_matrix(shaped).flatten()  # type: ignore
    return np.sum(values.astype('float64') * (1 + np.arange(len(values))))
