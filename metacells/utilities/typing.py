'''
Utilities for dealing with shaped data types.

The code has to deal with many different alternative data types for what is essentially two basic
data types: 2D matrices and 1D vectors.

Specifically, we have Pandas data frames and series, Scipy sparse matrices, and Numpy
multi-dimensional arrays (not to mention the deprecated Numpy matrix type).

Python has the great ability to "duck type", so in an ideal world, we could just pretend these are
just two data types and be done. In practice, this is hopelessly broken.

First, even operations that exists for all data types sometimes have different interfaces
(as in, ``np.foo(...)`` vs. ``matrix.foo``).

Second, operating on sparse and dense data often requires completely different code paths.

This makes it very easy to write code that works today and breaks tomorrow when someone passes a
Pandas series to a function that expects a Numpy array and it just *almost* works correctly (and god
help the poor soul that mixes up a Numpy matrix with a 2d array).

"Eternal vigilance is the cost of freedom" - the solution here is to define a bunch of fake types,
which are almost entirely for the benefit of the ``mypy`` type checker (with some run-time
assertions as well).

This not only makes the code intent explicit ("explicit is better than implicit") but also allows us
to leverage ``mypy`` to catch errors such as applying a Numpy operation on a sparse matrix, etc.

To put some order in this chaos, the following concepts are used:

* :py:const:`Shaped` is any 1d or 2d data in any format we can work with. :py:const:`Matrix` is any
  2d data, and :py:const:`Vector` is any 1d data.

* We support many data types that we can't directly operate on: :py:class:`SparseMatrix` comes to
  mind, but also :py:class:`PandasSeries`, :py:class:`PandasFrame` and :py:class:`NumpyMatrix` have
  strange quirks when it comes to directly operating on them. We therefore introduce the concept of
  "proper" types, which we can safely work on, as in :py:const:`ProperShaped`,
  :py:const:`ProperMatrix` and :py:const:`ProperVector` which we can obtain by using using
  :py:func:`to_proper`, :py:func:`to_proper_matrix` :py:func:`to_proper_matrices` and
  :py:func:`to_proper_vector`.

* We provide type aliases to distinguish (proper) dense data from (proper) sparse using
  :py:const:`DenseShaped`, :py:const:`DenseMatrix`, :py:const:`DenseVector`. The proper
  sparse data is :py:const:`CompressedMatrix`.

* In general, avoid using the type names that start with ``_``.
'''

from abc import abstractmethod
from typing import Any, Optional, Tuple, Type, TypeVar, Union, overload

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

import metacells.utilities.documentation as utd
import metacells.utilities.timing as utm

__all__ = [
    'CPP_DATA_TYPES',


    'Matrix',
    'ProperMatrix',
    'DenseMatrix',
    'CompressedMatrix',
    'ImproperMatrix',
    'SparseMatrix',
    'NumpyMatrix',
    'PandasFrame',

    'Vector',
    'ProperVector',
    'DenseVector',
    'ImproperVector',
    'PandasSeries',

    'Shaped',
    'ProperShaped',
    'DenseShaped',

    'to_proper',
    'to_proper_matrix',
    'to_proper_matrices',
    'to_proper_vector',
    'canonize',

    'frozen',
    'freeze',
    'unfreeze',

    'to_dense',
    'to_dense_matrix',
    'to_dense_vector',

    'DENSE_FAST_FLAG',
    'SPARSE_FAST_FORMAT',
    'SPARSE_SLOW_FORMAT',
    'LAYOUT_OF_AXIS',

    'is_layout',
    'is_row_major',
    'is_column_major',

    'is_contiguous',
    'to_contiguous',
]


#: The data types supported by the C++ extensions code.
CPP_DATA_TYPES = ['float32', 'float64', 'int32', 'int64', 'uint32', 'uint64']


S = TypeVar('S', bound='Shaped')


# pylint: disable=missing-function-docstring
# pylint: disable=abstract-method


class Shaped:
    '''
    A ``mypy`` type for any shaped (1- or 2-dimensional) data.
    '''
    ndim: int
    shape: Union[Tuple[int, int], Tuple[int]]

    @abstractmethod
    def dtype(self) -> str: ...

    @abstractmethod
    def __getitem__(self, key: Any) -> Any: ...

    @abstractmethod
    def __setitem__(self, key: Any, value: Any) -> Any: ...

    @staticmethod
    def am(data: Any) -> bool:
        return isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)) \
            or sp.issparse(data)

    @classmethod
    def be(cls: Type[S], data: Any, name: str = 'data') -> S:
        if not cls.am(data):
            raise ValueError('%s: %s is not %s'
                             % (name, data.__class__, cls.__name__))
        return data

    @classmethod
    def maybe(cls: Type[S], data: Any) -> Optional[S]:
        if not cls.am(data):
            return None
        return data


class NumpyShaped(Shaped):
    '''
    A ``mypy`` type for numpy 1- or 2-dimensional data.
    '''
    strides: Union[Tuple[int, int], Tuple[int]]
    flags: Any

    @abstractmethod
    def setflags(self, write: bool = False) -> None: ...

    @abstractmethod
    def __lt__(self: S,  # type: ignore
               value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __le__(self: S,  # type: ignore
               value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __eq__(self: S,  # type: ignore
               value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __ne__(self: S,  # type: ignore
               value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __ge__(self: S,  # type: ignore
               value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __gt__(self: S,  # type: ignore
               value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __add__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __iadd__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __sub__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __isub__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __mul__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __imul__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __matmul__(self, value: 'NumpyShaped') -> 'NumpyShaped': ...

    @abstractmethod
    def __imatmul__(self, value: 'NumpyShaped') -> 'NumpyShaped': ...

    @abstractmethod
    def __truediv__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __itruediv__(self: S,
                     value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __floordiv__(self: S,
                     value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __ifloordiv__(self: S,
                      value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __mod__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __imod__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __pow__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __ipow__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __lshift__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __ilshift__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __rshift__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __irshift__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __and__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __iand__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __xor__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __ixor__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __or__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __ior__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __neg__(self: S) -> S: ...

    @abstractmethod
    def __pos__(self: S) -> S: ...

    @abstractmethod
    def __abs__(self: S) -> S: ...

    @abstractmethod
    def __invert__(self: S) -> S: ...

    @staticmethod
    def am(data: Any) -> bool:
        return isinstance(data, (np.ndarray, np.matrix))


class NumpyMatrix(NumpyShaped):
    '''
    A ``mypy`` type for numpy matrix data.
    '''
    shape: Tuple[int, int]
    strides: Tuple[int, int]

    @staticmethod
    def am(data: Any) -> bool:
        return isinstance(data, np.matrix)


class DenseVector(NumpyShaped):
    '''
    A ``mypy`` type for numpy 1-dimensional data.
    '''
    size: int
    shape: Tuple[int]
    strides: Tuple[int]

    @staticmethod
    def am(data: Any) -> bool:
        return isinstance(data, np.ndarray) and data.ndim == 1

    @abstractmethod
    def sum(self) -> Union[float, int]: ...

    @abstractmethod
    def argmax(self) -> int: ...


class DenseMatrix(NumpyShaped):
    '''
    A ``mypy`` type for numpy 2-dimensional data.
    '''
    shape: Tuple[int, int]
    strides: Tuple[int, int]

    @abstractmethod
    def transpose(self) -> 'DenseMatrix': ...

    @abstractmethod
    def sum(self, *, axis: int) -> DenseVector: ...

    @abstractmethod
    def argmax(self, *, axis: int) -> DenseVector: ...

    @staticmethod
    def am(data: Any) -> bool:
        return isinstance(data, np.ndarray) and data.ndim == 2


SP = TypeVar('SP', bound='SparseMatrix')


class SparseMatrix(Shaped):
    '''
    A ``mypy`` type for sparse 2-dimensional data.
    '''
    shape: Tuple[int, int]
    nnz: int

    @abstractmethod
    def getformat(self) -> str: ...

    @abstractmethod
    def toarray(self) -> DenseMatrix: ...

    @abstractmethod
    def transpose(self: SP) -> SP: ...

    @abstractmethod
    def multiply(self: SP, other: Shaped) -> SP: ...

    @abstractmethod
    def getcol(self: SP, index: int) -> SP: ...

    @abstractmethod
    def getrow(self: SP, index: int) -> SP: ...

    @abstractmethod
    def sum(self, *, axis: int) -> DenseVector: ...

    @abstractmethod
    def max(self, *, axis: int) -> DenseVector: ...

    @abstractmethod
    def min(self, *, axis: int) -> DenseVector: ...

    @abstractmethod
    def getnnz(self, *, axis: int) -> DenseVector: ...

    @abstractmethod
    def argmax(self, *, axis: int) -> DenseVector: ...

    @abstractmethod
    def maximum(self: SP, other: SP) -> SP: ...

    @abstractmethod
    def tocsr(self) -> 'CompressedMatrix': ...

    @abstractmethod
    def tocsc(self) -> 'CompressedMatrix': ...

    @staticmethod
    def am(data: Any) -> bool:
        return sp.issparse(data)


class CompressedMatrix(SparseMatrix):
    '''
    A ``mypy`` type for sparse csr/csc 2-dimensional data.
    '''
    indices: DenseVector
    indptr: DenseVector
    data: DenseVector
    has_sorted_indices: bool
    has_canonical_format: bool

    @abstractmethod
    def sum_duplicates(self) -> None: ...

    @abstractmethod
    def eliminate_zeros(self) -> None: ...

    @staticmethod
    def am(data: Any) -> bool:
        return SparseMatrix.am(data) and data.getformat() in ('csr', 'csc')


class PandasIndex(Shaped):
    '''
    A ``mypy`` type for a Pandas index.
    '''
    values: DenseVector


class PandasFrame(Shaped):
    '''
    A ``mypy`` type for Pandas 2-dimensional data.
    '''
    shape: Tuple[int, int]
    values: DenseMatrix
    index: PandasIndex
    columns: PandasIndex

    @staticmethod
    def am(data: Any) -> bool:
        return isinstance(data, pd.DataFrame)


class PandasSeries(Shaped):
    '''
    A ``mypy`` type for Pandas 1-dimensional data.
    '''
    size: int
    shape: Tuple[int]
    values: DenseVector
    index: PandasIndex

    @staticmethod
    def am(data: Any) -> bool:
        return isinstance(data, pd.Series)


# pylint: enable=missing-function-docstring
# pylint: enable=abstract-method


#: A ``mypy`` type for "proper" 2-dimensional data.
#:
#: "Proper" data allows for direct processing without having
#: to mess with its formatting.
ProperMatrix = Union[DenseMatrix, CompressedMatrix]


#: A ``mypy`` type for "improper" 2-dimensional data.
#:
#: "Improper" data contains "proper" data somewhere inside it.
ImproperMatrix = Union[NumpyMatrix, PandasFrame, SparseMatrix]


#: A ``mypy`` type for any 2-dimensional data.
Matrix = Union[ProperMatrix, ImproperMatrix]


#: A ``mypy`` type for "proper" 2-dimensional data.
#:
#: "Proper" data allows for direct processing without having
#: to mess with its formatting.
ProperVector = DenseVector


#: A ``mypy`` type for "improper" 1-dimensional data.
#:
#: "Improper" data contains "proper" data somewhere inside it.
ImproperVector = PandasSeries


#: A ``mypy`` type for any 1-dimensional data.
#:
#: .. todo::
#:
#:    Is there any need for ``SparseVector``?
Vector = Union[ProperVector, ImproperVector]


#: A ``mypy`` type for any (proper) dense data.
#:
#: Dense data has a simple one-entry-per-element representation.
DenseShaped = Union[DenseMatrix, DenseVector]


#: A ``mypy`` type for any "proper" 1- or 2-dimensional data.
#:
#: "Proper" data allows for direct processing without having
#: to mess with its formatting.
ProperShaped = Union[ProperVector, ProperMatrix]


@utd.expand_doc()
def to_proper(
    shaped: Shaped,
    *,
    ndim: Optional[int] = None,
    layout: str = 'row_major'
) -> ProperShaped:
    '''
    Given some data, access the properly-formatted data within it.

    If ``ndim`` is specified, insist this is the number of dimensions.

    If the data is in some strange sparse format, use ``layout`` (default: {layout}) to decide
    whether to return it in ``row_major`` (csr) or ``column_major`` (csc) format. Otherwise, this is
    ignored.
    '''
    assert layout in LAYOUT_OF_AXIS

    if isinstance(shaped, (pd.Series, pd.DataFrame)):
        shaped = shaped.values

    if isinstance(shaped, np.matrix):
        shaped = np.array(shaped)

    sparse = SparseMatrix.maybe(shaped)
    if sparse is not None:
        compressed = CompressedMatrix.maybe(sparse)
        if compressed is None:
            if layout == 'row_major':
                with utm.timed_step('tocsr'):
                    compressed = sparse.tocsr()
                    utm.timed_parameters(results=compressed.shape[0],
                                         elements=compressed.nnz / compressed.shape[0])
            else:
                with utm.timed_step('tocsc'):
                    compressed = sparse.tocsc()
                    utm.timed_parameters(results=compressed.shape[1],
                                         elements=compressed.nnz / compressed.shape[1])
        return compressed

    assert isinstance(shaped, np.ndarray)

    if ndim is not None and shaped.ndim != ndim:
        raise ValueError('data is %s-dimensional, expected %s-dimensional'
                         % (shaped.ndim, ndim))

    return shaped


def to_proper_matrix(matrix: Matrix, *, layout: str = 'row_major') -> ProperMatrix:
    '''
    Same as :py:func:`to_proper` but also ensures the result is a :py:const:`ProperMatrix`.
    '''
    return to_proper(matrix, ndim=2, layout=layout)  # type: ignore


def to_proper_vector(vector: Vector) -> ProperVector:
    '''
    Same as :py:func:`to_proper` but also ensures the result is a :py:const:`ProperVector`.
    '''
    return to_proper(vector, ndim=1)  # type: ignore


def to_proper_matrices(
    matrix: Matrix,
    *,
    layout: str = 'row_major'
) -> Tuple[Optional[DenseMatrix], Optional[CompressedMatrix], ProperMatrix]:
    '''
    Given some matrix, return the properly-formatted data within it using
    :py:func:`to_proper_matrix`, which may be either dense or compressed (sparse).
    '''
    proper = to_proper_matrix(matrix, layout=layout)
    return (DenseMatrix.maybe(proper), CompressedMatrix.maybe(proper), proper)


@utm.timed_call()
def canonize(matrix: Matrix) -> None:
    '''
    If the ``matrix`` is sparse, ensure it is in "canonical format", which makes most operations on
    it much more efficient.

    Otherwise, keep the matrix as-is.
    '''
    compressed = CompressedMatrix.maybe(matrix)
    if compressed is not None:
        is_frozen = frozen(compressed)
        try:
            unfreeze(compressed)
            compressed.sum_duplicates()
        finally:
            if is_frozen:
                freeze(compressed)


def frozen(shaped: Shaped) -> bool:
    '''
    Test whether the ``shaped`` data is protected against future modification.
    '''
    proper = to_proper(shaped)

    numpy = NumpyShaped.maybe(proper)
    if numpy is not None:
        return not numpy.flags.writeable

    compressed = CompressedMatrix.maybe(proper)
    if compressed is not None:
        assert compressed.indices.flags.writeable \
            == compressed.indptr.flags.writeable \
            == compressed.data.flags.writeable
        return not compressed.data.flags.writeable

    raise NotImplementedError('frozen of %s' % shaped.__class__)


@utm.timed_call()
def freeze(shaped: Shaped) -> None:
    '''
    Protect the ``shaped`` data against future modification.
    '''
    proper = to_proper(shaped)

    numpy = NumpyShaped.maybe(proper)
    if numpy is not None:
        numpy.setflags(write=False)
        return

    compressed = CompressedMatrix.maybe(proper)
    if compressed is not None:
        compressed.indices.setflags(write=False)
        compressed.indptr.setflags(write=False)
        compressed.data.setflags(write=False)
        return

    raise NotImplementedError('freeze of %s' % shaped.__class__)


@utm.timed_call()
def unfreeze(shaped: Shaped) -> None:
    '''
    Permit future modification of some ``shaped`` data.
    '''
    proper = to_proper(shaped)

    numpy = NumpyShaped.maybe(proper)
    if numpy is not None:
        numpy.setflags(write=True)
        return

    compressed = CompressedMatrix.maybe(proper)
    if compressed is not None:
        compressed.indices.setflags(write=True)
        compressed.indptr.setflags(write=True)
        compressed.data.setflags(write=True)
        return

    raise NotImplementedError('unfreeze of %s' % shaped.__class__)


def to_dense(
    shaped: Shaped,
    *,
    ndim: Optional[int] = None,
    copy: Optional[bool] = False
) -> DenseShaped:
    '''
    Convert some (possibly sparse) ``shaped`` data to an (full dense size) 1/2-dimensional numpy
    array.

    If ``ndim`` is specified, insist this is the number of dimensions.

    If ``copy``, a copy of the data is returned even if it is already a numpy array. If ``copy`` is
    ``None``, always returns the data without copying (fails on sparse data).
    '''
    proper = to_proper(shaped, ndim=ndim)

    sparse = SparseMatrix.maybe(shaped)
    if sparse is not None:
        assert copy is not None
        with utm.timed_step('toarray'):
            if sparse.ndim == 2:
                utm.timed_parameters(results=sparse.shape[0],
                                     elements=sparse.shape[1])
            else:
                utm.timed_parameters(size=sparse.shape[0])
            return sparse.toarray()

    dense = NumpyShaped.be(proper)
    if copy:
        return np.copy(dense)

    return dense  # type: ignore


def to_dense_matrix(matrix: Matrix, *, copy: Optional[bool] = False) -> DenseMatrix:
    '''
    Convert some (possibly sparse) data to an (full dense size) 2-dimensional array.

    If ``copy``, a copy of the data is returned even if it is already a 2d numpy array. If ``copy``
    is ``None``, always returns the data without copying (fails on sparse data).
    '''
    return to_dense(matrix, copy=copy)  # type: ignore


def to_dense_vector(shaped: Shaped, *, copy: Optional[bool] = False) -> DenseVector:
    '''
    Convert some (possibly sparse) data to an (full dense size) 1-dimensional array.

    This will convert a matrix where one of the dimensions has size one to a flat array.

    If ``copy``, a copy of the data is returned even if it is already a 1d numpy array. If ``copy``
    is ``None``, always returns the data without copying (fails on sparse data).
    '''
    dense = to_dense(shaped, copy=copy)

    if dense.ndim == 1:
        return dense  # type: ignore

    seen_large_dimension = False
    for size in dense.shape:
        if size == 1:
            continue
        assert not seen_large_dimension
        seen_large_dimension = True

    array = np.reshape(dense, -1)
    assert array.ndim == 1

    return array


#: Which flag indicates efficient 2D dense matrix layout.
DENSE_FAST_FLAG = dict(column_major='F_CONTIGUOUS', row_major='C_CONTIGUOUS')

#: Which format indicates efficient 2D sparse matrix layout.
SPARSE_FAST_FORMAT = dict(column_major='csc', row_major='csr')

#: Which format indicates inefficient 2D sparse matrix layout.
SPARSE_SLOW_FORMAT = dict(column_major='csr', row_major='csc')

#: The layout by the ``axis`` parameter.
LAYOUT_OF_AXIS = ('row_major', 'column_major')


def is_layout(matrix: Matrix, layout: str) -> bool:
    '''
    Test whether the matrix layout is optimized for ``column_major`` or ``row_major`` layout.
    '''
    sparse = SparseMatrix.maybe(matrix)
    if sparse is not None:
        return sparse.getformat() == SPARSE_FAST_FORMAT[layout]

    dense = DenseMatrix.be(to_proper_matrix(matrix))
    return dense.flags[DENSE_FAST_FLAG[layout]]


def is_row_major(matrix: Matrix) -> bool:
    '''
    Test whether the matrix layout is optimized for per-row (variable, gene) processing.
    '''
    return is_layout(matrix, 'row_major')


def is_column_major(matrix: Matrix) -> bool:
    '''
    Test whether the matrix layout is optimized for per-column (observation, cell) processing.
    '''
    return is_layout(matrix, 'column_major')


@overload
def matrix_layout(matrix: ProperMatrix) -> str: ...


@overload
def matrix_layout(matrix: ImproperMatrix) -> Optional[str]: ...


def matrix_layout(matrix):  # type: ignore
    '''
    Return which layout the matrix is optimized for (``row_major`` or ``column_major``).

    .. note::

        The matrix may be neither one, in which case this returns ``None``.
    '''
    sparse = SparseMatrix.maybe(matrix)
    if sparse is not None:
        for layout, sparse_format in SPARSE_FAST_FORMAT.items():
            if sparse.getformat() == sparse_format:
                return layout
        return None

    dense = DenseMatrix.be(to_proper_matrix(matrix))
    for layout, flag in DENSE_FAST_FLAG.items():
        if dense.flags[flag]:
            return layout
    return None


def is_contiguous(vector: Vector) -> bool:
    '''
    Return whether the vector is contiguous in memory.

    This is only ``True`` for a dense vector.
    '''
    proper = to_proper_vector(vector)
    dense = DenseVector.be(proper)
    return dense.flags.c_contiguous and dense.flags.f_contiguous


def to_contiguous(vector: Vector, *, copy: bool = False) -> DenseVector:
    '''
    Return the vector in contiguous (dense) format.

    If ``copy``, a copy of the vector is returned even if it is already a contiguous numpy array.
    '''
    proper = to_proper_vector(vector)
    dense = DenseVector.be(proper)
    if copy or not dense.flags.c_contiguous or not dense.flags.f_contiguous:
        dense = np.copy(dense)
    return dense
