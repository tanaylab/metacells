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

* :py:const:`AnyShaped` is any 1d or 2d data in any format we can work with. :py:const:`Matrix` is
  any 2d data, and :py:const:`Vector` is any 1d data.

* For 2D data, we allow multiple data types that we can't directly operate on:
  most :py:class:`SparseMatrix` layouts, :py:class:`PandasFrame` and :py:class:`NumpyMatrix` have
  strange quirks when it comes to directly operating on them and should be avoided, while CSR and
  CSC :py:class:`CompressedMatrix` sparse matrices and properly-laid-out 2D numpy arrays
  :py:class:`DenseMatrix` are in general well-behaved. We therefore introduce the concept of
  :py:const:`ImproperMatrix` and :py:const:`ProperMatrix` types, and provide functions that
  manipulate whether the "proper" data is in row-major or column-major order.

* For 1D data, we just distinguish between :py:class:`PandasSeries` and 1D numpy
  :py:class:`DenseVector` arrays, as these are the only types we allow. In theory we could have also
  allowed for sparse vectors but mercifully these are very uncommon so we can just ignore them.
'''

from abc import abstractmethod
from contextlib import contextmanager
from typing import (Any, Iterable, Iterator, Optional, Sized, Tuple, Type,
                    TypeVar, Union, overload)

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore

import metacells.utilities.documentation as utd
import metacells.utilities.timing as utm

__all__ = [
    'CPP_DATA_TYPES',

    'Shaped',
    'AnyShaped',
    'ProperShaped',
    'ImproperShaped',
    'NumpyShaped',
    'DenseShaped',

    'Matrix',
    'ProperMatrix',
    'ImproperMatrix',
    'DenseMatrix',
    'SparseMatrix',
    'CompressedMatrix',
    'NumpyMatrix',
    'PandasFrame',

    'Vector',
    'PandasSeries',
    'DenseVector',

    'to_proper_matrix',
    'to_proper_matrices',

    'frozen',
    'freeze',
    'unfreeze',
    'unfrozen',

    'to_dense_matrix',
    'to_dense_vector',

    'DENSE_FAST_FLAG',
    'SPARSE_FAST_FORMAT',
    'SPARSE_SLOW_FORMAT',
    'LAYOUT_OF_AXIS',
    'PER_OF_AXIS',

    'matrix_layout',

    'is_layout',
    'is_row_major',
    'is_column_major',

    'is_contiguous',
    'to_contiguous',

    'is_canonical',
    'eliminate_zeros',
    'sort_indices',
    'sum_duplicates',
]


#: The data types supported by the C++ extensions code.
CPP_DATA_TYPES = ['float32', 'float64', 'int32', 'int64', 'uint32', 'uint64']


S = TypeVar('S', bound='Shaped')


# pylint: disable=missing-function-docstring
# pylint: disable=abstract-method


class Shaped:
    '''
    A ``mypy`` type for any shaped (1- or 2-dimensional, proper or improper) data.
    '''
    ndim: int
    shape: Union[Tuple[int, int], Tuple[int]]

    @abstractmethod
    def dtype(self) -> str: ...

    @abstractmethod
    def __getitem__(self, key: Any) -> Any: ...

    @abstractmethod
    def __setitem__(self, key: Any, value: Any) -> Any: ...

    @abstractmethod
    def transpose(self: S) -> S: ...

    @staticmethod
    def am(data: Any) -> bool:
        '''
        Check whether the ``data`` is ``Shaped``.
        '''
        return isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)) \
            or sp.issparse(data)

    @classmethod
    def be(cls: Type[S], data: Any, name: str = 'data') -> S:
        '''
        Return the ``data`` in the particular ``cls`` format.
        '''
        if not cls.am(data):
            raise ValueError('%s: %s is not  %s'
                             % (name, data.__class__, cls.__name__))
        return data

    @classmethod
    def maybe(cls: Type[S], data: Any) -> Optional[S]:
        '''
        Return the ``data`` in the particular ``cls`` format, if it already is of that class.
        '''
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
    def __iadd__(self: S, value: Union[S, float, int]) -> S: ...

    @abstractmethod
    def __sub__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __isub__(self: S, value: Union[S, float, int]) -> S: ...

    @abstractmethod
    def __mul__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __imul__(self: S, value: Union[S, float, int]) -> S: ...

    @abstractmethod
    def __matmul__(self, value: 'NumpyShaped') -> 'NumpyShaped': ...

    @abstractmethod
    def __imatmul__(self, value: S) -> 'NumpyShaped': ...

    @abstractmethod
    def __truediv__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __itruediv__(self: S, value: Union[S, float, int]) -> S: ...

    @abstractmethod
    def __floordiv__(self: S,
                     value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __ifloordiv__(self: S, value: Union[S, float, int]) -> S: ...

    @abstractmethod
    def __mod__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __imod__(self: S, value: Union[S, float, int]) -> S: ...

    @abstractmethod
    def __pow__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __ipow__(self: S, value: Union[S, float, int]) -> S: ...

    @abstractmethod
    def __lshift__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __ilshift__(self: S, value: Union[S, float, int]) -> S: ...

    @abstractmethod
    def __rshift__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __irshift__(self: S, value: Union[S, float, int]) -> S: ...

    @abstractmethod
    def __and__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __iand__(self: S, value: Union[S, float, int]) -> S: ...

    @abstractmethod
    def __xor__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __ixor__(self: S, value: Union[S, float, int]) -> S: ...

    @abstractmethod
    def __or__(self: S, value: Union['NumpyShaped', float, int]) -> S: ...

    @abstractmethod
    def __ior__(self: S, value: Union[S, float, int]) -> S: ...

    @abstractmethod
    def __neg__(self: S) -> S: ...

    @abstractmethod
    def __pos__(self: S) -> S: ...

    @abstractmethod
    def __abs__(self: S) -> S: ...

    @abstractmethod
    def __invert__(self: S) -> S: ...

    @abstractmethod
    def astype(self: S, typ: str) -> S: ...

    @staticmethod
    def am(data: Any) -> bool:
        return isinstance(data, (np.ndarray, np.matrix))


class NumpyMatrix(NumpyShaped):
    '''
    A ``mypy`` type for numpy ``matrix`` data. Avoid using if at all possible.
    '''
    shape: Tuple[int, int]
    strides: Tuple[int, int]

    @staticmethod
    def am(data: Any) -> bool:
        return isinstance(data, np.matrix)


class DenseVector(NumpyShaped, Iterable, Sized):
    '''
    A ``mypy`` type for numpy 1-dimensional data.
    '''
    size: int
    shape: Tuple[int]
    strides: Tuple[int]

    @abstractmethod
    def sum(self) -> Union[float, int]: ...

    @abstractmethod
    def argmax(self) -> int: ...

    @abstractmethod
    def nonzero(self) -> Tuple['DenseVector']: ...

    @staticmethod
    def am(data: Any) -> bool:
        return isinstance(data, np.ndarray) and data.ndim == 1


class DenseMatrix(NumpyShaped):
    '''
    A ``mypy`` type for numpy 2-dimensional data.
    '''
    shape: Tuple[int, int]
    strides: Tuple[int, int]
    T: 'DenseMatrix'

    @abstractmethod
    def sum(self, *, axis: int) -> DenseVector: ...

    @abstractmethod
    def argmax(self, *, axis: int) -> DenseVector: ...

    @abstractmethod
    def nonzero(self) -> Tuple[DenseVector, DenseVector]: ...

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
    def nanmax(self, *, axis: int) -> DenseVector: ...

    @abstractmethod
    def min(self, *, axis: int) -> DenseVector: ...

    @abstractmethod
    def nanmin(self, *, axis: int) -> DenseVector: ...

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

    @abstractmethod
    def nonzero(self) -> Tuple[DenseVector, DenseVector]: ...

    @abstractmethod
    def mean(self: SP, *, axis: int) -> 'DenseVector': ...

    @abstractmethod
    def copy(self: SP) -> SP: ...

    @staticmethod
    def am(data: Any) -> bool:
        return sp.issparse(data)


class CompressedMatrix(SparseMatrix):
    '''
    A ``mypy`` type for sparse CSR/CSC 2-dimensional data.
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

    @abstractmethod
    def sort_indices(self) -> None: ...

    @staticmethod
    def am(data: Any) -> bool:
        return SparseMatrix.am(data) and data.getformat() in ('csr', 'csc')


class PandasIndex(Shaped):
    '''
    A ``mypy`` type for a pandas index.
    '''
    values: DenseVector


class PandasFrame(Shaped):
    '''
    A ``mypy`` type for pandas 2-dimensional data.
    '''
    shape: Tuple[int, int]
    values: DenseMatrix
    index: PandasIndex
    columns: PandasIndex

    @staticmethod
    def am(data: Any) -> bool:
        return isinstance(data, pd.DataFrame)


class PandasSeries(Shaped, Sized):
    '''
    A ``mypy`` type for pandas 1-dimensional data.
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


#: A ``mypy`` type for any 1-dimensional data.
#:
#: .. todo::
#:
#:    Is there any need for ``SparseVector``?
Vector = Union[DenseVector, PandasSeries]

#: A "proper" 1- or 2-dimensional data.
ProperShaped = Union[ProperMatrix, DenseVector]

#: An "improper" 1- or 2- dimensional data.
ImproperShaped = Union[ImproperMatrix, PandasSeries]

#: Shaped data of any of the types we can deal with.
AnyShaped = Union[ProperShaped, ImproperShaped]

#: Dense 1- or 2-dimensional data.
DenseShaped = Union[DenseVector, DenseMatrix]


@utd.expand_doc()
def to_proper_matrix(
    matrix: Matrix,
    *,
    default_layout: Optional[str] = 'row_major'
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

    if default_layout is not None:
        assert default_layout in LAYOUT_OF_AXIS

    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values
        if isinstance(matrix, pd.core.arrays.categorical.Categorical):
            matrix = np.array(matrix)

    sparse = SparseMatrix.maybe(matrix)
    if sparse is not None:
        compressed = CompressedMatrix.maybe(sparse)
        if compressed is None:
            if default_layout == 'column_major':
                with utm.timed_step('sparse.tocsc'):
                    compressed = sparse.tocsc()
                    utm.timed_parameters(results=compressed.shape[1],
                                         elements=compressed.nnz / compressed.shape[1])
            else:
                with utm.timed_step('sparse.tocsr'):
                    compressed = sparse.tocsr()
                    utm.timed_parameters(results=compressed.shape[0],
                                         elements=compressed.nnz / compressed.shape[0])
        return compressed

    if isinstance(matrix, np.matrix) or not isinstance(matrix, np.ndarray):
        matrix = np.asarray(matrix)
    assert isinstance(matrix, np.ndarray)

    return matrix


def to_proper_matrices(
    matrix: Matrix,
    *,
    default_layout: Optional[str] = 'row_major'
) -> Tuple[ProperMatrix, Optional[DenseMatrix], Optional[CompressedMatrix]]:
    '''
    Given some matrix, return the properly-formatted data within it using
    :py:func:`to_proper_matrix`, which may be either dense or compressed (sparse).
    '''
    proper = to_proper_matrix(matrix, default_layout=default_layout)
    dense = DenseMatrix.maybe(proper)
    compressed = CompressedMatrix.maybe(proper)
    assert (dense is not None) or (compressed is not None)
    assert (dense is None) != (compressed is None)
    return (proper, dense, compressed)


def frozen(shaped: Union[ProperShaped, PandasFrame, PandasSeries]) -> bool:
    '''
    Test whether the ``shaped`` data is protected against future modification.
    '''
    compressed = CompressedMatrix.maybe(shaped)
    if compressed is not None:
        assert compressed.indices.flags.writeable \
            == compressed.indptr.flags.writeable \
            == compressed.data.flags.writeable
        return not compressed.data.flags.writeable

    if isinstance(shaped, (pd.DataFrame, pd.Series)):
        shaped = shaped.values

    numpy = NumpyShaped.maybe(shaped)
    if numpy is not None:
        return not numpy.flags.writeable

    raise NotImplementedError('frozen of %s' % shaped.__class__)


def freeze(shaped: Union[ProperShaped, PandasFrame, PandasSeries]) -> None:
    '''
    Protect the ``shaped`` data against future modification.
    '''
    compressed = CompressedMatrix.maybe(shaped)
    if compressed is not None:
        compressed.indices.setflags(write=False)
        compressed.indptr.setflags(write=False)
        compressed.data.setflags(write=False)
        return

    if isinstance(shaped, (pd.DataFrame, pd.Series)):
        shaped = shaped.values

    numpy = NumpyShaped.maybe(shaped)
    if numpy is not None:
        numpy.setflags(write=False)
        return

    raise NotImplementedError('freeze of %s' % shaped.__class__)


def unfreeze(shaped: Union[ProperShaped, PandasFrame, PandasSeries]) -> None:
    '''
    Permit future modification of some ``shaped`` data.
    '''
    compressed = CompressedMatrix.maybe(shaped)
    if compressed is not None:
        compressed.indices.setflags(write=True)
        compressed.indptr.setflags(write=True)
        compressed.data.setflags(write=True)
        return

    if isinstance(shaped, (pd.DataFrame, pd.Series)):
        shaped = shaped.values

    numpy = NumpyShaped.maybe(shaped)
    if numpy is not None:
        numpy.setflags(write=True)
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
def to_dense_matrix(
    matrix: Matrix,
    *,
    default_layout: Optional[str] = 'row_major',
    copy: Optional[bool] = False
) -> DenseMatrix:
    '''
    Convert some (possibly improper) ``matrix`` data to a dense 2-dimensional numpy array.

    If ``copy`` (default: {copy}), a copy of the data is returned even if it is already a numpy
    array. If ``copy`` is ``None``, always returns the data without copying (fails on sparse data).
    '''
    sparse = SparseMatrix.maybe(matrix)
    if sparse is not None:
        assert copy is not None
        with utm.timed_step('sparse.toarray'):
            if sparse.ndim == 2:
                utm.timed_parameters(results=sparse.shape[0],
                                     elements=sparse.shape[1])
            else:
                utm.timed_parameters(size=sparse.shape[0])
            return sparse.toarray()

    dense = \
        DenseMatrix.be(to_proper_matrix(matrix, default_layout=default_layout))

    if copy and id(dense) == id(matrix):
        dense = np.copy(dense)

    return DenseMatrix.be(dense)


@utd.expand_doc()
def to_dense_vector(shaped: Shaped, *, copy: Optional[bool] = False) -> DenseVector:
    '''
    Convert some (possibly improper) data to a dense 1-dimensional numpy array.

    This will convert a matrix where one of the dimensions has size one to a flat array.

    If ``copy`` (default: {copy}), a copy of the data is returned even if it is already a 1d numpy
    array. If ``copy`` is ``None``, always returns the data without copying (fails on sparse data).
    '''
    if shaped.ndim == 2:
        assert shaped.shape[0] == 1 or shaped.shape[1] == 1  # type: ignore
        dense: DenseVector = to_dense_matrix(shaped, copy=copy)  # type: ignore
        dense = np.reshape(dense, -1)

    else:
        assert shaped.ndim == 1
        if isinstance(shaped, pd.Series):
            dense = DenseVector.be(shaped.values)
            if isinstance(dense, pd.core.arrays.categorical.Categorical):
                dense = np.array(dense)
        else:
            dense = DenseVector.be(shaped)

    if copy and id(dense) == id(shaped):
        dense = np.copy(dense)

    return DenseVector.be(dense)


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

    sparse = SparseMatrix.maybe(matrix)
    if sparse is not None:
        return sparse.getformat() == SPARSE_FAST_FORMAT[layout]

    dense = DenseMatrix.be(to_proper_matrix(matrix))
    return dense.flags[DENSE_FAST_FLAG[layout]]


def is_row_major(matrix: Matrix) -> bool:
    '''
    Test whether the matrix ``layout`` is optimized for per-row (variable, gene) processing.
    '''
    return is_layout(matrix, 'row_major')


def is_column_major(matrix: Matrix) -> bool:
    '''
    Test whether the ``matrix`` layout is optimized for per-column (observation, cell) processing.
    '''
    return is_layout(matrix, 'column_major')


@overload
def matrix_layout(matrix: ProperMatrix) -> str: ...


@overload
def matrix_layout(matrix: ImproperMatrix) -> Optional[str]: ...


def matrix_layout(matrix):  # type: ignore
    '''
    Return which layout the ``matrix`` is optimized for (``row_major`` or ``column_major``).

    .. note::

        The matrix may be in neither layout, in which case this returns ``None``.
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
    Return whether the ``vector`` is contiguous in memory.

    This is only ``True`` for a dense vector.
    '''
    dense = to_dense_vector(vector)
    return dense.flags.c_contiguous and dense.flags.f_contiguous


def to_contiguous(vector: Vector, *, copy: bool = False) -> DenseVector:
    '''
    Return the ``vector`` in contiguous (dense) format.

    If ``copy`` (default: {copy}), a copy of the vector is returned even if it is already a
    contiguous numpy array.
    '''
    dense = to_dense_vector(vector)
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
    if shaped.ndim == 1:
        return is_contiguous(shaped)  # type: ignore

    sparse = SparseMatrix.maybe(shaped)
    if sparse is None:
        return matrix_layout(shaped) is not None  # type: ignore

    if sparse.getformat() not in ('coo', 'csr', 'csc'):
        return False

    if hasattr(sparse, 'has_canonical_format') \
            and not getattr(sparse, 'has_canonical_format'):
        return False

    if hasattr(sparse, 'has_sorted_indices') \
            and not getattr(sparse, 'has_sorted_indices'):
        return False

    compressed = CompressedMatrix.maybe(sparse)
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
