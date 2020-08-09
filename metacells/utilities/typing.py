'''
Utilities for dealing with shaped data types.
'''

from typing import Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy import sparse  # type: ignore

import metacells.utilities.timing as timed

__all__ = [
    'DATA_TYPES',

    'Matrix',
    'Vector',
    'Shaped',

    'canonize',

    'frozen',
    'freeze',
    'unfreeze',

    'unpandas',
    'to_np_array',
    'to_1d_array',

    'DENSE_FAST_FLAG',
    'SPARSE_FAST_FORMAT',
    'SPARSE_SLOW_FORMAT',

    'is_layout',
    'is_row_major',
    'is_column_major',

    'is_contiguous',
    'to_contiguous',
]


#: The data types supported by the C++ extensions code.
DATA_TYPES = ['float32', 'float64', 'int32', 'int64', 'uint32', 'uint64']

#: A ``mypy`` type for matrices.
Matrix = Union[sparse.spmatrix, np.ndarray, pd.DataFrame]

#: A ``mypy`` type for vectors.
Vector = Union[np.ndarray, pd.Series]

#: A ``mypy`` type for shaped data (matrix or vector).
Shaped = Union[Matrix, Vector]


@timed.call()
def canonize(matrix: Matrix) -> None:
    '''
    If the data is sparse, ensure it is in "canonical format", which makes most operations on it
    much more efficient.

    Otherwise, keep the matrix as-is.
    '''
    if sparse.issparse(matrix):
        is_frozen = frozen(matrix)
        try:
            unfreeze(matrix)
            matrix.sum_duplicates()
        finally:
            if is_frozen:
                freeze(matrix)


def frozen(data: Shaped) -> bool:
    '''
    Test whether the data is protected against future modification.
    '''
    if not sparse.issparse(data):
        data = unpandas(data)
        return not data.flags.writeable

    assert data.indices.flags.writeable \
        == data.indptr.flags.writeable \
        == data.data.flags.writeable

    return not data.data.flags.writeable


@timed.call()
def freeze(data: Shaped) -> None:
    '''
    Protect data against future modification.
    '''
    if not sparse.issparse(data):
        data = unpandas(data)
        data.setflags(write=False)

    elif data.getformat() in ['csc', 'csr']:
        data.indices.setflags(write=False)
        data.indptr.setflags(write=False)
        data.data.setflags(write=False)

    else:
        raise NotImplementedError('freeze of sparse data in %s format'
                                  % data.getformat())


@timed.call()
def unfreeze(data: Shaped) -> None:
    '''
    Permit data future modification of some data.
    '''
    if not sparse.issparse(data):
        data = unpandas(data)
        data.setflags(write=True)

    elif data.getformat() in ['csc', 'csr']:
        data.indices.setflags(write=True)
        data.indptr.setflags(write=True)
        data.data.setflags(write=True)

    else:
        raise NotImplementedError('unfreeze of sparse data in %s format'
                                  % data.getformat())


def unpandas(data: Shaped) -> Shaped:
    '''
    If the data is a Pandas data frame or series, access the underlying Numpy array,
    otherwise return it as is (sparse data).
    '''
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.values

    return data


def to_np_array(data: Shaped, *, copy: Optional[bool] = False) -> np.ndarray:
    '''
    Convert some (possibly sparse) data to an (full dense size) 1/2-dimensional Numpy array.

    If ``copy``, a copy of the data is returned even if it is already a Numpy array. If ``copy`` is
    ``None``, always returns the data without copying (fails on sparse data).
    '''
    if sparse.issparse(data):
        assert copy is not None
        with timed.step('toarray'):
            if data.ndim == 2:
                timed.parameters(results=data.shape[0], elements=data.shape[1])
            else:
                timed.parameters(size=data.shape[0])
            return data.toarray()

    data = unpandas(data)

    if isinstance(data, np.matrix):
        data = np.array(data)

    if copy:
        data = np.copy(data)

    return data


def to_1d_array(data: Shaped, *, copy: Optional[bool] = False) -> np.ndarray:
    '''
    Convert some (possibly sparse) data to an (full dense size) 1-dimensional array.

    This should only be applied if only one dimension has size greater than one.

    If ``copy``, a copy of the data is returned even if it is already a Numpy array. If ``copy`` is
    ``None``, always returns the data without copying (fails on sparse data).
    '''
    data = to_np_array(data, copy=copy)

    if data.ndim == 1:
        return data

    seen_large_dimension = False
    for size in data.shape:
        if size == 1:
            continue
        assert not seen_large_dimension
        seen_large_dimension = True

    array = np.reshape(data, -1)
    assert array.ndim == 1

    return array


#: Which flag indicates efficient 2D dense matrix layout.
DENSE_FAST_FLAG = dict(column_major='F_CONTIGUOUS', row_major='C_CONTIGUOUS')

#: Which format indicates efficient 2D sparse matrix layout.
SPARSE_FAST_FORMAT = dict(column_major='csc', row_major='csr')

#: Which format indicates inefficient 2D sparse matrix layout.
SPARSE_SLOW_FORMAT = dict(column_major='csr', row_major='csc')


def is_layout(matrix: Matrix, layout: str) -> bool:
    '''
    Test whether the matrix layout is optimized for ``column_major`` or ``row_major`` layout.
    '''
    if sparse.issparse(matrix):
        return matrix.getformat() == SPARSE_FAST_FORMAT[layout]

    matrix = unpandas(matrix)
    return matrix.flags[DENSE_FAST_FLAG[layout]]


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


def matrix_layout(matrix: Matrix) -> Optional[str]:
    '''
    Return which layout the matrix is optimized for (``row_major`` or ``column_major``).

    .. note::

        The matrix may be neither one, in which case this returns ``None``.
    '''
    assert matrix.ndim == 2

    if sparse.issparse(matrix):
        for layout, sparse_format in SPARSE_FAST_FORMAT.items():
            if matrix.getformat() == sparse_format:
                return layout

    else:
        matrix = unpandas(matrix)
        for layout, flag in DENSE_FAST_FLAG.items():
            if matrix.flags[flag]:
                return layout

    return None


def is_contiguous(vector: Vector) -> bool:
    '''
    Return whether the vector is contiguous in memory.

    This is only ``True`` for a dense vector.
    '''
    assert vector.ndim == 1

    if sparse.issparse(vector):
        return False

    vector = unpandas(vector)
    return vector.flags.c_contiguous and vector.flags.f_contiguous


def to_contiguous(vector: Vector, *, copy: bool = False) -> np.ndarray:
    '''
    Return the vector in contiguous (dense) format.

    If ``copy``, a copy of the vector is returned even if it is already a contiguous Numpy array.
    '''
    vector = unpandas(vector)

    if copy or not isinstance(vector, np.ndarray) or vector.ndim != 1 or not is_contiguous(vector):
        return to_1d_array(vector, copy=copy)

    return vector
