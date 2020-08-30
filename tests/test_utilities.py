'''
Test the utility functions.
'''

import os
from glob import glob
from typing import Any, List

import numpy as np  # type: ignore
from scipy import sparse  # type: ignore
from scipy import stats

import metacells.utilities as ut

# pylint: disable=missing-docstring


def test_expand_doc() -> None:
    @ut.expand_doc(foo=7)
    def bar(baz: Any, vaz: int = 5) -> None:  # pylint: disable=blacklisted-name,unused-argument
        '''
        Bar with {foo} foos and parameter vaz (default: {vaz}).
        '''

    assert bar.__doc__ == '''
        Bar with 7 foos and parameter vaz (default: 5).
        '''


def test_sparse_corrcoef() -> None:
    matrix = sparse.rand(100, 10000, density=0.1, format='csr')
    sparse_correlation = ut.corrcoef(matrix)
    numpy_correlation = np.corrcoef(matrix.toarray())
    assert numpy_correlation.shape == (100, 100)
    assert np.allclose(sparse_correlation, numpy_correlation)


def test_relayout_matrix() -> None:
    rvs = stats.poisson(10, loc=10).rvs
    csr_matrix = sparse.random(1000, 10000, format='csr',
                               dtype='int32', random_state=123456, data_rvs=rvs)
    assert csr_matrix.getformat() == 'csr'

    scipy_csc_matrix = csr_matrix.tocsc()
    metacells_csc_matrix = ut.relayout_compressed(csr_matrix)

    assert scipy_csc_matrix.getformat() == 'csc'
    assert metacells_csc_matrix.getformat() == 'csc'

    assert np.all(metacells_csc_matrix.indptr == scipy_csc_matrix.indptr)
    assert np.all(metacells_csc_matrix.toarray() == scipy_csc_matrix.toarray())

    scipy_csr_matrix = scipy_csc_matrix.tocsr()

    metacells_csr_matrix = ut.relayout_compressed(metacells_csc_matrix)

    assert scipy_csr_matrix.getformat() == 'csr'
    assert metacells_csr_matrix.getformat() == 'csr'

    assert np.all(metacells_csr_matrix.indptr == scipy_csr_matrix.indptr)
    assert np.all(metacells_csr_matrix.toarray() == scipy_csr_matrix.toarray())


def test_downsample_vector_inplace() -> None:
    size = 10
    samples = 20
    random_seed = 123456

    safe = np.arange(size)
    data = np.arange(size)

    ut.downsample_vector(data, samples, random_seed=random_seed)

    assert np.all(data <= safe)
    assert np.sum(data) == samples


def test_downsample_vector_too_few() -> None:
    size = 10
    total = (size * (size - 1)) // 2
    samples = total * 2
    random_seed = 123456

    safe = np.arange(size)
    data = np.arange(size)

    ut.downsample_vector(data, samples, random_seed=random_seed)

    assert np.all(data == safe)


def test_downsample_matrix() -> None:
    rvs = stats.poisson(10, loc=10).rvs
    matrix = sparse.random(1000, 10000, format='csr',
                           dtype='int32', random_state=123456, data_rvs=rvs)
    assert matrix.nnz == matrix.shape[0] * matrix.shape[1] * 0.01
    old_row_sums = ut.sum_per(matrix, per='row')
    min_sum = np.min(old_row_sums)
    result = ut.downsample_matrix(matrix, per='row', samples=int(min_sum))
    assert result.shape == matrix.shape
    new_row_sums = ut.sum_per(result, per='row')
    assert np.all(new_row_sums == min_sum)

    matrix = matrix.toarray()
    result = ut.downsample_matrix(matrix, per='row', samples=int(min_sum))
    assert result.shape == matrix.shape
    new_row_sums = ut.sum_per(result, per='row')
    assert np.all(new_row_sums == min_sum)


def test_bincount_vector() -> None:
    array = np.array(np.random.rand(100000) * 100, dtype='int32')
    numpy_bincount = np.bincount(array)
    metacells_bincount = ut.bincount_vector(array)
    assert np.all(numpy_bincount == metacells_bincount)

    numpy_bincount = np.bincount(array, minlength=110)
    metacells_bincount = ut.bincount_vector(array, minlength=110)
    assert np.all(numpy_bincount == metacells_bincount)


def test_freeze_dense() -> None:
    array = np.arange(10)

    assert not ut.frozen(array)
    array[0] = -1
    assert array[0] == -1

    ut.freeze(array)
    assert ut.frozen(array)
    try:
        array[0] = -2
        assert False, 'writing to frozen array'
    except ValueError as exception:
        assert 'is read-only' in str(exception)

    assert array[0] == -1

    ut.unfreeze(array)
    assert not ut.frozen(array)
    array[0] = -3
    assert array[0] == -3


def test_freeze_sparse() -> None:
    matrix = sparse.rand(100, 100, density=0.1, format='csr')

    row = 0
    while matrix.indptr[row + 1] == 0:
        row += 1
    column = matrix.indices[0]
    assert matrix[row, column] != 0

    assert not ut.frozen(matrix)
    matrix.data[0] = -1
    assert matrix[row, column] == -1

    ut.freeze(matrix)
    assert ut.frozen(matrix)
    try:
        matrix.data[0] = -2
        assert False, 'writing to frozen matrix'
    except ValueError as exception:
        assert 'is read-only' in str(exception)

    assert matrix.data[0] == -1

    ut.unfreeze(matrix)
    assert not ut.frozen(matrix)
    matrix.data[0] = -3
    assert matrix[row, column] == -3


def test_sliding_window_function() -> None:
    array = np.arange(3)
    actual_result = \
        ut.sliding_window_function(array, function='mean', window_size=3)
    expected_result = np.array([1/3, 1, 5/3])
    assert np.allclose(actual_result, expected_result)

    order_by = np.array([0, 2, 1])
    actual_result = ut.sliding_window_function(array, function='mean',
                                               window_size=3, order_by=order_by)
    expected_result = np.array([2/3, 4/3, 1])
    assert np.allclose(actual_result, expected_result)


def test_patterns_matches() -> None:
    strings = ['foo', 'bar', 'baz']
    actual = ut.patterns_matches('ba?', strings)
    expected = np.array([False, True, True])
    assert np.allclose(actual, expected)


def test_bin_pack() -> None:
    size_of_elements = np.arange(10) * 8 + 1
    bin_of_elements = ut.bin_pack(size_of_elements, 106)
    bins_count = np.max(bin_of_elements) + 1
    size_of_bins: List[int] = []
    elements_of_bins: List[List[int]] = []
    for bin_index in range(bins_count):
        elements_of_bin = np.where(bin_of_elements == bin_index)[0]
        size_of_bin = np.sum(size_of_elements[elements_of_bin])
        assert size_of_bin <= 106
        size_of_bins.append(size_of_bin)
        elements_of_bins.append(list(elements_of_bin))

    assert size_of_bins == [98, 98, 98, 76]
    assert elements_of_bins == [[3, 9], [4, 8], [5, 7], [0, 1, 2, 6]]


def test_compress_indices() -> None:
    assert list(ut.compress_indices(np.array([0, 3, 2]))) == [0, 2, 1]
    assert list(ut.compress_indices(np.array([0, -1, 2]))) == [0, -1, 1]


def test_parallel_map() -> None:
    @ut.timed_call('invocation')
    def invocation(index: int) -> int:
        return index

    actual = list(ut.parallel_map(invocation, 100))
    expected = list(range(100))
    assert actual == expected

    for path in glob('.coverage.*'):
        os.remove(path)
