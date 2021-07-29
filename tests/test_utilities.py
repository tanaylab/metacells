'''
Test the utility functions.
'''

import os
from glob import glob
from typing import Any, List

import numpy as np
from scipy import sparse  # type: ignore
from scipy import stats
from sklearn.metrics import roc_auc_score  # type: ignore

import metacells.utilities as ut

ut.allow_inefficient_layout(False)

# pylint: disable=missing-function-docstring


def test_expand_doc() -> None:
    @ut.expand_doc(foo=7)
    def bar(baz: Any, vaz: int = 5) -> None:  # pylint: disable=blacklisted-name,unused-argument
        '''
        Bar with {foo} foos and parameter vaz (default: {vaz}).
        '''

    assert bar.__doc__ == '''
        Bar with 7 foos and parameter vaz (default: 5).
        '''


def test_corrcoef() -> None:
    np.random.seed(123456)
    matrix = sparse.rand(100, 10000, density=0.1, format='csr')
    # matrix = sparse.rand(5, 20, density=0.3, format='csr')
    dense = ut.to_layout(matrix.toarray(), layout='row_major')
    numpy_correlation = np.corrcoef(dense)
    assert numpy_correlation.shape == (100, 100)

    dense = dense.T

    for reproducible in (False, True):
        dense_correlation = ut.corrcoef(dense, per='column',
                                        reproducible=reproducible)
        assert dense_correlation.shape == (100, 100)
        assert np.min(np.diag(dense_correlation)) == 1
        assert np.max(np.diag(dense_correlation)) == 1
        assert np.allclose(dense_correlation, numpy_correlation, atol=1e-6)

        sparse_correlation = ut.corrcoef(matrix, per='row',
                                         reproducible=reproducible)
        assert sparse_correlation.shape == (100, 100)
        assert np.min(np.diag(sparse_correlation)) == 1
        assert np.max(np.diag(sparse_correlation)) == 1
        assert np.allclose(sparse_correlation, numpy_correlation, atol=1e-6)

    dense[:, :] = 0
    for reproducible in (False, True):
        zeros_correlation = ut.corrcoef(dense, per='column',
                                        reproducible=reproducible)
        assert zeros_correlation.shape == (100, 100)
        assert np.min(np.diag(zeros_correlation)) == 1
        assert np.max(np.diag(zeros_correlation)) == 1
        np.fill_diagonal(zeros_correlation, 0)
        assert np.min(zeros_correlation) == 0
        assert np.max(zeros_correlation) == 0


def test_logistics() -> None:
    matrix = np.array([[0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 1, 1, 1]], dtype='float64')
    same_value = 1 / (1 + np.exp(-5 * (0 - 0.8)))
    diff_value = 1 / (1 + np.exp(-5 * (1 - 0.8)))
    results = ut.logistics(matrix, per='row')
    assert np.allclose(results[0, 0], 0)
    assert np.allclose(results[1, 1], 0)
    assert np.allclose(results[2, 2], 0)
    assert np.allclose(results[0, 1], diff_value)
    assert np.allclose(results[1, 0], diff_value)
    assert np.allclose(results[0, 2], (diff_value + same_value) / 2)
    assert np.allclose(results[2, 0], (diff_value + same_value) / 2)
    assert np.allclose(results[1, 2], (diff_value + same_value) / 2)
    assert np.allclose(results[2, 1], (diff_value + same_value) / 2)


def test_relayout_matrix() -> None:
    rvs = stats.poisson(10, loc=10).rvs
    csr_matrix = sparse.random(20, 20, format='csr',
                               dtype='int32', random_state=123456, data_rvs=rvs)
    assert csr_matrix.getformat() == 'csr'

    scipy_csc_matrix = csr_matrix.tocsc()
    metacells_csc_matrix = ut.to_layout(csr_matrix, layout='column_major')

    assert scipy_csc_matrix.getformat() == 'csc'
    assert metacells_csc_matrix.getformat() == 'csc'

    assert np.all(metacells_csc_matrix.indptr == scipy_csc_matrix.indptr)
    assert np.all(metacells_csc_matrix.toarray() == scipy_csc_matrix.toarray())

    scipy_csr_matrix = scipy_csc_matrix.tocsr()

    metacells_csr_matrix = \
        ut.to_layout(metacells_csc_matrix, layout='row_major')

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


def test_matrix_rows_folds_and_aurocs() -> None:
    np.random.seed(123456)

    signal = np.random.rand(100)
    zero_mask = signal < 1 / 3
    one_mask = signal > 2 / 3
    signal[zero_mask] = 0

    dense = np.random.rand(20, 100)
    dense[:, zero_mask] = 0
    for index in range(20):
        dense[index, :] *= index / 10
        dense[index, :] += signal

    in_data = ut.to_layout(dense[:, one_mask], layout='row_major')
    out_data = ut.to_layout(dense[:, ~one_mask], layout='row_major')
    in_means = ut.mean_per(in_data, per='row')
    out_means = ut.mean_per(out_data, per='row')

    normalization = 1e-4
    in_means += normalization
    out_means += normalization
    builtin_rows_folds = in_means / out_means

    builtin_rows_aurocs = \
        np.array([roc_auc_score(one_mask, dense[index, :])
                  for index in range(20)])

    dense_rows_folds, dense_rows_aurocs = \
        ut.matrix_rows_folds_and_aurocs(dense, columns_subset=one_mask,
                                        normalization=normalization)
    assert np.allclose(dense_rows_aurocs, builtin_rows_aurocs)
    assert np.allclose(dense_rows_folds, builtin_rows_folds)

    compressed = sparse.csr_matrix(dense)
    sparse_rows_folds, sparse_rows_aurocs = \
        ut.matrix_rows_folds_and_aurocs(compressed, columns_subset=one_mask,
                                        normalization=normalization)
    assert np.allclose(sparse_rows_aurocs, builtin_rows_aurocs)
    assert np.allclose(sparse_rows_folds, builtin_rows_folds)


def test_bincount_vector() -> None:
    array = np.array(np.random.rand(100000) * 100, dtype='int32')
    numpy_bincount = np.bincount(array)
    metacells_bincount = ut.bincount_vector(array)
    assert np.all(numpy_bincount == metacells_bincount)

    numpy_bincount = np.bincount(array, minlength=110)
    metacells_bincount = ut.bincount_vector(array, minlength=110)
    assert np.all(numpy_bincount == metacells_bincount)


def test_most_frequent_vector() -> None:
    array = np.array(['a', 'b', 'a', 'c'])
    assert ut.most_frequent(array) == 'a'


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
    array = np.arange(5)
    actual_result = \
        ut.sliding_window_function(array, function='mean', window_size=3)

    expected_result = np.array([1, 1, 2, 3, 3])
    assert np.allclose(actual_result, expected_result)

    order_by = np.array([0, 2, 1, 3, 4])
    actual_result = ut.sliding_window_function(array, function='mean',
                                               window_size=3, order_by=order_by)

    expected_result = np.array([1, 2, 1, 8/3, 8/3])
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


def test_bin_fill() -> None:
    size_of_elements = np.arange(10) * 8 + 1
    bin_of_elements = ut.bin_fill(size_of_elements, 106)
    bins_count = np.max(bin_of_elements) + 1
    size_of_bins: List[int] = []
    elements_of_bins: List[List[int]] = []
    for bin_index in range(bins_count):
        elements_of_bin = np.where(bin_of_elements == bin_index)[0]
        size_of_bin = np.sum(size_of_elements[elements_of_bin])
        assert size_of_bin >= 106
        size_of_bins.append(size_of_bin)
        elements_of_bins.append(list(elements_of_bin))

    assert size_of_bins == [131, 123, 116]
    assert elements_of_bins == [[3, 4, 9], [2, 5, 8], [0, 1, 6, 7]]


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

    # TODO: Why does pytest coverage error trying to read these files?
    for path in glob('.coverage.*'):
        os.remove(path)


def test_sum_groups() -> None:
    expected_sums = np.array([[5, 7, 2], [10, 8, 13]])
    expected_sizes = np.array([2, 2])
    groups = np.array([0, 1, 0, 1])

    dense_rows = \
        np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0], [7, 8, 9]], dtype='float')

    results = ut.sum_groups(dense_rows, groups, per='row')
    assert results is not None
    assert np.allclose(ut.to_numpy_matrix(results[0]), expected_sums)
    assert np.allclose(results[1], expected_sizes)

    results = ut.sum_groups(dense_rows.transpose(), groups, per='column')
    assert results is not None
    assert np.allclose(ut.to_numpy_matrix(results[0]),
                       expected_sums.transpose())
    assert np.allclose(results[1], expected_sizes)

    sparse_rows = sparse.csr_matrix(dense_rows)

    results = ut.sum_groups(sparse_rows, groups, per='row')
    assert results is not None
    assert np.allclose(ut.to_numpy_matrix(results[0]), expected_sums)
    assert np.allclose(results[1], expected_sizes)

    results = ut.sum_groups(sparse_rows.transpose(), groups, per='column')
    assert results is not None
    assert np.allclose(ut.to_numpy_matrix(results[0]),
                       expected_sums.transpose())
    assert np.allclose(results[1], expected_sizes)


def test_shuffle_dense_rows_matrix() -> None:
    dense = \
        np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0], [7, 8, 9]], dtype='float')
    ut.shuffle_matrix(dense, per='row', random_seed=123456)
    expected = \
        np.array([[0, 2, 1], [3, 4, 0], [5, 0, 6], [7, 8, 9]], dtype='float')
    assert np.allclose(dense, expected)


def test_shuffle_sparse_rows_matrix() -> None:
    sparse_csr = \
        sparse.csr_matrix(np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0], [7, 8, 9]],
                                   dtype='float'))
    ut.shuffle_matrix(sparse_csr, per='row', random_seed=123456)
    expected = \
        np.array([[1, 0, 2], [3, 0, 4], [5, 0, 6], [7, 8, 9]], dtype='float')
    assert np.allclose(sparse_csr.todense(), expected)


def test_shuffle_sparse_columns_matrix() -> None:
    sparse_csc = \
        sparse.csc_matrix(np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0], [7, 8, 9]],
                                   dtype='float'))
    ut.shuffle_matrix(sparse_csc, per='column', random_seed=123456)
    expected = \
        np.array([[3, 1, 9], [0, 0, 4], [7, 6, 2], [5, 8, 0]], dtype='float')
    assert np.allclose(sparse_csc.todense(), expected)


def test_rank_matrix() -> None:
    dense = \
        np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0], [7, 8, 9]], dtype='float')

    result = ut.rank_per(dense, per='row', rank=1)
    expected = np.array([1, 3, 5, 8])
    assert np.allclose(result, expected)

    dense = ut.to_layout(dense, layout='column_major')
    result = ut.rank_per(dense, per='column', rank=1)
    expected = np.array([3, 1, 2])
    assert np.allclose(result, expected)


def test_random_piles() -> None:
    result = ut.random_piles(10, target_pile_size=3, random_seed=123456)
    expected = np.array([2, 2, 1, 1, 1, 0, 0, 2, 0, 0])
    assert np.allclose(result, expected)


def test_group_piles() -> None:
    group_of_elements = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    group_of_groups = np.array([0, 1, 1, 0])
    expected = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1])
    result = ut.group_piles(group_of_elements, group_of_groups)
    assert np.allclose(result, expected)


def test_dense_per() -> None:
    matrix = np.array([[0, 1, 2], [3, 4, 5]], dtype='float')
    _test_per(matrix)


def test_sparse_per() -> None:
    matrix = np.array([[0, 1, 2], [3, 4, 5]], dtype='float')
    matrix = sparse.csr_matrix(matrix)
    _test_per(matrix)


def _test_per(rows_matrix: ut.Matrix) -> None:
    columns_matrix = ut.to_layout(rows_matrix, layout='column_major')

    assert np.allclose(ut.nnz_per(rows_matrix, per='row'),
                       np.array([2, 3]))
    assert np.allclose(ut.nnz_per(columns_matrix, per='column'),
                       np.array([1, 2, 2]))

    assert np.allclose(ut.sum_per(rows_matrix, per='row'),
                       np.array([3, 12]))
    assert np.allclose(ut.sum_per(columns_matrix, per='column'),
                       np.array([3, 5, 7]))

    assert np.allclose(ut.max_per(rows_matrix, per='row'),
                       np.array([2, 5]))
    assert np.allclose(ut.max_per(columns_matrix, per='column'),
                       np.array([3, 4, 5]))

    assert np.allclose(ut.min_per(rows_matrix, per='row'),
                       np.array([0, 3]))
    assert np.allclose(ut.min_per(columns_matrix, per='column'),
                       np.array([0, 1, 2]))

    assert np.allclose(ut.sum_squared_per(rows_matrix, per='row'),
                       np.array([5, 50]))
    assert np.allclose(ut.sum_squared_per(columns_matrix, per='column'),
                       np.array([9, 17, 29]))

    assert np.allclose(ut.fraction_per(rows_matrix, per='row'),
                       np.array([3/15, 12/15]))
    assert np.allclose(ut.fraction_per(columns_matrix, per='column'),
                       np.array([3/15, 5/15, 7/15]))

    assert np.allclose(ut.mean_per(rows_matrix, per='row'),
                       np.array([3/3, 12/3]))
    assert np.allclose(ut.mean_per(columns_matrix, per='column'),
                       np.array([3/2, 5/2, 7/2]))

    assert np.allclose(ut.variance_per(rows_matrix, per='row'),
                       np.array([5/3 - (3/3)**2, 50/3 - (12/3)**2]))

    assert np.allclose(ut.variance_per(columns_matrix, per='column'),
                       np.array([9/2 - (3/2)**2, 17/2 - (5/2)**2, 29/2 - (7/2)**2]))

    assert np.allclose(ut.normalized_variance_per(columns_matrix, per='column'),
                       np.array([(9/2 - (3/2)**2) / (3/2),
                                 (17/2 - (5/2)**2) / (5/2),
                                 (29/2 - (7/2)**2) / (7/2)]))

    dense = ut.to_numpy_matrix(ut.fraction_by(rows_matrix, by='row'))
    assert np.allclose(dense, np.array([[0/3, 1/3, 2/3], [3/12, 4/12, 5/12]]))

    dense = ut.to_numpy_matrix(ut.fraction_by(columns_matrix, by='column'))
    assert np.allclose(dense, np.array([[0/3, 1/5, 2/7], [3/3, 4/5, 5/7]]))
