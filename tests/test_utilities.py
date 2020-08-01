'''
Test the utility functions.
'''

from typing import Any

import numpy as np  # type: ignore
from anndata import AnnData
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


def test_parallel_map() -> None:
    actual = list(ut.parallel_map(lambda index: index, 10000))
    expected = list(range(10000))
    assert actual == expected


def test_unbatched_parallel_map() -> None:
    actual = list(ut.parallel_map(lambda index: index,
                                  10000, batches_per_thread=None))
    expected = list(range(10000))
    assert actual == expected


def test_parallel_for() -> None:
    mask = np.zeros(10000, dtype='bool')

    def invocation(indices: range) -> None:
        assert not np.any(mask[indices])
        mask[indices] = True

    ut.parallel_for(invocation, 10000)

    assert np.all(mask)


def test_unbatched_parallel_for() -> None:
    mask = np.zeros(10000, dtype='bool')

    def invocation(index: int) -> None:
        assert not mask[index]
        mask[index] = True

    ut.parallel_for(invocation, 10000, batches_per_thread=None)

    assert np.all(mask)


def test_parallel_collect() -> None:
    shared_storage = ut.SharedStorage()

    def compute(indices: range) -> None:
        tmp = shared_storage.get_private('tmp', make=list)
        tmp.extend(list(indices))

    def merge(from_thread: str, into_thread: str) -> None:
        from_tmp = shared_storage.get_private('tmp', thread=from_thread)
        into_tmp = shared_storage.get_private('tmp', thread=into_thread)
        into_tmp.extend(from_tmp)

    final_thread = ut.parallel_collect(compute, merge, 10000)
    final_tmp = shared_storage.get_private('tmp', thread=final_thread)
    assert len(final_tmp) == 10000
    assert np.all(np.arange(10000) == sorted(final_tmp))


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


def test_downsample_tmp_size() -> None:
    assert ut.downsample_tmp_size(0) == 0
    assert ut.downsample_tmp_size(1) == 0
    assert ut.downsample_tmp_size(2) == 3
    assert ut.downsample_tmp_size(3) == 7
    assert ut.downsample_tmp_size(4) == 7
    assert ut.downsample_tmp_size(5) == 15
    assert ut.downsample_tmp_size(6) == 15
    assert ut.downsample_tmp_size(7) == 15
    assert ut.downsample_tmp_size(8) == 15
    assert ut.downsample_tmp_size(9) == 31


def test_downsample_with_tmp() -> None:
    size = 10
    total = (size * (size - 1)) // 2
    samples = 20
    random_seed = 123456

    safe = np.arange(size)
    data = np.arange(size)
    tmp = np.empty(ut.downsample_tmp_size(size))
    output = np.empty(size)

    ut.downsample_array(data, samples, random_seed=random_seed,
                        tmp=tmp, output=output)
    assert tmp[-1] == total - samples

    assert np.all(data == safe)
    assert np.all(output <= data)
    assert np.sum(output) == samples


def test_downsample_array_inplace() -> None:
    size = 10
    samples = 20
    random_seed = 123456

    safe = np.arange(size)
    data = np.arange(size)

    ut.downsample_array(data, samples, random_seed=random_seed)

    assert np.all(data <= safe)
    assert np.sum(data) == samples


def test_downsample_array_too_few() -> None:
    size = 10
    total = (size * (size - 1)) // 2
    samples = total * 2
    random_seed = 123456

    safe = np.arange(size)
    data = np.arange(size)

    ut.downsample_array(data, samples, random_seed=random_seed)

    assert np.all(data == safe)


def test_downsample_matrix() -> None:
    rvs = stats.poisson(10, loc=10).rvs
    matrix = sparse.random(1000, 10000, format='csr',
                           dtype='int32', random_state=123456, data_rvs=rvs)
    assert matrix.nnz == matrix.shape[0] * matrix.shape[1] * 0.01
    old_row_sums = ut.sum_matrix(matrix, axis=1)
    min_sum = old_row_sums.min()
    result = ut.downsample_matrix(matrix, axis=1, samples=int(min_sum))
    assert result.shape == matrix.shape
    new_row_sums = ut.sum_matrix(result, axis=1)
    assert np.all(new_row_sums == min_sum)

    matrix = matrix.toarray()
    result = ut.downsample_matrix(matrix, axis=1, samples=int(min_sum))
    assert result.shape == matrix.shape
    new_row_sums = ut.sum_matrix(result, axis=1)
    assert np.all(new_row_sums == min_sum)


def test_bincount() -> None:
    array = np.array(np.random.rand(100000) * 100, dtype='int32')
    numpy_bincount = np.bincount(array)
    metacells_bincount = ut.bincount_array(array)
    assert np.all(numpy_bincount == metacells_bincount)

    numpy_bincount = np.bincount(array, minlength=110)
    metacells_bincount = ut.bincount_array(array, minlength=110)
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


def test_sum() -> None:
    matrix = np.random.rand(100, 200)
    adata = AnnData(matrix)
    ut.prepare(adata, 'test')

    metacells_sum_per_row = ut.get_sum_per_obs(adata).data
    numpy_sum_per_row = matrix.sum(axis=1)
    assert np.allclose(metacells_sum_per_row, numpy_sum_per_row)

    metacells_sum_per_column = ut.get_sum_per_var(adata).data
    numpy_sum_per_column = matrix.sum(axis=0)
    assert np.allclose(metacells_sum_per_column, numpy_sum_per_column)


def test_mean() -> None:
    matrix = np.random.rand(100, 200)
    adata = AnnData(matrix)
    ut.prepare(adata, 'test')

    metacells_mean_per_row = ut.get_mean_per_obs(adata).data
    numpy_mean_per_row = matrix.mean(axis=1)
    assert np.allclose(metacells_mean_per_row, numpy_mean_per_row)

    metacells_mean_per_column = ut.get_mean_per_var(adata).data
    numpy_mean_per_column = matrix.mean(axis=0)
    assert np.allclose(metacells_mean_per_column, numpy_mean_per_column)


def test_sum_squared() -> None:
    matrix = np.random.rand(100, 200)
    squared = np.square(matrix)
    adata = AnnData(matrix)
    ut.prepare(adata, 'test')

    metacells_sum_squared_per_row = ut.get_sum_squared_per_obs(adata).data
    numpy_sum_squared_per_row = squared.sum(axis=1)
    assert np.allclose(metacells_sum_squared_per_row,
                       numpy_sum_squared_per_row)

    metacells_sum_squared_per_column = ut.get_sum_squared_per_var(adata).data
    numpy_sum_squared_per_column = squared.sum(axis=0)
    assert np.allclose(metacells_sum_squared_per_column,
                       numpy_sum_squared_per_column)


def test_variance() -> None:
    matrix = np.random.rand(100, 200)
    adata = AnnData(matrix)
    ut.prepare(adata, 'test')

    metacells_variance_per_row = ut.get_variance_per_obs(adata).data
    numpy_variance_per_row = matrix.var(axis=1)
    assert np.allclose(metacells_variance_per_row, numpy_variance_per_row)

    metacells_variance_per_column = ut.get_variance_per_var(adata).data
    numpy_variance_per_column = matrix.var(axis=0)
    assert np.allclose(metacells_variance_per_column,
                       numpy_variance_per_column)


def test_max() -> None:
    matrix = np.random.rand(100, 200)
    adata = AnnData(matrix)
    ut.prepare(adata, 'test')

    metacells_max_per_row = ut.get_max_per_obs(adata).data
    numpy_max_per_row = np.amax(matrix, axis=1)
    assert np.allclose(metacells_max_per_row, numpy_max_per_row)

    metacells_max_per_column = ut.get_max_per_var(adata).data
    numpy_max_per_column = np.amax(matrix, axis=0)
    assert np.allclose(metacells_max_per_column, numpy_max_per_column)


def test_min() -> None:
    matrix = np.random.rand(100, 200)
    adata = AnnData(matrix)
    ut.prepare(adata, 'test')

    metacells_min_per_row = ut.get_min_per_obs(adata).data
    numpy_min_per_row = np.amin(matrix, axis=1)
    assert np.allclose(metacells_min_per_row, numpy_min_per_row)

    metacells_min_per_column = ut.get_min_per_var(adata).data
    numpy_min_per_column = np.amin(matrix, axis=0)
    assert np.allclose(metacells_min_per_column, numpy_min_per_column)


def test_nnz() -> None:
    matrix = sparse.rand(100, 200, density=0.1, format='csr')
    adata = AnnData(matrix)
    ut.prepare(adata, 'test')

    metacells_nnz_per_row = ut.get_nnz_per_obs(adata).data
    scipy_nnz_per_row = matrix.getnnz(axis=1)
    assert np.allclose(metacells_nnz_per_row, scipy_nnz_per_row)

    metacells_nnz_per_column = ut.get_nnz_per_var(adata).data
    scipy_nnz_per_column = matrix.getnnz(axis=0)
    assert np.allclose(metacells_nnz_per_column, scipy_nnz_per_column)


def test_focus_on() -> None:
    rvs = stats.poisson(10, loc=10).rvs
    matrix = sparse.random(100, 1000, format='csr',
                           dtype='int32', random_state=123456, data_rvs=rvs)
    adata = AnnData(matrix)

    ut.prepare(adata, 'test')
    assert ut.get_focus_name(adata) == 'vo:test'

    with ut.focus_on(ut.get_log_data, adata) as log_data:
        outer_focus = 'vo:log_e_of_1_plus_test'
        assert ut.get_focus_name(adata) == outer_focus
        assert id(log_data.data) == id(adata.layers[outer_focus[3:]])

        with ut.focus_on(ut.get_downsample_of_var_per_obs, adata, of='test', samples=10) \
                as downsamled_data:
            inner_focus = 'vo:downsample_10_v_per_o_of_vo:test'
            assert ut.get_focus_name(adata) == inner_focus
            assert id(downsamled_data.data) \
                == id(adata.layers[inner_focus[3:]])

        assert ut.get_focus_name(adata) == outer_focus
        assert id(log_data.data) == id(adata.layers[outer_focus[3:]])

        with ut.focus_on(ut.get_downsample_of_var_per_obs, adata, of='test', samples=10) \
                as downsamled_data:
            inner_focus = 'vo:downsample_10_v_per_o_of_vo:test'
            assert ut.get_focus_name(adata) == inner_focus
            assert id(downsamled_data.data) == \
                id(adata.layers[inner_focus[3:]])
            ut.del_vo_data(adata, outer_focus)

        assert ut.get_focus_name(adata) == 'vo:test'

    assert ut.get_focus_name(adata) == 'vo:test'


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


def test_dense_annotations() -> None:
    matrix = np.array([[0, 1, 2], [3, 4, 5]], dtype='float')
    _test_annotations(matrix)


def test_sparse_annotations() -> None:
    matrix = np.array([[0, 1, 2], [3, 4, 5]], dtype='float')
    matrix = sparse.csr_matrix(matrix)
    _test_annotations(matrix)


def _test_annotations(full_matrix: ut.Matrix) -> None:
    adata = AnnData(full_matrix)
    ut.prepare(adata, 'test')

    assert np.allclose(ut.get_nnz_per_obs(adata).data, np.array([2, 3]))
    assert np.allclose(ut.get_nnz_per_var(adata).data, np.array([1, 2, 2]))

    assert np.allclose(ut.get_sum_per_obs(adata).data, np.array([3, 12]))
    assert np.allclose(ut.get_sum_per_var(adata).data, np.array([3, 5, 7]))

    assert np.allclose(ut.get_nnz_per_obs(adata).data, np.array([2, 3]))
    assert np.allclose(ut.get_nnz_per_var(adata).data, np.array([1, 2, 2]))

    assert np.allclose(ut.get_sum_per_obs(adata).data, np.array([3, 12]))
    assert np.allclose(ut.get_sum_per_var(adata).data, np.array([3, 5, 7]))

    assert np.allclose(ut.get_max_per_obs(adata).data, np.array([2, 5]))
    assert np.allclose(ut.get_max_per_var(adata).data, np.array([3, 4, 5]))

    assert np.allclose(ut.get_min_per_obs(adata).data, np.array([0, 3]))
    assert np.allclose(ut.get_min_per_var(adata).data, np.array([0, 1, 2]))

    assert np.allclose(ut.get_sum_squared_per_obs(adata).data,
                       np.array([5, 50]))
    assert np.allclose(ut.get_sum_squared_per_var(adata)
                       .data, np.array([9, 17, 29]))

    assert np.allclose(ut.get_fraction_per_obs(adata)
                       .data, np.array([3/15, 12/15]))
    assert np.allclose(ut.get_fraction_per_var(adata).data,
                       np.array([3/15, 5/15, 7/15]))

    assert np.allclose(ut.get_mean_per_obs(adata).data, np.array([3/3, 12/3]))
    assert np.allclose(ut.get_mean_per_var(adata).data,
                       np.array([3/2, 5/2, 7/2]))

    assert np.allclose(ut.get_variance_per_obs(adata).data,
                       np.array([5/3 - (3/3)**2, 50/3 - (12/3)**2]))

    assert np.allclose(ut.get_variance_per_var(adata).data,
                       np.array([9/2 - (3/2)**2, 17/2 - (5/2)**2, 29/2 - (7/2)**2]))

    assert np.allclose(ut.get_relative_variance_per_obs(adata).data,
                       np.log2(np.array([(5/3 - (3/3)**2) / (3/3),
                                         (50/3 - (12/3)**2) / (12/3)])))
    assert np.allclose(ut.get_relative_variance_per_var(adata).data,
                       np.log2(np.array([(9/2 - (3/2)**2) / (3/2),
                                         (17/2 - (5/2)**2) / (5/2),
                                         (29/2 - (7/2)**2) / (7/2)])))

    data = ut.get_fraction_of_var_per_obs(adata).data
    if sparse.issparse(data):
        data = data.toarray()
    assert np.allclose(data, np.array([[0/3, 1/3, 2/3], [3/12, 4/12, 5/12]]))

    data = ut.get_fraction_of_obs_per_var(adata).data
    if sparse.issparse(data):
        data = data.toarray()
    assert np.allclose(data, np.array([[0/3, 1/5, 2/7], [3/3, 4/5, 5/7]]))
