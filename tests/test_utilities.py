'''
Test the utility functions.
'''

import numpy as np  # type: ignore
from scipy import sparse  # type: ignore
from scipy import stats  # type: ignore

import metacells.utilities as ut

# pylint: disable=missing-docstring


def test_expand_doc():
    @ut.expand_doc(foo=7)
    def bar(baz, vaz=5):  # pylint: disable=blacklisted-name,unused-argument
        '''
        Bar with {foo} foos and parameter vaz (default: {vaz}).
        '''

    assert bar.__doc__ == '''
        Bar with 7 foos and parameter vaz (default: 5).
        '''


def test_sparse_corrcoef():
    matrix = sparse.rand(100, 10000, density=0.1, format='csr')
    sparse_correlation = ut.corrcoef(matrix)
    numpy_correlation = np.corrcoef(matrix.todense())
    assert numpy_correlation.shape == (100, 100)
    assert np.allclose(sparse_correlation, numpy_correlation)


def test_parallel_map():
    actual = list(ut.parallel_map(lambda index: index, 10000))
    expected = list(range(10000))
    assert actual == expected


def test_unbatched_parallel_map():
    actual = list(ut.parallel_map(lambda index: index,
                                  10000, batches_per_thread=None))
    expected = list(range(10000))
    assert actual == expected


def test_parallel_for():
    mask = np.zeros(10000, dtype='bool')

    def invocation(indices):
        assert not np.any(mask[indices])
        mask[indices] = True

    ut.parallel_for(invocation, 10000)

    assert np.all(mask)


def test_unbatched_parallel_for():
    mask = np.zeros(10000, dtype='bool')

    def invocation(index):
        assert not mask[index]
        mask[index] = True

    ut.parallel_for(invocation, 10000, batches_per_thread=None)

    assert np.all(mask)


def test_parallel_collect():
    shared_storage = ut.SharedStorage()
    shared_storage.set_private('tmp', list)

    def compute(index: int) -> None:
        tmp = shared_storage.get_private('tmp')
        tmp.append(index)

    def merge(from_thread_name: str, into_thread_name: str) -> None:
        from_tmp = shared_storage.get_thread_private('tmp', from_thread_name)
        into_tmp = shared_storage.get_thread_private('tmp', into_thread_name)
        into_tmp.extend(from_tmp)

    final_thread_name = \
        ut.parallel_collect(compute, merge, 10000, batches_per_thread=None)
    final_tmp = shared_storage.get_thread_private('tmp', final_thread_name)
    assert len(final_tmp) == 10000
    assert np.all(np.arange(10000) == sorted(final_tmp))


def test_relayout_matrix():
    rvs = stats.poisson(10, loc=10).rvs
    csr_matrix = sparse.random(1000, 10000, format='csr',
                               dtype='int32', random_state=123456, data_rvs=rvs)
    assert csr_matrix.getformat() == 'csr'

    scipy_csc_matrix = csr_matrix.tocsc()
    metacells_csc_matrix = ut.relayout_compressed(csr_matrix, axis=0)

    assert scipy_csc_matrix.getformat() == 'csc'
    assert metacells_csc_matrix.getformat() == 'csc'

    assert np.all(metacells_csc_matrix.indptr == scipy_csc_matrix.indptr)
    assert np.all(metacells_csc_matrix.todense() == scipy_csc_matrix.todense())

    scipy_csr_matrix = scipy_csc_matrix.tocsr()

    metacells_csr_matrix = ut.relayout_compressed(metacells_csc_matrix, axis=1)

    assert scipy_csr_matrix.getformat() == 'csr'
    assert metacells_csr_matrix.getformat() == 'csr'

    assert np.all(metacells_csr_matrix.indptr == scipy_csr_matrix.indptr)
    assert np.all(metacells_csr_matrix.todense() == scipy_csr_matrix.todense())


def test_downsample_tmp_size():
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


def test_downsample_with_tmp():
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


def test_downsample_array_inplace():
    size = 10
    samples = 20
    random_seed = 123456

    safe = np.arange(size)
    data = np.arange(size)

    ut.downsample_array(data, samples, random_seed=random_seed)

    assert np.all(data <= safe)
    assert np.sum(data) == samples


def test_downsample_array_too_few():
    size = 10
    total = (size * (size - 1)) // 2
    samples = total * 2
    random_seed = 123456

    safe = np.arange(size)
    data = np.arange(size)

    ut.downsample_array(data, samples, random_seed=random_seed)

    assert np.all(data == safe)


def test_downsample_matrix():
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

    matrix = matrix.todense()
    result = ut.downsample_matrix(matrix, axis=1, samples=int(min_sum))
    assert result.shape == matrix.shape
    new_row_sums = ut.sum_matrix(result, axis=1)
    assert np.all(new_row_sums == min_sum)
