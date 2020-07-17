'''
Test the utility functions.
'''

import numpy as np  # type: ignore
from scipy import sparse  # type: ignore

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
    sparse_correlation = ut.sparse_corrcoef(matrix)
    numpy_correlation = np.corrcoef(matrix.todense())
    assert np.allclose(sparse_correlation, numpy_correlation)


def test_parallel_map():
    actual = list(ut.parallel_map(lambda index: index, 10000))
    expected = list(range(10000))
    assert actual == expected


def test_unbatcheded_parallel_map():
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


def test_unbatcheded_parallel_for():
    mask = np.zeros(10000, dtype='bool')

    def invocation(index):
        assert not mask[index]
        mask[index] = True

    ut.parallel_for(invocation, 10000, batches_per_thread=None)

    assert np.all(mask)
