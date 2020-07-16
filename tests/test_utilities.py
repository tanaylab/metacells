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
