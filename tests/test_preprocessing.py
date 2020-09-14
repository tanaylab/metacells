'''
Test the preprocessing functions.
'''

import numpy as np  # type: ignore
from anndata import AnnData
from scipy import sparse  # type: ignore
from scipy import stats

import metacells.preprocessing as pp
import metacells.utilities as ut

# pylint: disable=missing-function-docstring


def test_mean() -> None:
    matrix = np.random.rand(100, 200)
    adata = AnnData(matrix)
    ut.setup(adata, x_name='test')

    metacells_mean_per_row = pp.get_mean_per_obs(adata).proper
    numpy_mean_per_row = matrix.mean(axis=1)
    assert np.allclose(metacells_mean_per_row, numpy_mean_per_row)

    metacells_mean_per_column = pp.get_mean_per_var(adata).proper
    numpy_mean_per_column = matrix.mean(axis=0)
    assert np.allclose(metacells_mean_per_column, numpy_mean_per_column)


def test_variance() -> None:
    matrix = np.random.rand(100, 200)
    adata = AnnData(matrix)
    ut.setup(adata, x_name='test')

    metacells_variance_per_row = pp.get_variance_per_obs(adata).proper
    numpy_variance_per_row = matrix.var(axis=1)
    assert np.allclose(metacells_variance_per_row, numpy_variance_per_row)

    metacells_variance_per_column = pp.get_variance_per_var(adata).proper
    numpy_variance_per_column = matrix.var(axis=0)
    assert np.allclose(metacells_variance_per_column,
                       numpy_variance_per_column)


def test_focus_on() -> None:
    rvs = stats.poisson(10, loc=10).rvs
    matrix = sparse.random(100, 1000, format='csr',
                           dtype='int32', random_state=123456, data_rvs=rvs)
    adata = AnnData(matrix)

    ut.setup(adata, x_name='test')
    assert ut.get_focus_name(adata) == 'test'

    with ut.focus_on(pp.get_log, adata, normalization=1) as log_data:
        outer_focus = 'test|log_e_normalization_1'
        assert ut.get_focus_name(adata) == outer_focus
        assert id(log_data.proper) == id(adata.layers[outer_focus])

        with ut.focus_on(pp.get_downsample_of_var_per_obs, adata, 'test', samples=10) \
                as downsamled_data:
            inner_focus = 'test|downsample_10_var_per_obs'
            assert ut.get_focus_name(adata) == inner_focus
            assert id(downsamled_data.proper) == id(adata.layers[inner_focus])

        assert ut.get_focus_name(adata) == outer_focus
        assert id(log_data.proper) == id(adata.layers[outer_focus])

        with ut.focus_on(pp.get_downsample_of_var_per_obs, adata, 'test', samples=10) \
                as downsamled_data:
            inner_focus = 'test|downsample_10_var_per_obs'
            assert ut.get_focus_name(adata) == inner_focus
            assert id(downsamled_data.proper) == id(adata.layers[inner_focus])
            ut.del_data(adata, outer_focus, per='vo')

        assert ut.get_focus_name(adata) == 'test'

    assert ut.get_focus_name(adata) == 'test'


def test_dense_annotations() -> None:
    matrix = np.array([[0, 1, 2], [3, 4, 5]], dtype='float')
    _test_annotations(matrix)


def test_sparse_annotations() -> None:
    matrix = np.array([[0, 1, 2], [3, 4, 5]], dtype='float')
    matrix = sparse.csr_matrix(matrix)
    _test_annotations(matrix)


def _test_annotations(full_matrix: ut.Matrix) -> None:
    adata = AnnData(full_matrix)
    ut.setup(adata, x_name='test')

    assert np.allclose(pp.get_per_obs(adata, ut.nnz_per).proper,
                       np.array([2, 3]))
    assert np.allclose(pp.get_per_var(adata, ut.nnz_per).proper,
                       np.array([1, 2, 2]))

    assert np.allclose(pp.get_per_obs(adata, ut.sum_per).proper,
                       np.array([3, 12]))
    assert np.allclose(pp.get_per_var(adata, ut.sum_per).proper,
                       np.array([3, 5, 7]))

    assert np.allclose(pp.get_per_obs(adata, ut.max_per).proper,
                       np.array([2, 5]))
    assert np.allclose(pp.get_per_var(adata, ut.max_per).proper,
                       np.array([3, 4, 5]))

    assert np.allclose(pp.get_per_obs(adata, ut.min_per).proper,
                       np.array([0, 3]))
    assert np.allclose(pp.get_per_var(adata, ut.min_per).proper,
                       np.array([0, 1, 2]))

    assert np.allclose(pp.get_per_obs(adata, ut.sum_squared_per).proper,
                       np.array([5, 50]))
    assert np.allclose(pp.get_per_var(adata, ut.sum_squared_per).proper,
                       np.array([9, 17, 29]))

    assert np.allclose(pp.get_fraction_per_obs(adata).proper,
                       np.array([3/15, 12/15]))
    assert np.allclose(pp.get_fraction_per_var(adata).proper,
                       np.array([3/15, 5/15, 7/15]))

    assert np.allclose(pp.get_mean_per_obs(adata).proper,
                       np.array([3/3, 12/3]))
    assert np.allclose(pp.get_mean_per_var(adata).proper,
                       np.array([3/2, 5/2, 7/2]))

    assert np.allclose(pp.get_variance_per_obs(adata).proper,
                       np.array([5/3 - (3/3)**2, 50/3 - (12/3)**2]))

    assert np.allclose(pp.get_variance_per_var(adata).proper,
                       np.array([9/2 - (3/2)**2, 17/2 - (5/2)**2, 29/2 - (7/2)**2]))

    assert np.allclose(pp.get_normalized_variance_per_var(adata).proper,
                       np.log2(np.array([(9/2 - (3/2)**2) / (3/2),
                                         (17/2 - (5/2)**2) / (5/2),
                                         (29/2 - (7/2)**2) / (7/2)])))

    dense = ut.to_dense_matrix(pp.get_fraction_of_var_per_obs(adata).proper)
    assert np.allclose(dense, np.array([[0/3, 1/3, 2/3], [3/12, 4/12, 5/12]]))

    dense = ut.to_dense_matrix(pp.get_fraction_of_obs_per_var(adata).proper)
    assert np.allclose(dense, np.array([[0/3, 1/5, 2/7], [3/3, 4/5, 5/7]]))
