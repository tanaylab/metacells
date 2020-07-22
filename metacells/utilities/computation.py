'''
Utilities for performing efficient computations.
'''

import os
from math import ceil
from typing import Any, Callable, Dict, Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import yaml
from scipy import sparse  # type: ignore

import metacells.utilities.timing as timed
from metacells.utilities.threading import parallel_for

Matrix = Union[sparse.spmatrix, np.ndarray, pd.DataFrame]
Vector = Union[np.ndarray, pd.Series]

MINIMAL_INSTRUCTIONS_PER_BATCH = 1_000_000
REDUCE_MATRIX_STEPS = [('sum_matrix', 'n'),
                       ('nnz_matrix', 'n'), ('max_matrix', 'n')]
REDUCE_MATRIX_VARIANTS = ['sparse', 'dense']


__all__ = [
    'as_array',
    'corrcoef',
    'log_matrix',
    'sum_matrix',
    'nnz_matrix',
    'max_matrix',
]


class ReduceMatrixStepPredictor:
    '''
    A predictor of CPU instructions used when reducing a matrix along some axis.
    '''

    def __init__(self, data: Dict[str, Any]) -> None:
        '''
        Initialize the predictor from data loaded from YAML.
        '''
        assert isinstance(data, dict)
        #: The O(...) complexity of the step.
        self.complexity = data.pop('complexity')
        #: Number of always-needed instructions.
        self.constant = float(data.pop('constant', 0))
        #: Rate of growth of needed instructions.
        self.factor = float(data.pop('factor', 4))

        assert self.complexity in ['n', 'n_log_n']
        assert self.constant >= 0
        assert self.factor > 0

    def predict(self, matrix: Matrix, axis: int) -> float:
        '''
        Predict the number of instructions needed to process a single ``matrix`` row/column
        ``axis``.
        '''
        results_count = matrix.shape[1 - axis]

        if sparse.issparse(matrix):
            variant = 'sparse'
            size = len(matrix.indices) / results_count
        else:
            variant = 'dense'
            size = matrix.shape[axis]

        timed.parameters(variant=variant, m=results_count,
                         complexity=self.complexity, n=size)

        if self.complexity == 'n_log_n':
            size *= np.log2(size)

        return self.constant + self.factor * size


class ReduceMatrixStepsPredictor:
    '''
    Predictors of CPU instructions used for the computation steps that reduce a matrix along some
    axis.
    '''

    def __init__(self, data: Dict[str, Any]) -> None:
        '''
        Initialize the predictors from data loaded from YAML.
        '''
        self.steps: Dict[str, Dict[str, ReduceMatrixStepPredictor]] = {
        }  # The predictor of each step.

        assert isinstance(data, dict)
        for step, variants in data.items():
            step_data = self.steps[step] = {}
            assert isinstance(variants, dict)
            for variant, datum in variants.items():
                step_data[variant] = ReduceMatrixStepPredictor(datum)

        for step, complexity in REDUCE_MATRIX_STEPS:
            step_data = self.steps.get(step)  # type: ignore
            if step_data is None:
                step_data = self.steps[step] = {}
            for variant in REDUCE_MATRIX_VARIANTS:
                if variant not in step_data:
                    step_data[variant] = \
                        ReduceMatrixStepPredictor(dict(complexity=complexity))

    def predict(self, step: str, matrix: Matrix, axis: int) -> float:
        '''
        Predict the number of instructions needed for the ``step`` to process a single ``matrix``
        row/column ``axis``.
        '''
        if sparse.issparse(matrix):
            return self.steps[step]['sparse'].predict(matrix, axis)
        return self.steps[step]['dense'].predict(matrix, axis)


REDUCE_MATRIX_STEPS_PREDICTOR = \
    ReduceMatrixStepsPredictor(  #
        yaml.safe_load(  #
            open(  #
                os.environ.get(  #
                    'METACELL_TIMING_YAML',
                    os.path.join(os.path.dirname(__file__), 'timing.yaml'))
            ).read()
        )
    )


def as_array(data: Union[Matrix, Vector]) -> np.ndarray:
    '''
    Convert a matrix to an array.
    '''
    if sparse.issparse(data):
        data = data.todense()
    return np.ravel(data)


@timed.call()
def corrcoef(matrix: Matrix) -> np.ndarray:
    '''
    Compute correlations between all observations (rows, cells) containing variables (columns,
    genes).

    This should give the same results as ``numpy.corrcoef``, but faster for sparse matrices.

    .. note::

        To correlate between observations (cells), the expected layout is the transpose of the
        layout of ``X`` in ``AnnData``.
    '''
    if not sparse.issparse(matrix):
        return np.corrcoef(matrix)

    obs_count = matrix.shape[0]
    var_count = matrix.shape[1]
    timed.parameters(obs_count=obs_count, var_count=var_count)
    sum_of_rows = matrix.sum(axis=1)
    assert len(sum_of_rows) == obs_count
    centering = sum_of_rows.dot(sum_of_rows.T) / var_count
    correlations = (matrix.dot(matrix.T) - centering) / (var_count - 1)
    assert correlations.shape == (obs_count, obs_count)
    diagonal = np.diag(correlations)
    correlations /= np.sqrt(np.outer(diagonal, diagonal))
    return correlations


@timed.call()
def log_matrix(
    matrix: Matrix,
    *,
    base: Optional[float] = None,
    normalization: float = 1,
) -> np.ndarray:
    '''
    Compute the ``log2`` of some ``matrix``.

    The ``base`` is added to the count before ``log2`` is applied, to handle the common case of zero
    values in sparse data.

    The ``normalization`` (default: {normalization}) is added to the count before the log is
    applied, to handle the common case of sparse data.
    '''
    if sparse.issparse(matrix):
        matrix = matrix.todense()
    elif isinstance(matrix, pd.DataFrame):
        matrix = np.copy(matrix.values)
    else:
        assert isinstance(matrix, np.ndarray)
        matrix = np.copy(matrix)

    matrix += normalization
    if base == 2:
        np.log2(matrix, out=matrix)
    else:
        np.log(matrix, out=matrix)
        if base is not None:
            assert base > 0
            matrix /= np.log(base)

    return matrix


@timed.call()
def sum_matrix(matrix: Matrix, axis: int) -> np.ndarray:
    '''
    Compute the total per row (``axis`` = 1) or column (``axis`` = 0) of some ``matrix``.
    '''
    return _reduce_matrix('sum_matrix', matrix, axis, lambda matrix: matrix.sum(axis=axis))


@timed.call()
def nnz_matrix(matrix: Matrix, axis: int) -> np.ndarray:
    '''
    Compute the number of non-zero elements per row (``axis`` = 1) or column (``axis`` = 0) of some
    ``matrix``.
    '''
    if sparse.issparse(matrix):
        return _reduce_matrix('nnz_matrix', matrix, axis, lambda matrix: matrix.getnnz(axis=axis))
    return _reduce_matrix('nnz_matrix', matrix, axis,
                          lambda matrix: np.count_nonzero(matrix, axis=axis))


@timed.call()
def max_matrix(matrix: Matrix, axis: int) -> np.ndarray:
    '''
    Compute the maximal element per row (``axis`` = 1) or column (``axis`` = 0) of some ``matrix``.
    '''
    if sparse.issparse(matrix):
        return _reduce_matrix('max_matrix', matrix, axis, lambda matrix: matrix.max(axis=axis))
    return _reduce_matrix('max_matrix', matrix, axis, lambda matrix: np.amax(matrix, axis=axis))


def _reduce_matrix(
    step: str,
    matrix: Matrix,
    axis: int,
    reducer: Callable,
) -> np.ndarray:
    assert matrix.ndim == 2
    assert 0 <= axis <= 1

    results_count = matrix.shape[1 - axis]
    results = np.empty(results_count)

    if sparse.issparse(matrix):
        if axis == 0:
            matrix = matrix.tocsc()
        else:
            matrix = matrix.tocsr()

    if axis == 0:
        def batch_reducer(indices: range) -> None:
            results[indices] = as_array(reducer(matrix[:, indices]))
    else:
        def batch_reducer(indices: range) -> None:
            results[indices] = as_array(reducer(matrix[indices, :]))

    mean_instructions_per_result = \
        REDUCE_MATRIX_STEPS_PREDICTOR.predict(step, matrix, axis)
    minimal_invocations_per_batch = \
        ceil(MINIMAL_INSTRUCTIONS_PER_BATCH / mean_instructions_per_result)
    timed.parameters(expected_instructions=mean_instructions_per_result
                     * results_count)

    parallel_for(batch_reducer, results_count,
                 minimal_invocations_per_batch=minimal_invocations_per_batch)

    return results
