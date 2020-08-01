'''
Utilities for preparing commonly used data in ``AnnData``.

The functions here use the facilities of :py:mod:`metacells.utilities.annotation` to wrap the
functions of :py:mod:`metacells.utilities.computation` in an easily accessible way.
'''

from typing import Callable, List, Optional, Tuple, Union

import numpy as np  # type: ignore
import scipy.sparse as sparse  # type: ignore
from anndata import AnnData

import metacells.utilities.annotation as uta
import metacells.utilities.computation as utc
import metacells.utilities.timing as timed
import metacells.utilities.typing as utt

__all__ = [
    'prepare',
    'track_base_indices',

    'WhichData',

    'get_derived',

    'Reducer',

    'get_per_obs',
    'get_per_var',

    'get_fraction_of_var_per_obs',
    'get_fraction_of_obs_per_var',
    'get_downsample_of_var_per_obs',
    'get_downsample_of_obs_per_var',

    'get_mean_per_obs',
    'get_mean_per_var',

    'get_fraction_per_obs',
    'get_fraction_per_var',

    'get_variance_per_obs',
    'get_variance_per_var',

    'get_relative_variance_per_obs',
    'get_relative_variance_per_var',

    'get_obs_obs_correlation',
    'get_var_var_correlation',
]


def prepare(adata: AnnData, name: str) -> None:
    '''
    Prepare the annotated ``adata`` for use by the ``metacells`` package.

    This needs the ``name`` of the data contained in ``adata.X``, which also becomes
    the focus data.

    If the name does not start with a ``vo:`` prefix, one is assumed.

    This should be called after populating the ``X`` data for the first time (e.g., by importing the
    data). All the rest of the code in the package assumes this was done.

    .. note::

        This assumes it is safe to arbitrarily slice the ``X`` data.

    .. note::

        When using the layer utilities, do not directly read or write the value of ``X``. Instead
        use :py:func:`metacells.utilities.annotation.get_vo_data`.
    '''
    _, full_name = _assert_prefixed(adata, name, ['vo:'])

    X = adata.X
    assert X is not None
    assert '__x__' not in adata.uns_keys()

    uta.set_data(adata, 'm:__x__', full_name)
    uta.set_data(adata, 'm:__focus__', full_name)
    uta.safe_slicing_data(full_name, uta.ALWAYS_SAFE)


def track_base_indices(adata: AnnData, *, name: str = 'base_index') -> None:
    '''
    Create ``name`` (default: ``base_index``) per-observation (cells) and per-variable (genes) data,
    which will be preserved when creating any :py:func:`metacells.utilities.annotation.slice` of the
    data to easily refer back to the original full data.
    '''
    uta.set_data(adata, 'o' + name, np.arange(adata.n_obs), uta.ALWAYS_SAFE)
    uta.set_data(adata, 'v' + name, np.arange(adata.n_vars), uta.ALWAYS_SAFE)


#: All requests for preparing data take either the base data's name (typically allowing for an
#: implied prefix), or the actual previously fetched base data.
WhichData = Union[str, uta.NamedData]


@timed.call()
def get_derived(
    adata: AnnData,
    deriver: utc.Deriver,
    *,
    of: Optional[WhichData] = None,
    to: Optional[str] = None,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> uta.NamedData:
    '''
    Return a matrix which is derived from some 2D data.

    The ``deriver`` function is invoked as ``reducer(matrix)``. It is expected to take a
    matrix and return a matrix of the same size.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    If ``to`` is not specified, the ``__qualname__`` of the deriver function is used. The
    appropriate prefix is automatically added.

    If ``inplace`` (or ``infocus``), the data will be stored in ``<to>_of_<of>`` for future reuse.

    Use the ``<to>_of_<of>`` data if it exists. Otherwise, compute it, and if ``inplace`` store it
    for future reuse.

    If the data is per-variable-per-observation, and ``infocus`` (implies ``inplace``), also makes
    the result the new focus.

    If the data is per-variable-per-observation, and ``by`` is specified, it forces the layout of
    the returned data (see :py:func:`metacells.utilities.annotation.get_vo_data`).

    Returns the result and its full name.
    '''
    per, full_of = _assert_prefixed(adata, of, ['vo:', 'vv:', 'oo:'])
    if to is None:
        to = deriver.__qualname__
    full_to = '%s%s_of_%s' % (per, to, full_of[3:])

    @timed.call('.' + to)
    def compute() -> utt.Matrix:
        matrix = uta.get_data(adata, full_of, by=by)
        return deriver(matrix)

    return _derive_2d_data(adata, full_of, full_to,
                           compute, inplace, infocus, by)


try:
    from mypy_extensions import NamedArg

    #: A function that reduces an axis of a matrix to a single value.
    Reducer = Callable[[utt.Matrix, NamedArg(int, 'axis')], utt.Matrix]
except ModuleNotFoundError:
    __all__.remove('Reducer')


@timed.call()
def get_per_obs(
    adata: AnnData,
    reducer: 'Reducer',
    *,
    of: Optional[WhichData] = None,
    to: Optional[str] = None,
    inplace: bool = True,
) -> uta.NamedData:
    '''
    Return the some per-observation (cell) reduction of some 2D data.

    The ``reducer`` function is invoked as ``reducer(matrix, axis=1)``. It is expected to take a
    matrix whose rows are observations and merge all the values in each row into a single
    per-observation value.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    If ``to`` is not specified, the ``__qualname__`` of the reducer function is used. The ``o:``
    prefix is automatically added.

    Use the ``<to>_of_<of>`` data if it exists. Otherwise, compute it, and if ``inplace`` store it
    for future reuse.

    Returns the result and its full name.
    '''
    per, full_of = _assert_prefixed(adata, of, ['vo:', 'oo:'])
    to = reducer.__qualname__
    full_to = 'o:%s_of_%s' % (to, full_of)

    @timed.call('.' + to)
    def compute() -> utt.Vector:
        matrix = uta.get_data(adata, full_of, by='obs')
        return utt.to_1d_array(reducer(matrix, axis=1))

    return _derive_1d_data(adata, per, full_of, full_to, compute, inplace)


@timed.call()
def get_per_var(
    adata: AnnData,
    reducer: 'Reducer',
    *,
    of: Optional[WhichData] = None,
    to: Optional[str] = None,
    inplace: bool = True,
) -> uta.NamedData:
    '''
    Return the some per-observation (cell) reduction of some 2D data.

    The ``reducer`` function is invoked as ``reducer(matrix, axis=0)``. It is expected to take a
    matrix whose columns are variables and merge all the values in each column into a single
    per-variable value.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    If ``to`` is not specified, the ``__qualname__`` of the reducer function is used. The ``v:``
    prefix is automatically added.

    Use the ``<to>_of_<of>`` data if it exists. Otherwise, compute it, and if ``inplace`` store it
    for future reuse.

    Returns the result and its full name.
    '''
    per, full_of = _assert_prefixed(adata, of, ['vo:', 'oo:'])
    to = reducer.__qualname__
    full_to = 'v:%s_of_%s' % (to, full_of)

    @timed.call('.' + to)
    def compute() -> utt.Vector:
        matrix = uta.get_data(adata, full_of, by='var')
        return utt.to_1d_array(reducer(matrix, axis=0))

    return _derive_1d_data(adata, per, full_of, full_to, compute, inplace)


@timed.call()
def get_fraction_of_var_per_obs(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> uta.NamedData:
    '''
    Return a matrix containing, in each entry, the fraction the original data (e.g. UMIs) for this
    variable (gene) is in this observation (cell) out of the total for all variables (genes).

    .. note::

        This is probably the version you want: here, the sum of fraction of the genes in a cell is
        1. See :py:func:`get_fraction_of_obs_per_var` for the other way around.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    If ``inplace``, the data will be stored in ``vo:fraction_of_v_per_o_of_vo:<of>`` for future
    reuse, and will also store the intermediate ``o:sum_of_vo:<of>`` per-observation (cell) data.

    If ``infocus`` (implies ``inplace``), also makes the result the new focus.

    If ``by`` is specified, it forces the layout of the returned data (see
    :py:func:`metacells.utilities.annotation.get_vo_data`).

    Returns the result and its full name.

    .. note::

        This assumes all the data values are non-negative.
    '''
    _, full_of = _assert_prefixed(adata, of, ['vo:'])
    full_to = 'vo:fraction_of_v_per_o_of_' + full_of

    @timed.call('.compute')
    def compute() -> utt.Matrix:
        matrix = uta.get_data(adata, full_of, by='obs')
        sum_per_obs = \
            get_per_obs(adata, utc.sum_axis, of=full_of, inplace=inplace).data

        zeros_mask = sum_per_obs == 0
        tmp = np.reciprocal(sum_per_obs, where=~zeros_mask)
        tmp[zeros_mask] = 0

        if sparse.issparse(matrix):
            return matrix.multiply(tmp[:, None])
        return matrix * tmp[:, None]

    return _derive_vo_data(adata, full_of, full_to, uta.SAFE_WHEN_SLICING_OBS,
                           compute, inplace, infocus, by)


@timed.call()
def get_fraction_of_obs_per_var(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> uta.NamedData:
    '''
    Return a matrix containing, in each entry, the fraction the original data (e.g. UMIs) for this
    variable (gene) is in this observation (cell) out of the total for all observations (cells).

    .. note::

        This is probably not the version you want: here, the sum of fractions of the cells in each
        gene is 1. See :py:func:`get_fraction_of_var_per_obs` for the other way around.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    If ``inplace``, the data will be stored in ``vv:fraction_of_o_per_v_of_vo:<of>`` for future
    reuse, and will also store the intermediate ``v:sum_of_vo:<of>`` per-variable (gene) data.

    If ``infocus`` (implies ``inplace``), also makes the result the new focus.

    If ``by`` is specified, it forces the layout of the returned data (see
    :py:func:`metacells.utilities.annotation.get_vo_data`).

    Returns the result and its full name.

    .. note::

        This assumes all the data values are non-negative.
    '''
    _, full_of = _assert_prefixed(adata, of, ['vo:'])
    full_to = 'vo:fraction_of_o_per_v_of_' + full_of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        matrix = uta.get_data(adata, full_of, by='var')
        sum_per_var = \
            get_per_var(adata, utc.sum_axis, of=full_of, inplace=inplace).data

        zeros_mask = sum_per_var == 0
        tmp = np.reciprocal(sum_per_var, where=~zeros_mask)
        tmp[zeros_mask] = 0

        if sparse.issparse(matrix):
            return matrix.multiply(tmp[None, :])
        return matrix * tmp[None, :]

    return _derive_vo_data(adata, full_of, full_to, uta.SAFE_WHEN_SLICING_VAR,
                           compute, inplace, infocus, by)


@timed.call()
def get_downsample_of_var_per_obs(
    adata: AnnData,
    *,
    samples: int,
    random_seed: int = 0,
    of: Optional[WhichData] = None,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> uta.NamedData:
    '''
    Return a matrix containing, for each observation (cell), downsampled data
    for each variable (gene), such that the total sum would be no more
    than ``samples``.

    .. note::

        This is probably the version you want: here, the sum of the genes in a cell will be (at
        most) ``samples``. See :py:func:`get_downsample_of_obs_per_var` for the other way around.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    If ``inplace``, the data will be stored in ``vo:downsample_<samples>_v_per_o_of_vo:<of>`` for
    future reuse.

    If ``infocus`` (implies ``inplace``), also makes the result the new focus.

    If ``by`` is specified, it forces the layout of the returned data (see
    :py:func:`metacells.utilities.annotation.get_vo_data`).

    A ``random_seed`` can be provided to make the operation replicable.

    Returns the result and its full name.

    .. note::

        This assumes all the data values are non-negative integer values (even if the ``dtype`` is a
        floating-point type).
    '''
    _, full_of = _assert_prefixed(adata, of, ['vo:'])
    full_to = 'vo:downsample_%s_v_per_o_of_%s' % (samples, full_of)

    @timed.call('.compute')
    def compute() -> utt.Matrix:
        matrix = uta.get_data(adata, full_of, by=by)
        return utc.downsample_matrix(matrix, axis=1, samples=samples,
                                     random_seed=random_seed)

    return _derive_vo_data(adata, full_of, full_to, uta.SAFE_WHEN_SLICING_OBS,
                           compute, inplace, infocus, by)


@timed.call()
def get_downsample_of_obs_per_var(
    adata: AnnData,
    *,
    samples: int,
    random_seed: int = 0,
    of: Optional[WhichData] = None,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> uta.NamedData:
    '''
    Return a matrix containing, for each variable (gene), downsampled data
    for each observation (cell), such that the total sum would be no more
    than ``samples``.

    .. note::

        This is probably not the version you want: here, the sum of the cells in a gene will be (at
        most) ``samples``. See :py:func:`get_downsample_of_var_per_obs` for the other way around.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    If ``inplace``, the data will be stored in ``vv:downsample_<samples>_o_per_v_of_vo:<of>`` for
    future reuse.

    If ``infocus`` (implies ``inplace``), also makes the result the new focus.

    If ``by`` is specified, it forces the layout of the returned data (see
    :py:func:`metacells.utilities.annotation.get_vo_data`).

    A ``random_seed`` can be provided to make the operation replicable.

    Returns the result and its full name.

    .. note::

        This assumes all the data values are non-negative integer values (even if the ``dtype`` is a
        floating-point type).
    '''
    _, full_of = _assert_prefixed(adata, of, ['vo:'])
    full_to = 'vo:downsample_%s_o_per_v_of_%s' % (samples, full_of)

    @timed.call('.compute')
    def compute() -> utt.Matrix:
        matrix = uta.get_data(adata, full_of, by=by)
        return utc.downsample_matrix(matrix, axis=0, samples=samples,
                                     random_seed=random_seed)

    return _derive_vo_data(adata, full_of, full_to, uta.SAFE_WHEN_SLICING_VAR,
                           compute, inplace, infocus, by)


@timed.call()
def get_mean_per_obs(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> uta.NamedData:
    '''
    Return the mean of the values per-observation (cell) of some data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    Use the ``o:mean_of_vo:<of>`` per-observation (cell) data if it exists. Otherwise, compute it,
    and if ``inplace`` store it for future reuse.

    If ``inplace``, also store the intermediate per-observation ``o:sum_of_vo:<of>`` for future
    reuse.

    Returns the result and its full name.
    '''
    per, full_of = _assert_prefixed(adata, of, ['vo:', 'oo:'])
    full_to = 'o:mean_of_' + full_of

    if isinstance(of, uta.NamedData):
        data = of.data
    else:
        data = uta.get_data(adata, full_of)
    base_count = data.shape[1]

    @timed.call('.compute')
    def compute() -> utt.Vector:
        sum_per_obs = \
            get_per_obs(adata, utc.sum_axis, of=full_of, inplace=inplace).data
        return sum_per_obs / base_count

    return _derive_1d_data(adata, per, full_of, full_to, compute, inplace)


@timed.call()
def get_mean_per_var(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> uta.NamedData:
    '''
    Return the mean of the values per-variable (gene) of some data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    Use the ``v:mean_of_vo:<of>`` per-variable (cell) data if it exists. Otherwise, compute it,
    and if ``inplace`` store it for future reuse.

    If ``inplace``, will also store the intermediate per-variable ``v:sum_of_vo:<of>`` for future
    reuse.

    Returns the result and its full name.
    '''
    per, full_of = _assert_prefixed(adata, of, ['vo:', 'vv:'])
    full_to = 'v:mean_of_' + full_of

    if isinstance(of, uta.NamedData):
        data = of.data
    else:
        data = uta.get_data(adata, full_of)
    base_count = data.shape[0]

    @timed.call('.compute')
    def compute() -> utt.Vector:
        sum_per_var = \
            get_per_var(adata, utc.sum_axis, of=full_of, inplace=inplace).data
        return sum_per_var / base_count

    return _derive_1d_data(adata, per, full_of, full_to, compute, inplace)


@timed.call()
def get_fraction_per_obs(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> uta.NamedData:
    '''
    Return the fraction of the values per-observation (cell) of the total some
    per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    Use the ``o:fraction_of_vo:<of>`` per-observation (cell) data if it exists. Otherwise, compute
    it, and if ``inplace`` store it for future reuse.

    If ``inplace``, also store the intermediate per-observation ``o:sum_of_vo:<of>`` for future
    reuse.

    Returns the result and its full name.
    '''
    per, full_of = _assert_prefixed(adata, of, ['vo:', 'oo:'])
    full_to = 'o:fraction_of_' + full_of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        sum_per_obs = \
            get_per_obs(adata, utc.sum_axis, of=full_of, inplace=inplace).data
        return sum_per_obs / sum_per_obs.sum()

    return _derive_1d_data(adata, per, full_of, full_to, compute, inplace)


@timed.call()
def get_fraction_per_var(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> uta.NamedData:
    '''
    Return the fraction of the values per-variable (gene) of the total of some
    per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    Use the ``v:fraction_of_vo:<of>`` per-variable (cell) data if it exists. Otherwise, compute it,
    and if ``inplace`` store it for future reuse.

    If ``inplace``, will also store the intermediate per-variable ``v:sum_of_vo:<of>`` for future
    reuse.

    Returns the result and its full name.
    '''
    per, full_of = _assert_prefixed(adata, of, ['vo:', 'vv:'])
    full_to = 'v:fraction_of_' + full_of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        sum_per_var = \
            get_per_var(adata, utc.sum_axis, of=full_of, inplace=inplace).data
        return sum_per_var / sum_per_var.sum()

    return _derive_1d_data(adata, per, full_of, full_to, compute, inplace)


@timed.call()
def get_variance_per_obs(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> uta.NamedData:
    '''
    Return the variance of the values per-observation (cell) of some per-variable-per-observation
    data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    Use the ``o:variance_of_vo:<of>`` per-observation (cell) data if it exists. Otherwise, compute
    it, and if ``inplace`` store it for future reuse.

    If ``inplace``, also store the intermediate per-observation ``o:sum_of_vo:<of>`` and
    ``o:sum_squared_of_vo:<of>`` data for future reuse.
    '''
    per, full_of = _assert_prefixed(adata, of, ['vo:', 'oo:'])
    full_to = 'o:variance_of_' + full_of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        sum_per_obs = \
            get_per_obs(adata, utc.sum_axis, of=full_of, inplace=inplace).data
        sum_squared_per_obs = get_per_obs(adata, utc.sum_squared_axis,
                                          of=full_of, inplace=inplace).data
        result = np.square(sum_per_obs).astype(float)
        result /= -adata.n_vars
        result += sum_squared_per_obs
        result /= adata.n_vars
        return result

    return _derive_1d_data(adata, per, full_of, full_to, compute, inplace)


@timed.call()
def get_variance_per_var(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> uta.NamedData:
    '''
    Return the variance of the values per-variable (gene) of some per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    Use the ``v:variance_of_vo:<of>`` per-variable (gene) data if it exists. Otherwise, compute it,
    and if ``inplace`` store it for future reuse.

    If ``inplace``, also store the intermediate per-variable ``v:sum_of_vo:<of>`` and
    ``v:sum_squared_of_vo:<of>`` data for future reuse.
    '''
    per, full_of = _assert_prefixed(adata, of, ['vo:', 'vv:'])
    full_to = 'v:variance_of_' + full_of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        sum_per_var = \
            get_per_var(adata, utc.sum_axis, of=full_of, inplace=inplace).data
        sum_squared_per_var = get_per_var(adata, utc.sum_squared_axis,
                                          of=full_of, inplace=inplace).data
        result = np.square(sum_per_var).astype(float)
        result /= -adata.n_obs
        result += sum_squared_per_var
        result /= adata.n_obs
        return result

    return _derive_1d_data(adata, per, full_of, full_to, compute, inplace)


@timed.call()
def get_relative_variance_per_obs(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> uta.NamedData:
    '''
    Return the log_2(variance/mean) of the values per-observation (cell) of some
    per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    Use the ``o:relative_variance_of_vo:<of>`` per-observation (cell) data if it exists. Otherwise,
    compute it, and if ``inplace`` store it for future reuse.

    If ``inplace``, also store the intermediate per-observation ``o:variance_of_vo:<of>``,
    ``o:mean_of_vo:<of>``, ``o:sum_of_vo:<of>`` and the ``o:sum_squared_of_vo:<of>`` data for future
    reuse.
    '''
    per, full_of = _assert_prefixed(adata, of, ['vo:', 'oo:'])
    full_to = 'o:relative_variance_of_' + full_of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        variance_per_obs = \
            get_variance_per_obs(adata, of=full_of, inplace=inplace).data
        mean_per_obs = \
            get_mean_per_obs(adata, of=full_of, inplace=inplace).data
        zeros_mask = mean_per_obs == 0

        result = np.reciprocal(mean_per_obs, where=~zeros_mask)
        result *= variance_per_obs
        result[zeros_mask] = 1
        np.log2(result, out=result)

        return result

    return _derive_1d_data(adata, per, full_of, full_to, compute, inplace)


@timed.call()
def get_relative_variance_per_var(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> uta.NamedData:
    '''
    Return the log_2(variance/mean) of the values per-variable (gene) of some
    per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    Use the ``v:relative_variance_of_vo:<of>`` per-observation (cell) data if it exists. Otherwise,
    compute it, and if ``inplace`` store it for future reuse.

    If ``inplace``, also store the intermediate per-variable ``v:variance_of_vo:<of>``,
    ``v:mean_of_vo:<of>``, ``v:sum_of_vo:<of>`` and the ``v:sum_squared_of_vo:<of>`` data for future
    reuse.
    '''
    per, full_of = _assert_prefixed(adata, of, ['vo:', 'vv:'])
    full_to = 'v:relative_variance_of_' + full_of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        variance_per_var = \
            get_variance_per_var(adata, of=full_of, inplace=inplace).data
        mean_per_var = \
            get_mean_per_var(adata, of=full_of, inplace=inplace).data
        zeros_mask = mean_per_var == 0

        result = np.reciprocal(mean_per_var, where=~zeros_mask)
        result *= variance_per_var
        result[zeros_mask] = 1
        np.log2(result, out=result)

        return result

    return _derive_1d_data(adata, per, full_of, full_to, compute, inplace)


@timed.call()
def get_obs_obs_correlation(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> uta.NamedData:
    '''
    Compute correlation between observations (cells) of the ``adata``.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    If ``inplace``, store the result in ``adata`` for future reuse, and also stores the intermediate
    ``o:sum_of_vo:<of>`` per-observation (cell) data.

    Returns the matrix and its full name.
    '''
    per, full_of = _assert_prefixed(adata, of, ['vo:', 'oo:'])
    full_to = 'oo:correlation_of_' + full_of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        if per == 'vo:':
            matrix = uta.get_data(adata, full_of, by='obs')
            correlations = utc.corrcoef(matrix, rowvar=True)
        else:
            matrix = uta.get_data(adata, full_of)
            correlations = utc.corrcoef(matrix)

        return correlations

    return _derive_2d_data(adata, full_of, full_to, compute, inplace)


@timed.call()
def get_var_var_correlation(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> uta.NamedData:
    '''
    Compute correlation between variables (genes) of the ``adata``.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used. If the
    name does not start with a ``vo:`` prefix, one is assumed.

    If ``inplace``, store the result in ``adata`` for future reuse, and also stores the intermediate
    ``v:sum_of_vo:<of>`` per-variable (gene) data.

    Returns the matrix and its full name.
    '''
    per, full_of = _assert_prefixed(adata, of, ['vo:', 'vv:'])
    full_to = 'vv:correlation_of_' + full_of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        if per == 'vo:':
            matrix = uta.get_data(adata, full_of, by='var')
            correlations = utc.corrcoef(matrix, rowvar=False)
        else:
            matrix = uta.get_data(adata, full_of)
            correlations = utc.corrcoef(matrix)

        return correlations.T

    return _derive_2d_data(adata, full_of, full_to, compute, inplace)


def _assert_prefixed(
    adata: AnnData,
    of: Optional[WhichData],
    allowed: List[str]
) -> Tuple[str, str]:
    if of is None:
        name = adata.uns['__focus__']
    elif isinstance(of, uta.NamedData):
        name = of.name
    else:
        name = of

    for prefix in ['m:', 'vo:', 'o:', 'v:', 'oo:', 'vv:']:
        if not name.startswith(prefix):
            continue
        if prefix in allowed:
            return prefix, name
        raise ValueError('the data: %s does not have the expected prefix: %s'
                         % (name, ', '.join(['"%s"' % expected
                                             for expected in allowed])))
    return allowed[0], allowed[0] + name


def _derive_vo_data(
    adata: AnnData,
    full_of: str,
    full_to: str,
    slicing_mask: uta.SlicingMask,
    compute: Callable[[], utt.Matrix],
    inplace: bool,
    infocus: bool,
    by: Optional[str],
) -> uta.NamedData:
    if inplace or infocus:
        uta.safe_slicing_derived(base=full_of, derived=full_to,
                                 slicing_mask=slicing_mask)

    return uta.get_vo_data(adata, full_to, compute=compute,
                           inplace=inplace, infocus=infocus, by=by)


def _derive_2d_data(
    adata: AnnData,
    full_of: str,
    full_to: str,
    compute: Callable[[], utt.Matrix],
    inplace: bool,
    infocus: bool = False,
    by: Optional[str] = None,
) -> uta.NamedData:
    per_of = full_of[:3]
    per_to = full_to[:3]

    assert per_of in ['vo:', 'oo:', 'vv:']
    assert per_to in ['vo:', 'oo:', 'vv:']

    if per_to == per_of:
        slicing_mask = uta.ALWAYS_SAFE
    elif per_to == 'oo:':
        slicing_mask = uta.SAFE_WHEN_SLICING_OBS
    else:
        assert per_to == 'vv:'
        slicing_mask = uta.SAFE_WHEN_SLICING_VAR

    return _derive_data(adata, full_of, full_to, slicing_mask, compute, inplace,
                        infocus, by)


def _derive_1d_data(
    adata: AnnData,
    per_of: str,
    full_of: str,
    full_to: str,
    compute: Callable[[], utt.Vector],
    inplace: bool,
) -> uta.NamedData:
    per_of = full_of[:3]
    per_to = full_to[:2]

    assert per_of in ['vo:', 'oo:', 'vv:']
    assert per_to in ['v:', 'o:']

    if per_to[0] == per_of[0] == per_of[1]:
        slicing_mask = uta.ALWAYS_SAFE
    elif per_to == 'v:':
        slicing_mask = uta.SAFE_WHEN_SLICING_VAR
    else:
        assert per_to == 'o:'
        slicing_mask = uta.SAFE_WHEN_SLICING_OBS

    return _derive_data(adata, full_of, full_to, slicing_mask, compute, inplace)


def _derive_data(
    adata: AnnData,
    full_of: str,
    full_to: str,
    slicing_mask: uta.SlicingMask,
    compute: Callable[[], utt.Vector],
    inplace: bool,
    infocus: bool = False,
    by: Optional[str] = None,
) -> uta.NamedData:
    if inplace:
        uta.safe_slicing_derived(base=full_of, derived=full_to,
                                 slicing_mask=slicing_mask)

    return uta.get_named_data(adata, full_to, compute=compute, inplace=inplace,
                              infocus=infocus, by=by)
