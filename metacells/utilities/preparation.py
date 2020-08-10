'''
Utilities for preparing commonly used data in ``AnnData``.

The functions here use the facilities of :py:mod:`metacells.utilities.annotation` to wrap the
functions of :py:mod:`metacells.utilities.computation` in an easily accessible way.
'''

from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

import numpy as np  # type: ignore
import scipy.sparse as sparse  # type: ignore
from anndata import AnnData

import metacells.utilities.annotation as uta
import metacells.utilities.computation as utc
import metacells.utilities.timing as timed
import metacells.utilities.typing as utt
from metacells.utilities.annotation import _items

__all__ = [
    'prepare',
    'track_base_indices',

    'NamedData',
    'WhichData',

    'get_derived',

    'Reducer',

    'get_per_obs',
    'get_per_var',

    'get_fraction_of_var_per_obs',
    'get_fraction_of_obs_per_var',
    'get_downsample_of_var_per_obs',
    'get_downsample_of_obs_per_var',

    'get_log',

    'get_mean_per_obs',
    'get_mean_per_var',

    'get_fraction_per_obs',
    'get_fraction_per_var',

    'get_variance_per_obs',
    'get_variance_per_var',

    'get_relative_variance_per_var',
    'get_normalized_variance_per_var',

    'get_obs_obs_correlation',
    'get_var_var_correlation',
]


def prepare(adata: AnnData, name: str) -> None:
    '''
    Prepare the annotated ``adata`` for use by the ``metacells`` package.

    This needs the ``name`` of the data contained in ``adata.X``, which also becomes
    the focus data.

    This should be called after populating the ``X`` data for the first time (e.g., by importing the
    data). All the rest of the code in the package assumes this was done.

    .. note::

        This assumes it is safe to arbitrarily slice all the currently existing data.

    .. note::

        When using the layer utilities, do not directly read or write the value of ``X``. Instead
        use :py:func:`metacells.utilities.annotation.get_vo_data`.
    '''
    X = adata.X
    assert X is not None
    assert '__x__' not in adata.uns_keys()

    uta.safe_slicing_data(name, uta.ALWAYS_SAFE)
    adata.uns['__x__'] = name
    adata.uns['__focus__'] = name

    for annotations in (adata.layers,
                        adata.obs,
                        adata.var,
                        adata.obsp,
                        adata.varp,
                        adata.uns):
        for data_name, _ in _items(annotations):
            uta.safe_slicing_data(data_name, uta.ALWAYS_SAFE)


def track_base_indices(adata: AnnData, *, name: str = 'base_index') -> None:
    '''
    Create ``obs_<name>`` per-observation (cells) and ``var_<name>`` per-variable (genes) data,
    which will be preserved when creating any :py:func:`metacells.utilities.annotation.slice` of the
    data to easily refer back to the original full data.
    '''
    obs_name = 'obs_' + name
    var_name = 'var_' + name

    adata.obs[obs_name] = np.arange(adata.n_obs)
    adata.var[var_name] = np.arange(adata.n_vars)

    uta.safe_slicing_data(obs_name, uta.ALWAYS_SAFE)
    uta.safe_slicing_data(var_name, uta.ALWAYS_SAFE)


class NamedData(NamedTuple):
    '''
    Data stored in ``AnnData`` with its name.
    '''

    #: The name of the data.
    #:
    #: .. note::
    #:
    #:    This must be globally unique across all annotations of all kinds.
    name: str

    #: The actual data.
    #:
    #: This will be anything at all for metadata, This will be a
    #: :py:class:`metacells.utilities.typing.Matrix` for per-variable-per-observation,
    #: per-observation-per-observation, or per-variable-per-variable, data; a
    #: :py:class:`metacells.utilities.typing.Vector` for per-observation or per-variable data; and
    #: anything at all for metadata.
    data: Any


#: All requests for deriving data take either the base data's name or the actual previously fetched
#: base data.
WhichData = Union[str, NamedData]


@timed.call()
def get_derived(
    adata: AnnData,
    derive: Callable[[utt.Shaped], utt.Shaped],
    *,
    of: Optional[WhichData] = None,
    to: Optional[str] = None,
    slicing_mask: uta.SlicingMask = uta.ALWAYS_SAFE,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedData:
    '''
    Return some data which is derived from some base data.

    The ``derive`` function is invoked as ``reducer(data)``. It is expected to take an array or
    matrix, and return data of the same shape.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    If ``to`` is not specified, the ``__qualname__`` of the derive function is used.

    Use the ``<to>_of_<of>`` data if it exists. Otherwise, compute it, and if ``inplace`` store it
    for future reuse.

    If the data is per-variable-per-observation, and ``infocus`` (implies ``inplace``), also makes
    the result the new focus.

    If the data is per-variable-per-observation, and ``by`` is specified, it forces the layout of
    the returned data (see :py:func:`metacells.utilities.annotation.get_vo_data`).

    The default ``slicing_mask`` assumes it is safe to arbitrarily slice the results (as long as the
    base data is also safe to slice), that is, that this is an element-wise computation. Otherwise,
    modify the mask to reflect the restrictions imposed by the computation itself.

    Returns the result and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'vv', 'oo', 'v', 'o'])
    to = '%s_of_%s' % (to or derive.__qualname__, of)

    @timed.call('.' + to)
    def compute() -> utt.Shaped:
        assert isinstance(of, str)
        return derive(uta.get_data(adata, of, by=by))

    return _derive_data(adata, of, per_of, to, slicing_mask, compute, inplace, infocus, by)


@timed.call()
def get_log(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    base: Optional[float] = None,
    normalization: float = 0,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedData:
    '''
    Return the logarithm of some data.

    Computes log with the specified ``base`` (by default, the natiral logarithm) after adding
    ``normalization`` to the data. If the data contains zeros and no positive normalization was
    specified, then the result will contain ``NaN`` values.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``log_<base>_of_<of>`` or ``log_<base>_of_<normalization>_plus_<of>`` data if it exists
    (where the default base is reported as ``e``). Otherwise, compute it, and if ``inplace`` store
    it for future reuse.

    If the data is per-variable-per-observation, and ``infocus`` (implies ``inplace``), also makes
    the result the new focus.

    If the data is per-variable-per-observation, and ``by`` is specified, it forces the layout of
    the returned data (see :py:func:`metacells.utilities.annotation.get_vo_data`).

    Returns the result and its name.
    '''
    if normalization > 0:
        to = 'log_%s_plus_%s' % (base or 'e', normalization)
    else:
        to = 'log_%s' % (base or 'e')

    def derive(matrix: utt.Matrix) -> utt.Matrix:
        return utc.log_matrix(matrix, base=base, normalization=normalization)

    return get_derived(adata, of=of, inplace=inplace, infocus=infocus, by=by,
                       to=to, derive=derive)


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
) -> NamedData:
    '''
    Return a matrix containing, for each observation (cell), downsampled data for each variable
    (gene), such that the total sum would be no more than ``samples``.

    .. note::

        This is probably the version you want: here, the sum of the genes in a cell will be (at
        most) ``samples``. See :py:func:`get_downsample_of_obs_per_var` for the other way around.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    If ``inplace``, the data will be stored in ``downsampled_<samples>_var_per_obs_of_<of>`` for
    future reuse.

    If ``infocus`` (implies ``inplace``), also makes the result the new focus.

    If ``by`` is specified, it forces the layout of the returned data (see
    :py:func:`metacells.utilities.annotation.get_vo_data`).

    A ``random_seed`` can be provided to make the operation replicable.

    Returns the result and its name.

    .. note::

        This assumes all the data values are non-negative integer values (even if the ``dtype`` is a
        floating-point type).
    '''
    def derive(data: utt.Shaped) -> utt.Shaped:
        return utc.downsample_matrix(data, axis=1, samples=samples,
                                     eliminate_zeros=True,
                                     random_seed=random_seed)

    return get_derived(adata, of=of, inplace=inplace, infocus=infocus, by=by,
                       to='downsampled_%s_var_per_obs' % samples,
                       slicing_mask=uta.SAFE_WHEN_SLICING_OBS,
                       derive=derive)


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
) -> NamedData:
    '''
    Return a matrix containing, for each variable (gene), downsampled data for each observation
    (cell), such that the total sum would be no more than ``samples``.

    .. note::

        This is probably not the version you want: here, the sum of the cells in a gene will be (at
        most) ``samples``. See :py:func:`get_downsample_of_var_per_obs` for the other way around.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    If ``inplace``, the data will be stored in ``downsampled_<samples>_obs_per_var_of_<of>`` for
    future reuse.

    If ``infocus`` (implies ``inplace``), also makes the result the new focus.

    If ``by`` is specified, it forces the layout of the returned data (see
    :py:func:`metacells.utilities.annotation.get_vo_data`).

    A ``random_seed`` can be provided to make the operation replicable.

    Returns the result and its name.

    .. note::

        This assumes all the data values are non-negative integer values (even if the ``dtype`` is a
        floating-point type).
    '''
    def derive(data: utt.Shaped) -> utt.Shaped:
        return utc.downsample_matrix(data, axis=0, samples=samples,
                                     eliminate_zeros=True,
                                     random_seed=random_seed)

    return get_derived(adata, of=of, inplace=inplace, infocus=infocus, by=by,
                       to='downsampled_%s_obs_per_var' % samples,
                       slicing_mask=uta.SAFE_WHEN_SLICING_VAR,
                       derive=derive)


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
) -> NamedData:
    '''
    Return the some per-observation (cell) reduction of some 2D data.

    The ``reducer`` function is invoked as ``reducer(matrix, axis=1)``. It is expected to take a
    matrix whose rows are observations and merge all the values in each row into a single
    per-observation value.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    If ``to`` is not specified, the ``__qualname__`` of the reducer function is used.

    Use the ``<to>_per_obs_of_<of>`` data if it exists. Otherwise, compute it, and if ``inplace``
    store it for future reuse.

    Returns the result and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'oo'])
    to = '%s_per_obs_of_%s' % (to or reducer.__qualname__, of)

    @timed.call('.' + to)
    def compute() -> utt.Vector:
        assert isinstance(of, str)
        matrix = uta.get_data(adata, of, by='obs')
        return utt.to_1d_array(reducer(matrix, axis=1))

    return _derive_1d_data(adata, per_of, of, 'o', to, compute, inplace)


@timed.call()
def get_per_var(
    adata: AnnData,
    reducer: 'Reducer',
    *,
    of: Optional[WhichData] = None,
    to: Optional[str] = None,
    inplace: bool = True,
) -> NamedData:
    '''
    Return the some per-observation (cell) reduction of some 2D data.

    The ``reducer`` function is invoked as ``reducer(matrix, axis=0)``. It is expected to take a
    matrix whose columns are variables and merge all the values in each column into a single
    per-variable value.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    If ``to`` is not specified, the ``__qualname__`` of the reducer function is used.

    Use the ``<to>_per_var_of_<of>`` data if it exists. Otherwise, compute it, and if ``inplace``
    store it for future reuse.

    Returns the result and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'oo'])
    to = '%s_per_var_of_%s' % (to or reducer.__qualname__, of)

    @timed.call('.' + to)
    def compute() -> utt.Vector:
        assert isinstance(of, str)
        matrix = uta.get_data(adata, of, by='var')
        return utt.to_1d_array(reducer(matrix, axis=0))

    return _derive_1d_data(adata, per_of, of, 'v', to, compute, inplace)


@timed.call()
def get_fraction_of_var_per_obs(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedData:
    '''
    Return a matrix containing, in each entry, the fraction the original data (e.g. UMIs) for this
    variable (gene) is in this observation (cell) out of the total for all variables (genes).

    .. note::

        This is probably the version you want: here, the sum of fraction of the genes in a cell is
        1. See :py:func:`get_fraction_of_obs_per_var` for the other way around.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    If ``inplace``, the data will be stored in ``fraction_of_v_per_o_of_vo:<of>`` for future reuse,
    and will also store the intermediate ``sum_per_obs_of_<of>`` per-observation (cell) data.

    If ``infocus`` (implies ``inplace``), also makes the result the new focus.

    If ``by`` is specified, it forces the layout of the returned data (see
    :py:func:`metacells.utilities.annotation.get_vo_data`).

    Returns the result and its name.

    .. note::

        This assumes all the data values are non-negative.
    '''
    _, of = _per_of(adata, of, ['vo'])
    to = 'fraction_of_var_per_obs_of_' + of

    @timed.call('.compute')
    def compute() -> utt.Matrix:
        assert isinstance(of, str)
        matrix = utt.unpandas(uta.get_data(adata, of, by='obs'))
        sum_per_obs = \
            utt.unpandas(get_per_obs(adata, utc.sum_axis,
                                     of=of, inplace=inplace).data)

        zeros_mask = sum_per_obs == 0
        tmp = np.reciprocal(sum_per_obs, where=~zeros_mask)
        tmp[zeros_mask] = 0

        if sparse.issparse(matrix):
            return matrix.multiply(tmp[:, None])
        return matrix * tmp[:, None]

    return _derive_vo_data(adata, of, to, uta.SAFE_WHEN_SLICING_OBS,
                           compute, inplace, infocus, by)


@timed.call()
def get_fraction_of_obs_per_var(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedData:
    '''
    Return a matrix containing, in each entry, the fraction the original data (e.g. UMIs) for this
    variable (gene) is in this observation (cell) out of the total for all observations (cells).

    .. note::

        This is probably not the version you want: here, the sum of fractions of the cells in each
        gene is 1. See :py:func:`get_fraction_of_var_per_obs` for the other way around.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    If ``inplace``, the data will be stored in ``fraction_of_obs_per_var_of_<of>`` for future reuse,
    and will also store the intermediate ``sum_per_var_of_<of>`` per-variable (gene) data.

    If ``infocus`` (implies ``inplace``), also makes the result the new focus.

    If ``by`` is specified, it forces the layout of the returned data (see
    :py:func:`metacells.utilities.annotation.get_vo_data`).

    Returns the result and its name.

    .. note::

        This assumes all the data values are non-negative.
    '''
    _, of = _per_of(adata, of, ['vo'])
    to = 'fraction_of_obs_per_var_of_' + of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        assert isinstance(of, str)
        matrix = utt.unpandas(uta.get_data(adata, of, by='var'))
        sum_per_var = \
            utt.unpandas(get_per_var(adata, utc.sum_axis,
                                     of=of, inplace=inplace).data)

        zeros_mask = sum_per_var == 0
        tmp = np.reciprocal(sum_per_var, where=~zeros_mask)
        tmp[zeros_mask] = 0

        if sparse.issparse(matrix):
            return matrix.multiply(tmp[None, :])
        return matrix * tmp[None, :]

    return _derive_vo_data(adata, of, to, uta.SAFE_WHEN_SLICING_VAR,
                           compute, inplace, infocus, by)


@timed.call()
def get_mean_per_obs(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> NamedData:
    '''
    Return the mean of the values per-observation (cell) of some data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``mean_per_obs_of_<of>`` per-observation (cell) data if it exists. Otherwise, compute
    it, and if ``inplace`` store it for future reuse.

    If ``inplace``, also store the intermediate per-observation ``sum_per_obs_of_<of>`` for future
    reuse.

    Returns the result and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'oo'])
    to = 'mean_per_obs_of_' + of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        sum_per_obs = \
            utt.unpandas(get_per_obs(adata, utc.sum_axis,
                                     of=of, inplace=inplace).data)
        return sum_per_obs / adata.n_vars

    return _derive_1d_data(adata, per_of, of, 'o', to, compute, inplace)


@timed.call()
def get_mean_per_var(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> NamedData:
    '''
    Return the mean of the values per-variable (gene) of some data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``v:mean_of_vo:<of>`` per-variable (cell) data if it exists. Otherwise, compute it,
    and if ``inplace`` store it for future reuse.

    If ``inplace``, will also store the intermediate per-variable ``v:sum_of_vo:<of>`` for future
    reuse.

    Returns the result and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'vv'])
    to = 'mean_per_var_of_' + of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        sum_per_var = \
            utt.unpandas(get_per_var(adata, utc.sum_axis,
                                     of=of, inplace=inplace).data)
        return sum_per_var / adata.n_obs

    return _derive_1d_data(adata, per_of, of, 'v', to, compute, inplace)


@timed.call()
def get_fraction_per_obs(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> NamedData:
    '''
    Return the fraction of the values per-observation (cell) of the total some
    per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``fraction_per_obs_of_<of>`` per-observation (cell) data if it exists. Otherwise,
    compute it, and if ``inplace`` store it for future reuse.

    If ``inplace``, also store the intermediate per-observation ``sum_per_obs_of_<of>`` for future
    reuse.

    Returns the result and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'oo'])
    to = 'fraction_per_obs_of_' + of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        sum_per_obs = \
            utt.unpandas(get_per_obs(adata, utc.sum_axis,
                                     of=of, inplace=inplace).data)
        return sum_per_obs / sum_per_obs.sum()

    return _derive_1d_data(adata, per_of, of, 'o', to, compute, inplace)


@timed.call()
def get_fraction_per_var(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> NamedData:
    '''
    Return the fraction of the values per-variable (gene) of the total of some
    per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``fraction_per_var_of_<of>`` per-variable (cell) data if it exists. Otherwise, compute
    it, and if ``inplace`` store it for future reuse.

    If ``inplace``, will also store the intermediate per-variable ``sum_per_var_of_<of>`` for future
    reuse.

    Returns the result and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'vv'])
    to = 'fraction_per_var_of_' + of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        sum_per_var = \
            utt.unpandas(get_per_var(adata, utc.sum_axis,
                                     of=of, inplace=inplace).data)
        return sum_per_var / sum_per_var.sum()

    return _derive_1d_data(adata, per_of, of, 'v', to, compute, inplace)


@timed.call()
def get_variance_per_obs(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> NamedData:
    '''
    Return the variance of the values per-observation (cell) of some per-variable-per-observation
    data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``variance_per_obs_of_<of>`` per-observation (cell) data if it exists. Otherwise,
    compute it, and if ``inplace`` store it for future reuse.

    If ``inplace``, also store the intermediate per-observation ``sum_per_obs_of_<of>`` and
    ``sum_squared_per_obs_of_<of>`` data for future reuse.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'oo'])
    to = 'variance_per_obs_of_' + of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        sum_per_obs = \
            utt.unpandas(get_per_obs(adata, utc.sum_axis,
                                     of=of, inplace=inplace).data)
        sum_squared_per_obs = \
            utt.unpandas(get_per_obs(adata, utc.sum_squared_axis,
                                     of=of, inplace=inplace).data)
        result = np.square(sum_per_obs).astype(float)
        result /= -adata.n_vars
        result += sum_squared_per_obs
        result /= adata.n_vars
        return result

    return _derive_1d_data(adata, per_of, of, 'o', to, compute, inplace)


@timed.call()
def get_variance_per_var(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> NamedData:
    '''
    Return the variance of the values per-variable (gene) of some per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``variance_per_var_of_<of>`` per-variable (gene) data if it exists. Otherwise, compute
    it, and if ``inplace`` store it for future reuse.

    If ``inplace``, also store the intermediate per-variable ``sum_per_var_of_<of>`` and
    ``sum_squared_per_var_of_<of>`` data for future reuse.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'vv'])
    to = 'variance_per_var_of_' + of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        sum_per_var = \
            utt.unpandas(get_per_var(adata, utc.sum_axis,
                                     of=of, inplace=inplace).data)
        sum_squared_per_var = \
            utt.unpandas(get_per_var(adata, utc.sum_squared_axis,
                                     of=of, inplace=inplace).data)
        result = np.square(sum_per_var).astype(float)
        result /= -adata.n_obs
        result += sum_squared_per_var
        result /= adata.n_obs
        return result

    return _derive_1d_data(adata, per_of, of, 'v', to, compute, inplace)


@timed.call()
def get_relative_variance_per_var(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> NamedData:
    '''
    Return the log_2(variance/mean) of the values per-variable (gene) of some
    per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``relative_variance_per_var_of_<of>`` per-variable (cell) data if it exists. Otherwise,
    compute it, and if ``inplace`` store it for future reuse.

    If ``inplace``, also store the intermediate per-variable ``variance_of_<of>``,
    ``mean_per_var_of_<of>``, ``sum_per_var_of_<of>`` and the ``sum_squared_per_var_of_<of>`` data
    for future reuse.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'vv'])
    to = 'relative_variance_per_var_of_' + of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        variance_per_var = \
            utt.unpandas(get_variance_per_var(adata, of=of,
                                              inplace=inplace).data)
        mean_per_var = \
            utt.unpandas(get_mean_per_var(adata, of=of, inplace=inplace).data)
        zeros_mask = mean_per_var == 0

        result = np.reciprocal(mean_per_var, where=~zeros_mask)
        result *= variance_per_var
        result[zeros_mask] = 1
        np.log2(result, out=result)

        return result

    return _derive_1d_data(adata, per_of, of, 'v', to, compute, inplace)


@timed.call()
def get_normalized_variance_per_var(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    window_size: int = 100,
    inplace: bool = True,
) -> NamedData:
    '''
    Return the (relative_variance - median-relative-variance-of-similar) of the values per-variable
    (gene) of some per-variable-per-observation data.

    The median-relative-variance-of-similar is the median relative variance of the ``window_size``
    variables (genes) with the most similar mean to the one being normalized. In general, the
    relative variance tends to (but doesn't always) go up for higher-mean variables. By normalizing
    by the median relative variance of variables of a similar mean, we factor out this dependency on
    the mean to decide which variables (genes) carry more meaningful information (and are more
    suitable to be picked as "feature genes").

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``normalized_variance_by_<window_size>_per_var_of_<of>`` per-variable (gene) data if it
    exists. Otherwise, compute it, and if ``inplace`` store it for future reuse.

    If ``inplace``, also store the intermediate per-variable ``relative_variance_per_var_of_<of>``,
    ``variance_per_var_of_<of>``, ``mean_per_var_of_<of>``, ``sum_per_var_of_<of>`` and the
    ``sum_squared_per_var_of_<of>`` data for future reuse.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'vv'])
    to = 'normalized_variance_by_%s_per_var_of_%s' % (window_size, of)

    @timed.call('.compute')
    def compute() -> utt.Vector:
        relative_variance_per_var = \
            utt.unpandas(get_relative_variance_per_var(adata, of=of,
                                                       inplace=inplace).data)
        mean_per_var = \
            utt.unpandas(get_mean_per_var(adata, of=of, inplace=inplace).data)
        median_variance_per_var = utc.sliding_window_function(relative_variance_per_var,
                                                              function='median',
                                                              window_size=window_size,
                                                              order_by=mean_per_var)
        return relative_variance_per_var - median_variance_per_var

    return _derive_1d_data(adata, per_of, of, 'v', to, compute, inplace)


@timed.call()
def get_obs_obs_correlation(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> NamedData:
    '''
    Compute correlation between observations (cells) of the ``adata``.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``obs_obs_correlation_of_<of>`` per-observation-per-observation (cell) data if it
    exists. Otherwise, compute it, and if ``inplace`` store it for future reuse.

    Returns the matrix and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'oo'])
    to = 'obs_obs_correlation_of_' + of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        assert isinstance(of, str)
        if per_of == 'vo':
            matrix = utt.unpandas(uta.get_data(adata, of, by='obs'))
            correlations = utc.corrcoef(matrix, rowvar=True)
        else:
            matrix = utt.unpandas(uta.get_data(adata, of))
            correlations = utc.corrcoef(matrix)

        return correlations

    return _derive_2d_data(adata, per_of, of, 'oo', to, compute, inplace)


@timed.call()
def get_var_var_correlation(
    adata: AnnData,
    *,
    of: Optional[WhichData] = None,
    inplace: bool = True,
) -> NamedData:
    '''
    Compute correlation between variables (genes) of the ``adata``.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``var_var_correlation_of_<of>`` per-variable-per-variable (gene) data if it
    exists. Otherwise, compute it, and if ``inplace`` store it for future reuse.

    Returns the matrix and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'vv'])
    to = 'var_var_correlation_of_' + of

    @timed.call('.compute')
    def compute() -> utt.Vector:
        assert isinstance(of, str)
        if per_of == 'vo':
            matrix = utt.unpandas(uta.get_data(adata, of, by='var'))
            correlations = utc.corrcoef(matrix, rowvar=False)
        else:
            matrix = utt.unpandas(uta.get_data(adata, of))
            correlations = utc.corrcoef(matrix)

        return correlations.T

    return _derive_2d_data(adata, per_of, of, 'vv', to, compute, inplace)


def _per_of(
    adata: AnnData,
    of: Optional[WhichData],
    allowed: List[str]
) -> Tuple[str, str]:
    if of is None:
        name = adata.uns['__focus__']
    elif isinstance(of, NamedData):
        name = of.name
    else:
        name = of

    per_of = uta.which_data(adata, name)
    if per_of not in allowed:
        raise ValueError('the data: %s is not per: %s'
                         % (name, ', '.join(['"%s"' % expected
                                             for expected in allowed])))

    return per_of, name


def _derive_vo_data(
    adata: AnnData,
    of: str,
    to: str,
    slicing_mask: uta.SlicingMask,
    compute: Callable[[], utt.Matrix],
    inplace: bool,
    infocus: bool,
    by: Optional[str],
) -> NamedData:
    if inplace or infocus:
        uta.safe_slicing_derived(base=of, derived=to,
                                 slicing_mask=slicing_mask)

    data = uta.get_vo_data(adata, to, compute=compute,
                           inplace=inplace, infocus=infocus, by=by)

    return NamedData(name=to, data=data)


def _derive_2d_data(
    adata: AnnData,
    per_of: str,
    of: str,
    per_to: str,
    to: str,
    compute: Callable[[], utt.Matrix],
    inplace: bool,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedData:
    assert per_of in ('vo', 'oo', 'vv')
    assert per_to in ('vo', 'oo', 'vv')

    if per_to == per_of:
        slicing_mask = uta.ALWAYS_SAFE
    elif per_to == 'oo':
        slicing_mask = uta.SAFE_WHEN_SLICING_OBS
    else:
        assert per_to == 'vv'
        slicing_mask = uta.SAFE_WHEN_SLICING_VAR

    return _derive_data(adata, of, per_to, to, slicing_mask, compute,
                        inplace, infocus, by)


def _derive_1d_data(
    adata: AnnData,
    per_of: str,
    of: str,
    per_to: str,
    to: str,
    compute: Callable[[], utt.Vector],
    inplace: bool,
) -> NamedData:
    if per_to[0] == per_of[0] == per_of[1]:
        slicing_mask = uta.ALWAYS_SAFE
    elif per_to == 'v':
        slicing_mask = uta.SAFE_WHEN_SLICING_VAR
    else:
        assert per_to == 'o'
        slicing_mask = uta.SAFE_WHEN_SLICING_OBS

    return _derive_data(adata, of, per_to, to, slicing_mask, compute, inplace)


def _derive_data(
    adata: AnnData,
    of: str,
    per_to: str,
    to: str,
    slicing_mask: uta.SlicingMask,
    compute: Callable[[], utt.Vector],
    inplace: bool,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedData:
    if inplace:
        uta.safe_slicing_derived(base=of, derived=to,
                                 slicing_mask=slicing_mask)

    data = uta.get_data(adata, to, per=per_to, compute=compute,
                        inplace=inplace, infocus=infocus, by=by)

    return NamedData(name=to, data=data)
