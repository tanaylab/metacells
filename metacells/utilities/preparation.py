'''
Utilities for preparing commonly used data in ``AnnData``.

The functions here use the facilities of :py:mod:`metacells.utilities.annotation` to wrap the
functions of :py:mod:`metacells.utilities.computation` in an easily accessible way.
'''

from typing import Callable, List, NamedTuple, Optional, Tuple, Type, TypeVar

import numpy as np  # type: ignore
from anndata import AnnData

import metacells.utilities.annotation as uta
import metacells.utilities.computation as utc
import metacells.utilities.documentation as utd
import metacells.utilities.timing as utm
import metacells.utilities.typing as utt
from metacells.utilities.annotation import _items

__all__ = [
    'prepare',
    'track_base_indices',

    'NamedShaped',
    'NamedMatrix',
    'NamedVector',

    'get_derived',
    'get_derived_matrix',
    'get_derived_vector',

    'get_log',
    'get_log_matrix',
    'get_log_vector',

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
    assert utt.Shaped.be(X).ndim == 2
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
    (default ``name``: {name}), which will be preserved when creating any
    :py:func:`metacells.utilities.annotation.slice` of the data to easily refer back to the original
    full data.
    '''
    obs_name = 'obs_' + name
    var_name = 'var_' + name

    adata.obs[obs_name] = np.arange(adata.n_obs)
    adata.var[var_name] = np.arange(adata.n_vars)

    uta.safe_slicing_data(obs_name, uta.ALWAYS_SAFE)
    uta.safe_slicing_data(var_name, uta.ALWAYS_SAFE)


class NamedShaped(NamedTuple):
    '''
    Any :py:const:`metacells.utilities.typing.Shaped` data stored in ``AnnData``, with its name.
    '''

    #: The name of the data.
    #:
    #: .. note::
    #:
    #:    This must be globally unique across all annotations of all kinds.
    name: str

    #: The actual :py:const:`metacells.utilities.typing.Shaped` data.
    shaped: utt.Shaped

    @property
    def proper(self) -> utt.ProperShaped:
        '''
        Access the :py:const:`metacells.utilities.typing.ProperShaped` data
        using :py:func:`metacells.utilities.typing.to_proper`.
        '''
        return utt.to_proper(self.shaped)


NS = TypeVar('NS', bound=NamedShaped)


class NamedMatrix(NamedShaped):
    '''
    :py:const:`metacells.utilities.typing.Matrix` data stored in ``AnnData`` with its name.
    '''
    @property
    def matrix(self) -> utt.Matrix:
        '''
        Access the :py:const:`metacells.utilities.typing.Matrix` data.
        '''
        shaped = self.shaped
        assert shaped.ndim == 2
        return shaped  # type: ignore

    @property
    def proper(  # type: ignore # pylint: disable=arguments-differ
        self,
        *,
        layout: str = 'row_major'
    ) -> utt.ProperMatrix:
        '''
        Access the :py:const:`metacells.utilities.typing.ProperMatrix` data
        using :py:func:`metacells.utilities.typing.to_proper_matrix`.
        '''
        return utt.to_proper_matrix(self.matrix, layout=layout)

    @staticmethod
    def be(named: NamedShaped) -> 'NamedMatrix':
        '''
        Construct from arbitrary :py:const:`NamedShaped`.
        '''
        assert named.shaped.ndim == 2
        return NamedMatrix(name=named.name, shaped=named.shaped)


class NamedVector(NamedShaped):
    '''
    :py:const:`metacells.utilities.typing.Vector` data stored in ``AnnData`` with its name.
    '''
    @property
    def vector(self) -> utt.Vector:
        '''
        Access the :py:const:`metacells.utilities.typing.Vector` data.
        '''
        shaped = self.shaped
        assert shaped.ndim == 1
        return shaped  # type: ignore

    @property
    def proper(self) -> utt.ProperVector:
        '''
        Access the :py:const:`metacells.utilities.typing.ProperVector` data
        using :py:func:`metacells.utilities.typing.to_proper_vector`.
        '''
        return utt.to_proper_vector(self.vector)

    @staticmethod
    def be(named: NamedShaped) -> 'NamedVector':
        '''
        Construct from arbitrary :py:const:`NamedShaped`.
        '''
        assert named.shaped.ndim == 1
        return NamedVector(name=named.name, shaped=named.shaped)


S = TypeVar('S', bound=utt.Shaped)


@utm.timed_call()
@utd.expand_doc()
def get_derived(
    adata: AnnData,
    derive: Callable[[S], S],
    of: Optional[str] = None,
    *,
    to: Optional[str] = None,
    slicing_mask: uta.SlicingMask = uta.ALWAYS_SAFE,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedShaped:
    '''
    Return some data which is derived from some base data.

    The ``derive`` function is invoked as ``reducer(data)``. It is expected to take an array or
    matrix, and return data of the same shape.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    If ``to`` is not specified, the ``__qualname__`` of the derive function is used.

    Use the ``<of>|<to>`` data if it exists. Otherwise, compute it, and if ``inplace`` (default:
    {inplace}) store it for future reuse.

    If the data is per-variable-per-observation, and ``infocus`` (default: {infocus}, implies
    ``inplace``), also makes the result the new focus.

    If the data is per-variable-per-observation, and ``by`` is specified, it forces the layout of
    the returned data (see :py:func:`metacells.utilities.annotation.get_vo_data`).

    The default ``slicing_mask`` :py:const:`metacells.utilities.annotation.ALWAYS_SAFE` assumes it
    is safe to arbitrarily slice the results (as long as the base data is also safe to slice), that
    is, that this is an element-wise computation. Otherwise, modify the mask to reflect the
    restrictions imposed by the computation itself.

    Returns the result and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'vv', 'oo', 'v', 'o'])
    to = '%s|%s' % (of, to or derive.__qualname__)

    @utm.timed_call('.' + to)
    def compute() -> utt.Shaped:
        assert isinstance(of, str)
        return derive(uta.get_proper(adata, of, by=by))  # type: ignore

    return _derive_data(NamedShaped, adata, of, per_of, to, slicing_mask,
                        compute, inplace, infocus, by)


@utm.timed_call()
def get_derived_matrix(
    adata: AnnData,
    derive: Callable[[S], S],
    of: Optional[str] = None,
    *,
    to: Optional[str] = None,
    slicing_mask: uta.SlicingMask = uta.ALWAYS_SAFE,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedMatrix:
    '''
    Same as :py:func:`get_derived` when the result is a
    :py:const:`metacells.utilities.typing.Matrix`.
    '''
    return NamedMatrix.be(get_derived(adata, derive, of, to=to,
                                      slicing_mask=slicing_mask,
                                      inplace=inplace, infocus=infocus, by=by))


@utm.timed_call()
def get_derived_vector(
    adata: AnnData,
    derive: Callable[[S], S],
    of: Optional[str] = None,
    *,
    to: Optional[str] = None,
    slicing_mask: uta.SlicingMask = uta.ALWAYS_SAFE,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedVector:
    '''
    Same as :py:func:`get_derived` when the result is a
    :py:const:`metacells.utilities.typing.Vector`.
    '''
    return NamedVector.be(get_derived(adata, derive, of, to=to,
                                      slicing_mask=slicing_mask,
                                      inplace=inplace, infocus=infocus, by=by))


@utm.timed_call()
@utd.expand_doc()
def get_log(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    base: Optional[float] = None,
    normalization: float = 0,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedShaped:
    '''
    Return the logarithm of some data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Computes the log of the data with the specified ``base`` (default: {base}) and ``normalization``
    (default: {normalization}). See :py:func:`metacells.utilities.computation.log_data` for details.

    Use the ``<of>|log_<base>`` or ``<of>|log_<base>_normalization_<normalization>`` data if it
    exists (where the default base is reported as ``e``). Otherwise, compute it, and if ``inplace``
    (default: {inplace}) store it for future reuse.

    If the data is per-variable-per-observation, and ``infocus`` (default: {infocus}, implies
    ``inplace``), also makes the result the new focus.

    If the data is per-variable-per-observation, and ``by`` is specified, it forces the layout of
    the returned data (see :py:func:`metacells.utilities.annotation.get_vo_data`).

    Returns the result and its name.
    '''
    if normalization != 0:
        to = 'log_%s_normalization_%s' % (base or 'e', normalization)
    else:
        to = 'log_%s' % (base or 'e')

    def derive(data: utt.Shaped) -> utt.Shaped:
        return utc.log_data(data, base=base, normalization=normalization)

    return get_derived(adata, derive, of, to=to,
                       inplace=inplace, infocus=infocus, by=by)


@utm.timed_call()
def get_log_matrix(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    base: Optional[float] = None,
    normalization: float = 0,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedMatrix:
    '''
    Same as :py:func:`get_derived` when the result is a
    :py:const:`metacells.utilities.typing.Matrix`.
    '''
    return NamedMatrix.be(get_log(adata, of, base=base,
                                  normalization=normalization,
                                  inplace=inplace, infocus=infocus, by=by))


@utm.timed_call()
def get_log_vector(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    base: Optional[float] = None,
    normalization: float = 0,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedVector:
    '''
    Same as :py:func:`get_derived` when the result is a
    :py:const:`metacells.utilities.typing.Vector`.
    '''
    return NamedVector.be(get_log(adata, of, base=base,
                                  normalization=normalization,
                                  inplace=inplace, infocus=infocus, by=by))


@utm.timed_call()
@utd.expand_doc()
def get_downsample_of_var_per_obs(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    samples: int,
    random_seed: int = 0,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedMatrix:
    '''
    Return a matrix containing, for each observation (cell), downsampled data for each variable
    (gene), such that the total sum would be no more than ``samples``.

    .. note::

        This is probably the version you want: here, the sum of the genes in a cell will be (at
        most) ``samples``. See :py:func:`get_downsample_of_obs_per_var` for the other way around.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    If ``inplace`` (default: {inplace}), the data will be stored in
    ``<of>|downsample_<samples>_var_per_obs`` for future reuse.

    If ``infocus`` (default: {infocus}, implies ``inplace``), also makes the result the new focus.

    If ``by`` is specified, it forces the layout of the returned data (see
    :py:func:`metacells.utilities.annotation.get_vo_data`).

    A non-zero ``random_seed`` (default: {random_seed}) can be provided to make the operation
    replicable.

    Returns the result and its name.

    .. note::

        This assumes all the data values are non-negative integer values (even if the ``dtype`` is a
        floating-point type).
    '''
    def derive(data: utt.Matrix) -> utt.Matrix:
        return utc.downsample_matrix(data, axis=1, samples=samples,
                                     eliminate_zeros=True,
                                     random_seed=random_seed)

    return get_derived_matrix(adata, derive, of,
                              to='downsample_%s_var_per_obs' % samples,
                              slicing_mask=uta.SAFE_WHEN_SLICING_OBS,
                              inplace=inplace, infocus=infocus, by=by)


@utm.timed_call()
@utd.expand_doc()
def get_downsample_of_obs_per_var(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    samples: int,
    random_seed: int = 0,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedMatrix:
    '''
    Return a matrix containing, for each variable (gene), downsampled data for each observation
    (cell), such that the total sum would be no more than ``samples``.

    .. note::

        This is probably not the version you want: here, the sum of the cells in a gene will be (at
        most) ``samples``. See :py:func:`get_downsample_of_var_per_obs` for the other way around.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    If ``inplace`` (default: {inplace}), the data will be stored in
    ``<of>|downsample_<samples>_obs_per_var`` for future reuse.

    If ``infocus`` (default: {infocus}, implies ``inplace``), also makes the result the new focus.

    If ``by`` is specified, it forces the layout of the returned data (see
    :py:func:`metacells.utilities.annotation.get_vo_data`).

    A non-zero ``random_seed`` (default: {random_seed}) can be provided to make the operation
    replicable.

    Returns the result and its name.

    .. note::

        This assumes all the data values are non-negative integer values (even if the ``dtype`` is a
        floating-point type).
    '''
    def derive(data: utt.Matrix) -> utt.Matrix:
        return utc.downsample_matrix(data, axis=0, samples=samples,
                                     eliminate_zeros=True,
                                     random_seed=random_seed)

    return get_derived_matrix(adata, derive, of,
                              to='downsample_%s_obs_per_var' % samples,
                              slicing_mask=uta.SAFE_WHEN_SLICING_VAR,
                              inplace=inplace, infocus=infocus, by=by)


try:
    from mypy_extensions import NamedArg

    #: A function that reduces an axis of a matrix to a single value.
    Reducer = Callable[[utt.Matrix, NamedArg(int, 'axis')], utt.Matrix]
except ModuleNotFoundError:
    __all__.remove('Reducer')


@utm.timed_call()
@utd.expand_doc()
def get_per_obs(
    adata: AnnData,
    reducer: 'Reducer',
    of: Optional[str] = None,
    *,
    to: Optional[str] = None,
    inplace: bool = True,
) -> NamedVector:
    '''
    Return the some per-observation (cell) reduction of some 2D data.

    The ``reducer`` function is invoked as ``reducer(matrix, axis=1)``. It is expected to take a
    matrix whose rows are observations and merge all the values in each row into a single
    per-observation value.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    If ``to`` is not specified, the ``__qualname__`` of the reducer function is used.

    Use the ``<of>|<to>_per_obs`` data if it exists. Otherwise, compute it, and if ``inplace``
    (default: {inplace}), store it for future reuse.

    Returns the result and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'oo'])
    to = '%s|%s_per_obs' % (of, to or reducer.__qualname__)

    @utm.timed_call('.' + to)
    def compute() -> utt.Vector:
        assert isinstance(of, str)
        matrix = uta.get_proper_matrix(adata, of, by='obs')
        return utt.to_dense_vector(reducer(matrix, axis=1))

    return _derive_1d_data(adata, per_of, of, 'o', to, compute, inplace)


@utm.timed_call()
@utd.expand_doc()
def get_per_var(
    adata: AnnData,
    reducer: 'Reducer',
    of: Optional[str] = None,
    *,
    to: Optional[str] = None,
    inplace: bool = True,
) -> NamedVector:
    '''
    Return the some per-observation (cell) reduction of some 2D data.

    The ``reducer`` function is invoked as ``reducer(matrix, axis=0)``. It is expected to take a
    matrix whose columns are variables and merge all the values in each column into a single
    per-variable value.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    If ``to`` is not specified, the ``__qualname__`` of the reducer function is used.

    Use the ``<of>|<to>_per_var`` data if it exists. Otherwise, compute it, and if ``inplace``
    (default: {inplace}), store it for future reuse.

    Returns the result and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'oo'])
    to = '%s|%s_per_var' % (of, to or reducer.__qualname__)

    @utm.timed_call('.' + to)
    def compute() -> utt.Vector:
        assert isinstance(of, str)
        matrix = uta.get_proper_matrix(adata, of, by='var')
        return utt.to_dense_vector(reducer(matrix, axis=0))

    return _derive_1d_data(adata, per_of, of, 'v', to, compute, inplace)


@utm.timed_call()
@utd.expand_doc()
def get_fraction_of_var_per_obs(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedMatrix:
    '''
    Return a matrix containing, in each entry, the fraction the original data (e.g. UMIs) for this
    variable (gene) is in this observation (cell) out of the total for all variables (genes).

    .. note::

        This is probably the version you want: here, the sum of fraction of the genes in a cell is
        1. See :py:func:`get_fraction_of_obs_per_var` for the other way around.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    If ``inplace`` (default: {inplace}, the data will be stored in ``<of>|fraction_of_var_per_obs``
    for future reuse, and will also store the intermediate ``<of>|sum_per_obs`` per-observation
    (cell) data.

    If ``infocus`` (default: {infocus}, implies ``inplace``), also makes the result the new focus.

    If ``by`` is specified, it forces the layout of the returned data (see
    :py:func:`metacells.utilities.annotation.get_vo_data`).

    Returns the result and its name.

    .. note::

        This assumes all the data values are non-negative.
    '''
    _, of = _per_of(adata, of, ['vo'])
    to = of + '|fraction_of_var_per_obs'

    @utm.timed_call('.compute')
    def compute() -> utt.Matrix:
        assert isinstance(of, str)
        matrix = uta.get_proper_matrix(adata, of, by='obs')
        sum_per_obs = \
            get_per_obs(adata, utc.sum_axis, of, inplace=inplace).proper

        zeros_mask = sum_per_obs == 0
        tmp = np.reciprocal(sum_per_obs, where=~zeros_mask)
        tmp[zeros_mask] = 0

        sparse = utt.SparseMatrix.maybe(matrix)
        if sparse is not None:
            return sparse.multiply(tmp[:, None])

        dense = utt.DenseMatrix.be(matrix)
        return dense * tmp[:, None]

    return _derive_vo_data(adata, of, to, uta.SAFE_WHEN_SLICING_OBS,
                           compute, inplace, infocus, by)


@utm.timed_call()
@utd.expand_doc()
def get_fraction_of_obs_per_var(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedMatrix:
    '''
    Return a matrix containing, in each entry, the fraction the original data (e.g. UMIs) for this
    variable (gene) is in this observation (cell) out of the total for all observations (cells).

    .. note::

        This is probably not the version you want: here, the sum of fractions of the cells in each
        gene is 1. See :py:func:`get_fraction_of_var_per_obs` for the other way around.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    If ``inplace`` (default: {inplace}), the data will be stored in ``<of>|fraction_of_obs_per_var``
    for future reuse, and will also store the intermediate ``<of>|sum_per_var`` per-variable (gene)
    data.

    If ``infocus`` (default: {infocus}, implies ``inplace``), also makes the result the new focus.

    If ``by`` is specified, it forces the layout of the returned data (see
    :py:func:`metacells.utilities.annotation.get_vo_data`).

    Returns the result and its name.

    .. note::

        This assumes all the data values are non-negative.
    '''
    _, of = _per_of(adata, of, ['vo'])
    to = of + '|fraction_of_obs_per_var'

    @utm.timed_call('.compute')
    def compute() -> utt.Matrix:
        assert isinstance(of, str)
        matrix = uta.get_proper_matrix(adata, of, by='var')
        sum_per_var = \
            get_per_var(adata, utc.sum_axis, of, inplace=inplace).proper

        zeros_mask = sum_per_var == 0
        tmp = np.reciprocal(sum_per_var, where=~zeros_mask)
        tmp[zeros_mask] = 0

        sparse = utt.SparseMatrix.maybe(matrix)
        if sparse is not None:
            return sparse.multiply(tmp[None, :])

        dense = utt.DenseMatrix.be(matrix)
        return dense * tmp[None, :]

    return _derive_vo_data(adata, of, to, uta.SAFE_WHEN_SLICING_VAR,
                           compute, inplace, infocus, by)


@utm.timed_call()
@utd.expand_doc()
def get_mean_per_obs(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> NamedVector:
    '''
    Return the mean of the values per-observation (cell) of some data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``<of>|mean_per_obs`` per-observation (cell) data if it exists. Otherwise, compute
    it, and if ``inplace`` (default: {inplace}), store it for future reuse.

    If ``inplace``, also store the intermediate per-observation ``<of>|sum_per_obs`` for future
    reuse.

    Returns the result and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'oo'])
    to = of + '|mean_per_obs'

    @utm.timed_call('.compute')
    def compute() -> utt.Vector:
        sum_per_obs = \
            get_per_obs(adata, utc.sum_axis, of, inplace=inplace).proper
        return sum_per_obs / adata.n_vars

    return _derive_1d_data(adata, per_of, of, 'o', to, compute, inplace)


@utm.timed_call()
@utd.expand_doc()
def get_mean_per_var(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> NamedVector:
    '''
    Return the mean of the values per-variable (gene) of some data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``<of>|mean_per_var`` per-variable (cell) data if it exists. Otherwise, compute it, and
    if ``inplace`` (default: {inplace}), store it for future reuse.

    If ``inplace``, will also store the intermediate per-variable ``<of>|sum_per_var`` for future
    reuse.

    Returns the result and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'vv'])
    to = of + '|mean_per_var'

    @utm.timed_call('.compute')
    def compute() -> utt.Vector:
        sum_per_var = \
            get_per_var(adata, utc.sum_axis, of, inplace=inplace).proper
        return sum_per_var / adata.n_obs

    return _derive_1d_data(adata, per_of, of, 'v', to, compute, inplace)


@utm.timed_call()
@utd.expand_doc()
def get_fraction_per_obs(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> NamedVector:
    '''
    Return the fraction of the values per-observation (cell) of the total some
    per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``<of>|fraction_per_obs`` per-observation (cell) data if it exists. Otherwise, compute
    it, and if ``inplace`` (default: {inplace}), store it for future reuse.

    If ``inplace``, also store the intermediate per-observation ``<of>|sum_per_obs`` for future
    reuse.

    Returns the result and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'oo'])
    to = of + '|fraction_per_obs'

    @utm.timed_call('.compute')
    def compute() -> utt.Vector:
        sum_per_obs = \
            get_per_obs(adata, utc.sum_axis, of, inplace=inplace).proper
        return sum_per_obs / sum_per_obs.sum()

    return _derive_1d_data(adata, per_of, of, 'o', to, compute, inplace)


@utm.timed_call()
@utd.expand_doc()
def get_fraction_per_var(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> NamedVector:
    '''
    Return the fraction of the values per-variable (gene) of the total of some
    per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``<of>|fraction_per_var`` per-variable (cell) data if it exists. Otherwise, compute it,
    and if ``inplace`` (default: {inplace}) store it for future reuse.

    If ``inplace``, will also store the intermediate per-variable ``<of>|sum_per_var`` for future
    reuse.

    Returns the result and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'vv'])
    to = of + '|fraction_per_var'

    @utm.timed_call('.compute')
    def compute() -> utt.Vector:
        sum_per_var = \
            get_per_var(adata, utc.sum_axis, of, inplace=inplace).proper
        return sum_per_var / sum_per_var.sum()

    return _derive_1d_data(adata, per_of, of, 'v', to, compute, inplace)


@utm.timed_call()
@utd.expand_doc()
def get_variance_per_obs(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> NamedVector:
    '''
    Return the variance of the values per-observation (cell) of some per-variable-per-observation
    data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``<of>|variance_per_obs`` per-observation (cell) data if it exists. Otherwise, compute
    it, and if ``inplace`` (default: {inplace}), store it for future reuse.

    If ``inplace``, also store the intermediate per-observation ``<of>|sum_per_obs`` and
    ``<of>|sum_squared_per_obs`` data for future reuse.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'oo'])
    to = of + '|variance_per_obs'

    @utm.timed_call('.compute')
    def compute() -> utt.Vector:
        sum_per_obs = \
            get_per_obs(adata, utc.sum_axis, of, inplace=inplace).proper
        sum_squared_per_obs = \
            get_per_obs(adata, utc.sum_squared_axis,
                        of, inplace=inplace).proper
        result = np.square(sum_per_obs).astype(float)
        result /= -adata.n_vars
        result += sum_squared_per_obs
        result /= adata.n_vars
        return result

    return _derive_1d_data(adata, per_of, of, 'o', to, compute, inplace)


@utm.timed_call()
@utd.expand_doc()
def get_variance_per_var(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> NamedVector:
    '''
    Return the variance of the values per-variable (gene) of some per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``<of>|variance_per_var`` per-variable (gene) data if it exists. Otherwise, compute it,
    and if ``inplace`` (default: {inplace}), store it for future reuse.

    If ``inplace``, also store the intermediate per-variable ``<of>|sum_per_var`` and
    ``<of>|sum_squared_per_var`` data for future reuse.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'vv'])
    to = of + '|variance_per_var'

    @utm.timed_call('.compute')
    def compute() -> utt.Vector:
        sum_per_var = \
            get_per_var(adata, utc.sum_axis, of, inplace=inplace).proper
        sum_squared_per_var = \
            get_per_var(adata, utc.sum_squared_axis,
                        of, inplace=inplace).proper
        result = np.square(sum_per_var).astype(float)
        result /= -adata.n_obs
        result += sum_squared_per_var
        result /= adata.n_obs
        return result

    return _derive_1d_data(adata, per_of, of, 'v', to, compute, inplace)


@utm.timed_call()
@utd.expand_doc()
def get_relative_variance_per_var(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> NamedVector:
    '''
    Return the log_2(variance/mean) of the values per-variable (gene) of some
    per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``<of>|relative_variance_per_var`` per-variable (cell) data if it exists. Otherwise,
    compute it, and if ``inplace`` (default: {inplace}), store it for future reuse.

    If ``inplace``, also store the intermediate per-variable ``<of>|variance``,
    ``<of>|mean_per_var``, ``<of>|sum_per_var`` and the ``<of>|sum_squared_per_var`` data for future
    reuse.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'vv'])
    to = of + '|relative_variance_per_var'

    @utm.timed_call('.compute')
    def compute() -> utt.Vector:
        variance_per_var = \
            get_variance_per_var(adata, of, inplace=inplace).proper
        mean_per_var = \
            get_mean_per_var(adata, of, inplace=inplace).proper
        zeros_mask = mean_per_var == 0

        result = np.reciprocal(mean_per_var, where=~zeros_mask)
        result *= variance_per_var
        result[zeros_mask] = 1
        np.log2(result, out=result)

        return result

    return _derive_1d_data(adata, per_of, of, 'v', to, compute, inplace)


@utm.timed_call()
@utd.expand_doc()
def get_normalized_variance_per_var(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    window_size: int = 100,
    inplace: bool = True,
) -> NamedVector:
    '''
    Return the (relative_variance - median-relative-variance-of-similar) of the values per-variable
    (gene) of some per-variable-per-observation data.

    The median-relative-variance-of-similar is the median relative variance of the ``window_size``
    (default: {window_size}) variables (genes) with the most similar mean to the one being
    normalized. In general, the relative variance tends to (but doesn't always) go up for
    higher-mean variables. By normalizing by the median relative variance of variables of a similar
    mean, we factor out this dependency on the mean to decide which variables (genes) carry more
    meaningful information (and are more suitable to be picked as "feature genes").

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    Use the ``<of>|normalized_variance_by_<window_size>_per_var`` per-variable (gene) data if it
    exists. Otherwise, compute it, and if ``inplace`` (default: {inplace}), store it for future
    reuse.

    If ``inplace``, also store the intermediate per-variable ``<of>|relative_variance_per_var``,
    ``<of>|variance_per_var``, ``<of>|mean_per_var``, ``<of>|sum_per_var`` and the
    ``<of>|sum_squared_per_var`` data for future reuse.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'vv'])
    to = '%s|normalized_variance_by_%s_per_var' % (of, window_size)

    @utm.timed_call('.compute')
    def compute() -> utt.Vector:
        relative_variance_per_var = \
            get_relative_variance_per_var(adata, of, inplace=inplace).proper
        mean_per_var = get_mean_per_var(adata, of, inplace=inplace).proper
        median_variance_per_var = utc.sliding_window_function(relative_variance_per_var,
                                                              function='median',
                                                              window_size=window_size,
                                                              order_by=mean_per_var)
        return relative_variance_per_var - median_variance_per_var

    return _derive_1d_data(adata, per_of, of, 'v', to, compute, inplace)


@utm.timed_call()
@utd.expand_doc()
def get_obs_obs_correlation(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    by: Optional[str] = None,
    inplace: bool = True,
) -> NamedMatrix:
    '''
    Compute correlation between observations (cells) of the ``adata``.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    If the data is per-observation-per-observation, then ``by`` (default: {by}) controls how to
    correlate it. A value of ``None`` indicates the data is symmetric, so the most efficient
    direction is chosen. Otherwise, the correlation is done by ``rows`` or by ``columns``
    as specified.

    Use the ``<of>|obs_obs_correlation`` or ``<of>|obs_obs_correlation_by_<by>``
    per-observation-per-observation (cell) data if it exists. Otherwise, compute it, and if
    ``inplace`` (default: {inplace}) store it for future reuse.

    Returns the matrix and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'oo'])
    if per_of == 'vo':
        assert by is None

    if by is None:
        to = of + '|obs_obs_correlation'
    else:
        to = of + '|obs_obs_correlation_by_' + by

    @utm.timed_call('.compute')
    def compute() -> utt.Matrix:
        assert isinstance(of, str)
        if per_of == 'vo':
            matrix = uta.get_proper_matrix(adata, of, by='obs')
            correlations = utc.corrcoef(matrix, rowvar=True)
        else:
            matrix = uta.get_proper_matrix(adata, of)
            correlations = utc.corrcoef(matrix, symmetric=True)

        return correlations

    return _derive_2d_data(adata, per_of, of, 'oo', to, compute, inplace)


@utm.timed_call()
@utd.expand_doc()
def get_var_var_correlation(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    by: Optional[str] = None,
    inplace: bool = True,
) -> NamedMatrix:
    '''
    Compute correlation between variables (genes) of the ``adata``.

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    If the data is per-variable-per-variable, then ``by`` (default: {by}) controls how to
    correlate it. A value of ``None`` indicates the data is symmetric, so the most efficient
    direction is chosen. Otherwise, the correlation is done by ``rows`` or by ``columns``
    as specified.

    Use the ``<of>|var_var_correlation`` or ``<of>|var_var_correlation_by_<by>``
    per-variable-per-variable (gene) data if it exists. Otherwise, compute it, and if ``inplace``
    (default: {inplace}), store it for future reuse.

    Returns the matrix and its name.
    '''
    per_of, of = _per_of(adata, of, ['vo', 'vv'])
    if per_of == 'vo':
        assert by is None

    if by is None:
        to = of + '|obs_obs_correlation'
    else:
        to = of + '|obs_obs_correlation_by_' + by

    @utm.timed_call('.compute')
    def compute() -> utt.Matrix:
        assert isinstance(of, str)
        if per_of == 'vo':
            matrix = uta.get_proper_matrix(adata, of, by='var')
            correlations = utc.corrcoef(matrix, rowvar=False)
        else:
            matrix = uta.get_proper_matrix(adata, of)
            correlations = utc.corrcoef(matrix, symmetric=True)

        return correlations.transpose()

    return _derive_2d_data(adata, per_of, of, 'vv', to, compute, inplace)


def _per_of(
    adata: AnnData,
    of: Optional[str],
    allowed: List[str]
) -> Tuple[str, str]:
    if of is None:
        of = adata.uns['__focus__']

    per_of = uta.which_data(adata, of)
    if per_of not in allowed:
        raise ValueError('the data: %s is not per: %s'
                         % (of, ', '.join(['"%s"' % expected
                                           for expected in allowed])))

    return per_of, of


def _derive_vo_data(
    adata: AnnData,
    of: str,
    to: str,
    slicing_mask: uta.SlicingMask,
    compute: Callable[[], utt.Matrix],
    inplace: bool,
    infocus: bool,
    by: Optional[str],
) -> NamedMatrix:
    if inplace or infocus:
        uta.safe_slicing_derived(base=of, derived=to,
                                 slicing_mask=slicing_mask)

    matrix = uta.get_vo_data(adata, to, compute=compute,
                             inplace=inplace, infocus=infocus, by=by)

    return NamedMatrix(name=to, shaped=matrix)


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
) -> NamedMatrix:
    assert per_of in ('vo', 'oo', 'vv')
    assert per_to in ('vo', 'oo', 'vv')

    if per_to == per_of:
        slicing_mask = uta.ALWAYS_SAFE
    elif per_to == 'oo':
        slicing_mask = uta.SAFE_WHEN_SLICING_OBS
    else:
        assert per_to == 'vv'
        slicing_mask = uta.SAFE_WHEN_SLICING_VAR

    return _derive_data(NamedMatrix, adata, of, per_to, to, slicing_mask,
                        compute, inplace, infocus, by)


def _derive_1d_data(
    adata: AnnData,
    per_of: str,
    of: str,
    per_to: str,
    to: str,
    compute: Callable[[], utt.Vector],
    inplace: bool,
) -> NamedVector:
    if per_to[0] == per_of[0] == per_of[1]:
        slicing_mask = uta.ALWAYS_SAFE
    elif per_to == 'v':
        slicing_mask = uta.SAFE_WHEN_SLICING_VAR
    else:
        assert per_to == 'o'
        slicing_mask = uta.SAFE_WHEN_SLICING_OBS

    return _derive_data(NamedVector, adata, of, per_to, to, slicing_mask,
                        compute, inplace)


def _derive_data(
    cls: Type[NS],
    adata: AnnData,
    of: str,
    per_to: str,
    to: str,
    slicing_mask: uta.SlicingMask,
    compute: Callable[[], utt.Shaped],
    inplace: bool,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NS:
    if inplace:
        uta.safe_slicing_derived(base=of, derived=to,
                                 slicing_mask=slicing_mask)

    data = uta.get_data(adata, to, per=per_to, compute=compute,
                        inplace=inplace, infocus=infocus, by=by)

    return cls(name=to, shaped=utt.Shaped.be(data))
