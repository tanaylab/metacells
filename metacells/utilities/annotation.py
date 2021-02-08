'''Proper)
Annotation
----------

This tries to deal safely with slicing data layers, observation, variable and unstructured
annotations, as well as two-dimensional variable and observation annotations.

.. todo::

    Discuss the different models with the maintainers of ``AnnData`` to see whether it would be
    possible to reconcile the two approaches.

Data Model
..........

The data model for the managed ``AnnData`` is slightly different from that of the raw ``AnnData``.
Specifically, the mapping is:

.. table:: Data Model Mapping
    :align: left
    :widths: auto

    ======================  =========================================================================
    Managed                 Raw
    ======================  =========================================================================
    metadata                unstructured annotations (``.uns``)
    per_var_per_obs matrix  union of ``.X`` and ``.layers``
    per_obs vector          observation annotations (``.obs``)
    per_var vector          variable annotations (``.var``)
    per_obs_per_obs matrix  observation pair matrix annotations (``.obsp``)
    per_var_per_var matrix  variable pair matrix annotations (``.varp``)
    per_obs_per_any matrix  observation multidimensional annotations (``.obsm``)
    per_var_per_any matrix  variable multidimensional annotations (``.varm``)
    ======================  =========================================================================

Mapping Code Complexity
.......................

The mapping between the models is mostly simply 1-1 which is trivial to implement. However, there
are three sources of (implementation) code complexity here:

1.  Per-variable-per-observation data.

    In contrast, raw ``AnnData`` contains a magic ``.X`` member which is distinct from the rest of
    the data layers. There is a strong assumption that all work will be done on ``X`` and the other
    layers are secondary at best.

    The managed ``AnnData`` therefore keeps a second special metadata property ``__x__`` which must
    :py:func:`metacells.utilities.annotation.setup`, and pretends that the value of ``X`` is just
    another layer. This gives rise to some edge cases (e.g., one can't delete the ``X`` layer).

2.  Layout-optimized data.

    For both dense and sparse formats, having the data in a layout that is friendly for the type of
    processing done on it makes for a big performance difference. For dense data this is "FORTRAN"
    vs. "C" layouts; for sparse data it is "CSR" vs. "CSC" format.

    Therefore, the managed ``AnnData`` allows for requesting a specific layout, and caches the
    results. Instead of monkey-patching the ``AnnData`` object, this is done by creating "hidden"
    layers with suffixed names (``...:__row_major__`` and ``...:__column_major__``).

    Changing the layout of a matrix using the builtin operations is very slow; therefore a parallel
    C++ extension is provided.

3.  Safe and efficient slicing.

    The managed ``AnnData`` provides a :py:func:`slice` operation that performs fast and safe
    slicing.

    In general data is assumed to be safe to slice (e.g., ``gene_ids`` can be freely sliced when we
    select a subset of the genes). However, derived data isn't always safe to slice; e.g.,
    ``UMIs|sum_per_obs`` is not safe to slice when we select a subset of the variables (it is safe
    to slice when we select a subset of the observations).

    We denote derived data by its name containing ``|``, chaining the source data and the operation
    applied to it (e.g., the above ``UMIs|sum_per_obs`` example). The code tracks, for each derived
    data, whether it is/not safe to slice it along each of the axis.

    In addition, since slicing layout-optimized data is much faster, the code ensures
    that, as much as possible, slicing is always applied to the proper layout form of the data.

The interaction between the above three features is the cause of most of the implementation
complexity. The upside is that application code becomes simpler and is more efficient.

Usage Differences
.................

The managed ``AnnData`` model prefers to consider all data as immutable. That is, if one wants to
compute the log of some values, the common approach in raw ``AnnData`` is to mutate ``X`` to hold
the log data, while in the managed ``AnnData`` the approach would be to create a new data layer, so
that in principle one never needs to mutate any data. This is (not 100%) enforced by
:py:func:`metacells.utilities.typing.freeze`-ing the data whenever possible.

Data Names
..........

The names of annotated data must be unique across all the kinds of data, otherwise
functions such as ``sc.pl.scatter`` get confused and may silently(!) use the wrong
data.

In general the auto-generated names are tediously long and detailed (e.g.,
``UMIs|sum_per_obs|log_2_normalization_1``), but ensure that they are properly unique. The caller is
responsible to ensure that the names of manually set data are also unique.
'''

import logging
from logging import Logger
from typing import (Any, Callable, Collection, Dict, List, MutableMapping,
                    NamedTuple, Optional, Sized, Tuple, Union)
from warnings import warn

import anndata as ad
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.utilities.computation as utc
import metacells.utilities.documentation as utd
import metacells.utilities.logging as utl
import metacells.utilities.timing as utm
import metacells.utilities.typing as utt

__all__ = [
    'setup',

    'slice',

    'SlicingMask',
    'ALWAYS_SAFE',
    'NEVER_SAFE',
    'SAFE_WHEN_SLICING_OBS',
    'SAFE_WHEN_SLICING_VAR',
    'safe_slicing_mask',
    'safe_slicing_derived',

    'set_name',
    'get_name',

    'set_m_data',
    'get_m_data',

    'set_o_data',
    'get_o_series',
    'get_o_dense',

    'set_v_data',
    'get_v_series',
    'get_v_dense',

    'set_oo_data',
    'get_oo_data',
    'get_oo_frame',
    'get_oo_proper',

    'set_vv_data',
    'get_vv_data',
    'get_vv_frame',
    'get_vv_proper',

    'set_oa_data',
    'get_oa_data',
    'get_oa_frame',
    'get_oa_proper',

    'set_va_data',
    'get_va_data',
    'get_va_frame',
    'get_va_proper',

    'set_vo_data',
    'get_vo_data',
    'get_vo_frame',
    'get_vo_proper',

    'log_use',
    'has_data',
]


LOG = logging.getLogger(__name__)


Annotations = Union[MutableMapping[Any, Any], pd.DataFrame]


def setup(
    adata: AnnData,
    *,
    name: Optional[str] = None,
    tmp: bool = False
) -> None:
    '''
    Set up the annotated ``adata`` for use by the ``metacells`` package.

    The optional ``name``, if specified, is attached to log messages about setting annotation data.

    If ``tmp`` is set, logging of modifications to the result will use the ``DEBUG`` logging level.
    By default, logging of modifications is done using the ``INFO`` logging level.

    This should be called after populating the ``X`` data for the first time (e.g., by importing the
    data). All the rest of the code in the package assumes this was done.

    .. note::

        This assumes it is safe to arbitrarily slice all the currently existing data.

    .. note::

        When using the layer utilities, do not directly read or write the value of ``X``. Instead
        use :py:func:`metacells.utilities.annotation.get_vo_data`.
    '''
    X = adata.X
    if not utt.frozen(X):  # type: ignore
        utt.freeze(X)  # type: ignore
    X = utt.BaseShaped.be(X)
    assert X is not None
    assert X.ndim == 2
    assert '__x__' not in adata.layers

    compressed = utt.CompressedMatrix.maybe(X)
    if compressed is not None:
        if not compressed.has_sorted_indices:
            warn('%s does not have sorted indices' % (name or 'adata'))
        if not compressed.has_canonical_format:
            warn('%s does not have canonical format' % (name or 'adata'))

    if tmp:
        adata.uns['__tmp__'] = True
    if name is not None:
        adata.uns['__name__'] = name
        LOG.log(utl.get_log_level(adata),
                '  created %s shape %s', name, adata.shape)


class SlicingMask(NamedTuple):
    '''
    A mask for safe operations when slicing some ``AnnData``.
    '''

    #: Whether the operation is safe when slicing observations (cells).
    obs: bool

    #: Whether the operation is safe when slicing variables (genes).
    vars: bool

    def __and__(self, other: 'SlicingMask') -> 'SlicingMask':
        return SlicingMask(obs=self.obs and other.obs,
                           vars=self.vars and other.vars)

    def __or__(self, other: 'SlicingMask') -> 'SlicingMask':
        return SlicingMask(obs=self.obs or other.obs,
                           vars=self.vars or other.vars)


#: A mask for operations which are safe regardless of what is sliced.
ALWAYS_SAFE = SlicingMask(obs=True, vars=True)

#: A mask for operations which are not safe regardless of what is sliced.
NEVER_SAFE = SlicingMask(obs=False, vars=False)

#: A mask for operations which are only safe when slicing variables (genes),
#: but not when slicing observations (cells).
SAFE_WHEN_SLICING_VAR = SlicingMask(obs=False, vars=True)

#: A mask for operations which are only safe when slicing observations (cells),
#: but not when slicing variables (genes).
SAFE_WHEN_SLICING_OBS = SlicingMask(obs=True, vars=False)

SAFE_SLICING: Dict[str, SlicingMask] = {}


def safe_slicing_mask(
    name: str,
) -> SlicingMask:
    '''
    Return a :py:const:`SlicingMask` specifying when it is safe to slice the ``name``-d data.
    '''
    if '|' not in name:
        return ALWAYS_SAFE

    slicing_mask = SAFE_SLICING.get(name)
    if slicing_mask is None:
        slicing_mask = SAFE_SLICING.get(name)
        if slicing_mask is None:
            unknown_sliced_data = \
                'Slicing an unknown data: %s; ' \
                'assuming it is not safe to slice' \
                % name
            warn(unknown_sliced_data)
            slicing_mask = SAFE_SLICING[name] = NEVER_SAFE
    return slicing_mask


def safe_slicing_derived(
    *,
    base: Union[str, List[str]],
    derived: str,
    slicing_mask: SlicingMask,
) -> None:
    '''
    Specify how to slice some ``derived`` data which is computed from some ``base_data``.

    If, when slicing, any of the bases data is not preserved, then the derived data would not be
    either. Otherwise, the derived will be preserved based on the flags given here.
    '''
    if isinstance(base, str):
        if '|' in base:
            slicing_mask = slicing_mask & SAFE_SLICING[base]
    else:
        for name in base:
            if '|' in name:
                slicing_mask = slicing_mask & SAFE_SLICING[name]

    SAFE_SLICING[derived] = slicing_mask


@utm.timed_call()
def slice(  # pylint: disable=redefined-builtin,too-many-branches,too-many-statements
    adata: AnnData,
    *,
    obs: Optional[Union[Sized, utt.Vector]] = None,
    vars: Optional[Union[Sized, utt.Vector]] = None,
    name: Optional[str] = None,
    track_obs: Optional[str] = None,
    track_var: Optional[str] = None,
    tmp: bool = False,
) -> AnnData:
    '''
    Return new annotated data which includes a subset of the full ``adata``.

    If ``obs`` and/or ``vars`` are specified, they should include either a boolean
    mask or a collection of indices to include in the data slice. In the case of an indices array,
    it is assumed the indices are unique and sorted, that is that their effect is similar to a mask.

    If ``track_obs`` and/or ``track_var`` are specified, the result slice will include a
    per-observation and/or per-variable annotation containing the indices of the sliced elements in
    the original full data.

    In general, data might become invalid when slicing (e.g., ``UMIs|sum_per_obs`` data is
    invalidated when slicing some of the variables). Therefore, such data will be removed from the
    result.

    If ``name`` is not specified, the data will be unnamed. Otherwise, if it starts with a ``.``, it
    will be appended to the current name (if any). Otherwise, ``name`` is the new name.

    If ``tmp`` is set, logging of modifications to the result will use the ``DEBUG`` logging level.
    By default, logging of modifications is done using the ``INFO`` logging level.
    '''

    assert '__x__' not in adata.layers

    is_same_obs: Optional[bool] = None
    if obs is None:
        obs = range(adata.n_obs)
        is_same_obs = True
    else:
        assert 0 < len(obs) <= adata.n_obs
        if len(obs) < adata.n_obs:
            is_same_obs = False

        shaped = utt.BaseShaped.maybe(obs)
        if shaped is not None:
            obs = utt.to_dense_vector(shaped)  # type: ignore
            if obs.dtype == 'bool':
                assert obs.size == adata.n_obs
                assert np.any(obs)
                is_same_obs = np.all(obs)

    is_same_vars: Optional[bool] = None
    if vars is None:
        vars = range(adata.n_vars)
        is_same_vars = True
    else:
        assert 0 < len(vars) <= adata.n_vars
        if len(vars) < adata.n_vars:
            is_same_vars = False

        shaped = utt.BaseShaped.maybe(vars)
        if shaped is not None:
            vars = utt.to_dense_vector(shaped)  # type: ignore
            if vars.dtype == 'bool':
                assert vars.size == adata.n_vars
                assert np.any(vars)
                is_same_vars = np.all(vars)

    if is_same_obs and is_same_vars:
        with utm.timed_step('adata.copy'):
            bdata = adata.copy()
        assert not hasattr(bdata, '__derived__')
        if hasattr(adata, '__derived__'):
            setattr(bdata, '__derived__', getattr(adata, '__derived__'))

    else:
        with utm.timed_step('adata.slice'):
            bdata = adata[obs, vars].copy()

        assert not hasattr(bdata, '__derived__')
        if hasattr(adata, '__derived__'):
            if is_same_obs is None:
                is_same_obs = bdata.n_obs == adata.n_obs \
                    and np.all(bdata.obs_names == adata.obs_names)
            if is_same_obs and is_same_vars is None:
                is_same_vars = bdata.n_obs == adata.n_vars \
                    and np.all(bdata.var_names == adata.var_names)
            if is_same_obs and is_same_vars:
                setattr(bdata, '__derived__', getattr(adata, '__derived__'))

    if tmp:
        bdata.uns['__tmp__'] = True

    if name is not None:
        set_name(bdata, name)
        LOG.log(utl.get_log_level(bdata),
                '  sliced %s shape %s', name, bdata.shape)

    if track_obs is not None:
        set_o_data(bdata, track_obs, np.arange(adata.n_obs)[obs])

    if track_var is not None:
        set_v_data(bdata, track_var, np.arange(adata.n_vars)[vars])

    return bdata


def log_use(
    logger: Logger,
    adata: AnnData,
    what: Optional[Union[str, utt.Shaped]],
    *,
    name: str,
    per: str,
    default: str = 'None',
    indent: str = '  ',
) -> None:
    '''
    Log using some annotation data.
    '''
    assert per in ('m', 'v', 'o', 'vv', 'oo', 'vo', 'va', 'oa')
    if isinstance(what, str):
        data_name = get_name(adata)
        if data_name is None:
            logger.debug('%s%s: %s:%s',
                         indent, name, MEMBER_OF_PER[per], what)
        else:
            logger.debug('%s%s: %s:%s:%s',
                         indent, name, data_name, MEMBER_OF_PER[per], what)
    elif what is None:
        logger.debug('%s%s: %s', indent, name, default)
    else:
        logger.debug('%s%s: <data>', indent, name)


@utd.expand_doc()
def get_data(  # pylint: disable=too-many-return-statements
    adata: AnnData,
    name: Optional[str] = None,
    *,
    per: Optional[str] = None,
    layout: Optional[str] = None,
) -> Any:
    '''
    Lookup any data by its ``name`` (default: {name}).

    If no ``name`` is specified, return the focus.

    If the data is 2-dimensional, and ``layout`` (default: {layout}) is specified, it forces the
    layout of the returned data.
    '''
    assert layout is None or layout in utt.LAYOUT_OF_AXIS

    if name is None:
        name = '__x__'

    if per == 'vo' \
            or (per is None
                and (name == '__x__' or name in adata.layers)):
        return get_vo_data(adata, name, layout=layout)

    if per == 'oo' or (per is None and name in adata.obsp):
        return get_oo_data(adata, name, layout=layout)

    if per == 'vv' or (per is None and name in adata.varp):
        return get_vv_data(adata, name, layout=layout)

    if per == 'oa' or (per is None and name in adata.obsm):
        return get_oa_data(adata, name, layout=layout)

    if per == 'va' or (per is None and name in adata.varm):
        return get_va_data(adata, name, layout=layout)

    assert layout is None

    if per == 'm' or (per is None and name in adata.uns):
        return get_m_data(adata, name)

    if per == 'o' or (per is None and name in adata.obs):
        return _get_shaped_data(adata, 'o', adata.obs, shape=(adata.n_obs,), what=name)

    if per == 'v' or (per is None and name in adata.var):
        return _get_shaped_data(adata, 'v', adata.var, shape=(adata.n_vars,), what=name)

    raise _unknown_data(adata, name, per)


def has_data(
    adata: AnnData,
    name: str,
    layout: Optional[str] = None,
) -> bool:
    '''
    Test whether we have the specified data.

    If the data is per-variable-per-observation, and ``layout`` is specified (one of ``row_major``
    and ``column_major``), it returns whether the specific data layout is available without
    having to compute or relayout existing data.
    '''
    assert layout is None or layout in utt.LAYOUT_OF_AXIS

    for annotations in (adata.layers, adata.obsp, adata.varp, adata.obsm, adata.varm):
        if (id(annotations) == id(adata.layers) and name == '__x__') \
                or name in annotations:
            if layout is None:
                return True
            name = '%s:__%s__' % (name, layout)
            if name in annotations:
                return True
            return utt.matrix_layout(get_data(adata, name)) == layout

    if name in adata.uns or name in adata.obs or name in adata.var:
        assert layout is None
        return True

    return False


@utd.expand_doc()
def data_per(
    adata: AnnData,
    name: str,
    must_exist: bool = True,
) -> Optional[str]:
    '''
    Return the kind of data identified by the ``name``.

    Possible return values are ``vo``, ``vv``, ``oo``, ``v``, ``o`` and ``m``.

    If the data is missing, and ``must_exist`` (default: {must_exist}), raise ``KeyError``.
    Otherwise, return ``None``.
    '''

    if name == '__x__':
        return 'vo'

    for per, annotation in (('vo', adata.layers),
                            ('m', adata.uns),
                            ('o', adata.obs),
                            ('v', adata.var),
                            ('oo', adata.obsp),
                            ('vv', adata.varp),
                            ('oa', adata.obsm),
                            ('va', adata.varm)):
        if name in annotation:
            return per

    if must_exist:
        raise _unknown_data(adata, name)

    return None


def get_m_data(adata: AnnData, name: str) -> Any:
    '''
    Lookup metadata (unstructured annotation) in ``adata`` by its ``name``.
    '''
    data = adata.uns.get(name)
    if data is None:
        raise _unknown_data(adata, name, 'm')
    return data


def get_o_series(
    adata: AnnData,
    what: Union[str, utt.Vector],
) -> utt.PandasSeries:
    '''
    Lookup per-observation (cell) data in ``adata`` by its ``name`` as a Pandas series.

    If ``what`` is a string, it is the name of a per-observation annotation to fetch.
    Otherwise, it should be some vector of data of the appropriate size.
    '''
    data = _get_shaped_data(adata, 'o', adata.obs,
                            shape=(adata.n_obs,), what=what)
    series = utt.PandasSeries.maybe(data)
    if series is None:
        series = pd.Series(utt.to_dense_vector(data), index=adata.obs_names)
    return series


def get_o_dense(
    adata: AnnData,
    what: Union[str, utt.Vector],
) -> utt.DenseVector:
    '''
    Lookup per-observation (cell) data in ``adata`` by its ``name`` as a Numpy array.

    If ``what`` is a string, it is the name of a per-observation annotation to fetch.
    Otherwise, it should be some vector of data of the appropriate size.
    '''
    data = _get_shaped_data(adata, 'o', adata.obs,
                            shape=(adata.n_obs,), what=what)
    return utt.to_dense_vector(data)


def get_v_series(
    adata: AnnData,
    what: Union[str, utt.Vector]
) -> utt.PandasSeries:
    '''
    Lookup per-variable (gene) data in ``adata`` by its ``name`` as a pandas series.

    If ``what`` is a string, it is the name of a per-variable annotation to fetch.
    Otherwise, it should be some vector of data of the appropriate size.
    '''
    data = _get_shaped_data(adata, 'v', adata.var,
                            shape=(adata.n_vars,), what=what)
    series = utt.PandasSeries.maybe(data)
    if series is None:
        series = pd.Series(utt.to_dense_vector(data), index=adata.var_names)
    return series


def get_v_dense(
    adata: AnnData,
    what: Union[str, utt.Vector]
) -> utt.DenseVector:
    '''
    Lookup per-variable (gene) data in ``adata`` by its ``name`` as a numpy array.

    If ``what`` is a string, it is the name of a per-variable annotation to fetch.
    Otherwise, it should be some vector of data of the appropriate size.
    '''
    data = _get_shaped_data(adata, 'v', adata.var,
                            shape=(adata.n_vars,), what=what)
    return utt.to_dense_vector(data)


@utd.expand_doc()
def get_oo_data(
    adata: AnnData,
    what: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
) -> utt.Matrix:
    '''
    Lookup per-observation-per-observation (cell) data in ``adata`` by its ``name``.

    If ``what`` is a string, it is the name of a per-variable annotation to fetch.
    Otherwise, it should be some matrix of data of the appropriate size.

    If ``layout`` (default: {layout}) is specified, it must be one of ``row_major`` or
    ``column_major`` (genes). This returns the data in a layout optimized for by-observation
    (row-major / csr) or by-variable (column-major / csc). This is cached in an additional "hidden"
    annotation whose name is suffixed (e.g. ``...:__row_major__``).
    '''
    return _get_layout_data(adata, 'oo', adata.obsp,
                            shape=(adata.n_obs, adata.n_obs),
                            what=what, layout=layout)


def get_oo_frame(
    adata: AnnData,
    what: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
) -> utt.PandasFrame:
    '''
    Same as :py:func:`get_oo_data` but returns a pandas data frame.
    '''
    data = get_oo_data(adata, what, layout=layout)
    frame = utt.PandasFrame.maybe(data)
    if frame is None:
        frame = pd.DataFrame(utt.to_proper_matrix(data),
                             index=adata.obs_names, columns=adata.obs_names)
    return frame


def get_oo_proper(
    adata: AnnData,
    what: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
) -> utt.ProperMatrix:
    '''
    Same as :py:func:`get_oo_data` but returns a numpy
    :py:const:`metacells.utilities.typing.ProperMatrix`.
    '''
    return utt.to_proper_matrix(get_oo_data(adata, what, layout=layout),
                                default_layout=layout or 'row_major')


@utd.expand_doc()
def get_vv_data(
    adata: AnnData,
    what: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
) -> utt.Matrix:
    '''
    Lookup per-variable-per-variable (gene) data in ``adata``.

    If ``what`` is a string, it is the name of a per-variable annotation to fetch.
    Otherwise, it should be some matrix of data of the appropriate size.

    If ``layout`` (default: {layout}) is specified, it must be one of ``row_major`` or
    ``column_major`` (genes). This returns the data in a layout optimized for by-observation
    (row-major / csr) or by-variable (column-major / csc). This is cached in an additional "hidden"
    annotation whose name is suffixed (e.g. ``...:__row_major__``).
    '''
    return _get_layout_data(adata, 'vv', adata.varp,
                            shape=(adata.n_vars, adata.n_vars),
                            what=what, layout=layout)


def get_vv_frame(
    adata: AnnData,
    what: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
) -> utt.PandasFrame:
    '''
    Same as :py:func:`get_vv_data` but returns a pandas data frame.
    '''
    data = get_vv_data(adata, what, layout=layout)
    frame = utt.PandasFrame.maybe(data)
    if frame is None:
        frame = pd.DataFrame(utt.to_proper_matrix(data),
                             index=adata.var_names, columns=adata.var_names)
    return frame


def get_vv_proper(
    adata: AnnData,
    what: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
) -> utt.ProperMatrix:
    '''
    Same as :py:func:`get_vv_data` but returns a numpy
    :py:const:`metacells.utilities.typing.ProperMatrix`.
    '''
    return utt.to_proper_matrix(get_vv_data(adata, what, layout=layout),
                                default_layout=layout or 'row_major')


@utd.expand_doc()
def get_oa_data(
    adata: AnnData,
    what: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
) -> utt.Matrix:
    '''
    Lookup per-observation-per-any (cell) data in ``adata`` by its ``name``.

    If ``what`` is a string, it is the name of a per-variable annotation to fetch.
    Otherwise, it should be some matrix of data of the appropriate size.

    If ``layout`` (default: {layout}) is specified, it must be one of ``row_major`` or
    ``column_major`` (genes). This returns the data in a layout optimized for by-observation
    (row-major / csr) or by-variable (column-major / csc). This is cached in an additional "hidden"
    annotation whose name is suffixed (e.g. ``...:__row_major__``).
    '''
    return _get_layout_data(adata, 'oa', adata.obsm,
                            shape=(adata.n_obs, 0),
                            what=what, layout=layout)


def get_oa_frame(
    adata: AnnData,
    what: Union[str, utt.Matrix],
    *,
    columns: Optional[Collection],
    layout: Optional[str] = None,
) -> utt.PandasFrame:
    '''
    Same as :py:func:`get_oa_data` but returns a pandas data frame.
    '''
    data = get_oa_data(adata, what, layout=layout)
    frame = utt.PandasFrame.maybe(data)
    if frame is None:
        frame = pd.DataFrame(utt.to_proper_matrix(data), index=adata.obs_names)
    if columns is not None:
        frame.columns = columns
    return frame


def get_oa_proper(
    adata: AnnData,
    what: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
) -> utt.ProperMatrix:
    '''
    Same as :py:func:`get_oa_data` but returns a numpy
    :py:const:`metacells.utilities.typing.ProperMatrix`.
    '''
    return utt.to_proper_matrix(get_oa_data(adata, what, layout=layout),
                                default_layout=layout or 'row_major')


@utd.expand_doc()
def get_va_data(
    adata: AnnData,
    what: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
) -> utt.Matrix:
    '''
    Lookup per-variable-per-variable (gene) data in ``adata``.

    If ``what`` is a string, it is the name of a per-variable annotation to fetch.
    Otherwise, it should be some matrix of data of the appropriate size.

    If ``layout`` (default: {layout}) is specified, it must be one of ``row_major`` or
    ``column_major`` (genes). This returns the data in a layout optimized for by-observation
    (row-major / csr) or by-variable (column-major / csc). This is cached in an additional "hidden"
    annotation whose name is suffixed (e.g. ``...:__row_major__``).
    '''
    return _get_layout_data(adata, 'va', adata.varm,
                            shape=(adata.n_vars, 0),
                            what=what, layout=layout)


def get_va_frame(
    adata: AnnData,
    what: Union[str, utt.Matrix],
    *,
    columns: Optional[Collection],
    layout: Optional[str] = None,
) -> utt.PandasFrame:
    '''
    Same as :py:func:`get_va_data` but returns a pandas data frame.
    '''
    data = get_va_data(adata, what, layout=layout)
    frame = utt.PandasFrame.maybe(data)
    if frame is None:
        frame = pd.DataFrame(utt.to_proper_matrix(data), index=adata.var_names)
    if columns is not None:
        frame.columns = columns
    return frame


def get_va_proper(
    adata: AnnData,
    what: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
) -> utt.ProperMatrix:
    '''
    Same as :py:func:`get_va_data` but returns a numpy
    :py:const:`metacells.utilities.typing.ProperMatrix`.
    '''
    return utt.to_proper_matrix(get_va_data(adata, what, layout=layout),
                                default_layout=layout or 'row_major')


@utd.expand_doc()
def get_vo_data(
    adata: AnnData,
    what: Union[str, utt.Matrix, None] = None,
    *,
    layout: Optional[str] = None,
) -> utt.Matrix:
    '''
    Lookup a per-variable-per-observation matrix (data layer) in ``adata`` by its ``name``.

    If ``what`` is not specified, uses the value of the ``X`` member.
    If ``what`` is a string, it is the name of a per-variable annotation to fetch.
    Otherwise, it should be some matrix of data of the appropriate size.

    If ``layout`` (default: {layout}) is specified, it must be one of ``row_major`` or
    ``column_major`` (genes). This returns the data in a layout optimized for by-observation
    (row-major / csr) or by-variable (column-major / csc). This is cached in an additional "hidden"
    layer whose name is suffixed (e.g. ``...:__row_major__``).

    Returns the result data.
    '''
    if what is None:
        what = '__x__'

    data = _get_layout_data(adata, 'vo', adata.layers,
                            shape=(adata.n_obs, adata.n_vars),
                            what=what, layout=layout)

    return data


def get_vo_frame(
    adata: AnnData,
    what: Union[str, utt.Matrix, None] = None,
    *,
    layout: Optional[str] = None,
) -> utt.PandasFrame:
    '''
    Same as :py:func:`get_vo_data` but returns a pandas data frame.
    '''
    data = get_vo_data(adata, what, layout=layout)
    frame = utt.PandasFrame.maybe(data)
    if frame is None:
        frame = pd.DataFrame(utt.to_proper_matrix(data),
                             index=adata.obs_names, columns=adata.var_names)
    return frame


def get_vo_proper(
    adata: AnnData,
    what: Union[str, utt.Matrix, None] = None,
    *,
    layout: Optional[str] = None,
) -> utt.ProperMatrix:
    '''
    Same as :py:func:`get_vo_data` but returns a numpy
    :py:const:`metacells.utilities.typing.ProperMatrix`.
    '''
    return utt.to_proper_matrix(get_vo_data(adata, what, layout=layout),
                                default_layout=layout or 'row_major')


def _get_layout_data(
    adata: AnnData,
    per: str,
    annotations: Annotations,
    *,
    shape: Tuple[int, ...],
    what: Union[str, utt.Matrix],
    layout: Optional[str],
) -> Any:
    data = _get_shaped_data(adata, per, annotations, shape=shape, what=what)
    if not isinstance(what, str):
        if not utt.is_layout(data, layout):
            data = utc.to_layout(what, layout=layout)
        return data

    if utt.is_layout(data, layout):
        return data

    assert layout in utt.LAYOUT_OF_AXIS
    layout_name = '%s:%s' % (what, layout)

    if not hasattr(adata, '__derived__'):
        derived: Dict[str, utt.ProperMatrix] = dict()
        setattr(adata, '__derived__', derived)
    else:
        derived = getattr(adata, '__derived__')

    layout_data = derived.get(layout_name)
    if layout_data is not None:
        assert layout_data.shape == shape
        assert utt.is_layout(layout_data, layout)
    else:
        layout_data = utc.to_layout(data, layout=layout)
        derived[layout_name] = layout_data

    if not utt.frozen(layout_data):
        utt.freeze(layout_data)

    return layout_data


def _get_shaped_data(
    adata: AnnData,
    per: str,
    annotations: Annotations,
    *,
    shape: Tuple[int, ...],
    what: Union[str, utt.Shaped],
) -> Any:
    if isinstance(what, str):
        if per == 'vo' and what == '__x__':
            data = _fix_data(adata.X)
        elif (isinstance(annotations, pd.DataFrame) and what in annotations.columns) \
                or what in annotations:
            data = _fix_data(annotations[what])
        else:
            raise _unknown_data(adata, what, per)

        if not utt.frozen(data):
            utt.freeze(data)

    else:
        assert what.shape == shape
        if len(shape) == 1:
            data = utt.to_dense_vector(what)
        else:
            data = utt.to_proper_matrix(what)  # type: ignore

    assert data.shape == shape
    return data


def _fix_data(data: Any) -> Any:
    if isinstance(data, ad._core.sparse_dataset.SparseDataset):  # pylint: disable=protected-access
        return data.value
    return data


@utd.expand_doc()
def del_data(
    adata: AnnData,
    name: str,
    *,
    per: Optional[str] = None,
    layout: Optional[str] = None,
    must_exist: bool = False,
) -> None:
    '''
    Delete some data from the ``adata``.

    If ``layout`` (default: {layout}) is specified, it must be one of ``row_major`` (cells) or
    ``column_major`` (genes). This will only delete the cached layout-specific data, if any. If
    ``layout`` is not specified, both the data and any cached layout-specific data will be deleted.

    If ``must_exist`` (default: {must_exist}), will ``raise`` if the data does not currently exist.

    .. note::

        You can't delete the ``__x__`` (that is, the ``X`` pseudo-layer). Deleting the focus
        (default) data changes the focus to become whatever is in ``X``.

    .. todo::

        It would be better to replace :py:func:`del_data` with intercepting ``del`` of the
        ``AnnData`` fields, but we do not own them.
    '''
    if name.endswith(':__row_major__'):
        assert layout is None
        layout = 'row_major'
        name = name[:-14]
    elif name.endswith(':__column_major__'):
        assert layout is None
        layout = 'column_major'
        name = name[:-17]

    if per is None:
        per = data_per(adata, name)
        assert per is not None

    _log_del_data(adata, per, name, layout)

    if per == 'vo':
        assert name != '__x__'

    annotation = _annotation_per(adata, per)

    if layout is not None:
        layout_name = '%s:__%s__' % (name, layout)
        if layout_name in adata.layers:
            del annotation[layout_name]
        elif must_exist:
            assert layout_name in annotation
        return

    if must_exist:
        assert name in annotation

    for suffix in ('', ':__row_major__', ':__column_major__'):
        suffixed_name = name + suffix
        if suffixed_name in annotation:
            del annotation[suffixed_name]


def _annotation_per(  # pylint: disable=too-many-return-statements
    adata: AnnData, per: str
) -> Annotations:
    if per == 'm':
        return adata.uns
    if per == 'vo':
        return adata.layers
    if per == 'v':
        return adata.var
    if per == 'o':
        return adata.obs
    if per == 'vv':
        return adata.varp
    if per == 'oo':
        return adata.obsp
    if per == 'va':
        return adata.varm
    if per == 'oa':
        return adata.obsm
    raise ValueError('unknown per: %s' % per)


def set_name(adata: AnnData, name: str) -> None:
    '''
    Set the ``name`` of the data (for log messages).

    If the name starts with ``.`` it is appended to the current name.
    '''
    if name[0] == '.':
        old_name = get_name(adata)
        if old_name is None:
            name = name[1:]
        else:
            name = old_name + name
    adata.uns['__name__'] = name


def get_name(adata: AnnData, default: Optional[str] = None) -> Optional[str]:
    '''
    Return the name of the data (for log messages), if any.

    If no name was set, returns the ``default``.
    '''
    return adata.uns.get('__name__', default)


def set_m_data(
    adata: AnnData,
    name: str,
    data: Any,
    log_value: Optional[Callable[[Any], Optional[str]]] = None,
) -> Any:
    '''
    Set unstructured meta-data.

    If ``log_value`` is specified, its results is used when logging the operation.
    '''
    _log_set_data(adata, 'm', name, data, log_value=log_value)

    adata.uns[name] = data


def set_o_data(
    adata: AnnData,
    name: str,
    data: utt.DenseVector,
    log_value: Optional[Callable[[Any], Optional[str]]] = None,
) -> Any:
    '''
    Set per-observation (cell) meta-data.

    If ``log_value`` is specified, its results is used when logging the operation.
    '''
    _log_set_data(adata, 'o', name, data, log_value=log_value)

    if not isinstance(data, list):
        assert utt.is_canonical(data)
        if not utt.frozen(data):
            utt.freeze(data)

    adata.obs[name] = data


def set_v_data(
    adata: AnnData,
    name: str,
    data: utt.DenseVector,
    log_value: Optional[Callable[[Any], Optional[str]]] = None
) -> Any:
    '''
    Set per-variable (gene) meta-data.

    If ``log_value`` is specified, its results is used when logging the operation.
    '''
    _log_set_data(adata, 'v', name, data, log_value=log_value)

    assert utt.is_canonical(data)
    if not utt.frozen(data):
        utt.freeze(data)

    adata.var[name] = data


def set_oo_data(
    adata: AnnData,
    name: str,
    data: utt.ProperMatrix,
    log_value: Optional[Callable[[Any], Optional[str]]] = None
) -> Any:
    '''
    Set per-observation-per-observation (cell) meta-data.

    If ``log_value`` is specified, its results is used when logging the operation.
    '''
    _log_set_data(adata, 'oo', name, data, log_value=log_value)

    assert utt.is_canonical(data)
    if not utt.frozen(data):
        utt.freeze(data)

    adata.obsp[name] = data


def set_vv_data(
    adata: AnnData,
    name: str,
    data: utt.ProperMatrix,
    log_value: Optional[Callable[[Any], Optional[str]]] = None
) -> Any:
    '''
    Set per-variable-per-variable (gene) meta-data.

    If ``log_value`` is specified, its results is used when logging the operation.
    '''
    _log_set_data(adata, 'vv', name, data, log_value=log_value)

    assert utt.is_canonical(data)
    if not utt.frozen(data):
        utt.freeze(data)

    adata.varp[name] = data


def set_oa_data(
    adata: AnnData,
    name: str,
    data: utt.ProperMatrix,
    log_value: Optional[Callable[[Any], Optional[str]]] = None
) -> Any:
    '''
    Set per-observation-per-any (cell) meta-data.

    If ``log_value`` is specified, its results is used when logging the operation.
    '''
    _log_set_data(adata, 'oa', name, data, log_value=log_value)

    assert utt.is_canonical(data)
    if not utt.frozen(data):
        utt.freeze(data)

    adata.obsm[name] = data


def set_va_data(
    adata: AnnData,
    name: str,
    data: utt.ProperMatrix,
    log_value: Optional[Callable[[Any], Optional[str]]] = None
) -> Any:
    '''
    Set per-variable-per-any (gene) meta-data.

    If ``log_value`` is specified, its results is used when logging the operation.
    '''
    _log_set_data(adata, 'va', name, data, log_value=log_value)

    assert utt.is_canonical(data)
    if not utt.frozen(data):
        utt.freeze(data)

    adata.varm[name] = data


def set_vo_data(
    adata: AnnData,
    name: str,
    data: utt.ProperMatrix,
    *,
    log_value: Optional[Callable[[Any], Optional[str]]] = None,
) -> Any:
    '''
    Set per-variable-per-observation (per-gene-per-cell) meta-data.
    '''
    if name == '__x__':
        _log_set_data(adata, 'x', name, data, log_value=log_value)
    else:
        _log_set_data(adata, 'vo', name, data, log_value=log_value)

    assert utt.is_canonical(data)
    if not utt.frozen(data):
        utt.freeze(data)

    if name == '__x__':
        adata.X = data
    else:
        adata.layers[name] = data


MEMBER_OF_PER = \
    dict(m='uns',
         o='obs', v='var',
         oo='obsp', vv='varp',
         oa='obsm', va='varm',
         vo='layers', x='X')


def _log_set_data(  # pylint: disable=too-many-branches
    adata: AnnData,
    per: str,
    name: str,
    value: Any = None,
    log_value: Optional[Callable[[Any], Optional[str]]] = None,
) -> None:
    if '|' in name:
        level = logging.DEBUG
    else:
        level = utl.get_log_level(adata)

    if not LOG.isEnabledFor(level):
        return

    texts = ['  ']

    try:
        data_name = get_name(adata)

        texts.append('setting ')
        if data_name is not None:
            texts.append(data_name)
            texts.append(':')
        texts.append(MEMBER_OF_PER[per])
        texts.append(':')
        texts.append(name)

        if log_value is not None:
            value = log_value(value)
            assert isinstance(value, str)

        if value is None:
            return

        if per == 'vo' and isinstance(value, str):
            level = logging.DEBUG
            texts[1] = 'caching '
            texts.append(' ')
            texts.append(value)
            texts.append(' layout')
            return

        if len(per) > 1:
            return

        if isinstance(value, (str, int, float)):
            texts.append(' to ')
            texts.append(str(value))
            return

        if isinstance(value, (pd.Series, np.ndarray)) and value.dtype == 'bool':
            texts.append(' to a mask of ')
            texts.append(utl.mask_description(value))
            return

        if per == 'm' and hasattr(value, 'ndim'):
            if value.ndim == 2:
                texts.append(' to a matrix of type ')
                texts.append(str(value.dtype))
                texts.append(' shape ')
                texts.append(str(value.shape))
            elif value.ndim == 1:
                texts.append(' to a vector of type ')
                texts.append(str(value.dtype))
                texts.append(' size ')
                texts.append(str(value.size))

#       if hasattr(value, 'ndim'):
#           texts.append(' checksum ')
#           texts.append(str(utt.shaped_checksum(value)))

    finally:
        text = ''.join(texts)
        if text != '  ':
            LOG.log(level, text)


def _log_del_data(
    adata: AnnData,
    per: str,
    name: str,
    layout: Optional[str],
) -> None:
    level = logging.DEBUG

#   if '|' in name:
#       level = logging.DEBUG
#   else:
#       level = utl.get_log_level(adata)

    if not LOG.isEnabledFor(level):
        return

    texts = ['  ']

    if layout is None:
        texts.append('deleting ')
    else:
        texts.append('uncaching ')

    data_name = get_name(adata)
    if data_name is not None:
        texts.append(data_name)
        texts.append('.')
    texts.append(MEMBER_OF_PER[per])
    texts.append('.')
    texts.append(name)

    if layout is not None:
        texts.append(' ')
        texts.append(layout)
        texts.append(' layout')

    LOG.log(level, ''.join(texts))


def _unknown_data(adata: AnnData, name: str, per: Optional[str] = None) -> KeyError:
    texts = ['unknown']

    if per is not None:
        texts.append(' ')
        texts.append(per)

    texts.append(' data')

    data_name = get_name(adata)
    if data_name is not None:
        texts.append(': ')
        texts.append(data_name)

    texts.append(' name: ')
    texts.append(name)
    return KeyError(''.join(texts))
