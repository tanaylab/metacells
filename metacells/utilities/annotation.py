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

    In the managed ``AnnData``, all the layers are equal. A special metadata property ``__focus__``
    provides the default layer name for all layer functions. Similarly, operations that output a
    layer allow specifying that the result will be ``infocus``, and the :py:func:`focus_on` function
    allows for code to safely shift the focus for specific code regions.

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

The managed ``AnnData`` provides standard functions that compute and cache new data from existing
data, using arbitrary compute functions. See the :py:mod:`metacells.preprocessing.common` module for
these functions.

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
from contextlib import contextmanager
from logging import Logger
from typing import (Any, Callable, Collection, Dict, Iterable, Iterator, List,
                    MutableMapping, NamedTuple, Optional, Set, Sized, Tuple,
                    Union)
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

    'get_data',
    'get_proper_matrix',
    'get_dense_vector',
    'get_vector_parameter_data',
    'has_data',
    'data_per',

    'get_m_data',
    'get_o_data',
    'get_o_dense',
    'get_v_data',
    'get_v_dense',
    'get_oo_data',
    'get_oo_proper',
    'get_vv_data',
    'get_vv_proper',
    'get_oa_data',
    'get_oa_proper',
    'get_va_data',
    'get_va_proper',
    'get_vo_data',
    'get_vo_proper',

    'all_data',
    'del_data',
    'focus_on',
    'intermediate_step',
    'get_name',
    'get_focus_name',
    'get_x_name',

    'set_m_data',
    'set_o_data',
    'set_v_data',
    'set_oo_data',
    'set_vv_data',
    'set_vo_data',
    'set_va_data',
    'set_oa_data',

    'annotation_items',
]


LOG = logging.getLogger(__name__)


Annotations = Union[MutableMapping[Any, Any], pd.DataFrame]


def setup(
    adata: AnnData,
    *,
    x_name: str,
    name: Optional[str] = None,
    tmp: bool = False
) -> None:
    '''
    Set up the annotated ``adata`` for use by the ``metacells`` package.

    This needs the ``x_name`` of the data contained in ``adata.X``, which also becomes the focus
    data.

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
    assert '__x__' not in adata.uns_keys()
    assert '|' not in x_name

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
    adata.uns['__x__'] = x_name
    adata.uns['__focus__'] = x_name
    _log_set_data(adata, 'm', '__focus__', x_name, force=True)


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
    x_name = get_x_name(adata)
    focus = get_focus_name(adata)
    assert x_name not in adata.layers
    assert has_data(adata, focus)

    shaped: Optional[utt.Shaped] = utt.BaseShaped.maybe(obs)  # type: ignore
    if shaped is None:
        if obs is None:
            obs = range(adata.n_obs)
        will_slice_obs = len(obs) != adata.n_obs
    else:
        obs = utt.to_dense_vector(shaped)
        if obs.dtype == 'bool':
            assert np.any(obs)
            will_slice_obs = not np.all(obs)
        else:
            assert obs.size > 0
            will_slice_obs = obs.size != adata.n_obs

    shaped = utt.BaseShaped.maybe(vars)  # type: ignore
    if shaped is None:
        if vars is None:
            vars = range(adata.n_vars)
        will_slice_var = len(vars) != adata.n_vars
    else:
        vars = utt.to_dense_vector(shaped)
        if vars.dtype == 'bool':
            assert np.any(vars)
            will_slice_var = not np.all(vars)
        else:
            assert vars.size > 0
            will_slice_var = vars.size != adata.n_vars

    if not will_slice_obs and not will_slice_var:
        with utm.timed_step('adata.copy'):
            return adata.copy()

    saved_data = _save_data(adata, will_slice_obs, will_slice_var)

    with utm.timed_step('adata.slice'):
        bdata = adata[obs, vars].copy()

    did_slice_obs = bdata.n_obs != adata.n_obs
    did_slice_var = bdata.n_vars != adata.n_vars

    assert did_slice_obs == will_slice_obs
    assert did_slice_var == will_slice_var

    _restore_data(saved_data, adata)

    assert get_x_name(adata) == x_name
    assert get_focus_name(adata) == focus
    assert x_name not in adata.layers
    assert has_data(adata, focus)

    if tmp:
        bdata.uns['__tmp__'] = True
    if name is not None:
        if name.startswith('.'):
            base_name = get_name(adata)
            if base_name is None:
                name = name[1:]
            else:
                name = base_name + name
        bdata.uns['__name__'] = name
        LOG.log(utl.get_log_level(bdata),
                '  sliced %s shape %s', name, bdata.shape)

    assert get_x_name(bdata) == x_name
    if focus == x_name or focus in bdata.layers:
        assert get_focus_name(bdata) == focus
    else:
        focus = bdata.uns['__focus__'] = x_name
    _log_set_data(bdata, 'm', '__focus__', focus, force=True)

    if track_obs is not None:
        obs_indices = np.arange(adata.n_obs)[obs]
        set_o_data(bdata, track_obs, obs_indices)

    if track_var is not None:
        var_indices = np.arange(adata.n_vars)[vars]
        set_v_data(bdata, track_var, var_indices)

    assert x_name not in bdata.layers
    assert has_data(bdata, focus)

    return bdata


def _save_data(
    adata: AnnData,
    will_slice_obs: bool,
    will_slice_var: bool,
) -> Dict[Tuple[str, str], Any]:
    saved_data: Dict[Tuple[str, str], Any] = {}

    for per, annotations in (('vo', adata.layers),
                             ('o', adata.obs),
                             ('v', adata.var),
                             ('oo', adata.obsp),
                             ('vv', adata.varp),
                             ('oa', adata.obsm),
                             ('va', adata.varm),
                             ('m', adata.uns)):
        _save_per_data(saved_data, adata, per, annotations,
                       will_slice_obs, will_slice_var)

    return saved_data


def _save_per_data(
    saved_data: Dict[Tuple[str, str], Any],
    adata: AnnData,
    per: str,
    annotations: Annotations,
    will_slice_obs: bool,
    will_slice_var: bool,
) -> None:
    delete_names: Set[str] = set()

    if per not in ('vo', 'va', 'oa'):
        def patch(name: str) -> None:  # pylint: disable=unused-argument
            pass
    else:
        will_slice_only_obs = will_slice_obs and not will_slice_var
        will_slice_only_var = will_slice_var and not will_slice_obs

        def patch(name: str) -> None:
            if will_slice_only_obs:
                patch_only(name, ':__row_major__')
            if will_slice_only_var:
                patch_only(name, ':__column_major__')

        def patch_only(name: str, by_suffix: str) -> None:
            if not name.endswith(by_suffix):
                return

            data = adata.layers[name]
            delete_names.add(name)

            base_name = name[:-len(by_suffix)]
            if base_name == get_x_name(adata):
                adata.X = data
            else:
                base_data = adata.layers[base_name]
                saved_data[(per, base_name)] = base_data

    for name, data in annotation_items(annotations):
        if _preserve_data(name, will_slice_obs, will_slice_var):
            patch(name)
        else:
            saved_data[(per, name)] = data
            delete_names.add(name)

    for name in delete_names:
        del annotations[name]


def _restore_data(saved_data: Dict[Tuple[str, str], Any], adata: AnnData) -> None:
    per_annotations = dict(m=adata.uns,
                           o=adata.obs,
                           v=adata.var,
                           vo=adata.layers,
                           oo=adata.obsp,
                           vv=adata.varp,
                           oa=adata.obsm,
                           va=adata.varm)
    for (per, name), data in saved_data.items():
        per_annotations[per][name] = data


def annotation_items(annotations: Annotations) -> Iterable[Tuple[str, Any]]:
    '''
    Given some annotations data (such as ``adata.obs``, ``adata.layers``, etc.), iterate on all the
    name and value pairs it contains.
    '''
    if isinstance(annotations, pd.DataFrame):
        return annotations.iteritems()
    return annotations.items()


def _preserve_data(
    name: str,
    do_slice_obs: bool,
    do_slice_var: bool,
) -> bool:
    base_name = name
    if '__' in name:
        is_per_obs = name.endswith(':__row_major__')
        is_per_var = name.endswith(':__column_major__')
        if is_per_obs or is_per_var:
            base_name = ':'.join(name.split(':')[:-1])
    else:
        is_per_obs = False
        is_per_var = False

    slicing_mask = safe_slicing_mask(base_name)
    if is_per_var:
        slicing_mask = slicing_mask & SAFE_WHEN_SLICING_VAR
    if is_per_obs:
        slicing_mask = slicing_mask & SAFE_WHEN_SLICING_OBS

    slicing_mask = slicing_mask | SlicingMask(obs=not do_slice_obs,
                                              vars=not do_slice_var)

    return slicing_mask.obs and slicing_mask.vars


def get_vector_parameter_data(
    logger: Logger,
    adata: AnnData,
    value: Optional[Union[str, utt.Vector]],
    *,
    name: str,
    per: str,
    default: str = 'None',
    indent: str = '  ',
) -> Optional[utt.DenseVector]:
    '''
    Given a parameter ``value`` which is either a name or an explicit (optional) vector parameter,
    log it and either return it (as dense) or get it from the annotated data.
    '''
    assert per in ('o', 'v')

    if isinstance(value, str):
        logger.debug('%s%s: %s', indent, name, value)
        value = get_data(adata, value, per=per)
    elif value is None:
        logger.debug('%s%s: %s', indent, name, default)
    else:
        logger.debug('%s%s: <vector>', indent, name)

    if value is not None:
        value = utt.to_dense_vector(value)  # type: ignore

    return value


@utm.timed_call()
@utd.expand_doc()
def get_data(  # pylint: disable=too-many-return-statements
    adata: AnnData,
    name: Optional[str] = None,
    *,
    per: Optional[str] = None,
    compute: Optional[Callable[[], Any]] = None,
    inplace: bool = True,
    infocus: bool = False,
    layout: Optional[str] = None,
) -> Any:
    '''
    Lookup any data by its ``name`` (default: {name}).

    If no ``name`` is specified, return the focus.

    If the data does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace`` (default: {inplace}), store the data for future reuse. This requires ``per`` (one
    of: ``vo``, ``vv``, ``oo``, ``va``, ``oa``, ``v``, ``o``, ``m``, default: {per}) to specify
    where to store the data.

    If the data is per-variable-per-observation, and ``infocus`` (default: {infocus}, implies
    ``inplace``), also makes the result the new focus.

    If the data is 2-dimensional, and ``layout`` (default: {layout}) is specified, it forces the
    layout of the returned data.
    '''
    assert layout is None or layout in utt.LAYOUT_OF_AXIS

    if name is None:
        name = get_focus_name(adata)

    if per == 'vo' \
            or (per is None
                and (name == get_x_name(adata) or name in adata.layers)):
        return get_vo_data(adata, name, compute=compute,
                           inplace=inplace, infocus=infocus, layout=layout)

    assert not infocus

    if per == 'oo' or (per is None and name in adata.obsp):
        return get_oo_data(adata, name, compute=compute,
                           inplace=inplace, layout=layout)

    if per == 'vv' or (per is None and name in adata.varp):
        return get_vv_data(adata, name, compute=compute,
                           inplace=inplace, layout=layout)

    if per == 'oa' or (per is None and name in adata.obsm):
        return get_oa_data(adata, name, compute=compute,
                           inplace=inplace, layout=layout)

    if per == 'va' or (per is None and name in adata.varm):
        return get_va_data(adata, name, compute=compute,
                           inplace=inplace, layout=layout)

    assert layout is None

    if per == 'm' or (per is None and name in adata.uns):
        return get_m_data(adata, name, compute=compute, inplace=inplace)

    if per == 'o' or (per is None and name in adata.obs):
        return get_o_data(adata, name, compute=compute, inplace=inplace)

    if per == 'v' or (per is None and name in adata.var):
        return get_v_data(adata, name, compute=compute, inplace=inplace)

    raise _unknown_data(adata, name, per)


@utm.timed_call()
def get_proper_matrix(
    adata: AnnData,
    name: Optional[str] = None,
    *,
    per: Optional[str] = None,
    compute: Optional[Callable[[], Any]] = None,
    inplace: bool = True,
    infocus: bool = False,
    layout: Optional[str] = None,
) -> utt.ProperMatrix:
    '''
    Same as :py:func:`get_data`, except that the returned data is
    passed through :py:func:`metacells.utilities.typing.to_proper_matrix`
    ensuring it is a :py:const:`metacells.utilities.typing.ProperMatrix`.
    '''
    data = get_data(adata, name, per=per, compute=compute,
                    inplace=inplace, infocus=infocus, layout=layout)
    assert utt.is_layout(data, layout)
    return utt.to_proper_matrix(data, default_layout=layout)


@utm.timed_call()
def get_dense_vector(
    adata: AnnData,
    name: Optional[str] = None,
    *,
    per: Optional[str] = None,
    compute: Optional[Callable[[], Any]] = None,
    inplace: bool = True,
    infocus: bool = False,
) -> utt.DenseVector:
    '''
    Same as :py:func:`get_data`, except that the returned data is
    passed through :py:func:`metacells.utilities.typing.to_dense_vector`
    ensuring it is a :py:const:`metacells.utilities.typing.DenseVector`.
    '''
    return utt.to_dense_vector(get_data(adata, name, per=per, compute=compute,
                                        inplace=inplace, infocus=infocus))


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
        if (id(annotations) == id(adata.layers) and name == get_x_name(adata)) \
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

    if name == get_x_name(adata):
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


@utm.timed_call()
@utd.expand_doc()
def get_m_data(
    adata: AnnData,
    name: str,
    compute: Optional[Callable[[], utt.Vector]] = None,
    *,
    inplace: bool = True,
) -> Any:
    '''
    Lookup metadata (unstructured annotation) in ``adata`` by its ``name``.

    If the metadata does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace`` (default: {inplace}), store the result in ``adata``.
    '''
    data = adata.uns.get(name)
    if data is not None:
        return data

    if compute is None:
        raise _unknown_data(adata, name, 'm')

    data = compute()
    assert data is not None

    if inplace:
        _log_set_data(adata, 'm', name, data)
        adata.uns[name] = data

    return data


@utm.timed_call()
@utd.expand_doc()
def get_o_data(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Vector]] = None,
    inplace: bool = True,
) -> utt.Vector:
    '''
    Lookup per-observation (cell) data in ``adata`` by its ``name``.

    If the data does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace`` (default: {inplace}), store the result in ``adata`` for future reuse.
    '''
    return _get_shaped_data(adata, 'o', adata.obs, shape=(adata.n_obs,),
                            name=name, compute=compute, inplace=inplace)


def get_o_dense(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Vector]] = None,
    inplace: bool = True,
) -> utt.DenseVector:
    '''
    Same as :py:func:`get_o_data` but returns a numpy
    :py:const:`metacells.utilities.typing.DenseVector`.
    '''
    return utt.to_dense_vector(get_o_data(adata, name,
                                          compute=compute, inplace=inplace))


@utm.timed_call()
@utd.expand_doc()
def get_v_data(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Vector]] = None,
    inplace: bool = True,
) -> utt.Vector:
    '''
    Lookup per-variable (gene) data in ``adata`` by its ``name``.

    If the data does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace`` (default: {inplace}), store the result in ``adata`` for future reuse.
    '''
    return _get_shaped_data(adata, 'v', adata.var, shape=(adata.n_vars,),
                            name=name, compute=compute, inplace=inplace)


def get_v_dense(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Vector]] = None,
    inplace: bool = True,
) -> utt.DenseVector:
    '''
    Same as :py:func:`get_v_data` but returns a numpy
    :py:const:`metacells.utilities.typing.DenseVector`.
    '''
    return utt.to_dense_vector(get_v_data(adata, name,
                                          compute=compute, inplace=inplace))


@utm.timed_call()
@utd.expand_doc()
def get_oo_data(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Matrix]] = None,
    inplace: bool = True,
    layout: Optional[str] = None,
) -> utt.Matrix:
    '''
    Lookup per-observation-per-observation (cell) data in ``adata`` by its ``name``.

    If the data does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace`` (default: {inplace}), store the result in ``adata`` for future reuse.

    If ``layout`` (default: {layout}) is specified, it must be one of ``row_major`` or
    ``column_major`` (genes). This returns the data in a layout optimized for by-observation
    (row-major / csr) or by-variable (column-major / csc). If also ``inplace``, this is cached in an
    additional "hidden" annotation whose name is suffixed (e.g. ``...:__row_major__``).
    '''
    return _get_layout_data(adata, 'oo', adata.obsp,
                            shape=(adata.n_obs, adata.n_obs),
                            name=name, compute=compute,
                            inplace=inplace, layout=layout)


def get_oo_proper(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Matrix]] = None,
    inplace: bool = True,
    layout: Optional[str] = None,
) -> utt.ProperMatrix:
    '''
    Same as :py:func:`get_oo_data` but returns a numpy
    :py:const:`metacells.utilities.typing.ProperMatrix`.
    '''
    return utt.to_proper_matrix(get_oo_data(adata, name, compute=compute,
                                            inplace=inplace, layout=layout),
                                default_layout=layout)


@utm.timed_call()
@utd.expand_doc()
def get_vv_data(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Matrix]] = None,
    inplace: bool = True,
    layout: Optional[str] = None,
) -> utt.Matrix:
    '''
    Lookup per-variable-per-variable (gene) data in ``adata``.

    If the data does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace`` (default: {inplace}), store the result in ``adata`` for future reuse.

    If ``layout`` (default: {layout}) is specified, it must be one of ``row_major`` or
    ``column_major`` (genes). This returns the data in a layout optimized for by-observation
    (row-major / csr) or by-variable (column-major / csc). If also ``inplace``, this is cached in an
    additional "hidden" annotation whose name is suffixed (e.g. ``...:__row_major__``).
    '''
    return _get_layout_data(adata, 'vv', adata.varp,
                            shape=(adata.n_vars, adata.n_vars),
                            name=name, compute=compute,
                            inplace=inplace, layout=layout)


def get_vv_proper(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Matrix]] = None,
    inplace: bool = True,
    layout: Optional[str] = None,
) -> utt.ProperMatrix:
    '''
    Same as :py:func:`get_vv_data` but returns a numpy
    :py:const:`metacells.utilities.typing.ProperMatrix`.
    '''
    return utt.to_proper_matrix(get_vv_data(adata, name, compute=compute,
                                            inplace=inplace, layout=layout),
                                default_layout=layout)


@utm.timed_call()
@utd.expand_doc()
def get_oa_data(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Matrix]] = None,
    inplace: bool = True,
    layout: Optional[str] = None,
) -> utt.Matrix:
    '''
    Lookup per-observation-per-any (cell) data in ``adata`` by its ``name``.

    If the data does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace`` (default: {inplace}), store the result in ``adata`` for future reuse.

    If ``layout`` (default: {layout}) is specified, it must be one of ``row_major`` or
    ``column_major`` (genes). This returns the data in a layout optimized for by-observation
    (row-major / csr) or by-variable (column-major / csc). If also ``inplace``, this is cached in an
    additional "hidden" annotation whose name is suffixed (e.g. ``...:__row_major__``).
    '''
    return _get_layout_data(adata, 'oa', adata.obsm,
                            shape=(adata.n_obs, 0),
                            name=name, compute=compute,
                            inplace=inplace, layout=layout)


def get_oa_proper(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Matrix]] = None,
    inplace: bool = True,
    layout: Optional[str] = None,
) -> utt.ProperMatrix:
    '''
    Same as :py:func:`get_oa_data` but returns a numpy
    :py:const:`metacells.utilities.typing.ProperMatrix`.
    '''
    return utt.to_proper_matrix(get_oa_data(adata, name, compute=compute,
                                            inplace=inplace, layout=layout),
                                default_layout=layout)


@utm.timed_call()
@utd.expand_doc()
def get_va_data(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Matrix]] = None,
    inplace: bool = True,
    layout: Optional[str] = None,
) -> utt.Matrix:
    '''
    Lookup per-variable-per-variable (gene) data in ``adata``.

    If the data does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace`` (default: {inplace}), store the result in ``adata`` for future reuse.

    If ``layout`` (default: {layout}) is specified, it must be one of ``row_major`` or
    ``column_major`` (genes). This returns the data in a layout optimized for by-observation
    (row-major / csr) or by-variable (column-major / csc). If also ``inplace``, this is cached in an
    additional "hidden" annotation whose name is suffixed (e.g. ``...:__row_major__``).
    '''
    return _get_layout_data(adata, 'va', adata.varm,
                            shape=(adata.n_vars, 0),
                            name=name, compute=compute,
                            inplace=inplace, layout=layout)


def get_va_proper(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Matrix]] = None,
    inplace: bool = True,
    layout: Optional[str] = None,
) -> utt.ProperMatrix:
    '''
    Same as :py:func:`get_va_data` but returns a numpy
    :py:const:`metacells.utilities.typing.ProperMatrix`.
    '''
    return utt.to_proper_matrix(get_va_data(adata, name, compute=compute,
                                            inplace=inplace, layout=layout),
                                default_layout=layout)


@utm.timed_call()
@utd.expand_doc()
def get_vo_data(
    adata: AnnData,
    name: Optional[str] = None,
    *,
    compute: Optional[Callable[[], utt.Matrix]] = None,
    inplace: bool = True,
    infocus: bool = False,
    layout: Optional[str] = None,
) -> utt.Matrix:
    '''
    Lookup a per-variable-per-observation matrix (data layer) in ``adata`` by its ``name``.

    If the ``name`` is not specified, get the focus data.

    If the layer does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace`` (default: {inplace}), store the result in ``adata`` for future reuse.

    If ``infocus`` (default: {infocus}, implies ``inplace``), also makes the result the new focus
    data.

    If ``layout`` (default: {layout}) is specified, it must be one of ``row_major`` or
    ``column_major`` (genes). This returns the data in a layout optimized for by-observation
    (row-major / csr) or by-variable (column-major / csc). If also ``inplace`` (or ``infocus``),
    this is cached in an additional "hidden" layer whose name is suffixed (e.g.
    ``...:__row_major__``).

    Returns the result data.
    '''
    if name is None:
        name = get_focus_name(adata)

    data = _get_layout_data(adata, 'vo', adata.layers,
                            shape=(adata.n_obs, adata.n_vars),
                            name=name, compute=compute,
                            inplace=inplace or infocus, layout=layout)

    if infocus:
        _log_set_data(adata, 'm', '__focus__', name)
        adata.uns['__focus__'] = name

    return data


def get_vo_proper(
    adata: AnnData,
    name: Optional[str] = None,
    *,
    compute: Optional[Callable[[], utt.Matrix]] = None,
    inplace: bool = True,
    layout: Optional[str] = None,
) -> utt.ProperMatrix:
    '''
    Same as :py:func:`get_vo_data` but returns a numpy
    :py:const:`metacells.utilities.typing.ProperMatrix`.
    '''
    return utt.to_proper_matrix(get_vo_data(adata, name, compute=compute,
                                            inplace=inplace, layout=layout),
                                default_layout=layout)


def _get_layout_data(
    adata: AnnData,
    per: str,
    annotations: Annotations,
    *,
    shape: Tuple[int, ...],
    name: str,
    compute: Optional[Callable[[], utt.Shaped]],
    inplace: bool,
    layout: Optional[str],
) -> Any:
    assert '__' not in name

    def get_base_data() -> Any:
        return _get_shaped_data(adata, per, annotations, shape=shape,
                                name=name, compute=compute, inplace=inplace)

    if layout is None:
        return get_base_data()

    assert layout in utt.LAYOUT_OF_AXIS

    layout_name = '%s:__%s__' % (name, layout)
    data = adata.layers.get(layout_name)
    if data is not None:
        # TODO: assert utt.matrix_layout(data) == layout
        if utt.matrix_layout(data) == layout:
            return data
        warn('%s %s layout is actually: %s'
             % (get_name(adata, 'adata'),
                layout_name,
                utt.matrix_layout(data)))

    data = get_base_data()
    if utt.matrix_layout(data) == layout:
        return data

    data = utc.to_layout(data, layout=layout)
    assert utt.matrix_layout(data) == layout
    if inplace:
        if not utt.frozen(data):
            utt.freeze(data)
        _log_set_data(adata, 'vo', name, layout)
        assert utt.is_layout(data, layout)
        annotations[layout_name] = data

    return data


def _get_shaped_data(
    adata: AnnData,
    per: str,
    annotations: Annotations,
    *,
    shape: Tuple[int, ...],
    name: str,
    compute: Optional[Callable[[], utt.Shaped]],
    inplace: bool
) -> Any:
    assert '__' not in name

    if per == 'vo' and name == get_x_name(adata):
        data = _fix_data(adata.X)
        if not utt.frozen(data):
            utt.freeze(data)
        return data

    if (isinstance(annotations, pd.DataFrame) and name in annotations.columns) \
            or name in annotations:
        data = _fix_data(annotations[name])
        if not utt.frozen(data):
            utt.freeze(data)
        return data

    if compute is None:
        raise _unknown_data(adata, name, per)

    data = compute()
    assert data is not None
    if len(shape) == 2 and shape[1] == 0:
        assert data.shape[0] == shape[0]
    else:
        assert data.shape == shape
    assert utt.is_canonical(data)

    if inplace:
        if not utt.frozen(data):
            utt.freeze(data)
        _log_set_data(adata, per, name, data)
        annotations[name] = data

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

        You can't delete the ``x_name`` (layer of ``X``). Deleting the focus (default) data changes
        the focus to become whatever is in ``X``.

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

    assert '__' not in name

    if per is None:
        per = data_per(adata, name)
        assert per is not None

    _log_del_data(adata, per, name, layout)

    if per == 'vo':
        x_name = get_x_name(adata)
        if name == get_focus_name(adata):
            _log_set_data(adata, 'm', '__focus__', x_name)
            adata.uns['__focus__'] = x_name

    annotation = _annotation_per(adata, per)

    if layout is not None:
        layout_name = '%s:__%s__' % (name, layout)
        if layout_name in adata.layers:
            del annotation[layout_name]
        elif must_exist:
            assert layout_name in annotation
        return

    if per == 'vo':
        assert name != x_name

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


@contextmanager
@utd.expand_doc()
def focus_on(
    accessor: Callable,
    adata: AnnData,
    *args: Any,
    intermediate: bool = True,
    keep: Optional[Union[str, Collection[str]]] = None,
    **kwargs: Any
) -> Iterator[Any]:
    '''
    Get some per-variable-per-observation data by invoking the ``accessor`` with ``adata`` (and some
    ``args`` and ``kwargs``), and make it the focus for the duration of the ``with`` statement.

    If the original focus data is deleted inside the ``with`` statement, then when it is done, the
    focus will revert to whatever is in ``X``.

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it, unless it is listed in ``keep``.

    For example, in order to temporarily focus on the log of some linear measurements,
    write:

    .. code:: python

        ...
        # The focus data is some linear measurements.
        with ut.focus_on(ut.get_log, adata) as log_data:
            # log_data contains the log of the measurements.
            # The focus data is the log of the measurements.
            ...
        # The focus data is back to the linear measurements.
        ...

    It is also possible to focus on some data by its name by writing ``focus_on(get_vo_data, adata,
    name)``.

    .. note::

        Do not specify ``inplace`` and/or ``infocus`` in the ``kwargs``, as they are implied.
    '''
    for name in ('infocus', 'inplace'):
        if name in kwargs:
            ignoring_redundant_explict_flags_to_focus_on = \
                'ignoring explicit %s flag for focus_on' % name
            warn(ignoring_redundant_explict_flags_to_focus_on)
            del kwargs[name]

    with intermediate_step(adata, intermediate=intermediate, keep=keep):
        yield accessor(adata, *args, infocus=True, **kwargs)


@contextmanager
@utd.expand_doc()
def intermediate_step(
    adata: AnnData,
    *,
    intermediate: bool = True,
    keep: Optional[Union[str, Collection[str]]] = None,
) -> Iterator[None]:
    '''
    Execute some code in a ``with`` statements and restore the focus at the end, if it was modified
    by the wrapped code.

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it, unless it is listed in ``keep``.

    .. note::

        This does not protected against deletion of data by the wrapped code. If the original focus
        data is deleted inside the ``with`` statement, then when it is done, the focus will revert
        to whatever is in ``X``.
    '''
    if not intermediate:
        old_tmp = '__tmp__' in adata.uns
        adata.uns['__tmp__'] = True
        old_data = all_data(adata)

    old_focus = get_focus_name(adata)

    if keep is None:
        keep = []
    elif isinstance(keep, str):
        keep = [keep]

    try:
        yield

    finally:
        if not intermediate:
            if not old_tmp:
                del adata.uns['__tmp__']
            new_data = all_data(adata)
            for per, new_names in new_data.items():
                old_names = old_data[per]
                for name in new_names:
                    if name not in old_names and name not in keep:
                        del_data(adata, name, per=per)

        if has_data(adata, old_focus):
            adata.uns['__focus__'] = old_focus
        else:
            x_name = get_x_name(adata)
            _log_set_data(adata, 'm', '__focus__', x_name)
            adata.uns['__focus__'] = x_name


def all_data(adata: AnnData) -> Dict[str, Set[str]]:
    '''
    Return all the data stored in the ``adata``.
    '''
    names = dict(x=set([get_x_name(adata)]))

    for per, annotations in (('m', adata.uns),
                             ('vo', adata.layers),
                             ('o', adata.obs),
                             ('v', adata.var),
                             ('oo', adata.obsp),
                             ('vv', adata.varp),
                             ('oa', adata.obsm),
                             ('va', adata.varm)):
        names[per] = set()
        for name, _ in annotation_items(annotations):
            names[per].add(name)

    return names


def get_name(adata: AnnData, default: Optional[str] = None) -> Optional[str]:
    '''
    Return the name of the data (for log messages), if any.

    If no name was set, returns the ``default``.
    '''
    return adata.uns.get('__name__', default)


def get_focus_name(adata: AnnData) -> str:
    '''
    Return the name of the focus per-variable-per-observation data.
    '''
    return adata.uns['__focus__']


def get_x_name(adata: AnnData) -> str:
    '''
    Return the name of the ``X`` per-variable-per-observation data.
    '''
    return adata.uns['__x__']


@utm.timed_call()
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


@utm.timed_call()
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


@utm.timed_call()
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


@utm.timed_call()
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


@utm.timed_call()
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


@utm.timed_call()
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


@utm.timed_call()
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


@utm.timed_call()
@utd.expand_doc()
def set_vo_data(
    adata: AnnData,
    name: str,
    data: utt.ProperMatrix,
    *,
    log_value: Optional[Callable[[Any], Optional[str]]] = None,
    infocus: bool = False,
) -> Any:
    '''
    Set per-variable-per-observation (per-gene-per-cell) meta-data.

    If ``infocus`` (default: {infocus}, also make the result the new focus.
    '''
    if name == get_x_name(adata):
        _log_set_data(adata, 'x', name, data, log_value=log_value)
    else:
        _log_set_data(adata, 'vo', name, data, log_value=log_value)

    assert utt.is_canonical(data)
    if not utt.frozen(data):
        utt.freeze(data)

    if name == get_x_name(adata):
        adata.X = data
    else:
        adata.layers[name] = data

    if infocus:
        adata.uns['__focus__'] = name


MEMBER_OF_PER = \
    dict(m='uns',
         o='obs', v='var',
         oo='obsp', vv='varp',
         oa='obsm', va='varm',
         vo='layers', x='X')


def _log_set_data(  # pylint: disable=too-many-return-statements,too-many-branches,too-many-statements
    adata: AnnData,
    per: str,
    name: str,
    value: Any = None,
    force: bool = False,
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

        if name == '__focus__':
            if not force and value == adata.uns['__focus__']:
                return
            texts.append('focusing ')
            if data_name is not None:
                texts.append(data_name)
            texts.append(' on ')
            assert isinstance(value, str)
            texts.append(value)
            return

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
#           if value.ndim == 2:
#               value = utt.to_dense_matrix(value).astype('float64')
#               checksum = np.sum(np.sum(value * (1+np.arange(value.shape[1])), axis=1) * (1+np.arange(value.shape[0])))
#           else:
#               value = utt.to_proper_vector(value).astype('float64')
#               checksum = np.sum(value * (1+np.arange(len(value))))
#           texts.append(' checksum ')
#           texts.append(str(checksum))

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
