'''
Utilities for dealing with ``AnnData``.

This tries to deal safely with slicing data layers, observation, variable and unstructured
annotations, as well as two-dimensional variable and observation annotations.

.. todo::

    Move the description here to a more appropriate part of the documentation.

.. todo::

    Discuss the different models with the maintainers of ``AnnData`` to see whether it would be
    possible to reconcile the two approaches.

Data Model
----------

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
    (none)                  multidimensional annotations (``.obsp``, ``.varp``, ``.obsm``, ``.varm``)
    ======================  =========================================================================

.. todo::

    Support multidimensional annotations.

Mapping Code Complexity
-----------------------

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
    be initialized using :py:func:`metacells.utilities.preparation.prepare`, and pretends that the
    value of ``X`` is just another layer. This gives rise to some edge cases (e.g., one can't delete
    the ``X`` layer).

2.  Layout-optimized data.

    For both dense and sparse formats, having the data in a layout that is friendly for the type of
    processing done on it makes for a big performance difference. For dense data this is "FORTRAN"
    vs. "C" layouts; for sparse data it is "CSR" vs. "CSC" format.

    Therefore, the managed ``AnnData`` allows for requesting a specific layout, and caches the
    results. Instead of monkey-patching the ``AnnData`` object, this is done by creating "hidden"
    layers with suffixed names (``...:__by_obs__`` and ``...:__by_var__``).

    Changing the layout of a matrix using the builtin operations is very slow; therefore a parallel
    C++ extension is provided.

3.  Safe and efficient slicing.

    The managed ``AnnData`` provides a :py:func:`slice` operation that performs fast and safe
    slicing.

    To achieve this, the code tracks, for each possible data, whether it is/not safe to slice it
    along each of the axis (e.g., the per-observation ``o:sum_of_vo:...`` data is invalidated when
    slicing some of the variables).

    In addition, since slicing layout-optimized data is much faster, the code ensures that,
    as much as possible, slicing is always applied to the proper layout form of the data.

The interaction between the above three features is the cause of most of the implementation
complexity. The upside is that application code becomes simpler and is more efficient.

Usage Differences
-----------------

The managed ``AnnData`` model prefers to consider all data as immutable. That is, if one wants to
compute the log of some values, the common approach in raw ``AnnData`` is to mutate ``X`` to hold
the log data, while in the managed ``AnnData`` the approach would be to create a new data layer, so
that in principle one never needs to mutate any data. This is (not 100%) enforced by
:py:func:`metacells.utilities.computation.freeze`-ing the data whenever possible.

The managed ``AnnData`` provides standard functions that compute and cache new data from existing
data, using arbitrary compute functions (e.g., :py:func:`get_data`, :py:func:`get_vo_data`). See the
:py:mod:`metacells.utilities.preparation` module for these functions.

Data Names
----------

The name of data stored in a managed ``AnnData`` is ``prefix:suffix``, where the suffix is arbitrary
and the prefix is one of ``vo`` for per-variable-per-observation, ``v`` for per-variable, ``o`` for
per-observation, ``vv`` for per-observation-per-observation, or ``oo`` for for
per-observation-per-observation. Similarly, metadata names should start with ``m:``.

This ensures that names of different kinds of data do not get mixed up. For example, it allows for
the generic :py:func:`get_data` function to safely fetch any kind data without ambiguity.

.. note::

    The keys of the data structures containing the data do not contain the prefix. For example,
    ``v:gene_ids`` references the normal ``gene_ids`` per-variable data.
'''

from contextlib import contextmanager
from typing import (Any, Callable, Collection, Dict, Iterable, Iterator, List,
                    MutableMapping, NamedTuple, Optional, Set, Tuple, Union)
from warnings import warn

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy.sparse as sparse  # type: ignore
from anndata import AnnData
from readerwriterlock import rwlock

import metacells.utilities.computation as utc
import metacells.utilities.timing as timed
import metacells.utilities.typing as utt

__all__ = [
    'SlicingMask',
    'ALWAYS_SAFE',
    'NEVER_SAFE',
    'SAFE_WHEN_SLICING_OBS',
    'SAFE_WHEN_SLICING_VAR',
    'safe_slicing_data',
    'safe_slicing_derived',

    'slice',

    'NamedData',

    'get_data',
    'get_named_data',
    'has_data',
    'set_data',

    'get_m_data',
    'get_o_data',
    'get_v_data',
    'get_oo_data',
    'get_vv_data',

    'get_vo_data',
    'del_vo_data',
    'focus_on',
    'get_focus_name',
    'get_x_name',

]


LOCK = rwlock.RWLockRead()


Annotations = Union[MutableMapping[Any, Any], pd.DataFrame]


class SlicingMask(NamedTuple):
    '''
    A mask for safe operations when slicing some ``AnnData``.
    '''

    #: Whether the operation is safe when slicing observations (cells).
    per_obs: bool

    #: Whether the operation is safe when slicing variables (genes).
    per_var: bool

    def __and__(self, other: 'SlicingMask') -> 'SlicingMask':
        return SlicingMask(per_obs=self.per_obs and other.per_obs,
                           per_var=self.per_var and other.per_var)

    def __or__(self, other: 'SlicingMask') -> 'SlicingMask':
        return SlicingMask(per_obs=self.per_obs or other.per_obs,
                           per_var=self.per_var or other.per_var)


#: A mask for operations which are safe regardless of what is sliced.
ALWAYS_SAFE = SlicingMask(per_obs=True, per_var=True)

#: A mask for operations which are not safe regardless of what is sliced.
NEVER_SAFE = SlicingMask(per_obs=False, per_var=False)

#: A mask for operations which are only safe when slicing variables (genes),
#: but not when slicing observations (cells).
SAFE_WHEN_SLICING_VAR = SlicingMask(per_obs=False, per_var=True)

#: A mask for operations which are only safe when slicing observations (cells),
#: but not when slicing variables (genes).
SAFE_WHEN_SLICING_OBS = SlicingMask(per_obs=True, per_var=False)

SAFE_SLICING: Dict[str, SlicingMask] = {}

DIMENSIONAL_DATA_TYPES = (np.ndarray, sparse.spmatrix, pd.Series, pd.DataFrame)


def safe_slicing_data(name: str, slicing_mask: SlicingMask) -> None:
    '''
    Specify when it is safe to slice the ``name``d data, using the ``slicing_mask``.

    .. note::

        This is thread-safe.
    '''
    with LOCK.gen_wlock():
        SAFE_SLICING[name] = slicing_mask


def _known_safe_slicing() -> None:
    safe_slicing_data('m:__x__', ALWAYS_SAFE)
    safe_slicing_data('m:__focus__', ALWAYS_SAFE)
    safe_slicing_data('v:gene_ids', ALWAYS_SAFE)


_known_safe_slicing()


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
    with LOCK.gen_rlock():
        if isinstance(base, str):
            slicing_mask = slicing_mask & SAFE_SLICING[base]
        else:
            for name in base:
                slicing_mask = slicing_mask & SAFE_SLICING[name]

    with LOCK.gen_wlock():
        SAFE_SLICING[derived] = slicing_mask


@timed.call()
def slice(  # pylint: disable=redefined-builtin
    adata: AnnData,
    *,
    cells: Optional[Collection] = None,
    genes: Optional[Collection] = None,
    invalidated_prefix: Optional[str] = None,
) -> AnnData:
    '''
    Return new annotated data which includes a subset of the full ``adata``.

    If ``cells`` and/or ``genes`` are specified, they should include either a boolean
    mask or a collection of indices to include in the data slice. In the case of an indices array,
    it is assumed the indices are unique and sorted, that is that their effect is similar to a mask.

    In general, data might become invalid when slicing (e.g., the per-observation
    ``o:sum_of_vo:...`` data is invalidated when slicing some of the variables). Therefore, such
    data will be removed from the result unless it was explicitly marked as preserved using
    :py:func:`safe_slicing_data` or :py:func:`safe_slicing_derived`.

    If ``invalidated_prefix`` is specified, then invalidated data will not be removed; instead it
    will be renamed with the addition of the provided prefix.

    .. note::

        Setting the prefix to the empty string would simply preserve all the invalidated data. As
        this is unsafe, it will trigger a run-time exception. If you wish to perform such an unsafe
        slicing operation, invoke the built-in ``adata[..., ...]``.
    '''
    assert invalidated_prefix != ''
    x_name = get_x_name(adata)
    focus = get_focus_name(adata)
    assert x_name not in adata.layers
    assert has_data(adata, focus)

    if cells is None:
        cells = range(adata.n_obs)
    if genes is None:
        genes = range(adata.n_vars)

    will_slice_obs = len(cells) != adata.n_obs
    will_slice_var = len(genes) != adata.n_vars

    if not will_slice_obs and not will_slice_var:
        with timed.step('.copy'):
            return adata[:, :]

    saved_data = \
        _save_data(adata, will_slice_obs,
                   will_slice_var, invalidated_prefix)

    with timed.step('.builtin'):
        bdata = adata[cells, genes]

    did_slice_obs = bdata.n_obs != adata.n_obs
    did_slice_var = bdata.n_vars != adata.n_vars

    assert did_slice_obs == will_slice_obs
    assert did_slice_var == will_slice_var

    if invalidated_prefix is not None:
        _prefix_data(saved_data, bdata, did_slice_obs, did_slice_var,
                     invalidated_prefix)

    _restore_data(saved_data, adata)

    assert get_x_name(adata) == x_name
    assert get_focus_name(adata) == focus
    assert x_name not in adata.layers
    assert has_data(adata, focus)

    assert get_x_name(bdata) == x_name
    if focus == x_name or focus in bdata.layers:
        assert get_focus_name(bdata) == focus
    else:
        focus = bdata.uns['__focus__'] = x_name
    assert x_name not in bdata.layers
    assert has_data(bdata, focus)

    return bdata


def _save_data(
    adata: AnnData,
    will_slice_obs: bool,
    will_slice_var: bool,
    invalidated_prefix: Optional[str],
) -> Dict[str, Any]:
    saved_data: Dict[str, Any] = {}

    for per_prefix, annotations in [('vo:', adata.layers),
                                    ('o:', adata.obs),
                                    ('v:', adata.var),
                                    ('oo:', adata.obsp),
                                    ('vv:', adata.varp),
                                    ('m:', adata.uns)]:
        _save_per_data(saved_data, adata, per_prefix, annotations,
                       will_slice_obs, will_slice_var, invalidated_prefix)

    return saved_data


def _save_per_data(  # pylint: disable=too-many-locals
    saved_data: Dict[str, Any],
    adata: AnnData,
    per_prefix: str,
    annotations: Annotations,
    will_slice_obs: bool,
    will_slice_var: bool,
    invalidated_prefix: Optional[str],
) -> None:
    delete_names: Set[str] = set()
    prefixed_data: Dict[str, Any] = {}

    if per_prefix != 'vo:':
        def patch(name: str) -> None:  # pylint: disable=unused-argument
            pass
    else:
        will_slice_only_obs = will_slice_obs and not will_slice_var
        will_slice_only_var = will_slice_var and not will_slice_obs

        def patch(name: str) -> None:
            if will_slice_only_obs:
                patch_only(name, ':__by_obs__')
            if will_slice_only_var:
                patch_only(name, ':__by_var__')

        def patch_only(name: str, by_suffix: str) -> None:
            if not name.endswith(by_suffix):
                return

            data = adata.layers[name]
            delete_names.add(name)

            base_name = name[:-len(by_suffix)]
            if base_name == get_x_name(adata)[3:]:
                with timed.step('.swap_x'):
                    with _modify(adata):
                        adata.X = data
            else:
                base_data = adata.layers[base_name]
                saved_data['vo:' + base_name] = base_data

    for name, data in _items(annotations):
        which_data = per_prefix + name
        action = _slice_action(which_data, will_slice_obs, will_slice_var,
                               invalidated_prefix is not None)

        if action == 'discard':
            saved_data[which_data] = data
            delete_names.add(name)
            continue

        patch(name)

        if action == 'preserve':
            continue

        assert action == 'prefix'
        assert invalidated_prefix is not None
        prefixed_name = invalidated_prefix + name
        prefixed_data[prefixed_name] = data
        delete_names.add(name)
        continue

    for name in delete_names:
        del annotations[name]


def _restore_data(saved_data: Dict[str, Any], adata: AnnData) -> None:
    per_annotations = dict(vo=adata.layers,
                           o=adata.obs,
                           v=adata.var,
                           oo=adata.obsp,
                           vv=adata.varp,
                           m=adata.uns)
    for name, data in saved_data.items():
        per_prefix, name_suffix = name.split(':', 1)
        per_annotations[per_prefix][name_suffix] = data


def _prefix_data(
    saved_data: Dict[str, Any],
    bdata: AnnData,
    did_slice_obs: bool,
    did_slice_var: bool,
    invalidated_prefix: str,
) -> None:
    for per_prefix, annotations in [('vo:', bdata.layers),
                                    ('o:', bdata.obs),
                                    ('v:', bdata.var),
                                    ('oo:', bdata.obsp),
                                    ('vv:', bdata.varp),
                                    ('m:', bdata.uns)]:
        _prefix_per_data(saved_data, per_prefix, annotations,
                         did_slice_obs, did_slice_var, invalidated_prefix)


def _prefix_per_data(
    saved_data: Dict[str, Any],
    per_prefix: str,
    annotations: MutableMapping[str, Any],
    did_slice_obs: bool,
    did_slice_var: bool,
    invalidated_prefix: str,
) -> None:
    delete_names: List[str] = []

    for name, data in _items(annotations):
        full_name = per_prefix + name
        action = _slice_action(full_name, did_slice_obs, did_slice_var, True)

        if action == 'preserve':
            continue
        assert action == 'prefix'

        prefixed_name = invalidated_prefix + name
        assert prefixed_name != name
        full_prefixed_name = per_prefix + prefixed_name

        saved_data[full_prefixed_name] = data
        delete_names.append(name)

        with LOCK.gen_wlock():
            SAFE_SLICING[full_prefixed_name] = SAFE_SLICING[full_name]

    for name in delete_names:
        del annotations[name]


def _items(annotations: Annotations) -> Iterable[Tuple[str, Any]]:
    if isinstance(annotations, pd.DataFrame):
        return annotations.iteritems()
    return annotations.items()


def _get(annotations: Annotations, name: str) -> Any:
    if isinstance(annotations, pd.DataFrame):
        if name not in annotations.columns:
            return None
        return annotations[name]
    return annotations.get(name)


def _slice_action(
    name: str,
    do_slice_obs: bool,
    do_slice_var: bool,
    will_prefix_invalidated: bool,
) -> str:
    per_prefix, name_suffix = name.split(':', 1)
    if ':__by_' in name_suffix:
        is_per = True
        is_per_obs = name_suffix.endswith(':__by_obs__')
        is_per_var = name_suffix.endswith(':__by_var__')
        assert is_per_obs or is_per_var
        base_name = name_suffix[:-11]
    else:
        is_per = False
        is_per_obs = False
        is_per_var = False
        base_name = name_suffix

    full_base_name = '%s:%s' % (per_prefix, base_name)
    with LOCK.gen_rlock():
        slicing_mask = SAFE_SLICING.get(full_base_name)

    if slicing_mask is None:
        with LOCK.gen_wlock():
            slicing_mask = SAFE_SLICING.get(full_base_name)
            if slicing_mask is None:
                unknown_sliced_data = \
                    'Slicing an unknown data: %s; ' \
                    'assuming it should not be preserved' \
                    % full_base_name
                warn(unknown_sliced_data)
                slicing_mask = SAFE_SLICING[full_base_name] = NEVER_SAFE

    if is_per_var:
        slicing_mask = slicing_mask & SAFE_WHEN_SLICING_VAR
    if is_per_obs:
        slicing_mask = slicing_mask & SAFE_WHEN_SLICING_OBS

    slicing_mask = slicing_mask | SlicingMask(per_obs=not do_slice_obs,
                                              per_var=not do_slice_var)

    if slicing_mask.per_obs and slicing_mask.per_var:
        return 'preserve'

    if is_per or not will_prefix_invalidated:
        return 'discard'

    return 'prefix'


class NamedData(NamedTuple):
    '''
    Data stored in ``AnnData`` with its full name.
    '''

    #: The full name of the data, including the prefix (``m:``, ``o:``, ``v:``, ``vo:``', ``oo:``,
    #: ``vv:``).
    name: str

    #: The actual data.
    #:
    #: This will be anything at all for metadata, This will be a
    #: :py:typ:`metacells.utilities.computation.Matrix` for per-variable-per-observation,
    #: per-observation-per-observation, or per-variable-per-variable, data; a
    #: :py:typ:`metacells.utilities.computation.Vector` for per-observation or per-variable data;
    #: and anything at all for metadata.
    data: Any


@timed.call()
def get_data(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Vector]] = None,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> Any:
    '''
    Lookup any data by its full ``name``.

    If the data does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace``, store the result in ``adata`` for future reuse.

    If the data is per-variable-per-observation, and ``infocus`` (implies ``inplace``), also makes
    the result the new focus.

    If the data is per-variable-per-observation, and ``by`` is specified, it forces the layout of
    the returned data (see :py:func:`get_vo_data`).

    .. note::

        This is given the full name, and works for any kind data, based on the full name's prefix
        (e.g., ``m:`` for metadata, ``vo:`` for per-variable-per-observation). Therefore it returns
        just the data instead of a :py:typ:`NamedData`.
    '''
    return get_named_data(adata, name, compute=compute, inplace=inplace,
                          infocus=infocus, by=by).data


@timed.call()
def get_named_data(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Vector]] = None,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedData:
    '''
    Same as :py:func:`get_data` but return the full :py:typ:`NamedData`.
        just the data instead of a :py:typ:`NamedData`.
    '''
    if name.startswith('vo:'):
        return get_vo_data(adata, name, compute=compute, inplace=inplace,
                           infocus=infocus, by=by)

    assert not infocus

    if name.startswith('m:'):
        assert by is None
        return get_m_data(adata, name, compute=compute, inplace=inplace)

    if name.startswith('o:'):
        assert by is None or by == 'obs'
        return get_o_data(adata, name, compute=compute, inplace=inplace)

    if name.startswith('v:'):
        assert by is None or by == 'var'
        return get_v_data(adata, name, compute=compute, inplace=inplace)

    if name.startswith('oo:'):
        assert by is None or by == 'obs'
        return get_oo_data(adata, name, compute=compute, inplace=inplace)

    if name.startswith('vv:'):
        assert by is None or by == 'var'
        return get_vv_data(adata, name, compute=compute, inplace=inplace)

    raise ValueError('the data name: %s does not start with a valid prefix '
                     '("vo:", "v:", "o:", "vv:", "oo", or "m:")' % name)


def has_data(
    adata: AnnData,
    name: str,
) -> bool:
    '''
    Test whether we have the specified data.

    .. note::

        This works for any kind data, based on the name's prefix (e.g., ``m:`` for metadata, ``vo:``
        for per-variable-per-observation).
    '''
    if name.startswith('vo:'):
        return name == get_x_name(adata) or name[3:] in adata.layers

    if name.startswith('m:'):
        return name[2:] in adata.uns

    if name.startswith('o:'):
        return name[2:] in adata.obs

    if name.startswith('v:'):
        return name[2:] in adata.var

    if name.startswith('oo:'):
        return name[3:] in adata.obsp

    if name.startswith('vv:'):
        return name[3:] in adata.varp

    raise ValueError('the data name: %s does not start with a valid prefix '
                     '("vo:", "v:", "o:", "vv:", "oo", or "m:")' % name)


@timed.call()
def set_data(
    adata: AnnData,
    name: str,
    data: Any,
    slicing_mask: SlicingMask = ALWAYS_SAFE,
) -> None:
    '''
    Set some ``name``d ``data`` in the ``adata``.

    By default, it is assumed that the ``slicing_mask`` is :py:data:`ALWAYS_SAFE`, that is, it is
    allowed to preserve the data when slicing both observations (cell) and variables (genes).
    '''
    SAFE_SLICING[name] = slicing_mask

    if name.startswith('m:'):
        with _modify(adata):
            adata.uns[name[2:]] = data
            return

    if name.startswith('o:'):
        with _modify(adata):
            adata.obs[name[2:]] = data
            return

    if name.startswith('v:'):
        with _modify(adata):
            adata.var[name[2:]] = data
            return

    if name.startswith('oo:'):
        with _modify(adata):
            adata.obsp[name[3:]] = data
            return

    if name.startswith('vv:'):
        with _modify(adata):
            adata.varp[name[3:]] = data
            return

        raise ValueError('the data name: %s does not start with a valid prefix '
                         '("vo:", "v:", "o:", "vv:", "oo", or "m:")' % name)


def get_m_data(
    adata: AnnData,
    name: str,
    compute: Optional[Callable[[], utt.Vector]] = None,
    *,
    inplace: bool = True,
) -> NamedData:
    '''
    Lookup metadata (unstructured annotation) in ``adata``.

    If the name does not start with a ``m:`` prefix, one is assumed.

    If the metadata does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace``, store the result in ``adata``.

    .. note::

        The caller is responsible for specifying the slicing behavior of the data.
    '''
    if name.startswith('m:'):
        full_name = name
        name = name[2:]
    else:
        full_name = 'm:' + name

    data = adata.uns.get(name)
    if data is not None:
        return NamedData(name=full_name, data=data)

    if compute is None:
        raise KeyError('unavailable metadata: ' + name)

    data = compute()
    assert data is not None

    if inplace:
        with _modify(adata):
            adata.uns[name] = data

    return NamedData(name=full_name, data=data)


def get_o_data(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Vector]] = None,
    inplace: bool = True,
) -> utt.Vector:
    '''
    Lookup per-observation (cell) data in ``adata``.

    If the name does not start with a ``o:`` prefix, one is assumed.

    If the data does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace``, store the result in ``adata`` for future reuse.

    .. note::

        The caller is responsible for specifying the slicing behavior of the data.
    '''
    return _get_shaped_data(adata.obs, (adata.n_obs,),
                            'per-observation', 'o:', name, compute, inplace)


def get_v_data(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Vector]] = None,
    inplace: bool = True,
) -> utt.Vector:
    '''
    Lookup per-variable (gene) data in ``adata``.

    If the data does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace``, store the result in ``adata`` for future reuse.

    .. note::

        The caller is responsible for specifying the slicing behavior of the data.
    '''
    return _get_shaped_data(adata.var, (adata.n_vars,),
                            'per-variable', 'v:', name, compute, inplace)


def get_oo_data(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Matrix]] = None,
    inplace: bool = True,
) -> utt.Matrix:
    '''
    Lookup per-observation-per-observation (cell) data in ``adata``.

    If the data does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace``, store the result in ``adata`` for future reuse.

    .. note::

        The caller is responsible for specifying the slicing behavior of the data.
    '''
    return _get_shaped_data(adata.obsp, (adata.n_obs, adata.n_obs),
                            'per-observation-per-observation', 'oo:',
                            name, compute, inplace)


def get_vv_data(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utt.Vector]] = None,
    inplace: bool = True,
) -> utt.Matrix:
    '''
    Lookup per-variable-per-variable (gene) data in ``adata``.

    If the data does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace``, store the result in ``adata`` for future reuse.

    .. note::

        The caller is responsible for specifying the slicing behavior of the data.
    '''
    return _get_shaped_data(adata.varp, (adata.n_vars, adata.n_vars),
                            'per-variable-per-variable', 'vv:',
                            name, compute, inplace)


def _get_shaped_data(
    annotations: Annotations,
    shape: Tuple[int, ...],
    per_text: str,
    per_prefix: str,
    name: str,
    compute: Optional[Callable[[], Any]],
    inplace: bool
) -> Any:
    if name.startswith(per_prefix):
        full_name = name
        name = name[len(per_prefix):]
    else:
        full_name = per_prefix + name

    data = _get(annotations, name)
    if data is not None:
        if isinstance(data, (pd.Series, pd.DataFrame)):
            data = data.values
        return NamedData(name=full_name, data=data)

    if compute is None:
        raise KeyError('unavailable %s data: %s' % (per_text, name))

    data = compute()
    assert data is not None
    assert data.shape == shape
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.values

    if inplace:
        utt.freeze(data)
        annotations[name] = data

    return NamedData(name=full_name, data=data)


@timed.call()
def get_vo_data(
    adata: AnnData,
    name: Optional[str] = None,
    *,
    compute: Optional[Callable[[], utt.Vector]] = None,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> NamedData:
    '''
    Lookup a per-variable-per-observation matrix (data layer) in ``adata``.

    If the ``name`` is not specified, get the focus data. If the name does not start with a ``vo:``
    prefix, one is assumed.

    If the layer does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace``, store the result in ``adata`` for future reuse.

    If ``infocus`` (implies ``inplace``), also makes the result the new focus data.

    If ``by`` is specified, it must be one of ``obs`` (cells) or ``var`` (genes). This returns the
    data in a layout optimized for by-observation (row-major / csr) or by-variable (column-major
    / csc). If also ``inplace`` (or ``infocus``), this is cached in an additional "hidden" layer
    whose name is suffixed (e.g. ``...:__by_obs__``).

    Returns the result and its full name.

    .. note::

        In general names that contain ``__`` are reserved and should not be explicitly used.

    .. note::

        The original data returned by ``compute`` is always preserved under its non-prefixed name.

    .. todo::

        A better implementation of :py:func:`get_vo_data` would be to cache the
        layout-specific data in a private variable, but we do not own the ``AnnData`` object.
    '''
    if name is None:
        name = get_focus_name(adata)

    assert '__' not in name

    if name.startswith('vo:'):
        full_name = name
        name = full_name[3:]
    else:
        full_name = 'vo:' + name

    if infocus:
        with _modify(adata):
            adata.uns['__focus__'] = full_name

    if full_name == get_x_name(adata):
        def get_base_data() -> utt.Matrix:
            with timed.step('.get_x'):
                data = adata.X

            assert data is not None
            assert data.shape == (adata.n_obs, adata.n_vars)
            return data
    else:
        def get_base_data() -> utt.Matrix:
            assert name is not None
            data = adata.layers.get(name)
            if data is not None:
                return data

            if compute is None:
                assert name is not None
                raise \
                    KeyError('unavailable per-variable-per-observation data: '
                             + name)
            data = compute()
            if sparse.issparse(data):
                if by == 'var':
                    data = data.tocsc()
                else:
                    data = data.tocsr()

            if inplace or infocus:
                with _modify(adata):
                    adata.layers[name] = data

            return data

    if by is None:
        data = get_base_data()

    else:
        assert by in ['var', 'obs']
        by_name = '%s:__by_%s__' % (name, by)
        axis = ['var', 'obs'].index(by)
        layout = ['column_major', 'row_major'][axis]

        data = adata.layers.get(by_name)
        if data is None:
            data = utc.to_layout(get_base_data(), layout=layout)
            if inplace or infocus:
                with _modify(adata):
                    adata.layers[by_name] = data

    assert data.shape == (adata.n_obs, adata.n_vars)
    if inplace or infocus:
        utt.freeze(data)

    return NamedData(name=full_name, data=data)


def del_vo_data(
    adata: AnnData,
    name: str,
    *,
    by: Optional[str] = None,
    must_exist: bool = False,
) -> None:
    '''
    Delete a per-variable-per-observation matrix in ``adata`` by its ``name``.

    If the name does not start with a ``vo:`` prefix, one is assumed.

    If ``by`` is specified, it must be one of ``obs`` (cells) or ``var`` (genes). This will only
    delete the cached layout-specific data. If ``by`` is not specified, both the data and any
    cached layout-specific data will be deleted.

    If ``must_exist``, will ``raise`` if the data does not currently exist.

    .. note::

        You can't delete the ``x_name`` (layer of ``X``). Deleting the focus (default) data changes
        the focus to become whatever is in ``X``.

    .. todo::

        The function :py:func:`del_vo_data` is only required due to the need to delete cached
        layout-specific data. A better way would be to intercept ``del`` of the ``AnnData``
        ``layers`` field, but we do not own it.
    '''
    assert '__' not in name

    if name.startswith('vo:'):
        full_name = name
        name = name[3:]
    else:
        full_name = 'vo:' + name

    x_name = get_x_name(adata)
    assert full_name != x_name

    if full_name == get_focus_name(adata):
        with _modify(adata):
            adata.uns['__focus__'] = x_name

    if by is not None:
        by_name = '%s:__by_%s__' % (name, by)
        if by_name in adata.layers:
            del adata.layers[by_name]
        elif must_exist:
            assert by_name in adata.layers
        return

    if must_exist:
        assert name in adata.layers
    for suffix in ['', ':__by_obs__', ':__by_var__']:
        suffixed_name = name + suffix
        if suffixed_name in adata.layers:
            del adata.layers[suffixed_name]


@contextmanager
def focus_on(
    accessor: Callable,
    adata: AnnData,
    *args: Any,
    **kwargs: Any
) -> Iterator[Any]:
    '''
    Get some per-variable-per-observation data by invoking the ``accessor`` with ``adata`` (and some
    ``args`` and ``kwargs``), and make it the focus for the duration of the ``with`` statement.

    If the original focus data is deleted inside the ``with`` statement, then when it is done, the
    focus will revert to whatever is in ``X``.

    For example, in order to temporarily focus on the log of some linear measurements,
    write:

    .. code:: python

        ...
        # The focus data is some linear measurements
        with ut.focus_on(get_log_data_per_var_per_obs, adata) as log_data:
            # log_data is a matrix containing the log of the measurements
            # The focus data is the log of the measurements
            ...
        # The focus data is back to the linear measurements
        ...

    It is also possible to focus on some data by its name by writing ``focus_on(get_vo_data, adata,
    name)``.

    .. note::

        Do not specify ``inplace`` and/or ``infocus`` in the ``kwargs``, as they are implied by this
        call.
    '''
    for name in ['infocus', 'inplace']:
        if name in kwargs:
            ignoring_redundant_explict_flags_to_focus_on = \
                'ignoring explicit %s flag for focus_on' % name
            warn(ignoring_redundant_explict_flags_to_focus_on)
            del kwargs[name]

    old_focus = get_focus_name(adata)

    yield accessor(adata, *args, infocus=True, **kwargs)

    with _modify(adata):
        if has_data(adata, old_focus):
            adata.uns['__focus__'] = old_focus
        else:
            adata.uns['__focus__'] = get_x_name(adata)


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


@contextmanager
def _modify(adata: AnnData) -> Iterator[None]:
    if adata.is_view:
        with timed.step('unview'):
            yield
    else:
        yield
