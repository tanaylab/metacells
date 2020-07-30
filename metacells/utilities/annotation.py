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

    In the managed ``AnnData``, all the layers are equal. A special metadata property ``focus``
    provides the default layer name for all layer functions. Similarly, operations that output a
    layer allow specifying that the result will be ``infocus``, and the :py:func:`focus_on` function
    allows for code to safely shift the focus for specific code regions.

    In contrast, raw ``AnnData`` contains a magic ``.X`` member which is distinct from the rest of
    the data layers. There is a strong assumption that all work will be done on ``X`` and the other
    layers are secondary at best.

    The managed ``AnnData`` therefore keeps a second special metadata property ``x_name`` which must
    be initialized using :py:func:`set_x_name`, and pretends that the value of ``X`` is just another
    layer. This gives rise to some edge cases (e.g., one can't delete the ``X`` layer).

2.  Layout-optimized data.

    For both dense and sparse formats, having the data in a layout that is friendly for the type of
    processing done on it makes for a big performance difference. For dense data this is "FORTRAN"
    vs. "C" layouts; for sparse data it is "CSR" vs. "CSC" format.

    Therefore, the managed ``AnnData`` allows for requesting a specific layout, and caches the
    results. Instead of monkey-patching the ``AnnData`` object, this is done by creating "hidden"
    layers with prefixed names (``__by_obs__XXX`` and ``__by_var__XXX``).

    Changing the layout of a matrix using the builtin operations is very slow; therefore a parallel
    C++ extension is provided.

3.  Safe and efficient slicing.

    The managed ``AnnData`` provides a :py:func:`slice` operation that performs fast and safe
    slicing.

    To achieve this, the code tracks, for each possible data, whether it is/not safe to slice it
    along each of the axis (e.g., per-observation ``sum_of_UMIs`` data is invalidated when slicing
    some of the variables).

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
data, using arbitrary compute functions (e.g., :py:func:`get_data_per_var_per_obs`,
:py:func:`get_data_per_obs`, etc.). A set of standard common functions is also provided (e.g.,
:py:func:`get_downsample_of_var_per_obs`, :py:func:`get_mean_per_var`, etc.).
'''

from contextlib import contextmanager
from typing import (Any, Callable, Collection, Dict, Iterable, Iterator, List,
                    NamedTuple, Optional, Set, Tuple, Union)
from warnings import warn

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy.sparse as sparse  # type: ignore
from anndata import AnnData  # type: ignore
from readerwriterlock import rwlock

import metacells.utilities.computation as utc
import metacells.utilities.documentation as utd
import metacells.utilities.timing as timed

__all__ = [
    'WhichData',
    'SlicingMask',
    'ALWAYS_SAFE',
    'NEVER_SAFE',
    'SAFE_WHEN_SLICING_OBS',
    'SAFE_WHEN_SLICING_VAR',
    'safe_slicing_data',
    'safe_slicing_derived',

    'slice',

    'set_x_name',
    'track_base_indices',

    'get_data_per_var_per_obs',
    'del_data_per_var_per_obs',
    'focus_on',

    'get_log_data_per_var_per_obs',
    'get_fraction_of_var_per_obs',
    'get_fraction_of_obs_per_var',
    'get_downsample_of_var_per_obs',
    'get_downsample_of_obs_per_var',

    'get_metadata',

    'get_data_per_obs',
    'get_data_per_var',

    'get_sum_per_obs',
    'get_sum_per_var',

    'get_mean_per_obs',
    'get_mean_per_var',

    'get_sum_squared_per_obs',
    'get_sum_squared_per_var',

    'get_variance_per_obs',
    'get_variance_per_var',

    'get_max_per_obs',
    'get_max_per_var',

    'get_min_per_obs',
    'get_min_per_var',

    'get_nnz_per_obs',
    'get_nnz_per_var',
]


LOCK = rwlock.RWLockRead()


class WhichData(NamedTuple):
    '''
    Identify some data held in the ``AnnData`` object.
    '''

    #: One of ``metadata``, ``per_var_per_obs``, ``per_var``, ``per_obs``, ``per_var_per_var`` and
    #: ``per_obs_per_obs``. This determines the expected data type and, for matrix and vector data,
    #: the expected dimensions.
    per: str

    #: The name of the data, which should be unique within its "per".
    name: str


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

SAFE_SLICING: Dict[WhichData, SlicingMask] = {}

DIMENSIONAL_DATA_TYPES = (np.ndarray, sparse.spmatrix, pd.Series, pd.DataFrame)


def safe_slicing_data(data: WhichData, slicing_mask: SlicingMask) -> None:
    '''
    Specify when it is safe to slice some ``data`` using the ``slicing_mask``.

    .. note::

        This is thread-safe.
    '''
    with LOCK.gen_wlock():
        SAFE_SLICING[data] = slicing_mask


def _known_safe_slicing() -> None:
    safe_slicing_data(WhichData('metadata', 'x_name'), ALWAYS_SAFE)
    safe_slicing_data(WhichData('metadata', 'focus'), ALWAYS_SAFE)
    safe_slicing_data(WhichData('per_var', 'gene_ids'), ALWAYS_SAFE)


_known_safe_slicing()


def safe_slicing_derived(
    *,
    base_data: List[WhichData],
    derived_data: WhichData,
    slicing_mask: SlicingMask,
) -> None:
    '''
    Specify how to slice some ``derived`` data which is computed
    from some ``bases`` such data.

    Each data description is a tuple of its "per" (one of ``per_var_per_obs``, ``per_var``,
    ``per_obs``, ``per_var_per_var`` and ``per_obs_per_obs``) and its name.

    If, when slicing, any of the bases data is not preserved, then the derived data would not be
    either. Otherwise, the derived will be preserved based on the flags given here.
    '''
    with LOCK.gen_rlock():
        for base_datum in base_data:
            slicing_mask = slicing_mask & SAFE_SLICING[base_datum]

    with LOCK.gen_wlock():
        SAFE_SLICING[derived_data] = slicing_mask


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

    In general, data might become invalid when slicing (e.g., the ``sum_of_UMIs`` of genes is
    invalidated when specifying a subset of the cells). Therefore, such data will be removed from
    the result unless it was explicitly marked as preserved using :py:func:`safe_slicing_data` or
    :py:func:`safe_slicing_derived`.

    If ``invalidated_prefix`` is specified, then invalidated data will not be removed; instead it
    will be renamed with the addition of the provided prefix.

    .. note::

        Setting the prefix to the empty string would simply preserve all the invalidated data. As
        this is unsafe, it will trigger a run-time exception. If you wish to perform such an unsafe
        slicing operation, invoke the built-in ``adata[..., ...]``.
    '''
    assert invalidated_prefix != ''
    x_name = get_metadata(adata, 'x_name')
    focus = get_metadata(adata, 'focus')
    assert x_name not in adata.layers
    assert has_data(adata, 'per_var_per_obs', focus)

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

    assert adata.uns['x_name'] == x_name
    assert adata.uns['focus'] == focus
    assert x_name not in adata.layers
    assert has_data(adata, 'per_var_per_obs', focus)

    assert bdata.uns['x_name'] == x_name
    if focus == x_name or focus in bdata.layers:
        assert bdata.uns['focus'] == focus
    else:
        focus = bdata.uns['focus'] = x_name
    assert x_name not in bdata.layers
    assert has_data(bdata, 'per_var_per_obs', focus)

    return bdata


def _save_data(
    adata: AnnData,
    will_slice_obs: bool,
    will_slice_var: bool,
    invalidated_prefix: Optional[str],
) -> Dict[WhichData, Any]:
    saved_data: Dict[WhichData, Any] = {}

    for per, annotations in [('per_var_per_obs', adata.layers),
                             ('per_obs', adata.obs),
                             ('per_var', adata.var),
                             ('per_obs_per_obs', adata.obsp),
                             ('per_var_per_var', adata.varp)]:
        _save_per_data(saved_data, adata, per, annotations,
                       will_slice_obs, will_slice_var, invalidated_prefix)

    return saved_data


def _save_per_data(  # pylint: disable=too-many-locals
    saved_data: Dict[WhichData, Any],
    adata: AnnData,
    per: str,
    annotations: Union[Dict[str, Any], pd.DataFrame],
    will_slice_obs: bool,
    will_slice_var: bool,
    invalidated_prefix: Optional[str],
) -> None:
    delete_names: Set[str] = set()
    prefixed_data: Dict[str, Any] = {}

    if per != 'per_var_per_obs':
        def patch(name: str) -> None:  # pylint: disable=unused-argument
            pass
    else:
        will_slice_only_obs = will_slice_obs and not will_slice_var
        will_slice_only_var = will_slice_var and not will_slice_obs

        def patch(name: str) -> None:
            if will_slice_only_obs:
                patch_only(name, '__by_obs__')
            if will_slice_only_var:
                patch_only(name, '__by_var__')

        def patch_only(name: str, by_prefix: str) -> None:
            if not name.startswith(by_prefix):
                return

            data = adata.layers[name]
            delete_names.add(name)

            base_name = name[len(by_prefix):]
            if base_name == adata.uns['x_name']:
                with timed.step('.swap_x'):
                    adata.X = data
            else:
                base_data = adata.layers[base_name]
                saved_data[WhichData('per_var_per_obs', base_name)] = base_data

    for name, data in _items(annotations):
        which_data = WhichData(per, name)
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


def _restore_data(saved_data: Dict[WhichData, Any], adata: AnnData) -> None:
    per_annotations = dict(per_var_per_obs=adata.layers,
                           per_obs=adata.obs,
                           per_var=adata.var,
                           per_obs_per_obs=adata.obsp,
                           per_var_per_var=adata.varp)
    for which_data, data in saved_data.items():
        annotations = per_annotations[which_data.per]
        annotations[which_data.name] = data


def _prefix_data(
    saved_data: Dict[WhichData, Any],
    bdata: AnnData,
    did_slice_obs: bool,
    did_slice_var: bool,
    invalidated_prefix: str,
) -> None:
    for per, annotations in [('per_var_per_obs', bdata.layers),
                             ('per_obs', bdata.obs),
                             ('per_var', bdata.var),
                             ('per_obs_per_obs', bdata.obsp),
                             ('per_var_per_var', bdata.varp)]:
        _prefix_per_data(saved_data, per, annotations,
                         did_slice_obs, did_slice_var, invalidated_prefix)


def _prefix_per_data(
    saved_data: Dict[WhichData, Any],
    per: str,
    annotations: Dict[str, Any],
    did_slice_obs: bool,
    did_slice_var: bool,
    invalidated_prefix: str,
) -> None:
    delete_names: List[str] = []

    for name, data in _items(annotations):
        which_data = WhichData(per, name)
        action = _slice_action(which_data, did_slice_obs, did_slice_var, True)

        if action == 'preserve':
            continue
        assert action == 'prefix'

        prefixed_name = invalidated_prefix + name
        assert prefixed_name != name

        prefixed_which_data = WhichData(per, prefixed_name)
        saved_data[prefixed_which_data] = data
        delete_names.append(name)

        with LOCK.gen_wlock():
            SAFE_SLICING[prefixed_which_data] = SAFE_SLICING[which_data]

    for name in delete_names:
        del annotations[name]


def _items(annotation: Union[Dict, pd.DataFrame]) -> Iterable[Tuple[str, Any]]:
    if isinstance(annotation, pd.DataFrame):
        return annotation.iteritems()
    return annotation.items()


def _get(annotation: Union[Dict, pd.DataFrame], name: str) -> Any:
    if isinstance(annotation, pd.DataFrame):
        if name not in annotation.columns:
            return None
        return annotation[:, name]
    return annotation.get(name)


def _slice_action(
    which_data: WhichData,
    do_slice_obs: bool,
    do_slice_var: bool,
    will_prefix_invalidated: bool,
) -> str:
    if which_data.name[:2] != '__':
        is_per = False
        is_per_obs = False
        is_per_var = False
        base_name = which_data.name
    else:
        is_per = True
        is_per_obs = which_data.name.startswith('__by_obs__')
        is_per_var = which_data.name.startswith('__by_var__')
        assert is_per_obs or is_per_var
        base_name = which_data.name[10:]

    base_which_data = WhichData(which_data.per, base_name)
    with LOCK.gen_rlock():
        slicing_mask = SAFE_SLICING.get(base_which_data)

    if slicing_mask is None:
        with LOCK.gen_wlock():
            slicing_mask = SAFE_SLICING.get(base_which_data)
            if slicing_mask is None:
                unknown_sliced_data = \
                    'Slicing an unknown %s data: %s; ' \
                    'assuming it should not be preserved' \
                    % (base_which_data.per, base_which_data.name)
                warn(unknown_sliced_data)
                slicing_mask = SAFE_SLICING[base_which_data] = NEVER_SAFE

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


def set_x_name(adata: AnnData, name: str) -> None:
    '''
    Set the name of the per-variable-per-observation data that is contained in ``X``. This name is
    stored in the metadata (unstructured annotation) named ``x_name``.

    This should be called after populating the ``X`` data for the first time (e.g., by importing the
    data). It lets rest of the code to know what kind of data it holds. All the other
    per-variable-per-observation data utilities ``assert`` this was done.

    This also sets the name to be the ``focus``, which is also stored in the metadata (unstructured
    annotations).

    .. note::

        This assumes it is safe to arbitrarily slice the layer.

    .. note::

        When using the layer utilities, do not directly read or write the value of ``X``. Instead
        use :py:func:`get_data_per_var_per_obs`.

    .. todo::

        Better integration with ``AnnData`` would allow accessing and even setting ``X`` in addition
        to using :py:func:`get_data_per_var_per_obs`. Currently this isn't implemented since
        ``AnnData`` provides its own magic for handling ``X`` which is incompatible with the layers
        approach.
    '''
    X = adata.X
    assert X is not None
    assert 'x_name' not in adata.uns_keys()

    adata.uns['focus'] = adata.uns['x_name'] = name

    safe_slicing_data(WhichData('per_var_per_obs', name), ALWAYS_SAFE)


def track_base_indices(adata: AnnData, *, name: str = 'base_index') -> None:
    '''
    Create ``name`` (by default, ``base_index``) per-observation (cells) and per-variable (genes)
    data, which will be preserved when creating any :py:func:`slice` of the data to easily refer
    back to the original full data.
    '''
    adata.obs[name] = np.arange(adata.n_obs)
    safe_slicing_data(WhichData('per_obs', name), ALWAYS_SAFE)

    adata.var[name] = np.arange(adata.n_vars)
    safe_slicing_data(WhichData('per_var', name), ALWAYS_SAFE)


def has_data(
    adata: AnnData,
    per: str,
    name: str,
) -> bool:
    '''
    Test whether we have the specified data.
    '''
    if per == 'per_var_per_obs':
        return name == adata.uns['x_name'] or name in adata.layers
    if per == 'per_obs':
        return name in adata.obs
    if per == 'per_var':
        return name in adata.var
    if per == 'per_obs_per_obs':
        return name in adata.obsp
    if per == 'per_var_per_var':
        return name in adata.varp
    assert per in ['per_var_per_obs', 'per_obs', 'per_var',
                   'per_obs_per_obs', 'per_var_per_var']
    return False


@timed.call()
def get_data_per_var_per_obs(
    adata: AnnData,
    name: Optional[str] = None,
    *,
    compute: Optional[Callable[[], utc.Vector]] = None,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> utc.Matrix:
    '''
    Lookup a per-observation-per-variable matrix (data layer) in ``adata``.

    If the ``name`` is not specified, get the ``focus``.

    If the layer does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace``, store the result in ``adata`` for future reuse.

    If ``infocus`` (implies ``inplace``), also makes the result the new ``focus``.

    If ``by`` is specified, it must be one of ``obs`` (cells) or ``var`` (genes). This returns the
    data in a layout optimized for by-observation (row-major / csr) or by-variable (column-major
    / csc). If also ``inplace`` (or ``infocus``), this is cached in an additional layer whose name
    is prefixed (e.g. ``__by_obs__UMIs`` for the ``UMIs`` layer).

    .. note::

        In general names that start with ``__`` are reserved and should not be explicitly used.

    .. note::

        The original data returned by ``compute`` is always preserved under its non-prefixed name.

    .. todo::

        A better implementation of :py:func:`get_data_per_var_per_obs` would be to cache the
        layout-specific data in a private variable, but we do not own the ``AnnData`` object.
    '''
    if name is None:
        name = get_metadata(adata, 'focus')

    if infocus:
        adata.uns['focus'] = name

    assert name[:2] != '__'

    if name == adata.uns['x_name']:
        def get_data() -> utc.Matrix:
            with timed.step('.get_x'):
                timed.parameters(name=name, all='+'.join(adata.layers.keys()))
                data = adata.X
            return data
    else:
        def get_data() -> utc.Matrix:
            data = adata.layers.get(name)
            if data is not None:
                utc.freeze(data)
                return data

            if compute is None:
                assert name is not None
                raise \
                    RuntimeError('unavailable per-variable-per-observation data: '
                                 + name)
            data = compute()
            assert data.shape == adata.shape

            if inplace or infocus:
                utc.freeze(data)
                adata.layers[name] = data

            return data

    if by is None:
        return get_data()

    assert by in ['var', 'obs']
    by_name = '__by_%s__%s' % (by, name)
    axis = ['var', 'obs'].index(by)

    by_data = adata.layers.get(by_name)
    if by_data is None:
        by_data = utc.to_layout(get_data(), axis=axis)

        if inplace or infocus:
            utc.freeze(by_data)
            adata.layers[by_name] = by_data

    return by_data


def del_data_per_var_per_obs(
    adata: AnnData, name: str,
    *,
    by: Optional[str] = None,
    must_exist: bool = False,
) -> None:
    '''
    Delete a per-observation-per-variable matrix in ``adata`` by its ``name``.

    If ``by`` is specified, it must be one of ``obs`` (cells) or ``var`` (genes). This will only
    delete the cached layout-specific data. If ``by`` is not specified, both the data and any
    cached layout-specific data will be deleted.

    If ``must_exist``, will ``raise`` if the data does not currently exist.

    .. note::

        You can't delete the ``x_name`` (layer of ``X``). Deleting the ``focus`` (default data)
        changes it to become the same as the ``x_name``.

    .. todo::

        The function :py:func:`del_data_per_var_per_obs` is only required due to the need to delete
        cached layout-specific data. A better way would be to intercept ``del`` of the ``AnnData``
        ``layers`` field, but we do not own it.
    '''
    assert not name.startswith('__')
    assert name != get_metadata(adata, 'x_name')
    if name == get_metadata(adata, 'focus'):
        adata.uns['focus'] = adata.uns['x_name']

    if by is not None:
        by_name = '__by_%s__%s' % (by, name)
        if by_name in adata.layers:
            del adata.layers[by_name]
        elif must_exist:
            assert by_name in adata.layers
        return

    if must_exist:
        assert name in adata.layers
    for prefix in ['', '__by_obs__', '__by_var__']:
        prefixed_name = prefix + name
        if prefixed_name in adata.layers:
            del adata.layers[prefixed_name]


@contextmanager
def focus_on(
    accessor: Callable,
    adata: AnnData,
    *,
    of: Optional[str] = None,
    by: Optional[str] = None,
    **kwargs: Any
) -> Iterator[utc.Matrix]:
    '''
    Get some per-observation-per-variable data by invoking the ``accessor`` with ``adata``
    (and optionally, ``of`` ``by`` and any additional ``kwargs``), and make it the ``focus``
    for the duration of the ``with`` statement.

    If the original focus data is deleted inside the ``with`` statement, then when it is done, the
    ``focus`` will be set to the ``x_name``.

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

    old_focus = adata.uns['focus']
    data = accessor(adata, of=of, by=by, infocus=True, **kwargs)
    yield data
    if has_data(adata, 'per_var_per_obs', old_focus):
        adata.uns['focus'] = old_focus
    else:
        adata.uns['focus'] = adata.uns['x_name']


@timed.call()
@utd.expand_doc()
def get_log_data_per_var_per_obs(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    base: Optional[float] = None,
    normalization: float = 1,
    inplace: bool = True,
    infocus: bool = False,
    by: Optional[str] = None,
) -> utc.Matrix:
    '''
    Return a matrix with the log of some per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    If ``inplace`` (or ``infocus``), the data will be stored in
    ``log_<base>_of_<normalization>_plus_<of>`` for future reuse.

    If ``infocus`` (implies ``inplace``), also makes the result the new ``focus``.

    If ``by`` is specified, it must be one of ``obs`` (cells) or ``var`` (genes). This returns the
    data in a layout optimized for by-observation (row-major / csr) or by-variable (column-major
    / csc). If also ``inplace`` (or ``infocus``), this is cached in an additional layer whose name
    is prefixed (e.g. ``__by_obs__log_2_of_1_plus_UMIs``).

    The natural logarithm is used by default. Otherwise, the ``base`` is used.

    The ``normalization`` (default: {normalization}) is added to the count before the log is
    applied, to handle the common case of sparse data.

    .. note::

        The result is always a dense matrix, as even for sparse data, the log is rarely zero.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    to = 'log_%s_of_%s_plus_%s' % (base or 'e', normalization, of)

    if inplace or infocus:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_var_per_obs', to),
                             slicing_mask=ALWAYS_SAFE)

    @timed.call('.compute')
    def compute() -> utc.Matrix:
        assert of is not None
        matrix = get_data_per_var_per_obs(adata, of, by=by)
        return utc.log_matrix(matrix, base=base, normalization=normalization)

    return get_data_per_var_per_obs(adata, to, compute=compute,
                                    inplace=inplace, infocus=infocus, by=by)


@timed.call()
def get_fraction_of_var_per_obs(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    inplace: bool = True,
    infocus: bool = True,
    intermediate: bool = True,
    by: Optional[str] = None,
) -> utc.Matrix:
    '''
    Return a matrix containing, in each entry, the fraction the original data (e.g. UMIs) for this
    variable (gene) is in this observation (cell) out of the total for all variables (genes).

    .. note::

        This is probably the version you want: here, the sum of fraction of the genes in a cell is
        1. See :py:func:`get_fraction_of_obs_per_var` for the other way around.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    If ``inplace``, the data will be stored in ``fraction_of_var_per_obs_of_<of>`` for future reuse.

    If ``infocus`` (implies ``inplace``), also makes the result the new ``focus``.

    If ``intermediate``, also stores the ``sum_of_<of>`` per-observation (cell) data for future
    reuse.

    If ``by`` is specified, it must be one of ``obs`` (cells) or ``var`` (genes). This returns the
    data in a layout optimized for by-observation (row-major / csr) or by-variable (column-major
    / csc). If also ``inplace`` (or ``infocus``), this is cached in an additional layer whose name
    is prefixed (e.g. ``__by_obs__fraction_of_var_per_obs_of_UMIs``).

    .. note::

        This assumes all the data values are non-negative.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    to = 'fraction_of_var_per_obs_of_' + of

    if inplace or infocus:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_var_per_obs', to),
                             slicing_mask=SAFE_WHEN_SLICING_OBS)

    @timed.call('.compute')
    def compute() -> utc.Matrix:
        assert of is not None
        matrix = get_data_per_var_per_obs(adata, of, by=by)
        total_per_obs = get_sum_per_obs(adata, of, inplace=intermediate)
        return matrix / total_per_obs[:, None]

    return get_data_per_var_per_obs(adata, to, compute=compute,
                                    inplace=inplace, infocus=infocus, by=by)


@timed.call()
def get_fraction_of_obs_per_var(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    inplace: bool = True,
    infocus: bool = True,
    intermediate: bool = True,
    by: Optional[str] = None,
) -> utc.Matrix:
    '''
    Return a matrix containing, in each entry, the fraction the original data (e.g. UMIs) for this
    variable (gene) is in this observation (cell) out of the total for all observations (cells).

    .. note::

        This is probably not the version you want: here, the sum of fractions of the cells in each
        gene is 1. See :py:func:`get_fraction_of_var_per_obs` for the other way around.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    If ``inplace``, the data will be stored in ``fraction_of_obs_per_var_of_<of>`` for future reuse.

    If ``infocus`` (implies ``inplace``), also makes the result the new ``focus``.

    If ``intermediate``, also stores the ``sum_of_<of>`` per-variable (gene) data for future reuse.

    If ``by`` is specified, it must be one of ``obs`` (cells) or ``var`` (genes). This returns the
    data in a layout optimized for by-observation (row-major / csr) or by-variable (column-major
    / csc). If also ``inplace`` (or ``infocus``), this is cached in an additional layer whose name
    is prefixed (e.g. ``__by_obs__fraction_of_obs_per_var_of_UMIs``).

    .. note::

        This assumes all the data values are non-negative.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    to = 'fraction_of_obs_per_var_of_' + of

    if inplace or infocus:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_var_per_obs', to),
                             slicing_mask=SAFE_WHEN_SLICING_VAR)

    @timed.call('.compute')
    def compute() -> utc.Vector:
        assert of is not None
        matrix = get_data_per_var_per_obs(adata, of, by=by)
        total_per_var = get_sum_per_var(adata, of, inplace=intermediate)
        return matrix / total_per_var[None, :]

    return get_data_per_var_per_obs(adata, to, compute=compute,
                                    inplace=inplace, infocus=infocus, by=by)


@timed.call()
def get_downsample_of_var_per_obs(
    adata: AnnData,
    *,
    samples: int,
    random_seed: int = 0,
    of: Optional[str] = None,
    inplace: bool = True,
    infocus: bool = True,
    by: Optional[str] = None,
) -> utc.Matrix:
    '''
    Return a matrix containing, for each observation (cell), downsampled data
    for each variable (gene), such that the total sum would be no more
    than ``samples``.

    .. note::

        This is probably the version you want: here, the sum of the genes in a cell will be (at
        most) ``samples``. See :py:func:`get_downsample_of_obs_per_var` for the other way around.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    If ``inplace``, the data will be stored in ``downsample_<samples>_var_per_obs_of_<of>`` for
    future reuse.

    If ``infocus`` (implies ``inplace``), also makes the result the new ``focus``.

    If ``by`` is specified, it must be one of ``obs`` (cells) or ``var`` (genes). This returns the
    data in a layout optimized for by-observation (row-major / csr) or by-variable (column-major
    / csc). If also ``inplace`` (or ``infocus``), this is cached in an additional layer whose name
    is prefixed (e.g. ``__by_obs__downsample_<samples>_var_per_obs_of_<of>``).

    A ``random_seed`` can be provided to make the operation replicable.

    .. note::

        This assumes all the data values are non-negative integer values (even if the ``dtype`` is a
        floating-point type).
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    to = 'downsample_%s_var_per_obs_of_%s' % (samples, of)

    if inplace or infocus:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_var_per_obs', to),
                             slicing_mask=SAFE_WHEN_SLICING_OBS)

    @timed.call('.compute')
    def compute() -> utc.Matrix:
        assert of is not None
        matrix = get_data_per_var_per_obs(adata, of, by=by)
        return utc.downsample_matrix(matrix, axis=1, samples=samples, random_seed=random_seed)

    return get_data_per_var_per_obs(adata, to, compute=compute,
                                    inplace=inplace, infocus=infocus, by=by)


@timed.call()
def get_downsample_of_obs_per_var(
    adata: AnnData,
    *,
    samples: int,
    random_seed: int = 0,
    of: Optional[str] = None,
    to: Optional[str] = None,
    inplace: bool = True,
    infocus: bool = True,
    by: Optional[str] = None,
) -> utc.Matrix:
    '''
    Return a matrix containing, for each variable (gene), downsampled data
    for each observation (cell), such that the total sum would be no more
    than ``samples``.

    .. note::

        This is probably not the version you want: here, the sum of the cells in a gene will be (at
        most) ``samples``. See :py:func:`get_downsample_of_var_per_obs` for the other way around.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    If ``inplace``, the data will be stored in ``downsample_<samples>_obs_per_var_of_<of>`` for
    future reuse.

    If ``infocus`` (implies ``inplace``), also makes the result the new ``focus``.

    If ``by`` is specified, it must be one of ``obs`` (cells) or ``var`` (genes). This returns the
    data in a layout optimized for by-observation (row-major / csr) or by-variable (column-major
    / csc). If also ``inplace`` (or ``infocus``), this is cached in an additional layer whose name
    is prefixed (e.g. ``__by_obs__downsample_800_obs_per_var_of_UMIs``).

    A ``random_seed`` can be provided to make the operation replicable.

    .. note::

        This assumes all the data values are non-negative integer values (even if the ``dtype`` is a
        floating-point type).
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    to = 'downsample_%s_obs_per_var_of_%s' % (samples, of)

    if inplace or infocus:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_var_per_obs', to),
                             slicing_mask=SAFE_WHEN_SLICING_VAR)

    @timed.call('.compute')
    def compute() -> utc.Matrix:
        assert of is not None
        matrix = get_data_per_var_per_obs(adata, of, by=by)
        return utc.downsample_matrix(matrix, axis=0, samples=samples, random_seed=random_seed)

    return get_data_per_var_per_obs(adata, to, compute=compute,
                                    inplace=inplace, infocus=infocus, by=by)


def get_metadata(
    adata: AnnData,
    name: str,
    compute: Optional[Callable[[], utc.Vector]] = None,
    *,
    inplace: bool = True,
) -> Any:
    '''
    Lookup metadata (unstructured annotation) in ``adata``.

    If the metadata does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace``, store the result in ``adata``.

    .. note::

        The caller is responsible for specifying the slicing behavior of the data.
    '''
    data = adata.uns.get(name)
    if data is not None:
        return data

    if compute is None:
        raise RuntimeError('unavailable metadata: ' + name)

    data = compute()
    assert data is not None

    if inplace:
        adata.uns[name] = data

    return data


def get_data_per_obs(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utc.Vector]] = None,
    inplace: bool = True,
) -> utc.Vector:
    '''
    Lookup per-observation (cell) data in ``adata``.

    If the data does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace``, store the result in ``adata`` for future reuse.

    .. note::

        The caller is responsible for specifying the slicing behavior of the data.
    '''
    data = adata.obs.get(name)
    if data is not None:
        return data

    if compute is None:
        raise RuntimeError('unavailable per-observation data: ' + name)

    data = compute()
    assert data is not None
    assert data.shape == (adata.n_obs,)

    if inplace:
        utc.freeze(data)
        adata.obs[name] = data

    return data


def get_data_per_var(
    adata: AnnData,
    name: str,
    *,
    compute: Optional[Callable[[], utc.Vector]] = None,
    inplace: bool = True,
) -> utc.Vector:
    '''
    Lookup per-variable (gene) data in ``adata``.

    If the data does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace``, store the result in ``adata`` for future reuse.

    .. note::

        The caller is responsible for specifying the slicing behavior of the data.
    '''
    data = adata.var.get(name)
    if data is not None:
        return data

    if compute is None:
        raise RuntimeError('unavailable per-variable data: ' + name)

    data = compute()
    assert data is not None
    assert data.shape == (adata.n_vars,)

    if inplace:
        utc.freeze(data)
        adata.var[name] = data

    return data


@timed.call()
def get_sum_per_obs(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> utc.Vector:
    '''
    Return the sum of the values per-observation (cell) of some per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    Use the ``sum_of_<of>`` per-observation (cell) data if it exists. Otherwise, compute it, and if
    ``inplace`` store it for future reuse.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    @timed.call('.compute')
    def compute() -> utc.Vector:
        matrix = get_data_per_var_per_obs(adata, of, by='obs')
        return utc.sum_matrix(matrix, axis=1)

    name = 'sum_of_' + of
    if inplace:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_obs', name),
                             slicing_mask=SAFE_WHEN_SLICING_OBS)

    return get_data_per_obs(adata, name, compute=compute, inplace=inplace)


@timed.call()
def get_sum_per_var(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> utc.Vector:
    '''
    Return the sum of the values per-variable (gene) of some per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    Use the ``sum_of_<of>`` per-variable (gene) data if it exists. Otherwise, compute it, and if
    ``inplace`` store it for future reuse.

    If ``inplace``, store the result in ``adata`` for future reuse.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    @timed.call('.compute')
    def compute() -> utc.Vector:
        matrix = get_data_per_var_per_obs(adata, of, by='var')
        return utc.sum_matrix(matrix, axis=0)

    name = 'sum_of_' + of
    if inplace:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_var', name),
                             slicing_mask=SAFE_WHEN_SLICING_VAR)

    return get_data_per_var(adata, name, compute=compute, inplace=inplace)


@timed.call()
def get_mean_per_obs(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> utc.Vector:
    '''
    Return the mean of the values per-observation (cell) of some per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    Use the ``sum_of_<of>`` per-observation (cell) data if it exists. Otherwise, compute it, and if
    ``inplace`` store it for future reuse.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    sum_per_obs = get_sum_per_obs(adata, of, inplace=inplace)
    with timed.step('.compute'):
        return sum_per_obs / adata.n_vars


@timed.call()
def get_mean_per_var(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> utc.Vector:
    '''
    Return the mean of the values per-variable (gene) of some per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    Use the ``sum_of_<of>`` per-variable (gene) data if it exists. Otherwise, compute it, and if
    ``inplace`` store it for future reuse.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    sum_per_var = get_sum_per_var(adata, of, inplace=inplace)
    with timed.step('.compute'):
        return sum_per_var / adata.n_obs


@timed.call()
def get_sum_squared_per_obs(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> utc.Vector:
    '''
    Return the sum of the squared values per-observation (cell) of some per-variable-per-observation
    data.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    Use the ``sum_squared_of_<of>`` per-observation (cell) data if it exists. Otherwise, compute it,
    and if ``inplace`` store it for future reuse.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    @timed.call('.compute')
    def compute() -> utc.Vector:
        matrix = get_data_per_var_per_obs(adata, of, by='obs')
        return utc.sum_squared_matrix(matrix, axis=1)

    name = 'sum_squared_of_' + of
    if inplace:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_obs', name),
                             slicing_mask=SAFE_WHEN_SLICING_OBS)

    return get_data_per_obs(adata, name, compute=compute, inplace=inplace)


@timed.call()
def get_sum_squared_per_var(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> utc.Vector:
    '''
    Return the sum of the squared values per-variable (gene) of some per-variable-per-observation
    data.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    Use the ``sum_squared_of_<of>`` per-variable (gene) data if it exists. Otherwise, compute it,
    and if ``inplace`` store it for future reuse.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    @timed.call('.compute')
    def compute() -> utc.Vector:
        matrix = get_data_per_var_per_obs(adata, of, by='var')
        return utc.sum_squared_matrix(matrix, axis=0)

    name = 'sum_squared_of_' + of
    if inplace:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_var', name),
                             slicing_mask=SAFE_WHEN_SLICING_VAR)

    return get_data_per_var(adata, name, compute=compute, inplace=inplace)


@timed.call()
def get_variance_per_obs(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
    intermediate: bool = True,
) -> utc.Vector:
    '''
    Return the variance of the values per-observation (cell) of some per-variable-per-observation
    data.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    Use the ``variance_of_<of>`` per-observation (cell) data if it exists. Otherwise, compute it,
    and if ``inplace`` store it for future reuse.

    If ``intermediate``, also store the intermediate per-observation ``sum_of_<of>`` and
    ``sum_squared_of_<of>`` data for future reuse.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    @timed.call('.compute')
    def compute() -> utc.Vector:
        sum_per_obs = get_sum_per_obs(adata, of, inplace=intermediate)
        sum_squared_per_obs = \
            get_sum_squared_per_obs(adata, of, inplace=intermediate)
        result = np.square(sum_per_obs)
        result /= -adata.n_vars
        result += sum_squared_per_obs
        result /= adata.n_vars
        return result

    name = 'variance_of_' + of
    if inplace:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_obs', name),
                             slicing_mask=SAFE_WHEN_SLICING_OBS)

    return get_data_per_obs(adata, name, compute=compute, inplace=inplace)


@timed.call()
def get_variance_per_var(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
    intermediate: bool = True,
) -> utc.Vector:
    '''
    Return the variance of the values per-variable (gene) of some per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    Use the ``variance_of_<of>`` per-variable (gene) data if it exists. Otherwise, compute it,
    and if ``inplace`` store it for future reuse.

    If ``intermediate``, also store the intermediate per-variable ``sum_of_<of>`` and
    ``sum_squared_of_<of>`` data for future reuse.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    @timed.call('.compute')
    def compute() -> utc.Vector:
        sum_per_var = get_sum_per_var(adata, of, inplace=intermediate)
        sum_squared_per_var = \
            get_sum_squared_per_var(adata, of, inplace=intermediate)
        result = np.square(sum_per_var)
        result /= -adata.n_obs
        result += sum_squared_per_var
        result /= adata.n_obs
        return result

    name = 'variance_of_' + of
    if inplace:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_var', name),
                             slicing_mask=SAFE_WHEN_SLICING_VAR)

    return get_data_per_var(adata, name, compute=compute, inplace=inplace)


@timed.call()
def get_max_per_obs(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> utc.Vector:
    '''
    Return the maximal value per-observation (cell) of some per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    Use the ``max_of_<of>`` per-observation (cell) data if it exists. Otherwise, compute it, and if
    ``inplace`` store it for future reuse.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    @timed.call('.compute')
    def compute() -> utc.Vector:
        matrix = get_data_per_var_per_obs(adata, of, by='obs')
        return utc.max_matrix(matrix, axis=1)

    name = 'max_of_' + of
    if inplace:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_obs', name),
                             slicing_mask=SAFE_WHEN_SLICING_OBS)

    return get_data_per_obs(adata, name, compute=compute, inplace=inplace)


@timed.call()
def get_max_per_var(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> utc.Vector:
    '''
    Return the maximal value per-variable (gene) of some per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    Use the ``max_of_<of>`` per-variable (gene) data if it exists. Otherwise, compute it, and if
    ``inplace`` store it for future reuse.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    @timed.call('.compute')
    def compute() -> utc.Vector:
        matrix = get_data_per_var_per_obs(adata, of, by='var')
        return utc.max_matrix(matrix, axis=0)

    name = 'max_of_' + of
    if inplace:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_var', name),
                             slicing_mask=SAFE_WHEN_SLICING_VAR)

    return get_data_per_var(adata, name, compute=compute, inplace=inplace)


@timed.call()
def get_min_per_obs(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> utc.Vector:
    '''
    Return the minimal value per-observation (cell) of some per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    Use the ``min_of_<of>`` per-observation (cell) data if it exists. Otherwise, compute it, and if
    ``inplace`` store it for future reuse.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    @timed.call('.compute')
    def compute() -> utc.Vector:
        matrix = get_data_per_var_per_obs(adata, of, by='obs')
        return utc.min_matrix(matrix, axis=1)

    name = 'min_of_' + of
    if inplace:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_obs', name),
                             slicing_mask=SAFE_WHEN_SLICING_OBS)

    return get_data_per_obs(adata, name, compute=compute, inplace=inplace)


@timed.call()
def get_min_per_var(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> utc.Vector:
    '''
    Return the minimal value per-variable (gene) of some per-variable-per-observation data.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    Use the ``min_of_<of>`` per-variable (gene) data if it exists. Otherwise, compute it, and if
    ``inplace`` store it for future reuse.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    @timed.call('.compute')
    def compute() -> utc.Vector:
        matrix = get_data_per_var_per_obs(adata, of, by='var')
        return utc.min_matrix(matrix, axis=0)

    name = 'min_of_' + of
    if inplace:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_var', name),
                             slicing_mask=SAFE_WHEN_SLICING_VAR)

    return get_data_per_var(adata, name, compute=compute, inplace=inplace)


@timed.call()
def get_nnz_per_obs(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> utc.Vector:
    '''
    Return the number of non-zero values per-observation (cell) of some per-variable-per-observation
    data.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    Use the ``nnz_of_<of>`` per-observation (cell) data if it exists. Otherwise, compute it, and if
    ``inplace`` store it for future reuse.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    @timed.call('.compute')
    def compute() -> utc.Vector:
        matrix = get_data_per_var_per_obs(adata, of, by='obs')
        return utc.nnz_matrix(matrix, axis=1)

    name = 'nnz_of_' + of
    if inplace:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_obs', name),
                             slicing_mask=SAFE_WHEN_SLICING_OBS)

    return get_data_per_obs(adata, name, compute=compute, inplace=inplace)


@timed.call()
def get_nnz_per_var(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    inplace: bool = True,
) -> utc.Vector:
    '''
    Return the number of non-zero values per-variable (gene) of some per-variable-per-observation
    data.

    If ``of`` is specified, this specific data is used. Otherwise, the ``focus`` is used.

    Use the ``nnz_of_<of>`` per-variable (gene) data if it exists. Otherwise, compute it, and if
    ``inplace`` store it for future reuse.
    '''
    if of is None:
        of = get_metadata(adata, 'focus')
        assert of is not None

    @timed.call('.compute')
    def compute() -> utc.Vector:
        matrix = get_data_per_var_per_obs(adata, of, by='var')
        return utc.nnz_matrix(matrix, axis=0)

    name = 'nnz_of_' + of
    if inplace:
        safe_slicing_derived(base_data=[WhichData('per_var_per_obs', of)],
                             derived_data=WhichData('per_var', name),
                             slicing_mask=SAFE_WHEN_SLICING_VAR)

    return get_data_per_var(adata, name, compute=compute, inplace=inplace)
