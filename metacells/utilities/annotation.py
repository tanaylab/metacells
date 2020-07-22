'''
Utilities for dealing with data annotations.

This tries to deal safely with data layers, observation, variable and unstructured annotations,
especially in the presence to slicing the data.

.. todo::

    Also deal with multi-dimensional ``obsm`` and ``varm`` annotations.
'''

from typing import (Any, Callable, Collection, Dict, NamedTuple, Optional,
                    Tuple, Union)
from warnings import warn

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData  # type: ignore
from readerwriterlock import rwlock
from scipy import sparse  # type: ignore

import metacells.utilities.computation as utc
import metacells.utilities.documentation as utd

__all__ = [
    'slice',
    'slicing_data_layer',
    'slicing_obs_annotation',
    'slicing_var_annotation',
    'slicing_uns_annotation',
    'SlicingContext',

    'annotate_as_base',

    'get_obs_count',
    'get_var_count',

    'declare_focus_of_data',

    'get_data_layer',
    'get_log_data',

    'get_annotation_of_obs',
    'get_sum_per_obs',
    'get_nnz_per_obs',
    'get_max_per_obs',

    'get_annotation_of_var',
    'get_sum_per_var',
    'get_nnz_per_var',
    'get_max_per_var',

    'get_annotation_of_data',
]


SPARSE_LAYOUTS = ['bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil']


class SlicingContext(NamedTuple):
    '''
    Context of slicing operation.

    Used as a parameter for function fixing the value of affected annotations.
    '''
    full_adata: AnnData  #: The data before the slicing.
    sliced_adata: AnnData  #: The data sliced using the builtin operation.
    did_slice_obs: bool  #: Whether observations were sliced.
    did_slice_var: bool  #: Whether variables were sliced.
    #: If observations were sliced, the indices or mask of the kept ones.
    slice_obs: Optional[Collection]
    #: If variables were sliced, the indices or mask of the kept ones.
    slice_var: Optional[Collection]


LOCK = rwlock.RWLockRead()

WARN_WHEN_SLICING_UNKNOWN_ANNOTATIONS: bool = True

SAFE_SLICING_OBS_ANNOTATIONS: Dict[str,
                                   Tuple[bool, bool,
                                         Optional[Callable[[SlicingContext, str], None]]]] = {}

SAFE_SLICING_VAR_ANNOTATIONS: Dict[str,
                                   Tuple[bool, bool,
                                         Optional[Callable[[SlicingContext, str], None]]]] = {}

SAFE_SLICING_UNS_ANNOTATIONS: Dict[str,
                                   Tuple[bool, bool,
                                         Optional[Callable[[SlicingContext, str], None]]]] = {}

SAFE_SLICING_DATA_LAYERS: Dict[str,
                               Tuple[bool, bool,
                                     Optional[Callable[[SlicingContext, str], None]]]] = {}


def slicing_data_layer(
    name: str,
    *,
    preserve_when_slicing_obs: bool = False,
    preserve_when_slicing_var: bool = False,
    fix_sliced_layer: Optional[Callable[[SlicingContext, str], None]] = None,
) -> None:
    '''
    Specify whether the named data layer should be preserved when slicing either the observations
    (cells) or variables (genes).

    A data layer is always preserved if it is the focus layer (the value of ``X``). Otherwise,
    when the data is sliced, it may become invalidated, and will not be preserved.

    If ``fix_sliced_layer`` is provided, when a slicing occurs, it will be invoked as
    ``fix_sliced_layer(slicing_context: SlicingContext, layer_name: str)`` to allow for fixing the
    preserved sliced data layer values.
    '''
    with LOCK.gen_wlock():
        SAFE_SLICING_DATA_LAYERS[name] = (preserve_when_slicing_obs,
                                          preserve_when_slicing_var,
                                          fix_sliced_layer)


def _slicing_known_data_layers() -> None:
    slicing_data_layer('UMIs',
                       preserve_when_slicing_obs=True,
                       preserve_when_slicing_var=True)
    slicing_data_layer('log_UMIs',
                       preserve_when_slicing_obs=True,
                       preserve_when_slicing_var=True)
    slicing_data_layer('downsampled_UMIs',
                       preserve_when_slicing_obs=True,
                       preserve_when_slicing_var=False)


def slicing_obs_annotation(
    name: str,
    *,
    preserve_when_slicing_obs: bool = False,
    preserve_when_slicing_var: bool = False,
    fix_sliced_annotation: Optional[Callable[[SlicingContext, str], None]] = None,
) -> None:
    '''
    Specify whether the named per-observation (cell) annotation should be preserved when slicing
    either the observations (cells) or variables (genes).

    If ``fix_sliced_annotation`` is provided, when a slicing occurs, it will be invoked as
    ``fix_sliced_annotation(slicing_context: SlicingContext, annotation_name: str)`` to allow for
    fixing the preserved sliced annotation values.
    '''
    SAFE_SLICING_OBS_ANNOTATIONS[name] = (preserve_when_slicing_obs,
                                          preserve_when_slicing_var,
                                          fix_sliced_annotation)


def _slicing_known_obs_annotations() -> None:
    slicing_obs_annotation('base_index',
                           preserve_when_slicing_obs=True,
                           preserve_when_slicing_var=True)
    slicing_obs_annotation('sum_UMIs',
                           preserve_when_slicing_obs=True,
                           preserve_when_slicing_var=False)
    slicing_obs_annotation('nnz_UMIs',
                           preserve_when_slicing_obs=True,
                           preserve_when_slicing_var=False)
    slicing_obs_annotation('max_UMIs',
                           preserve_when_slicing_obs=True,
                           preserve_when_slicing_var=False)


def slicing_var_annotation(
    name: str,
    *,
    preserve_when_slicing_obs: bool = False,
    preserve_when_slicing_var: bool = False,
    fix_sliced_annotation: Optional[Callable[[SlicingContext, str], None]] = None,
) -> None:
    '''
    Specify whether the named per-variable (gene) annotation should be preserved when slicing
    either the observations (cells) or variables (genes).

    If ``fix_sliced_annotation`` is provided, when a slicing occurs, it will be invoked as
    ``fix_sliced_annotation(slicing_context: SlicingContext, annotation_name: str)`` to allow for
    fixing the preserved sliced annotation values.
    '''
    SAFE_SLICING_VAR_ANNOTATIONS[name] = (preserve_when_slicing_obs,
                                          preserve_when_slicing_var,
                                          fix_sliced_annotation)


def _slicing_known_var_annotations() -> None:
    slicing_var_annotation('gene_ids',
                           preserve_when_slicing_obs=True,
                           preserve_when_slicing_var=True)
    slicing_var_annotation('base_index',
                           preserve_when_slicing_obs=True,
                           preserve_when_slicing_var=True)
    slicing_var_annotation('sum_UMIs',
                           preserve_when_slicing_obs=False,
                           preserve_when_slicing_var=True)
    slicing_obs_annotation('nnz_UMIs',
                           preserve_when_slicing_obs=False,
                           preserve_when_slicing_var=True)
    slicing_obs_annotation('max_UMIs',
                           preserve_when_slicing_obs=False,
                           preserve_when_slicing_var=True)


def slicing_uns_annotation(
    name: str,
    *,
    preserve_when_slicing_obs: bool = False,
    preserve_when_slicing_var: bool = False,
    fix_sliced_annotation: Optional[Callable[[SlicingContext, str], None]] = None,
) -> None:
    '''
    Specify whether the named unstructured annotation should be preserved when slicing either the
    observations (cells) or variables (genes).

    If ``fix_sliced_annotation`` is provided, when a slicing occurs, it will be invoked as
    ``fix_sliced_annotation(slicing_context: SlicingContext, annotation_name: str)`` to allow for
    fixing the preserved sliced annotation values.
    '''
    with LOCK.gen_wlock():
        SAFE_SLICING_UNS_ANNOTATIONS[name] = (preserve_when_slicing_obs,
                                              preserve_when_slicing_var,
                                              fix_sliced_annotation)


def _slicing_known_uns_annotations() -> None:
    slicing_uns_annotation('layer',
                           preserve_when_slicing_obs=True,
                           preserve_when_slicing_var=True)


def _slicing_known_annotations() -> None:
    _slicing_known_data_layers()
    _slicing_known_obs_annotations()
    _slicing_known_var_annotations()
    _slicing_known_uns_annotations()


_slicing_known_annotations()


def slice(  # pylint: disable=redefined-builtin
    adata: AnnData,
    *,
    cells: Optional[Collection] = None,
    genes: Optional[Collection] = None,
    invalidated_annotations_prefix: Optional[str] = None,
) -> AnnData:
    '''
    Return new annotated data which includes a subset of the full ``adata``.

    If ``cells`` and/or ``genes`` are specified, they should include either a boolean
    mask or a collection of indices to include in the data slice. In the case of an indices array,
    it is assumed the indices are unique and sorted, that is that their effect is similar to a mask.

    In general, annotations data might become invalid when slicing (e.g., the ``sum_UMIs`` of
    genes is invalidated when specifying a subset of the cells). Therefore, annotations will be
    removed from the result unless they were explicitly marked as preserved using
    ``slicing_obs_annotation``, ``slicing_var_annotation``, and/or ``slicing_uns_annotation``.

    If ``invalidated_annotations_prefix`` is specified, then unpreserved annotations will not be
    removed; instead they will be renamed with the addition of the prefix.

    .. note::

        Setting the prefix to the empty string would preserves all the annotations including the
        invalidated ones. As this is unsafe, it will trigger a run-time exception. If you wish to
        perform such an unsafe slicing operation, invoke the built-in ``adata[..., ...]``.
    '''
    assert invalidated_annotations_prefix != ''

    if cells is None:
        cells = range(get_obs_count(adata))
    if genes is None:
        genes = range(get_var_count(adata))

    bdata = adata[cells, genes]

    did_slice_obs = get_obs_count(bdata) != get_obs_count(adata)
    did_slice_var = get_var_count(bdata) != get_var_count(adata)

    slicing_context = SlicingContext(full_adata=adata, sliced_adata=bdata,
                                     did_slice_obs=did_slice_obs, did_slice_var=did_slice_var,
                                     slice_obs=cells, slice_var=genes)

    if did_slice_obs or did_slice_var:
        _filter_annotations('obs', bdata.obs, slicing_context,
                            SAFE_SLICING_OBS_ANNOTATIONS, invalidated_annotations_prefix)
        _filter_annotations('var', bdata.var, slicing_context,
                            SAFE_SLICING_VAR_ANNOTATIONS, invalidated_annotations_prefix)
        _filter_annotations('uns', bdata.uns, slicing_context,
                            SAFE_SLICING_UNS_ANNOTATIONS, invalidated_annotations_prefix)
        _filter_layers(slicing_context, invalidated_annotations_prefix)

    return bdata


def _filter_annotations(  # pylint: disable=too-many-locals
    kind: str,
    new_annotations: Union[pd.DataFrame, Dict[str, Any]],
    slicing_context: SlicingContext,
    slicing_annotations: Dict[str,
                              Tuple[bool, bool,
                                    Optional[Callable[[SlicingContext, str], None]]]],
    invalidated_annotations_prefix: Optional[str],
) -> None:
    annotation_names = list(new_annotations.keys())
    for name in annotation_names:
        (preserve_when_slicing_obs, preserve_when_slicing_var, fix_sliced_annotation) = \
            _get_slicing_annotation(kind, 'annotation',
                                    slicing_annotations, name)

        preserve = (not slicing_context.did_slice_obs or preserve_when_slicing_obs) \
            and (not slicing_context.did_slice_var or preserve_when_slicing_var)

        if preserve:
            if fix_sliced_annotation is not None:
                fix_sliced_annotation(slicing_context, name)
            continue

        if invalidated_annotations_prefix is not None:
            new_name = invalidated_annotations_prefix + name
            assert new_name != name
            new_annotations[new_name] = new_annotations[name]
            _clone_slicing_annotation(slicing_annotations, name, new_name)

        del new_annotations[name]


def _filter_layers(
    slicing_context: SlicingContext,
    invalidated_annotations_prefix: Optional[str],
) -> None:
    layer_names = list(slicing_context.sliced_adata.layers.keys())
    for name in layer_names:
        if name[0] == '_' and name[4] == '_' and name[1:4] in SPARSE_LAYOUTS:
            base_name = name[5:]
        else:
            base_name = name

        (preserve_when_slicing_obs, preserve_when_slicing_var, fix_sliced_layer) = \
            _get_slicing_annotation('data', 'layer',
                                    SAFE_SLICING_DATA_LAYERS, base_name)

        preserve = (not slicing_context.did_slice_obs or preserve_when_slicing_obs) \
            and (not slicing_context.did_slice_var or preserve_when_slicing_var)

        if preserve:
            if fix_sliced_layer is not None:
                if base_name == name:
                    fix_sliced_layer(slicing_context, name)
                else:
                    del slicing_context.sliced_adata.layers[name]
            continue

        if invalidated_annotations_prefix is not None and name == base_name:
            new_name = invalidated_annotations_prefix + name
            assert new_name != name
            slicing_context.sliced_adata.layers[new_name] = \
                slicing_context.sliced_adata.layers[name]
            _clone_slicing_annotation(SAFE_SLICING_DATA_LAYERS, name, new_name)

        del slicing_context.sliced_adata.layers[name]


def _get_slicing_annotation(
    kind: str,
    what: str,
    slicing_annotations: Dict[str,
                              Tuple[bool, bool,
                                    Optional[Callable[[SlicingContext, str], None]]]],
    name: str
) -> Tuple[bool, bool, Optional[Callable[[SlicingContext, str], None]]]:
    with LOCK.gen_rlock():
        slicing_annotation = slicing_annotations.get(name)

    if slicing_annotation is None:
        with LOCK.gen_wlock():
            slicing_annotation = slicing_annotations.get(name)
            if slicing_annotation is None:
                unknown_sliced_annotation = \
                    'Slicing an unknown {kind} {what}: {name}; ' \
                    'assuming it should not be preserved' \
                    .format(kind=kind, what=what, name=name)
                warn(unknown_sliced_annotation)
                slicing_annotation = (False, False, None)
                slicing_annotations[name] = slicing_annotation

    return slicing_annotation


def _clone_slicing_annotation(
    slicing_annotations: Dict[str,
                              Tuple[bool, bool,
                                    Optional[Callable[[SlicingContext, str], None]]]],
    old_name: str,
    new_name: str,
) -> None:
    with LOCK.gen_rlock():
        if new_name in slicing_annotations:
            return

    with LOCK.gen_wlock():
        if new_name in slicing_annotations:
            return
        slicing_annotations[new_name] = slicing_annotations[old_name]


def annotate_as_base(adata: AnnData, *, name: str = 'base_index') -> None:
    '''
    Create ``name`` (by default, ``base_index``) per-observation (cells) and per-variable (genes)
    annotations, which will be preserved when creating a ``slice`` of the data to easily refer back
    to the original full data.
    '''
    adata.obs[name] = np.arange(get_obs_count(adata))
    adata.var[name] = np.arange(get_var_count(adata))


def get_obs_count(adata: AnnData) -> int:
    '''
    Return the number of observations (cells) in the ``adata``.
    '''
    return adata.shape[0]


def get_var_count(adata: AnnData) -> int:
    '''
    Return the number of variables (genes) per observation (cell) in the ``adata``.
    '''
    return adata.shape[1]


def declare_focus_of_data(adata: AnnData, name: str) -> None:
    '''
    Declare the current layer of the data that is contained in ``X``.

    This should be called after populating the ``X`` data for the first time (e.g., by importing the
    data). It lets rest of the code to know what kind of data it holds. All the other layer
    utilities ``assert`` this was done.

    .. note::

        When using the layer utilities, do not replace the value of ``X`` by writinf ``adata.X =
        ...``. Instead use ``get_data_layer(adata, ...)``.
    '''
    assert adata.X is not None
    assert 'layer' not in adata.uns_keys()
    adata.uns['layer'] = name
    adata.layers[name] = adata.X


@utd.expand_doc(sparse_layouts=', '.join(['``%s``' % layout for layout in SPARSE_LAYOUTS]))
def get_data_layer(  # pylint: disable=too-many-branches
    adata: AnnData,
    name: str,
    compute: Optional[Callable[[], utc.Vector]] = None,
    *,
    inplace: bool = False,
    infocus: bool = False,
    instead: bool = False,
    layout: Optional[str] = None,
) -> utc.Matrix:
    '''
    Lookup a per-obsevration-per-variable matrix in ``adata``.

    If the layer does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace``, store the annotation in ``adata``.

    If ``infocus`` (implies ``inplace``), also make it the current ``X`` of the data.

    If ``instead`` (implies ``infocus``), do not preserve the current ``X`` of the data, that is,
    remove it from the layers.

    If ``inplace``, and ``layout`` is specified (valid values: {sparse_layouts}), and the data
    returned by ``compute`` is sparse, then the specific layout data is stored as an additional
    layer whose name is prefixed with the layout (e.g., ``_csr_UMIs`` instead of ``UMIs``).

    .. note::

        This ``assert``s that the current ``X``, if not ``None``, was declared to be a layer of the
        data so it isn't lost even if ``infocus`` is specified.

    .. note::

        This never sets the focus layer (that is, ``X``) to a specific-layout data. The focus layer
        is always set to whatever was returned by ``compute`` under the unprefixed name.
    '''
    if name in adata.layers.keys():
        data = adata.layers[name]
    else:
        if compute is None:
            raise RuntimeError('unavailable layer data: ' + name)
        data = compute()
        assert data.shape == adata.shape

    if inplace or infocus or instead:
        adata.layers[name] = data

        if infocus or instead:
            if data.X is not None:
                old_name = get_annotation_of_data(adata, 'layer')
                assert id(adata.layers[old_name]) == id(adata.X)
                if instead:
                    del adata.layers[old_name]
            adata.uns['layer'] = name
            adata.X = data

    if layout is None or not sparse.issparse(data):
        return data
    assert layout in SPARSE_LAYOUTS

    layout_name = '_%s_%s' % (layout, name)
    if layout_name in adata.layers.keys():
        return adata.layers[layout_name]

    data = getattr(data, 'to' + layout)()
    adata.layers[layout_name] = data
    return data


@utd.expand_doc()
def get_log_data(
    adata: AnnData,
    *,
    of_layer: Optional[str] = None,
    to_layer: Optional[str] = None,
    base: Optional[float] = None,
    normalization: float = 1,
    inplace: bool = False,
    infocus: bool = False,
    instead: bool = False,
) -> utc.Matrix:
    '''
    Return a matrix with the log of some data layer.

    If ``of_layer`` is specified, this specific data is used. Otherwise, the focus data layer (in
    ``adata.X``) is used.

    If ``inplace`` or ``infocus`` or ``instead``, the data will be stored in ``to_layer`` (by
    default, this is the name of the source layer with a ``log_`` prefix).

    The natural logarithm is used by default. Otherwise, the ``base`` is used.

    The ``normalization`` (default: {normalization}) is added to the count before the log is
    applied, to handle the common case of sparse data.

    .. note::

        The result is always a dense matrix, as even for sparse data, the log is rarely zero.
    '''
    if of_layer is None:
        of_layer = get_annotation_of_data(adata, 'layer')
        assert of_layer is not None

    if to_layer is None:
        to_layer = 'log_' + of_layer

    def compute() -> np.ndarray:
        assert of_layer is not None
        data = get_data_layer(adata, of_layer)
        return utc.log_matrix(data, base=base, normalization=normalization)

    return get_data_layer(adata, to_layer, compute,
                          inplace=inplace, infocus=infocus, instead=instead)


def get_annotation_of_obs(
    adata: AnnData,
    name: str,
    compute: Optional[Callable[[], utc.Vector]] = None,
    *,
    inplace: bool = False
) -> np.ndarray:
    '''
    Lookup per-observation (cell) annotation in ``adata``.

    If the annotation does not exist, ``compute`` it. If no ``compute`` function was given,
    ``raise``.

    If ``inplace``, store the annotation in ``adata``.
    '''
    if name in adata.obs_keys():
        return adata.obs[name]

    if compute is None:
        raise RuntimeError('unavailable observation annotation: ' + name)
    data = compute()
    assert data is not None
    assert data.shape == (get_obs_count(adata),)

    if inplace:
        adata.obs[name] = data

    return data


def get_sum_per_obs(adata: AnnData, layer: str, *, inplace: bool = False) -> np.ndarray:
    '''
    Return the sum of the values per observation (cell) of some ``layer``.

    Use the existing ``sum_<layer>`` annotation if it exists.

    If ``inplace``, store the annotation in ``adata``.
    '''
    def compute() -> np.ndarray:
        return utc.sum_matrix(get_data_layer(adata, layer, layout='csr'), axis=1)

    return get_annotation_of_obs(adata, 'obs_' + layer, compute, inplace=inplace)


def get_nnz_per_obs(adata, layer: str, *, inplace: bool = False) -> np.ndarray:
    '''
    Return the number of non-zero values per observation (cell) of some ``layer``.

    Use the existing ``nnz_<layer>`` annotation if it exists.

    If ``inplace``, store the annotation in ``adata``.
    '''
    def compute() -> np.ndarray:
        return utc.nnz_matrix(get_data_layer(adata, layer, layout='csr'), axis=1)

    return get_annotation_of_obs(adata, 'obs_' + layer, compute, inplace=inplace)


def get_max_per_obs(adata, layer: str, *, inplace: bool = False) -> np.ndarray:
    '''
    Return the maximal value per observation (cell) of some ``layer``.

    Use the existing ``max_<layer>`` annotation if it exists.

    If ``inplace``, store the annotation in ``adata``.
    '''
    def compute() -> np.ndarray:
        return utc.max_matrix(get_data_layer(adata, layer, layout='csr'), axis=1)

    return get_annotation_of_obs(adata, 'obs_' + layer, compute, inplace=inplace)


def get_annotation_of_var(
    adata: AnnData,
    name: str,
    compute: Optional[Callable[[], utc.Vector]] = None,
    *,
    inplace: bool = False
) -> np.ndarray:
    '''
    Lookup per-variable (gene) annotation in ``adata``.

    If the annotation does not exist, ``compute`` it. If no ``compute`` function was given,
    ``raise``.

    If ``inplace``, store the annotation in ``adata``.
    '''
    if name in adata.var_keys():
        return adata.var[name]

    if compute is None:
        raise RuntimeError('unavailable variable annotation: ' + name)

    data = compute()
    assert data is not None
    assert data.shape == (get_var_count(adata),)

    if inplace:
        adata.var[name] = data

    return data


def get_sum_per_var(adata: AnnData, layer: str, *, inplace: bool = False) -> np.ndarray:
    '''
    Return the sum of the values per variable (gene) of some ``layer``.

    Use the existing ``sum_<layer>`` annotation if it exists.

    If ``inplace``, store the annotation in ``adata``.
    '''
    def compute() -> np.ndarray:
        return utc.sum_matrix(get_data_layer(adata, layer, layout='csc'), axis=0)

    return get_annotation_of_var(adata, 'sum_' + layer, compute, inplace=inplace)


def get_nnz_per_var(adata, layer: str, *, inplace: bool = False) -> np.ndarray:
    '''
    Return the number of non-zero values per variable (gene) of some ``layer``.

    Use the existing ``nnz_<layer>`` annotation if it exists.

    If ``inplace``, store the annotation in ``adata``.
    '''
    def compute() -> np.ndarray:
        return utc.nnz_matrix(get_data_layer(adata, layer, layout='csc'), axis=0)

    return get_annotation_of_var(adata, 'nnz_' + layer, compute, inplace=inplace)


def get_max_per_var(adata, layer: str, *, inplace: bool = False) -> np.ndarray:
    '''
    Return the maximal value per variable (gene) of some ``layer``.

    Use the existing ``max_<layer>`` annotation if it exists.

    If ``inplace``, store the annotation in ``adata``.
    '''
    def compute() -> np.ndarray:
        return utc.max_matrix(get_data_layer(adata, layer, layout='csc'), axis=0)

    return get_annotation_of_var(adata, 'max_' + layer, compute, inplace=inplace)


def get_annotation_of_data(
    adata: AnnData,
    name: str,
    compute: Optional[Callable[[], utc.Vector]] = None,
    *,
    inplace: bool = False
) -> Any:
    '''
    Lookup unstructured (data) annotation in ``adata``.

    If the annotation does not exist, ``compute`` it. If no ``compute`` function was given,
    ``raise``.

    If ``inplace``, store the annotation in ``adata``.
    '''
    if name in adata.uns_keys():
        return adata.uns[name]

    if compute is None:
        raise RuntimeError('unavailable unstructured annotation: ' + name)

    data = compute()
    assert data is not None

    if inplace:
        adata.uns[name] = data

    return data
