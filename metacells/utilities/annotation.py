'''
Utilities for dealing with data annotations.

This tries to deal safely with data layers, observation, variable and unstructured annotations,
especially in the presence to slicing the data.

.. todo::

    Deal with multi-dimensional ''obsm`` and ``varm`` annotations.
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

    'get_cells_count',
    'get_genes_count',

    'declare_focus_of_data',

    'get_data_layer',
    'get_csr_data',
    'get_csc_data',
    'get_log_data',

    'get_annotation_of_cells',
    'get_total_umis_of_cells',
    'get_non_zero_genes_of_cells',

    'get_annotation_of_genes',
    'get_total_umis_of_genes',
    'get_non_zero_cells_of_genes',
    'get_max_umis_of_genes',

    'get_annotation_of_data',
]


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
    slicing_obs_annotation('total_counts',
                           preserve_when_slicing_obs=True,
                           preserve_when_slicing_var=False)
    slicing_obs_annotation('n_genes_by_counts',
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
    slicing_var_annotation('total_counts',
                           preserve_when_slicing_obs=False,
                           preserve_when_slicing_var=True)
    slicing_obs_annotation('n_cells_by_counts',
                           preserve_when_slicing_obs=False,
                           preserve_when_slicing_var=True)
    slicing_obs_annotation('max_by_counts',
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


def _slice_csr(
    slicing_context: SlicingContext,
    _annotation_name: str,
) -> None:
    assert slicing_context.did_slice_obs
    assert not slicing_context.did_slice_var
    full_csr = slicing_context.full_adata.uns['csr']
    assert id(slicing_context.sliced_adata.uns['csr']) == id(full_csr)
    slicing_context.sliced_adata.uns['csr'] = full_csr[slicing_context.slice_obs, :]


def _slice_csc(
    slicing_context: SlicingContext,
    _annotation_name: str,
) -> None:
    assert slicing_context.did_slice_obs
    assert not slicing_context.did_slice_var
    full_csc = slicing_context.sliced_adata.uns['csc']
    assert id(slicing_context.sliced_adata.uns['csc']) == id(full_csc)
    slicing_context.sliced_adata.uns['csc'] = full_csc[:,
                                                       slicing_context.slice_var]


def _slicing_known_uns_annotations() -> None:
    slicing_uns_annotation('layer',
                           preserve_when_slicing_obs=True,
                           preserve_when_slicing_var=True)
    slicing_uns_annotation('csr',
                           preserve_when_slicing_obs=True,
                           preserve_when_slicing_var=False,
                           fix_sliced_annotation=_slice_csr)
    slicing_uns_annotation('csc',
                           preserve_when_slicing_obs=False,
                           preserve_when_slicing_var=True,
                           fix_sliced_annotation=_slice_csc)


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

    In general, annotations data might become invalid when slicing (e.g., the ``total_counts`` of
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
        cells = range(get_cells_count(adata))
    if genes is None:
        genes = range(get_genes_count(adata))

    bdata = adata[cells, genes]

    did_slice_obs = get_cells_count(bdata) != get_cells_count(adata)
    did_slice_var = get_genes_count(bdata) != get_genes_count(adata)

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
        (preserve_when_slicing_obs, preserve_when_slicing_var, fix_sliced_layer) = \
            _get_slicing_annotation('data', 'layer',
                                    SAFE_SLICING_DATA_LAYERS, name)

        preserve = (not slicing_context.did_slice_obs or preserve_when_slicing_obs) \
            and (not slicing_context.did_slice_var or preserve_when_slicing_var)

        if preserve:
            if fix_sliced_layer is not None:
                fix_sliced_layer(slicing_context, name)
            continue

        if invalidated_annotations_prefix is not None:
            new_name = invalidated_annotations_prefix + name
            assert new_name != name
            slicing_context.sliced_adata.layers[new_name] = slicing_context.sliced_adata.layers[name]
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
    adata.obs[name] = np.arange(get_cells_count(adata))
    adata.var[name] = np.arange(get_genes_count(adata))


def get_cells_count(adata: AnnData) -> int:
    '''
    Return the number of RNA cell profiles in the ``adata``.
    '''
    return adata.shape[0]


def get_genes_count(adata: AnnData) -> int:
    '''
    Return the number of genes per cell in the ``adata``.
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


def get_data_layer(
    adata: AnnData,
    name: str,
    compute: Optional[Callable[[AnnData], Any]] = None,
    *,
    inplace: bool = False,
    infocus: bool = False,
    instead: bool = False,
) -> Union[np.ndarray, sparse.spmatrix, pd.DataFrame]:
    '''
    Lookup a per-obsevration-per-variable matrix in ``adata``.

    If the layer does not exist, ``compute`` it. If no ``compute`` function was given, ``raise``.

    If ``inplace``, store the annotation in ``adata``.

    If ``infocus`` (implies ``inplace``), also make it the current ``X`` of the data.

    If ``instead`` (implies ``infocus``), do not preserve the current ``X`` of the data, that is,
    remove it from the layers.

    .. note::

        This ``assert``s that the current ``X``, if not ``None``, was declared to be a layer of the
        data so it isn't lost even if ``infocus`` is specified.
    '''
    if name in adata.layers.keys():
        data = adata.layers[name]
    else:
        if compute is None:
            raise RuntimeError('unavailable layer data: ' + name)
        data = compute(adata)
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

    return data


def get_csr_data(
    adata: AnnData,
    *,
    of_layer: str,
    to_layer: Optional[str] = None,
    inplace: bool = False,
    infocus: bool = False,
    instead: bool = False,
) -> Union[np.ndarray, sparse.spmatrix]:
    '''
    If the data layer is dense, return it immediately. Otherwise:

    Return the data in ``csr_matrix`` format for efficient per-row (cell) processing.

    If ``of_layer`` is specified, this specific data is used. Otherwise, the focus data layer (in
    ``adata.X``) is used.

    If ``inplace`` or ``infocus`` or ``instead``, the data will be stored in ``to_layer`` (by
    default, this is the name of the source layer with a ``csr_`` prefix).
    '''
    data = get_data_layer(adata, of_layer)
    if not sparse.issparse(data):
        return data

    if to_layer is None:
        to_layer = 'csr_' + of_layer

    return get_data_layer(adata, to_layer, lambda _adata: data.tocsr(),
                          inplace=inplace, infocus=infocus, instead=instead)


def get_csc_data(
    adata: AnnData,
    *,
    of_layer: str,
    to_layer: Optional[str] = None,
    inplace: bool = False,
    infocus: bool = False,
    instead: bool = False,
) -> Union[np.ndarray, sparse.spmatrix]:
    '''
    If the data layer is dense, return it immediately. Otherwise:

    Return the data in ``csc_matrix`` format for efficient per-column (gene) processing.

    If ``of_layer`` is specified, this specific data is used. Otherwise, the focus data layer (in
    ``adata.X``) is used.

    If ``inplace`` or ``infocus`` or ``instead``, the data will be stored in ``to_layer`` (by
    default, this is the name of the source layer with a ``csc_`` prefix).
    '''
    data = get_data_layer(adata, of_layer)
    if not sparse.issparse(data):
        return data

    if to_layer is None:
        to_layer = 'csc_' + of_layer

    return get_data_layer(adata, to_layer, lambda _: data.tocsc(),
                          inplace=inplace, infocus=infocus, instead=instead)


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
) -> Union[np.ndarray]:
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

    def compute(adata: AnnData) -> np.ndarray:
        assert of_layer is not None
        data = get_data_layer(adata, of_layer)
        return utc.log_data(data, base=base, normalization=normalization)

    return get_data_layer(adata, to_layer, compute,
                          inplace=inplace, infocus=infocus, instead=instead)


def get_annotation_of_cells(
    adata: AnnData,
    name: str,
    compute: Optional[Callable[[AnnData], Any]] = None,
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
    data = compute(adata)
    assert data is not None
    assert data.shape == (get_cells_count(adata),)

    if inplace:
        adata.obs[name] = data

    return data


def get_total_umis_of_cells(adata: AnnData, *, inplace: bool = False) -> np.ndarray:
    '''
    Return the total number of UMIs per cell.

    Use the existing ``total_counts`` annotation if it exists.

    If ``inplace``, store the annotation in ``adata``.
    '''
    return get_annotation_of_cells(adata, 'total_counts',
                                   utc.totals_of_cells, inplace=inplace)


def get_non_zero_genes_of_cells(adata, *, inplace: bool = False) -> np.ndarray:
    '''
    Return the number of genes with non-zero UMIs per cell.

    Use the existing ``n_genes_by_counts`` annotation if it exists.

    If ``inplace``, store the annotation in ``adata``.
    '''
    return get_annotation_of_cells(adata, 'n_cells_by_counts',
                                   utc.non_zero_genes_of_cells, inplace=inplace)


def get_annotation_of_genes(
    adata: AnnData,
    name: str,
    compute: Optional[Callable[[AnnData], Any]] = None,
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

    data = compute(adata)
    assert data is not None
    assert data.shape == (get_genes_count(adata),)

    if inplace:
        adata.var[name] = data

    return data


def get_total_umis_of_genes(adata: AnnData, *, inplace: bool = False) -> np.ndarray:
    '''
    Return the total number of UMIs per gene.

    Use the existing ``total_counts`` annotation if it exists.

    If ``inplace``, store the annotation in ``adata``.
    '''
    return get_annotation_of_genes(adata, 'total_counts',
                                   utc.totals_of_genes, inplace=inplace)


def get_non_zero_cells_of_genes(adata, *, inplace: bool = False) -> np.ndarray:
    '''
    Return the number of cells each gene has non-zero UMIs in.

    Use the existing ``n_cells_by_counts`` annotation if it exists.

    If ``inplace``, store the annotation in ``adata``.
    '''
    return get_annotation_of_genes(adata, 'n_cells_by_counts',
                                   utc.non_zero_cells_of_genes, inplace=inplace)


def get_max_umis_of_genes(adata, *, inplace: bool = False) -> np.ndarray:
    '''
    Return the maximal number of UMIs each gene has in a cell.

    Use the existing ``max_by_counts`` annotation if it exists.

    If ``inplace``, store the annotation in ``adata``.
    '''
    return get_annotation_of_genes(adata, 'max_by_counts',
                                   utc.max_umis_of_genes, inplace=inplace)


def get_annotation_of_data(
    adata: AnnData,
    name: str,
    compute: Optional[Callable[[AnnData], Any]] = None,
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

    data = compute(adata)
    assert data is not None

    if inplace:
        adata.uns[name] = data

    return data
