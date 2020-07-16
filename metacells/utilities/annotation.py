'''
Utilities for dealing with data annotations.
'''

from typing import Any, Callable, Collection, Dict, Optional, Tuple, Union
from warnings import warn

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData  # type: ignore
from readerwriterlock import rwlock

import metacells.utilities.computation as utc

__all__ = [
    'slice',
    'slicing_obs_annotation',
    'slicing_var_annotation',
    'slicing_uns_annotation',

    'annotate_as_base',
    'get_cells_count',
    'get_genes_count',

    'get_annotation_of_cells',
    'get_annotation_of_genes',
    'get_total_umis_of_cells',
    'get_total_umis_of_genes',
    'get_non_zero_genes_of_cells',
    'get_non_zero_cells_of_genes',
    'get_max_umis_of_genes',
]

LOCK = rwlock.RWLockRead()

WARN_WHEN_SLICING_UNKNOWN_ANNOTATIONS: bool = True

SAFE_SLICING_OBS_ANNOTATIONS: Dict[str,
                                   Tuple[bool, bool,
                                         Optional[Callable[[AnnData, AnnData, str, bool, bool],
                                                           None]]]] = {}
SAFE_SLICING_VAR_ANNOTATIONS: Dict[str,
                                   Tuple[bool, bool,
                                         Optional[Callable[[AnnData, AnnData, str, bool, bool],
                                                           None]]]] = {}
SAFE_SLICING_UNS_ANNOTATIONS: Dict[str,
                                   Tuple[bool, bool,
                                         Optional[Callable[[AnnData, AnnData, str, bool, bool],
                                                           None]]]] = {}


def slicing_obs_annotation(
    name: str,
    *,
    preserve_when_slicing_obs: bool = False,
    preserve_when_slicing_var: bool = False,
    fix_sliced_annotation: Optional[Callable[[AnnData, AnnData, str, bool, bool], None]] = None,
) -> None:
    '''
    Specify whether the named per-observation (cell) annotation should be preserved when slicing
    either the observations (cells) or variables (genes).

    If ``fix_sliced_annotation`` is provided, when a slicing occurs, it will be invoked as
    ``fix_sliced_annotation(full_adata, sliced_adata, annotation_name, did_slice_obs,
    did_slice_var)`` to allow for fixing the preserved sliced annotation values.
    '''
    with LOCK.gen_wlock():
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
    fix_sliced_annotation: Optional[Callable[[AnnData, AnnData, str, bool, bool], None]] = None,
) -> None:
    '''
    Specify whether the named per-variable (gene) annotation should be preserved when slicing
    either the observations (cells) or variables (genes).

    If ``fix_sliced_annotation`` is provided, when a slicing occurs, it will be invoked as
    ``fix_sliced_annotation(full_adata, sliced_adata, annotation_name, did_slice_obs,
    did_slice_var)`` to allow for fixing the preserved sliced annotation values.
    '''
    with LOCK.gen_wlock():
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
    fix_sliced_annotation: Optional[Callable[[AnnData, AnnData, str, bool, bool], None]] = None,
) -> None:
    '''
    Specify whether the named unstructured annotation should be preserved when slicing either the
    observations (cells) or variables (genes).

    If ``fix_sliced_annotation`` is provided, when a slicing occurs, it will be invoked as
    ``fix_sliced_annotation(full_adata, sliced_adata, annotation_name, did_slice_obs,
    did_slice_var)`` to allow for fixing the preserved sliced annotation values.
    '''
    with LOCK.gen_wlock():
        SAFE_SLICING_UNS_ANNOTATIONS[name] = (preserve_when_slicing_obs,
                                              preserve_when_slicing_var,
                                              fix_sliced_annotation)


def _slicing_known_uns_annotations() -> None:
    return


def _slicing_known_annotations() -> None:
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

    if did_slice_obs or did_slice_var:
        _filter_annotations('obs', adata, bdata, bdata.obs, SAFE_SLICING_OBS_ANNOTATIONS,
                            did_slice_obs, did_slice_var, invalidated_annotations_prefix)
        _filter_annotations('var', adata, bdata, bdata.var, SAFE_SLICING_VAR_ANNOTATIONS,
                            did_slice_obs, did_slice_var, invalidated_annotations_prefix)
        _filter_annotations('uns', adata, bdata, bdata.uns, SAFE_SLICING_UNS_ANNOTATIONS,
                            did_slice_obs, did_slice_var, invalidated_annotations_prefix)

    return bdata


def _filter_annotations(  # pylint: disable=too-many-locals
    kind: str,
    old_data: AnnData,
    new_data: AnnData,
    new_annotations: Union[pd.DataFrame, Dict[str, Any]],
    slicing_annotations: Dict[str,
                              Tuple[bool, bool,
                                    Optional[Callable[[AnnData, AnnData, str, bool, bool],
                                                      None]]]],
    did_slice_obs: bool,
    did_slice_var: bool,
    invalidated_annotations_prefix: Optional[str],
) -> None:
    annotation_names = list(new_annotations.keys())
    for name in annotation_names:
        with LOCK.gen_rlock():
            slicing_annotation = slicing_annotations.get(name)

        if slicing_annotation is None:
            with LOCK.gen_wlock():
                slicing_annotation = slicing_annotations.get(name)
                if slicing_annotation is None:
                    unknown_sliced_annotation_message = \
                        'Slicing an unknown {kind} annotation: {name}; ' \
                        'assuming it should not be preserved' \
                        .format(kind=kind, name=name)
                    warn(unknown_sliced_annotation_message)
                    slicing_annotation = (False, False, None)
                    slicing_annotations[name] = slicing_annotation

        (preserve_when_slicing_obs, preserve_when_slicing_var, fix_sliced_annotation) = \
            slicing_annotation

        preserve = (not did_slice_obs or preserve_when_slicing_obs) \
            and (not did_slice_var or preserve_when_slicing_var)

        if preserve:
            if fix_sliced_annotation is not None:
                fix_sliced_annotation(old_data, new_data,
                                      name, did_slice_obs, did_slice_var)
            continue

        if invalidated_annotations_prefix is not None:
            new_name = invalidated_annotations_prefix + name
            assert new_name != name
            new_annotations[new_name] = new_annotations[name]

        del new_annotations[name]


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
    return adata.X.shape[0]


def get_genes_count(adata: AnnData) -> int:
    '''
    Return the number of genes per cell in the ``adata``.
    '''
    return adata.X.shape[1]


def get_annotation_of_cells(
    adata: AnnData, name: str,
    compute: Callable[[AnnData], Any],
    inplace: bool = False
) -> np.ndarray:
    '''
    Lookup per-observation (cell) annotation in ``adata``.

    If the annotation does not exist, ``compute`` it.

    If ``inplace``, store the annotation in ``adata``.
    '''
    if name in adata.obs_keys():
        return adata.obs[name]

    data = compute(adata)
    assert data.shape == (get_cells_count(adata),)

    if inplace:
        adata.obs[name] = data

    return data


def get_annotation_of_genes(
    adata: AnnData, name: str,
    compute: Callable[[AnnData], Any],
    inplace: bool = False
) -> np.ndarray:
    '''
    Lookup per-variable (gene) annotation in ``adata``.

    If the annotation does not exist, ``compute`` it.

    If ``inplace``, store the annotation in ``adata``.
    '''
    if name in adata.var_keys():
        return adata.var[name]

    data = compute(adata)
    assert data.shape == (get_genes_count(adata),)

    if inplace:
        adata.var[name] = data

    return data


def get_total_umis_of_cells(adata: AnnData, inplace: bool = False) -> np.ndarray:
    '''
    Return the total number of UMIs per cell.

    Use the existing ``total_counts`` annotation if it exists.

    If ``inplace``, store the annotation in ``adata``.
    '''
    return get_annotation_of_cells(adata, 'total_counts',
                                   utc.totals_of_cells, inplace=inplace)


def get_total_umis_of_genes(adata: AnnData, inplace: bool = False) -> np.ndarray:
    '''
    Return the total number of UMIs per gene.

    Use the existing ``total_counts`` annotation if it exists.

    If ``inplace``, store the annotation in ``adata``.
    '''
    return get_annotation_of_genes(adata, 'total_counts',
                                   utc.totals_of_genes, inplace=inplace)


def get_non_zero_genes_of_cells(adata, inplace: bool = False) -> np.ndarray:
    '''
    Return the number of genes with non-zero UMIs per cell.

    Use the existing ``n_genes_by_counts`` annotation if it exists.

    If ``inplace``, store the annotation in ``adata``.
    '''
    return get_annotation_of_cells(adata, 'n_cells_by_counts',
                                   utc.non_zero_genes_of_cells, inplace=inplace)


def get_non_zero_cells_of_genes(adata, inplace: bool = False) -> np.ndarray:
    '''
    Return the number of cells each gene has non-zero UMIs in.

    Use the existing ``n_cells_by_counts`` annotation if it exists.

    If ``inplace``, store the annotation in ``adata``.
    '''
    return get_annotation_of_genes(adata, 'n_cells_by_counts',
                                   utc.non_zero_cells_of_genes, inplace=inplace)


def get_max_umis_of_genes(adata, inplace: bool = False) -> np.ndarray:
    '''
    Return the maximal number of UMIs each gene has in a cell.

    Use the existing ``max_by_counts`` annotation if it exists.

    If ``inplace``, store the annotation in ``adata``.
    '''
    return get_annotation_of_genes(adata, 'max_by_counts',
                                   utc.max_umis_of_genes, inplace=inplace)
