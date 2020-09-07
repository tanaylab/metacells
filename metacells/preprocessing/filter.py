'''
Filter
------
'''

import logging
from typing import List, Optional

import numpy as np  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'filter_data',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def filter_data(  # pylint: disable=too-many-branches
    adata: AnnData,
    masks: List[str],
    *,
    to: Optional[str] = 'included',
    name: Optional[str] = None,
    tmp: bool = False,
    invalidated_prefix: Optional[str] = None,
    invalidated_suffix: Optional[str] = None,
) -> Optional[AnnData]:
    '''
    Filter (slice) the data based on previously-computed masks.

    For example, it is useful to discard cell-cycle genes, cells which have too few UMIs for
    meaningful analysis, etc. In general, the "best" filter depends on the data set.

    This function makes it easy to combine different pre-computed per-observation (cell) and
    per-variable (gene) boolean mask annotations into a final overall inclusion mask, and slice the
    data accordingly, while tracking the base index of the cells and genes in the filtered data.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` ``adata``, where the observations are cells
    and the variables are genes.

    **Returns**

    An annotated data containing a subset of the cells and genes.

    If ``name`` is specified, this will be the logging name of the new data. Otherwise, it will be
    unnamed.

    If ``tmp`` (default: {tmp}) is set, logging of modifications to the result will use the
    ``DEBUG`` logging level. By default, logging of modifications is done using the ``INFO`` logging
    level.

    If ``to`` (default: {to}) is specified, store the mask in ``<to>_cells`` and ``<to>_genes``
    annotations of the full data.

    If no cells and/or no genes were selected by the filter, return ``None``.

    **Computation Parameters**

    1. For each of the mask in ``masks``, fetch it. Silently ignore missing masks if the name has a
       ``?`` suffix. Invert the mask if the name has a ``~`` suffix. Bitwise-AND the appropriate
       (cells or genes) mask with the result.

    2. If the final cells or genes mask is empty, return None. Otherwise, return a slice of the full
       data containing just the cells and genes specified by the final masks. If
       ``invalidated_prefix`` (default: {invalidated_prefix}) and/or ``invalidated_suffix``
       (default: {invalidated_suffix}) are specified, then invalidated data will not be removed;
       instead it will be renamed with the addition of the provided prefix and/or suffix.
    '''
    _, level = ut.log_operation(LOG, adata, 'filter_data')

    cells_mask = np.full(adata.n_obs, True, dtype='bool')
    genes_mask = np.full(adata.n_vars, True, dtype='bool')

    for mask_name in masks:
        log_mask_name = mask_name

        if mask_name[0] == '~':
            invert = True
            mask_name = mask_name[1:]
        else:
            invert = False

        if mask_name[-1] == '?':
            must_exist = False
            mask_name = mask_name[:-1]
        else:
            must_exist = True

        per = ut.data_per(adata, mask_name, must_exist=must_exist)
        if per is None:
            continue

        mask = ut.get_proper_vector(adata, mask_name)
        if mask.dtype != 'bool':
            raise ValueError('the data: %s is not a boolean mask')
        if invert:
            mask = ~mask

        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug('  %s: %s', log_mask_name, ut.mask_description(mask))

        if per == 'o':
            cells_mask = cells_mask & mask
        elif per == 'v':
            genes_mask = genes_mask & mask
        else:
            raise ValueError('the data: %s '
                             'is not per-observation or per-variable'
                             % mask_name)

    if to is not None:
        ut.set_o_data(adata, to + '_cells', cells_mask, ut.NEVER_SAFE)
        ut.set_v_data(adata, to + '_genes', genes_mask, ut.NEVER_SAFE)

    elif LOG.isEnabledFor(level):
        ut.log_mask(LOG, level, 'cells', cells_mask)
        ut.log_mask(LOG, level, 'genes', genes_mask)

    if not np.any(cells_mask) or not np.any(genes_mask):
        return None

    return ut.slice(adata, name=name, tmp=tmp,
                    obs=cells_mask, vars=genes_mask,
                    invalidated_prefix=invalidated_prefix,
                    invalidated_suffix=invalidated_suffix)