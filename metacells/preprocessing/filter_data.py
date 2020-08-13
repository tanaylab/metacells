'''
Filter the data for computing metacells.
'''

from typing import List, Optional

import numpy as np  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'filter_data',
]


@ut.timed_call()
@ut.expand_doc()
def filter_data(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches,dangerous-default-value
    adata: AnnData,
    masks: List[str],
    *,
    track_base_indices: Optional[str] = 'base_index',
    name: Optional[str] = 'included',
) -> Optional[AnnData]:
    '''
    Extract a subset of the data for further processing.

    For example, it is useful to discard cell-cycle genes, cells which have too few UMIs for
    meaningful analysis, etc. In general, the "best" filter depends on the data set.

    This function makes it easy to combine different pre-computed per-observation (cell) and
    per-variable (gene) boolean mask annotations into a final overall inclusion mask, and slice the
    data accordingly, while tracking the base index of the cells and genes in the filtered data.

    **Input**

    A :py:func:`metacells.utilities.preparation.prepare`-ed annotated ``adata``, where the
    observations are cells and the variables are genes.

    **Returns**

    An annotated data containing a subset of the cells and genes.

    If ``name`` (default: {name}) is not ``None``, store the mask in ``<name>_cells`` and
    ``<name>_genes`` annotations of the full data.

    If no cells and/or no genes were selected by the filter, return ``None``.

    **Computation Parameters**

    1. If ``track_base_indices`` (default: ``{track_base_indices}``) is not ``None``, then invoke
       :py:func:`metacells.utilities.preparation.track_base_indices` to allow for mapping the
       returned sliced data back to the full original data.

    2. For each of the mask in ``masks``, fetch it. Silently ignore missing masks if the name has a
       ``?`` suffix. Invert the mask if the name has a ``~`` suffix. Bitwise-AND the appropriate
       (cells or genes) mask with the result.

    4. If the final cells or genes mask is empty, return None. Otherwise, return a slice of the full
       data containing just the cells and genes specified by the final masks.
    '''
    if track_base_indices is not None:
        ut.track_base_indices(adata, name=track_base_indices)

    cells_mask = np.full(adata.n_obs, True, dtype='bool')
    genes_mask = np.full(adata.n_obs, True, dtype='bool')

    for mask_name in masks:
        if mask_name[0] == '~':
            invert = True
            mask_name = mask_name[1]
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

        if per == 'o':
            cells_mask = cells_mask & mask
        elif per == 'v':
            genes_mask = genes_mask & mask
        else:
            raise ValueError('the data: %s '
                             'is not per-observation or per-variable'
                             % mask_name)

    if name is not None:
        cells_name = name + '_cells'
        genes_name = name + '_genes'
        adata.obs[cells_name] = cells_mask
        adata.var[genes_name] = genes_mask
        ut.safe_slicing_data(cells_name, ut.NEVER_SAFE)
        ut.safe_slicing_data(genes_name, ut.NEVER_SAFE)

    if not np.any(cells_mask) or not np.any(genes_mask):
        return None

    return ut.slice(adata, obs=cells_mask, vars=genes_mask)
