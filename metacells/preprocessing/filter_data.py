'''
Filter the data for computing metacells.
'''

from typing import List, Optional

import numpy as np  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'DEFAULT_FILTER_MASKS',
    'filter_data',
]

#: The default masks used to filter the data.
DEFAULT_FILTER_MASKS = [
    'properly_sampled_cells?',
    'properly_sampled_genes?',
    '~noisy_lonely_genes?',
    '~excluded_genes?',
    '~excluded_cells?'
]


@ut.call()
@ut.expand_doc(masks='``, ``'.join(DEFAULT_FILTER_MASKS))
def filter_data(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches,dangerous-default-value
    adata: AnnData,
    *,
    track_base_indices: Optional[str] = 'base_index',
    masks: List[str] = DEFAULT_FILTER_MASKS,
    intermediate: bool = True,
) -> Optional[AnnData]:
    '''
    Filter the data in preparation for computing metacells.

    This picks a subset of the cells and the genes, so that running the metacells algorithm on the
    subset would give "optimal" results.

    In general, it is useful to discard some of the data before performing any form of an analysis
    algorithm. For example, it is useful to discard cell-cycle genes, cells which have too few UMIs
    for meaningful analysis, etc. In general, the "best" filter depends on the data set.

    This function makes it easy to combine different pre-computed per-observation (cell) and
    per-variable (gene) boolean mask annotations into a final overall inclusion mask, and slice the
    data accordingly, while tracking the base index of the cells and genes in the filtered data.

    **Input**

    A :py:func:`metacells.utilities.preparation.prepare`-ed annotated ``adata``, where the
    observations are cells and the variables are genes, containing the UMIs count in the ``of``
    (default: the ``focus``) per-variable-per-observation data.

    **Returns**

    An annotated data containing a subset of the cells and genes, ready for computing metacells.

    If ``intermediate``, store the ``included_cells`` and ``included_genes`` masks in the full data.

    If no cells and/or no genes were selected by the filter, return ``None``.

    **Computation Parameters**

    Given an annotated ``adata``, where the variables are cell RNA profiles and the observations are
    gene UMI counts, do the following:

    1. If ``track_base_indices`` is not ``None`` (default: ``{track_base_indices}``), then invoke
       :py:func:`metacells.utilities.preparation.track_base_indices` to allow for mapping the
       returned sliced data back to the full original data.

    2. For each of the mask in ``masks`` (default: ``{masks}``), fetch it. Silently ignore
       missing masks if the name has a ``?`` suffix. Invert the mask if the name has a ``~`` suffix.
       AND the mask with the appropriate (cells or genes) mask.

    4. If the final cells or genes mask is empty, return None. Otherwise, return a slice of the full
       data containing just the cells and genes specified by the final masks.
    '''
    if track_base_indices is not None:
        ut.track_base_indices(adata, name=track_base_indices)

    cells_mask = np.full(adata.n_obs, True, dtype='bool')
    genes_mask = np.full(adata.n_obs, True, dtype='bool')

    for name in masks:
        if name[0] == '~':
            invert = True
            name = name[1]
        else:
            invert = False

        if name[-1] == '?':
            must_exist = False
            name = name[:-1]
        else:
            must_exist = True

        per = ut.which_data(adata, name, must_exist=must_exist)
        if per is None:
            continue

        mask = ut.to_1d_array(ut.get_data(adata, name))
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
                             % name)

    if intermediate:
        adata.obs['included_cells'] = cells_mask
        adata.var['included_genes'] = genes_mask
        ut.safe_slicing_data('included_cells', ut.NEVER_SAFE)
        ut.safe_slicing_data('included_genes', ut.NEVER_SAFE)

    if not np.any(cells_mask) or not np.any(genes_mask):
        return None

    return ut.slice(adata, obs=cells_mask, vars=genes_mask)
