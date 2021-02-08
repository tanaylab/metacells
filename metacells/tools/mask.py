'''
Mask
----
'''

import logging
from typing import List, Optional

from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'combine_masks',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def combine_masks(  # pylint: disable=too-many-branches
    adata: AnnData,
    masks: List[str],
    *,
    invert: bool = False,
    to: Optional[str] = None,
) -> Optional[ut.PandasSeries]:
    '''
    Combine different pre-computed masks into a final overall mask.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` ``adata``, where the observations are cells
    and the variables are genes.

    **Returns**

    If ``to`` (default: {to}) is ``None``, returns the computed mask. Otherwise, sets the
    mask as an annotation (per-variable or per-observation depending on the type of the combined masks).

    **Computation Parameters**

    1. For each of the mask in ``masks``, fetch it. Silently ignore missing masks if the name has a
       ``?`` suffix. Invert the mask if the name has a ``~`` suffix. Bitwise-AND the appropriate
       (observations or variables) mask with the result.

    2. If ``invert`` (default: {invert}), invert the result combined mask.
    '''
    level = ut.log_operation(LOG, adata, 'combine_masks')
    assert len(masks) > 0

    per: Optional[str] = None

    for mask_name in masks:
        log_mask_name = mask_name

        if mask_name[0] == '~':
            invert_mask = True
            mask_name = mask_name[1:]
        else:
            invert_mask = False

        if mask_name[-1] == '?':
            must_exist = False
            mask_name = mask_name[:-1]
        else:
            must_exist = True

        if mask_name in adata.obs:
            mask_per = 'o'
            mask = ut.get_o_numpy(adata, mask_name)
        elif mask_name in adata.var:
            mask_per = 'v'
            mask = ut.get_v_numpy(adata, mask_name)
        else:
            if must_exist:
                raise KeyError(f'unknown mask data: {mask_name}')
            continue

        if mask.dtype != 'bool':
            raise ValueError(f'the data: {mask_name} is not a boolean mask')

        if invert_mask:
            mask = ~mask

        if per is None:
            per = mask_per
            combined_mask = mask
        else:
            if mask_per != per:
                raise \
                    ValueError('mixing per-observation and per-variable masks')
            combined_mask = combined_mask & mask

        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug('  %s: %s', log_mask_name, ut.mask_description(mask))

    if invert:
        combined_mask = ~combined_mask

    if LOG.isEnabledFor(level):
        ut.log_mask(LOG, level, 'combined', combined_mask)

    if to is None:
        if per == 'o':
            return ut.to_pandas_series(combined_mask, index=adata.obs_names)
        assert per == 'v'
        return ut.to_pandas_series(combined_mask, index=adata.var_names)

    if per == 'o':
        ut.set_o_data(adata, to, combined_mask)
    else:
        ut.set_v_data(adata, to, combined_mask)

    return None
