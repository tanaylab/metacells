'''
Mask
----
'''

from typing import List, Optional

from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'combine_masks',
]


@ut.logged()
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

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    If ``to`` (default: {to}) is ``None``, returns the computed mask. Otherwise, sets the
    mask as an annotation (per-variable or per-observation depending on the type of the combined masks).

    **Computation Parameters**

    1. For each of the mask in ``masks``, fetch it. Silently ignore missing masks if the name has a
       ``?`` suffix. Invert the mask if the name has a ``~`` suffix. Bitwise-AND the appropriate
       (observations or variables) mask with the result.

    2. If ``invert`` (default: {invert}), invert the result combined mask.
    '''
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
            mask = ut.get_o_numpy(adata, mask_name,
                                  formatter=ut.mask_description) > 0
        elif mask_name in adata.var:
            mask_per = 'v'
            mask = ut.get_v_numpy(adata, mask_name,
                                  formatter=ut.mask_description) > 0
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

        if ut.logging_calc():
            ut.log_calc(log_mask_name, mask)

    if invert:
        combined_mask = ~combined_mask

    if to is None:
        ut.log_return('combined', combined_mask)
        if per == 'o':
            return ut.to_pandas_series(combined_mask, index=adata.obs_names)
        assert per == 'v'
        return ut.to_pandas_series(combined_mask, index=adata.var_names)

    if per == 'o':
        ut.set_o_data(adata, to, combined_mask)
    else:
        ut.set_v_data(adata, to, combined_mask)
    return None
