"""
Mask
----
"""

from typing import List
from typing import Optional

from anndata import AnnData  # type: ignore

import metacells.utilities as ut

__all__ = [
    "combine_masks",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def combine_masks(  # pylint: disable=too-many-branches,too-many-statements
    adata: AnnData,
    masks: List[str],
    *,
    invert: bool = False,
    to: Optional[str] = None,
) -> Optional[ut.PandasSeries]:
    """
    Combine different pre-computed masks into a final overall mask.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    If ``to`` (default: {to}) is ``None``, returns the computed mask. Otherwise, sets the
    mask as an annotation (per-variable or per-observation depending on the type of the combined masks).

    **Computation Parameters**

    1. For each of the mask in ``masks``, fetch it. Silently ignore missing masks if the name has a
       ``?`` suffix. Invert the mask if the name has a ``~`` prefix. If the name has a ``|`` prefix
       (before the ``~`` prefix, if any), then bitwise-OR the mask into the OR mask, otherwise (or if
       it has a ``&`` prefix), bitwise-AND the mask into the AND mask.

    2. Combine (bitwise-AND) the AND mask and the OR mask into a single mask.

    3. If ``invert`` (default: {invert}), invert the result combined mask.
    """
    assert len(masks) > 0

    per: Optional[str] = None

    and_mask: Optional[ut.NumpyVector] = None
    or_mask: Optional[ut.NumpyVector] = None

    for mask_name in masks:
        log_mask_name = mask_name

        if mask_name[0] == "|":
            is_or = True
            mask_name = mask_name[1:]
        else:
            is_or = False
            if mask_name[0] == "&":
                mask_name = mask_name[1:]

        if mask_name[0] == "~":
            invert_mask = True
            mask_name = mask_name[1:]
        else:
            invert_mask = False

        if mask_name[-1] == "?":
            must_exist = False
            mask_name = mask_name[:-1]
        else:
            must_exist = True

        if mask_name in adata.obs:
            mask_per = "o"
            mask = ut.get_o_numpy(adata, mask_name, formatter=ut.mask_description) > 0
        elif mask_name in adata.var:
            mask_per = "v"
            mask = ut.get_v_numpy(adata, mask_name, formatter=ut.mask_description) > 0
        else:
            if must_exist:
                raise KeyError(f"unknown mask data: {mask_name}")
            continue

        if mask.dtype != "bool":
            raise ValueError(f"the data: {mask_name} is not a boolean mask")

        if invert_mask:
            mask = ~mask

        if ut.logging_calc():
            ut.log_calc(log_mask_name, mask)

        if per is None:
            per = mask_per
        else:
            if mask_per != per:
                raise ValueError("mixing per-observation and per-variable masks")

        if is_or:
            if or_mask is None:
                or_mask = mask
            else:
                or_mask = or_mask | mask
        else:
            if and_mask is None:
                and_mask = mask
            else:
                and_mask = and_mask & mask

    if and_mask is not None:
        if or_mask is not None:
            combined_mask = and_mask & or_mask
        else:
            combined_mask = and_mask
    else:
        if or_mask is not None:
            combined_mask = or_mask
        else:
            raise ValueError("no masks to combine")

    if invert:
        combined_mask = ~combined_mask

    if to is None:
        ut.log_return("combined", combined_mask)
        if per == "o":
            return ut.to_pandas_series(combined_mask, index=adata.obs_names)
        assert per == "v"
        return ut.to_pandas_series(combined_mask, index=adata.var_names)

    if per == "o":
        ut.set_o_data(adata, to, combined_mask)
    else:
        ut.set_v_data(adata, to, combined_mask)

    return None
