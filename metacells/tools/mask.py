"""
Mask
----
"""

from typing import Collection
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from anndata import AnnData  # type: ignore

import metacells.utilities as ut

__all__ = [
    "combine_masks",
    "sum_mask_genes",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def combine_masks(  # pylint: disable=too-many-branches,too-many-statements
    adata: AnnData,
    masks: Collection[str],
    *,
    invert: bool = False,
    to: Optional[str] = None,
) -> Optional[pd.Series]:
    """
    Combine different pre-computed masks into a final overall mask.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    If ``to`` (default: {to}) is ``None``, returns the computed mask. Otherwise, sets the
    mask as an annotation (per-variable or per-observation depending on the type of the combined masks).

    **Computation Parameters**

    1. For each of the mask in ``masks``, in order (left to right), fetch it. Silently ignore missing masks if the name
       has a ``?`` suffix. If the first character of the mask name is ``&``, restrict the current mask, otherwise the
       first character must be ``|`` and we'll expand the mask (for the 1st mask, the mask becomes the current mask
       regardless of the 1st character). If the following character is ``~``, first invert the mask before applying it.

    3. If ``invert`` (default: {invert}), invert the final result mask.
    """
    assert len(masks) > 0

    per: Optional[str] = None

    result_mask: Optional[ut.NumpyVector] = None

    for mask_name in masks:
        log_mask_name = mask_name

        if mask_name[0] == "|":
            is_or = True
            mask_name = mask_name[1:]
        elif mask_name[0] == "&":
            is_or = False
            mask_name = mask_name[1:]
        else:
            raise ValueError(f"invalid mask name: {mask_name} (does not start with & or |)")

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

        if result_mask is None:
            result_mask = mask
        elif is_or:
            result_mask = result_mask | mask
        else:
            result_mask = result_mask & mask

    if result_mask is None:
        raise ValueError("no masks to combine")

    if invert:
        result_mask = ~result_mask

    if to is None:
        ut.log_return("result", result_mask)
        if per == "o":
            return ut.to_pandas_series(result_mask, index=adata.obs_names)
        assert per == "v"
        return ut.to_pandas_series(result_mask, index=adata.var_names)

    if per == "o":
        ut.set_o_data(adata, to, result_mask)
    else:
        ut.set_v_data(adata, to, result_mask)

    return None


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def sum_mask_genes(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    mask_property: str,
    to: Optional[str] = None,
) -> Optional[pd.Series]:
    """
    Sum the total value (typically UMIs) of all the variables (typically genes) listed in some ``mask_property`` in each
    of the observations (typically cells).

    This is used to count things like the total UMIs of mitochondrial genes or ribosomal genes when choosing cells to
    exclude from the analyzed data.

    The ``mask_property`` name should be of a per-variable (gene) Boolean annotation.

    If ``to`` is specified, this is stored as a per-observation (cell) annotation with that name, and returns ``None``.
    Otherwise, it returns it as a pandas series (indexed by the observation names, typically cell barcodes).
    """
    mask = ut.get_v_numpy(adata, mask_property)
    assert mask.dtype == "bool"

    values = ut.get_vo_proper(adata, what, layout="row_major")

    sums = np.zeros(adata.n_obs, dtype=values.dtype)  # type: ignore
    for var_index in np.where(mask)[0]:
        sums += values[var_index, :]

    if to is not None:
        ut.set_v_data(adata, to, sums)
        return None

    ut.log_return("sums", sums)
    return ut.to_pandas_series(sums, index=adata.obs_names)
