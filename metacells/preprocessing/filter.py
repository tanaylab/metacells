'''
Filter
------
'''

import logging
from typing import List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    'filter_data',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def filter_data(  # pylint: disable=dangerous-default-value
    adata: AnnData,
    obs_masks: List[str] = [],
    var_masks: List[str] = [],
    *,
    mask_obs: Optional[str] = None,
    mask_var: Optional[str] = None,
    invert_obs: bool = False,
    invert_var: bool = False,
    track_obs: Optional[str] = None,
    track_var: Optional[str] = None,
    name: Optional[str] = None,
    tmp: bool = False,
) -> Optional[Tuple[AnnData, pd.Series, pd.Series]]:
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

    An annotated data containing a subset of the observations (cells) and variables (genes).

    If no observations and/or no variables were selected by the filter, return ``None``.

    If ``name`` is not specified, the data will be unnamed. Otherwise, if it starts with a ``.``, it
    will be appended to the current name (if any). Otherwise, ``name`` is the new name.

    If ``tmp`` (default: {tmp}) is set, logging of modifications to the result will use the
    ``DEBUG`` logging level. By default, logging of modifications is done using the ``INFO`` logging
    level.

    If ``mask_obs`` and/or ``mask_var`` are specified, store the mask of the selected data as a
    per-observation and/or per-variable annotation of the full ``adata``.

    If ``track_obs`` and/or ``track_var`` are specified, store the original indices of the selected
    data as a per-observation and/or per-variable annotation of the result data.

    **Computation Parameters**

    1. Combine the masks in ``obs_masks`` and/or ``var_masks`` using
       :py:func:`metacells.tools.mask.combine_masks` passing it ``invert_obs`` and ``invert_var``,
       and ``mask_obs`` and ``mask_var`` as the ``to`` parameter. If either list of masks is empty,
       use the full mask.

    2. If the obtained masks for either the observations or variables is empty, return ``None``.
       Otherwise, return a slice of the full data containing just the observations and variables
       specified by the final masks.
    '''
    ut.log_operation(LOG, adata, 'filter_data')

    if len(obs_masks) == 0:
        obs_mask = np.full(adata.n_obs, True, dtype='bool')
        if mask_obs is not None:
            ut.set_o_data(adata, mask_obs, obs_mask)
    else:
        obs_mask = \
            tl.combine_masks(adata, obs_masks, invert=invert_obs, to=mask_obs)
        if obs_mask is None:
            assert mask_obs is not None
            obs_mask = ut.to_proper_vector(ut.get_o_data(adata, mask_obs))

    if len(var_masks) == 0:
        var_mask = np.full(adata.n_vars, True, dtype='bool')
        if mask_var is not None:
            ut.set_o_data(adata, mask_var, var_mask)
    else:
        var_mask = \
            tl.combine_masks(adata, var_masks, invert=invert_var, to=mask_var)
        if var_mask is None:
            assert mask_var is not None
            var_mask = ut.to_proper_vector(ut.get_v_data(adata, mask_var))

    if not np.any(obs_mask) or not np.any(var_mask):
        return None

    fdata = ut.slice(adata, name=name, tmp=tmp,
                     obs=obs_mask, vars=var_mask,
                     track_obs=track_obs, track_var=track_var)

    return fdata, pd.Series(obs_mask, index=adata.obs_names), \
        pd.Series(var_mask, index=adata.var_names)
