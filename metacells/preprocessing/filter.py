'''
Filter
------
'''

import logging
from typing import List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
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
    mask_obs: Optional[str] = None,
    mask_var: Optional[str] = None,
    track_obs: Optional[str] = None,
    track_var: Optional[str] = None,
    name: Optional[str] = None,
    tmp: bool = False,
    invalidated_prefix: Optional[str] = None,
    invalidated_suffix: Optional[str] = None,
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

    If ``name`` is specified, this will be the logging name of the new data. Otherwise, it will be
    unnamed.

    If ``tmp`` (default: {tmp}) is set, logging of modifications to the result will use the
    ``DEBUG`` logging level. By default, logging of modifications is done using the ``INFO`` logging
    level.

    If ``mask_obs`` and/or ``mask_var`` are specified, store the mask of the selected data as a
    per-observation and/or per-variable annotation of the full ``adata``.

    If ``track_obs`` and/or ``track_var`` are specified, store the original indices of the selected
    data as a per-observation and/or per-variable annotation of the result data.

    **Computation Parameters**

    1. For each of the mask in ``masks``, fetch it. Silently ignore missing masks if the name has a
       ``?`` suffix. Invert the mask if the name has a ``~`` suffix. Bitwise-AND the appropriate
       (observations or variables) mask with the result.

    2. If the final observations or variables mask is empty, return None. Otherwise, return a slice
       of the full data containing just the observations and variables specified by the final masks.
       If ``invalidated_prefix`` (default: {invalidated_prefix}) and/or ``invalidated_suffix``
       (default: {invalidated_suffix}) are specified, then invalidated data will not be removed;
       instead it will be renamed with the addition of the provided prefix and/or suffix.
    '''
    _, level = ut.log_operation(LOG, adata, 'filter_data')

    obs_mask = np.full(adata.n_obs, True, dtype='bool')
    vars_mask = np.full(adata.n_vars, True, dtype='bool')

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

        if per == 'o':
            obs_mask = obs_mask & mask
            per_name = 'observations'
        elif per == 'v':
            vars_mask = vars_mask & mask
            per_name = 'variables'
        else:
            raise ValueError('the data: %s '
                             'is not per-observation or per-variable'
                             % mask_name)

        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug('  %s (%s): %s', log_mask_name,
                      per_name, ut.mask_description(mask))

    if mask_obs is not None:
        ut.set_o_data(adata, mask_obs, obs_mask, ut.ALWAYS_SAFE)

    elif LOG.isEnabledFor(level):
        ut.log_mask(LOG, level, 'observations', obs_mask)

    if mask_var is not None:
        ut.set_v_data(adata, mask_var, vars_mask, ut.ALWAYS_SAFE)

    elif LOG.isEnabledFor(level):
        ut.log_mask(LOG, level, 'variables', vars_mask)

    if not np.any(obs_mask) or not np.any(vars_mask):
        return None

    fdata = ut.slice(adata, name=name, tmp=tmp,
                     obs=obs_mask, vars=vars_mask,
                     track_obs=track_obs, track_var=track_var,
                     invalidated_prefix=invalidated_prefix,
                     invalidated_suffix=invalidated_suffix)

    return fdata, pd.Series(obs_mask, index=adata.obs_names), \
        pd.Series(vars_mask, index=adata.var_names)
