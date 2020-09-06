'''
Group Data
----------
'''

import logging
from typing import Optional, Union

import numpy as np  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'group_obs_data',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def group_obs_data(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    groups: Union[str, ut.Vector],
    name: Optional[str] = None,
    tmp: bool = False,
    intermediate: bool = True,
) -> Optional[AnnData]:
    '''
    Compute new data which has the sum ``of`` some data of the observations (cells) for each group.

    For example, having computed a metacell index for each cell, compute the per-metacell data
    for further analysis.

    If ``groups`` is a string, it is expected to be the name of a per-observation vector annotation.
    Otherwise it should be a vector. The group indices should be integers, where negative values
    indicate "no group" and non-negative values indicate the index of the group to which each
    observation (cell) belongs to.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` ``adata``, where the observations are cells
    and the variables are genes.

    **Returns**

    An annotated data where each observation is the sum of the group of original observations
    (cells). Observations with a negative group index are discarded. If all observations are
    discarded, return ``None``.

    The new data will contain only:

    * An ``X`` member holding the summed-per-group ``of`` data. This will also be the focus.

    * Any per-variable (gene) data or per-variable-per-variable data which is safe when slicing
      observations (cells).

    * A new ``grouped`` per-observation data which counts, for each group, the numnber
      of grouped observations summed into it.

    If ``name`` is specified, this will be the logging name of the new data. Otherwise, it will be
    unnamed.

    If ``tmp`` (default: {tmp}) is set, logging of modifications to the result will use the
    ``DEBUG`` logging level. By default, logging of modifications is done using the ``INFO`` logging
    level.

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. cached
    layouts) for future reuse. Otherwise, discard it.
    '''
    ut.log_operation(LOG, adata, 'group_obs_data')
    level = ut.get_log_level(adata)

    with ut.focus_on(ut.get_vo_data, adata, of, layout='row_major',
                     intermediate=intermediate) as data:
        group_of_cells = \
            ut.get_vector_parameter_data(LOG, level, adata, groups,
                                         per='o', name='groups')
        assert group_of_cells is not None

        results = ut.sum_groups(data, group_of_cells, per='row')
        if results is None:
            return None
        summed_data, cell_counts = results

        gdata = AnnData(summed_data)
        ut.setup(gdata, name=name, x_name=ut.get_focus_name(adata), tmp=tmp)

        ut.set_o_data(gdata, 'grouped', cell_counts,
                      log_value=lambda: str(np.mean(cell_counts)))

        for v_name, v_value in adata.var.items():
            if ut.safe_slicing_mask(v_name).obs:
                ut.set_v_data(gdata, v_name, v_value)

        for vv_name, vv_value in adata.varp.items():
            if ut.safe_slicing_mask(vv_name).obs:
                ut.set_vv_data(gdata, vv_name, vv_value)

        return gdata
