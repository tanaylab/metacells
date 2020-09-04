'''
Collect the Final Metacells
---------------------------
'''

import logging
from typing import Iterable, Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'collect_metacells',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def collect_metacells(
    adata: AnnData,
    cdata: Union[AnnData, Iterable[AnnData]],
    *,
    base_indices: str = 'obs_base_index',
    metacells: str = 'metacell',
    inplace: bool = True,
) -> Optional[pd.Series]:
    '''
    Collect the ``metacells`` annotation from the clean ``cdata`` to create the same annotation for
    the full ``adata``.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Observations (Cell) Annotations
        ``metacell``
            The index of the metacell each cell belongs to. This is ``-1`` for outlier cells and
            ``-2`` for excluded cells.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the observation names).

    **Computation Parameters**

    1. Iterate on all the ``cdata`` clean data, which we computed ``metacells`` (per-observation
       annotations) for.

    2. Combine the metacell indices from the clean data into the final metacells, using the
       ``base_indices`` pre-observation (cell) annotation to map the clean cell indices to the full
       cell indices. If multiple clean data assign a metacell index to the same cell, the last one
       wins. The clean metacell indices are shifted so that the result is unique metacell indices
       across the whole data.

    3. Cells which were "excluded", that is, not included in any of the clean data, are given the
       metacell index ``-2``. This distinguishes them from "outlier" cells which were not placed
       in any metacell in the clean data they were included in.
    '''
    ut.log_operation(LOG, adata, 'collect_metacells')
    metacell_of_all_cells = np.full(adata.n_obs, -2, dtype='int32')

    if isinstance(cdata, AnnData):
        cdata = [cdata]

    metacells_count = 0
    for cdatum in cdata:
        metacell_of_clean_cells = np.copy(ut.get_o_data(cdatum, metacells))
        grouped_clean_cells_mask = metacell_of_clean_cells >= 0

        clean_metacells_count = np.max(metacell_of_clean_cells) + 1
        metacell_of_clean_cells[grouped_clean_cells_mask] += metacells_count
        metacells_count += clean_metacells_count

        clean_cell_indices = ut.get_o_data(cdatum, base_indices)
        metacell_of_all_cells[clean_cell_indices] = metacell_of_clean_cells

        if LOG.isEnabledFor(logging.DEBUG):
            name = ut.get_name(cdatum)
            if name is None:
                LOG.debug('  - collect metacells from clean data')
            else:
                LOG.debug('  - collect metacells from: %s', name)

            LOG.debug('    metacells: %s', clean_metacells_count)
            LOG.debug('    cells: %s',
                      ut.mask_description(grouped_clean_cells_mask))
            LOG.debug('    outliers: %s',
                      ut.mask_description(~grouped_clean_cells_mask))

    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug('  collected metacells: %s',
                  np.max(metacell_of_all_cells) + 1)
        LOG.debug('  collected cells: %s',
                  ut.mask_description(metacell_of_all_cells >= 0))
        LOG.debug('  collected outliers: %s',
                  ut.mask_description(metacell_of_all_cells == -1))
        LOG.debug('  collected excluded: %s',
                  ut.mask_description(metacell_of_all_cells == -2))

    if inplace:
        ut.set_o_data(adata, metacells, metacell_of_all_cells,
                      log_value=lambda: str(np.max(metacell_of_all_cells) + 1))
        return None

    return pd.Series(metacell_of_all_cells, index=adata.obs_names)
