'''
Result Metacells
----------------
'''

import logging
from typing import Iterable, Optional, Union

from anndata import AnnData

import metacells.preprocessing as pp
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    'collect_result_metacells',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def collect_result_metacells(
    adata: AnnData,
    cdata: Union[AnnData, Iterable[AnnData]],
    of: Optional[str] = None,
    *,
    name: str = 'METACELLS',
    tmp: bool = False,
    intermediate: bool = True,
) -> Optional[AnnData]:
    '''
    Collect the result metacells computed using the clean data.

    **Input**

    The full :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where
    the observations are cells and the variables are genes, and the clean data we
    have computed metacells for.

    **Returns**

    Annotated metacell data containing for each observation the sum ``of`` the data (by default,
    the focus) of the cells for each metacell.

    Also sets the following in the full data:

    Observations (Cell) Annotations
        ``metacell``
            The index of the metacell each cell belongs to. This is ``-1`` for outlier cells and
            ``-2`` for excluded cells.

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.apply_metacells.apply_metacells` to add a ``metacell``
       per-observation (cell) annotation to the full data.

    2. Invoke :py:func:`metacells.preprocessing.group_data.group_obs_data` to sum the ``of`` data
       into a new metacells annotated data, using the ``name`` (default: {name}) and ``tmp``
       (default: {tmp}).
    '''
    ut.log_pipeline_step(LOG, adata, 'collect_result_metacells')

    tl.apply_metacells(adata, cdata)
    return pp.group_obs_data(adata, of=of, groups='metacell',
                             name=name, tmp=tmp, intermediate=intermediate)
