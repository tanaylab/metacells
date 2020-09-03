'''
Downsample Cells
----------------
'''

import logging
from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.preprocessing as pp
import metacells.utilities as ut

__all__ = [
    'downsample_cells',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def downsample_cells(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    downsample_cell_quantile: float = 0.05,
    random_seed: int = 0,
    inplace: bool = True,
    infocus: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasFrame]:
    '''
    Downsample the values ``of`` some data.

    Downsampling is an effective way to get the same number of samples in multiple cells
    (that is, the same number of total UMIs in multiple cells), and serves as an alternative to
    normalization (e.g., working with UMI fractions instead of raw UMI counts).

    Downsampling is especially important when computing correlations between cells. When there is
    high variance between the total UMI count in different cells, then normalization will return
    higher correlation values between cells with a higher total UMI count, which will result in an
    inflated estimation of their similarity to other cells. Downsampling avoids this effect.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Variable-Observation (Gene-Cell) Annotations
        ``<of>|downsample_<samples>_var_per_obs``
            The downsampled data where the total number of samples in each cell is at most
            ``samples``.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas data frame (indexed by the cell and gene
    names).

    If ``infocus`` (default: {infocus}, implies ``inplace``), also makes the result the new focus.

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Compute the total samples in each cell, and use the ``downsample_cell_quantile`` (default:
       {downsample_cell_quantile}) to select our target number of samples. That is, if this quantile
       is 0.1, then 10% of the cells would have less samples than the target and will be unchanged,
       while 90% of the cells will have more samples than the target, and will be downsampled so
       that they will have exactly the target number of samples.

    2. Downsample each cell so that it has at most the selected number of samples. Use the
       ``random_seed`` to allow making this replicable.
    '''
    of, level = \
        ut.log_operation(LOG, adata, 'downsample_cells', of)

    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        total_per_cell = pp.get_per_obs(adata, ut.sum_per).proper

        LOG.log(level, '  downsample_cell_quantile: %s',
                downsample_cell_quantile)
        samples = round(np.quantile(total_per_cell, downsample_cell_quantile))
        LOG.log(level, '  samples: %s', samples)

        downsampled = pp.get_downsample_of_var_per_obs(adata,
                                                       samples=samples,
                                                       random_seed=random_seed)

    if inplace or infocus:
        ut.set_vo_data(adata, downsampled.name,
                       downsampled.matrix, infocus=infocus)
        return None

    if ut.SparseMatrix.am(downsampled.matrix):
        return pd.DataFrame.from_spmatrix(downsampled.matrix,
                                          index=adata.obs_names,
                                          columns=adata.var_names)

    return pd.DataFrame(downsampled.matrix,
                        index=adata.obs_names, columns=adata.var_names)
