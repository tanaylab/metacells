'''
Downsample
----------
'''

from typing import Optional, Union

import numpy as np
from anndata import AnnData

import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    'downsample_cells',
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def downsample_cells(
    adata: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    *,
    downsample_min_cell_quantile: float = pr.downsample_min_cell_quantile,
    downsample_min_samples: float = pr.downsample_min_samples,
    downsample_max_cell_quantile: float = pr.downsample_max_cell_quantile,
    random_seed: int = pr.random_seed,
    inplace: bool = True,
) -> Optional[ut.PandasFrame]:
    '''
    Downsample the values of ``what`` (default: {what}) data.

    Downsampling is an effective way to get the same number of samples in multiple cells
    (that is, the same number of total UMIs in multiple cells), and serves as an alternative to
    normalization (e.g., working with UMI fractions instead of raw UMI counts).

    Downsampling is especially important when computing correlations between cells. When there is
    high variance between the total UMI count in different cells, then normalization will return
    higher correlation values between cells with a higher total UMI count, which will result in an
    inflated estimation of their similarity to other cells. Downsampling avoids this effect.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    **Returns**

    Variable-Observation (Gene-Cell) Annotations
        ``downsampled``
            The downsampled data where the total number of samples in each cell is at most
            ``samples``.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas data frame (indexed by the cell and gene
    names).

    **Computation Parameters**

    1. Compute the total samples in each cell.

    2. Decide on the value to downsample to. We would like all cells to end up with at least some
       reasonable number of samples (total UMIs) ``downsample_min_samples`` (default:
       {downsample_min_samples}). We'd also like all (most) cells to end up with the highest
       reasonable downsampled total number of samples, so if possible we increase the number of
       samples, as long as at most ``downsample_min_cell_quantile`` (default:
       {downsample_min_cell_quantile}) cells will have lower number of samples. We'd also like all
       (most) cells to end up with the same downsampled total number of samples, so if we have to we
       decrease the number of samples to ensure at most ``downsample_max_cell_quantile`` (default:
       {downsample_max_cell_quantile}) cells will have a lower number of samples.

    3. Downsample each cell so that it has at most the selected number of samples. Use the
       ``random_seed`` to allow making this replicable.
    '''
    total_per_cell = ut.get_o_numpy(adata, what, sum=True)

    samples = int(round(min(max(downsample_min_samples,
                                np.quantile(total_per_cell, downsample_min_cell_quantile)),
                            np.quantile(total_per_cell, downsample_max_cell_quantile))))

    ut.log_calc('samples', samples)

    data = ut.get_vo_proper(adata, what, layout='row_major')
    assert ut.matrix_dtype(data) == 'float32'
    downsampled = ut.downsample_matrix(data, per='row', samples=samples,
                                       random_seed=random_seed)
    if inplace:
        ut.set_vo_data(adata, 'downsampled', downsampled)
        return None

    return ut.to_pandas_frame(downsampled,
                              index=adata.obs_names, columns=adata.var_names)
