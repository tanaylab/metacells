'''
Detect properly-sampled cells and genes.
'''

import logging
from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'find_properly_sampled_cells',
    'find_properly_sampled_genes',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def find_properly_sampled_cells(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    minimal_total_of_cells: Optional[int] = 800,
    maximal_total_of_cells: Optional[int] = None,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Detect cells with a "proper" amount ``of`` some data sampled (by default, the focus).

    Due to both technical effects and natural variance between cells, the total number of UMIs
    varies from cell to cell. We often would like to work on cells that contain a sufficient number
    of UMIs for meaningful analysis; we sometimes also wish to exclude cells which have "too many"
    UMIs.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Observation (Cell) Annotations
        ``properly_sampled_cells``
            A boolean mask indicating whether each cell has a "proper" amount of samples.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the observation names).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Exclude all cells whose total data is less than the ``minimal_total_of_cells`` (default:
       {minimal_total_of_cells}), unless it is ``None``

    2. Exclude all cells whose total data is more than the ``maximal_total_of_cells`` (default:
       {maximal_total_of_cells}), unless it is ``None``
    '''
    of, level = ut.log_operation(LOG, adata, 'find_properly_sampled_cells', of)

    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        total_of_cells = ut.get_per_obs(adata, ut.sum_per).proper
        cells_mask = np.full(adata.n_obs, True, dtype='bool')

        if minimal_total_of_cells is not None:
            LOG.log(level, '  minimal_total_of_cells: %s',
                    minimal_total_of_cells)
            cells_mask = \
                cells_mask & (total_of_cells >= minimal_total_of_cells)

        if maximal_total_of_cells is not None:
            LOG.log(level, '  maximal_total_of_cells: %s',
                    maximal_total_of_cells)
            cells_mask = \
                cells_mask & (total_of_cells <= maximal_total_of_cells)

    if inplace:
        ut.set_o_data(adata, 'properly_sampled_cells',
                      cells_mask, ut.SAFE_WHEN_SLICING_OBS)
        return None

    ut.log_mask(LOG, level, 'properly_sampled_cells', cells_mask)

    return pd.Series(cells_mask, index=adata.obs_names)


@ut.timed_call()
@ut.expand_doc()
def find_properly_sampled_genes(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    minimal_total_of_genes: int = 1,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Detect genes with a "proper" amount ``of`` some data samples (by default, the focus).

    Due to both technical effects and natural variance between genes, the expression of genes varies
    greatly between cells. This is exactly the information we are trying to analyze. We often would
    like to work on genes that have a sufficient level of expression for meaningful analysis.
    Specifically, it doesn't make sense to analyze genes that have zero expression in all the cells.

    .. todo::

        Provide additional optional criteria for "properly sampled genes", such as a minimal value
        for their maximal fraction and/or number of UMIs?

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Variable (Gene) Annotations
        ``properly_sampled_genes``
            A boolean mask indicating whether each gene has a "proper" number of samples.

    If ``inplace`` (default: {inplace}), this is written to the data and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Exclude all genes whose total data is less than the ``minimal_total_of_genes`` (default:
       {minimal_total_of_genes}).
    '''
    of, level = ut.log_operation(LOG, adata, 'find_properly_sampled_genes', of)

    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        total_of_genes = ut.get_per_var(adata, ut.sum_per).proper
        LOG.log(level, '  minimal_total_of_genes: %s',
                minimal_total_of_genes)
        genes_mask = total_of_genes >= minimal_total_of_genes

    if inplace:
        ut.set_v_data(adata, 'properly_sampled_genes',
                      genes_mask, ut.SAFE_WHEN_SLICING_OBS)
        return None

    ut.log_mask(LOG, level, 'properly_sampled_genes', genes_mask)

    return pd.Series(genes_mask, index=adata.obs_names)
