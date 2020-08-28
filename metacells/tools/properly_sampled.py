'''
Detect Properly-Sampled Cells and Genes
---------------------------------------
'''

import logging
from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.preprocessing as pp
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
    min_total_of_cells: Optional[int] = 800,
    max_total_of_cells: Optional[int] = None,
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

    1. Exclude all cells whose total data is less than the ``min_total_of_cells`` (default:
       {min_total_of_cells}), unless it is ``None``

    2. Exclude all cells whose total data is more than the ``max_total_of_cells`` (default:
       {max_total_of_cells}), unless it is ``None``
    '''
    of, level = ut.log_operation(LOG, adata, 'find_properly_sampled_cells', of)

    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        total_of_cells = pp.get_per_obs(adata, ut.sum_per).proper
        cells_mask = np.full(adata.n_obs, True, dtype='bool')

        if min_total_of_cells is not None:
            LOG.log(level, '  min_total_of_cells: %s', min_total_of_cells)
            cells_mask = cells_mask & (total_of_cells >= min_total_of_cells)

        if max_total_of_cells is not None:
            LOG.log(level, '  max_total_of_cells: %s',
                    max_total_of_cells)
            cells_mask = \
                cells_mask & (total_of_cells <= max_total_of_cells)

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
    min_total_of_genes: int = 1,
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

        Provide additional optional criteria for "properly sampled genes"?

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

    1. Exclude all genes whose total data is less than the ``min_total_of_genes`` (default:
       {min_total_of_genes}).
    '''
    of, level = ut.log_operation(LOG, adata, 'find_properly_sampled_genes', of)

    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        total_of_genes = pp.get_per_var(adata, ut.sum_per).proper
        LOG.log(level, '  min_total_of_genes: %s', min_total_of_genes)
        genes_mask = total_of_genes >= min_total_of_genes

    if inplace:
        ut.set_v_data(adata, 'properly_sampled_genes',
                      genes_mask, ut.SAFE_WHEN_SLICING_OBS)
        return None

    ut.log_mask(LOG, level, 'properly_sampled_genes', genes_mask)

    return pd.Series(genes_mask, index=adata.obs_names)
