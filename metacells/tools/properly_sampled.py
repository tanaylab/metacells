'''
Properly Sampled
----------------
'''

import logging
from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.parameters as pr
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
    *,
    min_cell_total: Optional[int],
    max_cell_total: Optional[int],
    excluded_adata: Optional[AnnData] = None,
    max_excluded_genes_fraction: Optional[float],
    inplace: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Detect cells with a "proper" amount of UMIs.

    Due to both technical effects and natural variance between cells, the total number of UMIs
    varies from cell to cell. We often would like to work on cells that contain a sufficient number
    of UMIs for meaningful analysis; we sometimes also wish to exclude cells which have "too many"
    UMIs.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Observation (Cell) Annotations
        ``properly_sampled_cell``
            A boolean mask indicating whether each cell has a "proper" amount of UMIs.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the observation names).

    **Computation Parameters**

    1. Exclude all cells whose total data is less than the ``min_cell_total`` (no default), unless
       it is ``None``.

    2. Exclude all cells whose total data is more than the ``max_cell_total`` (no default), unless
       it is ``None``.

    3. If ``max_excluded_genes_fraction`` (no default) is not ``None``, then ``excluded_adata`` must
       not be ``None`` and should contain just the excluded genes data for each cell. Exclude all
       cells whose sum of the excluded data divided by the total data is more than the specified
       threshold.
    '''
    _, level = ut.log_operation(LOG, adata, 'find_properly_sampled_cells')

    assert (max_excluded_genes_fraction is None) == (excluded_adata is None)

    data = ut.get_vo_proper(adata, layout='row_major')
    total_of_cells = ut.sum_per(data, per='row')

    cells_mask = np.full(adata.n_obs, True, dtype='bool')

    if min_cell_total is not None:
        LOG.debug('  min_cell_total: %s', min_cell_total)
        cells_mask = cells_mask & (total_of_cells >= min_cell_total)

    if max_cell_total is not None:
        LOG.debug('  max_cell_total: %s', max_cell_total)
        cells_mask = cells_mask & (total_of_cells <= max_cell_total)

    if excluded_adata is not None:
        assert max_excluded_genes_fraction is not None
        LOG.debug('  max_excluded_genes_fraction: %s',
                  max_excluded_genes_fraction)
        excluded_data = ut.get_vo_proper(excluded_adata, layout='row_major')
        excluded_of_cells = ut.sum_per(excluded_data, per='row')
        if np.min(total_of_cells) == 0:
            total_of_cells = np.copy(total_of_cells)
            total_of_cells[total_of_cells == 0] = 1
        excluded_fraction = excluded_of_cells / total_of_cells
        cells_mask = \
            cells_mask & (excluded_fraction <= max_excluded_genes_fraction)

    if inplace:
        ut.set_o_data(adata, 'properly_sampled_cell', cells_mask)
        return None

    ut.log_mask(LOG, level, 'properly_sampled_cell', cells_mask)

    return pd.Series(cells_mask, index=adata.obs_names)


@ut.timed_call()
@ut.expand_doc()
def find_properly_sampled_genes(
    adata: AnnData,
    *,
    min_gene_total: int = pr.properly_sampled_min_gene_total,
    inplace: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Detect genes with a "proper" amount of UMIs.

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
        ``properly_sampled_gene``
            A boolean mask indicating whether each gene has a "proper" number of UMIs.

    If ``inplace`` (default: {inplace}), this is written to the data and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    **Computation Parameters**

    1. Exclude all genes whose total data is less than the ``min_gene_total`` (default:
       {min_gene_total}).
    '''
    _, level = ut.log_operation(LOG, adata, 'find_properly_sampled_genes')

    data = ut.get_vo_proper(adata, layout='column_major')
    total_of_genes = ut.sum_per(data, per='column')

    LOG.debug('  min_gene_total: %s', min_gene_total)
    genes_mask = total_of_genes >= min_gene_total

    if inplace:
        ut.set_v_data(adata, 'properly_sampled_gene', genes_mask)
        return None

    ut.log_mask(LOG, level, 'properly_sampled_gene', genes_mask)

    return pd.Series(genes_mask, index=adata.obs_names)
