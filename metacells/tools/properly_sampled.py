'''
Detect properly-sampled cells and genes.
'''

from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'find_properly_sampled_cells',
    'find_properly_sampled_genes',
]


@ut.call()
@ut.expand_doc()
def find_properly_sampled_cells(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    minimal_total_umis_of_cells: Optional[int] = 800,
    maximal_total_umis_of_cells: Optional[int] = None,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[pd.Series]:
    '''
    Detect properly-sampled cells.

    Due to both technical effects and natural variance between cells, the total number of UMIs
    varies from cell to cell.

    We often would like to work on cells that contain a sufficient number of UMIs for meaningful
    analysis; we sometimes also wish to exclude cells which have "too many" UMIs.

    **Input**

    A :py:func:`metacells.utilities.preparation.prepare`-ed annotated ``adata``, where the
    observations are cells and the variables are genes, containing the UMIs count in the ``of``
    (default: the focus) per-variable-per-observation data.

    **Returns**

    Observation (Cell) Annotations
        ``properly_sampled_cells``
            A boolean mask indicating whether each cell has a "proper" amount of samples.

    If ``inplace`` (default: {inplace}), this is written to ``adata`` and the function returns
    ``None``. Otherwise this is returned as a Pandas series (indexed by the observation names).

    If not ``intermediate``, this discards all the intermediate data used (e.g. sums). Otherwise,
    such data is kept for future reuse.

    **Computation Parameters**

    Given an annotated ``adata``, where the variables are cell RNA profiles and the observations are
    gene UMI counts, do the following:

    1. Exclude all cells whose total number of umis is less than the
       ``minimal_total_umis_of_cells``, unless it is ``None`` (default:
       {minimal_total_umis_of_cells}).

    2. Exclude all cells whose total number of umis is more than the
       ``maximal_total_umis_of_cells``, unless it is ``None`` (default:
       {maximal_total_umis_of_cells}).
    '''
    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        total_umis_of_cells = ut.get_per_obs(adata, ut.sum_axis).data
        cells_mask = np.full(adata.n_obs, True, dtype='bool')

        if minimal_total_umis_of_cells is not None:
            cells_mask = \
                cells_mask & (total_umis_of_cells >=
                              minimal_total_umis_of_cells)

        if maximal_total_umis_of_cells is not None:
            cells_mask = \
                cells_mask & (total_umis_of_cells <=
                              maximal_total_umis_of_cells)

    if inplace:
        adata.var['properly_sampled_cells'] = cells_mask
        ut.safe_slicing_data('properly_sampled_cells',
                             ut.SAFE_WHEN_SLICING_OBS)
        return None

    return pd.Series(cells_mask, index=adata.obs_names)


@ut.call()
@ut.expand_doc()
def find_properly_sampled_genes(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    minimal_total_umis_of_genes: Optional[int] = 1,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[pd.Series]:
    '''
    Detect properly-sampled genes.

    Due to both technical effects and natural variance between genes, the expression of genes
    varies greatly between cells. This is exactly the information we are trying to analyze.

    We often would like to work on genes that have a sufficient level of expression for meaningful
    analysis. Specifically, it hardly (if ever) make sense to consider genes that have zero
    expression in all the cells.

    .. todo::

        Provide additional optional criteria for "properly sampled genes", such as a minimal value
        for their maximal fraction and/or number of UMIs?

    **Input**

    A :py:func:`metacells.utilities.preparation.prepare`-ed annotated ``adata``, where the
    observations are cells and the variables are genes, containing the UMIs count in the ``of``
    (default: the focus) per-variable-per-observation data.

    **Returns**

    Variable (Gene) Annotations
        ``properly_sampled_genes``
            A boolean mask indicating whether each gene has a "proper" number of samples.

    If ``inplace`` (default: {inplace}), this is written to ``adata`` and the function returns
    ``None``. Otherwise this is returned as a Pandas series (indexed by the variable names).

    If not ``intermediate``, this discards all the intermediate data used (e.g. sums). Otherwise,
    such data is kept for future reuse.

    **Computation Parameters**

    Given an annotated ``adata``, where the variables are cell RNA profiles and the observations are
    gene UMI counts, do the following:

    1. Exclude all genes whose total number of umis is less than the
       ``minimal_total_umis_of_genes``, unless it is ``None`` (default:
       {minimal_total_umis_of_genes}).
    '''
    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        total_umis_of_genes = ut.get_per_var(adata, ut.sum_axis).data
        genes_mask = np.full(adata.n_obs, True, dtype='bool')

        if minimal_total_umis_of_genes is not None:
            genes_mask = \
                genes_mask & (total_umis_of_genes >=
                              minimal_total_umis_of_genes)

    if inplace:
        adata.var['properly_sampled_genes'] = genes_mask
        ut.safe_slicing_data('properly_sampled_genes',
                             ut.SAFE_WHEN_SLICING_OBS)
        return None

    return pd.Series(genes_mask, index=adata.obs_names)
