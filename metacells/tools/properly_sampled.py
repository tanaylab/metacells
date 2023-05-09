"""
Properly Sampled
----------------
"""

from typing import Optional
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    "compute_excluded_gene_umis",
    "find_properly_sampled_cells",
    "find_properly_sampled_genes",
]


def compute_excluded_gene_umis(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
) -> None:
    """
    Given an ``excluded_gene`` mask, compute the total ``excluded_umis`` of each cell.
    """
    umis_per_gene_per_cell = ut.get_vo_proper(adata, what, layout="column_major")
    excluded_genes_mask = ut.get_v_numpy(adata, "excluded_gene")
    umis_per_excluded_gene_per_cell = umis_per_gene_per_cell[:, excluded_genes_mask]
    umis_per_excluded_gene_per_cell = ut.to_layout(umis_per_excluded_gene_per_cell, layout="row_major")
    excluded_umis_per_cell = ut.sum_per(umis_per_excluded_gene_per_cell, per="row")
    ut.set_o_data(adata, "excluded_umis", excluded_umis_per_cell)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def find_properly_sampled_cells(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    min_cell_total: Optional[int],
    max_cell_total: Optional[int],
    max_excluded_genes_fraction: Optional[float],
    inplace: bool = True,
) -> Optional[ut.PandasSeries]:
    """
    Detect cells with a "proper" amount of ``what`` (default: {what}) data.

    Due to both technical effects and natural variance between cells, the total number of UMIs
    varies from cell to cell. We often would like to work on cells that contain a sufficient number
    of UMIs for meaningful analysis; we sometimes also wish to exclude cells which have "too many"
    UMIs.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

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

    3. If ``max_excluded_genes_fraction`` (no default) is not ``None``, then exclude all cells whose sum of the excluded
       data (as defined by the ``excluded_gene`` mask) divided by the total data is more than the specified threshold.
    """
    total_umis_per_cell = ut.get_o_numpy(adata, what, sum=True)

    cells_mask = np.full(adata.n_obs, True, dtype="bool")

    if min_cell_total is not None:
        cells_mask = cells_mask & (total_umis_per_cell >= min_cell_total)

    if max_cell_total is not None:
        cells_mask = cells_mask & (total_umis_per_cell <= max_cell_total)

    if max_excluded_genes_fraction is not None:
        if not ut.has_data(adata, "excluded_umis"):
            compute_excluded_gene_umis(adata, what)
        excluded_umis_per_cell = ut.get_o_numpy(adata, "excluded_umis")
        excluded_umis_fraction_per_cell = excluded_umis_per_cell / total_umis_per_cell
        cells_mask = cells_mask & (excluded_umis_fraction_per_cell <= max_excluded_genes_fraction)

    if inplace:
        ut.set_o_data(adata, "properly_sampled_cell", cells_mask)
        return None

    ut.log_return("properly_sampled_cell", cells_mask)
    return ut.to_pandas_series(cells_mask, index=adata.obs_names)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def find_properly_sampled_genes(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    min_gene_total: int = pr.properly_sampled_min_gene_total,
    inplace: bool = True,
) -> Optional[ut.PandasSeries]:
    """
    Detect genes with a "proper" amount of ``what`` (default: {what}) data.

    Due to both technical effects and natural variance between genes, the expression of genes varies
    greatly between cells. This is exactly the information we are trying to analyze. We often would
    like to work on genes that have a sufficient level of expression for meaningful analysis.
    Specifically, it doesn't make sense to analyze genes that have zero expression in all the cells.

    .. todo::

        Provide additional optional criteria for "properly sampled genes"?

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    **Returns**

    Variable (Gene) Annotations
        ``properly_sampled_gene``
            A boolean mask indicating whether each gene has a "proper" number of UMIs.

    If ``inplace`` (default: {inplace}), this is written to the data and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    **Computation Parameters**

    1. Exclude all genes whose total data is less than the ``min_gene_total`` (default:
       {min_gene_total}).
    """
    total_of_genes = ut.get_v_numpy(adata, what, sum=True)

    genes_mask = total_of_genes >= min_gene_total

    if inplace:
        ut.set_v_data(adata, "properly_sampled_gene", genes_mask)
        return None

    ut.log_return("properly_sampled_gene", genes_mask)
    return ut.to_pandas_series(genes_mask, index=adata.obs_names)
