"""
Selection
---------
"""

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    "extract_selected_data",
]


# pylint: disable=dangerous-default-value


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def extract_selected_data(  # pylint: disable=too-many-branches, too-many-statements
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    name: Optional[str] = ".select",
    downsample_min_samples: float = pr.select_downsample_min_samples,
    downsample_min_cell_quantile: float = pr.select_downsample_min_cell_quantile,
    downsample_max_cell_quantile: float = pr.select_downsample_max_cell_quantile,
    min_gene_relative_variance: Optional[float],  # = pr.select_min_gene_relative_variance,
    min_genes: int = pr.select_min_genes,
    min_gene_total: Optional[int] = pr.select_min_gene_total,
    min_gene_top3: Optional[int] = pr.select_min_gene_top3,
    additional_gene_masks: List[str] = ["&~lateral_gene"],
    random_seed: int,
    top_level: bool = True,
) -> AnnData:
    """
    Select a subset of ``what`` (default: {what} data, to compute metacells by.

    When computing metacells (or clustering cells in general), it makes sense to use a subset of the genes for computing
    cell-cell similarity, for both technical (e.g., too low an expression level) and biological (e.g., ignoring
    bookkeeping and cell cycle genes) reasons. The steps provided here are expected to be generically useful, but as
    always specific data sets may require custom gene selection steps on a case-by-case basis.

    **Input**

    A presumably "clean" Annotated ``adata``, where the observations are cells and the variables are
    genes, where ``what`` is a per-variable-per-observation matrix or the name of a
    per-variable-per-observation annotation containing such a matrix.

    Will obey the following annotations in the full ``adata``, if they exist:

    Variable (Gene) Annotations

        ``select_gene``
            If exists, force a mask of genes to use as "select" genes, ignoring everything else.

        ``lateral_gene``
            A boolean mask of genes which are lateral from being chosen as "select" genes based
            on their name.

    **Returns**

    Returns annotated sliced data containing the "select" subset of the original data. By default,
    the ``name`` of this data is {name}. If no selects were selected, return ``None``.

    Also sets the following annotations in the full ``adata``:

    Unstructured Annotations
        ``downsample_samples``
            The target total number of samples in each downsampled cell.

    Observation-Variable (Cell-Gene) Annotations:
        ``downsampled``
            The downsampled data where the total number of samples in each cell is at most ``downsample_samples``.

    Variable (Gene) Annotations
        ``high_total_gene``
            A boolean mask of genes with "high" expression level (unless a ``select_gene`` mask exists).

        ``high_relative_variance_gene``
            A boolean mask of genes with "high" normalized variance, relative to other genes with a similar expression
            level (unless a ``select_gene`` mask exists).

        ``selected_gene``
            A boolean mask of the actually selected genes.

    **Computation Parameters**

    0. If a ``select_gene`` mask exists, just use these genes and go directly to the last step 6.

    1. Invoke :py:func:`metacells.tools.downsample.downsample_cells` to downsample the cells to the same total number of
       UMIs, using the ``downsample_min_samples`` (default: {downsample_min_samples}), ``downsample_min_cell_quantile``
       (default: {downsample_min_cell_quantile}), ``downsample_max_cell_quantile`` (default:
       {downsample_max_cell_quantile}) and the ``random_seed`` (non-zero for reproducible results).

    2. Invoke :py:func:`metacells.tools.high.find_high_total_genes` to select high-expression genes (based on the
       downsampled data), using ``min_gene_total``.

    3. Invoke :py:func:`metacells.tools.high.find_high_relative_variance_genes` to select high-variance genes (based on
       the downsampled data), using ``min_gene_relative_variance``.

    4. Compute the set of genes that pass the above test, as well as match the ``additional_gene_masks`` (default:
       {additional_gene_masks}).

    5. If we found less than ``min_genes`` genes (default: {min_genes}, and ``min_gene_relative_variance`` was
       specified, try to achieve the required minimal number of genes by reducing the ``min_gene_relative_variance``. In
       extreme cases, give up on the relative variance requirement altogether.

    6. Invoke :py:func:`metacells.tools.filter.filter_data` to slice just the selected genes.
    """

    assert min_genes > 0

    tl.downsample_cells(
        adata,
        what,
        downsample_min_samples=downsample_min_samples,
        downsample_min_cell_quantile=downsample_min_cell_quantile,
        downsample_max_cell_quantile=downsample_max_cell_quantile,
        random_seed=random_seed,
    )

    results: Optional[Tuple[AnnData, ut.PandasSeries, ut.PandasSeries]] = None
    if ut.has_data(adata, "select_gene"):
        results = tl.filter_data(
            adata,
            name=name,
            top_level=top_level,
            track_var="full_gene_index",
            var_masks=["&select_gene"],
            mask_var="selected_gene",
        )
        assert results is not None

    else:
        var_masks = []

        if min_gene_top3 is not None:
            var_masks.append("&high_top3_gene")
            tl.find_high_topN_genes(adata, "downsampled", topN=3, min_gene_topN=min_gene_top3)

        if min_gene_total is not None:
            var_masks.append("&high_total_gene")
            tl.find_high_total_genes(adata, "downsampled", min_gene_total=min_gene_total)

        if min_gene_relative_variance is not None:
            var_masks.append("&high_relative_variance_gene")
            tl.find_high_relative_variance_genes(
                adata, "downsampled", min_gene_relative_variance=min_gene_relative_variance
            )

        candidate_genes_mask = tl.combine_masks(adata, var_masks + additional_gene_masks)
        assert candidate_genes_mask is not None
        if np.sum(candidate_genes_mask.values) >= min_genes:
            results = tl.filter_data(
                adata,
                name=name,
                top_level=top_level,
                track_var="full_gene_index",
                var_masks=var_masks + additional_gene_masks,
                mask_var="selected_gene",
            )

    if results is None and min_gene_relative_variance is not None:
        valid_candidate_genes_mask = tl.combine_masks(adata, var_masks[:-1] + additional_gene_masks)
        assert valid_candidate_genes_mask is not None

        is_valid_set = False
        if np.sum(valid_candidate_genes_mask.values) <= min_genes:
            var_masks.pop()

        else:
            while True:
                high_gene_relative_variance = min_gene_relative_variance
                min_gene_relative_variance -= 1 / 8
                tl.find_high_relative_variance_genes(
                    adata, "downsampled", min_gene_relative_variance=min_gene_relative_variance
                )
                valid_candidate_genes_mask = tl.combine_masks(adata, var_masks + additional_gene_masks)
                assert valid_candidate_genes_mask is not None
                if np.sum(valid_candidate_genes_mask.values) >= min_genes:
                    is_valid_set = True
                    break

            for _ in range(4):
                mid_gene_relative_variance = (high_gene_relative_variance + min_gene_relative_variance) / 2.0
                tl.find_high_relative_variance_genes(
                    adata, "downsampled", min_gene_relative_variance=mid_gene_relative_variance
                )
                mid_candidate_genes_mask = tl.combine_masks(adata, var_masks + additional_gene_masks)
                assert mid_candidate_genes_mask is not None
                if np.sum(mid_candidate_genes_mask.values) >= min_genes:
                    min_gene_relative_variance = mid_gene_relative_variance
                    valid_candidate_genes_mask = mid_candidate_genes_mask
                    is_valid_set = True
                else:
                    high_gene_relative_variance = mid_gene_relative_variance
                    is_valid_set = False

        if not is_valid_set:
            ut.set_v_data(adata, "high_relative_variance_gene", valid_candidate_genes_mask.values)

        results = tl.filter_data(
            adata,
            name=name,
            top_level=top_level,
            track_var="full_gene_index",
            var_masks=var_masks + additional_gene_masks,
            mask_var="selected_gene",
        )

    if results is None:
        raise ValueError(f"Failed to select {min_genes} genes")

    return results[0]
