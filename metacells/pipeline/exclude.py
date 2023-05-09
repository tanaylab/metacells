"""
Exclude
-------

Raw single-cell RNA sequencing data is notoriously noisy and "dirty". The pipeline steps here performs initial analysis
of the data and exclude some of it, so it would not harm the metacells computation. The steps provided here are expected
to be generically useful, but as always specific data sets may require custom cleaning steps on a case-by-case basis.
"""

from re import Pattern
from typing import Collection
from typing import List
from typing import Optional
from typing import Union

from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    "exclude_genes",
    "exclude_cells",
    "extract_clean_data",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def exclude_genes(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    bursty_max_sampled_cells: Optional[int] = pr.bursty_lonely_max_sampled_cells,
    bursty_downsample_min_samples: int = pr.bursty_lonely_downsample_min_samples,
    bursty_downsample_min_cell_quantile: float = pr.bursty_lonely_downsample_max_cell_quantile,
    bursty_downsample_max_cell_quantile: float = pr.bursty_lonely_downsample_min_cell_quantile,
    bursty_min_gene_total: int = pr.bursty_lonely_min_gene_total,
    bursty_min_gene_normalized_variance: float = pr.bursty_lonely_min_gene_normalized_variance,
    bursty_max_gene_similarity: float = pr.bursty_lonely_max_gene_similarity,
    properly_sampled_min_gene_total: Optional[int] = pr.properly_sampled_min_gene_total,
    excluded_gene_names: Optional[Collection[str]] = None,
    excluded_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    random_seed: int,
) -> None:
    """
    Exclude a subset of the genes from the metacells computation.

    You can also just manually set the ``excluded_gene`` mask, or further manipulate it after calling this function.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Sets the following in the data:

    Variable (gene) annotations:
        ``properly_sampled_gene``
            A mask of the "properly sampled" genes.

        ``excluded_gene``
            A mask of the genes which were excluded (by name or due to not being properly sampled).

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.bursty_lonely.find_bursty_lonely_genes` using
       ``bursty_max_sampled_cells`` (default: {bursty_max_sampled_cells}),
       ``bursty_downsample_min_samples`` (default: {bursty_downsample_min_samples}),
       ``bursty_downsample_min_cell_quantile`` (default: {bursty_downsample_min_cell_quantile}),
       ``bursty_downsample_max_cell_quantile`` (default: {bursty_downsample_max_cell_quantile}),
       ``bursty_min_gene_total`` (default: {bursty_min_gene_total}),
       ``bursty_min_gene_normalized_variance`` (default: {bursty_min_gene_normalized_variance}),
       ``bursty_max_gene_similarity`` (default: {bursty_max_gene_similarity}),
       and ``random_seed``.
       Any "bursty_lonely_genes" will be excluded.

    2. Invoke :py:func:`metacells.tools.properly_sampled.find_properly_sampled_genes` using
       ``properly_sampled_min_gene_total`` (default: {properly_sampled_min_gene_total}). Genes which are not properly
       sampled will be excluded.

    3. Invoke :py:func:`metacells.tools.named.find_named_genes` to also exclude genes based on their name, using the
       ``excluded_gene_names`` (default: {excluded_gene_names}) and ``excluded_gene_patterns`` (default:
       {excluded_gene_patterns}).
    """
    if (
        bursty_max_sampled_cells is None
        and properly_sampled_min_gene_total is None
        and excluded_gene_names is None
        and excluded_gene_patterns is None
    ):
        return

    bursty_lonely_genes_mask: Optional[ut.NumpyVector] = None
    if bursty_max_sampled_cells is not None:
        tl.find_bursty_lonely_genes(
            adata,
            what,
            max_sampled_cells=bursty_max_sampled_cells,
            downsample_min_samples=bursty_downsample_min_samples,
            downsample_min_cell_quantile=bursty_downsample_min_cell_quantile,
            downsample_max_cell_quantile=bursty_downsample_max_cell_quantile,
            min_gene_total=bursty_min_gene_total,
            min_gene_normalized_variance=bursty_min_gene_normalized_variance,
            max_gene_similarity=bursty_max_gene_similarity,
            random_seed=random_seed,
        )
        bursty_lonely_genes_mask = ut.get_v_numpy(adata, "bursty_lonely_gene")

    properly_sampled_genes_mask: Optional[ut.NumpyVector] = None
    if properly_sampled_min_gene_total is not None:
        tl.find_properly_sampled_genes(adata, what, min_gene_total=properly_sampled_min_gene_total)
        properly_sampled_genes_mask = ut.get_v_numpy(adata, "properly_sampled_gene")

    named_genes_mask: Optional[ut.NumpyVector] = None
    if excluded_gene_names is not None or excluded_gene_patterns is not None:
        named_genes_series = tl.find_named_genes(
            adata, names=excluded_gene_names, patterns=excluded_gene_patterns, to=None
        )
        assert named_genes_series is not None
        named_genes_mask = named_genes_series.values

    excluded_genes_mask: Optional[ut.NumpyVector] = None
    if bursty_lonely_genes_mask is not None:
        excluded_genes_mask = bursty_lonely_genes_mask

    if properly_sampled_genes_mask is not None:
        if excluded_genes_mask is None:
            excluded_genes_mask = ~properly_sampled_genes_mask
        else:
            excluded_genes_mask = excluded_genes_mask | ~properly_sampled_genes_mask

    if named_genes_mask is not None:
        if excluded_genes_mask is None:
            excluded_genes_mask = named_genes_mask
        else:
            excluded_genes_mask = excluded_genes_mask | named_genes_mask

    assert excluded_genes_mask is not None
    ut.set_v_data(adata, "excluded_gene", excluded_genes_mask)


@ut.logged()
@ut.timed_call()
def exclude_cells(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    properly_sampled_min_cell_total: Optional[int],
    properly_sampled_max_cell_total: Optional[int],
    properly_sampled_max_excluded_genes_fraction: Optional[float],
    additional_cells_masks: Optional[List[str]] = None,
) -> None:
    """
    Exclude a subset of the cells from the metacells computation.

    You can also just manually set the ``excluded_cell`` mask, or further manipulate it after calling this function.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes. Optionally, may contain an
    ``excluded_gene`` mask of genes to be excluded from the metacells computation. That is, invoke this after
    calling :py:func:`exclude_genes` (if you wish to exclude any genes).

    **Returns**

    Sets the following in the full data:

    Observation (cell) annotations:
        ``properly_sampled_cell``
            A mask of the "properly sampled" cells.

        ``excluded_cell``
            A mask of the genes which were excluded (inverse of ``properly_sampled_cell``).

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.properly_sampled.find_properly_sampled_cells` using
       ``properly_sampled_min_cell_total`` (no default), ``properly_sampled_max_cell_total`` (no default) and
       ``properly_sampled_max_excluded_genes_fraction`` (no default).

    2. Exclude any cells which are not properly sampled (``|~properly_sampled_cell``), with optional additional
       following ``additional_cells_masks`` (using  :py:func:`metacells.tools.mask.combine_masks`).
    """
    tl.find_properly_sampled_cells(
        adata,
        what,
        min_cell_total=properly_sampled_min_cell_total,
        max_cell_total=properly_sampled_max_cell_total,
        max_excluded_genes_fraction=properly_sampled_max_excluded_genes_fraction,
    )

    excluded_cells_masks = ["|~properly_sampled_cell"] + (additional_cells_masks or [])
    tl.combine_masks(adata, excluded_cells_masks, to="excluded_cell")


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def extract_clean_data(
    adata: AnnData,
    *,
    name: Optional[str] = ".clean",
    top_level: bool = True,
) -> Optional[AnnData]:
    """
    Extract a "clean" subset of the ``adata`` to compute metacells for.

    .. todo::

        Allow computing metacells on the full data, obeying the ``excluded_cell`` and ``excluded_gene`` masks.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes. This may contain per-gene
    ``excluded_gene`` and per-cell ``excluded_cell`` annotations.

    **Returns**

    Annotated sliced data containing the "clean" subset of the original data (that is, everything not
    marked as excluded).

    The returned data will have ``full_cell_index`` and ``full_gene_index`` per-observation (cell) and per-variable
    (gene) annotations to allow mapping the results back to the original data.

    **Computation Parameters**

    1. This simply :py:func:`metacells.tools.filter.filter_data` to slice just the genes not in the ``excluded_gene``
       mask and the cells not in the ``excluded_cell`` mask, data using the ``name`` (default: {name}), and tracking the
       original ``full_cell_index`` and ``full_gene_index``.
    """
    results = tl.filter_data(
        adata,
        name=name,
        top_level=top_level,
        track_obs="full_cell_index",
        track_var="full_gene_index",
        obs_masks=["&~excluded_cell?"],
        var_masks=["&~excluded_gene?"],
    )

    if results is None:
        return None
    return results[0]
