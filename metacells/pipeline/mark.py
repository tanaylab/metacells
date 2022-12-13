"""
Mark (Genes)
------------

Even when working with "clean" data, computing metacells requires marking genes for special treatment, specifically
"lateral" genes which should not be selected as feature genes and "noisy" genes which should be neither selected as
feature genes not used to detect deviant (outlier) cells in the metacells.

Proper choice of lateral and noisy genes is mandatory for high-quality metacells generation. However, this choice can't
be fully automated, and therefore relies on the analyst to manually provide the list of genes.
"""

from re import Pattern
from typing import Collection
from typing import Optional
from typing import Union

from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    "mark_lateral_genes",
    "mark_noisy_genes",
    "mark_feature_genes",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def mark_lateral_genes(
    adata: AnnData,
    *,
    lateral_gene_names: Optional[Collection[str]] = None,
    lateral_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
) -> None:
    """
    Mark a subset of the genes as "lateral", that is, prevent them from being used as feature genes when computing
    metacell.

    Lateral genes are still used to detect outliers, that is, the cells in resulting metacells should still have
    similar expression level for such genes.

    You can also just manually set the ``lateral_gene`` mask, or further manipulate it after calling this function.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Sets the following in the data:

    Variable (gene) annotations:
        ``lateral_gene``
            A mask of the "lateral" genes.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.named.find_named_genes` to also mark lateral genes based on their name, using
       the ``lateral_gene_names`` (default: {lateral_gene_names}) and ``lateral_gene_patterns`` (default:
       {lateral_gene_patterns}).
    """
    tl.find_named_genes(adata, names=lateral_gene_names, patterns=lateral_gene_patterns, to="lateral_gene")


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def mark_noisy_genes(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    bursty_lonely_max_sampled_cells: int = pr.bursty_lonely_max_sampled_cells,
    bursty_lonely_downsample_min_samples: int = pr.bursty_lonely_downsample_min_samples,
    bursty_lonely_downsample_min_cell_quantile: float = pr.bursty_lonely_downsample_min_cell_quantile,
    bursty_lonely_downsample_max_cell_quantile: float = pr.bursty_lonely_downsample_max_cell_quantile,
    bursty_lonely_min_gene_total: int = pr.bursty_lonely_min_gene_total,
    bursty_lonely_min_gene_normalized_variance: float = pr.bursty_lonely_min_gene_normalized_variance,
    bursty_lonely_max_gene_similarity: float = pr.bursty_lonely_max_gene_similarity,
    noisy_gene_names: Optional[Collection[str]] = None,
    noisy_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    random_seed: int = pr.random_seed,
) -> None:
    """
    Mark a subset of the genes as "noisy", that is, prevent them from both being used as feature genes and for detecting
    deviant (outlier) cells when computing metacell.

    You can also just manually set the ``noisy_gene`` mask, or further manipulate it after calling this function.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Sets the following in the data:

    Variable (gene) annotations:
        ``bursty_lonely_genes``
            A boolean mask indicating whether each gene was found to be a "bursty lonely" gene.

        ``noisy_gene``
            A mask of the "noisy" genes.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.bursty_lonely.find_bursty_lonely_genes` using
       ``bursty_lonely_max_sampled_cells`` (default: {bursty_lonely_max_sampled_cells}),
       ``bursty_lonely_downsample_min_samples`` (default: {bursty_lonely_downsample_min_samples}),
       ``bursty_lonely_downsample_min_cell_quantile`` (default:
       {bursty_lonely_downsample_min_cell_quantile}), ``bursty_lonely_downsample_max_cell_quantile``
       (default: {bursty_lonely_downsample_max_cell_quantile}), ``bursty_lonely_min_gene_total``
       (default: {bursty_lonely_min_gene_total}), ``bursty_lonely_min_gene_normalized_variance``
       (default: {bursty_lonely_min_gene_normalized_variance}), and
       ``bursty_lonely_max_gene_similarity`` (default: {bursty_lonely_max_gene_similarity}). Mark all
       bursty lonely genes as "noisy".

    2. Invoke :py:func:`metacells.tools.named.find_named_genes` to also mark noisy genes based on their name, using
       the ``noisy_gene_names`` (default: {noisy_gene_names}) and ``noisy_gene_patterns`` (default:
       {noisy_gene_patterns}).
    """
    tl.find_bursty_lonely_genes(
        adata,
        what,
        max_sampled_cells=bursty_lonely_max_sampled_cells,
        downsample_min_samples=bursty_lonely_downsample_min_samples,
        downsample_min_cell_quantile=bursty_lonely_downsample_min_cell_quantile,
        downsample_max_cell_quantile=bursty_lonely_downsample_max_cell_quantile,
        min_gene_total=bursty_lonely_min_gene_total,
        min_gene_normalized_variance=bursty_lonely_min_gene_normalized_variance,
        max_gene_similarity=bursty_lonely_max_gene_similarity,
        random_seed=random_seed,
    )

    noisy_genes_mask = ut.get_v_numpy(adata, "bursty_lonely_gene")

    if noisy_gene_names is not None or noisy_gene_patterns is not None:
        noisy_genes_series = tl.find_named_genes(adata, to=None, names=noisy_gene_names, patterns=noisy_gene_patterns)
        assert noisy_genes_series is not None
        noisy_genes_mask |= noisy_genes_series.values

    ut.set_v_data(adata, "noisy_gene", noisy_genes_mask)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def mark_feature_genes(
    adata: AnnData,
    *,
    feature_gene_names: Optional[Collection[str]] = None,
    feature_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
) -> None:
    """
    Mark a subset of the genes as "feature", that is, force them to be used as feature genes when computing metacell.

    If this is done, then it overrides the default feature gene selection mechanism, forcing the algorithm to
    use a fixed set of feature genes. In general, this will result in lower-quality metacells, especially when
    using the divide-and-conquer algorithm, so **do not use this unless you really know what you are doing**.

    You can also just manually set the ``feature_gene`` mask, or further manipulate it after calling this function.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Sets the following in the data:

    Variable (gene) annotations:
        ``feature_gene``
            A mask of the "feature" genes.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.named.find_named_genes` to also mark feature genes based on their name, using
       the ``feature_gene_names`` (default: {feature_gene_names}) and ``feature_gene_patterns`` (default:
       {feature_gene_patterns}).
    """
    tl.find_named_genes(adata, names=feature_gene_names, patterns=feature_gene_patterns, to="feature_gene")
