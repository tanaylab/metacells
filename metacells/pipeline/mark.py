"""
Mark (Genes)
------------

Even when working with "clean" data, computing metacells requires marking genes for special treatment, specifically
"lateral" genes which should not be selected for computing metacells and "noisy" genes which should be neither selected
as nor be used to detect deviant (outlier) cells in the metacells.

Proper choice of lateral and noisy genes is mandatory for high-quality metacells generation. However, this choice can't
be fully automated, and therefore relies on the analyst to manually provide the list of genes.
"""

from re import Pattern
from typing import Collection
from typing import Optional
from typing import Union

from anndata import AnnData  # type: ignore

import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    "mark_lateral_genes",
    "mark_noisy_genes",
    "mark_select_genes",
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
    Mark a subset of the genes as "lateral", that is, prevent them from being selected for computing metacells.

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
    *,
    noisy_gene_names: Optional[Collection[str]] = None,
    noisy_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
) -> None:
    """
    Mark a subset of the genes as "noisy", that is, prevent them from both being selected for computing metacells and
    for detecting deviant (outlier) cells.

    You can also just manually set the ``noisy_gene`` mask, or further manipulate it after calling this function.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Sets the following in the data:

    Variable (gene) annotations:
        ``noisy_gene``
            A mask of the "noisy" genes.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.named.find_named_genes` to also mark noisy genes based on their name, using
       the ``noisy_gene_names`` (default: {noisy_gene_names}) and ``noisy_gene_patterns`` (default:
       {noisy_gene_patterns}).
    """
    tl.find_named_genes(adata, names=noisy_gene_names, patterns=noisy_gene_patterns, to="noisy_gene")


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def mark_select_genes(
    adata: AnnData,
    *,
    select_gene_names: Optional[Collection[str]] = None,
    select_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
) -> None:
    """
    Mark a subset of the genes as "select", that is, force them to be used when computing metacell.

    If this is done, then it overrides the default gene selection mechanism, forcing the algorithm to use a fixed set of
    genes. In general, this will result in lower-quality metacells, especially when using the divide-and-conquer
    algorithm, so **do not use this unless you really know what you are doing**.

    You can also just manually set the ``select_gene`` mask, or further manipulate it after calling this function.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Sets the following in the data:

    Variable (gene) annotations:
        ``select_gene``
            A mask of the "select" genes.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.named.find_named_genes` to also select genes based on their name, using
       the ``select_gene_names`` (default: {select_gene_names}) and ``select_gene_patterns`` (default:
       {select_gene_patterns}).
    """
    tl.find_named_genes(adata, names=select_gene_names, patterns=select_gene_patterns, to="select_gene")
