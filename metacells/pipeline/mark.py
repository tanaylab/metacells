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
from typing import Dict
from typing import Optional
from typing import Union

from anndata import AnnData  # type: ignore

import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    "mark_lateral_genes",
    "mark_noisy_genes",
    "mark_select_genes",
    "mark_ignored_genes",
    "mark_essential_genes",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def mark_lateral_genes(
    adata: AnnData,
    *,
    lateral_gene_names: Optional[Collection[str]] = None,
    lateral_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    op: str = "set",
) -> None:
    """
    Mark a subset of the genes as "lateral", that is, prevent them from being selected for computing metacells.

    Lateral genes are still used to detect outliers, that is, the cells in resulting metacells should still have
    similar expression level for such genes.

    Depending on ``op``, this will either ``set`` (override/create) the mask, ``add`` (to an existing mask), or
    ``remove`` (from an existing mask).

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Sets the following in the data:

    Variable (gene) annotations:
        ``lateral_gene``
            A mask of the "lateral" genes.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.named.find_named_genes` to also mark lateral genes based on their name, using
       the ``lateral_gene_names`` (default: {lateral_gene_names}), ``lateral_gene_patterns`` (default:
       {lateral_gene_patterns}) and ``op`` (default: {op}).
    """
    tl.find_named_genes(adata, names=lateral_gene_names, patterns=lateral_gene_patterns, to="lateral_gene", op=op)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def mark_noisy_genes(
    adata: AnnData,
    *,
    noisy_gene_names: Optional[Collection[str]] = None,
    noisy_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    op: str = "set",
) -> None:
    """
    Mark a subset of the genes as "noisy", that is, prevent them from being used for detecting deviant (outlier) cells.

    Depending on ``op``, this will either ``set`` (override/create) the mask, ``add`` (to an existing mask), or
    ``remove`` (from an existing mask).

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Sets the following in the data:

    Variable (gene) annotations:
        ``noisy_gene``
            A mask of the "noisy" genes.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.named.find_named_genes` to also mark noisy genes based on their name, using
       the ``noisy_gene_names`` (default: {noisy_gene_names}), ``noisy_gene_patterns`` (default:
       {noisy_gene_patterns}) and ``op`` (default: {op}).
    """
    tl.find_named_genes(adata, names=noisy_gene_names, patterns=noisy_gene_patterns, to="noisy_gene", op=op)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def mark_select_genes(
    adata: AnnData,
    *,
    select_gene_names: Optional[Collection[str]] = None,
    select_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    op: str = "set",
) -> None:
    """
    Mark a subset of the genes as "select", that is, force them to be used when computing metacell.

    If this is done, then it overrides the default gene selection mechanism, forcing the algorithm to use a fixed set of
    genes. In general, this will result in lower-quality metacells, especially when using the divide-and-conquer
    algorithm, so **do not use this unless you really know what you are doing**.

    Depending on ``op``, this will either ``set`` (override/create) the mask, ``add`` (to an existing mask), or
    ``remove`` (from an existing mask).

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Sets the following in the data:

    Variable (gene) annotations:
        ``select_gene``
            A mask of the "select" genes.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.named.find_named_genes` to select genes based on their name, using
       the ``select_gene_names`` (default: {select_gene_names}), ``select_gene_patterns`` (default:
       {select_gene_patterns}) and ``op`` (default: {op}).
    """
    tl.find_named_genes(adata, names=select_gene_names, patterns=select_gene_patterns, to="select_gene", op=op)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def mark_ignored_genes(
    adata: AnnData,
    *,
    ignored_gene_names: Optional[Collection[str]] = None,
    ignored_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    ignored_gene_names_of_types: Optional[Dict[str, Collection[str]]] = None,
    ignored_gene_patterns_of_types: Optional[Dict[str, Collection[str]]] = None,
    op: str = "set",
) -> None:
    """
    Mark a subset of the genes as "ignored", that is, do not attempt to match them when projecting this (query)
    data onto an atlas.

    Depending on ``op``, this will either ``set`` (override/create) the mask(s), ``add`` (to an existing mask(s)), or
    ``remove`` (from an existing mask(s)).

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Sets the following in the data:

    Variable (gene) annotations:
        ``ignored_gene``
            A mask of the "ignored" genes for all the query metacells regardless of their type.

        ``ignored_gene_of_<type>``
            A mask of the "ignored" genes for query metacells that are assigned a specific type.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.named.find_named_genes` to ignore genes based on their name, using
       the ``ignored_gene_names`` (default: {ignored_gene_names}), ``ignored_gene_patterns`` (default:
       {ignored_gene_patterns}) and ``op`` (default: {op}).

    2. Similarly for each type specified in ``ignored_gene_names_of_types`` and/or ``ignored_gene_patterns_of_types``.
    """
    tl.find_named_genes(adata, names=ignored_gene_names, patterns=ignored_gene_patterns, to="ignored_gene", op=op)

    ignored_gene_names_of_types = ignored_gene_names_of_types or {}
    ignored_gene_patterns_of_types = ignored_gene_patterns_of_types or {}
    types = set(ignored_gene_names_of_types.keys()) | set(ignored_gene_patterns_of_types.keys())

    for type_name in types:
        tl.find_named_genes(
            adata,
            names=ignored_gene_names_of_types.get(type_name),
            patterns=ignored_gene_patterns_of_types.get(type_name),
            to=f"ignored_gene_of_{type_name}",
            op=op,
        )


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def mark_essential_genes(
    adata: AnnData,
    *,
    essential_gene_names_of_types: Optional[Dict[str, Collection[str]]] = None,
    essential_gene_patterns_of_types: Optional[Dict[str, Collection[str]]] = None,
    op: str = "set",
) -> None:
    """
    Mark a subset of the genes as "essential", that is, require that (most of them) will be projected "well" to accept a
    query metacell as having the same type as the atlas projection.

    Depending on ``op``, this will either ``set`` (override/create) the mask(s), ``add`` (to an existing mask(s)), or
    ``remove`` (from an existing mask(s)).

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Sets the following in the data:

    Variable (gene) annotations:
        ``essential_gene_of_<type>``
            A mask of the "essential" genes for query metacells to be assigned a specific type.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.named.find_named_genes` to ignore genes based on their name, using the
       ``essential_gene_names_of_types`` (default: {essential_gene_names_of_types}),
       ``essential_gene_patterns_of_types`` (default: {essential_gene_patterns_of_types}) and ``op`` (default: {op}).
    """
    essential_gene_names_of_types = essential_gene_names_of_types or {}
    essential_gene_patterns_of_types = essential_gene_patterns_of_types or {}
    types = set(essential_gene_names_of_types.keys()) | set(essential_gene_patterns_of_types.keys())

    for type_name in types:
        tl.find_named_genes(
            adata,
            names=essential_gene_names_of_types.get(type_name),
            patterns=essential_gene_patterns_of_types.get(type_name),
            to=f"essential_gene_of_{type_name}",
            op=op,
        )
