'''
Find genes which have high expression.
'''

import logging
from typing import Optional

import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'find_high_fraction_genes',
    'find_high_normalized_variance_genes',
    'find_high_relative_variance_genes',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def find_high_fraction_genes(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    minimal_fraction_of_genes: float = 1e-5,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Find genes which have high fraction of the total ``of`` some data of the cells.

    Genes with too-low expression are typically excluded from computations. In particular,
    genes may have all-zero expression, in which case including them just slows the
    computations (and triggers numeric edge cases).

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Variable (Gene) Annotations
        ``high_fraction_genes``
            A boolean mask indicating whether each gene was found to have a high normalized
            variance.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Use :py:func:`metacells.utilities.preparation.get_fraction_per_var` to get the
       normalized variance of each gene.

    2. Select the genes whose fraction is at least ``minimal_fraction_of_genes`` (default:
       {minimal_fraction_of_genes}).
    '''
    of, level = \
        ut.log_operation(LOG, adata, 'find_high_fraction_genes', of)

    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        fraction_of_genes = \
            ut.get_fraction_per_var(adata).proper
        genes_mask = \
            fraction_of_genes >= minimal_fraction_of_genes

    if inplace:
        ut.set_v_data(adata, 'high_fraction_genes',
                      genes_mask, ut.NEVER_SAFE)
        return None

    ut.log_mask(LOG, level, 'high_fraction_genes', genes_mask)

    return pd.Series(genes_mask, index=adata.var_names)


@ut.timed_call()
@ut.expand_doc()
def find_high_normalized_variance_genes(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    minimal_normalized_variance_of_genes: float = 2.5,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Find genes which have high normalized variance ``of`` some data (by default, the focus).

    The normalized variance measures the variance / mean of each gene.
    See :py:func:`metacells.utilities.preparation.get_normalized_variance_per_var`
    for details.

    Genes with a high normalized variance are "noisy", that is, have significantly different
    expression level in different cells.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Variable (Gene) Annotations
        ``high_normalized_variance_genes``
            A boolean mask indicating whether each gene was found to have a high normalized
            variance.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Use :py:func:`metacells.utilities.preparation.get_normalized_variance_per_var` to get the
       normalized variance of each gene.

    2. Select the genes whose normalized variance is at least
       ``minimal_normalized_variance_of_genes`` (default: {minimal_normalized_variance_of_genes}).
    '''
    of, level = \
        ut.log_operation(LOG, adata, 'find_high_normalized_variance_genes', of)

    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        normalized_variance_of_genes = \
            ut.get_normalized_variance_per_var(adata).proper
        genes_mask = \
            normalized_variance_of_genes >= minimal_normalized_variance_of_genes

    if inplace:
        ut.set_v_data(adata, 'high_normalized_variance_genes',
                      genes_mask, ut.NEVER_SAFE)
        return None

    ut.log_mask(LOG, level, 'high_normalized_variance_genes', genes_mask)

    return pd.Series(genes_mask, index=adata.var_names)


@ut.timed_call()
@ut.expand_doc()
def find_high_relative_variance_genes(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    minimal_relative_variance_of_genes: float = 2.5,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Find genes which have high relative variance ``of`` some data (by default, the focus).

    The relative variance measures the variance / mean of each gene relative to the other genes with
    a similar level of expression. See
    :py:func:`metacells.utilities.preparation.get_relative_variance_per_var` for details.

    Genes with a high relative variance are good candidates for being selected as "feature genes",
    that is, be used to compute the similarity between cells. Using the relative variance
    compensates for the bias for selecting higher-expression genes, whose normalized variance can to
    be larger due to random noise alone.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Variable (Gene) Annotations
        ``high_relative_variance_genes``
            A boolean mask indicating whether each gene was found to have a high relative
            variance.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Use :py:func:`metacells.utilities.preparation.get_relative_variance_per_var` to get the
       relative variance of each gene.

    2. Select the genes whose relative variance is at least
       ``minimal_relative_variance_of_genes`` (default: {minimal_relative_variance_of_genes}).
    '''
    of, level = \
        ut.log_operation(LOG, adata, 'find_high_relative_variance_genes', of)

    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        relative_variance_of_genes = \
            ut.get_relative_variance_per_var(adata).proper
        genes_mask = \
            relative_variance_of_genes >= minimal_relative_variance_of_genes

    if inplace:
        ut.set_v_data(adata, 'high_relative_variance_genes',
                      genes_mask, ut.NEVER_SAFE)
        return None

    ut.log_mask(LOG, level, 'high_relative_variance_genes', genes_mask)

    return pd.Series(genes_mask, index=adata.var_names)
