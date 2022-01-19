"""
High
----
"""

from typing import Optional
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    "find_top_feature_genes",
    "find_high_total_genes",
    "find_high_topN_genes",
    "find_high_fraction_genes",
    "find_high_normalized_variance_genes",
    "find_high_relative_variance_genes",
    "find_metacells_significant_genes",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def find_top_feature_genes(
    adata: AnnData,
    *,
    max_genes: int = pr.max_top_feature_genes,
    inplace: bool = True,
) -> Optional[ut.PandasSeries]:
    """
    Find genes which have high ``feature_gene`` value.

    This is applied after computing metacells to pick the "strongest" feature genes. If using the
    direct algorithm (:py:func:`metacells.pipeline.direct.compute_direct_metacells`) then all
    feature genes are equally "strong"; however, if using the divide-and-conquer algorithm
    (:py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`,
    :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`) then this
    will pick the genes which were most commonly used as features across all the piles.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``feature_gene`` is a per-variable (gene) annotation counting how many times each gene was used
    as a feature.

    **Returns**

    Variable (Gene) Annotations
        ``top_feature_gene``
            A boolean mask indicating whether each gene was found to be a top feature gene.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    **Computation Parameters**

    1. Look for the lowest positive ``feature_gene`` threshold such that at most ``max_genes`` are
       picked as top feature genes. Note we may still pick more than ``max_genes``, for example when
       using the direct algorithm, we always return all feature genes as there's no way to
       distinguish between them using the ``feature_gene`` data.
    """
    feature_of_gene = ut.get_v_numpy(adata, "feature_gene", formatter=ut.mask_description)
    max_threshold = np.max(feature_of_gene)
    assert max_threshold > 0
    threshold = 0
    selected_count = max_genes + 1
    while selected_count > max_genes and threshold < max_threshold:
        threshold = threshold + 1
        genes_mask = feature_of_gene >= threshold
        selected_count = np.sum(genes_mask)
        ut.log_calc(f"threshold: {threshold} selected: {selected_count}")

    if inplace:
        ut.set_v_data(adata, "top_feature_gene", genes_mask)
        return None

    ut.log_return("top_feature_gene", genes_mask)
    return ut.to_pandas_series(genes_mask, index=adata.var_names)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def find_high_total_genes(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    min_gene_total: int,
    inplace: bool = True,
) -> Optional[ut.PandasSeries]:
    """
    Find genes which have high total number of ``what`` (default: {what}) data.

    This should typically only be applied to downsampled data to ensure that variance in sampling
    depth does not affect the result.

    Genes with too-low expression are typically excluded from computations. In particular,
    genes may have all-zero expression, in which case including them just slows the
    computations (and triggers numeric edge cases).

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    **Returns**

    Variable (Gene) Annotations
        ``high_total_gene``
            A boolean mask indicating whether each gene was found to have a high normalized
            variance.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    **Computation Parameters**

    1. Use :py:func:`metacells.utilities.computation.sum_per` to get the total UMIs of each gene.

    2. Select the genes whose fraction is at least ``min_gene_total``.
    """
    total_of_genes = ut.get_v_numpy(adata, what, sum=True)
    genes_mask = total_of_genes >= min_gene_total

    if inplace:
        ut.set_v_data(adata, "high_total_gene", genes_mask)
        return None

    ut.log_return("high_total_genes", genes_mask)
    return ut.to_pandas_series(genes_mask, index=adata.var_names)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def find_high_topN_genes(  # pylint: disable=invalid-name
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    topN: int,  # pylint: disable=invalid-name
    min_gene_topN: int,  # pylint: disable=invalid-name
    inplace: bool = True,
) -> Optional[ut.PandasSeries]:
    """
    Find genes which have high total top-Nth value of ``what`` (default: {what}) data.

    This should typically only be applied to downsampled data to ensure that variance in sampling
    depth does not affect the result.

    Genes with too-low expression are typically excluded from computations. In particular,
    genes may have all-zero expression, in which case including them just slows the
    computations (and triggers numeric edge cases).

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    **Returns**

    Variable (Gene) Annotations
        ``high_top<topN>_gene``
            A boolean mask indicating whether each gene was found to have a high top-Nth value.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    **Computation Parameters**

    1. Use :py:func:`metacells.utilities.computation.top_per` to get the top-Nth UMIs of each gene.

    2. Select the genes whose fraction is at least ``min_gene_topN``.
    """
    data_of_genes = ut.get_vo_proper(adata, what, layout="column_major")
    rank = max(adata.n_obs - topN - 1, 1)
    topN_of_genes = ut.rank_per(data_of_genes, per="column", rank=rank)  # pylint: disable=invalid-name
    genes_mask = topN_of_genes >= min_gene_topN

    if inplace:
        ut.set_v_data(adata, f"high_top{topN}_gene", genes_mask)
        return None

    ut.log_return(f"high_top{topN}_genes", genes_mask)
    return ut.to_pandas_series(genes_mask, index=adata.var_names)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def find_high_fraction_genes(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    min_gene_fraction: float = pr.significant_gene_fraction,
    inplace: bool = True,
) -> Optional[ut.PandasSeries]:
    """
    Find genes which have high fraction of the total ``what`` (default: {what}) data of the cells.

    Genes with too-low expression are typically excluded from computations. In particular,
    genes may have all-zero expression, in which case including them just slows the
    computations (and triggers numeric edge cases).

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    **Returns**

    Variable (Gene) Annotations
        ``high_fraction_gene``
            A boolean mask indicating whether each gene was found to have a high normalized
            variance.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    **Computation Parameters**

    1. Use :py:func:`metacells.utilities.computation.fraction_per` to get the fraction of each gene.

    2. Select the genes whose fraction is at least ``min_gene_fraction`` (default:
       {min_gene_fraction}).
    """
    data = ut.get_vo_proper(adata, what, layout="column_major")
    fraction_of_genes = ut.fraction_per(data, per="column")

    genes_mask = fraction_of_genes >= min_gene_fraction

    if inplace:
        ut.set_v_data(adata, "high_fraction_gene", genes_mask)
        return None

    ut.log_return("high_fraction_genes", genes_mask)
    return ut.to_pandas_series(genes_mask, index=adata.var_names)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def find_high_normalized_variance_genes(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    min_gene_normalized_variance: float = pr.significant_gene_normalized_variance,
    inplace: bool = True,
) -> Optional[ut.PandasSeries]:
    """
    Find genes which have high normalized variance of ``what`` (default: {what}) data.

    The normalized variance measures the variance / mean of each gene. See
    :py:func:`metacells.utilities.computation.normalized_variance_per` for details.

    Genes with a high normalized variance are "noisy", that is, have significantly different
    expression level in different cells.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    **Returns**

    Variable (Gene) Annotations
        ``high_normalized_variance_gene``
            A boolean mask indicating whether each gene was found to have a high normalized
            variance.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    **Computation Parameters**

    1. Use :py:func:`metacells.utilities.computation.normalized_variance_per` to get the normalized
       variance of each gene.

    2. Select the genes whose normalized variance is at least
       ``min_gene_normalized_variance`` (default: {min_gene_normalized_variance}).
    """
    data = ut.get_vo_proper(adata, what, layout="column_major")
    normalized_variance_of_genes = ut.normalized_variance_per(data, per="column")

    genes_mask = normalized_variance_of_genes >= min_gene_normalized_variance

    if inplace:
        ut.set_v_data(adata, "high_normalized_variance_gene", genes_mask)
        return None

    ut.log_return("high_normalized_variance_genes", genes_mask)
    return ut.to_pandas_series(genes_mask, index=adata.var_names)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def find_high_relative_variance_genes(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    min_gene_relative_variance: float = pr.significant_gene_relative_variance,
    window_size: int = pr.relative_variance_window_size,
    inplace: bool = True,
) -> Optional[ut.PandasSeries]:
    """
    Find genes which have high relative variance of ``what`` (default: {what}) data.

    The relative variance measures the variance / mean of each gene relative to the other genes with
    a similar level of expression. See
    :py:func:`metacells.utilities.computation.relative_variance_per` for details.

    Genes with a high relative variance are good candidates for being selected as "feature genes",
    that is, be used to compute the similarity between cells. Using the relative variance
    compensates for the bias for selecting higher-expression genes, whose normalized variance can to
    be larger due to random noise alone.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    **Returns**

    Variable (Gene) Annotations
        ``high_relative_variance_gene``
            A boolean mask indicating whether each gene was found to have a high relative
            variance.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    **Computation Parameters**

    1. Use :py:func:`metacells.utilities.computation.relative_variance_per` to get the relative
       variance of each gene.

    2. Select the genes whose relative variance is at least
       ``min_gene_relative_variance`` (default: {min_gene_relative_variance}).
    """
    data = ut.get_vo_proper(adata, what, layout="column_major")
    relative_variance_of_genes = ut.relative_variance_per(data, per="column", window_size=window_size)

    genes_mask = relative_variance_of_genes >= min_gene_relative_variance

    if inplace:
        ut.set_v_data(adata, "high_relative_variance_gene", genes_mask)
        return None

    ut.log_return("high_relative_variance_genes", genes_mask)
    return ut.to_pandas_series(genes_mask, index=adata.var_names)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def find_metacells_significant_genes(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    min_gene_range_fold: float = pr.min_significant_metacells_gene_range_fold_factor,
    normalization: float = pr.metacells_gene_range_normalization,
    min_gene_fraction: float = pr.min_significant_metacells_gene_fraction,
    inplace: bool = True,
) -> Optional[ut.PandasSeries]:
    """
    Find genes which have a significant signal in metacells data. This computation is too unreliable to be used on
    cells.

    Find genes which have a high maximal expression in at least one metacell, and a wide range of expression across the
    metacells. Such genes are good candidates for being used as marker genes and/or to compute distances between
    metacells.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    **Returns**

    Variable (Gene) Annotations
        ``significant_gene``
            A boolean mask indicating whether each gene was found to be significant.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    **Computation Parameters**

    1. Compute the minimal and maximal expression level of each gene.

    2. Select the genes whose fold factor (log2 of maximal over minimal value, using the ``normalization``
       (default: {normalization}) is at least ``min_gene_range_fold`` (default: {min_gene_range_fold}).

    3. Select the genes whose maximal expression is at least ``min_gene_fraction`` (default: {min_gene_fraction}).
    """
    assert normalization >= 0

    data = ut.get_vo_proper(adata, what, layout="row_major")
    fractions_of_genes = ut.to_layout(ut.fraction_by(data, by="row"), layout="column_major")

    min_fraction_of_genes = ut.min_per(fractions_of_genes, per="column")
    max_fraction_of_genes = ut.max_per(fractions_of_genes, per="column")

    high_max_fraction_genes_mask = max_fraction_of_genes >= min_gene_fraction
    ut.log_calc("high max fraction genes", high_max_fraction_genes_mask)

    min_fraction_of_genes += normalization
    max_fraction_of_genes += normalization

    max_fraction_of_genes /= min_fraction_of_genes
    range_fold_of_genes = np.log2(max_fraction_of_genes, out=max_fraction_of_genes)

    high_range_genes_mask = range_fold_of_genes >= min_gene_range_fold
    ut.log_calc("high range genes", high_range_genes_mask)

    significant_genes_mask = high_max_fraction_genes_mask & high_range_genes_mask

    if inplace:
        ut.set_v_data(adata, "significant_gene", significant_genes_mask)
        return None

    ut.log_return("significant_genes", significant_genes_mask)
    return ut.to_pandas_series(significant_genes_mask, index=adata.var_names)
