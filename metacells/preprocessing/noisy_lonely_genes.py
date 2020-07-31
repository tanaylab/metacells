'''
Detect noisy lonely genes.
'''

from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'find_noisy_lonely_genes',
]


@ut.call()
@ut.expand_doc()
def find_noisy_lonely_genes(  # pylint: disable=too-many-locals
    adata: AnnData,
    *,
    of: Optional[str] = None,
    minimal_gene_fraction: float = 1e-5,
    minimal_gene_relative_variance: float = 2.5,
    maximal_gene_correlation: float = 0.15,
    inplace: bool = True,
) -> Optional[pd.Series]:
    '''
    Detect noisy lonely genes.

    Noisy genes have high expression and high variance. Lonely genes have low correlation with all
    other genes.

    Genes that are both noisy and lonely tend to throw off clustering algorithms. Since they are
    noisy, they are given significant weight in the algorithm. Since they are lonely, they don't
    contribute to meaningful clustering of the cells.

    In the best case, noisy lonely genes are a distracting noise for the clustering algorithm. In
    the worst case, the algorithm may choose to cluster cells based on such genes, which defeats the
    purpose of clustering cells with similar overall transcriptome state.

    It is therefore useful to explicitly identify, in a pre-processing step, the few noisy lonely
    genes, and exclude them from the rest of the analysis.

    .. note::

        Detecting such genes requires computing gene-gene correlation. This is acceptable for small
        data sets, but for large data sets it suffices to first randomly select a subset of the
        cells of a reasonable size (we recommend 20,000 cells), and use just these cells to detect
        the noisy lonely genes, before clustering the complete large data set.

        It is also strongly recommended these cells be downsampled so their total number of UMIs
        is the same, to get an unbiased sampling of each gene in each cell.

    **Input**

    An annotated ``adata``, where the observations are cells and the variables are genes, containing
    the UMIs count in the ``of`` (default: the ``focus``) per-variable-per-observation data.

    Various intermediate data (sums, variances, etc.) is used if already available. Otherwise it is
    computed and cached for future reuse.

    **Returns**

    Variable (Gene) Annotations
        ``noisy_lonely_mask``
            A boolean indicating whether the gene was found to be a noisy lonely gene.

    If ``inplace`` (default: {inplace}), these are written to ``adata`` and the function returns
    ``None``. Otherwise this is returned as a Pandas series (indexed by the variable names).

    **Computation Parameters**

    Given an annotated ``adata``, where the variables are cell RNA profiles and the observations are
    gene UMI counts, do the following:

    1. Pick as candidates all genes whose fraction of the UMIs is at least
       ``minimal_gene_fraction`` (default: {minimal_gene_fraction}).

    2. Restrict the genes to include only genes whose relative variance is at least
       ``minimal_gene_relative_variance`` (default: {minimal_gene_relative_variance}).

    3. Finally restrict the genes to include only genes which have a correlation of at most
       ``maximal_gene_correlation`` (default: {maximal_gene_correlation}) with at least one other
       gene.

    .. todo::

        Should we correlate the normalized (fraction) and/or log of the data for
        :py:func:`find_noisy_lonely_genes`?
    '''
    with ut.focus_on(ut.get_vo_data, adata, of):
        fraction_of_genes = ut.get_fraction_per_var(adata)[0]
        relative_variance_of_genes = ut.get_relative_variance_per_var(adata)[0]

        fraction_mask = fraction_of_genes >= minimal_gene_fraction
        variance_mask = relative_variance_of_genes >= minimal_gene_relative_variance
        noisy_mask = fraction_mask & variance_mask

        candidate_adata = adata[:, noisy_mask]
        correlation_of_genes = \
            np.copy(np.array(ut.get_var_var_correlation(candidate_adata)[0]))
        np.fill_diagonal(correlation_of_genes, 0)
        max_correlation_of_genes = \
            ut.to_array(correlation_of_genes.max(axis=0))

        lonely_mask = max_correlation_of_genes <= maximal_gene_correlation

        noisy_lonely_mask = noisy_mask
        noisy_lonely_mask[noisy_mask] = lonely_mask

        if inplace:
            ut.set_data(adata, 'v:noisy_lonely_mask', noisy_lonely_mask,
                        ut.SAFE_WHEN_SLICING_VAR)
            return None

        return pd.Series(noisy_lonely_mask, index=adata.var_names)
