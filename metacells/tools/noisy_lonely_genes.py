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


@ut.timed_call()
@ut.expand_doc()
def find_noisy_lonely_genes(  # pylint: disable=too-many-locals
    adata: AnnData,
    of: Optional[str] = None,
    *,
    minimal_fraction_of_genes: float = 1e-5,
    minimal_relative_variance_of_genes: float = 2.5,
    maximal_correlation_of_genes: float = 0.15,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasSeries]:
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

    A :py:func:`metacells.utilities.preparation.prepare`-ed annotated ``adata``, where the
    observations are cells and the variables are genes, containing the UMIs count in the ``of``
    (default: the focus) per-variable-per-observation data.

    **Returns**

    Variable (Gene) Annotations
        ``noisy_lonely_genes``
            A boolean mask indicating whether each gene was found to be a noisy lonely gene.

    If ``inplace`` (default: {inplace}), these are written to ``adata`` and the function returns
    ``None``. Otherwise this is returned as a Pandas series (indexed by the variable names).

    If not ``intermediate`` (default: {intermediate}), this discards all the intermediate data used
    (e.g. sums). Otherwise, such data is kept for future reuse.

    **Computation Parameters**

    Given an annotated ``adata``, where the variables are cell RNA profiles and the observations are
    gene UMI counts, do the following:

    1. Pick as candidates all genes whose fraction of the UMIs is at least
       ``minimal_fraction_of_genes`` (default: {minimal_fraction_of_genes}).

    2. Restrict the genes to include only genes whose relative variance is at least
       ``minimal_relative_variance_of_genes`` (default: {minimal_relative_variance_of_genes}).

    3. Finally restrict the genes to include only genes which have a correlation of at most
       ``maximal_correlation_of_genes`` (default: {maximal_correlation_of_genes}) with at least one
       other gene.

    .. todo::

        Should we correlate the normalized (fraction) and/or log of the data for
        :py:func:`find_noisy_lonely_genes`?
    '''
    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        fraction_of_genes = ut.get_fraction_per_var(adata).proper
        relative_variance_of_genes = \
            ut.get_relative_variance_per_var(adata).proper

        fraction_mask = fraction_of_genes >= minimal_fraction_of_genes
        variance_mask = relative_variance_of_genes >= minimal_relative_variance_of_genes
        noisy_mask = fraction_mask & variance_mask

        noisy_adata = adata[:, noisy_mask]

        correlation_of_noisy_genes = \
            ut.get_var_var_correlation(noisy_adata, inplace=False).proper
        np.fill_diagonal(correlation_of_noisy_genes, None)

        max_correlation_of_noisy_genes = \
            np.nanmax(correlation_of_noisy_genes, axis=0)

        lonely_mask = max_correlation_of_noisy_genes <= maximal_correlation_of_genes

        noisy_lonely_mask = noisy_mask
        noisy_lonely_mask[noisy_mask] = lonely_mask

    if inplace:
        adata.var['noisy_lonely_genes'] = noisy_lonely_mask
        ut.safe_slicing_data('noisy_lonely_genes', ut.SAFE_WHEN_SLICING_VAR)
        return None

    return pd.Series(noisy_lonely_mask, index=adata.var_names)
