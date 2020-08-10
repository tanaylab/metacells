'''
Compute cells cross-similarity based on some gene mask(s).
'''

from typing import Optional

import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'compute_similarity'
]


@ut.timed_call()
@ut.expand_doc()
def compute_similarity(
    adata: AnnData,
    elements: str,
    of: Optional[str] = None,
    *,
    log: bool = False,
    log_base: Optional[float] = None,
    log_normalization: float = -1,
    repeated: bool = True,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[pd.DataFrame]:
    '''
    Compute a measure of the similarity for each pair of ``elements`` (either ``cells`` or
    ``genes``).

    If ``of`` is specified, this specific data is used. Otherwise, the focus data is used.

    **Input**

    A :py:func:`metacells.utilities.preparation.prepare`-ed annotated ``adata``, where the
    observations are cells and the variables are genes, containing the UMIs count in the ``of``
    (default: the ``focus``) per-variable-per-observation data.

    **Returns**

    Observations-Pair (cells) or Variable-Pair (genes) Annotations
        ``<per>_similarity``
            A square matrix where each entry is the similarity between a pair of cells or genes.

    If ``inplace`` (default: {inplace}), this is written to ``adata`` and the function returns
    ``None``. Otherwise this is returned as a Pandas data frame (indexed by the observation or
    variable names).

    If not ``intermediate`` (default: {intermediate}), this discards all the intermediate data used
    (e.g. sums). Otherwise, such data is kept for future reuse.

    **Computation Parameters**

    Given an annotated ``adata``, where the variables are cell RNA profiles and the observations are
    gene UMI counts, do the following:

    1. Compute the cross-correlation between all the elements. If ``log`` (default: {log}),
       correlate the logarithm of the values using ``log_base`` (default: {log_base}) and
       ``log_normalization`` (default: {log_normalization}). See
       :py:func:`metacells.utilities.computation.log_data` for details.

    2. If ``repeated`` (default: {repeated}), compute the cross-correlation of the correlations.
       That is, for two elements to be similar, they would need to be similar to the rest of the
       elements in the same way. This compensates for the extreme sparsity of the data.
    '''
    assert elements in ('cells', 'genes')

    with ut.intermediate_step(adata, intermediate=intermediate):
        if log:
            of = ut.get_log(adata, of, base=log_base,
                            normalization=log_normalization).name

        if elements == 'cells':
            similarity = ut.get_obs_obs_correlation(adata, of,
                                                    inplace=inplace or repeated)
            if repeated:
                similarity = ut.get_obs_obs_correlation(adata, similarity.name,
                                                        inplace=inplace)

        else:
            similarity = ut.get_var_var_correlation(adata, of,
                                                    inplace=inplace or repeated)
            if repeated:
                similarity = ut.get_var_var_correlation(adata, similarity.name,
                                                        inplace=inplace)

    if inplace:
        adata.obsp['cells_similarity'] = similarity.data
        ut.safe_slicing_data('cells_similarity', ut.SAFE_WHEN_SLICING_OBS)
        return None

    if elements == 'cells':
        names = adata.obs_names
    else:
        names = adata.var_names
    return pd.DataFrame(similarity.data, index=names, columns=names)
