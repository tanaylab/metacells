'''
Compute cells cross-similarity based on some gene mask(s).
'''

import logging
from typing import Optional

import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

__all__ = [
    'compute_obs_obs_similarity',
    'compute_var_var_similarity',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def compute_obs_obs_similarity(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    repeated: bool = True,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasFrame]:
    '''
    Compute a measure of the similarity between the observations (cells) ``of`` some data (by
    default, the focus).

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Observations-Pair (cells) Annotations
        ``obs_similarity``
            A square matrix where each entry is the similarity between a pair of cells.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas data frame (indexed by the observation names).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Compute the cross-correlation between all the cells.

    2. If ``repeated`` (default: {repeated}), compute the cross-correlation of the correlations.
       That is, for two observations (cells) to be similar, they would need to be similar to the
       rest of the observations (cells) in the same way. This compensates for the extreme sparsity
       of the data.
    '''
    return _compute_elements_similarity(adata, 'obs', of, repeated=repeated,
                                        inplace=inplace, intermediate=intermediate)


@ut.timed_call()
@ut.expand_doc()
def compute_var_var_similarity(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    repeated: bool = True,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasFrame]:
    '''
    Compute a measure of the similarity between the variables (genes) ``of`` some data (by
    default, the focus).

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Variable-Pair (genes) Annotations
        ``var_similarity``
            A square matrix where each entry is the similarity between a pair of cells.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas data frame (indexed by the variable names).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Compute the cross-correlation between all the genes.

    2. If ``repeated`` (default: {repeated}), compute the cross-correlation of the correlations.
       That is, for two variables (genes) to be similar, they would need to be similar to the rest
       of the variables (genes) in the same way. This compensates for the extreme sparsity of the
       data.
    '''
    return _compute_elements_similarity(adata, 'var', of, repeated=repeated,
                                        inplace=inplace, intermediate=intermediate)


def _compute_elements_similarity(
    adata: AnnData,
    elements: str,
    of: Optional[str] = None,
    *,
    repeated: bool = True,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasFrame]:
    assert elements in ('obs', 'var')

    of, level = \
        ut.log_operation(LOG, adata,
                         'compute_%s_%s_similarity' % (elements, elements),
                         of, elements + '_similarity')

    with ut.intermediate_step(adata, intermediate=intermediate):
        if elements == 'obs':
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

    LOG.log(level, '  repeated: %s', repeated)

    if inplace:
        to = elements + '_similarity'
        if elements == 'obs':
            ut.set_oo_data(adata, to,
                           similarity.matrix, ut.SAFE_WHEN_SLICING_OBS)
        else:
            ut.set_vv_data(adata, to,
                           similarity.matrix, ut.SAFE_WHEN_SLICING_VAR)
        return None

    if elements == 'obs':
        names = adata.obs_names
    else:
        names = adata.var_names
    return pd.DataFrame(similarity.matrix, index=names, columns=names)
