'''
Cross-Similarity
----------------
'''

import logging
from typing import Optional

import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.parameters as pr
import metacells.preprocessing as pp
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
    method: str = pr.similarity_method,
    repeated: bool = False,
    location: float = pr.logistics_location,
    scale: float = pr.logistics_scale,
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

    1. If ``method`` (default: {method}) is ``logistics``, compute the mean value of the logistics
       function between the variables of each pair of observations.

    2. Otherwise, the ``method`` must be ``pearson``. Compute the cross-correlation between all the
       observations (cells), and if ``repeated`` (default: {repeated}), compute the
       cross-correlation of the correlations. That is, for two observations (cells) to be similar,
       they would need to be similar to the rest of the observations (cells) in the same way. This
       compensates for the extreme sparsity of the data.
    '''
    return _compute_elements_similarity(adata, 'obs', of,
                                        method=method,
                                        repeated=repeated,
                                        location=location,
                                        scale=scale,
                                        inplace=inplace,
                                        intermediate=intermediate)


@ut.timed_call()
@ut.expand_doc()
def compute_var_var_similarity(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    method: str = pr.similarity_method,
    repeated: bool = False,
    location: float = pr.logistics_location,
    scale: float = pr.logistics_scale,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[ut.PandasFrame]:
    '''
    Compute a measure of the similarity between the variables (genes) ``of`` some data (by
    default, the focus).

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    The ``method`` (default: {method}) can be one of ``pearson`` for computing Pearson correlation
    or ``logistics`` for computing the logistics function. If the latter is, used, the function is
    parameterized using the ``location`` (default: {location}) and ``scale`` (default: {scale}).

    **Returns**

    Variable-Pair (genes) Annotations
        ``var_similarity``
            A square matrix where each entry is the similarity between a pair of cells.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas data frame (indexed by the variable names).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. If ``method`` (default: {method}) is ``logistics``, compute the mean value of the logistics
       function between the observations of each pair of variables.

    2. Otherwise, the ``method`` must be ``pearson``. Compute the cross-correlation between all the
       varibles, and if ``repeated`` (default: {repeated}), compute the cross-correlation of the
       correlations. That is, for two variables (genes) to be similar, they would need to be
       similar to the rest of the variables (genes) in the same way. This compensates for the
       extreme sparsity of the data.
    '''
    return _compute_elements_similarity(adata, 'var', of,
                                        method=method,
                                        repeated=repeated,
                                        location=location,
                                        scale=scale,
                                        inplace=inplace,
                                        intermediate=intermediate)


def _compute_elements_similarity(  # pylint: disable=too-many-branches
    adata: AnnData,
    elements: str,
    of: Optional[str],
    *,
    repeated: bool,
    method: str,
    location: float,
    scale: float,
    inplace: bool,
    intermediate: bool,
) -> Optional[ut.PandasFrame]:
    assert elements in ('obs', 'var')

    assert method in ('pearson', 'logistics')

    of, _ = \
        ut.log_operation(LOG, adata,
                         'compute_%s_%s_similarity' % (elements, elements), of)

    LOG.debug('  method: %s', method)

    with ut.intermediate_step(adata, intermediate=intermediate):
        if method == 'logistics':
            LOG.debug('  location: %s', location)
            LOG.debug('  scale: %s', scale)
            if elements == 'obs':
                similarity = pp.get_obs_obs_logistics(adata, of,
                                                      location=location, scale=scale,
                                                      inplace=inplace)
            else:
                similarity = pp.get_var_var_logistics(adata, of,
                                                      location=location, scale=scale,
                                                      inplace=inplace)
        else:
            LOG.debug('  repeated: %s', repeated)

            if elements == 'obs':
                similarity = pp.get_obs_obs_correlation(adata, of,
                                                        inplace=inplace or repeated)
                if repeated:
                    similarity = pp.get_obs_obs_correlation(adata, similarity.name,
                                                            inplace=inplace)
            else:
                similarity = pp.get_var_var_correlation(adata, of,
                                                        inplace=inplace or repeated)
                if repeated:
                    similarity = pp.get_var_var_correlation(adata, similarity.name,
                                                            inplace=inplace)

    if inplace:
        to = elements + '_similarity'
        if elements == 'obs':
            ut.set_oo_data(adata, to, similarity.proper)
        else:
            ut.set_vv_data(adata, to, similarity.proper)
        return None

    if elements == 'obs':
        names = adata.obs_names
    else:
        names = adata.var_names

    return pd.DataFrame(similarity.proper, index=names, columns=names)
