'''
Cross-Similarity
----------------
'''

import logging
from typing import Optional, Union

from anndata import AnnData

import metacells.parameters as pr
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
    what: Union[str, ut.Matrix] = '__x__',
    *,
    method: str = pr.similarity_method,
    location: float = pr.logistics_location,
    scale: float = pr.logistics_scale,
    inplace: bool = True,
) -> Optional[ut.PandasFrame]:
    '''
    Compute a measure of the similarity between the observations (cells) ``of`` some data (by
    default, the focus).

    The ``method`` (default: {method}) can be one of:
    * ``pearson`` for computing Pearson correlation.
    * ``repeated_pearson`` for computing correlations-of-correlations.
    * ``logistics`` for computing the logistics function.
    * ``logistics_pearson`` for computing correlations-of-logistics.

    If using the logistics function, use the ``scale`` (default: {scale}) and ``location`` (default:
    {location}).

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Observations-Pair (cells) Annotations
        ``obs_similarity``
            A square matrix where each entry is the similarity between a pair of cells.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas data frame (indexed by the observation names).

    **Computation Parameters**

    1. If ``method`` (default: {method}) is ``logistics`` or ``logistics_pearson``, compute the mean
       value of the logistics function between the variables of each pair of observations (cells).
       Otherwise, it should be ``pearson`` or ``repeated_pearson``, so compute the cross-correlation
       between all the observations.

    2. If the ``method`` is ``logistics_pearson`` or ``repeated_pearson``, then compute the
       cross-correlation of the results of the previous step. That is, two observations (cells) will
       be similar if they are similar to the rest of the observations (cells) in the same way. This
       compensates for the extreme sparsity of the data.
    '''
    return _compute_elements_similarity(adata, 'obs', 'row', what,
                                        method=method,
                                        location=location,
                                        scale=scale,
                                        inplace=inplace)


@ut.timed_call()
@ut.expand_doc()
def compute_var_var_similarity(
    adata: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    *,
    method: str = pr.similarity_method,
    location: float = pr.logistics_location,
    scale: float = pr.logistics_scale,
    inplace: bool = True,
) -> Optional[ut.PandasFrame]:
    '''
    Compute a measure of the similarity between the variables (genes) ``of`` some data (by
    default, the focus).

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    The ``method`` (default: {method}) can be one of:
    * ``pearson`` for computing Pearson correlation.
    * ``repeated_pearson`` for computing correlations-of-correlations.
    * ``logistics`` for computing the logistics function.
    * ``logistics_pearson`` for computing correlations-of-logistics.

    If using the logistics function, use the ``scale`` (default: {scale}) and ``location`` (default:
    {location}).

    **Returns**

    Variable-Pair (genes) Annotations
        ``var_similarity``
            A square matrix where each entry is the similarity between a pair of genes.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas data frame (indexed by the variable names).

    **Computation Parameters**

    1. If ``method`` (default: {method}) is ``logistics`` or ``logistics_pearson``, compute the mean
       value of the logistics function between the variables of each pair of variables (genes).
       Otherwise, it should be ``pearson`` or ``repeated_pearson``, so compute the cross-correlation
       between all the variables.

    2. If the ``method`` is ``logistics_pearson`` or ``repeated_pearson``, then compute the
       cross-correlation of the results of the previous step. That is, two variables (genes) will
       be similar if they are similar to the rest of the variables (genes) in the same way. This
       compensates for the extreme sparsity of the data.
    '''
    return _compute_elements_similarity(adata, 'var', 'column', what,
                                        method=method,
                                        location=location,
                                        scale=scale,
                                        inplace=inplace)


def _compute_elements_similarity(
    adata: AnnData,
    elements: str,
    per: str,
    what: Union[str, ut.Matrix],
    *,
    method: str,
    location: float,
    scale: float,
    inplace: bool,
) -> Optional[ut.PandasFrame]:
    assert elements in ('obs', 'var')

    assert method in ('pearson', 'repeated_pearson',
                      'logistics', 'logistics_pearson')

    ut.log_operation(LOG, adata,
                     'compute_%s_%s_similarity' % (elements, elements),
                     what if isinstance(what, str) else '<data>')

    LOG.debug('  method: %s', method)

    data = ut.get_vo_proper(adata, what, layout=f'{per}_major')

    if method.startswith('logistics'):
        LOG.debug('  location: %s', location)
        LOG.debug('  scale: %s', scale)
        similarity = ut.logistics(data, per=per)
    else:
        similarity = ut.corrcoef(data, per=per)

    if method.endswith('_pearson'):
        similarity = ut.corrcoef(similarity, per=None)

    if inplace:
        to = elements + '_similarity'
        if elements == 'obs':
            ut.set_oo_data(adata, to, similarity)
        else:
            ut.set_vv_data(adata, to, similarity)
        return None

    if elements == 'obs':
        names = adata.obs_names
    else:
        names = adata.var_names

    return ut.to_pandas_frame(similarity, index=names, columns=names)
