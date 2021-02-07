'''
Collect
-------
'''

import logging
from typing import Union

from anndata import AnnData

import metacells.preprocessing as pp
import metacells.utilities as ut

__all__ = [
    'collect_metacells',
]


LOG = logging.getLogger(__name__)


@ut.timed_call('collect_metacells')
def collect_metacells(
    adata: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    *,
    name: str = 'metacells',
    tmp: bool = False,
    intermediate: bool = True,
) -> AnnData:
    '''
    Collect computed metacells data.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated (presumably "clean") ``adata``,
    where the observations are cells and the variables are genes, and the "clean" ``cdata`` we have
    computed metacells for.

    **Returns**

    Annotated metacell data containing for each observation the sum ``of`` the data (by default,
    the focus) of the cells for each metacell, which contains the following annotations:

    Variable (Gene) Annotations
        ``excluded_gene`` (if ``intermediate``)
            A mask of the genes which were excluded by name.

        ``clean_gene`` (if ``intermediate``)
            A boolean mask of the clean genes.

        ``forbidden_gene`` (if ``intermediate``)
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name. This is ``False`` for non-"clean" genes.

        If directly computing metecalls:

        ``feature``
            A boolean mask of the "feature" genes. This is ``False`` for non-"clean" genes.

        If using divide-and-conquer:

        ``pre_feature`` (if ``intermediate``), ``feature``
            The number of times the gene was used as a feature when computing the preliminary and
            final metacells. This is zero for non-"clean" genes.

    Observations (Cell) Annotations
        ``grouped``
            The number of ("clean") cells grouped into each metacell.

        ``candidate`` (if ``intermediate``)
            The index of the candidate metacell each cell was assigned to to. This is ``-1`` for
            non-"clean" cells.

    Also sets all relevant annotations in the full data based on their value in the clean data, with
    appropriate defaults for non-"clean" data.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.preprocessing.group.group_obs_data` to sum the cells into
       metacells.

    2. Pass all relevant per-gene and per-cell annotations to the result.
    '''
    ut.log_pipeline_step(LOG, adata, 'collect_metacells')

    mdata = \
        pp.group_obs_data(adata, what, groups='metacell', name=name, tmp=tmp)
    assert mdata is not None

    for annotation_name, always in (('excluded_gene', False),
                                    ('clean_gene', False),
                                    ('forbidden_gene', False),
                                    ('pre_feature_gene', False),
                                    ('feature_gene', True)):
        if not always and not intermediate:
            continue
        if not ut.has_data(adata, annotation_name) \
                and (annotation_name.startswith('pre_')
                     or annotation_name in ('excluded_gene', 'clean_gene')):
            continue
        value_per_gene = ut.get_v_dense(adata, annotation_name)
        ut.set_v_data(mdata, annotation_name, value_per_gene,
                      log_value=ut.mask_description)

    if intermediate:
        for annotation_name, always in (('pile', False),
                                        ('candidate', True)):
            if always or ut.has_data(adata, annotation_name):
                pp.group_obs_annotation(adata, mdata, groups='metacell',
                                        name=annotation_name, method='unique')

    return mdata
