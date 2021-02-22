'''
Collect
-------
'''

from typing import Union

from anndata import AnnData

import metacells.preprocessing as pp
import metacells.utilities as ut

__all__ = [
    'collect_metacells',
]


@ut.logged()
@ut.timed_call()
def collect_metacells(
    adata: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    *,
    name: str = 'metacells',
    top_level: bool = True,
) -> AnnData:
    '''
    Collect computed metacells data.

    **Input**

    Annotated (presumably "clean") ``adata``, where the observations are cells and the variables are
    genes, and the "clean" ``cdata`` we have computed metacells for.

    **Returns**

    Annotated metacell data containing for each observation the sum ``of`` the data (by default,
    the focus) of the cells for each metacell, which contains the following annotations:

    Variable (Gene) Annotations
        ``excluded_gene``
            A mask of the genes which were excluded by name.

        ``clean_gene``
            A boolean mask of the clean genes.

        ``forbidden_gene``
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name. This is ``False`` for non-"clean" genes.

        If directly computing metecalls:

        ``feature``
            A boolean mask of the "feature" genes. This is ``False`` for non-"clean" genes.

        If using divide-and-conquer:

        ``pre_feature``, ``feature``
            The number of times the gene was used as a feature when computing the preliminary and
            final metacells. This is zero for non-"clean" genes.

    Observations (Cell) Annotations
        ``grouped``
            The number of ("clean") cells grouped into each metacell.

        ``candidate``
            The index of the candidate metacell each cell was assigned to to. This is ``-1`` for
            non-"clean" cells.

    Also sets all relevant annotations in the full data based on their value in the clean data, with
    appropriate defaults for non-"clean" data.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.preprocessing.group.group_obs_data` to sum the cells into
       metacells.

    2. Pass all relevant per-gene and per-cell annotations to the result.
    '''
    mdata = pp.group_obs_data(adata, what, groups='metacell', name=name)
    assert mdata is not None
    if top_level:
        ut.top_level(mdata)

    for annotation_name in ('excluded_gene',
                            'clean_gene',
                            'forbidden_gene',
                            'pre_feature_gene',
                            'feature_gene'):
        if not ut.has_data(adata, annotation_name):
            continue
        value_per_gene = ut.get_v_numpy(adata, annotation_name,
                                        formatter=ut.mask_description)
        ut.set_v_data(mdata, annotation_name, value_per_gene,
                      formatter=ut.mask_description)

    for annotation_name in ('pile', 'candidate'):
        if ut.has_data(adata, annotation_name):
            pp.group_obs_annotation(adata, mdata, groups='metacell',
                                    name=annotation_name, method='unique')

    return mdata
