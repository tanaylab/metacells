'''
Feature
-------
'''

import logging
from re import Pattern
from typing import Collection, Optional, Union

from anndata import AnnData

import metacells.parameters as pr
import metacells.preprocessing as pp
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    'extract_feature_data',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def extract_feature_data(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    name: Optional[str] = '.feature',
    tmp: bool = False,
    downsample_cell_quantile: float = pr.feature_downsample_cell_quantile,
    min_gene_fraction: float = pr.feature_min_gene_fraction,
    min_gene_relative_variance: float = pr.feature_min_gene_relative_variance,
    forbidden_gene_names: Optional[Collection[str]] = None,
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    random_seed: int = 0,
    intermediate: bool = True,
) -> Optional[AnnData]:
    '''
    Extract a "feature" subset of the ``adata`` to compute metacells for.

    When computing metacells (or clustering cells in general), it makes sense to use a subset of the
    genes for computing cell-cell similarity, for both technical (e.g., too low an expression level)
    and biological (e.g., ignoring bookkeeping and cell cycle genes) reasons. The steps provided
    here are expected to be generically useful, but as always specific data sets may require custom
    feature selection steps on a case-by-case basis.

    **Input**

    A presumably "clean" :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where
    the observations are cells and the variables are genes.

    All the computations will use the ``of`` data (by default, the focus).

    **Returns**

    Returns annotated sliced data containing the "feature" subset of the original data. The focus of
    the data will be the (slice) ``of`` the (downsampled) input data. By default, the ``name`` of
    this data is {name}. If no features were selected, return ``None``.

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse, and set the ``feature`` per-variable boolean mask of ``adata``. Otherwise,
    discard all intermediate data.

    If ``intermediate``, this will also set the following annotations in the full ``adata``:

    Variable (Gene) Annotations
        ``high_fraction_gene``
            A boolean mask of genes with "high" expression level.

        ``high_relative_variance_gene``
            A boolean mask of genes with "high" normalized variance, relative to other genes with a
            similar expression level.

        ``forbidden``
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name.

        ``feature``
            A boolean mask of the "feature" genes.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.downsample.downsample_cells` to downsample the cells to the
       same total number of UMIs, using the ``downsample_cell_quantile`` (default:
       {downsample_cell_quantile}) and the ``random_seed`` (default: {random_seed}).

    2. Invoke :py:func:`metacells.tools.high.find_high_fraction_genes` to select high-expression
       feature genes (based on the downsampled data), using ``min_gene_fraction``.

    3. Invoke :py:func:`metacells.tools.high.find_high_relative_variance_genes` to select
       high-variance feature genes (based on the downsampled data), using
       ``min_gene_relative_variance``.

    4. Invoke :py:func:`metacells.tools.named.find_named_genes` to forbid genes from being used as
       feature genes, based on their name. using the ``forbidden_gene_names`` (default:
       {forbidden_gene_names}) and ``forbidden_gene_patterns`` (default: {forbidden_gene_patterns}).
       This is stored in an intermediate per-variable (gene) ``forbidden_genes`` boolean mask.

    5. Invoke :py:func:`metacells.preprocessing.filter.filter_data` to slice just the selected
       "feature" genes using the ``name`` (default: {name}) and ``tmp`` (default: {tmp}).
    '''
    ut.log_pipeline_step(LOG, adata, 'extract_feature_data')

    with ut.focus_on(ut.get_vo_data, adata, of,
                     intermediate=intermediate, keep='feature'):
        tl.downsample_cells(adata,
                            downsample_cell_quantile=downsample_cell_quantile,
                            random_seed=random_seed,
                            infocus=True)

        tl.find_high_fraction_genes(adata,
                                    min_gene_fraction=min_gene_fraction)

        tl.find_high_relative_variance_genes(adata,
                                             min_gene_relative_variance=min_gene_relative_variance)

        if forbidden_gene_names is not None \
                or forbidden_gene_patterns is not None:
            tl.find_named_genes(adata,
                                to='forbidden',
                                names=forbidden_gene_names,
                                patterns=forbidden_gene_patterns)

        results = pp.filter_data(adata, name=name, tmp=tmp,
                                 mask_var='feature',
                                 masks=['high_fraction_gene',
                                        'high_relative_variance_gene',
                                        '~forbidden'])
        if results is None:
            raise ValueError('Empty feature data, giving up')

    fdata = results[0]

    ut.get_vo_data(fdata, ut.get_focus_name(adata), infocus=True)

    return fdata
