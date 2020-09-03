'''
Clean Data
----------
'''

import logging
from re import Pattern
from typing import Collection, Optional, Union

from anndata import AnnData

import metacells.preprocessing as pp
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    'extract_clean_data',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def extract_clean_data(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    name: Optional[str] = 'CLEAN',
    tmp: bool = True,
    min_total_of_cells: Optional[int] = 800,
    max_total_of_cells: Optional[int] = None,
    min_total_of_genes: int = 1,
    excluded_gene_names: Optional[Collection[str]] = None,
    excluded_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    intermediate: bool = True,
) -> Optional[AnnData]:
    '''
    Extract a "clean" subset of the ``adata`` to compute metacells for.

    Raw single-cell RNA sequencing data is notoriously noisy and "dirty". This pipeline step
    performs initial filtering of the data so that the rest of the pipeline works on "clean" data.
    The steps provided here are expected to be generically useful, but as always specific data sets
    may require custom cleaning steps on a case-by-case basis.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    All the computations will use the ``of`` data (by default, the focus).

    **Returns**

    Annotated sliced data containing the "clean" subset of the original data. The focus of the data
    will be the (slice) ``of`` the input data. By default, the ``name`` of this data is ``CLEAN``.

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.preprocessing.common.track_base_indices`. This ensures we can map
       each element of the clean data back to its index in the input full data.

    2. Invoke :py:func:`metacells.tools.properly_sampled.find_properly_sampled_cells` using
       ``min_total_of_cells`` (default: {min_total_of_cells}) and ``max_total_of_cells`` (default:
       {max_total_of_cells}).

    3. Invoke :py:func:`metacells.tools.properly_sampled.find_properly_sampled_genes` using
       ``min_total_of_genes`` (default: {min_total_of_genes}).

    4. Invoke :py:func:`metacells.tools.named_genes.find_named_genes` to exclude genes based on
       their name, using the ``excluded_gene_names`` (default: {excluded_gene_names}) and
       ``excluded_gene_patterns`` (default: {excluded_gene_patterns}). This is stored in an
       intermediate per-variable (gene) ``excluded_genes`` boolean mask.

    5. Invoke :py:func:`metacells.preprocessing.filter_data.filter_data` to slice just the surviving
       cells and genes using the ``name`` (default: {name}) and ``tmp`` (default: {tmp}).
    '''
    ut.log_pipeline_step(LOG, adata, 'clean_data')

    pp.track_base_indices(adata)

    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        tl.find_properly_sampled_cells(adata,
                                       min_total_of_cells=min_total_of_cells,
                                       max_total_of_cells=max_total_of_cells)

        tl.find_properly_sampled_genes(adata,
                                       min_total_of_genes=min_total_of_genes)

        if excluded_gene_names is not None \
                or excluded_gene_patterns is not None:
            tl.find_named_genes(adata,
                                to='excluded_genes',
                                names=excluded_gene_names,
                                patterns=excluded_gene_patterns)

        cdata = pp.filter_data(adata, name=name, tmp=tmp,
                               masks=['properly_sampled_cells',
                                      'properly_sampled_genes',
                                      '~excluded_genes'])

    if cdata is not None:
        ut.get_vo_data(cdata, ut.get_focus_name(adata), infocus=True)

    return cdata
