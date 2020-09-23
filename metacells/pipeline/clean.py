'''
Clean
-----
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
    'extract_clean_data',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def extract_clean_data(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    properly_sampled_min_cell_total: Optional[int] = pr.properly_sampled_min_cell_total,
    properly_sampled_max_cell_total: Optional[int] = pr.properly_sampled_max_cell_total,
    properly_sampled_min_gene_total: int = pr.properly_sampled_min_gene_total,
    noisy_lonely_max_sampled_cells: int = pr.noisy_lonely_max_sampled_cells,
    noisy_lonely_downsample_cell_quantile: float = pr.noisy_lonely_downsample_cell_quantile,
    noisy_lonely_min_gene_fraction: float = pr.noisy_lonely_min_gene_fraction,
    noisy_lonely_min_gene_normalized_variance: float = pr.noisy_lonely_min_gene_normalized_variance,
    noisy_lonely_max_gene_similarity: float = pr.noisy_lonely_max_gene_similarity,
    excluded_gene_names: Optional[Collection[str]] = None,
    excluded_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    random_seed: int = pr.random_seed,
    name: Optional[str] = '.clean',
    tmp: bool = False,
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
    will be the (slice) ``of`` the input data. By default, the ``name`` of this data is {name}.
    If this starts with a ``.``, this will be appended to the current name of the data (if any).

    The returned data will have ``full_cell_index`` and ``full_gene_index`` per-observation (cell)
    and per-variable (gene) annotations to allow mapping the results back to the original data.

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    If ``intermediate``, also set the following in the full data:

    Observation (cell) annotations:
        ``properly_sampled_cell``
            A mask of the "properly sampled" cells.

        ``clean_cell`` (if ``intermediate``)
            A boolean mask of the clean cells.

    Variable (gene) annotations:
        ``properly_sampled_gene``
            A mask of the "properly sampled" genes.

        ``noisy_lonely_gene``
            A mask of the "noisy lonely" genes.

        ``excluded``
            A mask of the genes which were excluded by name.

        ``clean_gene``
            A boolean mask of the clean genes.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.properly_sampled.find_properly_sampled_cells` using
       ``properly_sampled_min_cell_total`` (default: {properly_sampled_min_cell_total}) and
       ``properly_sampled_max_cell_total`` (default: {properly_sampled_max_cell_total}).

    2. Invoke :py:func:`metacells.tools.properly_sampled.find_properly_sampled_genes` using
       ``properly_sampled_min_gene_total`` (default: {properly_sampled_min_gene_total}).

    3. Invoke :py:func:`metacells.tools.noisy_lonely.find_noisy_lonely_genes` using
       ``noisy_lonely_max_sampled_cells`` (default: {noisy_lonely_max_sampled_cells}),
       ``noisy_lonely_downsample_cell_quantile`` (default: {noisy_lonely_downsample_cell_quantile}),
       ``noisy_lonely_min_gene_fraction`` (default: {noisy_lonely_min_gene_fraction}),
       ``noisy_lonely_min_gene_normalized_variance`` (default:
       {noisy_lonely_min_gene_normalized_variance}), and ``noisy_lonely_max_gene_similarity``
       (default: {noisy_lonely_max_gene_similarity}).

    4. Invoke :py:func:`metacells.tools.named.find_named_genes` to exclude genes based on their
       name, using the ``excluded_gene_names`` (default: {excluded_gene_names}) and
       ``excluded_gene_patterns`` (default: {excluded_gene_patterns}). This is stored in an
       intermediate per-variable (gene) ``excluded_genes`` boolean mask.

    5. Invoke :py:func:`metacells.preprocessing.filter.filter_data` to slice just the surviving
       cells and genes using the ``name`` (default: {name}) and ``tmp`` (default: {tmp}), and
       tracking the original ``full_cell_index`` and ``full_gene_index``.
    '''
    ut.log_pipeline_step(LOG, adata, 'clean_data')

    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        tl.find_properly_sampled_cells(adata,
                                       min_cell_total=properly_sampled_min_cell_total,
                                       max_cell_total=properly_sampled_max_cell_total)

        tl.find_properly_sampled_genes(adata,
                                       min_gene_total=properly_sampled_min_gene_total)

        tl.find_noisy_lonely_genes(adata,
                                   max_sampled_cells=noisy_lonely_max_sampled_cells,
                                   downsample_cell_quantile=noisy_lonely_downsample_cell_quantile,
                                   min_gene_fraction=noisy_lonely_min_gene_fraction,
                                   min_gene_normalized_variance=noisy_lonely_min_gene_normalized_variance,
                                   max_gene_similarity=noisy_lonely_max_gene_similarity,
                                   random_seed=random_seed)

        if excluded_gene_names is not None \
                or excluded_gene_patterns is not None:
            tl.find_named_genes(adata,
                                to='excluded',
                                names=excluded_gene_names,
                                patterns=excluded_gene_patterns)

        mask_obs = 'clean_cell' if intermediate else None
        mask_var = 'clean_gene' if intermediate else None

        results = pp.filter_data(adata, name=name, tmp=tmp,
                                 mask_obs=mask_obs,
                                 mask_var=mask_var,
                                 track_obs='full_cell_index',
                                 track_var='full_gene_index',
                                 masks=['properly_sampled_cell',
                                        'properly_sampled_gene',
                                        '~noisy_lonely_gene',
                                        '~excluded'])
        if results is None:
            return None

        cdata = results[0]
        if not intermediate:
            ut.del_data(cdata, 'excluded', per='v')

    ut.get_vo_data(cdata, ut.get_focus_name(adata), infocus=True)
    return cdata
