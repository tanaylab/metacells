'''
Clean
-----

Raw single-cell RNA sequencing data is notoriously noisy and "dirty". The pipeline steps here
performs initial analysis of the data and extract just the "clean" data for actually computing the
metacells. The steps provided here are expected to be generically useful, but as always specific
data sets may require custom cleaning steps on a case-by-case basis.
'''

from re import Pattern
from typing import Collection, List, Optional, Union

from anndata import AnnData

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    'analyze_clean_genes',
    'pick_clean_genes',
    'analyze_clean_cells',
    'pick_clean_cells',
    'extract_clean_data',
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def analyze_clean_genes(
    adata: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    *,
    properly_sampled_min_gene_total: int = pr.properly_sampled_min_gene_total,
    noisy_lonely_max_sampled_cells: int = pr.noisy_lonely_max_sampled_cells,
    noisy_lonely_downsample_min_samples: int = pr.noisy_lonely_downsample_min_samples,
    noisy_lonely_downsample_min_cell_quantile: float = pr.noisy_lonely_downsample_min_cell_quantile,
    noisy_lonely_downsample_max_cell_quantile: float = pr.noisy_lonely_downsample_max_cell_quantile,
    noisy_lonely_min_gene_total: int = pr.noisy_lonely_min_gene_total,
    noisy_lonely_min_gene_normalized_variance: float = pr.noisy_lonely_min_gene_normalized_variance,
    noisy_lonely_max_gene_similarity: float = pr.noisy_lonely_max_gene_similarity,
    excluded_gene_names: Optional[Collection[str]] = None,
    excluded_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    random_seed: int = pr.random_seed,
) -> None:
    '''
    Analyze genes in preparation for picking the "clean" subset of the ``adata``.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Sets the following in the data:

    Variable (gene) annotations:
        ``properly_sampled_gene``
            A mask of the "properly sampled" genes.

        ``noisy_lonely_gene``
            A mask of the "noisy lonely" genes.

        ``excluded_gene``
            A mask of the genes which were excluded by name.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.properly_sampled.find_properly_sampled_genes` using
       ``properly_sampled_min_gene_total`` (default: {properly_sampled_min_gene_total}).

    2. Invoke :py:func:`metacells.tools.noisy_lonely.find_noisy_lonely_genes` using
       ``noisy_lonely_max_sampled_cells`` (default: {noisy_lonely_max_sampled_cells}),
       ``noisy_lonely_downsample_min_samples`` (default: {noisy_lonely_downsample_min_samples}),
       ``noisy_lonely_downsample_min_cell_quantile`` (default:
       {noisy_lonely_downsample_min_cell_quantile}), ``noisy_lonely_downsample_max_cell_quantile``
       (default: {noisy_lonely_downsample_max_cell_quantile}), ``noisy_lonely_min_gene_total``
       (default: {noisy_lonely_min_gene_total}), ``noisy_lonely_min_gene_normalized_variance``
       (default: {noisy_lonely_min_gene_normalized_variance}), and
       ``noisy_lonely_max_gene_similarity`` (default: {noisy_lonely_max_gene_similarity}).

    3. Invoke :py:func:`metacells.tools.named.find_named_genes` to exclude genes based on their
       name, using the ``excluded_gene_names`` (default: {excluded_gene_names}) and
       ``excluded_gene_patterns`` (default: {excluded_gene_patterns}). This is stored in a
       per-variable (gene) ``excluded_genes`` boolean mask.
    '''
    tl.find_properly_sampled_genes(adata, what,
                                   min_gene_total=properly_sampled_min_gene_total)

    excluded_genes_mask: Optional[str]
    if excluded_gene_names is not None or excluded_gene_patterns is not None:
        excluded_genes_mask = 'excluded_gene'
        tl.find_named_genes(adata,
                            to='excluded_gene',
                            names=excluded_gene_names,
                            patterns=excluded_gene_patterns)
    else:
        excluded_genes_mask = None

    tl.find_noisy_lonely_genes(adata, what,
                               excluded_genes_mask=excluded_genes_mask,
                               max_sampled_cells=noisy_lonely_max_sampled_cells,
                               downsample_min_samples=noisy_lonely_downsample_min_samples,
                               downsample_min_cell_quantile=noisy_lonely_downsample_min_cell_quantile,
                               downsample_max_cell_quantile=noisy_lonely_downsample_max_cell_quantile,
                               min_gene_total=noisy_lonely_min_gene_total,
                               min_gene_normalized_variance=noisy_lonely_min_gene_normalized_variance,
                               max_gene_similarity=noisy_lonely_max_gene_similarity,
                               random_seed=random_seed)


CLEAN_GENES_MASKS = ['properly_sampled_gene',
                     '~noisy_lonely_gene', '~excluded_gene']


@ut.timed_call()
@ut.expand_doc(masks=', '.join(CLEAN_GENES_MASKS))
def pick_clean_genes(  # pylint: disable=dangerous-default-value
    adata: AnnData,
    *,
    masks: List[str] = CLEAN_GENES_MASKS,
    to: str = 'clean_gene'
) -> None:
    '''
    Create a mask of the "clean" genes that will be used to actually compute the metacells.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Sets the following in the data:

    Variable (gene) annotations:
        ``to`` (default: {to})
            A mask of the "clean" genes to use for actually computing the metacells.

    **Computation Parameters**

    1. This simply AND-s the specified ``masks`` (default: {masks}) using
       :py:func:`metacells.tools.mask.combine_masks`.
    '''
    tl.combine_masks(adata, masks, to=to)


@ut.logged()
@ut.timed_call()
def analyze_clean_cells(
    adata: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    *,
    properly_sampled_min_cell_total: Optional[int],
    properly_sampled_max_cell_total: Optional[int],
    properly_sampled_max_excluded_genes_fraction: Optional[float],
) -> None:
    '''
    Analyze cells in preparation for extracting the "clean" subset of the ``adata``.

    Raw single-cell RNA sequencing data is notoriously noisy and "dirty". This pipeline step
    performs initial analysis of the cells to allow us to extract just the "clean" data for
    processing. The steps provided here are expected to be generically useful, but as always
    specific data sets may require custom cleaning steps on a case-by-case basis.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Sets the following in the full data:

    Observation (cell) annotations:
        ``properly_sampled_cell``
            A mask of the "properly sampled" cells.

    **Computation Parameters**

    1. If ``properly_sampled_max_excluded_genes_fraction`` is not ``None``, then consider all the
       genes not covered by the ``clean_gene`` per-variable mask as "excluded" for computing the
       excluded genes fraction for each cell.

    2. Invoke :py:func:`metacells.tools.properly_sampled.find_properly_sampled_cells` using
       ``properly_sampled_min_cell_total`` (no default), ``properly_sampled_max_cell_total`` (no
       default) and ``properly_sampled_max_excluded_genes_fraction`` (no default).
    '''
    excluded_adata: Optional[AnnData] = None
    if properly_sampled_max_excluded_genes_fraction is not None:
        excluded_genes = \
            tl.filter_data(adata, name='dirty_genes', top_level=False,
                           var_masks=['~clean_gene'])
        if excluded_genes is not None:
            excluded_adata = excluded_genes[0]

    if excluded_genes is None:
        max_excluded_genes_fraction = None
    else:
        max_excluded_genes_fraction = properly_sampled_max_excluded_genes_fraction

    tl.find_properly_sampled_cells(adata, what,
                                   min_cell_total=properly_sampled_min_cell_total,
                                   max_cell_total=properly_sampled_max_cell_total,
                                   excluded_adata=excluded_adata,
                                   max_excluded_genes_fraction=max_excluded_genes_fraction)


CLEAN_CELLS_MASKS = ['properly_sampled_cell']


@ut.timed_call()
@ut.expand_doc(masks=', '.join(CLEAN_CELLS_MASKS))
def pick_clean_cells(  # pylint: disable=dangerous-default-value
    adata: AnnData,
    *,
    masks: List[str] = CLEAN_CELLS_MASKS,
    to: str = 'clean_cell'
) -> None:
    '''
    Create a mask of the "clean" cells that will be used to actually compute the metacells.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Sets the following in the data:

    Observation (cell) annotations:
        ``to`` (default: {to})
            A mask of the "clean" cells to use for actually computing the metacells.

    **Computation Parameters**

    1. This simply AND-s the specified ``masks`` (default: {masks}) using
       :py:func:`metacells.tools.mask.combine_masks`.
    '''
    tl.combine_masks(adata, masks, to=to)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def extract_clean_data(
    adata: AnnData,
    obs_mask: str = 'clean_cell',
    var_mask: str = 'clean_gene',
    *,
    name: Optional[str] = '.clean',
    top_level: bool = True,
) -> Optional[AnnData]:
    '''
    Extract a "clean" subset of the ``adata`` to compute metacells for.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Annotated sliced data containing the "clean" subset of the original data. By default, the
    ``name`` of this data is {name}. If this starts with a ``.``, this will be appended to the
    current name of the data (if any).

    The returned data will have ``full_cell_index`` and ``full_gene_index`` per-observation (cell)
    and per-variable (gene) annotations to allow mapping the results back to the original data.

    **Computation Parameters**

    1. This simply :py:func:`metacells.tools.filter.filter_data` to slice just the
       ``obs_mask`` (default: {obs_mask}) and ``var_mask`` (default: {var_mask}) data using the
       ``name`` (default: {name}), and tracking the original ``full_cell_index`` and
       ``full_gene_index``.
    '''
    results = tl.filter_data(adata, name=name, top_level=top_level,
                             track_obs='full_cell_index',
                             track_var='full_gene_index',
                             obs_masks=[obs_mask],
                             var_masks=[var_mask])
    if results is None:
        return None

    return results[0]
