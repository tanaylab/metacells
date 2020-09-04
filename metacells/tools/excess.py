'''
Compute the excess gene-gene coefficient of determination (R^2).
'''

import logging
from typing import Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    'compute_excess_r2',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def compute_excess_r2(  # pylint: disable=too-many-branches,too-many-statements
    adata: AnnData,
    of: Optional[str] = None,
    *,
    metacells: Union[str, ut.Vector] = 'metacell',
    downsample_cell_quantile: float = pr.excess_downsample_cell_quantile,
    min_gene_total: float = pr.excess_min_gene_total,
    top_gene_rank: int = pr.excess_top_gene_rank,
    shuffles_count: int = pr.excess_shuffles_count,
    random_seed: int = pr.random_seed,
    inplace: bool = True,
    intermediate: bool = True,
    mdata: Optional[AnnData] = None,
    mindices: Optional[Union[str, ut.Vector]] = None,
) -> Optional[ut.PandasSeries]:
    '''
    Compute the excess gene-gene coefficient of determination (R^2) for the metacells based ``of``
    some data.

    In an ideal metacell, all cells have the same biological state, and the only variation is due to
    sampling noise. In such an ideal metacell, there would be zero R^2 between the expression of
    different genes. Naturally, ideal metacells do not exist. Measuring internal gene-gene
    correlation in each metacell gives us an estimate of the quality of the metacells we got.

    Due to technical sampling issues, even if we randomize each gene's expression across the cells
    in each metacell, we would still get some residual R^2 value. We therefore subtract this
    from the actual R^2 value to obtain the "excess R^2" value, which we use as a rough measure
    of the metacell's quality.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    **Returns**

    Variable (Gene) Annotations

        ``gene_max_excess_r2``
            For each gene, the maximal difference (across all metacells, every other gene) between
            the R^2 coefficient of determination of the two genes, and the shuffled coefficient of
            determination of the two genes (see below).

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the gene names).

    If ``mdata`` is specified, it is assumed to have one observation for each metacell, and use the
    same genes as ``adata`` (if it doesn't, specify ``mindices`` to the name of an annotation or a
    vector that maps each ``adata`` gene index to a ``mdata`` gene index). The ``excess_r2`` for
    each metacell will be stored there as a per-variable-per-observation annotation (layer).

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it. In addition, ``gene_max_top_r2`` and
    ``gene_max_top_shuffled_r2`` will be stored for each variable in ``adata``, and if ``mdata`` was
    specified, store ``top_r2`` and ``top_shuffled_r2`` per-variable-per-observation in it as well.

    **Computation Parameters**

    1. For each metacell:

        1.1. Downsample the cells so their total number of UMIs would be the
             ``downsample_cell_quantile`` (default: {downsample_cell_quantile}). That is, if
             ``downsample_cell_quantile`` is 0.1, then 90% of the cells would have more UMIs than
             the target number of samples and will be downsampled; 10% of the cells would have too
             few UMIs and will be left unchanged.

        1.2 Ignore all genes whose total data in the metacell (in the downsampled data) is below the
            ``min_gene_total`` (default: {min_gene_total}). This ensures we only correlate genes
            with sufficient data to be meaningful. We also ignore any genes which have exactly the
            same value in all cells of the metacell, regardless of their expression level.

        1.3 Compute the cross-correlation between all the remaining genes and square it to obtain
            the metacell's per-gene-per-gene R^2 data. Collect for each gene the ``top_gene_rank``
            value, which is the gene's top R^2 for this metacell.

        1.4 Randomly shuffle the remaining gene expressions between the cells of each metacells
            using the ``random_seed``, re-compute the cross-correlation and square it into the
            metacell's per-gene-per-gene shuffled R^2 data. Collect for each gene the
            ``top_gene_rank`` value. Repeat this ``shuffles_count`` times and use the average as the
            gene's top shuffled R^2 for this metacell.

        1.5 The difference, for each gene, between the gene's top R^2, and the gene's (averaged) top
            shuffled R^2, is the gene's excess R^2 for this metacell.

    2. Collect, for each gene, the maximal excess R^2 across all metacells into
       ``gene_max_excess_r2``. If ``intermediate``, also compute the global ``gene_max_top_r2`` and
       ``gene_max_top_shuffled_r2``. Note
       that ``gene_max_excess_r2 != gene_max_top_r2 - gene_max_top_shuffled_r2``.
    '''
    of, level = ut.log_operation(LOG, adata, 'compute_excess_r2', of)
    assert shuffles_count > 0

    with ut.focus_on(ut.get_vo_data, adata, of, layout='row_major',
                     intermediate=intermediate) as data:
        data = ut.to_proper_matrix(data)

        metacell_of_cells = \
            ut.get_vector_parameter_data(LOG, level, adata, metacells,
                                         per='o', name='metacells')
        assert metacell_of_cells is not None

        metacells_count = np.max(metacell_of_cells) + 1
        assert metacells_count > 0

        if mdata is not None:
            LOG.log(level, '  mdata: %s', ut.get_name(mdata) or '<adata>')
            mindex_of_genes = \
                ut.get_vector_parameter_data(LOG, level, adata, mindices,
                                             per='v', name='mindices',
                                             default='same')
        if mindex_of_genes is None:
            mindex_of_genes = np.arange(adata.n_vars)

        max_excess_r2_per_gene = np.full(adata.n_vars, -10, dtype='float')
        max_top_r2_per_gene = None
        max_top_shuffled_r2_per_gene = None
        if intermediate:
            max_top_r2_per_gene = np.full(adata.n_vars, -10, dtype='float')
            max_top_shuffled_r2_per_gene = \
                np.full(adata.n_vars, -10, dtype='float')

        excess_r2_per_gene_per_metacell = None
        top_r2_per_gene_per_metacell = None
        top_shuffled_r2_per_gene_per_metacell = None
        if mdata is not None:
            assert mdata.n_obs == metacells_count
            excess_r2_per_gene_per_metacell = \
                np.full(mdata.shape, None, dtype='float')
            if intermediate:
                top_r2_per_gene_per_metacell = \
                    np.full(mdata.shape, None, dtype='float')
                top_shuffled_r2_per_gene_per_metacell = \
                    np.full(mdata.shape, None, dtype='float')

        LOG.log(level, '  downsample_cell_quantile: %s',
                downsample_cell_quantile)
        LOG.log(level, '  min_gene_total: %s', min_gene_total)
        LOG.log(level, '  top_gene_rank: %s', top_gene_rank)

        for metacell_index in range(metacells_count):
            _collect_metacell_excess(metacell_index, metacells_count,
                                     metacell_of_cells=metacell_of_cells,
                                     data=data,
                                     downsample_cell_quantile=downsample_cell_quantile,
                                     random_seed=random_seed,
                                     shuffles_count=shuffles_count,
                                     top_gene_rank=top_gene_rank,
                                     min_gene_total=min_gene_total,
                                     max_excess_r2_per_gene=max_excess_r2_per_gene,
                                     max_top_r2_per_gene=max_top_r2_per_gene,
                                     max_top_shuffled_r2_per_gene=max_top_shuffled_r2_per_gene,
                                     mindex_of_genes=mindex_of_genes,
                                     excess_r2_per_gene_per_metacell=excess_r2_per_gene_per_metacell,
                                     top_r2_per_gene_per_metacell=top_r2_per_gene_per_metacell,
                                     top_shuffled_r2_per_gene_per_metacell=top_shuffled_r2_per_gene_per_metacell)

    if excess_r2_per_gene_per_metacell is not None:
        excess_r2_per_gene_per_metacell[excess_r2_per_gene_per_metacell < -5] = None
        assert mdata is not None
        ut.set_vo_data(mdata, 'excess_r2', excess_r2_per_gene_per_metacell)
        if top_r2_per_gene_per_metacell is not None:
            top_r2_per_gene_per_metacell[top_r2_per_gene_per_metacell < -5] = None
            ut.set_vo_data(mdata, 'top_r2', top_r2_per_gene_per_metacell)
        if top_shuffled_r2_per_gene_per_metacell is not None:
            top_shuffled_r2_per_gene_per_metacell[top_shuffled_r2_per_gene_per_metacell < -5] = None
            ut.set_vo_data(mdata, 'top_shuffled_r2',
                           top_shuffled_r2_per_gene_per_metacell)

    max_excess_r2_per_gene[max_excess_r2_per_gene < -5] = None

    if inplace:
        ut.set_v_data(adata, 'gene_max_excess_r2', max_excess_r2_per_gene,
                      log_value=lambda: _log_r2(max_excess_r2_per_gene))
        if max_top_r2_per_gene is not None:
            max_top_r2_per_gene[max_top_r2_per_gene < -5] = None
            ut.set_v_data(adata, 'gene_max_top_r2', max_top_r2_per_gene,
                          log_value=lambda: _log_r2(max_top_r2_per_gene))
        if max_top_shuffled_r2_per_gene is not None:
            max_top_shuffled_r2_per_gene[max_top_shuffled_r2_per_gene < -5] = None
            ut.set_v_data(adata, 'gene_max_top_shuffled_r2',
                          max_top_shuffled_r2_per_gene,
                          log_value=lambda: _log_r2(max_top_shuffled_r2_per_gene))
        return None

    if LOG.isEnabledFor(level):
        LOG.log(level, '  gene_max_excess_r2: %s',
                _log_r2(max_excess_r2_per_gene))
        if max_top_r2_per_gene is not None:
            LOG.log(level, '  gene_max_top_r2: %s',
                    _log_r2(max_top_r2_per_gene))
        if max_top_shuffled_r2_per_gene is not None:
            LOG.log(level, '  gene_max_top_shuffled_r2: %s',
                    _log_r2(max_top_shuffled_r2_per_gene))

    return pd.DataSeries(max_excess_r2_per_gene, index=adata.var_names)


def _log_r2(values: Optional[ut.DenseVector]) -> str:
    assert values is not None
    return '%s for %s' % (np.nanmean(values), ut.mask_description(~np.isnan(values)))


def _collect_metacell_excess(  # pylint: disable=too-many-statements
    metacell_index: int,
    metacells_count: int,
    *,
    metacell_of_cells: ut.DenseVector,
    data: ut.ProperMatrix,
    downsample_cell_quantile: float,
    random_seed: int,
    shuffles_count: int,
    min_gene_total: float,
    top_gene_rank: int,
    max_excess_r2_per_gene: ut.DenseVector,
    max_top_r2_per_gene: Optional[ut.DenseVector],
    max_top_shuffled_r2_per_gene: Optional[ut.DenseVector],
    mindex_of_genes: ut.DenseVector,
    excess_r2_per_gene_per_metacell: Optional[ut.DenseMatrix],
    top_r2_per_gene_per_metacell: Optional[ut.DenseMatrix],
    top_shuffled_r2_per_gene_per_metacell: Optional[ut.DenseMatrix],
) -> None:
    LOG.debug('  - metacell: %s / %s', metacell_index, metacells_count)
    metacell_mask = metacell_of_cells == metacell_index
    assert ut.matrix_layout(data) == 'row_major'
    metacell_data = data[metacell_mask, :]
    assert metacell_data.shape[0] > 0

    total_per_cell = ut.sum_per(metacell_data, per='row')
    samples = round(np.quantile(total_per_cell, downsample_cell_quantile))
    LOG.debug('    samples: %s', samples)
    downsampled_data_rows = \
        ut.downsample_matrix(metacell_data, per='row', samples=samples)

    downsampled_data_columns = \
        ut.to_layout(downsampled_data_rows, layout='column_major')
    total_per_gene = \
        ut.to_dense_vector(ut.sum_per(downsampled_data_columns, per='column'))
    min_per_gene = \
        ut.to_dense_vector(ut.min_per(downsampled_data_columns, per='column'))
    max_per_gene = \
        ut.to_dense_vector(ut.max_per(downsampled_data_columns, per='column'))
    correlated_genes_mask = \
        (total_per_gene >= min_gene_total) & (min_per_gene < max_per_gene)
    correlated_genes_count = np.sum(correlated_genes_mask)
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug('    correlate genes: %s',
                  ut.ratio_description(correlated_genes_count,
                                       correlated_genes_mask.size))
    assert correlated_genes_count > top_gene_rank
    correlated_gene_rank = correlated_genes_count - top_gene_rank - 1

    correlated_gene_indices = np.where(correlated_genes_mask)[0]
    mindex_of_correlated_genes = mindex_of_genes[correlated_gene_indices]
    correlated_data = downsampled_data_columns[:, correlated_gene_indices]
    assert ut.matrix_layout(correlated_data) == 'column_major'

    metacell_r2 = ut.corrcoef(correlated_data, per='column')
    metacell_r2 *= metacell_r2
    np.fill_diagonal(metacell_r2, 0)

    top_r2_per_correlated_gene = \
        ut.rank_per(metacell_r2, per=None, rank=correlated_gene_rank)

    if top_r2_per_gene_per_metacell is not None:
        top_r2_per_gene_per_metacell[metacell_index,
                                     mindex_of_correlated_genes] = \
            top_r2_per_correlated_gene

    if max_top_r2_per_gene is not None:
        max_top_r2_per_gene[correlated_genes_mask] = \
            np.maximum(max_top_r2_per_gene[correlated_genes_mask],
                       top_r2_per_correlated_gene)

    shuffled_data = correlated_data.copy()

    top_shuffled_r2_per_correlated_gene = \
        np.zeros(top_r2_per_correlated_gene.size)

    with ut.timed_step('.shuffle'):
        for shuffle_index in range(shuffles_count):
            if random_seed == 0:
                shuffle_seed = 0
            else:
                shuffle_seed = random_seed + 977 * shuffle_index

            ut.shuffle_matrix(shuffled_data, per='column',
                              random_seed=shuffle_seed)

            shuffled_r2 = ut.corrcoef(shuffled_data, per='column')

            shuffled_r2 *= shuffled_r2
            assert ut.matrix_layout(shuffled_r2) == 'row_major'
            np.fill_diagonal(shuffled_r2, 0)

            top_shuffled_r2_per_correlated_gene += \
                ut.rank_per(shuffled_r2, per=None, rank=correlated_gene_rank)

        top_shuffled_r2_per_correlated_gene /= shuffles_count

    if top_shuffled_r2_per_gene_per_metacell is not None:
        top_shuffled_r2_per_gene_per_metacell[metacell_index,
                                              mindex_of_correlated_genes] = \
            top_shuffled_r2_per_correlated_gene

    if max_top_shuffled_r2_per_gene is not None:
        max_top_shuffled_r2_per_gene[correlated_genes_mask] = \
            np.maximum(max_top_shuffled_r2_per_gene[correlated_genes_mask],
                       top_shuffled_r2_per_correlated_gene)

    excess_r2_per_correlated_gene = \
        top_r2_per_correlated_gene - top_shuffled_r2_per_correlated_gene

    if excess_r2_per_gene_per_metacell is not None:
        excess_r2_per_gene_per_metacell[metacell_index, mindex_of_correlated_genes] = \
            excess_r2_per_correlated_gene

    max_excess_r2_per_gene[correlated_genes_mask] = \
        np.maximum(max_excess_r2_per_gene[correlated_genes_mask],
                   excess_r2_per_correlated_gene)
