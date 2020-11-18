'''
Excess R^2
----------
'''

import logging
from typing import Optional, Union

import numpy as np  # type: ignore
from anndata import AnnData

import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    'compute_excess_r2',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def compute_excess_r2(
    adata: AnnData,
    of: Optional[str] = None,
    *,
    metacells: Union[str, ut.Vector] = 'metacell',
    downsample_cell_quantile: float = pr.excess_downsample_cell_quantile,
    min_gene_total: float = pr.excess_min_gene_total,
    top_gene_rank: int = pr.excess_top_gene_rank,
    shuffles_count: int = pr.excess_shuffles_count,
    random_seed: int = pr.random_seed,
    intermediate: bool = True,
    mdata: AnnData,
) -> None:
    '''
    Compute the excess gene-gene coefficient of determination (R^2) for the metacells based ``of``
    some data.

    In an ideal metacell, all cells have the same biological state, and the only variation is due to
    sampling noise. In such an ideal metacell, there would be zero R^2 between the expression of
    different genes. Naturally, ideal metacells do not exist. Measuring internal gene-gene
    correlation in each metacell gives us an estimate of the quality of the metacells we got.

    Since we are dealing with sparse data, small metacell sizes, and a large number of genes,
    sometimes we get high gene-gene correlation "just because" two genes happened to have non-zero
    measured expression in the same cell. To estimate this effect, we shuffle the gene expressions
    between the cells. In theory, this should give us zero gene-gene correlation, but due to the
    above effect, we get a significant non-zero correlation (and therefore R^2) value.

    We therefore compensate for this effect by subtracting the maximal R^2 value from the real data
    from the maximal R^2 value from the shuffled data. The average over all genes of this "excess"
    R^2 value serves as a rough measure of the metacell's quality, which we can average over all
    metacells to get an even rougher estimate of the metacells algorithm results. In general we'd
    like to see this being low, with values of a "few" * 0.01 - the lower the better.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are cells and the variables are genes.

    In addition, ``mdata`` is assumed to have one observation for each metacell, and use the same
    genes as ``adata``.

    **Returns**

    Sets the following in ``mdata`` (with ``None`` values for "insignificant" genes, see below):

    Per-Variable Per-Observation (Gene-Cell) Annotations

        ``top_r2``
            For each gene and metacell, the top-``top_gene_rank`` R^2 coefficient of determination
            between the gene and any other gene.

        ``top_shuffled_r2``
            For each gene and metacell, the top-``top_gene_rank`` R^2 coefficient of determination
            between the gene and any other gene, when shuffling the genes between the cells in the
            metacell (this is averaged over multiple shuffles).

        ``excess_r2``
            For each gene and metacell, the difference between the ``top_r2`` and the
            ``top_shuffled_r2``, that is, the excess R^2 coefficient of determination between the

            R^2 coefficient of determination between the genes. Note that ``excess_r2 != top_r2
            - top_shuffled_r2`` because the top difference is different than the difference of the
            top values.

        ``inner_variance``
            For each gene and metacell, the variance of the gene in the metacell.

        ``inner_normalized_variance``
            For each gene and metacell, the normalized variance (variance over mean) of the
            gene in the metacell.

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse.

    **Computation Parameters**

    For each metacell:

    1. Downsample the cells so their total number of UMIs would be the
       ``downsample_cell_quantile`` (default: {downsample_cell_quantile}). That is, if
       ``downsample_cell_quantile`` is 0.1, then 90% of the cells would have more UMIs than
       the target number of samples and will be downsampled; 10% of the cells would have too
       few UMIs and will be left unchanged.

    2 Ignore all genes whose total data in the metacell (in the downsampled data) is below the
      ``min_gene_total`` (default: {min_gene_total}). This ensures we only correlate genes
      with sufficient data to be meaningful. We also ignore any genes which have exactly the
      same value in all cells of the metacell, regardless of their expression level.

    3 Compute the cross-correlation between all the remaining genes, and square it to obtain
      the metacell's per-gene-per-gene R^2 data. Collect for each gene the ``top_gene_rank``
      value, which is the gene's top R^2 for this metacell.

    4 Randomly shuffle the remaining gene expressions between the cells of each metacells
      using the ``random_seed``, re-compute the cross-correlation and square it into the
      metacell's per-gene-per-gene shuffled R^2 data. Collect for each gene the
      ``top_gene_rank`` value. Repeat this ``shuffles_count`` times and use the average as the
      gene's top shuffled R^2 for this metacell.

    5 The difference, for each gene, between the gene's top R^2, and the gene's (averaged) top
      shuffled R^2, is the gene's excess R^2 for this metacell.
    '''
    of, _ = ut.log_operation(LOG, adata, 'compute_excess_r2', of)
    assert shuffles_count > 0

    with ut.focus_on(ut.get_vo_data, adata, of, layout='row_major',
                     intermediate=intermediate) as data:
        data = ut.to_proper_matrix(data)

        metacell_of_cells = \
            ut.get_vector_parameter_data(LOG, adata, metacells,
                                         per='o', name='metacells')
        assert metacell_of_cells is not None

        metacells_count = np.max(metacell_of_cells) + 1
        assert metacells_count > 0

        LOG.debug('  mdata: %s', ut.get_name(mdata) or '<adata>')

        assert mdata.n_obs == metacells_count
        top_r2_per_gene_per_metacell = \
            np.full(mdata.shape, None, dtype='float')
        top_shuffled_r2_per_gene_per_metacell = \
            np.full(mdata.shape, None, dtype='float')
        excess_r2_per_gene_per_metacell = \
            np.full(mdata.shape, None, dtype='float')
        variance_per_gene_per_metacell = \
            np.full(mdata.shape, None, dtype='float')
        normalized_variance_per_gene_per_metacell = \
            np.full(mdata.shape, None, dtype='float')

        LOG.debug('  downsample_cell_quantile: %s',
                  downsample_cell_quantile)
        LOG.debug('  min_gene_total: %s', min_gene_total)
        LOG.debug('  top_gene_rank: %s', top_gene_rank)

        for metacell_index in range(metacells_count):
            _collect_metacell_excess(metacell_index, metacells_count,
                                     metacell_of_cells=metacell_of_cells,
                                     data=data,
                                     downsample_cell_quantile=downsample_cell_quantile,
                                     random_seed=random_seed,
                                     shuffles_count=shuffles_count,
                                     top_gene_rank=top_gene_rank,
                                     min_gene_total=min_gene_total,
                                     excess_r2_per_gene_per_metacell=excess_r2_per_gene_per_metacell,
                                     top_r2_per_gene_per_metacell=top_r2_per_gene_per_metacell,
                                     top_shuffled_r2_per_gene_per_metacell=top_shuffled_r2_per_gene_per_metacell,
                                     variance_per_gene_per_metacell=variance_per_gene_per_metacell,
                                     normalized_variance_per_gene_per_metacell=normalized_variance_per_gene_per_metacell)

    ut.set_vo_data(mdata, 'excess_r2', excess_r2_per_gene_per_metacell)
    top_r2_per_gene_per_metacell[top_r2_per_gene_per_metacell < -5] = None
    ut.set_vo_data(mdata, 'top_r2', top_r2_per_gene_per_metacell)
    top_shuffled_r2_per_gene_per_metacell[top_shuffled_r2_per_gene_per_metacell < -5] = None
    ut.set_vo_data(mdata, 'top_shuffled_r2',
                   top_shuffled_r2_per_gene_per_metacell)
    ut.set_vo_data(mdata, 'inner_variance',
                   variance_per_gene_per_metacell)
    ut.set_vo_data(mdata, 'inner_normalized_variance',
                   normalized_variance_per_gene_per_metacell)


def _log_r2(values: Optional[ut.DenseVector]) -> str:
    assert values is not None
    return '%s <- %s' % (ut.mask_description(~np.isnan(values)), np.nanmean(values))


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
    excess_r2_per_gene_per_metacell: ut.DenseMatrix,
    top_r2_per_gene_per_metacell: ut.DenseMatrix,
    top_shuffled_r2_per_gene_per_metacell: ut.DenseMatrix,
    variance_per_gene_per_metacell: ut.DenseMatrix,
    normalized_variance_per_gene_per_metacell: ut.DenseMatrix,
) -> None:
    LOG.debug('  - metacell: %s / %s', metacell_index, metacells_count)
    metacell_mask = metacell_of_cells == metacell_index
    assert ut.matrix_layout(data) == 'row_major'
    metacell_data = data[metacell_mask, :]
    cells_count = metacell_data.shape[0]
    LOG.debug('    cells: %s', cells_count)
    if cells_count < 2:
        return

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
                  ut.mask_description(correlated_genes_mask))
    if correlated_genes_count < 2:
        return

    correlated_gene_rank = max(correlated_genes_count - top_gene_rank - 1, 1)

    correlated_gene_indices = np.where(correlated_genes_mask)[0]
    correlated_data = downsampled_data_columns[:, correlated_gene_indices]
    assert ut.matrix_layout(correlated_data) == 'column_major'

    metacell_r2 = ut.corrcoef(correlated_data, per='column')
    metacell_r2 *= metacell_r2
    np.fill_diagonal(metacell_r2, 0)

    top_r2_per_correlated_gene = \
        ut.rank_per(metacell_r2, per=None, rank=correlated_gene_rank)

    top_r2_per_gene_per_metacell[metacell_index,
                                 correlated_gene_indices] = \
        top_r2_per_correlated_gene

    assert normalized_variance_per_gene_per_metacell is not None
    assert correlated_data.shape == (cells_count, correlated_genes_count)

    if ut.SparseMatrix.am(correlated_data):
        correlated_squared_data = correlated_data.multiply(correlated_data)
    else:
        correlated_squared_data = correlated_data * correlated_data

    correlated_total_squared_per_gene = \
        ut.to_dense_vector(ut.sum_per(correlated_squared_data,
                                      per='column'))

    correlated_total_per_gene = total_per_gene[correlated_gene_indices]
    correlated_variance_per_gene = \
        np.square(correlated_total_per_gene).astype(float)
    correlated_variance_per_gene /= -cells_count
    correlated_variance_per_gene += correlated_total_squared_per_gene
    correlated_variance_per_gene /= cells_count
    variance_per_gene_per_metacell[metacell_index, correlated_gene_indices] = \
        correlated_variance_per_gene

    correlated_mean_per_gene = correlated_total_per_gene / cells_count
    correlated_mean_per_gene[correlated_mean_per_gene <= 0] = 1
    correlated_normalized_variance_per_gene = \
        correlated_variance_per_gene / correlated_mean_per_gene
    normalized_variance_per_gene_per_metacell[metacell_index,
                                              correlated_gene_indices] = \
        correlated_normalized_variance_per_gene

    assert ut.matrix_layout(correlated_data) == 'column_major'
    if ut.SparseMatrix.am(correlated_data):
        shuffled_data = correlated_data.copy()
    else:
        shuffled_data = np.copy(correlated_data)
    assert ut.matrix_layout(shuffled_data) == 'column_major'

    top_shuffled_r2_per_correlated_gene = \
        np.zeros(top_r2_per_correlated_gene.size)

    with ut.timed_step('.shuffle'):
        LOG.debug('    random_seed: %s', random_seed)
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

    top_shuffled_r2_per_gene_per_metacell[metacell_index,
                                          correlated_gene_indices] = \
        top_shuffled_r2_per_correlated_gene

    excess_r2_per_correlated_gene = \
        top_r2_per_correlated_gene - top_shuffled_r2_per_correlated_gene

    excess_r2_per_gene_per_metacell[metacell_index, correlated_gene_indices] = \
        excess_r2_per_correlated_gene
