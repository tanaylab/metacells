'''
Excess R^2
----------
'''

from typing import Any, List, Optional, Set, TypeVar, Union

import numpy as np
from anndata import AnnData

import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    'compute_type_compatible_sizes',
    'compute_excess_r2',
]


T = TypeVar('T')


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_type_compatible_sizes(
    adatas: List[AnnData],
    *,
    size: str = 'grouped',
    kind: str = 'type',
) -> None:
    '''
    Given multiple annotated data of groups, compute a "compatible" sizes for each one to
    allow for consistent excess R^2 and inner normalized variance comparison.

    Since excess R^2 and inner normalized variance quality measures are sensitive to the group
    (metacell) sizes, it is useful to artifically shrink the groups so the sizes will be similar
    between the compared data sets. Assuming each group (metacell) has a type annotation, for each
    such type, we give each one a "compatible" size (less than or equal to its actual size) so that
    using this reduced size will give us comparable measures between all the data sets.

    The "compatible" sizes are chosen such that the density distributions of the sizes in all data
    sets would be as similar to each other as possible.

    **Input**

    Several annotated ``adatas`` where each observation is a group. Should contain per-observation
    ``size`` annotation (default: {size}) and ``kind`` annotation (default: {kind}).

    **Returns**

    Sets the following in each ``adata``:

    Per-Observation (group) Annotations:

        ``compatible_size``
            The number of grouped cells in the group to use for computing excess R^2 and inner
            normalized variance.

    **Computation**

    1. For each type, sort the groups (metacells) in increasing number of grouped observations (cells).

    2. Consider the maximal quantile (rank) of the next smallest group (metacell) in each data set.

    3. Compute the minimal number of grouped observations in all the metacells whose quantile is up
       to this maximal quantile.

    4. Use this as the "compatible" size for all these metacells, and remove them from consideration.

    5. Loop until all metacells are assigned a "compatible" size.
    '''
    if len(adatas) == 1:
        ut.set_o_data(adatas[0], 'compatible_size',
                      ut.get_o_numpy(adatas[0], size,
                                     formatter=ut.sizes_description))
        return

    metacell_sizes_of_data = \
        [ut.get_o_numpy(adata, size, formatter=ut.sizes_description)
         for adata in adatas]
    metacell_types_of_data = [ut.get_o_numpy(adata, kind) for adata in adatas]

    unique_types: Set[Any] = set()
    for metacell_types in metacell_types_of_data:
        unique_types.update(metacell_types)

    compatible_size_of_data = [np.full(adata.n_obs, -1) for adata in adatas]

    for metacell_type in unique_types:
        sorted_metacell_indices_of_data = \
            [np.argsort(metacell_sizes)[metacell_types == metacell_type]
             for metacell_sizes, metacell_types
             in zip(metacell_sizes_of_data, metacell_types_of_data)]

        metacells_count_of_data = \
            [len(sorted_metacell_indices)
             for sorted_metacell_indices
             in sorted_metacell_indices_of_data]

        def _for_each(value_of_data: List[T]) -> List[T]:
            return [value
                    for metacells_count, value
                    in zip(metacells_count_of_data, value_of_data)
                    if metacells_count > 0]

        metacells_count_of_each = _for_each(metacells_count_of_data)
        if len(metacells_count_of_each) == 0:
            continue

        sorted_metacell_indices_of_each = \
            _for_each(sorted_metacell_indices_of_data)
        metacell_sizes_of_each = _for_each(metacell_sizes_of_data)
        compatible_size_of_each = _for_each(compatible_size_of_data)

        if len(metacells_count_of_each) == 1:
            compatible_size_of_each[0][sorted_metacell_indices_of_each[0]] = \
                metacell_sizes_of_each[0][sorted_metacell_indices_of_each[0]]

        metacell_quantile_of_each = \
            [(np.arange(len(sorted_metacell_indices)) + 1) / len(sorted_metacell_indices)
             for sorted_metacell_indices
             in sorted_metacell_indices_of_each]

        next_position_of_each = np.full(len(metacell_quantile_of_each), 0)

        while True:
            next_quantile_of_each = \
                [metacell_quantile[next_position]
                 for metacell_quantile, next_position
                 in zip(metacell_quantile_of_each, next_position_of_each)]

            next_quantile = max(next_quantile_of_each)

            last_position_of_each = next_position_of_each

            next_position_of_each[:] = \
                [np.sum(metacell_quantile <= next_quantile)
                 for metacell_quantile
                 in metacell_quantile_of_each]

            positions_of_each = \
                [range(last_position, next_position)
                 for last_position, next_position
                 in zip(last_position_of_each, next_position_of_each)]

            sizes_of_each = \
                [metacell_sizes[sorted_metacell_indices[positions]]
                 for metacell_sizes, sorted_metacell_indices, positions
                 in zip(metacell_sizes_of_each, sorted_metacell_indices_of_each, positions_of_each)]

            min_size_of_each = \
                [np.min(sizes)
                 for sizes, positions
                 in zip(sizes_of_each, positions_of_each)]

            min_size = min(min_size_of_each)

            for sorted_metacell_indices, positions, compatible_size \
                    in zip(sorted_metacell_indices_of_each, positions_of_each, compatible_size_of_each):
                compatible_size[sorted_metacell_indices[positions]] = min_size

            is_done_of_each = [next_position == metacells_count
                               for next_position, metacells_count
                               in zip(next_position_of_each, metacells_count_of_each)]

            if all(is_done_of_each):
                break

            assert not any(is_done_of_each)

    for adata, compatible_size in zip(adatas, compatible_size_of_data):
        assert np.min(compatible_size) > 0
        ut.set_o_data(adata, 'compatible_size', compatible_size)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_excess_r2(
    adata: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    *,
    metacells: Union[str, ut.Vector] = 'metacell',
    compatible_size: Optional[str] = 'compatible_size',
    downsample_cell_quantile: float = pr.excess_downsample_cell_quantile,
    min_gene_total: int = pr.excess_min_gene_total,
    top_gene_rank: int = pr.excess_top_gene_rank,
    shuffles_count: int = pr.excess_shuffles_count,
    random_seed: int = pr.random_seed,
    mdata: AnnData,
) -> None:
    '''
    Compute the excess gene-gene coefficient of determination (R^2) for the metacells based of
    ``what`` data.

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

    Annotated ``adata``, where the observations are cells and the variables are genes.

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

    **Computation Parameters**

    For each metacell:

    1. If ``compatible_size`` (default: {compatible_size}) is specified, it should be an
       integer per-observation annotation of the metacells, whose value is at most the
       number of grouped cells in the metacell. Pick a random subset of the cells of
       this size. If ``compatible_size`` is ``None``, use all the cells of the metacell.

    2. Downsample the cells so their total number of UMIs would be the
       ``downsample_cell_quantile`` (default: {downsample_cell_quantile}). That is, if
       ``downsample_cell_quantile`` is 0.1, then 90% of the cells would have more UMIs than
       the target number of samples and will be downsampled; 10% of the cells would have too
       few UMIs and will be left unchanged.

    3 Ignore all genes whose total data in the metacell (in the downsampled data) is below the
      ``min_gene_total`` (default: {min_gene_total}). This ensures we only correlate genes
      with sufficient data to be meaningful. We also ignore any genes which have exactly the
      same value in all cells of the metacell, regardless of their expression level.

    4 Compute the cross-correlation between all the remaining genes, and square it to obtain
      the metacell's per-gene-per-gene R^2 data. Collect for each gene the ``top_gene_rank``
      value, which is the gene's top R^2 for this metacell.

    5 Randomly shuffle the remaining gene expressions between the cells of each metacells
      using the ``random_seed``, re-compute the cross-correlation and square it into the
      metacell's per-gene-per-gene shuffled R^2 data. Collect for each gene the
      ``top_gene_rank`` value. Repeat this ``shuffles_count`` times and use the average as the
      gene's top shuffled R^2 for this metacell.

    6 The difference, for each gene, between the gene's top R^2, and the gene's (averaged) top
      shuffled R^2, is the gene's excess R^2 for this metacell.
    '''
    assert shuffles_count > 0

    data = ut.get_vo_proper(adata, what, layout='row_major')

    metacell_of_cells = \
        ut.get_o_numpy(adata, metacells, formatter=ut.groups_description)

    metacells_count = np.max(metacell_of_cells) + 1
    assert metacells_count > 0

    if compatible_size is not None:
        compatible_size_of_metacells: Optional[ut.NumpyVector] = \
            ut.get_o_numpy(mdata, compatible_size,
                           formatter=ut.sizes_description)
    else:
        compatible_size_of_metacells = None

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

    for metacell_index in range(metacells_count):
        with ut.log_step('- metacell', metacell_index,
                         formatter=lambda metacell_index:
                         ut.ratio_description(metacell_index,
                                              metacells_count)):
            if compatible_size_of_metacells is not None:
                compatible_size_of_metacell = \
                    compatible_size_of_metacells[metacell_index]
            else:
                compatible_size_of_metacell = None

            _collect_metacell_excess(metacell_index,
                                     compatible_size=compatible_size_of_metacell,
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


def _log_r2(values: Optional[ut.NumpyVector]) -> str:
    assert values is not None
    return '%s <- %s' % (ut.mask_description(~np.isnan(values)), np.nanmean(values))


def _collect_metacell_excess(  # pylint: disable=too-many-statements,too-many-branches
    metacell_index: int,
    *,
    compatible_size: Optional[int],
    metacell_of_cells: ut.NumpyVector,
    data: ut.ProperMatrix,
    downsample_cell_quantile: float,
    random_seed: int,
    shuffles_count: int,
    min_gene_total: float,
    top_gene_rank: int,
    excess_r2_per_gene_per_metacell: ut.NumpyMatrix,
    top_r2_per_gene_per_metacell: ut.NumpyMatrix,
    top_shuffled_r2_per_gene_per_metacell: ut.NumpyMatrix,
    variance_per_gene_per_metacell: ut.NumpyMatrix,
    normalized_variance_per_gene_per_metacell: ut.NumpyMatrix,
) -> None:
    metacell_mask = metacell_of_cells == metacell_index
    metacell_indices = np.where(metacell_mask)[0]

    if compatible_size is None:
        if ut.logging_calc():
            ut.log_calc(f'cells: {len(metacell_indices)}')
    else:
        assert 0 < compatible_size <= len(metacell_indices)
        if compatible_size < len(metacell_indices):
            np.random.seed(random_seed)
            if ut.logging_calc():
                ut.log_calc(  #
                    f'cells: {ut.ratio_description(compatible_size, len(metacell_indices))}')
            metacell_indices = np.random.choice(metacell_indices,
                                                size=compatible_size,
                                                replace=False)
            assert len(metacell_indices) == compatible_size

    cells_count = len(metacell_indices)
    if cells_count < 2:
        return

    assert ut.matrix_layout(data) == 'row_major'
    metacell_data = data[metacell_indices, :]

    total_per_cell = ut.sum_per(metacell_data, per='row')
    samples = round(np.quantile(total_per_cell, downsample_cell_quantile))
    if ut.logging_calc():
        ut.log_calc(f'samples: {samples}')
    downsampled_data_rows = \
        ut.downsample_matrix(metacell_data, per='row', samples=samples)

    downsampled_data_columns = \
        ut.to_layout(downsampled_data_rows, layout='column_major')
    total_per_gene = ut.sum_per(downsampled_data_columns, per='column')

    min_per_gene = ut.min_per(downsampled_data_columns, per='column')
    max_per_gene = ut.max_per(downsampled_data_columns, per='column')
    correlated_genes_mask = \
        (total_per_gene >= min_gene_total) & (min_per_gene < max_per_gene)
    correlated_genes_count = np.sum(correlated_genes_mask)
    if ut.logging_calc():
        ut.log_calc(  #
            f'correlate genes: {ut.mask_description(correlated_genes_mask)}')
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

    sparse = ut.maybe_sparse_matrix(correlated_data)
    if sparse is not None:
        correlated_squared_data = sparse.multiply(sparse)
    else:
        correlated_squared_data = correlated_data * correlated_data

    correlated_total_squared_per_gene = \
        ut.sum_per(correlated_squared_data, per='column')

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
    sparse = ut.maybe_sparse_matrix(correlated_data)
    if sparse is not None:
        shuffled_data = sparse.copy()
    else:
        shuffled_data = np.copy(correlated_data)
    assert ut.matrix_layout(shuffled_data) == 'column_major'

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

    top_shuffled_r2_per_gene_per_metacell[metacell_index,
                                          correlated_gene_indices] = \
        top_shuffled_r2_per_correlated_gene

    excess_r2_per_correlated_gene = \
        top_r2_per_correlated_gene - top_shuffled_r2_per_correlated_gene

    excess_r2_per_gene_per_metacell[metacell_index, correlated_gene_indices] = \
        excess_r2_per_correlated_gene
