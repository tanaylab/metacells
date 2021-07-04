'''
Quality
-------
'''

from typing import Any, List, Optional, Set, TypeVar, Union

import numpy as np
from anndata import AnnData

import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    'compute_type_compatible_sizes',
    'compute_inner_normalized_variance',
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
    Given multiple annotated data of groups, compute a "compatible" size for each one to allow for
    consistent inner normalized variance comparison.

    Since the inner normalized variance quality measure is sensitive to the group (metacell) sizes,
    it is useful to artificially shrink the groups so the sizes will be similar between the compared
    data sets. Assuming each group (metacell) has a type annotation, for each such type, we give
    each one a "compatible" size (less than or equal to its actual size) so that using this reduced
    size will give us comparable measures between all the data sets.

    The "compatible" sizes are chosen such that the density distributions of the sizes in all data
    sets would be as similar to each other as possible.

    .. note::

        This is only effective if the groups are "similar" in size. Using this to compare very coarse
        grouping (few thousands of cells) with fine-grained ones (few dozens of cells) will still
        result in very different results.

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

    4. Use this as the "compatible" size for all these groups, and remove them from consideration.

    5. Loop until all groups are assigned a "compatible" size.
    '''
    assert len(adatas) > 0
    if len(adatas) == 1:
        ut.set_o_data(adatas[0], 'compatible_size',
                      ut.get_o_numpy(adatas[0], size,
                                     formatter=ut.sizes_description))
        return

    group_sizes_of_data = \
        [ut.get_o_numpy(adata, size, formatter=ut.sizes_description)
         for adata in adatas]
    group_types_of_data = [ut.get_o_numpy(adata, kind) for adata in adatas]

    unique_types: Set[Any] = set()
    for group_types in group_types_of_data:
        unique_types.update(group_types)

    compatible_size_of_data = [np.full(adata.n_obs, -1) for adata in adatas]

    for type_index, group_type in enumerate(sorted(unique_types)):
        with ut.log_step(f'- {group_type}',
                         ut.progress_description(len(unique_types),
                                                 type_index, 'type')):
            sorted_group_indices_of_data = \
                [np.argsort(group_sizes)[group_types == group_type]
                 for group_sizes, group_types
                 in zip(group_sizes_of_data, group_types_of_data)]

            groups_count_of_data = \
                [len(sorted_group_indices)
                 for sorted_group_indices
                 in sorted_group_indices_of_data]

            ut.log_calc('group_counts', groups_count_of_data)

            def _for_each(value_of_data: List[T]) -> List[T]:
                return [value
                        for groups_count, value
                        in zip(groups_count_of_data, value_of_data)
                        if groups_count > 0]

            groups_count_of_each = _for_each(groups_count_of_data)

            if len(groups_count_of_each) == 0:
                continue

            sorted_group_indices_of_each = \
                _for_each(sorted_group_indices_of_data)
            group_sizes_of_each = _for_each(group_sizes_of_data)
            compatible_size_of_each = _for_each(compatible_size_of_data)

            if len(groups_count_of_each) == 1:
                compatible_size_of_each[0][sorted_group_indices_of_each[0]] = \
                    group_sizes_of_each[0][sorted_group_indices_of_each[0]]

            group_quantile_of_each = \
                [(np.arange(len(sorted_group_indices)) + 1) / len(sorted_group_indices)
                 for sorted_group_indices
                 in sorted_group_indices_of_each]

            next_position_of_each = np.full(len(group_quantile_of_each), 0)

            while True:
                next_quantile_of_each = \
                    [group_quantile[next_position]
                     for group_quantile, next_position
                     in zip(group_quantile_of_each, next_position_of_each)]
                next_quantile = max(next_quantile_of_each)

                last_position_of_each = next_position_of_each.copy()
                next_position_of_each[:] = \
                    [np.sum(group_quantile <= next_quantile)
                     for group_quantile
                     in group_quantile_of_each]

                positions_of_each = \
                    [range(last_position, next_position)
                     for last_position, next_position
                     in zip(last_position_of_each, next_position_of_each)]

                sizes_of_each = \
                    [group_sizes[sorted_group_indices[positions]]
                     for group_sizes, sorted_group_indices, positions
                     in zip(group_sizes_of_each, sorted_group_indices_of_each, positions_of_each)]

                min_size_of_each = \
                    [np.min(sizes)
                     for sizes, positions
                     in zip(sizes_of_each, positions_of_each)]
                min_size = min(min_size_of_each)

                for sorted_group_indices, positions, compatible_size \
                        in zip(sorted_group_indices_of_each, positions_of_each, compatible_size_of_each):
                    compatible_size[sorted_group_indices[positions]] = min_size

                is_done_of_each = [next_position == groups_count
                                   for next_position, groups_count
                                   in zip(next_position_of_each, groups_count_of_each)]
                if all(is_done_of_each):
                    break

                assert not any(is_done_of_each)

    for adata, compatible_size in zip(adatas, compatible_size_of_data):
        assert np.min(compatible_size) > 0
        ut.set_o_data(adata, 'compatible_size', compatible_size)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_inner_normalized_variance(
    what: Union[str, ut.Matrix] = '__x__',
    *,
    compatible_size: Optional[str] = None,
    downsample_min_samples: int = pr.downsample_min_samples,
    downsample_min_cell_quantile: float = pr.downsample_min_cell_quantile,
    downsample_max_cell_quantile: float = pr.downsample_max_cell_quantile,
    min_gene_total: int = pr.quality_min_gene_total,
    adata: AnnData,
    gdata: AnnData,
    group: Union[str, ut.Vector] = 'metacell',
    random_seed: int = pr.random_seed,
) -> None:
    '''
    Compute the inner normalized variance (variance / mean) for each gene in each group.

    This is also known as the "index of dispersion" and can serve as a quality measure for
    the groups. An ideal group would contain only cells with "the same" biological state
    and all remaining inner variance would be due to technical sampling noise.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    In addition, ``gdata`` is assumed to have one observation for each group, and use the same
    genes as ``adata``.

    **Returns**

    Sets the following in ``gdata``:

    Per-Variable Per-Observation (Gene-Cell) Annotations

        ``inner_variance``
            For each gene and group, the variance of the gene in the group.

        ``inner_normalized_variance``
            For each gene and group, the normalized variance (variance over mean) of the
            gene in the group.

    **Computation Parameters**

    For each group (metacell):

    1. If ``compatible_size`` (default: {compatible_size}) is specified, it should be an
       integer per-observation annotation of the groups, whose value is at most the
       number of grouped cells in the group. Pick a random subset of the cells of
       this size. If ``compatible_size`` is ``None``, use all the cells of the group.

    2. Invoke :py:func:`metacells.tools.downsample.downsample_cells` to downsample the surviving
       cells to the same total number of UMIs, using the ``downsample_min_samples`` (default:
       {downsample_min_samples}), ``downsample_min_cell_quantile`` (default:
       {downsample_min_cell_quantile}), ``downsample_max_cell_quantile`` (default:
       {downsample_max_cell_quantile}) and the ``random_seed`` (default: {random_seed}).

    3. Compute the normalized variance of each gene based on the downsampled data. Set the
       result to ``nan`` for genes with less than ``min_gene_total`` (default: {min_gene_total}).
    '''
    cells_data = ut.get_vo_proper(adata, what, layout='row_major')

    if compatible_size is not None:
        compatible_size_of_groups: Optional[ut.NumpyVector] = \
            ut.get_o_numpy(gdata, compatible_size,
                           formatter=ut.sizes_description)
    else:
        compatible_size_of_groups = None

    group_of_cells = \
        ut.get_o_numpy(adata, group, formatter=ut.groups_description)

    groups_count = np.max(group_of_cells) + 1
    assert groups_count > 0

    assert gdata.n_obs == groups_count
    variance_per_gene_per_group = np.full(gdata.shape, None, dtype='float32')
    normalized_variance_per_gene_per_group = \
        np.full(gdata.shape, None, dtype='float32')

    for group_index in range(groups_count):
        with ut.log_step('- group', group_index,
                         formatter=lambda group_index:
                         ut.progress_description(groups_count,
                                                 group_index, 'group')):
            if compatible_size_of_groups is not None:
                compatible_size_of_group = \
                    compatible_size_of_groups[group_index]
            else:
                compatible_size_of_group = None

            _collect_group_data(group_index,
                                group_of_cells=group_of_cells,
                                cells_data=cells_data,
                                compatible_size=compatible_size_of_group,
                                downsample_min_samples=downsample_min_samples,
                                downsample_min_cell_quantile=downsample_min_cell_quantile,
                                downsample_max_cell_quantile=downsample_max_cell_quantile,
                                min_gene_total=min_gene_total,
                                random_seed=random_seed,
                                variance_per_gene_per_group=variance_per_gene_per_group,
                                normalized_variance_per_gene_per_group=normalized_variance_per_gene_per_group)

    ut.set_vo_data(gdata, 'inner_variance',
                   variance_per_gene_per_group)
    ut.set_vo_data(gdata, 'inner_normalized_variance',
                   normalized_variance_per_gene_per_group)


def _collect_group_data(
    group_index: int,
    *,
    group_of_cells: ut.NumpyVector,
    cells_data: ut.ProperMatrix,
    compatible_size: Optional[int],
    downsample_min_samples: int,
    downsample_min_cell_quantile: float,
    downsample_max_cell_quantile: float,
    min_gene_total: int,
    random_seed: int,
    variance_per_gene_per_group: ut.NumpyMatrix,
    normalized_variance_per_gene_per_group: ut.NumpyMatrix,
) -> None:
    cell_indices = np.where(group_of_cells == group_index)[0]
    cells_count = len(cell_indices)
    if cells_count < 2:
        return

    if compatible_size is None:
        ut.log_calc('  cells', cells_count)
    else:
        assert 0 < compatible_size <= cells_count
        if compatible_size < cells_count:
            np.random.seed(random_seed)
            if ut.logging_calc():
                ut.log_calc('  cells: '
                            + ut.ratio_description(len(cell_indices), 'cell',
                                                   compatible_size, 'compatible'))
            cell_indices = np.random.choice(cell_indices,
                                            size=compatible_size,
                                            replace=False)
            assert len(cell_indices) == compatible_size

    assert ut.matrix_layout(cells_data) == 'row_major'
    group_data = cells_data[cell_indices, :]

    total_per_cell = ut.sum_per(group_data, per='row')
    samples = int(round(min(max(downsample_min_samples,
                                np.quantile(total_per_cell, downsample_min_cell_quantile)),
                            np.quantile(total_per_cell, downsample_max_cell_quantile))))
    if ut.logging_calc():
        ut.log_calc(f'  samples: {samples}')
    downsampled_data = \
        ut.downsample_matrix(group_data, per='row',
                             samples=samples, random_seed=random_seed)

    downsampled_data = ut.to_layout(downsampled_data, layout='column_major')
    total_per_gene = ut.sum_per(downsampled_data, per='column')
    too_small_genes = total_per_gene < min_gene_total
    if ut.logging_calc():
        included_genes_count = len(too_small_genes) - np.sum(too_small_genes)
        ut.log_calc(f'  included genes: {included_genes_count}')

    variance_per_gene = ut.variance_per(downsampled_data, per='column')
    normalized_variance_per_gene = \
        ut.normalized_variance_per(downsampled_data, per='column')

    variance_per_gene[too_small_genes] = None
    normalized_variance_per_gene[too_small_genes] = None

    variance_per_gene_per_group[group_index, :] = variance_per_gene
    normalized_variance_per_gene_per_group[group_index, :] = \
        normalized_variance_per_gene
