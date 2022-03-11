"""
Quality
-------
"""

from typing import Any
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore
from scipy import sparse  # type: ignore

import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    "compute_type_compatible_sizes",
    "compute_inner_normalized_variance",
    "compute_inner_fold_factors",
    "compute_significant_projected_fold_factors",
    "compute_similar_query_metacells",
    "compute_outliers_matches",
    "compute_deviant_fold_factors",
    "compute_outliers_fold_factors",
    "compute_type_genes_normalized_variances",
]


T = TypeVar("T")


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_type_compatible_sizes(
    adatas: List[AnnData],
    *,
    size: str = "grouped",
    kind: str = "type",
) -> None:
    """
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
    """
    assert len(adatas) > 0
    if len(adatas) == 1:
        ut.set_o_data(adatas[0], "compatible_size", ut.get_o_numpy(adatas[0], size, formatter=ut.sizes_description))
        return

    group_sizes_of_data = [ut.get_o_numpy(adata, size, formatter=ut.sizes_description) for adata in adatas]
    group_types_of_data = [ut.get_o_numpy(adata, kind) for adata in adatas]

    unique_types: Set[Any] = set()
    for group_types in group_types_of_data:
        unique_types.update(group_types)

    compatible_size_of_data = [np.full(adata.n_obs, -1) for adata in adatas]

    groups_count_of_data: List[int] = []
    for type_index, group_type in enumerate(sorted(unique_types)):
        with ut.log_step(f"- {group_type}", ut.progress_description(len(unique_types), type_index, "type")):
            sorted_group_indices_of_data = [
                np.argsort(group_sizes)[group_types == group_type]
                for group_sizes, group_types in zip(group_sizes_of_data, group_types_of_data)
            ]

            groups_count_of_data = [len(sorted_group_indices) for sorted_group_indices in sorted_group_indices_of_data]

            ut.log_calc("group_counts", groups_count_of_data)

            def _for_each(value_of_data: List[T]) -> List[T]:
                return [value for groups_count, value in zip(groups_count_of_data, value_of_data) if groups_count > 0]

            groups_count_of_each = _for_each(groups_count_of_data)

            if len(groups_count_of_each) == 0:
                continue

            sorted_group_indices_of_each = _for_each(sorted_group_indices_of_data)
            group_sizes_of_each = _for_each(group_sizes_of_data)
            compatible_size_of_each = _for_each(compatible_size_of_data)

            if len(groups_count_of_each) == 1:
                compatible_size_of_each[0][sorted_group_indices_of_each[0]] = group_sizes_of_each[0][
                    sorted_group_indices_of_each[0]
                ]

            group_quantile_of_each = [
                (np.arange(len(sorted_group_indices)) + 1) / len(sorted_group_indices)
                for sorted_group_indices in sorted_group_indices_of_each
            ]

            next_position_of_each = np.full(len(group_quantile_of_each), 0)

            while True:
                next_quantile_of_each = [
                    group_quantile[next_position]
                    for group_quantile, next_position in zip(group_quantile_of_each, next_position_of_each)
                ]
                next_quantile = max(next_quantile_of_each)

                last_position_of_each = next_position_of_each.copy()
                next_position_of_each[:] = [
                    np.sum(group_quantile <= next_quantile) for group_quantile in group_quantile_of_each
                ]

                positions_of_each = [
                    range(last_position, next_position)
                    for last_position, next_position in zip(last_position_of_each, next_position_of_each)
                ]

                sizes_of_each = [
                    group_sizes[sorted_group_indices[positions]]
                    for group_sizes, sorted_group_indices, positions in zip(
                        group_sizes_of_each, sorted_group_indices_of_each, positions_of_each
                    )
                ]

                min_size_of_each = [np.min(sizes) for sizes, positions in zip(sizes_of_each, positions_of_each)]
                min_size = min(min_size_of_each)

                for sorted_group_indices, positions, compatible_size in zip(
                    sorted_group_indices_of_each, positions_of_each, compatible_size_of_each
                ):
                    compatible_size[sorted_group_indices[positions]] = min_size

                is_done_of_each = [
                    next_position == groups_count
                    for next_position, groups_count in zip(next_position_of_each, groups_count_of_each)
                ]
                if all(is_done_of_each):
                    break

                assert not any(is_done_of_each)

    for adata, compatible_size in zip(adatas, compatible_size_of_data):
        assert np.min(compatible_size) > 0
        ut.set_o_data(adata, "compatible_size", compatible_size)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_inner_normalized_variance(
    what: Union[str, ut.Matrix] = "__x__",
    *,
    compatible_size: Optional[str] = None,
    downsample_min_samples: int = pr.downsample_min_samples,
    downsample_min_cell_quantile: float = pr.downsample_min_cell_quantile,
    downsample_max_cell_quantile: float = pr.downsample_max_cell_quantile,
    min_gene_total: int = pr.quality_min_gene_total,
    adata: AnnData,
    gdata: AnnData,
    group: Union[str, ut.Vector] = "metacell",
    random_seed: int = pr.random_seed,
) -> None:
    """
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
    """
    cells_data = ut.get_vo_proper(adata, what, layout="row_major")

    if compatible_size is not None:
        compatible_size_of_groups: Optional[ut.NumpyVector] = ut.get_o_numpy(
            gdata, compatible_size, formatter=ut.sizes_description
        )
    else:
        compatible_size_of_groups = None

    group_of_cells = ut.get_o_numpy(adata, group, formatter=ut.groups_description)

    groups_count = np.max(group_of_cells) + 1
    assert groups_count > 0

    assert gdata.n_obs == groups_count
    variance_per_gene_per_group = np.full(gdata.shape, None, dtype="float32")
    normalized_variance_per_gene_per_group = np.full(gdata.shape, None, dtype="float32")

    for group_index in range(groups_count):
        with ut.log_step(
            "- group",
            group_index,
            formatter=lambda group_index: ut.progress_description(groups_count, group_index, "group"),
        ):
            if compatible_size_of_groups is not None:
                compatible_size_of_group = compatible_size_of_groups[group_index]
            else:
                compatible_size_of_group = None

            _collect_group_data(
                group_index,
                group_of_cells=group_of_cells,
                cells_data=cells_data,
                compatible_size=compatible_size_of_group,
                downsample_min_samples=downsample_min_samples,
                downsample_min_cell_quantile=downsample_min_cell_quantile,
                downsample_max_cell_quantile=downsample_max_cell_quantile,
                min_gene_total=min_gene_total,
                random_seed=random_seed,
                variance_per_gene_per_group=variance_per_gene_per_group,
                normalized_variance_per_gene_per_group=normalized_variance_per_gene_per_group,
            )

    ut.set_vo_data(gdata, "inner_variance", variance_per_gene_per_group)
    ut.set_vo_data(gdata, "inner_normalized_variance", normalized_variance_per_gene_per_group)


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
        ut.log_calc("  cells", cells_count)
    else:
        assert 0 < compatible_size <= cells_count
        if compatible_size < cells_count:
            np.random.seed(random_seed)
            if ut.logging_calc():
                ut.log_calc(
                    "  cells: " + ut.ratio_description(len(cell_indices), "cell", compatible_size, "compatible")
                )
            cell_indices = np.random.choice(cell_indices, size=compatible_size, replace=False)
            assert len(cell_indices) == compatible_size

    assert ut.is_layout(cells_data, "row_major")
    group_data = cells_data[cell_indices, :]

    total_per_cell = ut.sum_per(group_data, per="row")
    samples = int(
        round(
            min(
                max(downsample_min_samples, np.quantile(total_per_cell, downsample_min_cell_quantile)),
                np.quantile(total_per_cell, downsample_max_cell_quantile),
            )
        )
    )
    if ut.logging_calc():
        ut.log_calc(f"  samples: {samples}")
    downsampled_data = ut.downsample_matrix(group_data, per="row", samples=samples, random_seed=random_seed)

    downsampled_data = ut.to_layout(downsampled_data, layout="column_major")
    total_per_gene = ut.sum_per(downsampled_data, per="column")
    too_small_genes = total_per_gene < min_gene_total
    if ut.logging_calc():
        included_genes_count = len(too_small_genes) - np.sum(too_small_genes)
        ut.log_calc(f"  included genes: {included_genes_count}")

    variance_per_gene = ut.variance_per(downsampled_data, per="column")
    normalized_variance_per_gene = ut.normalized_variance_per(downsampled_data, per="column")

    variance_per_gene[too_small_genes] = None
    normalized_variance_per_gene[too_small_genes] = None

    variance_per_gene_per_group[group_index, :] = variance_per_gene
    normalized_variance_per_gene_per_group[group_index, :] = normalized_variance_per_gene


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_inner_fold_factors(
    what: Union[str, ut.Matrix] = "__x__",
    *,
    adata: AnnData,
    gdata: AnnData,
    group: Union[str, ut.Vector] = "metacell",
    min_gene_inner_fold_factor: float = pr.min_gene_inner_fold_factor,
    min_entry_inner_fold_factor: float = pr.min_entry_inner_fold_factor,
    inner_abs_folds: bool = pr.inner_abs_folds,
) -> None:
    """
    Compute the inner fold factors of genes within in each metacell.

    This computes, for each cell of the metacell, the same fold factors that are used to detect deviant cells (see
    :py:func:`metacells.tools.deviants.find_deviant_cells`), and keeps the maximal fold for each gene in the metacell.
    The result per-metacell-per-gene matrix is then made sparse by discarding too-low values (setting them to zero).
    Ideally, this matrix should be "very" sparse. If it contains "too many" non-zero values, this indicates the
    metacells contains "too much" variability. This may be due to actual biology (e.g. immune cells or olfactory nerves
    which are all similar except for each one expressing one different gene), due to batch effects (similar cells in
    distinct batches differing in some genes due to technical issues), due to low data quality (the overall noise level
    is so high that this is simply the best the algorithm can do), or worse - a combination of the above.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where ``what`` is a
    per-variable-per-observation matrix or the name of a per-variable-per-observation annotation containing such a
    matrix.

    In addition, ``gdata`` is assumed to have one observation for each group, and use the same
    genes as ``adata``.

    **Returns**

    Sets the following in ``gdata``:

    Per-Variable Per-Observation (Gene-Cell) Annotations

        ``inner_fold``
            For each gene and group, the maximal fold factor of this gene in any cell contained in the group (unless the
            value is too low to be of interest, in which case it will be zero).

    **Computation Parameters**

    1. For each group (metacell), for each gene, compute the gene's maximal (in all the cells of the group) fold factor
       log2((actual UMIs + 1) / (expected UMIs + 1)), similarly to
       :py:func:`metacells.tools.deviants.find_deviant_cells`.

    2. If the maximal fold factor for a gene (across all metacells) is below ``min_gene_inner_fold_factor`` (default:
       {min_gene_inner_fold_factor}), then set all the gene's fold factors to zero (too low to be of interest). If
       ``inner_abs_folds`` (default: {inner_abs_folds}), consider the absolute fold factors.

    3. Otherwise, for any metacell whose fold factor for the gene is less than ``min_entry_inner_fold_factor`` (default:
       {min_entry_inner_fold_factor}), set the fold factor to zero (too low to be of interest).
    """
    assert 0 <= min_entry_inner_fold_factor <= min_gene_inner_fold_factor

    cells_data = ut.get_vo_proper(adata, what, layout="row_major")
    metacells_data = ut.get_vo_proper(gdata, what, layout="row_major")
    group_of_cells = ut.get_o_numpy(adata, group, formatter=ut.groups_description)
    total_umis_per_cell = ut.sum_per(cells_data, per="row")
    total_umis_per_metacell = ut.sum_per(metacells_data, per="row")

    @ut.timed_call("compute_metacell_inner_folds")
    def _compute_single_metacell_inner_folds(metacell_index: int) -> ut.NumpyVector:
        return _compute_metacell_inner_folds(
            metacell_index=metacell_index,
            cells_data=cells_data,
            metacells_data=metacells_data,
            group_of_cells=group_of_cells,
            total_umis_per_cell=total_umis_per_cell,
            total_umis_per_metacell=total_umis_per_metacell,
        )

    results = list(ut.parallel_map(_compute_single_metacell_inner_folds, gdata.n_obs))
    dense_inner_folds_by_row = np.array(results)
    dense_inner_folds_by_column = ut.to_layout(dense_inner_folds_by_row, "column_major")
    sparse_inner_folds = ut.sparsify_matrix(
        dense_inner_folds_by_column,
        min_column_max_value=min_gene_inner_fold_factor,
        min_entry_value=min_entry_inner_fold_factor,
        abs_values=inner_abs_folds,
    )
    ut.set_vo_data(gdata, "inner_fold", sparse_inner_folds)


@ut.logged()
def _compute_metacell_inner_folds(
    *,
    metacell_index: int,
    cells_data: ut.Matrix,
    metacells_data: ut.Matrix,
    group_of_cells: ut.NumpyVector,
    total_umis_per_cell: ut.NumpyVector,
    total_umis_per_metacell: ut.NumpyVector,
) -> ut.NumpyVector:
    grouped_cells_mask = group_of_cells == metacell_index
    assert np.sum(grouped_cells_mask) > 0
    grouped_cells_data = ut.to_numpy_matrix(cells_data[grouped_cells_mask, :])
    total_umis_per_grouped_cell = total_umis_per_cell[grouped_cells_mask]
    metacell_data = metacells_data[metacell_index, :]
    total_umis_of_metacell = total_umis_per_metacell[metacell_index]
    expected_scale_per_grouped_cell = total_umis_per_grouped_cell / total_umis_of_metacell
    expected_cells_data = expected_scale_per_grouped_cell[:, np.newaxis] * metacell_data[np.newaxis, :]
    assert expected_cells_data.shape == grouped_cells_data.shape
    fold_factors_per_grouped_cell = np.log2((expected_cells_data + 1) / (grouped_cells_data + 1))
    fold_factors = ut.max_per(ut.to_layout(fold_factors_per_grouped_cell, "column_major"), per="column")
    max_fold_factor = np.max(fold_factors)
    ut.log_calc("max_fold_factor", max_fold_factor)
    return fold_factors


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_significant_projected_fold_factors(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    total_umis: Optional[ut.Vector] = None,
    projected: Union[str, ut.Matrix] = "projected",
    fold_normalization: float = pr.project_fold_normalization,
    min_significant_gene_value: float = pr.project_min_significant_gene_value,
    min_gene_fold_factor: float = pr.project_max_projection_fold_factor,
    min_entry_fold_factor: float = pr.min_entry_project_fold_factor,
    abs_folds: bool = pr.project_abs_folds,
) -> None:
    """
    Compute the significant projected fold factors of genes for each query metacell.

    This computes, for each metacell of the query, the fold factors between the actual query UMIs and the UMIs of the
    projection of the metacell onto the atlas (see :py:func:`metacells.tools.project.project_query_onto_atlas`). The
    result per-metacell-per-gene matrix is then made sparse by discarding too-low values (setting them to zero).
    Ideally, this matrix should be "very" sparse. If it contains "too many" non-zero values, more genes need to
    be ignored by the projection, or somehow corrected for batch effects prior to computing the projection.

    **Input**

    Annotated ``adata``, where the observations are query metacells and the variables are genes, where ``what`` is a
    per-variable-per-observation matrix or the name of a per-variable-per-observation annotation containing such a
    matrix.

    In addition, the ``projected`` UMIs of each query metacells onto the atlas.

    **Returns**

    Sets the following in ``gdata``:

    Per-Variable Per-Observation (Gene-Cell) Annotations
        ``projected_fold``
            For each gene and query metacell, the fold factor of this gene between the query and its projection (unless
            the value is too low to be of interest, in which case it will be zero).

    **Computation Parameters**

    1. For each group (metacell), for each gene, compute the gene's fold factor
       log2((actual UMIs + ``fold_normalization``) / (expected UMIs + ``fold_normalization``)), similarly to
       :py:func:`metacells.tools.project.project_query_onto_atlas` (the default ``fold_normalization`` is
       {fold_normalization}).

    2. Set the fold factor to zero for every case where the total UMIs in the query metacell and the projected image is
       not at least ``min_significant_gene_value`` (default: {min_significant_gene_value}).

    3. If the maximal fold factor for a gene (across all metacells) is below ``min_gene_fold_factor`` (default:
       {min_gene_fold_factor}), then set all the gene's fold factors to zero (too low to be of interest).

    4. Otherwise, for any metacell whose fold factor for the gene is less than ``min_entry_fold_factor`` (default:
       {min_entry_fold_factor}), set the fold factor to zero (too low to be of interest). If ``abs_folds`` (default:
       {abs_folds}), consider the absolute fold factors.
    """
    assert 0 <= min_entry_fold_factor <= min_gene_fold_factor
    assert fold_normalization >= 0

    metacells_data = ut.get_vo_proper(adata, what, layout="row_major")
    projected_data = ut.get_vo_proper(adata, projected, layout="row_major")

    metacells_fractions = ut.fraction_by(metacells_data, by="row", sums=total_umis)
    projected_fractions = ut.fraction_by(projected_data, by="row", sums=total_umis)

    metacells_fractions += fold_normalization  # type: ignore
    projected_fractions += fold_normalization  # type: ignore

    dense_folds = metacells_fractions / projected_fractions  # type: ignore
    dense_folds = np.log2(dense_folds, out=dense_folds)

    total_umis = ut.to_numpy_matrix(metacells_data + projected_data)  # type: ignore
    insignificant_folds_mask = total_umis < min_significant_gene_value
    ut.log_calc("insignificant entries", insignificant_folds_mask)
    dense_folds[insignificant_folds_mask] = 0.0
    dense_folds_by_column = ut.to_layout(dense_folds, layout="column_major")
    sparse_folds = ut.sparsify_matrix(
        dense_folds_by_column,
        min_column_max_value=min_gene_fold_factor,
        min_entry_value=min_entry_fold_factor,
        abs_values=abs_folds,
    )

    ut.set_vo_data(adata, "projected_fold", sparse_folds)


@ut.logged()
@ut.timed_call()
def compute_similar_query_metacells(
    adata: AnnData,
    max_projection_fold_factor: float = pr.project_max_projection_fold_factor,
    max_dissimilar_genes: int = pr.project_max_dissimilar_genes,
    abs_folds: bool = pr.project_abs_folds,
) -> None:
    """
    Mark query metacells that are similar to their projection on the atlas.

    This does not guarantee the query metacell is "the same as" its projection on the atlas; rather, it means the two
    are sufficiently similar that one can be "reasonably confident" in applying atlas metadata to the query metacell
    based on the projection, which is a much lower bar.

    **Input**

    Annotated query ``adata``, where the observations are cells and the variables are genes, where ``what`` is a
    per-variable-per-observation matrix or the name of a per-variable-per-observation annotation containing such a
    matrix.

    The data should contain per-observation-per-variable annotations ``projected_fold`` with the significant projection
    folds factors, as computed by :py:func:`compute_significant_projected_fold_factors`.

    **Returns**

    Sets the following in ``adata``:

    Per-Observation (Cell) Annotations

        ``similar``
            A boolean mask indicating the query metacell is similar to its projection in the atlas.
        ``dissimilar_genes_count``
            The number of genes whose fold factor is above the threshold.

    **Computation Parameters**

    1. Mark as dissimilar any query metacells which have more than ``max_dissimilar_genes`` (default:
       {max_dissimilar_genes}) genes whose projection fold is above ``max_projection_fold_factor``.
    """
    assert max_projection_fold_factor >= 0
    assert max_dissimilar_genes >= 0

    projected_folds = ut.get_vo_proper(adata, "projected_fold", layout="row_major")
    if abs_folds:
        projected_folds = np.abs(projected_folds)  # type: ignore
    high_folds = projected_folds > max_projection_fold_factor  # type: ignore
    high_folds_per_metacell = ut.sum_per(high_folds, per="row")  # type: ignore
    similar_mask = high_folds_per_metacell <= max_dissimilar_genes
    ut.log_calc("max dissimilar_genes_count", np.max(high_folds_per_metacell))
    ut.set_o_data(adata, "dissimilar_genes_count", high_folds_per_metacell, formatter=ut.mask_description)
    ut.set_o_data(adata, "similar", similar_mask)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_outliers_matches(
    what: Union[str, ut.Matrix] = "__x__",
    *,
    adata: AnnData,
    gdata: AnnData,
    group: Union[str, ut.Vector] = "metacell",
    most_similar: str = "most_similar",
    value_normalization: float = pr.outliers_fold_normalization,
    reproducible: bool,
) -> None:
    """
    Given an assignment of observations (cells) to groups (metacells), compute for each outlier the "most similar"
    group.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where ``what`` is a
    per-variable-per-observation matrix or the name of a per-variable-per-observation annotation containing such a
    matrix.

    In addition, ``gdata`` is assumed to have one observation for each group, and use the same genes as ``adata``. Note
    that there's no requirement that the ``gdata`` will contain the groups defined in ``adata``. That is, it is possible
    to give query cells data in ``adata`` and atlas metacells in ``gdata`` to find the most similar atlas metacell for
    each outlier query metacell.

    **Returns**

    Sets the following in ``adata``:

    Per-Observation (Cell) Annotations

        ``most_similar`` (default: {most_similar})
            For each observation (cell), the index of the "most similar" group.

    **Computation Parameters**

    1. Compute the log2 of the fraction of each gene in each of the outlier cells and the group metacells using
       the ``value_normalization`` (default: {value_normalization}).

    2. Cross-correlate each of the outlier cells with each of the group metacells, in a ``reproducible`` manner.
    """
    assert list(adata.var_names) == list(gdata.var_names)

    group_of_cells = ut.get_o_numpy(adata, group)
    outliers_mask = group_of_cells < 0
    odata = ut.slice(adata, obs=outliers_mask)

    outliers_data = ut.get_vo_proper(odata, what, layout="row_major")
    groups_data = ut.get_vo_proper(gdata, what, layout="row_major")

    outliers_fractions = ut.fraction_by(outliers_data, by="row")
    groups_fractions = ut.fraction_by(groups_data, by="row")

    outliers_fractions = ut.to_numpy_matrix(outliers_fractions)
    groups_fractions = ut.to_numpy_matrix(groups_fractions)

    outliers_fractions += value_normalization
    groups_fractions += value_normalization

    outliers_log_fractions = np.log2(outliers_fractions, out=outliers_fractions)
    groups_log_fractions = np.log2(groups_fractions, out=groups_fractions)

    outliers_groups_correlation = ut.cross_corrcoef_rows(
        outliers_log_fractions, groups_log_fractions, reproducible=reproducible
    )
    outliers_similar_group_indices = np.argmax(outliers_groups_correlation, axis=1)
    assert len(outliers_similar_group_indices) == odata.n_obs

    cells_similar_group_indices = np.full(adata.n_obs, -1, dtype="int32")
    cells_similar_group_indices[outliers_mask] = outliers_similar_group_indices
    ut.set_o_data(adata, most_similar, cells_similar_group_indices)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_deviant_fold_factors(
    what: Union[str, ut.Matrix] = "__x__",
    *,
    adata: AnnData,
    gdata: AnnData,
    group: Union[str, ut.Vector] = "metacell",
    most_similar: Union[str, ut.Vector] = "most_similar",
    significant_gene_fold_factor: float = pr.significant_gene_fold_factor,
) -> None:
    """
    Given an assignment of observations (cells) to groups (metacells) or, if an outlier, to the most similar groups,
    compute for each observation and gene the fold factor relative to its group for the purpose of detecting deviant
    cells.

    Ideally, all grouped cells would have no genes with high enough fold factors to be considered deviants, and all
    outlier cells would. In practice grouped cells might have a (few) such genes to the restriction on the fraction
    of deviants.

    It is important not to read too much into the results for a single cell, but looking at which genes appear for cell
    populations (e.g., cells with specific metadata such as batch identification) might be instructive.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where ``what`` is a
    per-variable-per-observation matrix or the name of a per-variable-per-observation annotation containing such a
    matrix.

    In addition, ``gdata`` is assumed to have one observation for each group, and use the same genes as ``adata``.

    **Returns**

    Sets the following in ``adata``:

    Per-Variable Per-Observation (Gene-Cell) Annotations

        ``deviant_fold``
            The fold factor between the cell's UMIs and the expected number of UMIs for the purpose of computing
            deviant cells.

    **Computation Parameters**

    1. For each cell, compute the expected UMIs for each gene given the fraction of the gene in the metacells associated
       with the cell (the one it is belongs to, or the most similar one for outliers). If this is less than
       ``significant_gene_fold_factor`` (default: {significant_gene_fold_factor}), set it to zero so the result will be
       sparse.
    """
    cells_data = ut.get_vo_proper(adata, what, layout="row_major")
    metacells_data = ut.get_vo_proper(gdata, what, layout="row_major")
    total_umis_per_cell = ut.sum_per(cells_data, per="row")
    total_umis_per_metacell = ut.sum_per(metacells_data, per="row")

    group_of_cells = ut.get_o_numpy(adata, group, formatter=ut.groups_description)
    most_similar_of_cells = ut.get_o_numpy(adata, most_similar, formatter=ut.groups_description)

    @ut.timed_call("compute_cell_deviant_certificates")
    def _compute_cell_deviant_certificates(cell_index: int) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
        return _compute_cell_certificates(
            cell_index=cell_index,
            cells_data=cells_data,
            metacells_data=metacells_data,
            group_of_cells=group_of_cells,
            most_similar_of_cells=most_similar_of_cells,
            total_umis_per_cell=total_umis_per_cell,
            total_umis_per_metacell=total_umis_per_metacell,
            significant_gene_fold_factor=significant_gene_fold_factor,
        )

    results = list(ut.parallel_map(_compute_cell_deviant_certificates, adata.n_obs))

    cell_indices = np.concatenate(
        [np.full(len(result[0]), cell_index, dtype="int32") for cell_index, result in enumerate(results)]
    )
    gene_indices = np.concatenate([result[0] for result in results])
    fold_factors = np.concatenate([result[1] for result in results])

    deviant_folds = sparse.csr_matrix((fold_factors, (cell_indices, gene_indices)), shape=adata.shape)
    ut.set_vo_data(adata, "deviant_folds", deviant_folds)


@ut.logged()
def _compute_cell_certificates(
    *,
    cell_index: int,
    cells_data: ut.Matrix,
    metacells_data: ut.Matrix,
    group_of_cells: ut.NumpyVector,
    most_similar_of_cells: ut.NumpyVector,
    total_umis_per_cell: ut.NumpyVector,
    total_umis_per_metacell: ut.NumpyVector,
    significant_gene_fold_factor: float,
) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
    group_index = group_of_cells[cell_index]
    most_similar_index = most_similar_of_cells[cell_index]
    assert (group_index < 0) != (most_similar_index < 0)
    metacell_index = max(group_index, most_similar_index)

    expected_scale = total_umis_per_cell[cell_index] / total_umis_per_metacell[metacell_index]
    expected_data = ut.to_numpy_vector(metacells_data[metacell_index, :], copy=True)
    expected_data *= expected_scale

    actual_data = ut.to_numpy_vector(cells_data[cell_index, :], copy=True)

    expected_data += 1.0
    actual_data += 1.0

    fold_factors = actual_data
    fold_factors /= expected_data
    fold_factors = np.log2(fold_factors, out=fold_factors)
    fold_factors = np.abs(fold_factors, out=fold_factors)

    significant_folds_mask = fold_factors >= significant_gene_fold_factor
    ut.log_calc("significant_folds_mask", significant_folds_mask)

    significant_gene_indices = np.where(significant_folds_mask)[0].astype("int32")
    significant_gene_folds = fold_factors[significant_folds_mask].astype("float32")
    return (significant_gene_indices, significant_gene_folds)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_outliers_fold_factors(
    what: Union[str, ut.Matrix] = "__x__",
    *,
    adata: AnnData,
    gdata: AnnData,
    most_similar: Union[str, ut.Vector] = "most_similar",
    fold_normalization: float = pr.outliers_fold_normalization,
    min_gene_outliers_fold_factor: float = pr.min_gene_outliers_fold_factor,
    min_entry_outliers_fold_factor: float = pr.min_entry_outliers_fold_factor,
    abs_folds: bool = pr.outliers_abs_folds,
) -> None:
    """
    Given annotated data which is a slice containing just the outliers, where each has a "most similar" group, compute
    for each observation and gene the fold factor relative to its group.

    All outliers should have at least one (typically several) genes with high fold factors, which are the reason they
    couldn't be merged into their most similar group.

    **Input**

    Annotated ``adata``, where the observations are outlier cells and the variables are genes, where ``what`` is a
    per-variable-per-observation matrix or the name of a per-variable-per-observation annotation containing such a
    matrix.

    In addition, ``gdata`` is assumed to have one observation for each group, and use the same genes as ``adata``.
    It should have a ``significant_gene`` mask.

    **Returns**

    Sets the following in ``adata``:

    Per-Variable Per-Observation (Gene-Cell) Annotations

        ``<most_similar>_fold`` (default: {most_similar}_fold)
            The fold factor between the outlier gene expression and their expression in the most similar group, (unless
            the value is too low to be of interest, in which case it will be zero).

    **Computation Parameters**

    1. For each outlier, compute the expected UMIs for each gene given the fraction of the gene in the metacell
       associated with the outlier by the ``most_similar`` (default: {most_similar}). If this is less than
       ``min_entry_outliers_fold_factor`` (default: {min_entry_outliers_fold_factor}), or if the gene doesn't
       have a fold factor of at least ``min_gene_outliers_fold_factor`` (default: {min_gene_outliers_fold_factor})
       in any outlier, this is set to zero to make the matrix sparse. Also, all columns for genes not listed
       in the ``significant_gene`` mask are also set to zero.
    """
    assert list(adata.var_names) == list(gdata.var_names)

    most_similar_of_outliers = ut.get_o_numpy(adata, most_similar, formatter=ut.groups_description)
    assert np.min(most_similar_of_outliers) >= 0

    outliers_data = ut.to_numpy_matrix(ut.get_vo_proper(adata, what, layout="row_major"))
    groups_data = ut.to_numpy_matrix(ut.get_vo_proper(gdata, what, layout="row_major"))
    most_similar_data = groups_data[most_similar_of_outliers, :]

    outliers_fractions = ut.fraction_by(outliers_data, by="row")
    most_similar_fractions = ut.fraction_by(most_similar_data, by="row")

    outliers_fractions += fold_normalization  # type: ignore
    most_similar_fractions += fold_normalization  # type: ignore

    outliers_log_fractions = np.log2(outliers_fractions, out=outliers_fractions)  # type: ignore
    most_similar_log_fractions = np.log2(most_similar_fractions, out=most_similar_fractions)  # type: ignore

    fold_factors = ut.to_layout(outliers_log_fractions - most_similar_log_fractions, layout="column_major")

    significant_gene_mask = ut.get_v_numpy(gdata, "significant_gene")
    fold_factors[:, ~significant_gene_mask] = 0

    sparse_folds = ut.sparsify_matrix(
        fold_factors,
        min_column_max_value=min_gene_outliers_fold_factor,
        min_entry_value=min_entry_outliers_fold_factor,
        abs_values=abs_folds,
    )

    ut.set_vo_data(adata, f"{most_similar}_fold", sparse_folds)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_type_genes_normalized_variances(
    what: Union[str, ut.Matrix] = "__x__",
    *,
    adata: AnnData,
    gdata: AnnData,
    group_property: str = "metacell",
    type_property: str = "type",
    type_gene_normalized_variance_quantile: float = pr.type_gene_normalized_variance_quantile,
) -> None:
    """
    Given metacells annotated data with type annotations, compute for each gene for each type how variable
    it is in the cells of the metacells of that type.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    In addition, ``gdata`` is assumed to have one observation for each group, and use the same
    genes as ``adata``. This should have a type annotation.

    **Returns**

    Sets the following in ``gdata``:

    Per-Variable (gene) Annotations:

        ``normalized_variance_in_<type>``
            For each type, the normalized variance (variance over mean) of the gene in the cells of the
            metacells of this type.

    **Computation Parameters**

    1. For each ``type_property`` (default: {type_property}) of metacell in ``gdata``, for each metacell of this type,
       consider all the cells in ``adata`` whose ``group_property`` (default: {group_property}) is that metacell,
       compute the normalized variance (variance over mean) of each gene's expression level, when normalizing each
       cell's total UMIs to the median in its metacell.

    2. Take the ``type_gene_normalized_variance_quantile`` (default: {type_gene_normalized_variance_quantile}) of the
       normalized variance of each gene across all metacells of each type.
    """
    assert list(adata.var_names) == list(gdata.var_names)

    type_of_metacell = ut.get_o_numpy(gdata, type_property)
    unique_types = np.unique(type_of_metacell)
    ut.log_calc("types_count", len(unique_types))

    metacell_of_cells = ut.get_o_numpy(adata, group_property)
    cells_umis = ut.get_vo_proper(adata, what, layout="row_major")

    @ut.logged()
    @ut.timed_call()
    def _compute_single_type_genes_normalized_variances(type_index: int) -> Tuple[str, ut.NumpyVector]:
        type_name = unique_types[type_index]
        ut.log_calc("type_name", type_name)
        type_metacells_indices = np.where(type_of_metacell == type_name)[0]
        ut.log_calc("type_metacells_count", len(type_metacells_indices))
        type_gene_normalized_variances_per_metacell = np.empty(
            (len(type_metacells_indices), adata.n_vars), dtype="float32"
        )

        for position, metacell_index in enumerate(type_metacells_indices):
            _compute_metacell_genes_normalized_variances(
                metacell_of_cells=metacell_of_cells,
                cells_umis=cells_umis,
                position=position,
                metacell_index=metacell_index,
                type_gene_normalized_variances_per_metacell=type_gene_normalized_variances_per_metacell,
            )

        type_gene_normalized_variances_per_metacell = ut.to_layout(
            type_gene_normalized_variances_per_metacell, layout="column_major"
        )
        return type_name, ut.quantile_per(
            type_gene_normalized_variances_per_metacell, quantile=type_gene_normalized_variance_quantile, per="column"
        )

    for (type_name, type_gene_normalized_variances) in list(
        ut.parallel_map(_compute_single_type_genes_normalized_variances, len(unique_types))
    ):
        ut.set_v_data(gdata, f"normalized_variance_in_{type_name}", type_gene_normalized_variances)


@ut.logged()
@ut.timed_call()
def _compute_metacell_genes_normalized_variances(
    *,
    metacell_of_cells: ut.NumpyVector,
    cells_umis: ut.Matrix,
    position: int,
    metacell_index: int,
    type_gene_normalized_variances_per_metacell: ut.NumpyVector,
) -> None:
    metacell_cells_mask = metacell_of_cells == metacell_index
    metacell_umis = cells_umis[metacell_cells_mask, :]
    total_umis = ut.sum_per(metacell_umis, per="row")
    median_total_umis = np.median(total_umis)
    ut.log_calc("median_total_umis", median_total_umis)
    metacell_fractions = ut.fraction_by(metacell_umis, by="row")
    metacell_fractions = ut.to_layout(metacell_fractions, layout="column_major")
    metacell_fractions *= median_total_umis
    genes_normalized_variance = ut.normalized_variance_per(metacell_fractions, per="column")
    type_gene_normalized_variances_per_metacell[position, :] = genes_normalized_variance
