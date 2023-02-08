"""
Quality
-------
"""

from typing import Collection
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
import scipy.sparse as sp  # type: ignore
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.utilities as ut

from .mask import combine_masks

__all__ = [
    "compute_inner_variance_folds",
    "compute_projected_folds",
    "compute_metacells_projection_correlation",
    "compute_similar_query_metacells",
    "compute_outliers_matches",
    "compute_deviant_folds",
    "compute_inner_folds",
    "compute_type_genes_normalized_variances",
    "compute_outliers_fold_factors",
]


T = TypeVar("T")


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_inner_variance_folds(
    what: Union[str, ut.Matrix] = "__x__",
    *,
    min_gene_total: int = pr.quality_min_gene_total,
    adata: AnnData,
    gdata: AnnData,
    group: Union[str, ut.Vector] = "metacell",
) -> None:
    """
    Compute the inner normalized variance (variance / mean) for each gene in each metacell.

    This is also known as the "index of dispersion" and can serve as a quality measure for the groups. An ideal metacell
    would contain only cells with "the same" biological state and all remaining inner variance would be due to technical
    sampling noise, so the inner normalized variance should be 1. In practice we see higher values - the lower, the
    better.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where ``what`` is a
    per-variable-per-observation (UMIs) matrix or the name of a per-variable-per-observation annotation containing such
    a matrix.

    In addition, ``gdata`` is assumed to have one (fraction) observation for each metacell, a ``total_umis`` per
    metacell, and use the same genes as ``adata``.

    **Returns**

    Sets the following in ``gdata``:

    Per-Variable Per-Observation (Gene-Cell) Annotations

        ``inner_variance_fold``
            For each gene and metacell, the normalized variance (variance over mean) of the gene in the metacell,
            if it has a sufficient number of UMIs to make this meaningful (otherwise, is 0).

    **Computation Parameters**

    For each metacell:

    1. Compute the median number of UMIs for a cell in the metacell. Scale all the UMIs so that the total number per
       cell is this median.

    2. Compute the ratio between the variance of these normalized UMIs, and the expected number of UMIs if using
       the metacells overall per-gene fractions.

    3. Set the ratio to 1.0 for genes that do not have at least ``min_gene_total`` (default: {min_gene_total}) UMIs in
       the metacell.
    """
    assert list(adata.var_names) == list(gdata.var_names)

    metacell_of_cells = ut.get_o_numpy(adata, group, formatter=ut.groups_description)
    assert gdata.n_obs == np.max(metacell_of_cells) + 1

    umis_per_gene_per_cell = ut.get_vo_proper(adata, what, layout="row_major")
    total_umis_per_metacell = ut.get_o_numpy(gdata, "total_umis")
    fraction_per_gene_per_metacell = ut.to_numpy_matrix(ut.get_vo_proper(gdata, what, layout="row_major"), copy=True)
    umis_per_gene_per_metacell = ut.to_numpy_matrix(
        ut.get_vo_proper(gdata, "total_umis", layout="row_major"), copy=True
    )

    @ut.timed_call()
    def _single_metacell_inner_variance(metacell_index: int) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
        return _compute_metacell_inner_variance(
            metacell_index=metacell_index,
            metacell_of_cells=metacell_of_cells,
            umis_per_gene_per_cell=umis_per_gene_per_cell,
            fraction_per_gene_per_metacell=fraction_per_gene_per_metacell,
            umis_per_gene_per_metacell=umis_per_gene_per_metacell,
            total_umis_per_metacell=total_umis_per_metacell,
            min_gene_total=min_gene_total,
        )

    results = list(ut.parallel_map(_single_metacell_inner_variance, gdata.n_obs))
    data = np.concatenate([metacell_fold_factors for _metacell_gene_indices, metacell_fold_factors in results])
    indices = np.concatenate([metacell_gene_indices for metacell_gene_indices, _metacell_fold_factors in results])
    indptr = np.array(
        [0] + [len(metacell_gene_indices) for metacell_gene_indices, _metacell_fold_factors in results], dtype="int32"
    )
    np.cumsum(indptr, out=indptr)

    assert data.dtype == "float32"
    assert indices.dtype == "int32"
    assert indptr.dtype == "int32"

    fold_per_gene_per_metacell = sp.csr_matrix((data, indices, indptr), shape=gdata.shape)
    ut.set_vo_data(gdata, "inner_variance_fold", sp.csr_matrix(fold_per_gene_per_metacell))


@ut.logged()
def _compute_metacell_inner_variance(
    *,
    metacell_index: int,
    metacell_of_cells: ut.NumpyVector,
    umis_per_gene_per_cell: ut.ProperMatrix,
    fraction_per_gene_per_metacell: ut.NumpyMatrix,
    umis_per_gene_per_metacell: ut.NumpyMatrix,
    total_umis_per_metacell: ut.NumpyVector,
    min_gene_total: int,
) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
    cell_indices = np.where(metacell_of_cells == metacell_index)[0]

    umis_per_gene_per_cell = ut.to_numpy_matrix(umis_per_gene_per_cell[cell_indices, :])
    total_umis_per_cell = ut.sum_per(umis_per_gene_per_cell, per="row")
    median_total_umis = np.median(total_umis_per_cell)

    total_umis_per_cell[total_umis_per_cell == 0] = 1
    scale_per_cell = median_total_umis / total_umis_per_cell
    scaled_umis_per_gene_per_cell = ut.scale_by(umis_per_gene_per_cell, by="row", scale=scale_per_cell)
    scaled_umis_per_gene_per_cell = ut.to_layout(scaled_umis_per_gene_per_cell, "column_major")
    variance_per_gene_of_metacell = ut.variance_per(scaled_umis_per_gene_per_cell, per="column")

    mean_per_gene_of_metacell = ut.to_numpy_vector(fraction_per_gene_per_metacell[metacell_index, :], copy=True)
    mean_per_gene_of_metacell *= median_total_umis

    mean_per_gene_of_metacell += 1
    variance_per_gene_of_metacell += 1
    fold_per_gene_of_metacell = np.log2(variance_per_gene_of_metacell) - np.log2(mean_per_gene_of_metacell)

    umis_per_gene_of_metacell = umis_per_gene_per_metacell[metacell_index, :]
    mask_per_gene_of_metacell = umis_per_gene_of_metacell < min_gene_total / total_umis_per_metacell[metacell_index]
    fold_per_gene_of_metacell[mask_per_gene_of_metacell] = 0.0

    gene_indices = np.where(fold_per_gene_of_metacell != 0)[0]
    ut.log_calc("nnz_fold_genes_count", len(gene_indices))
    return (gene_indices.astype("int32"), fold_per_gene_of_metacell[gene_indices].astype("float32"))


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_projected_folds(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    total_umis: Optional[ut.Vector] = None,
    projected: Union[str, ut.Matrix] = "projected",
    fold_normalization: float = pr.project_fold_normalization,
    min_significant_gene_value: float = pr.project_min_significant_gene_value,
) -> None:
    """
    Compute the projected fold factors of genes for each query metacell.

    This computes, for each metacell of the query, the fold factors between the actual query UMIs and the UMIs of the
    projection of the metacell onto the atlas (see :py:func:`metacells.tools.project.project_query_onto_atlas`).

    **Input**

    Annotated ``adata``, where the observations are query metacells and the variables are genes, where ``what`` is a
    per-variable-per-observation matrix or the name of a per-variable-per-observation annotation containing such a
    matrix.

    In addition, the ``projected`` UMIs of each query metacells onto the atlas.

    **Returns**

    Sets the following in ``gdata``:

    Per-Variable Per-Observation (Gene-Cell) Annotations
        ``projected_fold``
            For each gene and query metacell, the fold factor of this gene between the query and its projection.

    **Computation Parameters**

    1. For each group (metacell), for each gene, compute the gene's fold factor
       log2((actual UMIs + ``fold_normalization``) / (expected UMIs + ``fold_normalization``)), similarly to
       :py:func:`metacells.tools.project.project_query_onto_atlas` (the default ``fold_normalization`` is
       {fold_normalization}).

    2. Set the fold factor to zero for every case where the total UMIs in the query metacell and the projected image is
       not at least ``min_significant_gene_value`` (default: {min_significant_gene_value}).
    """
    assert fold_normalization >= 0

    metacells_data = ut.get_vo_proper(adata, what, layout="row_major")
    projected_data = ut.get_vo_proper(adata, projected, layout="row_major")

    metacells_fractions = ut.fraction_by(metacells_data, by="row", sums=total_umis)
    projected_fractions = ut.fraction_by(projected_data, by="row", sums=total_umis)

    metacells_fractions = ut.to_numpy_matrix(metacells_fractions)
    projected_fractions = ut.to_numpy_matrix(projected_fractions)

    metacells_fractions += fold_normalization
    projected_fractions += fold_normalization

    dense_folds = np.log2(metacells_fractions) - np.log2(projected_fractions)

    total_umis = ut.to_numpy_matrix(metacells_data + projected_data)  # type: ignore
    insignificant_folds_mask = total_umis < min_significant_gene_value
    ut.log_calc("insignificant entries", insignificant_folds_mask)

    dense_folds[insignificant_folds_mask] = 0.0
    dense_folds_by_column = ut.to_layout(dense_folds, layout="column_major")

    ut.set_vo_data(adata, "projected_fold", dense_folds_by_column)


@ut.logged()
@ut.timed_call()
def compute_metacells_projection_correlation(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    gene_masks: Collection[str] = [
        "&atlas_gene?",
        "&~lateral_gene?",
        "&~atlas_lateral_gene?",
        "&marker_gene?",
        "&atlas_marker_gene?",
        "&~noisy_gene?",
        "&~atlas_noisy_gene?",
    ],
    projected: Union[str, ut.Matrix] = "projected",
    reproducible: bool,
) -> None:
    """
    Compute the correlation between the metacells UMIs and their projection on the atlas.

    **Input**

    Annotated query ``adata``, where the observations are metacells and the variables are genes and ``what`` is the
    (corrected) UMIs of the metacells.

    The data should contain per-observation-per-variable annotations ``projected`` with the projection of the
    metacells on some atlas.

    **Returns**

    Sets the ``projected_correlation`` per-observation annotation to the correlation between the corrected and the
    projected UMIs for each metacell. Correlation only looks at a subset of the genes specified by the
    ``mask_names``; by default, it looks only at genes common to the atlas and the query, that were "marker" in
    both, and that were not lateral or by noisy (forbidden from being selected for computing metacells).

    If ``reproducible``, a slower (still parallel) but reproducible algorithm will be used.
    """

    mask = combine_masks(adata, gene_masks)
    included_adata = ut.slice(adata, vars=mask, name=".included")
    corrected = ut.to_numpy_matrix(ut.get_vo_proper(included_adata, what))
    projected = ut.to_numpy_matrix(ut.get_vo_proper(included_adata, projected))
    projected_correlation = ut.pairs_corrcoef_rows(corrected, projected, reproducible=reproducible)
    ut.set_o_data(adata, "projected_correlation", projected_correlation)


@ut.logged()
@ut.timed_call()
def compute_similar_query_metacells(
    adata: AnnData,
    max_projection_fold_factor: float = pr.project_max_projection_fold_factor,
    max_dissimilar_genes: int = pr.project_max_dissimilar_genes,
    essential_genes_property: Union[None, str, Collection[str]] = None,
    min_similar_essential_genes: Optional[float] = None,
    abs_folds: bool = pr.project_abs_folds,
) -> None:
    """
    Mark query metacells that are similar to their projection on the atlas.

    This does not guarantee the query metacell is "the same as" its projection on the atlas; rather, it means the two
    are sufficiently similar that one can be "reasonably confident" in applying atlas metadata to the query metacell
    based on the projection, which is a much lower bar.

    **Input**

    Annotated query ``adata``, where the observations are metacells and the variables are genes.

    The data should contain per-observation-per-variable annotations ``projected_fold`` with the significant projection
    folds factors, as computed by :py:func:`compute_projected_folds`. If
    ``min_essential_significant_genes_fraction``, and ``essential_genes_property`` are specified, then the data may
    contain additional per-observation (gene) mask(s) denoting the essential genes.

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

    2. If ``essential_genes_property`` and ``min_similar_essential_genes`` are specified, the former should be the
       name(s) of boolean per-gene property/ies, and we will mark as dissimilar any query metacells which do not have at
       least this number of genes denoted by that mask(s) with a projection fold at most ``max_projection_fold_factor``.
    """
    assert max_projection_fold_factor >= 0
    assert max_dissimilar_genes >= 0

    projected_folds = ut.get_vo_proper(adata, "projected_fold", layout="row_major")
    if abs_folds:
        projected_folds = np.abs(projected_folds)  # type: ignore
    high_folds = projected_folds > max_projection_fold_factor  # type: ignore
    high_folds_per_metacell = ut.sum_per(high_folds, per="row")  # type: ignore
    ut.set_o_data(adata, "dissimilar_genes_count", high_folds_per_metacell, formatter=ut.mask_description)

    ut.log_calc("max dissimilar_genes_count", np.max(high_folds_per_metacell))
    similar_mask = high_folds_per_metacell <= min(max_dissimilar_genes, adata.n_vars)

    if essential_genes_property is not None and min_similar_essential_genes is not None:
        essential_genes_mask = np.zeros(adata.n_vars, dtype="bool")
        if isinstance(essential_genes_property, str):
            essential_genes_property = [essential_genes_property]
        for property_name in essential_genes_property:
            essential_genes_mask |= ut.get_v_numpy(adata, property_name)

        ut.log_calc("essential_genes_mask", essential_genes_mask)
        ut.log_calc("essential_gene_names", adata.var_names[essential_genes_mask])
        essential_similar_genes_mask = ~high_folds[:, essential_genes_mask]  # type: ignore
        essential_similar_genes_per_metacell = ut.sum_per(
            ut.to_layout(essential_similar_genes_mask, layout="row_major"), per="row"
        )
        ut.set_o_data(adata, "similar_essential_genes_count", essential_similar_genes_per_metacell)
        essential_metacells_similar_mask = essential_similar_genes_per_metacell >= min_similar_essential_genes
        ut.log_calc("essential_metacells_similar_mask", essential_metacells_similar_mask)
        similar_mask &= essential_metacells_similar_mask

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
    cells_similar_metacell_indices = np.full(adata.n_obs, -1, dtype="int32")

    metacell_per_cell = ut.get_o_numpy(adata, group)
    outliers_mask = metacell_per_cell < 0
    if not np.any(outliers_mask):
        ut.set_o_data(adata, most_similar, cells_similar_metacell_indices)
        return

    odata = ut.slice(adata, obs=outliers_mask)

    outliers_data = ut.get_vo_proper(odata, what, layout="row_major")
    metacells_data = ut.get_vo_proper(gdata, what, layout="row_major")

    outliers_fractions = ut.fraction_by(outliers_data, by="row")
    metacells_fractions = ut.fraction_by(metacells_data, by="row")

    outliers_fractions = ut.to_numpy_matrix(outliers_fractions)
    metacells_fractions = ut.to_numpy_matrix(metacells_fractions)

    outliers_fractions += value_normalization
    metacells_fractions += value_normalization

    outliers_log_fractions = np.log2(outliers_fractions, out=outliers_fractions)
    metacells_log_fractions = np.log2(metacells_fractions, out=metacells_fractions)

    outliers_metacells_correlation = ut.cross_corrcoef_rows(
        outliers_log_fractions, metacells_log_fractions, reproducible=reproducible
    )
    outliers_similar_metacell_indices = np.argmax(outliers_metacells_correlation, axis=1)
    assert len(outliers_similar_metacell_indices) == odata.n_obs

    cells_similar_metacell_indices[outliers_mask] = outliers_similar_metacell_indices
    ut.set_o_data(adata, most_similar, cells_similar_metacell_indices)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_deviant_folds(
    what: Union[str, ut.Matrix] = "__x__",
    *,
    adata: AnnData,
    gdata: AnnData,
    group: Union[str, ut.Vector] = "metacell",
    most_similar: Union[str, ut.Vector] = "most_similar",
    min_gene_total: int = pr.quality_min_gene_total,
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
       with the cell (the one it is belongs to, or the most similar one for outliers).

    2. If the number of UMIs in the metacell (for grouped cells), or sum of the UMIs of the gene in an outlier cell and
       the metacell, is less than ``min_gene_total`` (default: {min_gene_total}), set the fold factor to 0 as we do not
       have sufficient data to robustly estimate it.
    """
    umis_per_gene_per_cell = ut.get_vo_proper(adata, what, layout="row_major")
    fraction_per_gene_per_metacell = ut.get_vo_proper(gdata, what, layout="row_major")
    umis_per_gene_per_metacell = ut.get_vo_proper(gdata, "total_umis", layout="row_major")

    metacell_per_cell = ut.get_o_numpy(adata, group, formatter=ut.groups_description)
    most_similar_per_cell = ut.get_o_numpy(adata, most_similar, formatter=ut.groups_description)
    outliers_mask = metacell_per_cell < 0
    assert np.all(most_similar_per_cell[outliers_mask] >= 0)
    assert np.all(most_similar_per_cell[~outliers_mask] < 0)
    combined_per_cell = metacell_per_cell.copy()
    combined_per_cell[outliers_mask] = most_similar_per_cell[outliers_mask]

    @ut.timed_call()
    def _single_cell_deviant_folds(cell_index: int) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
        return _compute_cell_deviant_folds(
            cell_index=cell_index,
            umis_per_gene_per_cell=umis_per_gene_per_cell,
            fraction_per_gene_per_metacell=fraction_per_gene_per_metacell,
            umis_per_gene_per_metacell=umis_per_gene_per_metacell,
            outliers_mask=outliers_mask,
            metacell_per_cell=combined_per_cell,
            min_gene_total=min_gene_total,
        )

    results = list(ut.parallel_map(_single_cell_deviant_folds, adata.n_obs))
    data = np.concatenate([cell_fold_factors for _cell_gene_indices, cell_fold_factors in results])
    indices = np.concatenate([cell_gene_indices for cell_gene_indices, _cell_fold_factors in results])
    indptr = np.array(
        [0] + [len(cell_gene_indices) for cell_gene_indices, _cell_fold_factors in results], dtype="int32"
    )
    np.cumsum(indptr, out=indptr)

    assert data.dtype == "float32"
    assert indices.dtype == "int32"
    assert indptr.dtype == "int32"

    fold_factors_per_gene_per_cell = sp.csr_matrix((data, indices, indptr), shape=adata.shape)
    ut.set_vo_data(adata, "deviant_fold", fold_factors_per_gene_per_cell)


@ut.logged()
def _compute_cell_deviant_folds(
    *,
    cell_index: int,
    umis_per_gene_per_cell: ut.Matrix,
    fraction_per_gene_per_metacell: ut.Matrix,
    umis_per_gene_per_metacell: ut.Matrix,
    metacell_per_cell: ut.NumpyVector,
    outliers_mask: ut.NumpyVector,
    min_gene_total: int,
) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
    actual_umis_per_gene = ut.to_numpy_vector(umis_per_gene_per_cell[cell_index, :], copy=True)
    total_umis_of_cell = np.sum(actual_umis_per_gene)

    metacell_index = metacell_per_cell[cell_index]
    metacell_fraction_per_gene = ut.to_numpy_vector(fraction_per_gene_per_metacell[metacell_index, :])
    expected_umis_per_gene = total_umis_of_cell * metacell_fraction_per_gene

    actual_umis_per_gene += 1
    expected_umis_per_gene += 1
    fold_factors = np.log2(actual_umis_per_gene) - np.log2(expected_umis_per_gene)

    metacell_umis_per_gene = ut.to_numpy_vector(umis_per_gene_per_metacell[metacell_index, :])
    if outliers_mask[cell_index]:
        total_umis_per_gene = actual_umis_per_gene + metacell_umis_per_gene
    else:
        total_umis_per_gene = metacell_umis_per_gene

    gene_indices = np.where(total_umis_per_gene >= min_gene_total)[0].astype("int32")
    ut.log_calc("nnz_fold_genes_count", len(gene_indices))
    return (gene_indices.astype("int32"), fold_factors[gene_indices].astype("float32"))


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_inner_folds(
    *,
    adata: AnnData,
    gdata: AnnData,
    group: Union[str, ut.Vector] = "metacell",
    abs_folds: bool = pr.inner_abs_folds,
) -> None:
    """
    Given ``adata`` with computed ``deviant_fold`` for each gene for each cell, set in ``inner_fold`` in ``gdata``, for
    each gene for each metacell the ``deviant_fold`` with the maximal absolute value (if ``abs_folds``, default:
    {abs_folds}), otherwise just with the maximal value.
    """
    group_per_cell = ut.get_o_numpy(adata, group)
    deviant_fold_per_gene_per_cell = ut.get_vo_proper(adata, "deviant_fold")

    @ut.timed_call()
    def _single_metacell_inner_folds(metacell_index: int) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
        return _compute_metacell_inner_folds(
            metacell_index=metacell_index,
            group_per_cell=group_per_cell,
            deviant_fold_per_gene_per_cell=deviant_fold_per_gene_per_cell,
            abs_folds=abs_folds,
        )

    results = list(ut.parallel_map(_single_metacell_inner_folds, gdata.n_obs))

    data = np.concatenate([metacell_fold_factors for _metacell_gene_indices, metacell_fold_factors in results])
    indices = np.concatenate([metacell_gene_indices for metacell_gene_indices, _metacell_fold_factors in results])
    assert len(data) == len(indices)
    indptr = np.array(
        [0] + [len(metacell_gene_indices) for metacell_gene_indices, _metacell_fold_factors in results], dtype="int32"
    )
    np.cumsum(indptr, out=indptr)

    assert data.dtype == "float32"
    assert indices.dtype == "int32"
    assert indptr.dtype == "int32"

    fold_factors_per_gene_per_metacell = sp.csr_matrix((data, indices, indptr), shape=gdata.shape)
    ut.set_vo_data(gdata, "inner_fold", fold_factors_per_gene_per_metacell)


@ut.logged()
def _compute_metacell_inner_folds(
    *,
    metacell_index: int,
    group_per_cell: ut.NumpyVector,
    deviant_fold_per_gene_per_cell: ut.ProperMatrix,
    abs_folds: bool,
) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
    cells_mask = group_per_cell == metacell_index
    cells_count = np.sum(cells_mask)
    assert cells_count > 0
    genes_count = deviant_fold_per_gene_per_cell.shape[1]

    tmp_deviant_fold_per_gene_per_cell = ut.to_numpy_matrix(
        deviant_fold_per_gene_per_cell[cells_mask, :], copy=abs_folds
    )
    tmp_deviant_fold_per_gene_per_cell = ut.to_layout(tmp_deviant_fold_per_gene_per_cell, layout="column_major")
    assert tmp_deviant_fold_per_gene_per_cell.shape == (cells_count, genes_count)

    if abs_folds:
        negative_folds_mask = tmp_deviant_fold_per_gene_per_cell < 0
        tmp_deviant_fold_per_gene_per_cell[negative_folds_mask] *= -1

    max_index_per_gene = np.argmax(tmp_deviant_fold_per_gene_per_cell, axis=0)
    assert len(max_index_per_gene) == genes_count

    if abs_folds:
        tmp_deviant_fold_per_gene_per_cell[negative_folds_mask] *= -1

    max_deviant_fold_per_gene = tmp_deviant_fold_per_gene_per_cell[max_index_per_gene, range(genes_count)]
    assert len(max_deviant_fold_per_gene) == genes_count

    gene_indices = np.where(max_deviant_fold_per_gene != 0.0)[0].astype("int32")
    ut.log_calc("nnz_fold_genes_count", len(gene_indices))
    return (gene_indices.astype("int32"), max_deviant_fold_per_gene[gene_indices].astype("float32"))


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

    metacell_per_cell = ut.get_o_numpy(adata, group_property)
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
                metacell_per_cell=metacell_per_cell,
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
    metacell_per_cell: ut.NumpyVector,
    cells_umis: ut.Matrix,
    position: int,
    metacell_index: int,
    type_gene_normalized_variances_per_metacell: ut.NumpyVector,
) -> None:
    metacell_cells_mask = metacell_per_cell == metacell_index
    metacell_umis = cells_umis[metacell_cells_mask, :]
    total_umis = ut.sum_per(metacell_umis, per="row")
    median_total_umis = np.median(total_umis)
    ut.log_calc("median_total_umis", median_total_umis)
    metacell_fractions = ut.fraction_by(metacell_umis, by="row")
    metacell_fractions = ut.to_layout(metacell_fractions, layout="column_major")
    metacell_fractions *= median_total_umis
    genes_normalized_variance = ut.normalized_variance_per(metacell_fractions, per="column")
    type_gene_normalized_variances_per_metacell[position, :] = genes_normalized_variance


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_outliers_fold_factors(
    what: Union[str, ut.Matrix] = "__x__",
    *,
    adata: AnnData,
    gdata: AnnData,
    most_similar: Union[str, ut.Vector] = "most_similar",
    min_gene_total: int = pr.quality_min_gene_total,
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
    It should have a ``marker_gene`` mask.

    **Returns**

    Sets the following in ``adata``:

    Per-Variable Per-Observation (Gene-Cell) Annotations

        ``<most_similar>_fold`` (default: {most_similar}_fold)
            The fold factor between the outlier gene expression and their expression in the most similar group, (unless
            the value is too low to be of interest, in which case it will be zero).

    **Computation Parameters**

    1. For each outlier, compute the expected UMIs for each gene given the fraction of the gene in the metacell
       associated with the outlier by the ``most_similar`` (default: {most_similar}).

    2. If the sum of the UMIs of the gene in cell and the metacell are less than ``min_gene_total`` (default:
       {min_gene_total}), set the fold factor to 0 as we do not have sufficient data to robustly estimate it.
    """
    assert list(adata.var_names) == list(gdata.var_names)

    actual_umis_per_gene_per_outlier = ut.to_numpy_matrix(ut.get_vo_proper(adata, what, layout="row_major"), copy=True)
    total_umis_per_outlier = ut.sum_per(actual_umis_per_gene_per_outlier, per="row")

    most_similar_of_outliers = ut.get_o_numpy(adata, most_similar, formatter=ut.groups_description)
    assert np.min(most_similar_of_outliers) >= 0

    metacells_fractions = ut.to_numpy_matrix(ut.get_vo_proper(gdata, what, layout="row_major"))
    metacells_umis = ut.to_numpy_matrix(ut.get_vo_proper(gdata, "total_umis", layout="row_major"))
    fraction_per_gene_per_most_similar = ut.to_numpy_matrix(metacells_fractions[most_similar_of_outliers, :])
    umis_per_gene_per_most_similar = ut.to_numpy_matrix(metacells_umis[most_similar_of_outliers, :])

    expected_umis_per_gene_per_outlier = fraction_per_gene_per_most_similar * total_umis_per_outlier[:, np.newaxis]
    assert actual_umis_per_gene_per_outlier.shape == expected_umis_per_gene_per_outlier.shape

    actual_umis_per_gene_per_outlier += 1
    expected_umis_per_gene_per_outlier += 1

    fold_factor_per_gene_per_outlier = np.log2(actual_umis_per_gene_per_outlier) - np.log2(
        expected_umis_per_gene_per_outlier
    )

    total_umis_per_gene_per_most_similar = umis_per_gene_per_most_similar[most_similar_of_outliers, :]
    total_umis_per_gene_per_fold = actual_umis_per_gene_per_outlier + total_umis_per_gene_per_most_similar
    fold_factor_per_gene_per_outlier[total_umis_per_gene_per_fold < min_gene_total] = 0.0

    ut.set_vo_data(adata, f"{most_similar}_fold", sp.csr_matrix(fold_factor_per_gene_per_outlier))
