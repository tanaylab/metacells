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

__all__ = [
    "compute_stdev_logs",
    "compute_projected_folds",
    "compute_similar_query_metacells",
    "compute_outliers_matches",
    "compute_deviant_folds",
    "compute_inner_folds",
    "compute_type_genes_normalized_variances",
    "compute_outliers_fold_factors",
    "count_significant_inner_folds",
]


T = TypeVar("T")


@ut.logged()
@ut.timed_call()
def compute_stdev_logs(
    what: Union[str, ut.Matrix] = "__x__",
    *,
    min_gene_total: int = pr.quality_min_gene_total,
    adata: AnnData,
    gdata: AnnData,
    group: Union[str, ut.Vector] = "metacell",
) -> None:
    """
    Compute the standard deviation of the log (base 2) of the fraction of each gene in the cells of the metacell.

    Ideally, the standard deviation should be ~1/3rd of the ``deviants_min_gene_fold_factor`` (which is ``3`` by
    default), indicating that (all)most cells are within that maximal fold factor. In practice we may see higher values.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where ``what`` is a
    per-variable-per-observation (UMIs) matrix or the name of a per-variable-per-observation annotation containing such
    a matrix.

    In addition, ``gdata`` is assumed to have one (fraction) observation for each metacell, a ``total_umis`` per
    metacell, and use the same genes as ``adata``.

    **Returns**

    Sets the following in ``gdata``:

    Per-Variable Per-Observation (Gene-Cell) Annotations

        ``inner_stdev_log``
            For each gene and metacell, the normalized variance (variance over mean) of the gene in the metacell,
            if it has a sufficient number of UMIs to make this meaningful (otherwise, is 0).

    **Computation Parameters**

    For each metacell:

    1. Compute the log (base 2) of the fractions of the UMIs of each gene in each cell, regularized by 1 UMI.

    2. Compute the standard deviation of these logs for each gene across all cells of each metacell.
    """
    assert list(adata.var_names) == list(gdata.var_names)

    metacell_of_cells = ut.get_o_numpy(adata, group, formatter=ut.groups_description)
    assert gdata.n_obs == np.max(metacell_of_cells) + 1

    umis_per_gene_per_cell = ut.get_vo_proper(adata, what, layout="row_major")

    @ut.timed_call()
    def _single_metacell_stdev_log(metacell_index: int) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
        return _compute_metacell_stdev_log(
            metacell_index=metacell_index,
            metacell_of_cells=metacell_of_cells,
            umis_per_gene_per_cell=umis_per_gene_per_cell,
            min_gene_total=min_gene_total,
        )

    results = list(ut.parallel_map(_single_metacell_stdev_log, gdata.n_obs))
    data = np.concatenate([metacell_fold_factors for _metacell_gene_indices, metacell_fold_factors in results])
    indices = np.concatenate([metacell_gene_indices for metacell_gene_indices, _metacell_fold_factors in results])
    indptr = np.array(
        [0] + [len(metacell_gene_indices) for metacell_gene_indices, _metacell_fold_factors in results], dtype="int64"
    )
    np.cumsum(indptr, out=indptr)

    assert data.dtype == "float32"
    assert indices.dtype == "int32"
    assert indptr.dtype == "int64"

    inner_stdev_log_per_gene_per_metacell = sp.csr_matrix((data, indices, indptr), shape=gdata.shape)
    inner_stdev_log_per_gene_per_metacell.has_sorted_indices = True
    inner_stdev_log_per_gene_per_metacell.has_canonical_format = True
    ut.set_vo_data(gdata, "inner_stdev_log", inner_stdev_log_per_gene_per_metacell)


@ut.logged()
def _compute_metacell_stdev_log(
    *,
    metacell_index: int,
    metacell_of_cells: ut.NumpyVector,
    umis_per_gene_per_cell: ut.ProperMatrix,
    min_gene_total: int,
) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
    ut.log_calc("metacell_index", metacell_index)
    cell_indices = np.where(metacell_of_cells == metacell_index)[0]
    umis_per_gene_per_cell_by_rows = ut.to_numpy_matrix(umis_per_gene_per_cell[cell_indices, :])
    umis_per_gene_per_cell_by_columns = ut.to_layout(umis_per_gene_per_cell_by_rows, layout="column_major")
    umis_per_gene = ut.sum_per(umis_per_gene_per_cell_by_columns, per="column")
    genes_mask = umis_per_gene > min_gene_total
    ut.log_calc("genes_mask", genes_mask)
    genes_indices = np.where(genes_mask)[0].astype("int32")
    if len(genes_indices) == 0:
        return (genes_indices, genes_indices.astype("float32"))

    regularized_umis_per_gene_per_cell = ut.to_layout(
        umis_per_gene_per_cell_by_rows[:, genes_indices] + 1, layout="row_major"
    )
    fraction_per_gene_per_cell = ut.fraction_by(regularized_umis_per_gene_per_cell, by="row")
    log_fraction_per_gene_per_cell = np.log2(fraction_per_gene_per_cell, out=fraction_per_gene_per_cell)  # type: ignore
    log_fraction_per_gene_per_cell = ut.to_layout(log_fraction_per_gene_per_cell, layout="column_major")
    stdev_per_gene = ut.stdev_per(log_fraction_per_gene_per_cell, per="column").astype("float32")
    ut.log_calc("stdev_per_gene", stdev_per_gene, formatter=ut.sizes_description)

    gene_indices = np.where(genes_mask)[0].astype("int32")
    return (gene_indices, stdev_per_gene)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_projected_folds(
    qdata: AnnData,
    from_query_layer: str = "corrected_fraction",
    to_query_layer: str = "projected_fraction",
    fold_regularization: float = pr.project_fold_regularization,
    min_significant_gene_umis: float = pr.project_min_significant_gene_umis,
) -> None:
    """
    Compute the projected fold factors of genes for each query metacell.

    This computes, for each metacell of the query, the fold factors between the corrected and projected gene fractions
    projection of the metacell onto the atlas (see :py:func:`metacells.tools.project.compute_projection_weights`).

    **Input**

    Annotated query ``qdata``, where the observations are query metacells and the variables are genes, where ``what`` is
    a per-variable-per-observation matrix or the name of a per-variable-per-observation annotation containing such a
    matrix.

    In addition, the ``projected`` UMIs of each query metacells onto the atlas.

    **Returns**

    Sets the following in ``qdata``:

    Per-Variable Per-Observation (Gene-Cell) Annotations
        ``projected_fold``
            For each gene and query metacell, the fold factor of this gene between the query and its projection.

    **Computation Parameters**

    1. For each group (metacell), for each gene, compute the gene's fold factor log2((``from_query_layer`` (default:
       {from_query_layer}) + ``fold_regularization``) / (``to_query_layer`` (default: {to_query_layer}) fractions +
       ``fold_regularization``)), similarly to :py:func:`metacells.tools.project.compute_projection_weights` (the
       default ``fold_regularization`` is {fold_regularization}).

    2. Set the fold factor to zero for every case where the total UMIs of the gene in the query metacell are not at
       least ``min_significant_gene_umis`` (default: {min_significant_gene_umis}).
    """
    assert fold_regularization >= 0

    corrected_fractions = ut.get_vo_proper(qdata, from_query_layer, layout="row_major")
    projected_fractions = ut.get_vo_proper(qdata, to_query_layer, layout="row_major")

    corrected_fractions = ut.to_numpy_matrix(corrected_fractions, copy=True)
    projected_fractions = ut.to_numpy_matrix(projected_fractions, copy=True)

    corrected_fractions += fold_regularization
    projected_fractions += fold_regularization

    dense_folds = np.log2(corrected_fractions) - np.log2(projected_fractions)

    total_umis = ut.to_numpy_matrix(ut.get_vo_proper(qdata, "total_umis", layout="row_major"))
    insignificant_folds_mask = total_umis < min_significant_gene_umis
    ut.log_calc("insignificant folds mask", insignificant_folds_mask)

    dense_folds[insignificant_folds_mask] = 0.0
    sparse_folds = sp.csr_matrix(dense_folds)
    sparse_folds.has_sorted_indices = True
    sparse_folds.has_canonical_format = True
    ut.set_vo_data(qdata, "projected_fold", sparse_folds)


@ut.logged()
@ut.timed_call()
def compute_similar_query_metacells(  # pylint: disable=too-many-statements
    qdata: AnnData,
    max_projection_fold_factor: float = pr.project_max_projection_fold_factor,
    max_projection_noisy_fold_factor: float = pr.project_max_projection_noisy_fold_factor,
    min_fitted_query_marker_genes: float = 0,
    max_misfit_genes: int = pr.project_max_misfit_genes,
    essential_genes_property: Union[None, str, Collection[str]] = None,
    min_essential_genes: Optional[int] = None,
    fitted_genes_mask: Optional[ut.NumpyVector] = None,
) -> None:
    """
    Mark query metacells that are "similar" to their projection on the atlas.

    This does not guarantee the query metacell is "the same as" its projection on the atlas; rather, it means the two
    are "sufficiently similar" that one can be reasonably confident in applying atlas metadata to the query metacell
    based on the projection.

    **Input**

    Annotated query ``qdata``, where the observations are metacells and the variables are genes.

    The data should contain per-observation-per-variable annotations ``projected_fold`` with the significant projection
    folds factors, as computed by :py:func:`compute_projected_folds`. If ``min_essential_significant_genes_fraction``,
    and ``essential_genes_property`` are specified, then the data may contain additional per-observation (gene) mask(s)
    denoting the essential genes.

    If a ``projected_noisy_gene`` mask exists, then the genes in it allow for a higher fold factor than normal genes.

    **Returns**

    Sets the following in ``qdata``:

    Per-Observation (Cell) Annotations

        ``similar``
            A boolean mask indicating the query metacell is similar to its projection in the atlas.

    Per-Variable Per-Observation (Gene-Cell) Annotations
        ``misfit``
            Whether the gene has a too-high fold factor between the query and its projection in the atlas.

    **Computation Parameters**

    1. If ``fitted_genes_mask`` is not ``None``, restrict the analysis to the genes listed in it.

    2. Mark as dissimilar any query metacells which have more than ``max_misfit_genes`` (default:
       {max_misfit_genes}) genes whose projection fold is above ``max_projection_fold_factor``,
       or, for genes in ``projected_noisy_gene``, above an additional ``max_projection_noisy_fold_factor``.

    3. Mark as dissimilar any query metacells which did not fit at least ``min_fitted_query_marker_genes`` of the
       query marker genes.

    4. If ``essential_genes_property`` and ``min_essential_genes`` are specified, the former should be the name(s) of
       boolean per-gene property/ies, and we will mark as dissimilar any query metacells which have at least this number
       of essential genes with a low projection fold factor.
    """
    assert max_projection_fold_factor >= 0
    assert max_projection_noisy_fold_factor >= 0
    assert max_misfit_genes >= 0

    projected_fold_per_gene_per_metacell = ut.get_vo_proper(qdata, "projected_fold", layout="row_major")
    projected_fold_per_gene_per_metacell = np.abs(projected_fold_per_gene_per_metacell)  # type: ignore

    if ut.has_data(qdata, "projected_noisy_gene"):
        max_projection_fold_factor_per_gene = np.full(qdata.n_vars, max_projection_fold_factor, dtype="float32")
        noisy_per_gene = ut.get_v_numpy(qdata, "projected_noisy_gene")
        max_projection_fold_factor_per_gene[noisy_per_gene] += max_projection_noisy_fold_factor
        misfit_per_gene_per_metacell_proper = (
            projected_fold_per_gene_per_metacell > max_projection_fold_factor_per_gene[np.newaxis, :]
        )
    else:
        misfit_per_gene_per_metacell_proper = projected_fold_per_gene_per_metacell > max_projection_fold_factor
    misfit_per_gene_per_metacell = sp.csr_matrix(misfit_per_gene_per_metacell_proper)
    misfit_per_gene_per_metacell.has_sorted_indices = True
    misfit_per_gene_per_metacell.has_canonical_format = True
    ut.set_vo_data(qdata, "misfit", misfit_per_gene_per_metacell)

    if fitted_genes_mask is None:
        misfit_per_fitted_gene_per_metacell = misfit_per_gene_per_metacell
    else:
        misfit_per_fitted_gene_per_metacell = misfit_per_gene_per_metacell[:, fitted_genes_mask]

    misfit_per_metacell = ut.sum_per(misfit_per_fitted_gene_per_metacell, per="row")
    ut.log_calc("misfit_per_metacell", misfit_per_metacell, formatter=ut.sizes_description)

    similar_mask = misfit_per_metacell <= max_misfit_genes
    ut.log_calc("similar_mask", similar_mask)

    query_marker_genes_mask = ut.get_v_numpy(qdata, "marker_gene")
    if fitted_genes_mask is not None:
        fitted_query_marker_genes_mask = query_marker_genes_mask & fitted_genes_mask
        ut.log_calc("fitted_query_marker_genes_mask", fitted_query_marker_genes_mask)
    else:
        fitted_query_marker_genes_mask = query_marker_genes_mask
    fitted_query_marker_genes = np.sum(fitted_query_marker_genes_mask)
    ut.log_calc("fitted_query_marker_genes", fitted_query_marker_genes)
    if fitted_query_marker_genes < min_fitted_query_marker_genes:
        similar_mask[:] = False
        ut.log_calc("similar_mask", similar_mask)

    if essential_genes_property is not None and min_essential_genes is not None:
        essential_genes_mask = np.zeros(qdata.n_vars, dtype="bool")
        if isinstance(essential_genes_property, str):
            essential_genes_property = [essential_genes_property]
        for property_name in essential_genes_property:
            essential_genes_mask |= ut.get_v_numpy(qdata, property_name)

        ut.log_calc("essential_genes_mask", essential_genes_mask)
        essential_genes_count = np.sum(essential_genes_mask)
        if essential_genes_count > 0:
            ut.log_calc("essential_gene_names", qdata.var_names[essential_genes_mask])
            misfit_per_essential_gene_per_metacell = misfit_per_gene_per_metacell[:, essential_genes_mask]
            misfit_essential_per_metacell = ut.sum_per(misfit_per_essential_gene_per_metacell, per="row")
            fitted_essential_per_metacell = essential_genes_count - misfit_essential_per_metacell
        else:
            fitted_essential_per_metacell = np.zeros(len(misfit_per_metacell), dtype="bool")

        ut.log_calc("fitted_essential_per_metacell", fitted_essential_per_metacell, formatter=ut.sizes_description)
        essential_similar_mask = fitted_essential_per_metacell >= min_essential_genes
        ut.log_calc("essential_similar_mask", essential_similar_mask)

        similar_mask &= essential_similar_mask

    ut.set_o_data(qdata, "similar", similar_mask)


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
    value_regularization: float = pr.outliers_fold_regularization,
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
       the ``value_regularization`` (default: {value_regularization}).

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

    outliers_fractions += value_regularization
    metacells_fractions += value_regularization

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
    most_similar: Union[str, ut.Vector, None] = "most_similar",
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
    combined_per_cell = metacell_per_cell.copy()
    outliers_mask: Optional[ut.NumpyVector] = None
    if most_similar is not None:
        most_similar_per_cell = ut.get_o_numpy(adata, most_similar, formatter=ut.groups_description)
        outliers_mask = metacell_per_cell < 0
        assert outliers_mask is not None
        assert np.all(most_similar_per_cell[outliers_mask] >= 0)
        assert np.all(most_similar_per_cell[~outliers_mask] < 0)
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
        [0] + [len(cell_gene_indices) for cell_gene_indices, _cell_fold_factors in results], dtype="int64"
    )
    np.cumsum(indptr, out=indptr)

    assert data.dtype == "float32"
    assert indices.dtype == "int32"
    assert indptr.dtype == "int64"

    fold_factors_per_gene_per_cell = sp.csr_matrix((data, indices, indptr), shape=adata.shape)
    fold_factors_per_gene_per_cell.has_sorted_indices = True
    fold_factors_per_gene_per_cell.has_canonical_format = True
    ut.set_vo_data(adata, "deviant_fold", fold_factors_per_gene_per_cell)


@ut.logged()
def _compute_cell_deviant_folds(
    *,
    cell_index: int,
    umis_per_gene_per_cell: ut.Matrix,
    fraction_per_gene_per_metacell: ut.Matrix,
    umis_per_gene_per_metacell: ut.Matrix,
    metacell_per_cell: ut.NumpyVector,
    outliers_mask: Optional[ut.NumpyVector],
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
    if outliers_mask is not None and outliers_mask[cell_index]:
        total_umis_per_gene = actual_umis_per_gene + metacell_umis_per_gene
    else:
        total_umis_per_gene = metacell_umis_per_gene

    gene_indices = np.where(total_umis_per_gene >= min_gene_total)[0].astype("int32")
    ut.log_calc("nnz_fold_genes_count", len(gene_indices))
    return (gene_indices.astype("int32"), fold_factors[gene_indices].astype("float32"))


@ut.logged()
@ut.timed_call()
def compute_inner_folds(
    *,
    adata: AnnData,
    gdata: AnnData,
    group: Union[str, ut.Vector] = "metacell",
) -> None:
    """
    Given ``adata`` with computed ``deviant_fold`` for each gene for each cell, set in ``inner_fold`` in ``gdata``, for
    each gene for each metacell the ``deviant_fold`` with the maximal absolute value.
    """
    group_per_cell = ut.get_o_numpy(adata, group)
    deviant_fold_per_gene_per_cell = ut.get_vo_proper(adata, "deviant_fold")

    @ut.timed_call()
    def _single_metacell_inner_folds(metacell_index: int) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
        return _compute_metacell_inner_folds(
            metacell_index=metacell_index,
            group_per_cell=group_per_cell,
            deviant_fold_per_gene_per_cell=deviant_fold_per_gene_per_cell,
        )

    results = list(ut.parallel_map(_single_metacell_inner_folds, gdata.n_obs))

    data = np.concatenate([metacell_fold_factors for _metacell_gene_indices, metacell_fold_factors in results])
    indices = np.concatenate([metacell_gene_indices for metacell_gene_indices, _metacell_fold_factors in results])
    assert len(data) == len(indices)
    indptr = np.array(
        [0] + [len(metacell_gene_indices) for metacell_gene_indices, _metacell_fold_factors in results], dtype="int64"
    )
    np.cumsum(indptr, out=indptr)

    assert data.dtype == "float32"
    assert indices.dtype == "int32"
    assert indptr.dtype == "int64"

    fold_factors_per_gene_per_metacell = sp.csr_matrix((data, indices, indptr), shape=gdata.shape)
    fold_factors_per_gene_per_metacell.has_sorted_indices = True
    fold_factors_per_gene_per_metacell.has_canonical_format = True
    ut.set_vo_data(gdata, "inner_fold", fold_factors_per_gene_per_metacell)


@ut.logged()
def _compute_metacell_inner_folds(
    *,
    metacell_index: int,
    group_per_cell: ut.NumpyVector,
    deviant_fold_per_gene_per_cell: ut.ProperMatrix,
) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
    cells_mask = group_per_cell == metacell_index
    cells_count = np.sum(cells_mask)
    assert cells_count > 0
    genes_count = deviant_fold_per_gene_per_cell.shape[1]

    tmp_deviant_fold_per_gene_per_cell = ut.to_numpy_matrix(deviant_fold_per_gene_per_cell[cells_mask, :], copy=True)
    tmp_deviant_fold_per_gene_per_cell = ut.to_layout(tmp_deviant_fold_per_gene_per_cell, layout="column_major")
    assert tmp_deviant_fold_per_gene_per_cell.shape == (cells_count, genes_count)

    negative_folds_mask = tmp_deviant_fold_per_gene_per_cell < 0
    tmp_deviant_fold_per_gene_per_cell[negative_folds_mask] *= -1

    max_index_per_gene = np.argmax(tmp_deviant_fold_per_gene_per_cell, axis=0)
    assert len(max_index_per_gene) == genes_count

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

    for type_name, type_gene_normalized_variances in list(
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

    fold_factor_per_gene_per_outlier = sp.csr_matrix(fold_factor_per_gene_per_outlier)
    fold_factor_per_gene_per_outlier.has_sorted_indices = True
    fold_factor_per_gene_per_outlier.has_canonical_format = True
    ut.set_vo_data(adata, f"{most_similar}_fold", fold_factor_per_gene_per_outlier)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def count_significant_inner_folds(
    adata: AnnData,
    *,
    min_gene_fold_factor: float = pr.significant_gene_fold_factor,
) -> None:
    """
    Given grouped (metacells) data, count for each gene in how many metacells there is at least one cell
    with a fold factor above some threshold.

    **Input**

    Annotated ``adata``, where the observations are metacells and the variables are genes, with an ``inner_fold``
    layer (as computed by ``compute_inner_folds``).

    **Returns**

    Sets the ``significant_inner_folds_count`` annotation, counting for each gene the number of metacells
    where the ``inner_fold`` is at least ``min_gene_fold_factor`` (default: {min_gene_fold_factor}), that is,
    where at least one cell in the metacell has a high fold factor for the gene's expression compared to the
    estimated overall gene expression in the metacell.
    """
    inner_fold_per_gene_per_metacell = ut.get_vo_proper(adata, "inner_fold", layout="column_major")
    significant_per_gene_per_metacell = inner_fold_per_gene_per_metacell >= min_gene_fold_factor  # type: ignore
    significant_per_gene = ut.sum_per(significant_per_gene_per_metacell, per="column")  # type: ignore
    ut.set_v_data(adata, "significant_inner_folds_count", significant_per_gene, formatter=ut.sizes_description)
