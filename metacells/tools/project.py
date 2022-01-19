"""
Project
-------
"""

from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore
from scipy import sparse  # type: ignore

import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    "project_query_onto_atlas",
    "find_systematic_genes",
    "project_atlas_to_query",
    "find_biased_genes",
    "compute_query_projection",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def project_query_onto_atlas(
    what: Union[str, ut.Matrix] = "__x__",
    *,
    adata: AnnData,
    qdata: AnnData,
    atlas_total_umis: Optional[ut.Vector] = None,
    query_total_umis: Optional[ut.Vector] = None,
    project_log_data: bool = pr.project_log_data,
    fold_normalization: float = pr.project_fold_normalization,
    min_significant_gene_value: float = pr.project_min_significant_gene_value,
    candidates_count: int = pr.project_candidates_count,
    min_usage_weight: float = pr.project_min_usage_weight,
    abs_folds: bool = pr.project_abs_folds,
) -> ut.CompressedMatrix:
    """
    Project query metacells onto atlas metacells.

    **Input**

    Annotated query ``qdata`` and atlas ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation annotation
    containing such a matrix.

    Typically this data excludes any genes having a systematic difference between the query and the atlas, e.g. genes
    detected by by :py:func:`metacells.tools.project.find_systematic_genes`.

    **Returns**

    A matrix whose rows are query metacells and columns are atlas metacells, where each entry is the weight of the atlas
    metacell in the projection of the query metacells. The sum of weights in each row (that is, for a single query
    metacell) is 1. The weighted sum of the atlas metacells using these weights is the "projected" image of the query
    metacell onto the atlas.

    In addition, sets the following annotations in ``qdata``:

    Observation (Cell) Annotations
        ``similar``
            A boolean mask indicating whether the query metacell is similar to its projection onto the atlas. If
            ``False`` the metacells is said to be "dissimilar", which may indicate the query contains cell states that
            do not appear in the atlas.

    **Computation Parameters**

    0. All fold computations (log2 of the ratio between gene expressions as a fraction of the total UMIs) use the
       ``fold_normalization`` (default: {fold_normalization}). Fractions are computed based on the total UMIs, unless
       ``atlas_total_umis`` and/or ``query_total_umis`` are specified.

    For each query metacell:

    1. Find the ``candidates_count`` (default: {candidates_count}) atlas metacells with the lowest maximal gene fold
       factor relative to the query metacell. Ignore the fold factors of genes whose sum of UMIs in the query and
       the atlas metacells is less than ``min_significant_gene_value`` (default: {min_significant_gene_value}).

    2. Compute the non-negative weights (with a sum of 1) of the atlas candidates that give the best projection of the
       query metacells onto the atlas. Since the algorithm for computing these weights rarely produces an exact 0
       weight, reduce all weights less than the ``min_usage_weight`` (default: {min_usage_weight}) to
       zero. If ``project_log_data`` (default: {project_log_data}), compute the match on the log of the data instead
       of the actual data.
    """
    assert fold_normalization > 0
    assert candidates_count > 0
    assert min_usage_weight >= 0
    assert np.all(adata.var_names == qdata.var_names)

    atlas_umis = ut.get_vo_proper(adata, what, layout="row_major")
    query_umis = ut.get_vo_proper(qdata, what, layout="row_major")

    if atlas_total_umis is None:
        atlas_total_umis = ut.sum_per(atlas_umis, per="row")
    atlas_total_umis = ut.to_numpy_vector(atlas_total_umis)

    if query_total_umis is None:
        query_total_umis = ut.sum_per(query_umis, per="row")
    query_total_umis = ut.to_numpy_vector(query_total_umis)

    atlas_fractions = ut.to_numpy_matrix(ut.fraction_by(atlas_umis, by="row", sums=atlas_total_umis))
    query_fractions = ut.to_numpy_matrix(ut.fraction_by(query_umis, by="row", sums=query_total_umis))

    atlas_fractions += fold_normalization
    query_fractions += fold_normalization

    atlas_log_fractions = np.log2(atlas_fractions)
    query_log_fractions = np.log2(query_fractions)

    atlas_fractions -= fold_normalization
    query_fractions -= fold_normalization

    if project_log_data:
        atlas_project_data = atlas_log_fractions
        query_project_data = query_log_fractions
    else:
        atlas_project_data = atlas_fractions
        query_project_data = query_fractions

    @ut.timed_call("project_single_metacell")
    def _project_single(query_metacell_index: int) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
        return _project_single_metacell(
            atlas_umis=atlas_umis,
            query_umis=query_umis,
            atlas_project_data=atlas_project_data,
            query_project_data=query_project_data,
            atlas_log_fractions=atlas_log_fractions,
            query_log_fractions=query_log_fractions,
            candidates_count=candidates_count,
            min_significant_gene_value=min_significant_gene_value,
            min_usage_weight=min_usage_weight,
            abs_folds=abs_folds,
            query_metacell_index=query_metacell_index,
        )

    results = list(ut.parallel_map(_project_single, qdata.n_obs))

    indices = np.concatenate([result[0] for result in results], dtype="int32")
    data = np.concatenate([result[1] for result in results], dtype="float32")

    atlas_used_sizes = [len(result[0]) for result in results]
    atlas_used_sizes.insert(0, 0)
    indptr = np.cumsum(np.array(atlas_used_sizes))

    return sparse.csr_matrix((data, indices, indptr), shape=(qdata.n_obs, adata.n_obs))


@ut.logged()
def _project_single_metacell(
    *,
    atlas_umis: ut.Matrix,
    query_umis: ut.Matrix,
    atlas_project_data: ut.NumpyMatrix,
    query_project_data: ut.NumpyMatrix,
    atlas_log_fractions: ut.NumpyMatrix,
    query_log_fractions: ut.NumpyMatrix,
    candidates_count: int,
    min_significant_gene_value: float,
    min_usage_weight: float,
    abs_folds: bool,
    query_metacell_index: int,
) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
    query_metacell_umis = ut.to_numpy_vector(query_umis[query_metacell_index, :])
    ut.log_calc("query_total_umis", np.sum(query_metacell_umis))
    query_metacell_project_data = query_project_data[query_metacell_index, :]
    query_metacell_log_fractions = query_log_fractions[query_metacell_index, :]

    if candidates_count < atlas_log_fractions.shape[0]:
        query_atlas_fold_factors = atlas_log_fractions - query_metacell_log_fractions[np.newaxis, :]
        query_atlas_fold_factors = np.abs(query_atlas_fold_factors, out=query_atlas_fold_factors)
        total_umis = atlas_umis + query_metacell_umis[np.newaxis, :]
        insignificant_folds_mask = total_umis < min_significant_gene_value
        ut.log_calc("significant_folds_mask", ~insignificant_folds_mask)
        query_atlas_fold_factors[insignificant_folds_mask] = 0.0
        if abs_folds:
            query_atlas_fold_factors = np.abs(query_atlas_fold_factors)
        query_atlas_max_fold_factors = ut.max_per(query_atlas_fold_factors, per="row")
        atlas_candidate_indices = np.argpartition(query_atlas_max_fold_factors, candidates_count)[0:candidates_count]
        atlas_candidate_indices.sort()
        atlas_candidate_project_data = atlas_project_data[atlas_candidate_indices, :]
    else:
        atlas_candidate_indices = np.arange(atlas_log_fractions.shape[0])
        atlas_candidate_project_data = atlas_project_data

    represent_result = ut.represent(query_metacell_project_data, atlas_candidate_project_data)
    assert represent_result is not None
    atlas_candidate_weights = represent_result[1]
    atlas_candidate_weights[atlas_candidate_weights < min_usage_weight] = 0
    atlas_candidate_weights[atlas_candidate_weights < min_usage_weight] /= np.sum(atlas_candidate_weights)

    atlas_candidate_used_mask = atlas_candidate_weights > 0

    atlas_used_indices = atlas_candidate_indices[atlas_candidate_used_mask]
    atlas_used_indices = atlas_used_indices.astype("int32")
    ut.log_return("atlas_used_indices", atlas_used_indices)

    atlas_used_weights = atlas_candidate_weights[atlas_candidate_used_mask]
    atlas_used_weights = atlas_used_weights.astype("float32")
    ut.log_return("atlas_used_weights", atlas_used_weights)

    return (atlas_used_indices, atlas_used_weights)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def find_systematic_genes(
    what: Union[str, ut.Matrix] = "__x__",
    *,
    adata: AnnData,
    qdata: AnnData,
    atlas_total_umis: Optional[ut.Vector] = None,
    query_total_umis: Optional[ut.Vector] = None,
    low_gene_quantile: float = pr.systematic_low_gene_quantile,
    high_gene_quantile: float = pr.systematic_high_gene_quantile,
) -> None:
    """
    Find genes that

    **Input**

    Annotated query ``qdata`` and atlas ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation annotation
    containing such a matrix.

    **Returns**

    A matrix whose rows are query metacells and columns are atlas metacells, where each entry is the weight of the atlas
    metacell in the projection of the query metacells. The sum of weights in each row (that is, for a single query
    metacell) is 1. The weighted sum of the atlas metacells using these weights is the "projected" image of the query
    metacell onto the atlas.

    In addition, sets the following annotations in ``qdata``:

    Variable (Gene) Annotations
        ``systematic_gene``
            A boolean mask indicating whether the gene is systematically higher or lower in the query compared to the
            atlas.

    **Computation Parameters**

    1. Compute the fraction of each gene out of the total UMIs in both the atlas and the query. If ``atlas_total_umis``
       and/or ``query_total_umis`` are given, use them as the basis instead of the sum of the UMIs.

    2. Compute for each gene its ``low_gene_quantile`` (default: {low_gene_quantile}) fraction in the query, and its
       ``high_gene_quantile`` (default: {high_gene_quantile}) fraction in the atlas.

    3. Compute for each gene its standard deviation in the atlas.

    4. Mark as systematic the genes for which the low quantile value in the query is at least the atlas high quantile
       value.

    5. Mark as systematic the genes for which the low quantile value in the atlas is at least the query high quantile
       value.
    """
    assert 0 <= low_gene_quantile <= 1
    assert 0 <= high_gene_quantile <= 1
    assert np.all(adata.var_names == qdata.var_names)

    query_umis = ut.get_vo_proper(qdata, what, layout="row_major")
    atlas_umis = ut.get_vo_proper(adata, what, layout="row_major")

    atlas_fractions = ut.to_numpy_matrix(ut.fraction_by(atlas_umis, by="row", sums=atlas_total_umis))
    query_fractions = ut.to_numpy_matrix(ut.fraction_by(query_umis, by="row", sums=query_total_umis))

    query_fractions = ut.to_layout(query_fractions, layout="column_major")
    atlas_fractions = ut.to_layout(atlas_fractions, layout="column_major")

    query_low_gene_values = ut.quantile_per(query_fractions, low_gene_quantile, per="column")
    atlas_low_gene_values = ut.quantile_per(atlas_fractions, low_gene_quantile, per="column")

    query_high_gene_values = ut.quantile_per(query_fractions, high_gene_quantile, per="column")
    atlas_high_gene_values = ut.quantile_per(atlas_fractions, high_gene_quantile, per="column")

    query_above_atlas = query_low_gene_values > atlas_high_gene_values
    atlas_above_query = atlas_low_gene_values >= query_high_gene_values

    systematic = query_above_atlas | atlas_above_query

    ut.set_v_data(qdata, "systematic_gene", systematic)


@ut.logged()
@ut.timed_call()
def project_atlas_to_query(
    *,
    adata: AnnData,
    qdata: AnnData,
    weights: ut.ProperMatrix,
    property_name: str,
    formatter: Optional[Callable[[Any], Any]] = None,
    to_property_name: Optional[str] = None,
    method: Callable[[ut.Vector, ut.Vector], Any] = ut.highest_weight,
) -> None:
    """
    Project the value of a property from per-observation atlas data to per-observation query data.

    The input annotated ``adata`` is expected to contain a per-observation (cell) annotation named ``property_name``.
    Given the ``weights`` matrix, where each row specifies the weights of the atlas metacells used to project a single
    query metacell, this will generate a new per-observation (group) annotation in ``qdata``, named ``to_property_name``
    (by default, the same as ``property_name``), containing the aggregated value of the property of all the observations
    (cells) that belong to the group.

    The aggregation method (by default, :py:func:`metacells.utilities.computation.highest_weight`) is any function
    taking two array, weights and values, and returning a single value.
    """
    if to_property_name is None:
        to_property_name = property_name

    property_of_atlas_metacells = ut.get_o_numpy(adata, property_name, formatter=formatter)
    property_of_query_metacells = []
    for query_metacell_index in range(qdata.n_obs):
        metacell_weights = ut.to_numpy_vector(weights[query_metacell_index, :])
        metacell_mask = metacell_weights > 0
        metacell_weights = ut.to_numpy_vector(metacell_weights[metacell_mask])
        metacell_values = property_of_atlas_metacells[metacell_mask]
        property_of_query_metacells.append(method(metacell_weights, metacell_values))

    ut.set_o_data(qdata, to_property_name, np.array(property_of_query_metacells))


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def find_biased_genes(
    adata: AnnData,
    *,
    max_projection_fold_factor: float = pr.project_max_projection_fold_factor,
    min_metacells_fraction: float = pr.biased_min_metacells_fraction,
    abs_folds: bool = pr.project_abs_folds,
) -> None:
    """
    Find genes that have a strong bias in the query compared to the atlas.

    **Input**

    Annotated query ``adata`` where the observations are cells and the variables are genes, where ``what`` is a
    per-variable-per-observation matrix or the name of a per-variable-per-observation annotation containing such a
    matrix.

    This should contain a ``projected_fold`` per-variable-per-observation matrix with the fold factor between each query
    metacell and its projected image on the atlas.

    **Returns**

    Sets the following annotations in ``adata``:

    Variable (Gene) Annotations
        ``biased_gene``
            A boolean mask indicating whether the gene has a strong bias in the query compared to the atlas.

    **Computation Parameters**

    1. Count for each such gene the number of query metacells for which the ``projected_fold`` is above
       ``max_projection_fold_factor``. If ``abs_folds`` (default: {abs_folds}), consider the absolute fold factor.

    2. Mark the gene as biased if either count is at least a ``min_metacells_fraction`` (default:
       {min_metacells_fraction}) of the metacells.
    """
    assert max_projection_fold_factor >= 0
    assert 0 <= min_metacells_fraction <= 1

    projected_fold = ut.get_vo_proper(adata, "projected_fold", layout="column_major")
    if abs_folds:
        projected_fold = np.abs(projected_fold)  # type: ignore

    high_projection_folds = ut.to_numpy_matrix(projected_fold > max_projection_fold_factor)  # type: ignore
    ut.log_calc("high_projection_folds", high_projection_folds)

    count_of_genes = ut.sum_per(high_projection_folds, per="column")
    min_count = adata.n_obs * min_metacells_fraction
    mask_of_genes = count_of_genes >= min_count

    ut.log_calc("biased genes", mask_of_genes)
    ut.set_v_data(adata, "biased_gene", mask_of_genes)


@ut.logged()
@ut.timed_call()
def compute_query_projection(
    what: Union[str, ut.Matrix] = "__x__",
    *,
    adata: AnnData,
    qdata: AnnData,
    weights: ut.Matrix,
    atlas_total_umis: Optional[ut.Vector] = None,
    query_total_umis: Optional[ut.Vector] = None,
) -> None:
    """
    Compute the projected image of the query on the atlas.

    **Input**

    Annotated query ``qdata`` and atlas ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation annotation
    containing such a matrix.

    The ``weights`` of the projection where each row is a query metacell, each column is an atlas metacell, and the
    value is the weight of the atlas cell for projecting the metacell, such that the sum of weights in each row
    is one.

    **Returns**

    In addition, sets the following annotations in ``qdata``:

    Observation (Cell) Annotations
        ``projection``
            The number of UMIs of each gene in the projected image of the query to the metacell, if the total number of
            UMIs in the projection is equal to the total number of UMIs in the query metacell.

    **Computation Parameters**

    1. Compute the fraction of each gene in the atlas and the query based on the total UMIs, unless ``atlas_total_umis``
       and/or ``query_total_umis`` are specified.

    2. Compute the projected image of each query metacell on the atlas using the weights.

    3. Convert this image to UMIs count based on the total UMIs of each metacell. Note that if overriding the total
       atlas or query UMIs, this means that the result need not sum to this total.
    """
    assert np.all(adata.var_names == qdata.var_names)

    atlas_umis = ut.get_vo_proper(adata, what, layout="row_major")
    query_umis = ut.get_vo_proper(qdata, what, layout="row_major")

    if atlas_total_umis is None:
        atlas_total_umis = ut.sum_per(atlas_umis, per="row")
    atlas_total_umis = ut.to_numpy_vector(atlas_total_umis)

    if query_total_umis is None:
        query_total_umis = ut.sum_per(query_umis, per="row")
    query_total_umis = ut.to_numpy_vector(query_total_umis)

    atlas_fractions = ut.to_numpy_matrix(ut.fraction_by(atlas_umis, by="row", sums=atlas_total_umis))
    projected_fractions = weights @ atlas_fractions  # type: ignore
    projected_umis = ut.scale_by(projected_fractions, scale=query_total_umis, by="row")
    ut.set_vo_data(qdata, "projected", projected_umis)
