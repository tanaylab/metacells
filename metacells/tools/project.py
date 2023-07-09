"""
Project
-------
"""

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import scipy.sparse as sp  # type: ignore
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    "compute_projection_weights",
    "compute_projected_fractions",
    "convey_atlas_to_query",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_projection_weights(
    *,
    adata: AnnData,
    qdata: AnnData,
    from_atlas_layer: str = "corrected_fraction",
    from_query_layer: str = "corrected_fraction",
    to_query_layer: str = "projected_fraction",
    log_data: bool = pr.project_log_data,
    fold_regularization: float = pr.project_fold_regularization,
    min_significant_gene_umis: float = pr.project_min_significant_gene_umis,
    max_consistency_fold_factor: float = pr.project_max_consistency_fold_factor,
    candidates_count: int = pr.project_candidates_count,
    min_candidates_fraction: float = pr.project_min_candidates_fraction,
    min_usage_weight: float = pr.project_min_usage_weight,
    second_anchor_indices: Optional[List[int]] = None,
    reproducible: bool,
) -> ut.CompressedMatrix:
    """
    Compute the weights and results of projecting a query onto an atlas.

    **Input**

    Annotated query ``qdata`` and atlas ``adata``, where the observations are cells and the variables are genes. The
    atlas should contain ``from_atlas_layer`` (default: {from_atlas_layer}) containing gene fractions, and the query
    should similarly contain ``from_query_layer`` (default: {from_query_layer}) containing gene fractions.

    **Returns**

    A CSR matrix whose rows are query metacells and columns are atlas metacells, where each entry is the weight of the
    atlas metacell in the projection of the query metacells. The sum of weights in each row (that is, for a single query
    metacell) is 1. The weighted sum of the atlas metacells using these weights is the "projected" image of the query
    metacell onto the atlas.

    In addition, sets the following annotations in ``qdata``:

    Observation (Cell) Annotations
        ``similar``
            A boolean mask indicating whether the query metacell is similar to its projection onto the atlas. If
            ``False`` the metacells is said to be "dissimilar", which may indicate the query contains cell states that
            do not appear in the atlas.

    Observation-Variable (Cell-Gene) Annotations
        ``to_query_layer`` (default: {to_query_layer})
            A matrix of gene fractions describing the "projected" image of the query metacell onto the atlas. This
            projection is a weighted average of some atlas metacells (using the computed weights returned by this
            function).

    **Computation Parameters**

    0. All fold computations (log2 of the ratio between gene fractions) use the ``fold_regularization`` (default:
       {fold_regularization}).

    For each query metacell:

    1. Correlate the metacell with all the atlas metacells, and pick the highest-correlated one as the "anchor".
       If ``second_anchor_indices`` is not ``None``, then the ``qdata`` must contain only a single query metacell, and
       is expected to contain a ``projected`` per-observation-per-variable matrix containing the projected image of this
       query metacell on the atlas using a single anchor. The code will compute the residual of the query and the atlas
       relative to this projection and pick a second atlas anchor whose residuals are the most correlated to the query
       metacell's residuals. If ``reproducible``, a slower (still parallel) but reproducible algorithm will be used.

    2. Consider (for each anchor) the ``candidates_count`` (default: {candidates_count}) candidate metacells with the
       highest correlation with the query metacell.

    3. Keep as candidates only atlas metacells whose maximal gene fold factor compared to the anchor(s) is at most
       ``max_consistency_fold_factor`` (default: {max_consistency_fold_factor}). Keep at least
       ``min_candidates_fraction`` (default: {min_candidates_fraction}) of the original candidates even if they are less
       consistent. For this computation, Ignore the fold factors of genes whose sum of UMIs in the anchor(s) and the
       candidate metacells is less than ``min_significant_gene_umis`` (default: {min_significant_gene_umis}).

    4. Compute the non-negative weights (with a sum of 1) of the selected candidates that give the best projection of
       the query metacells onto the atlas. If ``log_data`` (default: {log_data}), try to fit the log (base 2) of the
       fractions, otherwise, try to fit the fractions themselves. Since the algorithm for computing these weights rarely
       produces an exact 0 weight, reduce all weights less than the ``min_usage_weight`` (default: {min_usage_weight})
       to zero. If ``second_anchor_indices`` is not ``None``, it is set to the list of indices of the used atlas
       metacells candidates correlated with the second anchor.
    """
    prepared_arguments = _project_query_atlas_data_arguments(
        adata=adata,
        qdata=qdata,
        from_atlas_layer=from_atlas_layer,
        from_query_layer=from_query_layer,
        to_query_layer=to_query_layer,
        log_data=log_data,
        fold_regularization=fold_regularization,
        min_significant_gene_umis=min_significant_gene_umis,
        max_consistency_fold_factor=max_consistency_fold_factor,
        candidates_count=candidates_count,
        min_candidates_fraction=min_candidates_fraction,
        min_usage_weight=min_usage_weight,
        second_anchor_indices=second_anchor_indices,
        reproducible=reproducible,
    )

    @ut.timed_call()
    def _project_single(query_metacell_index: int) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
        return _project_single_metacell(
            query_metacell_index=query_metacell_index,
            **prepared_arguments,
        )

    if ut.is_main_process():
        results = ut.parallel_map(_project_single, qdata.n_obs)
    else:
        results = [_project_single(query_metacell_index) for query_metacell_index in range(qdata.n_obs)]

    indices = np.concatenate([result[0] for result in results], dtype="int32")
    data = np.concatenate([result[1] for result in results], dtype="float32")

    atlas_used_sizes = [len(result[0]) for result in results]
    atlas_used_sizes.insert(0, 0)
    indptr = np.cumsum(np.array(atlas_used_sizes))

    return sp.csr_matrix((data, indices, indptr), shape=(qdata.n_obs, adata.n_obs))


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_projected_fractions(
    *,
    adata: AnnData,
    qdata: AnnData,
    from_atlas_layer: str = "corrected_fraction",
    to_query_layer: str = "projected_fraction",
    log_data: bool = pr.project_log_data,
    fold_regularization: float = pr.project_fold_regularization,
    weights: ut.ProperMatrix,
) -> None:
    """
    Compute the projected image of a query on an atlas.

    **Input**

    Annotated query ``qdata`` and atlas ``adata``, where the observations are cells and the variables are genes. The
    atlas should contain ``from_atlas_layer`` (default: {from_atlas_layer}) containing gene fractions.

    **Returns**

    Sets ``to_query_layer`` (default: {to_query_layer}) in the query containing the gene fractions of the projection of
    the atlas fractions using the ``weights`` matrix.

    .. note::

        It is important to use the same ``log_data`` value as that given to ``compute_projection_weights`` to compute
        the weights (default: {log_data}).
    """
    assert fold_regularization > 0

    atlas_fractions = ut.get_vo_proper(adata, from_atlas_layer, layout="row_major")

    if log_data:
        atlas_log_fractions = ut.to_numpy_matrix(atlas_fractions, copy=True)
        atlas_log_fractions += fold_regularization
        atlas_log_fractions = np.log2(atlas_log_fractions, out=atlas_log_fractions)
        projected_fractions = weights @ atlas_log_fractions
        projected_fractions = np.power(2.0, projected_fractions, out=projected_fractions)
        projected_fractions -= fold_regularization

    else:
        projected_fractions = weights @ atlas_fractions  # type: ignore

    assert projected_fractions.shape == qdata.shape
    projected_fractions = ut.fraction_by(projected_fractions, by="row").astype("float32")  # type: ignore
    ut.set_vo_data(qdata, to_query_layer, sp.csr_matrix(projected_fractions))


def _project_query_atlas_data_arguments(
    adata: AnnData,
    qdata: AnnData,
    from_atlas_layer: str,
    from_query_layer: str,
    to_query_layer: str,
    log_data: bool,
    fold_regularization: float,
    min_significant_gene_umis: float,
    max_consistency_fold_factor: float,
    candidates_count: int,
    min_candidates_fraction: float,
    min_usage_weight: float,
    second_anchor_indices: Optional[List[int]],
    reproducible: bool,
) -> Dict[str, Any]:
    assert fold_regularization > 0
    assert candidates_count > 0
    assert 0 <= min_candidates_fraction <= 1.0
    assert min_usage_weight >= 0
    assert max_consistency_fold_factor >= 0
    assert np.all(adata.var_names == qdata.var_names)

    atlas_umis = ut.get_vo_proper(adata, "total_umis", layout="row_major")

    atlas_fractions = ut.get_vo_proper(adata, from_atlas_layer, layout="row_major")
    query_fractions = ut.get_vo_proper(qdata, from_query_layer, layout="row_major")

    atlas_fractions = ut.to_numpy_matrix(atlas_fractions, copy=True)
    query_fractions = ut.to_numpy_matrix(query_fractions, copy=True)

    if second_anchor_indices is not None:
        assert qdata.n_obs == 1
        query_single_fractions = ut.to_numpy_vector(ut.get_vo_proper(qdata, to_query_layer))

        query_residual_fractions = query_fractions - query_single_fractions[np.newaxis, :]
        query_residual_fractions[query_residual_fractions < 0] = 0

        atlas_residual_fractions = atlas_fractions - ut.to_numpy_vector(query_residual_fractions)[np.newaxis, :]
        atlas_residual_fractions[atlas_residual_fractions < 0] = 0

        if log_data:
            atlas_residual_fractions += fold_regularization
            query_residual_fractions += fold_regularization

            atlas_residual_data = np.log2(atlas_residual_fractions, out=atlas_residual_fractions)
            query_residual_data = np.log2(query_residual_fractions, out=query_residual_fractions)

        else:
            atlas_residual_data = atlas_residual_fractions
            query_residual_data = query_residual_fractions

        query_atlas_corr_residual: Optional[ut.NumpyMatrix] = ut.cross_corrcoef_rows(
            query_residual_data, atlas_residual_data, reproducible=reproducible
        )

    else:
        query_atlas_corr_residual = None

    atlas_log_fractions = atlas_fractions + fold_regularization
    atlas_log_fractions = np.log2(atlas_log_fractions, out=atlas_log_fractions)

    query_log_fractions = query_fractions + fold_regularization
    query_log_fractions = np.log2(query_log_fractions, out=query_log_fractions)

    if log_data:
        atlas_data = atlas_log_fractions
        query_data = query_log_fractions

    else:
        atlas_data = atlas_fractions
        query_data = query_fractions

    query_atlas_corr = ut.cross_corrcoef_rows(query_log_fractions, atlas_log_fractions, reproducible=reproducible)

    return {
        "atlas_umis": atlas_umis,
        "query_atlas_corr": query_atlas_corr,
        "atlas_data": atlas_data,
        "atlas_log_fractions": atlas_log_fractions,
        "query_data": query_data,
        "candidates_count": candidates_count,
        "min_candidates_fraction": min_candidates_fraction,
        "min_significant_gene_umis": min_significant_gene_umis,
        "min_usage_weight": min_usage_weight,
        "max_consistency_fold_factor": max_consistency_fold_factor,
        "second_anchor_indices": second_anchor_indices,
        "query_atlas_corr_residual": query_atlas_corr_residual,
    }


@ut.logged()
def _project_single_metacell(  # pylint: disable=too-many-statements,too-many-branches
    *,
    query_metacell_index: int,
    atlas_umis: ut.Matrix,
    query_atlas_corr: ut.NumpyMatrix,
    atlas_data: ut.NumpyMatrix,
    atlas_log_fractions: ut.NumpyMatrix,
    query_data: ut.NumpyMatrix,
    candidates_count: int,
    min_candidates_fraction: float,
    min_significant_gene_umis: float,
    min_usage_weight: float,
    max_consistency_fold_factor: float,
    second_anchor_indices: Optional[List[int]],
    query_atlas_corr_residual: Optional[ut.NumpyMatrix],
) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
    query_metacell_data = query_data[query_metacell_index, :]

    query_metacell_atlas_correlations = query_atlas_corr[query_metacell_index, :]
    query_metacell_atlas_order = np.argsort(-query_metacell_atlas_correlations)

    atlas_anchor_index = query_metacell_atlas_order[0]
    ut.log_calc("atlas_anchor_index", atlas_anchor_index)
    atlas_anchor_log_fractions = atlas_log_fractions[atlas_anchor_index, :]
    atlas_anchor_umis = ut.to_numpy_vector(atlas_umis[atlas_anchor_index, :])

    atlas_candidates_consistency = [0.0]
    atlas_candidates_indices = [atlas_anchor_index]

    position = 1
    while len(atlas_candidates_indices) < candidates_count and position < len(query_metacell_atlas_order):
        atlas_metacell_index = query_metacell_atlas_order[position]
        position += 1
        atlas_metacell_log_fractions = atlas_log_fractions[atlas_metacell_index, :]
        atlas_metacell_consistency_fold_factors = np.abs(atlas_metacell_log_fractions - atlas_anchor_log_fractions)
        atlas_metacell_umis = ut.to_numpy_vector(atlas_umis[atlas_metacell_index, :])
        atlas_metacell_significant_genes_mask = atlas_metacell_umis + atlas_anchor_umis >= min_significant_gene_umis
        if np.any(atlas_metacell_significant_genes_mask):
            atlas_metacell_consistency = np.max(
                atlas_metacell_consistency_fold_factors[atlas_metacell_significant_genes_mask]
            )
            atlas_candidates_consistency.append(atlas_metacell_consistency)
            atlas_candidates_indices.append(atlas_metacell_index)

    sorted_locations = list(np.argsort(np.array(atlas_candidates_consistency)))
    min_candidates_count = candidates_count * min_candidates_fraction

    while (
        len(sorted_locations) > min_candidates_count
        and atlas_candidates_consistency[sorted_locations[-1]] > max_consistency_fold_factor
    ):
        sorted_locations.pop()

    atlas_candidate_indices_set = set([atlas_anchor_index])
    for location in sorted_locations:
        atlas_metacell_index = atlas_candidates_indices[location]
        atlas_candidate_indices_set.add(atlas_metacell_index)

    ut.log_calc("atlas_candidates", len(atlas_candidate_indices_set))

    if query_atlas_corr_residual is None:
        atlas_candidate_indices = np.array(sorted(atlas_candidate_indices_set))
    else:
        query_metacell_atlas_residual_correlations = query_atlas_corr_residual[query_metacell_index, :]
        query_metacell_atlas_residual_order = np.argsort(-query_metacell_atlas_residual_correlations)

        atlas_secondary_anchor_index = query_metacell_atlas_residual_order[0]
        ut.log_calc("atlas_secondary_anchor_index", atlas_secondary_anchor_index)
        atlas_secondary_anchor_log_fractions = atlas_log_fractions[atlas_anchor_index, :]
        atlas_secondary_anchor_umis = ut.to_numpy_vector(atlas_umis[atlas_secondary_anchor_index, :])

        atlas_secondary_candidates_consistency = [0.0]
        atlas_secondary_candidates_indices = [atlas_secondary_anchor_index]

        position = 1
        while len(atlas_secondary_candidates_indices) < candidates_count and position < len(
            query_metacell_atlas_residual_order
        ):
            atlas_metacell_index = query_metacell_atlas_residual_order[position]
            position += 1
            atlas_metacell_log_fractions = atlas_log_fractions[atlas_metacell_index, :]
            atlas_metacell_consistency_fold_factors = np.abs(
                atlas_metacell_log_fractions - atlas_secondary_anchor_log_fractions
            )
            atlas_metacell_umis = ut.to_numpy_vector(atlas_umis[atlas_metacell_index, :])
            atlas_metacell_significant_genes_mask = (
                atlas_metacell_umis + atlas_secondary_anchor_umis >= min_significant_gene_umis
            )
            if np.any(atlas_metacell_significant_genes_mask):
                atlas_metacell_consistency = np.max(
                    atlas_metacell_consistency_fold_factors[atlas_metacell_significant_genes_mask]
                )
                atlas_secondary_candidates_consistency.append(atlas_metacell_consistency)
                atlas_secondary_candidates_indices.append(atlas_metacell_index)

        sorted_secondary_locations = list(np.argsort(np.array(atlas_secondary_candidates_consistency)))

        while (
            len(sorted_secondary_locations) > min_candidates_count
            and atlas_secondary_candidates_consistency[sorted_secondary_locations[-1]] > max_consistency_fold_factor
        ):
            sorted_secondary_locations.pop()

        atlas_secondary_candidate_indices_set = set([atlas_secondary_anchor_index])
        for location in sorted_secondary_locations:
            atlas_metacell_index = atlas_secondary_candidates_indices[location]
            atlas_secondary_candidate_indices_set.add(atlas_metacell_index)

        ut.log_calc("atlas_secondary_candidates", len(atlas_candidate_indices_set))

        atlas_candidate_indices = np.array(sorted(atlas_candidate_indices_set | atlas_secondary_candidate_indices_set))

    atlas_candidates_data = atlas_data[atlas_candidate_indices, :]
    represent_result = ut.represent(query_metacell_data, atlas_candidates_data)
    assert represent_result is not None
    atlas_candidate_weights = represent_result[1]
    atlas_candidate_weights[atlas_candidate_weights < min_usage_weight] = 0
    atlas_candidate_weights[:] /= np.sum(atlas_candidate_weights)

    atlas_used_mask = atlas_candidate_weights > 0

    atlas_used_indices = atlas_candidate_indices[atlas_used_mask].astype("int32")
    ut.log_return("atlas_used_indices", atlas_used_indices)

    if second_anchor_indices is not None:
        for atlas_metacell_index in atlas_used_indices:
            if atlas_metacell_index not in atlas_candidate_indices_set:
                second_anchor_indices.append(atlas_metacell_index)

    atlas_used_weights = atlas_candidate_weights[atlas_used_mask]
    atlas_used_weights = atlas_used_weights.astype("float32")
    ut.log_return("atlas_used_weights", atlas_used_weights)

    return (atlas_used_indices, atlas_used_weights)


@ut.logged()
@ut.timed_call()
def convey_atlas_to_query(
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
    Convey the value of a property from per-observation atlas data to per-observation query data.

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
        atlas_metacell_weights = ut.to_numpy_vector(weights[query_metacell_index, :])
        used_atlas_metacells_mask = atlas_metacell_weights > 0
        assert np.any(used_atlas_metacells_mask)
        used_atlas_metacell_weights = ut.to_numpy_vector(atlas_metacell_weights[used_atlas_metacells_mask])
        used_atlas_metacell_values = property_of_atlas_metacells[used_atlas_metacells_mask]
        property_of_query_metacells.append(method(used_atlas_metacell_weights, used_atlas_metacell_values))

    query_property_data = np.array(property_of_query_metacells, dtype=property_of_atlas_metacells.dtype)
    ut.set_o_data(qdata, to_property_name, query_property_data)
