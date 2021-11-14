"""
Project
-------
"""

from typing import Tuple
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore
from scipy import sparse  # type: ignore

import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    "project_query_onto_atlas",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def project_query_onto_atlas(
    *,
    adata: AnnData,
    qdata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    project_fold_normalization: float = pr.project_fold_normalization,
    project_candidates_count: int = pr.project_candidates_count,
    project_min_usage_weight: float = pr.project_min_usage_weight,
    project_min_consistency_weight: float = pr.project_min_consistency_weight,
    project_max_consistency_fold: float = pr.project_max_consistency_fold,
    project_max_projection_fold: float = pr.project_max_projection_fold,
) -> ut.CompressedMatrix:
    """
    Project query metacells onto atlas metacells.

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

    Observation (Cell) Annotations
        ``charted``
            A boolean mask indicating whether the query metacell is meaningfully projected onto the atlas. If ``False``
            the metacells is said to be "uncharted", which may indicate the query contains cell states that do not
            appear in the atlas.

    Observation-Variable (Cell-Gene) Annotations
        ``projected``
            A matrix of UMIs where the sum of UMIs for each query metacell is the same as the sum of ``what`` UMIs,
            describing the "projected" image of the query metacell onto the atlas.


    **Computation Parameters**

    0. All fold computations (log2 of the ratio between gene expressions as a fraction of the total UMIs) use the
       ``project_fold_normalization`` (default: {project_fold_normalization}).

    For each query metacell:

    1. Find the ``project_candidates_count`` (default: {project_candidates_count}) atlas metacells with the lowest
       maximal gene fold factor relative to the query metacell.

    2. Compute the non-negative weights (with a sum of 1) of the atlas candidates that give the best projection of the
       query metacells onto the atlas. Since the algorithm for computing these weights rarely produces an exact 0
       weight, reduce all weights less than the ``project_min_usage_weight`` (default: {project_min_usage_weight}) to
       zero.

    4. Consider for each gene the fold factor between the query metacell and its projection. If this is above
       ``project_max_projection_fold`` (default: {project_max_projection_fold}), declare the query metacell as
       uncharted.

    5. Consider for each gene the fold factor between its maximal and minimal expression in the atlas candidates whose
       weight is at least ``project_min_consistency_weight`` (default: {project_min_consistency_weight}). If this is
       is above ``project_max_consistency_fold`` (default: {project_max_consistency_fold}), declare the query metacell
       as uncharted.
    """
    assert project_fold_normalization > 0
    assert project_candidates_count > 0
    assert project_min_usage_weight >= 0
    assert project_min_consistency_weight >= 0
    assert project_max_consistency_fold >= 0

    atlas_umis = ut.get_vo_proper(adata, what, layout="row_major")
    query_umis = ut.get_vo_proper(qdata, what, layout="row_major")

    atlas_fractions = ut.to_numpy_matrix(ut.fraction_by(atlas_umis, by="row"))
    query_fractions = ut.to_numpy_matrix(ut.fraction_by(query_umis, by="row"))

    atlas_log_fractions = np.log2(atlas_fractions)
    query_log_fractions = np.log2(query_fractions)

    atlas_log_fractions += project_fold_normalization
    query_log_fractions += project_fold_normalization

    @ut.timed_call("project_single_metacell")
    def _project_single(query_metacell_index: int) -> Tuple[bool, ut.NumpyVector, ut.NumpyVector]:
        return _project_single_metacell(
            atlas_fractions=atlas_fractions,
            query_fractions=query_fractions,
            atlas_log_fractions=atlas_log_fractions,
            query_log_fractions=query_log_fractions,
            project_fold_normalization=project_fold_normalization,
            project_candidates_count=project_candidates_count,
            project_min_usage_weight=project_min_usage_weight,
            project_min_consistency_weight=project_min_consistency_weight,
            project_max_consistency_fold=project_max_consistency_fold,
            project_max_projection_fold=project_max_projection_fold,
            query_metacell_index=query_metacell_index)
    results = list(ut.parallel_map(_project_single, qdata.n_obs))

    charted = np.array([result[0] for result in results])
    ut.set_o_data(qdata, "charted", charted)

    data = np.concatenate([result[2] for result in results], dtype="float32")
    indices = np.concatenate([result[1] for result in results], dtype="int32")

    atlas_used_sizes = [len(result[1]) for result in results]
    atlas_used_sizes.insert(0, 0)
    indptr = np.cumsum(np.array(atlas_used_sizes))

    sparse_weights = sparse.csr_matrix((data, indices, indptr), shape=atlas_fractions.shape)

    dense_weights = ut.to_numpy_matrix(sparse_weights)
    projected = dense_weights @ atlas_fractions.transpose()
    assert projected.shape == query_fractions.shape
    query_total_umis = ut.sum_per(query_umis, per="row")
    projected *= query_total_umis[:, np.newaxis]
    ut.set_vo_data(qdata, "projected", projected)

    return sparse_weights


@ut.logged()
def _project_single_metacell(
    *,
    atlas_fractions: ut.NumpyMatrix,
    query_fractions: ut.NumpyMatrix,
    atlas_log_fractions: ut.NumpyMatrix,
    query_log_fractions: ut.NumpyMatrix,
    project_fold_normalization: float,
    project_candidates_count: int,
    project_min_usage_weight: float,
    project_min_consistency_weight: float,
    project_max_consistency_fold: float,
    project_max_projection_fold: float,
    query_metacell_index: int
) -> Tuple[bool, ut.NumpyVector, ut.NumpyVector]:
    query_metacell_fractions = query_fractions[query_metacell_index, :]
    query_metacell_log_fractions = query_log_fractions[query_metacell_index, :]

    if project_candidates_count < atlas_log_fractions.shape[0]:
        query_atlas_fold_factors = atlas_log_fractions - query_metacell_log_fractions[np.newaxis, :]
        query_atlas_fold_factors = np.abs(query_atlas_fold_factors, out=query_atlas_fold_factors)
        query_atlas_max_fold_factors = ut.max_per(query_atlas_fold_factors, per="row")
        query_atlas_max_fold_factors *= -1
        atlas_candidate_indices = np.argpartition(query_atlas_fold_factors, project_candidates_count)
        atlas_candidate_fractions = atlas_fractions[atlas_candidate_indices]
    else:
        atlas_candidate_indices = np.arange(atlas_log_fractions.shape[0])
        atlas_candidate_fractions = atlas_fractions

    represent_result = ut.represent(query_metacell_fractions, atlas_candidate_fractions)
    assert represent_result is not None
    atlas_candidate_weights = represent_result[1]
    atlas_candidate_weights[atlas_candidate_weights < project_min_usage_weight] = 0
    atlas_candidate_weights[atlas_candidate_weights < project_min_usage_weight] /= np.sum(atlas_candidate_weights)

    atlas_candidate_used_mask = atlas_candidate_weights > 0
    atlas_used_weights = atlas_candidate_weights[atlas_candidate_used_mask]
    atlas_used_indices = atlas_candidate_indices[atlas_candidate_used_mask]

    projected_fractions = atlas_candidate_fractions @ atlas_candidate_weights
    projected_log_fractions = np.log2(projected_fractions, out=projected_fractions)
    projected_log_fractions += project_fold_normalization

    query_projected_fold_factors = projected_log_fractions - query_metacell_log_fractions
    query_projected_fold_factors = np.abs(query_projected_fold_factors, out=query_projected_fold_factors)
    query_max_projected_fold_factors = np.max(query_projected_fold_factors)
    charted = query_max_projected_fold_factors > project_max_projection_fold

    if charted:
        atlas_consistency_mask = atlas_used_weights >= project_min_consistency_weight
        atlas_consistency_indices = atlas_used_indices[atlas_consistency_mask]
        atlas_consistency_log_fractions = \
            ut.to_layout(atlas_log_fractions[atlas_consistency_indices], layout="column_major")

        atlas_consistency_min_log_fractions = ut.min_per(atlas_consistency_log_fractions, per="column")
        atlas_consistency_max_log_fractions = ut.max_per(atlas_consistency_log_fractions, per="column")
        atlas_consistency_fold_factors = atlas_consistency_max_log_fractions - atlas_consistency_min_log_fractions
        atlas_consistency_max_fold_factor = np.max(atlas_consistency_fold_factors)
        charted = atlas_consistency_max_fold_factor <= project_max_consistency_fold

    atlas_used_indices = atlas_used_indices.astype("int32")
    atlas_used_weights = atlas_used_weights.astype("float32")
    ut.log_return("charted", charted)
    ut.log_return("atlas_used_indices", atlas_used_indices)
    ut.log_return("atlas_used_weights", atlas_used_weights)
    return (charted, atlas_used_indices, atlas_used_weights)
