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
from typing import Union

import numpy as np
import scipy.sparse as sp  # type: ignore
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    "renormalize_query_by_atlas",
    "project_query_onto_atlas",
    "find_systematic_genes",
    "project_atlas_to_query",
    "find_biased_genes",
    "compute_query_projection",
]


@ut.logged()
@ut.timed_call()
def renormalize_query_by_atlas(  # pylint: disable=too-many-statements,too-many-branches
    what: str = "__x__",
    *,
    adata: AnnData,
    qdata: AnnData,
    var_annotations: Dict[str, Any],
    layers: Dict[str, Any],
    varp_annotations: Dict[str, Any],
) -> Optional[AnnData]:
    """
    Add an ``ATLASNORM`` pseudo-gene to query metacells data to compensate for the query having filtered out many genes.

    This renormalizes the gene fractions in the query to fit the atlas in case the query has aggressive filtered a
    significant amount of genes.

    **Input**

    Annotated query ``qdata`` and atlas ``adata``, where the observations are cells and the variables are genes, where
    ``X`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation annotation containing
    such a matrix.

    **Returns**

    None if no normalization is needed (or possible). Otherwise, a copy of the query metacells data, with an additional
    variable (gene) called ``ATLASNORM`` to the query data, such that the total number of UMIs for each query metacells
    is as expected given the total number of UMIs of the genes common to the query and the atlas. This is skipped if the
    query and the atlas have exactly the same list of genes, or if if the query already contains a high number of genes
    missing from the atlas so that the total number of UMIs for the query metacells is already at least the expected
    based on the common genes.

    **Computation Parameters**

    1. Computes how many UMIs should be added to each query metacell so that its (total UMIs / total common gene UMIs)
       would be the same as the (total atlas UMIs / total atlas common UMIs). If this is zero (or negative), stop.

    2. Add an ``ATLASNORM`` pseudo-gene to the query with the above amount of UMIs. For each per-variable (gene)
       observation, add the value specified in ``var_annotations``, whose list of keys must cover the set of
       per-variable annotations in the query data. For each per-observation-per-variable layer, add the value specified
       in ``layers``, whose list of keys must cover the existing layers. For each per-variable-per-variable annotation,
       add the value specified in ``varp_annotations``.
    """
    for name in qdata.var.keys():
        if "|" not in name and name not in var_annotations.keys():
            raise RuntimeError(f"missing default value for variable annotation {name}")

    for name in qdata.layers.keys():
        if name not in layers.keys():
            raise RuntimeError(f"missing default value for layer {name}")

    for name in qdata.varp.keys():
        if name not in varp_annotations.keys():
            raise RuntimeError(f"missing default value for variable-variable {name}")

    if list(qdata.var_names) == list(adata.var_names):
        return None

    query_genes_list = list(qdata.var_names)
    atlas_genes_list = list(adata.var_names)
    common_genes_list = list(sorted(set(qdata.var_names) & set(adata.var_names)))
    query_gene_indices = np.array([query_genes_list.index(gene) for gene in common_genes_list])
    atlas_gene_indices = np.array([atlas_genes_list.index(gene) for gene in common_genes_list])
    common_qdata = ut.slice(qdata, name=".common", vars=query_gene_indices)
    common_adata = ut.slice(adata, name=".common", vars=atlas_gene_indices)

    assert list(common_qdata.var_names) == list(common_adata.var_names)

    atlas_total_umis_per_metacell = ut.get_o_numpy(adata, what, sum=True)
    atlas_common_umis_per_metacell = ut.get_o_numpy(common_adata, what, sum=True)
    atlas_total_umis = np.sum(atlas_total_umis_per_metacell)
    atlas_common_umis = np.sum(atlas_common_umis_per_metacell)
    atlas_disjoint_umis_fraction = atlas_total_umis / atlas_common_umis - 1.0

    ut.log_calc("atlas_total_umis", atlas_total_umis)
    ut.log_calc("atlas_common_umis", atlas_common_umis)
    ut.log_calc("atlas_disjoint_umis_fraction", atlas_disjoint_umis_fraction)

    query_total_umis_per_metacell = ut.get_o_numpy(qdata, what, sum=True)
    query_common_umis_per_metacell = ut.get_o_numpy(common_qdata, what, sum=True)
    query_total_umis = np.sum(query_total_umis_per_metacell)
    query_common_umis = np.sum(query_common_umis_per_metacell)
    query_disjoint_umis_fraction = query_total_umis / query_common_umis - 1.0

    ut.log_calc("query_total_umis", query_total_umis)
    ut.log_calc("query_common_umis", query_common_umis)
    ut.log_calc("query_disjoint_umis_fraction", query_disjoint_umis_fraction)

    if query_disjoint_umis_fraction >= atlas_disjoint_umis_fraction:
        return None

    query_normalization_umis_fraction = atlas_disjoint_umis_fraction - query_disjoint_umis_fraction
    ut.log_calc("query_normalization_umis_fraction", query_normalization_umis_fraction)
    query_normalization_umis_per_metacell = query_common_umis_per_metacell * query_normalization_umis_fraction

    _proper, dense, compressed = ut.to_proper_matrices(qdata.X)

    if dense is None:
        assert compressed is not None
        dense = ut.to_numpy_matrix(compressed)
    added = np.concatenate([dense, query_normalization_umis_per_metacell[:, np.newaxis]], axis=1)

    if compressed is not None:
        added = sp.csr_matrix(added)

    assert added.shape[0] == qdata.shape[0]
    assert added.shape[1] == qdata.shape[1] + 1

    ndata = AnnData(added)
    ndata.obs_names = qdata.obs_names
    var_names = list(qdata.var_names)
    var_names.append("ATLASNORM")
    ndata.var_names = var_names

    for name, value in qdata.uns.items():
        ut.set_m_data(ndata, name, value)

    for name, value in qdata.obs.items():
        ut.set_o_data(ndata, name, value)

    for name, value in qdata.obsp.items():
        ut.set_oo_data(ndata, name, value)

    for name in qdata.var.keys():
        if "|" in name:
            continue
        value = ut.get_v_numpy(qdata, name)
        value = np.append(value, [var_annotations[name]])
        ut.set_v_data(ndata, name, value)

    for name in qdata.layers.keys():
        data = ut.get_vo_proper(qdata, name)
        _proper, dense, compressed = ut.to_proper_matrices(data)

        if dense is None:
            assert compressed is not None
            dense = ut.to_numpy_matrix(compressed)

        values = np.full(qdata.n_obs, layers[name], dtype=dense.dtype)
        added = np.concatenate([dense, values[:, np.newaxis]], axis=1)

        if compressed is not None:
            added = sp.csr_matrix(added)

        ut.set_vo_data(ndata, name, added)

    for name in qdata.varp.keys():
        data = ut.get_vv_proper(qdata, name)
        _proper, dense, compressed = ut.to_proper_matrices(data)

        if dense is None:
            assert compressed is not None
            dense = ut.to_numpy_matrix(compressed)

        values = np.full(qdata.n_vars, varp_annotations[name], dtype=dense.dtype)
        added = np.concatenate([dense, values[:, np.newaxis]], axis=1)
        values = np.full(qdata.n_vars + 1, varp_annotations[name], dtype=dense.dtype)
        added = np.concatenate([added, values[np.newaxis, :]], axis=0)

        if compressed is not None:
            added = sp.csr_matrix(added)

        ut.set_vv_data(ndata, name, added)

    return ndata


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
    max_consistency_fold_factor: float = pr.project_max_consistency_fold_factor,
    candidates_count: int = pr.project_candidates_count,
    min_usage_weight: float = pr.project_min_usage_weight,
    reproducible: bool,
    second_anchor_indices: Optional[List[int]] = None,
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

    1. Correlate the metacell with all the atlas metacells, and pick the highest-correlated one as the "anchor".
       If ``second_anchor_indices`` is not ``None``, then the ``qdata`` must contain only a single query metacell, and
       is expected to contain a ``projected`` per-observation-per-variable matrix containing the projected image of this
       query metacell on the atlas using a single anchor. The code will compute the residual of the query and the atlas
       relative to this projection and pick a second atlas anchor whose residuals are the most correlated to the query
       metacell's residuals. If ``reproducible``, a slower (still parallel) but reproducible algorithm will be used.

    2. Consider as candidates only atlas metacells whose maximal gene fold factor compared to the anchor(s) is at most
       ``max_consistency_fold_factor`` (default: {max_consistency_fold_factor}). Ignore the fold factors of genes whose
       sum of UMIs in the anchor(s) and the candidate metacells is less than ``min_significant_gene_value`` (default:
       {min_significant_gene_value}).

    3. Select (for each anchor) the ``candidates_count`` (default: {candidates_count}) candidate metacells with the
       highest correlation with the query metacell.

    4. Compute the non-negative weights (with a sum of 1) of the selected candidates that give the best projection of
       the query metacells onto the atlas. Since the algorithm for computing these weights rarely produces an exact 0
       weight, reduce all weights less than the ``min_usage_weight`` (default: {min_usage_weight}) to zero. If
       ``project_log_data`` (default: {project_log_data}), compute the match on the log of the data instead of the
       actual data. If ``second_anchor_indices`` is not ``None``, it is set to the list of indices of the used atlas
       metacells candidates correlated with the second anchor.
    """
    prepared_arguments = _project_query_atlas_data_arguments(
        what,
        adata=adata,
        qdata=qdata,
        atlas_total_umis=atlas_total_umis,
        query_total_umis=query_total_umis,
        project_log_data=project_log_data,
        fold_normalization=fold_normalization,
        min_significant_gene_value=min_significant_gene_value,
        max_consistency_fold_factor=max_consistency_fold_factor,
        candidates_count=candidates_count,
        min_usage_weight=min_usage_weight,
        reproducible=reproducible,
        second_anchor_indices=second_anchor_indices,
    )

    @ut.timed_call("project_single_metacell")
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


def _project_query_atlas_data_arguments(
    what: Union[str, ut.Matrix],
    *,
    adata: AnnData,
    qdata: AnnData,
    atlas_total_umis: Optional[ut.Vector],
    query_total_umis: Optional[ut.Vector],
    project_log_data: bool,
    fold_normalization: float,
    min_significant_gene_value: float,
    max_consistency_fold_factor: float,
    candidates_count: int,
    min_usage_weight: float,
    reproducible: bool,
    second_anchor_indices: Optional[List[int]],
) -> Dict[str, Any]:
    assert fold_normalization > 0
    assert candidates_count > 0
    assert min_usage_weight >= 0
    assert max_consistency_fold_factor >= 0
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

    if second_anchor_indices is not None:
        assert qdata.n_obs == 1
        query_single_fractions = ut.to_numpy_vector(ut.get_vo_proper(qdata, "projected")) / query_total_umis[0]

        query_residual_fractions = query_fractions - query_single_fractions[np.newaxis, :]
        query_residual_fractions[query_residual_fractions < 0] = 0

        atlas_residual_fractions = atlas_fractions - ut.to_numpy_vector(query_residual_fractions)[np.newaxis, :]
        atlas_residual_fractions[atlas_residual_fractions < 0] = 0

        if project_log_data:
            atlas_residual_fractions += fold_normalization
            query_residual_fractions += fold_normalization

            atlas_project_residual_data = np.log2(atlas_residual_fractions)
            query_project_residual_data = np.log2(query_residual_fractions)
        else:
            atlas_project_residual_data = atlas_residual_fractions
            query_project_residual_data = query_residual_fractions

        query_atlas_corr_residual: Optional[ut.NumpyMatrix] = ut.cross_corrcoef_rows(
            query_project_residual_data, atlas_project_residual_data, reproducible=reproducible
        )

    else:
        query_atlas_corr_residual = None

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

    query_atlas_corr = ut.cross_corrcoef_rows(query_project_data, atlas_project_data, reproducible=reproducible)

    return dict(
        atlas_umis=atlas_umis,
        query_atlas_corr=query_atlas_corr,
        atlas_project_data=atlas_project_data,
        query_project_data=query_project_data,
        atlas_log_fractions=atlas_log_fractions,
        candidates_count=candidates_count,
        min_significant_gene_value=min_significant_gene_value,
        min_usage_weight=min_usage_weight,
        max_consistency_fold_factor=max_consistency_fold_factor,
        second_anchor_indices=second_anchor_indices,
        query_atlas_corr_residual=query_atlas_corr_residual,
    )


@ut.logged()
def _project_single_metacell(  # pylint: disable=too-many-statements
    *,
    query_metacell_index: int,
    atlas_umis: ut.Matrix,
    query_atlas_corr: ut.NumpyMatrix,
    atlas_project_data: ut.NumpyMatrix,
    query_project_data: ut.NumpyMatrix,
    atlas_log_fractions: ut.NumpyMatrix,
    candidates_count: int,
    min_significant_gene_value: float,
    min_usage_weight: float,
    max_consistency_fold_factor: float,
    second_anchor_indices: Optional[List[int]],
    query_atlas_corr_residual: Optional[ut.NumpyMatrix],
) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
    query_metacell_project_data = query_project_data[query_metacell_index, :]

    query_metacell_atlas_correlations = query_atlas_corr[query_metacell_index, :]
    query_metacell_atlas_order = np.argsort(-query_metacell_atlas_correlations)

    atlas_anchor_index = query_metacell_atlas_order[0]
    ut.log_calc("atlas_anchor_index", atlas_anchor_index)
    atlas_anchor_log_fractions = atlas_log_fractions[atlas_anchor_index, :]
    atlas_anchor_umis = ut.to_numpy_vector(atlas_umis[atlas_anchor_index, :])

    atlas_candidate_indices_set = set([atlas_anchor_index])
    position = 1
    while len(atlas_candidate_indices_set) < candidates_count and position < len(query_metacell_atlas_order):
        atlas_metacell_index = query_metacell_atlas_order[position]
        position += 1
        atlas_metacell_log_fractions = atlas_log_fractions[atlas_metacell_index, :]
        atlas_metacell_consistency_fold_factors = np.abs(atlas_metacell_log_fractions - atlas_anchor_log_fractions)
        atlas_metacell_umis = ut.to_numpy_vector(atlas_umis[atlas_metacell_index, :])
        atlas_metacell_significant_genes_mask = atlas_metacell_umis + atlas_anchor_umis >= min_significant_gene_value
        atlas_metacell_consistency = np.max(
            atlas_metacell_consistency_fold_factors[atlas_metacell_significant_genes_mask]
        )
        if atlas_metacell_consistency <= max_consistency_fold_factor:
            atlas_candidate_indices_set.add(atlas_metacell_index)

    if query_atlas_corr_residual is not None:
        query_metacell_atlas_residual_correlations = query_atlas_corr_residual[query_metacell_index, :]
        query_metacell_atlas_residual_order = np.argsort(-query_metacell_atlas_residual_correlations)

        atlas_secondary_anchor_index = query_metacell_atlas_residual_order[0]
        ut.log_calc("atlas_secondary_anchor_index", atlas_secondary_anchor_index)

        atlas_secondary_candidate_indices_set = set([atlas_secondary_anchor_index])
        position = 1
        while len(atlas_secondary_candidate_indices_set) < candidates_count and position < len(
            query_metacell_atlas_residual_order
        ):
            atlas_metacell_index = query_metacell_atlas_order[position]
            position += 1
            atlas_metacell_log_fractions = atlas_log_fractions[atlas_metacell_index, :]
            atlas_metacell_consistency_fold_factors = np.abs(atlas_metacell_log_fractions - atlas_anchor_log_fractions)
            atlas_metacell_umis = ut.to_numpy_vector(atlas_umis[atlas_metacell_index, :])
            atlas_metacell_significant_genes_mask = (
                atlas_metacell_umis + atlas_anchor_umis >= min_significant_gene_value
            )
            atlas_metacell_consistency = np.max(
                atlas_metacell_consistency_fold_factors[atlas_metacell_significant_genes_mask]
            )
            if atlas_metacell_consistency <= max_consistency_fold_factor:
                atlas_secondary_candidate_indices_set.add(atlas_metacell_index)

        atlas_candidate_indices = np.array(sorted(atlas_candidate_indices_set | atlas_secondary_candidate_indices_set))
    else:
        atlas_candidate_indices = np.array(sorted(atlas_candidate_indices_set))

    atlas_candidates_project_data = atlas_project_data[atlas_candidate_indices, :]

    represent_result = ut.represent(query_metacell_project_data, atlas_candidates_project_data)
    assert represent_result is not None
    atlas_candidate_weights = represent_result[1]
    atlas_candidate_weights[atlas_candidate_weights < min_usage_weight] = 0
    atlas_candidate_weights[atlas_candidate_weights < min_usage_weight] /= np.sum(atlas_candidate_weights)

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
    to_property_name: str = "systematic_gene",
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
        ``systematic_gene`` (or ``to_property_name``)
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

    if adata.n_obs == 1:
        systematic = np.zeros(adata.n_vars, dtype="bool")
    else:
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

    ut.set_v_data(qdata, to_property_name, systematic)


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
        assert np.any(metacell_mask)
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
    to_property_name: str = "biased_gene",
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
        ``biased_gene`` (or ``to_property_name``):
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

    ut.set_v_data(adata, to_property_name, mask_of_genes)


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
