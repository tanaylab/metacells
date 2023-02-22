"""
Projection
----------
"""

from math import ceil
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np
import scipy.sparse as sp  # type: ignore
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    "projection_pipeline",
    "outliers_projection_pipeline",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def projection_pipeline(
    what: str = "__x__",
    *,
    adata: AnnData,
    qdata: AnnData,
    only_atlas_marker_genes: bool = pr.only_atlas_marker_genes,
    only_query_marker_genes: bool = pr.only_query_marker_genes,
    ignore_atlas_lateral_genes: bool = pr.ignore_atlas_lateral_genes,
    ignore_query_lateral_genes: bool = pr.ignore_query_lateral_genes,
    consider_atlas_noisy_genes: bool = pr.consider_atlas_noisy_genes,
    consider_query_noisy_genes: bool = pr.consider_query_noisy_genes,
    misfit_min_metacells_fraction: float = pr.misfit_min_metacells_fraction,
    project_fold_normalization: float = pr.project_fold_normalization,
    project_candidates_count: int = pr.project_candidates_count,
    project_min_candidates_fraction: float = pr.project_min_candidates_fraction,
    project_min_significant_gene_umis: int = pr.project_min_significant_gene_umis,
    project_min_usage_weight: float = pr.project_min_usage_weight,
    project_max_consistency_fold_factor: float = pr.project_max_consistency_fold_factor,
    project_max_projection_fold_factor: float = pr.project_max_projection_fold_factor,
    project_max_projection_noisy_fold_factor: float = pr.project_max_projection_noisy_fold_factor,
    project_max_misfit_genes: int = pr.project_max_misfit_genes,
    project_min_essential_genes_fraction: Optional[float] = pr.project_min_essential_genes_fraction,
    atlas_type_property_name: str = "type",
    project_corrections: bool = pr.project_corrections,
    project_min_corrected_gene_correlation: float = pr.project_min_corrected_gene_correlation,
    project_min_corrected_gene_factor: float = pr.project_min_corrected_gene_factor,
    reproducible: bool,
    top_level_parallel: bool = True,
) -> ut.CompressedMatrix:
    """
    Complete pipeline for projecting query metacells onto an atlas of metacells for the ``what`` (default: {what}) data.

    **Input**

    Annotated query ``qdata`` and atlas ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation annotation
    containing such a matrix. The atlas should also contain a ``type`` per-observation (metacell) annotation.

    The ``qdata`` may include per-gene masks, ``manually_ignored_gene`` and ``manually_ignored_gene_of_<type>``, which
    force the code below to ignore the marked genes from either the preliminary projection or the following refined
    type-specific projections.

    **Returns**

    A matrix whose rows are query metacells and columns are atlas metacells, where each entry is the weight of the atlas
    metacell in the projection of the query metacells. The sum of weights in each row (that is, for a single query
    metacell) is 1. The weighted sum of the atlas metacells using these weights is the "projected" image of the query
    metacell onto the atlas.

    For "similar" query metacells (see below), these weights are all located in a small region of the atlas manifold.
    For "dissimilar" query metacells, these weights will include atlas metacells in a secondary region of the atlas
    attempting to cover the residual differences between the query and the main atlas manifold region. This tries to
    identify doublets.

    In addition, a modified version of ``qdata`` with a new ``ATLASNORM`` gene (if requested), and the following
    additional annotations:

    TODOX

    Variable (Gene) Annotations
        ``correction_factor``
            The ratio between the original query and the corrected values (>1 if the gene was increased, <1 if it was
            reduced).

        ``projected_correlation``
            For each gene, the correlation between the (corrected) query expression level and the projected expression
            level.

        ``correlated_gene``
            A boolean mask of genes which were ignored because they were very correlated between the query and the
            atlas.

        ``atlas_gene``
            A boolean mask indicating whether the gene exists in the atlas.

        ``atlas_marker_gene`` (``only_atlas_marker_genes`` requires an atlas ``marker_gene`` mask)
            A boolean mask indicating whether the gene is considered a "marker" in the atlas.

        ``atlas_lateral_gene`` (``ignore_atlas_lateral_genes`` requires an atlas has ``lateral_gene`` mask) A boolean
            mask indicating whether the gene was forbidden from being selected when computing the atlas metacell (and
            hence is ignored).

        ``atlas_noisy_gene`` (``consider_atlas_noisy_genes`` requires an atlas has ``noisy_gene`` mask) A
            boolean mask indicating whether the gene was forbidden from being selected or used for detecting deviant
            (outlier) cells when computing the atlas metacells (and hence is ignored).

        ``ignored_gene``
            A boolean mask indicating whether the gene was ignored by the projection (for any reason).

        ``essential_gene_of_<type>``
            For each query metacell type, a boolean mask indicating whether properly projecting this gene is essential
            for assigning this type to a query metacell. This is copied from the atlas if
            ``project_min_essential_genes_fraction`` is specified.

        ``misfit_gene_of_<type>``
            For each query metacell type, a boolean mask indicating whether the gene has a strong bias in the query
            metacells of this type compared to the atlas metacells of this type.

        ``ignored_gene_of_<type>``
            For each query metacell type, a boolean mask indicating whether the gene was ignored by the projection (for
            any reason) when computing the projection for metacells of this type.

    Observation (Cell) Annotations
        ``similar``
            A boolean mask indicating whether the corrected query metacell is similar to its projection onto the atlas
            (ignoring the ``ignored_genes``). If ``False`` the metacell is said to be "dissimilar", which may indicate
            the query contains cell states that do not appear in the atlas. If ``True`` the metacell is said to be
            "similar", but there may still be significant systematic differences between the query and the atlas as
            captured by the various per-gene annotations. In addition, the projection on the atlas might indicate the
            query metacell is a doublet combining two different points in the atlas (see below). In either case, being
            "similar" doesn't guarantee we have a good query metacell that exists in the atlas.

        ``essential_genes_count``
            If ``project_min_essential_genes_fraction`` was specified, the number of essential genes for the
            metacell projected type(s).

        ``similar_essential_genes_count``
            If ``project_min_essential_genes_fraction`` was specified, the number of essential genes for the
            metacell projected type(s) that were projected within the ``project_max_projection_fold_factor`` similarity
            threshold.

        ``dissimilar_genes_count``
            The number of genes whose fold factor between the query metacell and its projection on the atlas is above
            the threshold.

        ``projected_type``
            The type assigned to each query metacell by its projection on the atlas, if it is "similar" and matches a
            single region in the atlas. This will be the special value "Dissimilar" for query metacells that are not
            similar to their projection, and "Mixture" for query metacells that are similar to a mixture of two regions
            of the same type in the atlas, or "Doublet" if the two regions have different types. Even so, this should
            not be taken as gospel, as there may still be significant systematic differences between the query and the
            atlas as captured by the various per-gene annotations.

        ``projected_secondary_type``
            If the query metacell is "Dissimilar", "Doublet" or "Mixture", this contains the additional type information
            (the type that best describes the projection for "Dissimilar" query metacells, or the ``;``-separated two
            types that are mixed for a "Doublet" or "Mixture" query metacells).

        ``projected_correlation``
            The correlation between the projected and the corrected UMIs for each metacell.

    Observation-Variable (Cell-Gene) Annotations
        ``projected``
            A matrix of UMIs where the sum of UMIs for each corrected query metacell is the same as the sum of ``what``
            UMIs, describing the "projected" image of the query metacell onto the atlas. This projection is a weighted
            average of some atlas metacells (using the computed weights returned by this function).

        ``projected_fold``
            For each gene and query metacell, the fold factor of this gene between the query and its projection (unless
            the value is too low to be of interest, in which case it will be zero).

    **Computation Parameters**

    0. Find the subset of genes that exist in both the query and the atlas. All computations will be done on this common
       subset.

    Compute preliminary projection:

    1. Compute a mask of ignored genes, containing any genes named in ``ignored_gene_names`` or that match any of the
       ``ignored_gene_patterns``. If ``only_atlas_marker_genes`` (default:
       ``only_atlas_marker_genes``), ignore genes the atlas did not mark as "marker" (and store the
       ``atlas_marker_gene`` mask). If ``only_query_marker_genes`` (default:
       ``only_query_marker_genes``), ignore genes the query did not mark as "marker". If
       ``ignore_atlas_lateral_genes`` (default: {ignore_atlas_lateral_genes}), also ignore the ``lateral_gene`` of the
       atlas (and store them as an ``atlas_lateral_gene`` mask). If ``consider_atlas_noisy_genes`` (default:
       {consider_atlas_noisy_genes}), also ignore the ``noisy_gene`` of the atlas (and store them as an
       ``atlas_noisy_gene`` mask). If ``ignore_query_lateral_genes`` (default: {ignore_query_lateral_genes}), also
       ignore the ``lateral_gene`` of the query. If ``consider_query_noisy_genes`` (default:
       {consider_query_noisy_genes}), also ignore the ``noisy_gene`` of the query. All these genes are ignored by
       the following code. In addition, ignore any genes marked in ``manually_ignored_gene``, if this annotation exists
       in the query.

    2. Invoke :py:func:`metacells.tools.project.compute_projection_weights` and
       :py:func:`metacells.tools.project.compute_projected_fractions` to project each query metacell onto the atlas,
       using the ``project_fold_normalization`` (default: {project_fold_normalization}),
       ``project_min_significant_gene_umis`` (default: {project_min_significant_gene_umis}),
       ``project_candidates_count`` (default: {project_candidates_count}),
       ``project_min_candidates_fraction`` (default: {project_min_candidates_fraction}),
       ``project_min_usage_weight`` (default:
       {project_min_usage_weight}) and ``reproducible``.

    3. Invoke :py:func:`metacells.tools.quality.compute_projected_fold_factors` to compute the significant
       fold factors between the query and its projection, using the ``project_fold_normalization`` (default:
       {project_fold_normalization}) and ``project_min_significant_gene_umis`` (default:
       {project_min_significant_gene_umis}).

    4. Correlate the expression levels of each gene between the query and projection. If this is at least
       ``project_min_corrected_gene_correlation`` (default: {project_min_corrected_gene_correlation}), compute the ratio
       between the mean expression of the gene in the projection and the query. If this is at most
       1/(1+``project_min_corrected_gene_factor``) or at least (1+``project_min_corrected_gene_factor``) (default:
       {project_min_corrected_gene_factor}), then multiply the gene's value by this factor so its level would match the
       atlas. If any genes were ignored or corrected, then repeat steps 1-4. However, if not ``project_corrections``,
       only record the correction factor and proceed without actually performing the correction.

    5. Invoke :py:func:`metacells.tools.project.convey_atlas_to_query` to assign a projected type to each of the
       query metacells based on the ``atlas_type_property_name`` (default: {atlas_type_property_name}).

    Then, for each type of query metacells:

    If ``top_level_parallel`` (default: ``top_level_parallel``), do this in parallel. This seems to work better than
    doing this serially and using parallelism within each type. However this is still inefficient compared to using both
    types of parallelism at once, which the code currently can't do without non-trivial coding (this would have been
    trivial in Julia...).

    6. Enhance the mask of type-specific ignored genes with any genes marked in ``manually_ignored_gene_of_<type>``, if
       this annotation exists in the query.

    7. Invoke :py:func:`metacells.tools.project.compute_projection_weights` to project each query metacell of the type
       onto the atlas, using the type-specific ignored genes in addition to the global ignored genes. Note that even
       though we only look at query metacells (tentatively) assigned the type, their projection on the atlas may use
       metacells of any type.

    8. Invoke :py:func:`metacells.tools.project.compute_projected_fractions` and
       :py:func:`metacells.tools.quality.compute_projected_fold_factors` to compute the significant fold factors between
       the query and its projection.

    9. Invoke :py:func:`metacells.tools.project.find_misfit_genes` to identify type-specific misfit genes. If any such
       genes are found, add them to the type-specific ignored genes (instead of the global misfit genes list) and repeat
       steps 6-9.

    10. Invoke :py:func:`metacells.tools.quality.compute_similar_query_metacells` to verify which query metacells
        ended up being sufficiently similar to their projection, using the ``essential_gene_of_<type>`` if needed.

    And then:

    11. Invoke :py:func:`metacells.tools.project.convey_atlas_to_query` to assign an updated projected type to each of
        the query metacells based on the ``atlas_type_property_name`` (default: {atlas_type_property_name}). If this
        changed the type assigned to any query metacell, repeat steps 6-10 (but do these steps no more than 3 times).

    For each query metacell that ended up being dissimilar:

    12. Update the list of ignored genes for the metacell based on the ``ignored_gene_of_<type>`` for both the primary
        (initially from the above) query metacell type and the secondary query metacell type (initially empty).

    13. Invoke :py:func:`metacells.tools.project.compute_projection_weights` just for this metacell, but allow the
        projection to use a secondary location in the atlas based on the residuals of the atlas metacells relative to
        the primary query projection.

    14. Invoke :py:func:`metacells.tools.project.convey_atlas_to_query` twice, once for the weights of the primary
        location and once for the weights of the secondary location, to obtain a primary and secondary type for
        the query metacell. If these have changed, repeat steps 12-14 (but do these steps no more than 3 times; note
        will will always do them twice as the 1st run will generate some non-empty secondary type).

    15. Invoke :py:func:`metacells.tools.project.compute_projected_fractions`,
        :py:func:`metacells.tools.quality.compute_projected_fold_factors` and
        :py:func:`metacells.tools.quality.compute_similar_query_metacells` to update the projection and evaluation of
        the query metacell. If it is now similar, mark it as a doublet or a mixture depending on whether the primary and
        the secondary types are different or identical.

    16. Invoke :py:func:`metacells.tools.quality.compute_metacells_projection_correlation` to compute the correlation
        between the projected and the corrected UMIs of each query metacell.
    """
    assert project_min_corrected_gene_factor >= 0
    use_essential_genes = project_min_essential_genes_fraction is not None

    ut.set_m_data(qdata, "project_max_projection_fold_factor", project_max_projection_fold_factor)
    ut.set_m_data(qdata, "project_max_projection_noisy_fold_factor", project_max_projection_noisy_fold_factor)
    ut.set_m_data(qdata, "project_max_misfit_genes", project_max_misfit_genes)

    atlas_common_gene_indices, query_common_gene_indices = _common_gene_indices(adata, qdata)

    type_names = _initialize_atlas_data_in_query(
        adata,
        qdata,
        atlas_type_property_name=atlas_type_property_name,
        atlas_common_gene_indices=atlas_common_gene_indices,
        query_common_gene_indices=query_common_gene_indices,
        consider_atlas_noisy_genes=consider_atlas_noisy_genes,
        consider_query_noisy_genes=consider_query_noisy_genes,
        use_essential_genes=use_essential_genes,
    )

    common_adata = _initialize_common_adata(adata, atlas_common_gene_indices)
    common_qdata = _initialize_common_qdata(qdata, query_common_gene_indices)

    min_essential_genes_of_type = _min_essential_genes_of_type(
        adata=adata,
        common_adata=common_adata,
        type_names=type_names,
        project_min_essential_genes_fraction=project_min_essential_genes_fraction,
    )

    initial_fitted_genes_mask = _initial_fitted_genes_mask(
        common_adata=common_adata,
        common_qdata=common_qdata,
        only_atlas_marker_genes=only_atlas_marker_genes,
        only_query_marker_genes=only_query_marker_genes,
        ignore_atlas_lateral_genes=ignore_atlas_lateral_genes,
        ignore_query_lateral_genes=ignore_query_lateral_genes,
        min_significant_gene_umis=project_min_significant_gene_umis,
    )

    _compute_preliminary_projected_type(
        what=what,
        common_adata=common_adata,
        common_qdata=common_qdata,
        initial_fitted_genes_mask=initial_fitted_genes_mask,
        project_fold_normalization=project_fold_normalization,
        project_min_significant_gene_umis=project_min_significant_gene_umis,
        project_max_consistency_fold_factor=project_max_consistency_fold_factor,
        project_candidates_count=project_candidates_count,
        project_min_usage_weight=project_min_usage_weight,
        project_corrections=project_corrections,
        project_min_corrected_gene_correlation=project_min_corrected_gene_correlation,
        project_min_corrected_gene_factor=project_min_corrected_gene_factor,
        atlas_type_property_name=atlas_type_property_name,
        reproducible=reproducible,
    )

    weights = _compute_per_type_projection(
        what=what,
        common_adata=common_adata,
        common_qdata=common_qdata,
        type_names=type_names,
        initial_fitted_genes_mask=initial_fitted_genes_mask,
        misfit_min_metacells_fraction=misfit_min_metacells_fraction,
        project_fold_normalization=project_fold_normalization,
        project_candidates_count=project_candidates_count,
        project_min_significant_gene_umis=project_min_significant_gene_umis,
        project_min_usage_weight=project_min_usage_weight,
        project_max_consistency_fold_factor=project_max_consistency_fold_factor,
        project_max_projection_fold_factor=project_max_projection_fold_factor,
        project_max_projection_noisy_fold_factor=project_max_projection_noisy_fold_factor,
        project_max_misfit_genes=project_max_misfit_genes,
        min_essential_genes_of_type=min_essential_genes_of_type,
        atlas_type_property_name=atlas_type_property_name,
        top_level_parallel=top_level_parallel,
        reproducible=reproducible,
    )

    _compute_dissimilar_residuals_projection(
        what=what,
        weights_per_atlas_per_query_metacell=weights,
        common_adata=common_adata,
        common_qdata=common_qdata,
        project_fold_normalization=project_fold_normalization,
        project_candidates_count=project_candidates_count,
        project_min_candidates_fraction=project_min_candidates_fraction,
        project_min_significant_gene_umis=project_min_significant_gene_umis,
        project_min_usage_weight=project_min_usage_weight,
        project_max_consistency_fold_factor=project_max_consistency_fold_factor,
        project_max_projection_fold_factor=project_max_projection_fold_factor,
        project_max_projection_noisy_fold_factor=project_max_projection_noisy_fold_factor,
        project_max_misfit_genes=project_max_misfit_genes,
        min_essential_genes_of_type=min_essential_genes_of_type,
        atlas_type_property_name=atlas_type_property_name,
        top_level_parallel=top_level_parallel,
        reproducible=reproducible,
    )

    _common_data_to_full(
        qdata=qdata,
        common_qdata=common_qdata,
        project_corrections=project_corrections,
        type_names=type_names,
        use_essential_genes=use_essential_genes,
    )

    return sp.csr_matrix(weights)


def _common_gene_indices(adata: AnnData, qdata: AnnData) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
    if list(qdata.var_names) == list(adata.var_names):
        atlas_common_gene_indices = query_common_gene_indices = np.array(range(qdata.n_vars), dtype="int32")
    else:
        atlas_genes_list = list(adata.var_names)
        query_genes_list = list(qdata.var_names)
        common_genes_list = list(sorted(set(atlas_genes_list) & set(query_genes_list)))
        assert len(common_genes_list) > 0
        atlas_common_gene_indices = np.array([atlas_genes_list.index(gene) for gene in common_genes_list])
        query_common_gene_indices = np.array([query_genes_list.index(gene) for gene in common_genes_list])

    return atlas_common_gene_indices, query_common_gene_indices


def _initialize_atlas_data_in_query(
    adata: AnnData,
    qdata: AnnData,
    *,
    atlas_type_property_name: str,
    atlas_common_gene_indices: ut.NumpyVector,
    query_common_gene_indices: ut.NumpyVector,
    consider_atlas_noisy_genes: bool,
    consider_query_noisy_genes: bool,
    use_essential_genes: bool,
) -> List[str]:
    atlas_genes_mask = np.zeros(qdata.n_vars, dtype="bool")
    atlas_genes_mask[query_common_gene_indices] = True
    ut.set_v_data(qdata, "atlas_gene", atlas_genes_mask)

    type_names = list(np.unique(ut.get_o_numpy(adata, atlas_type_property_name)))

    genes_mask_names = ["lateral_gene", "noisy_gene", "marker_gene"]
    if consider_atlas_noisy_genes:
        genes_mask_names.append("noisy_gene")
    if use_essential_genes:
        genes_mask_names += [f"essential_gene_of_{type_name}" for type_name in type_names]

    for genes_mask_name in genes_mask_names:
        if not ut.has_data(adata, genes_mask_name):
            continue
        atlas_mask = ut.get_v_numpy(adata, genes_mask_name)
        query_mask = np.zeros(qdata.n_vars, dtype="bool")
        query_mask[query_common_gene_indices] = atlas_mask[atlas_common_gene_indices]
        ut.set_v_data(
            qdata,
            genes_mask_name if genes_mask_name.startswith("essential_gene_of_") else "atlas_" + genes_mask_name,
            query_mask,
        )

    noisy_masks: List[str] = []
    if consider_atlas_noisy_genes and ut.has_data(qdata, "atlas_noisy_gene"):
        noisy_masks.append("|atlas_noisy_gene")

    if consider_query_noisy_genes and ut.has_data(qdata, "noisy_gene"):
        noisy_masks.append("|noisy_gene")

    if len(noisy_masks) > 0:
        tl.combine_masks(qdata, noisy_masks, to="projected_noisy_gene")

    return type_names


def _initialize_common_adata(adata: AnnData, atlas_common_gene_indices: ut.NumpyVector) -> AnnData:
    common_adata = ut.slice(adata, name=".common", vars=atlas_common_gene_indices)

    atlas_total_common_umis = ut.get_o_numpy(common_adata, "total_umis", sum=True)
    ut.set_o_data(common_adata, "total_umis", atlas_total_common_umis)

    return common_adata


def _initialize_common_qdata(
    qdata: AnnData,
    query_common_gene_indices: ut.NumpyVector,
) -> AnnData:
    common_qdata = ut.slice(
        qdata,
        name=".common",
        vars=query_common_gene_indices,
        track_var="full_gene_index_of_qdata",
    )

    query_total_common_umis = ut.get_o_numpy(common_qdata, "total_umis", sum=True)
    ut.set_o_data(common_qdata, "total_umis", query_total_common_umis)
    ut.set_o_data(qdata, "total_atlas_umis", query_total_common_umis)

    return common_qdata


def _min_essential_genes_of_type(
    *,
    adata: AnnData,
    common_adata: AnnData,
    type_names: List[str],
    project_min_essential_genes_fraction: Optional[float],
) -> Dict[Tuple[str, str], Optional[int]]:
    min_essential_genes_of_type: Dict[Tuple[str, str], Optional[int]] = {
        (type_name, other_type_name): None for type_name in type_names for other_type_name in type_names
    }

    has_any_essential_genes = False
    if project_min_essential_genes_fraction is not None:
        for type_name in type_names:
            if type_name != "Outliers" and ut.has_data(adata, f"essential_gene_of_{type_name}"):
                has_any_essential_genes = True
                break

    if not has_any_essential_genes:
        return min_essential_genes_of_type
    assert project_min_essential_genes_fraction is not None

    for type_name in type_names:
        if type_name == "Outliers":
            min_essential_genes_count: Optional[int] = None
        else:
            atlas_essential_genes_mask = ut.get_v_numpy(adata, f"essential_gene_of_{type_name}")
            atlas_essential_genes_count = np.sum(atlas_essential_genes_mask)

            common_essential_genes_mask = ut.get_v_numpy(common_adata, f"essential_gene_of_{type_name}")
            common_essential_genes_count = np.sum(common_essential_genes_mask)

            min_essential_genes_count = ceil(project_min_essential_genes_fraction * atlas_essential_genes_count)
            assert min_essential_genes_count is not None
            assert min_essential_genes_count >= 0

            if min_essential_genes_count > common_essential_genes_count:
                atlas_essential_genes_names = sorted(adata.var_names[atlas_essential_genes_mask])
                common_essential_genes_names = sorted(common_adata.var_names[common_essential_genes_mask])
                missing_essential_genes_names = sorted(
                    set(atlas_essential_genes_names) - set(common_essential_genes_names)
                )
                ut.logger().warning(  # pylint: disable=logging-fstring-interpolation
                    f"the {common_essential_genes_count} "
                    f"common essential gene(s) {', '.join(common_essential_genes_names)} "
                    f"for the type {type_name} "
                    f"are not enough for the required {min_essential_genes_count}; "
                    "reducing the minimal requirement "
                    f" (the non-common essential gene(s) are: {', '.join(missing_essential_genes_names)})"
                )
                min_essential_genes_count = common_essential_genes_count

        min_essential_genes_of_type[(type_name, type_name)] = min_essential_genes_count
        ut.log_calc(f"min_essential_genes_of_type[{type_name}]", min_essential_genes_count)

        if min_essential_genes_count is None:
            continue

        for other_type_name in type_names:
            if other_type_name <= type_name or other_type_name == "Outliers":
                continue

            other_atlas_essential_genes_mask = ut.get_v_numpy(adata, f"essential_gene_of_{type_name}")
            pair_atlas_essential_genes_mask = atlas_essential_genes_mask | other_atlas_essential_genes_mask
            pair_atlas_essential_genes_count = np.sum(pair_atlas_essential_genes_mask)

            other_common_essential_genes_mask = ut.get_v_numpy(common_adata, f"essential_gene_of_{type_name}")
            pair_common_essential_genes_mask = common_essential_genes_mask | other_common_essential_genes_mask
            pair_common_essential_genes_count = np.sum(pair_common_essential_genes_mask)

            missing_essential_genes_count = pair_atlas_essential_genes_count - pair_common_essential_genes_count
            assert missing_essential_genes_count >= 0

            min_essential_genes_count = ceil(project_min_essential_genes_fraction * pair_atlas_essential_genes_count)
            assert min_essential_genes_count is not None
            ut.log_calc(f"min_essential_genes_of_type[{type_name}, {other_type_name}]", min_essential_genes_count)
            min_essential_genes_of_type[(type_name, other_type_name)] = min_essential_genes_count
            min_essential_genes_of_type[(other_type_name, type_name)] = min_essential_genes_count

    return min_essential_genes_of_type


def _initial_fitted_genes_mask(
    *,
    common_adata: AnnData,
    common_qdata: AnnData,
    only_atlas_marker_genes: bool,
    only_query_marker_genes: bool,
    ignore_atlas_lateral_genes: bool,
    ignore_query_lateral_genes: bool,
    min_significant_gene_umis: int,
) -> ut.NumpyVector:
    atlas_total_umis_per_gene = ut.get_v_numpy(common_adata, "total_umis", sum=True)
    query_total_umis_per_gene = ut.get_v_numpy(common_qdata, "total_umis", sum=True)
    total_umis_per_gene = atlas_total_umis_per_gene + query_total_umis_per_gene
    initial_fitted_genes_mask = total_umis_per_gene >= min_significant_gene_umis
    ut.log_calc("total_umis_mask", initial_fitted_genes_mask)

    if only_atlas_marker_genes and ut.has_data(common_adata, "marker_gene"):
        initial_fitted_genes_mask &= ut.get_v_numpy(common_adata, "marker_gene")

    if only_query_marker_genes and ut.has_data(common_qdata, "marker_gene"):
        initial_fitted_genes_mask &= ut.get_v_numpy(common_qdata, "marker_gene")

    if ignore_atlas_lateral_genes and ut.has_data(common_adata, "lateral_gene"):
        initial_fitted_genes_mask &= ~ut.get_v_numpy(common_adata, "lateral_gene")

    if ignore_query_lateral_genes and ut.has_data(common_qdata, "lateral_gene"):
        initial_fitted_genes_mask &= ~ut.get_v_numpy(common_qdata, "lateral_gene")

    return initial_fitted_genes_mask


def _common_data_to_full(
    *,
    qdata: AnnData,
    common_qdata: AnnData,
    project_corrections: bool,
    use_essential_genes: bool,
    type_names: List[str],
) -> None:
    full_gene_index_of_common_qdata = ut.get_v_numpy(common_qdata, "full_gene_index_of_qdata")

    property_names = [f"fitted_gene_of_{type_name}" for type_name in type_names]
    if project_corrections:
        property_names.append("correction_factor")

    for property_name in property_names:
        data_per_common_gene = ut.get_v_numpy(common_qdata, property_name)
        data_per_full_gene = np.zeros(qdata.n_vars, dtype=data_per_common_gene.dtype)
        data_per_full_gene[full_gene_index_of_common_qdata] = data_per_common_gene
        ut.set_v_data(qdata, property_name, data_per_full_gene)

    for property_name in (
        "projected_type",
        "projected_secondary_type",
        "projected_correlation",
        "similar",
    ):
        data_per_metacell = ut.get_o_numpy(common_qdata, property_name)
        ut.set_o_data(qdata, property_name, data_per_metacell)

    for property_name in (
        "corrected_fraction",
        "projected_fraction",
        "fitted",
        "misfit",
        "projected_fold",
    ):
        data_per_common_gene_per_metacell = ut.get_vo_proper(common_qdata, property_name)
        data_per_gene_per_metacell = np.zeros(qdata.shape, dtype=ut.shaped_dtype(data_per_common_gene_per_metacell))
        data_per_gene_per_metacell[:, full_gene_index_of_common_qdata] = ut.to_numpy_matrix(
            data_per_common_gene_per_metacell
        )
        ut.set_vo_data(qdata, property_name, sp.csr_matrix(data_per_gene_per_metacell))

    if use_essential_genes:
        primary_type_per_metacell = ut.get_o_numpy(qdata, "projected_type")
        secondary_type_per_metacell = ut.get_o_numpy(qdata, "projected_secondary_type")
        essential_per_gene_per_metacell = np.zeros(qdata.shape, dtype="bool")
        for metacell_index in range(qdata.n_obs):
            primary_type_of_metacell = primary_type_per_metacell[metacell_index]
            if primary_type_of_metacell != "Outliers":
                essential_per_gene_per_metacell[metacell_index, :] = ut.get_v_numpy(
                    qdata, f"essential_gene_of_{primary_type_of_metacell}"
                )

            secondary_type_of_metacell = secondary_type_per_metacell[metacell_index]
            if secondary_type_of_metacell not in ("", "Outliers"):
                essential_per_gene_per_metacell[metacell_index, :] |= ut.get_v_numpy(
                    qdata, f"essential_gene_of_{secondary_type_of_metacell}"
                )

        ut.set_vo_data(qdata, "essential", sp.csr_matrix(essential_per_gene_per_metacell))


@ut.logged()
@ut.timed_call()
def _compute_preliminary_projected_type(
    *,
    what: str,
    common_adata: AnnData,
    common_qdata: AnnData,
    initial_fitted_genes_mask: ut.NumpyVector,
    project_fold_normalization: float,
    project_min_significant_gene_umis: float,
    project_max_consistency_fold_factor: float,
    project_candidates_count: int,
    project_min_usage_weight: float,
    project_corrections: bool,
    project_min_corrected_gene_correlation: float,
    project_min_corrected_gene_factor: float,
    atlas_type_property_name: str,
    reproducible: bool,
) -> None:
    fitted_adata = ut.slice(common_adata, name=".fitted", vars=initial_fitted_genes_mask)
    _normalize_corrected_fractions(fitted_adata, what)

    fitted_qdata = ut.slice(
        common_qdata,
        name=".fitted",
        vars=initial_fitted_genes_mask,
        track_var="common_gene_index_of_qdata",
    )
    _normalize_corrected_fractions(fitted_qdata, what)

    fitted_genes_correction_factors = np.full(fitted_qdata.n_vars, 1.0, dtype="float32")

    repeat = 0
    while True:
        repeat += 1
        ut.log_calc("preliminary repeat", repeat)

        weights = tl.compute_projection(
            adata=fitted_adata,
            qdata=fitted_qdata,
            fold_normalization=project_fold_normalization,
            min_significant_gene_umis=project_min_significant_gene_umis,
            max_consistency_fold_factor=project_max_consistency_fold_factor,
            candidates_count=project_candidates_count,
            min_usage_weight=project_min_usage_weight,
            reproducible=reproducible,
        )

        if (
            repeat > 2
            or not project_corrections
            or not _correct_correlated_genes(
                fitted_qdata=fitted_qdata,
                project_min_corrected_gene_correlation=project_min_corrected_gene_correlation,
                project_min_corrected_gene_factor=project_min_corrected_gene_factor,
                fitted_genes_correction_factors=fitted_genes_correction_factors,
                reproducible=reproducible,
            )
        ):
            break

    tl.convey_atlas_to_query(
        adata=fitted_adata,
        qdata=common_qdata,
        weights=weights,
        property_name=atlas_type_property_name,
        to_property_name="projected_type",
    )

    _fitted_data_to_common(
        common_qdata=common_qdata,
        fitted_qdata=fitted_qdata,
        fitted_genes_correction_factors=fitted_genes_correction_factors,
        project_corrections=project_corrections,
    )


def _fitted_data_to_common(
    *,
    common_qdata: AnnData,
    fitted_qdata: AnnData,
    fitted_genes_correction_factors: ut.NumpyVector,
    project_corrections: bool,
) -> None:
    common_gene_index_of_fitted_qdata = ut.get_v_numpy(fitted_qdata, "common_gene_index_of_qdata")

    for property_name in (
        "corrected_fraction",
        "projected_fraction",
    ):
        data_per_fitted_gene_per_metacell = ut.get_vo_proper(fitted_qdata, property_name)
        data_per_gene_per_metacell = np.zeros(
            common_qdata.shape, dtype=ut.shaped_dtype(data_per_fitted_gene_per_metacell)
        )
        data_per_gene_per_metacell[:, common_gene_index_of_fitted_qdata] = ut.to_numpy_matrix(
            data_per_fitted_gene_per_metacell
        )
        ut.set_vo_data(common_qdata, property_name, sp.csr_matrix(data_per_gene_per_metacell))

    if project_corrections:
        common_genes_correction_factors = np.zeros(common_qdata.n_vars, dtype="float32")
        common_genes_correction_factors[common_gene_index_of_fitted_qdata] = fitted_genes_correction_factors
        ut.set_v_data(common_qdata, "correction_factor", common_genes_correction_factors)


def _correct_correlated_genes(
    *,
    fitted_qdata: AnnData,
    project_min_corrected_gene_correlation: float,
    project_min_corrected_gene_factor: float,
    fitted_genes_correction_factors: ut.NumpyVector,
    reproducible: bool,
) -> bool:
    corrected_fractions_per_gene_per_cell = ut.to_numpy_matrix(
        ut.get_vo_proper(fitted_qdata, "corrected_fraction", layout="column_major")
    )
    projected_fractions_per_gene_per_cell = ut.to_numpy_matrix(
        ut.get_vo_proper(fitted_qdata, "projected_fraction", layout="column_major")
    )

    corrected_fractions_per_cell_per_gene = corrected_fractions_per_gene_per_cell.transpose()
    projected_fractions_per_cell_per_gene = projected_fractions_per_gene_per_cell.transpose()

    gene_correlations = ut.pairs_corrcoef_rows(
        projected_fractions_per_cell_per_gene, corrected_fractions_per_cell_per_gene, reproducible=reproducible
    )
    assert len(gene_correlations) == fitted_qdata.n_vars

    correlated_genes_mask = gene_correlations >= project_min_corrected_gene_correlation
    ut.log_calc("correlated_genes_mask", correlated_genes_mask)
    if not np.any(correlated_genes_mask):
        return False

    correlated_corrected_fractions_per_gene_per_cell = corrected_fractions_per_gene_per_cell[correlated_genes_mask, :]
    correlated_projected_fractions_per_gene_per_cell = projected_fractions_per_gene_per_cell[correlated_genes_mask, :]

    correlated_total_corrected_fractions_per_gene = ut.sum_per(
        correlated_corrected_fractions_per_gene_per_cell, per="row"
    )
    correlated_total_projected_fractions_per_gene = ut.sum_per(
        correlated_projected_fractions_per_gene_per_cell, per="row"
    )

    correlated_gene_correction_factors = (
        correlated_total_projected_fractions_per_gene / correlated_total_corrected_fractions_per_gene
    )

    high_factor = 1 + project_min_corrected_gene_factor
    low_factor = 1.0 / high_factor
    corrected_genes_mask = (correlated_gene_correction_factors <= low_factor) | (
        correlated_gene_correction_factors >= high_factor
    )

    ut.log_calc("corrected_genes_mask", corrected_genes_mask)
    if not np.any(corrected_genes_mask):
        return False

    correlated_gene_indices = np.where(correlated_genes_mask)[0]
    corrected_gene_indices = correlated_gene_indices[corrected_genes_mask]
    corrected_gene_factors = correlated_gene_correction_factors[corrected_genes_mask]
    fitted_genes_correction_factors[corrected_gene_indices] *= corrected_gene_factors

    corrected_fractions = ut.get_vo_proper(fitted_qdata, "corrected_fraction")
    corrected_fractions = ut.to_numpy_matrix(corrected_fractions, copy=True)
    corrected_fractions[:, corrected_gene_indices] *= corrected_gene_factors[np.newaxis, corrected_gene_indices]
    corrected_fractions = ut.fraction_by(corrected_fractions, by="row")
    ut.set_vo_data(fitted_qdata, "corrected_fraction", corrected_fractions)

    return True


@ut.logged()
@ut.timed_call()
def _compute_per_type_projection(
    *,
    what: str,
    common_adata: AnnData,
    common_qdata: AnnData,
    type_names: List[str],
    initial_fitted_genes_mask: ut.NumpyVector,
    misfit_min_metacells_fraction: float,
    project_fold_normalization: float,
    project_candidates_count: int,
    project_min_significant_gene_umis: float,
    project_min_usage_weight: float,
    project_max_consistency_fold_factor: float,
    project_max_projection_fold_factor: float,
    project_max_projection_noisy_fold_factor: float,
    project_max_misfit_genes: int,
    min_essential_genes_of_type: Dict[Tuple[str, str], Optional[int]],
    atlas_type_property_name: str,
    top_level_parallel: bool,
    reproducible: bool,
) -> ut.NumpyMatrix:
    old_types_per_metacell: List[Set[str]] = []
    for _metacell_index in range(common_qdata.n_obs):
        old_types_per_metacell.append(set())

    fitted_genes_mask_per_type = {type_name: initial_fitted_genes_mask.copy() for type_name in type_names}
    misfit_per_gene_per_metacell = np.empty(common_qdata.shape, dtype="bool")
    projected_correlation_per_metacell = np.empty(common_qdata.n_obs, dtype="float32")
    projected_fold_per_gene_per_metacell = np.empty(common_qdata.shape, dtype="float32")
    weights_per_atlas_per_query_metacell = np.empty((common_qdata.n_obs, common_adata.n_obs), dtype="float32")
    similar_per_metacell = np.empty(common_qdata.n_obs, dtype="bool")

    repeat = 0
    while True:
        repeat += 1
        ut.log_calc("types repeat", repeat)

        type_of_query_metacells = ut.get_o_numpy(common_qdata, "projected_type")

        misfit_per_gene_per_metacell[:, :] = False
        projected_fold_per_gene_per_metacell[:, :] = 0.0

        query_type_names = list(np.unique(type_of_query_metacells))

        @ut.timed_call("single_type_projection")
        def _single_type_projection(
            type_index: int,
        ) -> Dict[str, Any]:
            return _compute_single_type_projection(
                what=what,
                type_name=query_type_names[type_index],
                common_adata=common_adata,
                common_qdata=common_qdata,
                fitted_genes_mask_per_type=fitted_genes_mask_per_type,
                misfit_min_metacells_fraction=misfit_min_metacells_fraction,
                project_fold_normalization=project_fold_normalization,
                project_candidates_count=project_candidates_count,
                project_min_significant_gene_umis=project_min_significant_gene_umis,
                project_min_usage_weight=project_min_usage_weight,
                project_max_consistency_fold_factor=project_max_consistency_fold_factor,
                project_max_projection_fold_factor=project_max_projection_fold_factor,
                project_max_projection_noisy_fold_factor=project_max_projection_noisy_fold_factor,
                project_max_misfit_genes=project_max_misfit_genes,
                min_essential_genes_of_type=min_essential_genes_of_type,
                top_level_parallel=top_level_parallel,
                reproducible=reproducible,
            )

        def _collect_single_type_result(
            type_index: int,
            *,
            query_metacell_indices_of_type: ut.NumpyVector,
            fitted_genes_indices_of_type: ut.NumpyVector,
            similar_per_metacell_of_type: ut.NumpyVector,
            misfit_per_fitted_gene_per_metacell_of_type: ut.Matrix,
            projected_correlation_per_metacell_of_type: ut.NumpyVector,
            projected_fold_per_fitted_gene_per_metacell_of_type: ut.Matrix,
            weights_per_atlas_per_query_metacell_of_type: ut.Matrix,
        ) -> None:
            fitted_genes_mask_per_type[query_type_names[type_index]][:] = False
            fitted_genes_mask_per_type[query_type_names[type_index]][fitted_genes_indices_of_type] = True
            similar_per_metacell[query_metacell_indices_of_type] = similar_per_metacell_of_type
            projected_correlation_per_metacell[
                query_metacell_indices_of_type
            ] = projected_correlation_per_metacell_of_type
            genes_meshgrid = np.meshgrid(
                query_metacell_indices_of_type, fitted_genes_indices_of_type, sparse=True, indexing="ij"
            )
            misfit_per_gene_per_metacell[genes_meshgrid] = ut.to_numpy_matrix(
                misfit_per_fitted_gene_per_metacell_of_type
            )
            projected_fold_per_gene_per_metacell[genes_meshgrid] = ut.to_numpy_matrix(
                projected_fold_per_fitted_gene_per_metacell_of_type
            )
            weights_per_atlas_per_query_metacell[query_metacell_indices_of_type, :] = ut.to_numpy_matrix(
                weights_per_atlas_per_query_metacell_of_type
            )

        if top_level_parallel:
            for type_index, result in enumerate(ut.parallel_map(_single_type_projection, len(query_type_names))):
                _collect_single_type_result(type_index, **result)
        else:
            for type_index in range(len(query_type_names)):
                result = _single_type_projection(type_index)
                _collect_single_type_result(type_index, **result)

        if repeat > 2 or not _changed_projected_types(
            common_adata=common_adata,
            common_qdata=common_qdata,
            old_type_of_query_metacells=type_of_query_metacells,
            weights_per_atlas_per_query_metacell=weights_per_atlas_per_query_metacell,
            atlas_type_property_name=atlas_type_property_name,
            old_types_per_metacell=old_types_per_metacell,
        ):
            break

    for type_name, fitted_genes_mask_of_type in fitted_genes_mask_per_type.items():
        ut.set_v_data(common_qdata, f"fitted_gene_of_{type_name}", fitted_genes_mask_of_type)

    projected_type_per_metacell = ut.get_o_numpy(common_qdata, "projected_type")
    fitted_mask_per_gene_per_metacell = np.vstack(
        [fitted_genes_mask_per_type[type_name] for type_name in projected_type_per_metacell]
    )
    ut.set_vo_data(common_qdata, "fitted", fitted_mask_per_gene_per_metacell)

    ut.set_o_data(common_qdata, "similar", similar_per_metacell)
    ut.set_o_data(common_qdata, "projected_correlation", projected_correlation_per_metacell)

    ut.set_vo_data(common_qdata, "misfit", misfit_per_gene_per_metacell)
    ut.set_vo_data(common_qdata, "projected_fold", projected_fold_per_gene_per_metacell)

    return weights_per_atlas_per_query_metacell


@ut.logged()
@ut.timed_call()
def _compute_single_type_projection(
    *,
    what: str,
    type_name: str,
    common_adata: AnnData,
    common_qdata: AnnData,
    fitted_genes_mask_per_type: Dict[str, ut.NumpyVector],
    misfit_min_metacells_fraction: float,
    project_fold_normalization: float,
    project_candidates_count: int,
    project_min_significant_gene_umis: float,
    project_min_usage_weight: float,
    project_max_consistency_fold_factor: float,
    project_max_projection_fold_factor: float,
    project_max_projection_noisy_fold_factor: float,
    project_max_misfit_genes: int,
    min_essential_genes_of_type: Dict[Tuple[str, str], Optional[int]],
    top_level_parallel: bool,
    reproducible: bool,
) -> Dict[str, Any]:
    projected_type_per_metacell = ut.get_o_numpy(common_qdata, "projected_type")
    query_metacell_mask_of_type = projected_type_per_metacell == type_name
    ut.log_calc("query_metacell_mask_of_type", query_metacell_mask_of_type)
    assert np.any(query_metacell_mask_of_type)
    min_essential_genes = min_essential_genes_of_type[(type_name, type_name)]

    fitted_genes_mask_of_type = fitted_genes_mask_per_type[type_name]
    assert np.any(fitted_genes_mask_of_type)

    name: Optional[str] = f".fitted.{type_name}"
    repeat = 0
    while True:
        repeat += 1
        ut.log_calc(f"{type_name} misfit repeat", repeat)

        type_fitted_adata = ut.slice(
            common_adata, name=name or ut.get_name(common_adata), vars=fitted_genes_mask_of_type
        )
        _normalize_corrected_fractions(type_fitted_adata, what)

        type_fitted_qdata = ut.slice(
            common_qdata,
            name=name or ut.get_name(common_qdata),
            obs=query_metacell_mask_of_type,
            vars=fitted_genes_mask_of_type,
            track_var="common_gene_index_of_qdata",
        )
        _normalize_corrected_fractions(type_fitted_qdata, what)
        corrected_fractions = ut.get_vo_proper(type_fitted_qdata, "corrected_fraction")

        name = None

        type_weights = tl.compute_projection(
            adata=type_fitted_adata,
            qdata=type_fitted_qdata,
            fold_normalization=project_fold_normalization,
            min_significant_gene_umis=project_min_significant_gene_umis,
            max_consistency_fold_factor=project_max_consistency_fold_factor,
            candidates_count=project_candidates_count,
            min_usage_weight=project_min_usage_weight,
            reproducible=reproducible,
        )
        projected_fractions = ut.get_vo_proper(type_fitted_qdata, "projected_fraction")

        tl.compute_projected_folds(
            type_fitted_qdata,
            fold_normalization=project_fold_normalization,
            min_significant_gene_umis=project_min_significant_gene_umis,
        )
        projected_fold_per_fitted_gene_per_metacell_of_type = ut.get_vo_proper(type_fitted_qdata, "projected_fold")

        if not _detect_type_misfit_genes(
            type_fitted_qdata=type_fitted_qdata,
            max_projection_fold_factor=project_max_projection_fold_factor,
            max_projection_noisy_fold_factor=project_max_projection_noisy_fold_factor,
            misfit_min_metacells_fraction=misfit_min_metacells_fraction,
            fitted_genes_mask_of_type=fitted_genes_mask_of_type,
        ):
            break

    tl.compute_similar_query_metacells(
        type_fitted_qdata,
        max_projection_fold_factor=project_max_projection_fold_factor,
        max_projection_noisy_fold_factor=project_max_projection_noisy_fold_factor,
        max_misfit_genes=project_max_misfit_genes,
        essential_genes_property=f"essential_gene_of_{type_name}",
        min_essential_genes=min_essential_genes,
    )
    similar_per_metacell_of_type = ut.get_o_numpy(type_fitted_qdata, "similar")
    misfit_per_fitted_gene_per_metacell_of_type = ut.get_vo_proper(type_fitted_qdata, "misfit")

    projected_correlation_per_metacell_of_type = ut.pairs_corrcoef_rows(
        ut.to_numpy_matrix(corrected_fractions), ut.to_numpy_matrix(projected_fractions), reproducible=reproducible
    )

    if top_level_parallel:
        if not isinstance(misfit_per_fitted_gene_per_metacell_of_type, sp.csr_matrix):
            misfit_per_fitted_gene_per_metacell_of_type = sp.csr_matrix(misfit_per_fitted_gene_per_metacell_of_type)
        if not isinstance(projected_fold_per_fitted_gene_per_metacell_of_type, sp.csr_matrix):
            projected_fold_per_fitted_gene_per_metacell_of_type = sp.csr_matrix(
                projected_fold_per_fitted_gene_per_metacell_of_type
            )
        if not isinstance(type_weights, sp.csr_matrix):
            type_weights = sp.csr_matrix(type_weights)

    ut.log_return("query_metacell_mask_of_type", query_metacell_mask_of_type)
    ut.log_return("fitted_genes_mask_of_type", fitted_genes_mask_of_type)
    ut.log_return("similar_per_metacell_of_type", similar_per_metacell_of_type)
    ut.log_return("misfit_per_fitted_gene_per_metacell_of_type", misfit_per_fitted_gene_per_metacell_of_type)
    ut.log_return("projected_correlation_per_metacell_of_type", projected_correlation_per_metacell_of_type)
    ut.log_return(
        "projected_fold_per_fitted_gene_per_metacell_of_type", projected_fold_per_fitted_gene_per_metacell_of_type
    )
    ut.log_return("weights", type_weights)

    return dict(
        query_metacell_indices_of_type=np.where(query_metacell_mask_of_type)[0],
        fitted_genes_indices_of_type=np.where(fitted_genes_mask_of_type)[0],
        similar_per_metacell_of_type=similar_per_metacell_of_type,
        misfit_per_fitted_gene_per_metacell_of_type=misfit_per_fitted_gene_per_metacell_of_type,
        projected_correlation_per_metacell_of_type=projected_correlation_per_metacell_of_type,
        projected_fold_per_fitted_gene_per_metacell_of_type=projected_fold_per_fitted_gene_per_metacell_of_type,
        weights_per_atlas_per_query_metacell_of_type=type_weights,
    )


def _detect_type_misfit_genes(
    *,
    type_fitted_qdata: AnnData,
    max_projection_fold_factor: float,
    max_projection_noisy_fold_factor: float,
    misfit_min_metacells_fraction: float,
    fitted_genes_mask_of_type: ut.NumpyVector,
) -> bool:
    assert max_projection_fold_factor >= 0
    assert max_projection_noisy_fold_factor >= 0
    assert 0 <= misfit_min_metacells_fraction <= 1

    projected_fold_per_gene_per_metacell = ut.get_vo_proper(type_fitted_qdata, "projected_fold", layout="column_major")
    projected_fold_per_gene_per_metacell = np.abs(projected_fold_per_gene_per_metacell)  # type: ignore

    if ut.has_data(type_fitted_qdata, "projected_noisy_gene"):
        max_projection_fold_factor_per_gene = np.full(
            type_fitted_qdata.n_vars, max_projection_fold_factor, dtype="float32"
        )
        noisy_per_gene = ut.get_v_numpy(type_fitted_qdata, "projected_noisy_gene")
        max_projection_fold_factor_per_gene[noisy_per_gene] += max_projection_noisy_fold_factor
        high_projection_fold_per_gene_per_metacell = (
            projected_fold_per_gene_per_metacell > max_projection_fold_factor_per_gene[np.newaxis, :]
        )
    else:
        high_projection_fold_per_gene_per_metacell = (
            ut.to_numpy_matrix(projected_fold_per_gene_per_metacell) > max_projection_fold_factor
        )
        ut.log_calc("high_projection_fold_per_gene_per_metacell", high_projection_fold_per_gene_per_metacell)

    high_projection_cells_per_gene = ut.sum_per(high_projection_fold_per_gene_per_metacell, per="column")
    ut.log_calc("high_projection_cells_per_gene", high_projection_cells_per_gene, formatter=ut.sizes_description)

    min_high_projection_cells = type_fitted_qdata.n_obs * misfit_min_metacells_fraction
    ut.log_calc("min_high_projection_cells", min_high_projection_cells)

    type_misfit_genes_mask = high_projection_cells_per_gene >= min_high_projection_cells
    ut.log_calc("type_misfit_genes_mask", type_misfit_genes_mask)
    if not np.any(type_misfit_genes_mask):
        return False

    common_gene_index_of_type_fitted_qdata = ut.get_v_numpy(type_fitted_qdata, "common_gene_index_of_qdata")
    fitted_genes_mask_of_type[common_gene_index_of_type_fitted_qdata] = ~type_misfit_genes_mask
    ut.log_calc("fitted_genes_mask_of_type", fitted_genes_mask_of_type)

    return True


@ut.logged()
@ut.timed_call()
def _changed_projected_types(
    *,
    common_adata: AnnData,
    common_qdata: AnnData,
    old_type_of_query_metacells: ut.NumpyVector,
    weights_per_atlas_per_query_metacell: ut.NumpyMatrix,
    atlas_type_property_name: str,
    old_types_per_metacell: List[Set[str]],
) -> bool:
    tl.convey_atlas_to_query(
        adata=common_adata,
        qdata=common_qdata,
        weights=weights_per_atlas_per_query_metacell,
        property_name=atlas_type_property_name,
        to_property_name="projected_type",
    )
    new_type_of_query_metacells = ut.get_o_numpy(common_qdata, "projected_type")

    has_changed = False
    for metacell_index, old_types_of_metacell in enumerate(old_types_per_metacell):
        old_type = old_type_of_query_metacells[metacell_index]
        old_types_of_metacell.add(old_type)

        new_type = new_type_of_query_metacells[metacell_index]
        if new_type not in old_types_of_metacell:
            ut.log_calc(f"metacell: {metacell_index} changed from old type: {old_type} to new type", new_type)
            has_changed = True
        elif new_type != old_type:
            ut.log_calc(f"metacell: {metacell_index} changed from old type: {old_type} to older type", new_type)

    ut.log_return("has_changed", has_changed)
    return has_changed


@ut.timed_call()
@ut.logged()
def _compute_dissimilar_residuals_projection(
    *,
    what: str,
    common_adata: AnnData,
    common_qdata: AnnData,
    weights_per_atlas_per_query_metacell: ut.NumpyMatrix,
    project_fold_normalization: float,
    project_candidates_count: int,
    project_min_candidates_fraction: float,
    project_min_significant_gene_umis: float,
    project_min_usage_weight: float,
    project_max_consistency_fold_factor: float,
    project_max_projection_fold_factor: float,
    project_max_projection_noisy_fold_factor: float,
    project_max_misfit_genes: int,
    min_essential_genes_of_type: Dict[Tuple[str, str], Optional[int]],
    atlas_type_property_name: str,
    top_level_parallel: bool,
    reproducible: bool,
) -> None:
    secondary_type = [""] * common_qdata.n_obs
    dissimilar_mask = ~ut.get_o_numpy(common_qdata, "similar")
    if not np.any(dissimilar_mask):
        ut.set_o_data(common_qdata, "projected_secondary_type", np.array(secondary_type))
        return

    dissimilar_qdata = ut.slice(
        common_qdata, obs=dissimilar_mask, name=".dissimilar", track_var="common_gene_index_of_qdata"
    )

    @ut.timed_call("single_metacell_residuals")
    def _single_metacell_residuals(
        dissimilar_metacell_index: int,
    ) -> Optional[Dict[str, Any]]:
        return _compute_single_metacell_residuals(
            what=what,
            dissimilar_metacell_index=dissimilar_metacell_index,
            common_adata=common_adata,
            dissimilar_qdata=dissimilar_qdata,
            project_fold_normalization=project_fold_normalization,
            project_candidates_count=project_candidates_count,
            project_min_candidates_fraction=project_min_candidates_fraction,
            project_min_significant_gene_umis=project_min_significant_gene_umis,
            project_min_usage_weight=project_min_usage_weight,
            project_max_consistency_fold_factor=project_max_consistency_fold_factor,
            project_max_projection_fold_factor=project_max_projection_fold_factor,
            project_max_projection_noisy_fold_factor=project_max_projection_noisy_fold_factor,
            project_max_misfit_genes=project_max_misfit_genes,
            min_essential_genes_of_type=min_essential_genes_of_type,
            atlas_type_property_name=atlas_type_property_name,
            reproducible=reproducible,
        )

    if top_level_parallel:
        results = ut.parallel_map(_single_metacell_residuals, dissimilar_qdata.n_obs)
    else:
        results = [
            _single_metacell_residuals(dissimilar_metacell_index)
            for dissimilar_metacell_index in range(dissimilar_qdata.n_obs)
        ]

    similar_per_metacell = ut.get_o_numpy(common_qdata, "similar").copy()
    primary_type_per_metacell = ut.get_o_numpy(common_qdata, "projected_type").copy()
    secondary_type_per_metacell = [""] * common_qdata.n_obs
    fitted_per_gene_per_metacell = ut.to_numpy_matrix(ut.get_vo_proper(common_qdata, "fitted"), copy=True)
    misfit_per_gene_per_metacell = ut.to_numpy_matrix(ut.get_vo_proper(common_qdata, "misfit"), copy=True)
    projected_correlation_per_metacell = ut.get_o_numpy(common_qdata, "projected_correlation").copy()
    projected_fold_per_gene_per_metacell = ut.to_numpy_matrix(
        ut.get_vo_proper(common_qdata, "projected_fold"), copy=True
    )

    def _collect_metacell_residuals(
        dissimilar_metacell_index: int,
        primary_type: str,
        secondary_type: str,
        fitted_genes_mask: ut.NumpyVector,
        misfit_genes_mask: ut.NumpyVector,
        projected_correlation: float,
        projected_fold: ut.NumpyVector,
        weights: ut.ProperMatrix,
    ) -> None:
        similar_per_metacell[dissimilar_metacell_index] = True
        primary_type_per_metacell[dissimilar_metacell_index] = primary_type
        secondary_type_per_metacell[dissimilar_metacell_index] = secondary_type
        fitted_per_gene_per_metacell[dissimilar_metacell_index, :] = fitted_genes_mask
        misfit_per_gene_per_metacell[dissimilar_metacell_index, fitted_genes_mask] = misfit_genes_mask
        projected_correlation_per_metacell[dissimilar_metacell_index] = projected_correlation
        projected_fold_per_gene_per_metacell[dissimilar_metacell_index, fitted_genes_mask] = projected_fold
        weights_per_atlas_per_query_metacell[dissimilar_metacell_index, :] = weights

    for dissimilar_metacell_index, result in enumerate(results):
        if result is not None:
            _collect_metacell_residuals(dissimilar_metacell_index, **result)

    ut.set_o_data(common_qdata, "projected_type", primary_type_per_metacell)
    ut.set_o_data(common_qdata, "projected_secondary_type", np.array(secondary_type_per_metacell))
    ut.set_o_data(common_qdata, "similar", similar_per_metacell)
    ut.set_o_data(common_qdata, "projected_correlation", projected_correlation_per_metacell)

    ut.set_vo_data(common_qdata, "fitted", fitted_per_gene_per_metacell)
    ut.set_vo_data(common_qdata, "misfit", misfit_per_gene_per_metacell)
    ut.set_vo_data(common_qdata, "projected_fold", projected_fold_per_gene_per_metacell)


@ut.timed_call()
@ut.logged()
def _compute_single_metacell_residuals(  # pylint: disable=too-many-statements
    *,
    what: str,
    dissimilar_metacell_index: int,
    common_adata: AnnData,
    dissimilar_qdata: AnnData,
    project_fold_normalization: float,
    project_candidates_count: int,
    project_min_candidates_fraction: float,
    project_min_significant_gene_umis: float,
    project_min_usage_weight: float,
    project_max_consistency_fold_factor: float,
    project_max_projection_fold_factor: float,
    project_max_projection_noisy_fold_factor: float,
    project_max_misfit_genes: int,
    min_essential_genes_of_type: Dict[Tuple[str, str], Optional[int]],
    atlas_type_property_name: str,
    reproducible: bool,
) -> Optional[Dict[str, Any]]:
    primary_type = ut.get_o_numpy(dissimilar_qdata, "projected_type")[dissimilar_metacell_index]
    secondary_type = ""

    repeat = 0
    while True:
        repeat += 1
        ut.log_calc("residuals repeat", repeat)
        ut.log_calc("primary_type", primary_type)
        ut.log_calc("secondary_type", secondary_type)

        fitted_genes_mask = ut.get_v_numpy(dissimilar_qdata, f"fitted_gene_of_{primary_type}")
        if secondary_type != "":
            fitted_genes_mask = fitted_genes_mask | ut.get_v_numpy(dissimilar_qdata, f"fitted_gene_of_{secondary_type}")

        name = f".fitted.{dissimilar_metacell_index}"
        fitted_adata = ut.slice(common_adata, vars=fitted_genes_mask, name=name)
        _normalize_corrected_fractions(fitted_adata, what)

        metacell_fitted_qdata = ut.slice(
            dissimilar_qdata, obs=[dissimilar_metacell_index], vars=fitted_genes_mask, name=name
        )
        _normalize_corrected_fractions(metacell_fitted_qdata, "corrected_fraction")
        corrected_fractions = ut.get_vo_proper(metacell_fitted_qdata, "corrected_fraction")

        second_anchor_indices: List[int] = []
        weights = tl.compute_projection(
            adata=fitted_adata,
            qdata=metacell_fitted_qdata,
            fold_normalization=project_fold_normalization,
            min_significant_gene_umis=project_min_significant_gene_umis,
            max_consistency_fold_factor=project_max_consistency_fold_factor,
            candidates_count=project_candidates_count,
            min_candidates_fraction=project_min_candidates_fraction,
            min_usage_weight=project_min_usage_weight,
            reproducible=reproducible,
            second_anchor_indices=second_anchor_indices,
        )
        projected_fractions = ut.get_vo_proper(metacell_fitted_qdata, "projected_fraction")
        first_anchor_weights = weights.copy()
        first_anchor_weights[:, second_anchor_indices] = 0.0

        if len(second_anchor_indices) == 0:
            new_secondary_type = ""
        else:
            second_anchor_weights = weights - first_anchor_weights  # type: ignore

            tl.convey_atlas_to_query(
                adata=fitted_adata,
                qdata=metacell_fitted_qdata,
                weights=second_anchor_weights,
                property_name=atlas_type_property_name,
                to_property_name="projected_secondary_type",
            )
            new_secondary_type = ut.get_o_numpy(metacell_fitted_qdata, "projected_secondary_type")[0]

        if np.sum(first_anchor_weights.data) == 0:
            new_primary_type = new_secondary_type
            new_secondary_type = ""
        else:
            tl.convey_atlas_to_query(
                adata=fitted_adata,
                qdata=metacell_fitted_qdata,
                weights=first_anchor_weights,
                property_name=atlas_type_property_name,
                to_property_name="projected_type",
            )
            new_primary_type = ut.get_o_numpy(metacell_fitted_qdata, "projected_type")[0]

        if repeat > 2 or (new_primary_type == primary_type and new_secondary_type == secondary_type):
            break

        primary_type = new_primary_type
        secondary_type = new_secondary_type

    projected_correlation = ut.pairs_corrcoef_rows(
        ut.to_numpy_matrix(corrected_fractions), ut.to_numpy_matrix(projected_fractions), reproducible=reproducible
    )[0]

    tl.compute_projected_folds(
        metacell_fitted_qdata,
        fold_normalization=project_fold_normalization,
        min_significant_gene_umis=project_min_significant_gene_umis,
    )
    projected_fold = ut.to_numpy_vector(ut.get_vo_proper(metacell_fitted_qdata, "projected_fold"))

    essential_genes_properties = [f"essential_gene_of_{primary_type}"]
    if secondary_type == "":
        min_essential_genes = min_essential_genes_of_type[(primary_type, primary_type)]
    else:
        min_essential_genes = min_essential_genes_of_type[(primary_type, secondary_type)]
        essential_genes_properties.append(f"essential_gene_of_{secondary_type}")

    tl.compute_similar_query_metacells(
        metacell_fitted_qdata,
        max_projection_fold_factor=project_max_projection_fold_factor,
        max_projection_noisy_fold_factor=project_max_projection_noisy_fold_factor,
        max_misfit_genes=project_max_misfit_genes,
        essential_genes_property=essential_genes_properties,
        min_essential_genes=min_essential_genes,
    )

    similar = ut.get_o_numpy(metacell_fitted_qdata, "similar")[0]
    ut.log_return("similar", False)
    if not similar:
        return None

    misfit_genes_mask = ut.to_numpy_vector(ut.get_vo_proper(metacell_fitted_qdata, "misfit")[0, :])

    ut.log_return("primary_type", primary_type)
    ut.log_return("secondary_type", secondary_type)
    ut.log_return("fitted_genes_mask", fitted_genes_mask)
    ut.log_return("misfit_genes_mask", misfit_genes_mask)
    ut.log_return("projected_correlation", projected_correlation)
    ut.log_return("projected_fold", projected_fold)

    return dict(
        primary_type=primary_type,
        secondary_type=secondary_type,
        fitted_genes_mask=fitted_genes_mask,
        misfit_genes_mask=misfit_genes_mask,
        projected_correlation=projected_correlation,
        projected_fold=projected_fold,
        weights=ut.to_numpy_vector(weights),
    )


def _normalize_corrected_fractions(adata: AnnData, what: str) -> None:
    fractions_data = ut.get_vo_proper(adata, what, layout="row_major")
    corrected_data = ut.fraction_by(fractions_data, by="row")
    ut.set_vo_data(adata, "corrected_fraction", corrected_data)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def outliers_projection_pipeline(
    what: str = "__x__",
    *,
    adata: AnnData,
    odata: AnnData,
    fold_normalization: float = pr.outliers_fold_normalization,
    project_min_significant_gene_umis: int = pr.project_min_significant_gene_umis,
    reproducible: bool,
) -> None:
    """
    Project outliers on an atlas.

    **Returns**

    Sets the following in ``odata``:

    Per-Observation (Cell) Annotations

        ``atlas_most_similar``
            For each observation (outlier), the index of the "most similar" atlas metacell.

    Per-Variable Per-Observation (Gene-Cell) Annotations

        ``atlas_most_similar_fold``
            The fold factor between the outlier gene expression and their expression in the most similar atlas metacell
            (unless the value is too low to be of interest, in which case it will be zero).

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.quality.compute_outliers_matches` using the ``fold_normalization``
       (default: {fold_normalization}) and ``reproducible``.

    2. Invoke :py:func:`metacells.tools.quality.compute_outliers_fold_factors` using the ``fold_normalization``
       (default: {fold_normalization}).
    """
    if list(odata.var_names) != list(adata.var_names):
        atlas_genes_list = list(adata.var_names)
        query_genes_list = list(odata.var_names)
        common_genes_list = list(sorted(set(atlas_genes_list) & set(query_genes_list)))
        atlas_common_gene_indices = np.array([atlas_genes_list.index(gene) for gene in common_genes_list])
        query_common_gene_indices = np.array([query_genes_list.index(gene) for gene in common_genes_list])
        common_adata = ut.slice(adata, name=".common", vars=atlas_common_gene_indices)
        common_odata = ut.slice(
            odata,
            name=".common",
            vars=query_common_gene_indices,
        )
    else:
        common_adata = adata
        common_odata = odata

    tl.compute_outliers_matches(
        what,
        adata=common_odata,
        gdata=common_adata,
        most_similar="atlas_most_similar",
        value_normalization=fold_normalization,
        reproducible=reproducible,
    )

    tl.compute_outliers_fold_factors(
        what,
        adata=common_odata,
        gdata=common_adata,
        most_similar="atlas_most_similar",
        min_gene_total=project_min_significant_gene_umis,
    )

    if list(odata.var_names) != list(adata.var_names):
        atlas_most_similar = ut.get_o_numpy(common_odata, "atlas_most_similar")
        ut.set_o_data(odata, "atlas_most_similar", atlas_most_similar)

        common_folds = ut.get_vo_proper(common_odata, "atlas_most_similar_fold")
        atlas_most_similar_fold = np.zeros(odata.shape, dtype="float32")
        atlas_most_similar_fold[:, query_common_gene_indices] = common_folds
        ut.set_vo_data(odata, "atlas_most_similar_fold", sp.csr_matrix(atlas_most_similar_fold))
