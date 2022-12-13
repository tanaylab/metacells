"""
Projection
----------
"""

from re import Pattern
from typing import Any
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

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
    ignored_gene_names: Optional[Collection[str]] = None,
    ignored_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    ignore_atlas_insignificant_genes: bool = pr.ignore_atlas_insignificant_genes,
    ignore_query_insignificant_genes: bool = pr.ignore_query_insignificant_genes,
    ignore_atlas_lateral_genes: bool = pr.ignore_atlas_lateral_genes,
    ignore_atlas_bystander_genes: bool = pr.ignore_atlas_bystander_genes,
    ignore_query_lateral_genes: bool = pr.ignore_query_lateral_genes,
    ignore_query_bystander_genes: bool = pr.ignore_query_bystander_genes,
    misfit_min_metacells_fraction: float = pr.misfit_min_metacells_fraction,
    project_fold_normalization: float = pr.project_fold_normalization,
    project_candidates_count: int = pr.project_candidates_count,
    project_min_significant_gene_value: float = pr.project_min_significant_gene_value,
    project_min_usage_weight: float = pr.project_min_usage_weight,
    project_max_consistency_fold_factor: float = pr.project_max_consistency_fold_factor,
    project_max_projection_fold_factor: float = pr.project_max_projection_fold_factor,
    project_max_dissimilar_genes: int = pr.project_max_dissimilar_genes,
    min_entry_project_fold_factor: float = pr.min_entry_project_fold_factor,
    project_min_similar_essential_genes_fraction: Optional[float] = pr.project_min_similar_essential_genes_fraction,
    project_abs_folds: bool = pr.project_abs_folds,
    ignored_gene_names_of_type: Optional[Dict[str, Collection[str]]] = None,
    ignored_gene_patterns_of_type: Optional[Dict[str, Collection[str]]] = None,
    atlas_type_property_name: str = "type",
    project_corrections: bool = pr.project_corrections,
    project_min_corrected_gene_correlation: float = pr.project_min_corrected_gene_correlation,
    project_min_corrected_gene_factor: float = pr.project_min_corrected_gene_factor,
    renormalize_query_by_atlas: bool = pr.renormalize_query_by_atlas,
    renormalize_var_annotations: Optional[Dict[str, Any]] = None,
    renormalize_layers: Optional[Dict[str, Any]] = None,
    renormalize_varp_annotations: Optional[Dict[str, Any]] = None,
    reproducible: bool,
    top_level_parallel: bool = True,
) -> Tuple[ut.CompressedMatrix, AnnData]:
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

        ``atlas_significant_gene`` (``ignore_atlas_insignificant_genes`` requires an atlas ``significant_gene`` mask)
            A boolean mask indicating whether the gene is considered significant in the atlas.

        ``atlas_lateral_gene`` (``ignore_atlas_lateral_genes`` requires an atlas has ``lateral_gene`` mask)
            A boolean mask indicating whether the gene was forbidden from being a feature in the atlas (and hence
            ignored).

        ``atlas_bystander_gene`` (``ignore_atlas_bystander_genes`` requires an atlas has ``bystander_gene`` mask)
            A boolean mask indicating whether the gene was forbidden from being a feature in the atlas (and hence
            ignored).

        ``ignored_gene``
            A boolean mask indicating whether the gene was ignored by the projection (for any reason).

        ``essential_gene_of_<type>``
            For each query metacell type, a boolean mask indicating whether properly projecting this gene is essential
            for assigning this type to a query metacell. This is copied from the atlas if
            ``project_min_similar_essential_genes_fraction`` is specified.

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
            If ``project_min_similar_essential_genes_fraction`` was specified, the number of essential genes for the
            metacell projected type(s).

        ``similar_essential_genes_count``
            If ``project_min_similar_essential_genes_fraction`` was specified, the number of essential genes for the
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
       ``ignored_gene_patterns``. If ``ignore_atlas_insignificant_genes`` (default:
       ``ignore_atlas_insignificant_genes``), ignore genes the atlas did not mark as significant (and store the
       ``atlas_significant_gene`` mask). If ``ignore_query_insignificant_genes`` (default:
       ``ignore_query_insignificant_genes``), ignore genes the query did not mark as significant. If
       ``ignore_atlas_lateral_genes`` (default: {ignore_atlas_lateral_genes}), also ignore the ``lateral_gene`` of the
       atlas (and store them as an ``atlas_lateral_gene`` mask). If ``ignore_atlas_bystander_genes`` (default:
       {ignore_atlas_bystander_genes}), also ignore the ``bystander_gene`` of the atlas (and store them as an
       ``atlas_bystander_gene`` mask). If ``ignore_query_lateral_genes`` (default: {ignore_query_lateral_genes}), also
       ignore the ``lateral_gene`` of the query. If ``ignore_query_bystander_genes`` (default:
       {ignore_query_bystander_genes}), also ignore the ``bystander_gene`` of the query. All these genes are ignored by
       the following code. In addition, ignore any genes marked in ``manually_ignored_gene``, if this annotation exists
       in the query.

    2. Invoke :py:func:`metacells.tools.project.project_query_onto_atlas` and
       :py:func:`metacells.tools.project.compute_query_projection` to project each query metacell onto the atlas, using
       the ``project_fold_normalization`` (default: {project_fold_normalization}),
       ``project_min_significant_gene_value`` (default: {project_min_significant_gene_value}),
       ``project_candidates_count`` (default: {project_candidates_count}), ``project_min_usage_weight`` (default:
       {project_min_usage_weight}), ``project_abs_folds`` (default: {project_abs_folds}) and ``reproducible``.

    3. Invoke :py:func:`metacells.tools.quality.compute_projected_fold_factors` to compute the significant
       fold factors between the query and its projection, using the ``project_fold_normalization`` (default:
       {project_fold_normalization}), ``project_min_significant_gene_value`` (default:
       {project_min_significant_gene_value}) and ``project_abs_folds`` (default: {project_abs_folds}).

    4. Correlate the expression levels of each gene between the query and projection. If this is at least
       ``project_min_corrected_gene_correlation`` (default: {project_min_corrected_gene_correlation}), compute the ratio
       between the mean expression of the gene in the projection and the query. If this is at most
       1/(1+``project_min_corrected_gene_factor``) or at least (1+``project_min_corrected_gene_factor``) (default:
       {project_min_corrected_gene_factor}), then multiply the gene's value by this factor so its level would match the
       atlas. If any genes were ignored or corrected, then repeat steps 1-4. However, if not ``project_corrections``,
       only record the correction factor and proceed without actually performing the correction.

    5. Invoke :py:func:`metacells.tools.project.project_atlas_to_query` to assign a projected type to each of the
       query metacells based on the ``atlas_type_property_name`` (default: {atlas_type_property_name}).

    Then, for each type of query metacells:

    If ``top_level_parallel`` (default: ``top_level_parallel``), do this in parallel. This seems to work better than
    doing this serially and using parallelism within each type. However this is still inefficient compared to using both
    types of parallelism at once, which the code currently can't do without non-trivial coding (this would have been
    trivial in Julia...).

    6. Enhance the mask of type-specific ignored genes with any genes marked in ``manually_ignored_gene_of_<type>``, if
       this annotation exists in the query.

    7. Invoke :py:func:`metacells.tools.project.project_query_onto_atlas` to project each query metacell of the type
       onto the atlas, using the type-specific ignored genes in addition to the global ignored genes. Note that even
       though we only look at query metacells (tentatively) assigned the type, their projection on the atlas may use
       metacells of any type.

    8. Invoke :py:func:`metacells.tools.project.compute_query_projection` and
       :py:func:`metacells.tools.quality.compute_projected_fold_factors` to compute the significant fold factors between
       the query and its projection.

    9. Invoke :py:func:`metacells.tools.project.find_misfit_genes` to identify type-specific misfit genes. If any such
       genes are found, add them to the type-specific ignored genes (instead of the global misfit genes list) and repeat
       steps 6-9.

    10. Invoke :py:func:`metacells.tools.quality.compute_similar_query_metacells` to verify which query metacells
        ended up being sufficiently similar to their projection, using the ``essential_gene_of_<type>`` if needed.

    And then:

    11. Invoke :py:func:`metacells.tools.project.project_atlas_to_query` to assign an updated projected type to each of
        the query metacells based on the ``atlas_type_property_name`` (default: {atlas_type_property_name}). If this
        changed the type assigned to any query metacell, repeat steps 6-10 (but do these steps no more than 3 times).

    For each query metacell that ended up being dissimilar:

    12. Update the list of ignored genes for the metacell based on the ``ignored_gene_of_<type>`` for both the primary
        (initially from the above) query metacell type and the secondary query metacell type (initially empty).

    13. Invoke :py:func:`metacells.tools.project.project_query_onto_atlas` just for this metacell, but allow the
        projection to use a secondary location in the atlas based on the residuals of the atlas metacells relative to
        the primary query projection.

    14. Invoke :py:func:`metacells.tools.project.project_atlas_to_query` twice, once for the weights of the primary
        location and once for the weights of the secondary location, to obtain a primary and secondary type for
        the query metacell. If these have changed, repeat steps 12-14 (but do these steps no more than 3 times; note
        will will always do them twice as the 1st run will generate some non-empty secondary type).

    15. Invoke :py:func:`metacells.tools.project.compute_query_projection`,
        :py:func:`metacells.tools.quality.compute_projected_fold_factors` and
        :py:func:`metacells.tools.quality.compute_similar_query_metacells` to update the projection and evaluation of
        the query metacell. If it is now similar, mark it as a doublet or a mixture depending on whether the primary and
        the secondary types are different or identical.

    16. Invoke :py:func:`metacells.utilities.computation.sparsify_matrix` to preserve only the significant
        ``projected_fold`` values, using ``project_max_projection_fold_factor`` (default:
        {project_max_projection_fold_factor}) and ``min_entry_project_fold_factor`` (default:
        ``min_entry_project_fold_factor``).

    17. Invoke :py:func:`metacells.tools.quality.compute_metacells_projection_correlation` to compute the correlation
        between the projected and the corrected UMIs of each query metacell.

    18. If ``renormalize_query_by_atlas`` (default: {renormalize_query_by_atlas}), then invoke
        :py:func:`metacells.tools.project.renormalize_query_by_atlas` using the ``renormalize_var_annotations``,
        ``renormalize_layers`` and ``renormalize_varp_annotations``, if any, to add an ``ATLASNORM`` pseudo-gene so that
        the fraction out of the total UMIs in the query of the genes common to the atlas would be the same on average as
    """
    assert project_min_corrected_gene_factor >= 0

    qdata = qdata.copy()

    _fill_manually_ignored_genes(
        qdata,
        ignored_gene_names=ignored_gene_names,
        ignored_gene_patterns=ignored_gene_patterns,
        ignored_gene_names_of_type=ignored_gene_names_of_type,
        ignored_gene_patterns_of_type=ignored_gene_patterns_of_type,
    )

    common_adata, common_qdata, min_essential_genes_of_type = _common_data(
        adata=adata,
        qdata=qdata,
        project_max_projection_fold_factor=project_max_projection_fold_factor,
        project_max_dissimilar_genes=project_max_dissimilar_genes,
        project_min_similar_essential_genes_fraction=project_min_similar_essential_genes_fraction,
        ignore_atlas_insignificant_genes=ignore_atlas_insignificant_genes,
        ignore_atlas_lateral_genes=ignore_atlas_lateral_genes,
        ignore_atlas_bystander_genes=ignore_atlas_bystander_genes,
        atlas_type_property_name=atlas_type_property_name,
    )

    atlas_total_common_umis = ut.get_o_numpy(common_adata, what, sum=True)

    query_total_common_umis = _compute_preliminary_projection(
        what=what,
        qdata=qdata,
        common_adata=common_adata,
        common_qdata=common_qdata,
        atlas_total_common_umis=atlas_total_common_umis,
        ignore_atlas_insignificant_genes=ignore_atlas_insignificant_genes,
        ignore_query_insignificant_genes=ignore_query_insignificant_genes,
        ignore_atlas_lateral_genes=ignore_atlas_lateral_genes,
        ignore_atlas_bystander_genes=ignore_atlas_bystander_genes,
        ignore_query_lateral_genes=ignore_query_lateral_genes,
        ignore_query_bystander_genes=ignore_query_bystander_genes,
        project_fold_normalization=project_fold_normalization,
        project_min_significant_gene_value=project_min_significant_gene_value,
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
        atlas_total_common_umis=atlas_total_common_umis,
        query_total_common_umis=query_total_common_umis,
        misfit_min_metacells_fraction=misfit_min_metacells_fraction,
        project_fold_normalization=project_fold_normalization,
        project_candidates_count=project_candidates_count,
        project_min_significant_gene_value=project_min_significant_gene_value,
        project_min_usage_weight=project_min_usage_weight,
        project_max_consistency_fold_factor=project_max_consistency_fold_factor,
        project_max_projection_fold_factor=project_max_projection_fold_factor,
        project_max_dissimilar_genes=project_max_dissimilar_genes,
        min_essential_genes_of_type=min_essential_genes_of_type,
        project_abs_folds=project_abs_folds,
        atlas_type_property_name=atlas_type_property_name,
        top_level_parallel=top_level_parallel,
        reproducible=reproducible,
    )

    weights = _compute_dissimilar_residuals_projection(
        what=what,
        weights=weights,
        common_adata=common_adata,
        common_qdata=common_qdata,
        atlas_total_common_umis=atlas_total_common_umis,
        query_total_common_umis=query_total_common_umis,
        project_fold_normalization=project_fold_normalization,
        project_candidates_count=project_candidates_count,
        project_min_significant_gene_value=project_min_significant_gene_value,
        project_min_usage_weight=project_min_usage_weight,
        project_max_consistency_fold_factor=project_max_consistency_fold_factor,
        project_max_projection_fold_factor=project_max_projection_fold_factor,
        project_max_dissimilar_genes=project_max_dissimilar_genes,
        min_essential_genes_of_type=min_essential_genes_of_type,
        project_abs_folds=project_abs_folds,
        atlas_type_property_name=atlas_type_property_name,
        top_level_parallel=top_level_parallel,
        reproducible=reproducible,
    )

    _convey_query_common_to_full_data(
        what=what,
        adata=adata,
        qdata=qdata,
        common_qdata=common_qdata,
        project_max_projection_fold_factor=project_max_projection_fold_factor,
        min_entry_project_fold_factor=min_entry_project_fold_factor,
        project_min_similar_essential_genes_fraction=project_min_similar_essential_genes_fraction,
        atlas_type_property_name=atlas_type_property_name,
        project_abs_folds=project_abs_folds,
    )

    tl.compute_metacells_projection_correlation(qdata, reproducible=reproducible)

    qdata = _renormalize_query(
        adata=adata,
        qdata=qdata,
        atlas_type_property_name=atlas_type_property_name,
        renormalize_query_by_atlas=renormalize_query_by_atlas,
        renormalize_var_annotations=renormalize_var_annotations,
        renormalize_layers=renormalize_layers,
        renormalize_varp_annotations=renormalize_varp_annotations,
    )

    return sp.csr_matrix(weights), qdata


@ut.logged()
@ut.timed_call()
def _fill_manually_ignored_genes(
    qdata: AnnData,
    ignored_gene_names: Optional[Collection[str]],
    ignored_gene_patterns: Optional[Collection[Union[str, Pattern]]],
    ignored_gene_names_of_type: Optional[Dict[str, Collection[str]]],
    ignored_gene_patterns_of_type: Optional[Dict[str, Collection[str]]],
) -> None:
    if ignored_gene_names is not None or ignored_gene_patterns is not None:
        tl.find_named_genes(qdata, to="manually_ignored_gene", names=ignored_gene_names, patterns=ignored_gene_patterns)

    if ignored_gene_names_of_type is None:
        ignored_gene_names_of_type = {}
    if ignored_gene_patterns_of_type is None:
        ignored_gene_patterns_of_type = {}

    ignored_type_names = list(set(ignored_gene_names_of_type.keys()) | set(ignored_gene_patterns_of_type.keys()))
    for type_name in ignored_type_names:
        tl.find_named_genes(
            qdata,
            to=f"manually_ignored_gene_of_{type_name}",
            names=ignored_gene_names_of_type.get(type_name),
            patterns=ignored_gene_patterns_of_type.get(type_name),
        )


@ut.logged()
@ut.timed_call()
def _common_data(
    *,
    qdata: AnnData,
    adata: AnnData,
    project_max_projection_fold_factor: float,
    project_max_dissimilar_genes: int,
    project_min_similar_essential_genes_fraction: Optional[float],
    ignore_atlas_insignificant_genes: bool,
    ignore_atlas_lateral_genes: bool,
    ignore_atlas_bystander_genes: bool,
    atlas_type_property_name: str,
) -> Tuple[AnnData, AnnData, Optional[Dict[str, Optional[float]]]]:
    ut.set_m_data(qdata, "project_max_projection_fold_factor", project_max_projection_fold_factor)
    ut.set_m_data(qdata, "project_max_dissimilar_genes", project_max_dissimilar_genes)

    if list(qdata.var_names) != list(adata.var_names):
        atlas_genes_list = list(adata.var_names)
        query_genes_list = list(qdata.var_names)
        common_genes_list = list(sorted(set(atlas_genes_list) & set(query_genes_list)))
        assert len(common_genes_list) > 0
        atlas_gene_indices = np.array([atlas_genes_list.index(gene) for gene in common_genes_list])
        query_gene_indices = np.array([query_genes_list.index(gene) for gene in common_genes_list])
        common_adata = ut.slice(adata, name=".common", vars=atlas_gene_indices)
        common_qdata = ut.slice(
            qdata,
            name=".common",
            vars=query_gene_indices,
            track_obs="full_metacell_index_of_qdata",
            track_var="full_gene_index_of_qdata",
        )
    else:
        common_adata = adata
        common_qdata = qdata
        ut.set_o_data(common_qdata, "full_metacell_index_of_qdata", np.arange(qdata.n_obs))
        ut.set_v_data(common_qdata, "full_gene_index_of_qdata", np.arange(qdata.n_vars))

    assert list(common_qdata.var_names) == list(common_adata.var_names)

    ut.set_v_data(qdata, "projected_correlation", np.zeros(qdata.n_vars, dtype="float32"))
    ut.set_v_data(qdata, "correlated_gene", np.zeros(qdata.n_vars, dtype="bool"))
    ut.set_o_data(common_qdata, "common_cell_index_of_qdata", np.arange(common_qdata.n_obs))
    ut.set_v_data(common_qdata, "common_gene_index_of_qdata", np.arange(common_qdata.n_vars))
    ut.set_v_data(common_qdata, "correction_factor", np.full(common_qdata.n_vars, 1.0, dtype="float32"))
    ut.set_v_data(common_qdata, "atlas_gene", np.full(common_qdata.n_vars, True))

    if ignore_atlas_insignificant_genes and ut.has_data(common_adata, "significant_gene"):
        atlas_significant_mask = ut.get_v_numpy(common_adata, "significant_gene")
        ut.set_v_data(common_qdata, "atlas_significant_gene", atlas_significant_mask)

    if ignore_atlas_lateral_genes and ut.has_data(common_adata, "lateral_gene"):
        atlas_lateral_mask = ut.get_v_numpy(common_adata, "lateral_gene")
        ut.set_v_data(common_qdata, "atlas_lateral_gene", atlas_lateral_mask)

    if ignore_atlas_bystander_genes and ut.has_data(common_adata, "bystander_gene"):
        atlas_bystander_mask = ut.get_v_numpy(common_adata, "bystander_gene")
        ut.set_v_data(common_qdata, "atlas_bystander_gene", atlas_bystander_mask)

    min_essential_genes_of_type: Optional[Dict[str, Optional[float]]] = None
    if project_min_similar_essential_genes_fraction is not None:
        type_names = np.unique(ut.get_o_numpy(common_adata, atlas_type_property_name))
        for type_name in type_names:
            if type_name != "Outliers" and ut.has_data(common_adata, f"essential_gene_of_{type_name}"):
                min_essential_genes_of_type = {}
                break

    if min_essential_genes_of_type is not None:
        for type_name in type_names:
            if type_name == "Outliers":
                min_essential_genes_of_type[type_name] = None
            else:
                essential_genes_name = f"essential_gene_of_{type_name}"
                essential_genes_mask = ut.get_v_numpy(common_adata, essential_genes_name)
                ut.set_v_data(common_qdata, essential_genes_name, essential_genes_mask)
                min_essential_genes_count = project_min_similar_essential_genes_fraction * np.sum(essential_genes_mask)
                assert min_essential_genes_count > 0
                min_essential_genes_of_type[type_name] = min_essential_genes_count

    return common_adata, common_qdata, min_essential_genes_of_type


def _final_primary_of(primary: str, secondary: str, is_similar: bool) -> str:
    if not is_similar:
        return "Dissimilar"
    if secondary == primary:
        return "Mixture"
    if secondary != "":
        return "Doublet"
    return primary


def _final_secondary_of(primary: str, secondary: str, is_similar: bool) -> str:
    if not is_similar:
        if secondary not in (primary, ""):
            return f"{primary};{secondary}"
        return primary
    if secondary == primary:
        return primary
    if secondary != "":
        return f"{primary};{secondary}"
    return ""


@ut.logged()
@ut.timed_call()
def _convey_query_common_to_full_data(
    *,
    what: str,
    adata: AnnData,
    qdata: AnnData,
    common_qdata: AnnData,
    project_max_projection_fold_factor: float,
    min_entry_project_fold_factor: float,
    project_min_similar_essential_genes_fraction: Optional[float],
    atlas_type_property_name: str,
    project_abs_folds: bool,
) -> None:
    similar_mask = ut.get_o_numpy(common_qdata, "similar")
    primary_type = ut.get_o_numpy(common_qdata, "projected_type")
    secondary_type = ut.get_o_numpy(common_qdata, "projected_secondary_type")

    final_primary_type = np.array(
        [
            _final_primary_of(primary, secondary, is_similar)
            for primary, secondary, is_similar in zip(primary_type, secondary_type, similar_mask)
        ],
        dtype="U",
    )

    final_secondary_type = np.array(
        [
            _final_secondary_of(primary, secondary, is_similar)
            for primary, secondary, is_similar in zip(primary_type, secondary_type, similar_mask)
        ],
        dtype="U",
    )

    ut.set_o_data(qdata, "projected_type", final_primary_type)
    ut.set_o_data(qdata, "projected_secondary_type", final_secondary_type)

    if id(qdata) == id(common_qdata):
        for data_name in ["full_metacell_index_of_qdata", "common_cell_index_of_qdata"]:
            del qdata.obs[data_name]

        for data_name in ["full_gene_index_of_qdata", "common_gene_index_of_qdata"]:
            del qdata.var[data_name]

        return

    assert common_qdata.n_obs == qdata.n_obs
    ut.set_o_data(qdata, "similar", similar_mask)

    dissimilar_genes_count = ut.get_o_numpy(common_qdata, "dissimilar_genes_count")
    ut.log_calc("max dissimilar_genes_count", np.max(dissimilar_genes_count))
    ut.set_o_data(qdata, "dissimilar_genes_count", dissimilar_genes_count, formatter=ut.mask_description)

    full_gene_index_of_common_qdata = ut.get_v_numpy(common_qdata, "full_gene_index_of_qdata")

    for (data_name, dtype, default) in [
        ("atlas_gene", "bool", False),
        ("atlas_lateral_gene", "bool", False),
        ("atlas_bystander_gene", "bool", False),
        ("atlas_significant_gene", "bool", False),
        ("ignored_gene", "bool", True),
        ("correction_factor", "float32", 1.0),
    ]:
        if ut.has_data(common_qdata, data_name):
            full_data = np.full(qdata.n_vars, default, dtype=dtype)
            full_data[full_gene_index_of_common_qdata] = ut.get_v_numpy(common_qdata, data_name)
            ut.set_v_data(qdata, data_name, full_data)

    all_types = np.unique(ut.get_o_numpy(adata, atlas_type_property_name))
    for type_name in all_types:
        data_names = ["ignored_gene", "misfit_gene"]
        if project_min_similar_essential_genes_fraction is not None:
            data_names.append("essential_gene")
        for data_name in data_names:
            type_data_name = f"{data_name}_of_{type_name}"
            full_data = np.zeros(qdata.n_vars, dtype="bool")
            full_data[full_gene_index_of_common_qdata] = ut.get_v_numpy(common_qdata, type_data_name)
            ut.set_v_data(qdata, type_data_name, full_data)

    common_corrected_data = ut.to_numpy_matrix(ut.get_vo_proper(common_qdata, what)).copy()
    full_corrected_data = ut.to_numpy_matrix(ut.get_vo_proper(qdata, what)).copy()
    full_corrected_data[:, full_gene_index_of_common_qdata] = common_corrected_data
    ut.set_vo_data(qdata, what, full_corrected_data)

    common_projected_fold = ut.get_vo_proper(common_qdata, "projected_fold")
    full_projected_fold = np.zeros(qdata.shape, dtype="float32")
    full_projected_fold[:, full_gene_index_of_common_qdata] = common_projected_fold
    sparse_projected_fold = ut.sparsify_matrix(
        full_projected_fold,
        min_column_max_value=project_max_projection_fold_factor,
        min_entry_value=min_entry_project_fold_factor,
        abs_values=project_abs_folds,
    )
    ut.set_vo_data(qdata, "projected_fold", sp.csr_matrix(sparse_projected_fold))

    common_projected = ut.get_vo_proper(common_qdata, "projected")
    full_projected = np.zeros(qdata.shape, dtype="float32")
    full_projected[:, full_gene_index_of_common_qdata] = ut.to_numpy_matrix(common_projected)
    ut.set_vo_data(qdata, "projected", full_projected)


@ut.logged()
@ut.timed_call()
def _renormalize_query(
    *,
    adata: AnnData,
    qdata: AnnData,
    atlas_type_property_name: str,
    renormalize_query_by_atlas: bool,
    renormalize_var_annotations: Optional[Dict[str, Any]],
    renormalize_layers: Optional[Dict[str, Any]],
    renormalize_varp_annotations: Optional[Dict[str, Any]],
) -> AnnData:
    if not renormalize_query_by_atlas:
        return qdata

    var_annotations = dict(
        # Standard MC2 gene annotations.
        clean_gene=False,
        excluded_gene=False,
        feature_gene=False,
        lateral_gene=False,
        bystander_gene=False,
        gene_deviant_votes=0,
        high_relative_variance_gene=False,
        high_total_gene=False,
        bursty_lonely_gene=False,
        pre_feature_gene=False,
        pre_gene_deviant_votes=0,
        pre_high_relative_variance_gene=False,
        pre_high_total_gene=False,
        properly_sampled_gene=False,
        rare_gene=False,
        rare_gene_module=-1,
        related_genes_module=-1,
        significant_gene=False,
        top_feature_gene=False,
        # Annotations added here (except for per-type).
        atlas_lateral_gene=False,
        atlas_bystander_gene=False,
        atlas_gene=False,
        atlas_significant_gene=False,
        essential_gene=False,
        correction_factor=1.0,
        correlated_gene=False,
        ignored_gene=False,
        manually_ignored_gene=False,
        projected_correlation=0.0,
    )

    if renormalize_var_annotations is not None:
        var_annotations.update(renormalize_var_annotations)

    layers = dict(inner_fold=0.0, projected=0.0, projected_fold=0.0)

    if renormalize_layers is not None:
        layers.update(renormalize_layers)

    varp_annotations = dict(var_similarity=0.0)
    if renormalize_varp_annotations is not None:
        varp_annotations.update(renormalize_varp_annotations)

    for type_name in np.unique(ut.get_o_numpy(adata, atlas_type_property_name)):
        var_annotations[f"misfit_gene_of_{type_name}"] = False
        var_annotations[f"manually_ignored_gene_of_{type_name}"] = False
        var_annotations[f"ignored_gene_of_{type_name}"] = False
    for essential_genes_name in adata.var.keys():
        if essential_genes_name.startswith("essential_gene_of_"):
            var_annotations[essential_genes_name] = False

    renormalized_qdata = tl.renormalize_query_by_atlas(
        adata=adata,
        qdata=qdata,
        var_annotations=var_annotations,
        layers=layers,
        varp_annotations=varp_annotations,
    )

    if renormalized_qdata is not None:
        return renormalized_qdata

    return qdata


@ut.logged()
@ut.timed_call()
def _compute_preliminary_projection(
    *,
    what: str,
    qdata: AnnData,
    common_adata: AnnData,
    common_qdata: AnnData,
    atlas_total_common_umis: ut.NumpyVector,
    ignore_atlas_insignificant_genes: bool,
    ignore_query_insignificant_genes: bool,
    ignore_atlas_lateral_genes: bool,
    ignore_atlas_bystander_genes: bool,
    ignore_query_lateral_genes: bool,
    ignore_query_bystander_genes: bool,
    project_fold_normalization: float,
    project_min_significant_gene_value: float,
    project_max_consistency_fold_factor: float,
    project_candidates_count: int,
    project_min_usage_weight: float,
    project_corrections: bool,
    project_min_corrected_gene_correlation: float,
    project_min_corrected_gene_factor: float,
    atlas_type_property_name: str,
    reproducible: bool,
) -> ut.NumpyVector:
    query_total_common_umis = ut.get_o_numpy(common_qdata, what, sum=True)

    included_adata, included_qdata = _included_global_data(
        common_adata=common_adata,
        common_qdata=common_qdata,
        ignore_atlas_insignificant_genes=ignore_atlas_insignificant_genes,
        ignore_query_insignificant_genes=ignore_query_insignificant_genes,
        ignore_atlas_lateral_genes=ignore_atlas_lateral_genes,
        ignore_atlas_bystander_genes=ignore_atlas_bystander_genes,
        ignore_query_lateral_genes=ignore_query_lateral_genes,
        ignore_query_bystander_genes=ignore_query_bystander_genes,
    )

    repeat = 0
    while True:
        repeat += 1
        ut.log_calc("preliminary repeat", repeat)

        weights = tl.project_query_onto_atlas(
            what,
            adata=included_adata,
            qdata=included_qdata,
            atlas_total_umis=atlas_total_common_umis,
            query_total_umis=query_total_common_umis,
            fold_normalization=project_fold_normalization,
            min_significant_gene_value=project_min_significant_gene_value,
            max_consistency_fold_factor=project_max_consistency_fold_factor,
            candidates_count=project_candidates_count,
            min_usage_weight=project_min_usage_weight,
            reproducible=reproducible,
        )

        tl.compute_query_projection(
            adata=included_adata,
            qdata=included_qdata,
            weights=weights,
            atlas_total_umis=atlas_total_common_umis,
            query_total_umis=query_total_common_umis,
        )

        tl.compute_projected_fold_factors(
            included_qdata,
            what,
            total_umis=query_total_common_umis,
            fold_normalization=project_fold_normalization,
            min_significant_gene_value=project_min_significant_gene_value,
        )

        if repeat > 2 or not _correct_correlated_genes(
            what=what,
            qdata=qdata,
            common_qdata=common_qdata,
            included_qdata=included_qdata,
            project_corrections=project_corrections,
            project_min_corrected_gene_correlation=project_min_corrected_gene_correlation,
            project_min_corrected_gene_factor=project_min_corrected_gene_factor,
            reproducible=reproducible,
        ):
            break

        query_total_common_umis = ut.get_o_numpy(common_qdata, what, sum=True)

    tl.project_atlas_to_query(
        adata=included_adata,
        qdata=included_qdata,
        weights=weights,
        property_name=atlas_type_property_name,
        to_property_name="projected_type",
    )

    projected_types = ut.get_o_numpy(included_qdata, "projected_type")
    ut.set_o_data(common_qdata, "projected_type", projected_types)

    return query_total_common_umis


@ut.logged()
@ut.timed_call()
def _included_global_data(
    *,
    common_adata: AnnData,
    common_qdata: AnnData,
    ignore_atlas_insignificant_genes: bool,
    ignore_query_insignificant_genes: bool,
    ignore_atlas_lateral_genes: bool,
    ignore_atlas_bystander_genes: bool,
    ignore_query_lateral_genes: bool,
    ignore_query_bystander_genes: bool,
) -> Tuple[AnnData, AnnData]:
    ignored_mask_names = ["|manually_ignored_gene?"]

    if ignore_atlas_insignificant_genes:
        ignored_mask_names.append("|~atlas_significant_gene?")

    if ignore_query_insignificant_genes:
        ignored_mask_names.append("|~significant_gene?")

    if ignore_atlas_lateral_genes:
        ignored_mask_names.append("|atlas_lateral_gene?")

    if ignore_atlas_bystander_genes:
        ignored_mask_names.append("|atlas_bystander_gene?")

    if ignore_query_lateral_genes:
        ignored_mask_names.append("|lateral_gene?")

    if ignore_query_bystander_genes:
        ignored_mask_names.append("|bystander_gene?")

    tl.combine_masks(common_qdata, ignored_mask_names, to="ignored_gene")
    included_genes_mask = ~ut.get_v_numpy(common_qdata, "ignored_gene")
    assert np.any(included_genes_mask), "all genes are ignored"

    included_adata = ut.slice(common_adata, name=".included", vars=included_genes_mask)
    included_qdata = ut.slice(common_qdata, name=".included", vars=included_genes_mask)

    return included_adata, included_qdata


@ut.logged()
@ut.timed_call()
def _compute_per_type_projection(
    *,
    what: str,
    common_adata: AnnData,
    common_qdata: AnnData,
    atlas_total_common_umis: ut.NumpyVector,
    query_total_common_umis: ut.NumpyVector,
    misfit_min_metacells_fraction: float,
    project_fold_normalization: float,
    project_candidates_count: int,
    project_min_significant_gene_value: float,
    project_min_usage_weight: float,
    project_max_consistency_fold_factor: float,
    project_max_projection_fold_factor: float,
    project_max_dissimilar_genes: int,
    min_essential_genes_of_type: Optional[Dict[str, Optional[float]]],
    project_abs_folds: bool,
    atlas_type_property_name: str,
    top_level_parallel: bool,
    reproducible: bool,
) -> ut.NumpyMatrix:
    taboo_types: List[Set[str]] = []
    for _metacell_index in range(common_qdata.n_obs):
        taboo_types.append(set())

    repeat = 0
    while True:
        repeat += 1
        ut.log_calc("types repeat", repeat)

        all_types = np.unique(ut.get_o_numpy(common_adata, atlas_type_property_name))
        for type_name in all_types:
            full_name = f"ignored_gene_of_{type_name}"
            if full_name in common_qdata.var:
                del common_qdata.var[full_name]

        type_of_atlas_metacells = ut.get_o_numpy(common_adata, atlas_type_property_name)
        atlas_unique_types = np.unique(type_of_atlas_metacells)

        type_of_query_metacells = ut.get_o_numpy(common_qdata, "projected_type")
        query_unique_types = np.unique(type_of_query_metacells)

        similar = np.zeros(common_qdata.n_obs, dtype="bool")
        dissimilar_genes_count = np.zeros(common_qdata.n_obs, dtype="int32")
        weights = np.zeros((common_qdata.n_obs, common_adata.n_obs), dtype="float32")
        projected_folds = np.zeros(common_qdata.shape, dtype="float32")
        misfit_gene_of_types = np.zeros((len(query_unique_types), common_qdata.n_vars), dtype="bool")
        ignored_gene_of_types = np.zeros((len(query_unique_types), common_qdata.n_vars), dtype="bool")

        @ut.timed_call("single_type_projection")
        def _single_type_projection(
            type_index: int,
        ) -> Tuple[
            ut.NumpyVector,  # similar,
            ut.NumpyVector,  # dissimilar_genes_count,
            ut.NumpyMatrix,  # weights,
            ut.NumpyMatrix,  # projected_folds,
            ut.NumpyVector,  # misfit_gene_of_type,
            ut.NumpyVector,  # ignored_gene_of_type,
        ]:
            return _compute_single_type_projection(
                what=what,
                type_name=query_unique_types[type_index],
                common_adata=common_adata,
                common_qdata=common_qdata,
                atlas_total_common_umis=atlas_total_common_umis,
                query_total_common_umis=query_total_common_umis,
                misfit_min_metacells_fraction=misfit_min_metacells_fraction,
                project_fold_normalization=project_fold_normalization,
                project_candidates_count=project_candidates_count,
                project_min_significant_gene_value=project_min_significant_gene_value,
                project_min_usage_weight=project_min_usage_weight,
                project_max_consistency_fold_factor=project_max_consistency_fold_factor,
                project_max_projection_fold_factor=project_max_projection_fold_factor,
                project_max_dissimilar_genes=project_max_dissimilar_genes,
                min_essential_genes_of_type=min_essential_genes_of_type,
                project_abs_folds=project_abs_folds,
                top_level_parallel=top_level_parallel,
                reproducible=reproducible,
            )

        def _collect_type_result(
            type_index: int,
            similar_of_type: ut.NumpyVector,
            dissimilar_genes_count_of_type: ut.NumpyVector,
            weights_of_type: ut.ProperMatrix,
            projected_folds_of_type: ut.NumpyMatrix,
            misfit_gene_of_type: ut.NumpyVector,
            ignored_gene_of_type: ut.NumpyVector,
        ) -> None:
            nonlocal similar
            similar |= similar_of_type
            nonlocal dissimilar_genes_count
            dissimilar_genes_count += dissimilar_genes_count_of_type
            nonlocal weights
            weights += weights_of_type  # type: ignore
            nonlocal projected_folds
            projected_folds += projected_folds_of_type
            nonlocal misfit_gene_of_types
            misfit_gene_of_types[type_index, :] = misfit_gene_of_type
            nonlocal ignored_gene_of_types
            ignored_gene_of_types[type_index, :] = ignored_gene_of_type

        if top_level_parallel:
            for type_index, result in enumerate(ut.parallel_map(_single_type_projection, len(query_unique_types))):
                _collect_type_result(type_index, *result)
        else:
            for type_index in range(len(query_unique_types)):
                result = _single_type_projection(type_index)
                _collect_type_result(type_index, *result)

        for type_index, type_name in enumerate(query_unique_types):
            ut.set_v_data(
                common_qdata, f"misfit_gene_of_{type_name}", ut.to_numpy_vector(misfit_gene_of_types[type_index, :])
            )
            ut.set_v_data(
                common_qdata, f"ignored_gene_of_{type_name}", ut.to_numpy_vector(ignored_gene_of_types[type_index, :])
            )

        for type_name in set(atlas_unique_types) - set(query_unique_types):
            ut.set_v_data(common_qdata, f"misfit_gene_of_{type_name}", np.zeros(common_qdata.n_vars, dtype="bool"))
            ut.set_v_data(common_qdata, f"ignored_gene_of_{type_name}", np.zeros(common_qdata.n_vars, dtype="bool"))

        ut.set_o_data(common_qdata, "similar", similar)
        ut.log_calc("max dissimilar_genes_count", np.max(dissimilar_genes_count))
        ut.set_o_data(common_qdata, "dissimilar_genes_count", dissimilar_genes_count, formatter=ut.mask_description)

        if repeat > 2 or not _changed_projected_types(
            common_adata=common_adata,
            common_qdata=common_qdata,
            weights=weights,
            atlas_type_property_name=atlas_type_property_name,
            taboo_types=taboo_types,
        ):
            break

    tl.compute_query_projection(
        adata=common_adata,
        qdata=common_qdata,
        weights=weights,
        atlas_total_umis=atlas_total_common_umis,
        query_total_umis=query_total_common_umis,
    )

    ut.set_vo_data(common_qdata, "projected_fold", projected_folds)

    return weights


@ut.logged()
@ut.timed_call()
def _compute_single_type_projection(
    *,
    what: str,
    type_name: str,
    common_adata: AnnData,
    common_qdata: AnnData,
    atlas_total_common_umis: ut.NumpyVector,
    query_total_common_umis: ut.NumpyVector,
    misfit_min_metacells_fraction: float,
    project_fold_normalization: float,
    project_candidates_count: int,
    project_min_significant_gene_value: float,
    project_min_usage_weight: float,
    project_max_consistency_fold_factor: float,
    project_max_projection_fold_factor: float,
    project_max_dissimilar_genes: int,
    min_essential_genes_of_type: Optional[Dict[str, Optional[float]]],
    project_abs_folds: bool,
    top_level_parallel: bool,
    reproducible: bool,
) -> Tuple[
    ut.NumpyVector,  # similar,
    ut.NumpyVector,  # dissimilar_genes_count,
    ut.NumpyMatrix,  # weights,
    ut.NumpyMatrix,  # projected_folds,
    ut.NumpyVector,  # misfit_gene_of_type,
    ut.NumpyVector,  # ignored_gene_of_type,
]:
    similar = np.zeros(common_qdata.n_obs, dtype="bool")
    dissimilar_genes_count = np.zeros(common_qdata.n_obs, dtype="int32")
    weights = np.zeros((common_qdata.n_obs, common_adata.n_obs), dtype="float32")
    projected_folds = np.zeros(common_qdata.shape, dtype="float32")
    misfit_gene_of_type = np.zeros(common_qdata.n_vars, dtype="bool")
    ignored_gene_of_type = np.zeros(common_qdata.n_vars, dtype="bool")

    (type_common_qdata, query_type_total_common_umis,) = _common_type_data(
        type_name=type_name,
        common_qdata=common_qdata,
        query_total_common_umis=query_total_common_umis,
    )

    type_ignored_mask_names = ["|ignored_gene", f"|manually_ignored_gene_of_{type_name}?"]
    type_included_adata = common_adata
    type_included_qdata = type_common_qdata

    name: Optional[str] = ".included"
    repeat = 0
    while True:
        repeat += 1
        ut.log_calc(f"{type_name} misfit repeat", repeat)

        type_included_adata, type_included_qdata, type_weights = _compute_type_projection(
            what=what,
            name=name,
            type_name=type_name,
            type_included_adata=type_included_adata,
            type_included_qdata=type_included_qdata,
            atlas_total_common_umis=atlas_total_common_umis,
            query_type_total_common_umis=query_type_total_common_umis,
            type_ignored_mask_names=type_ignored_mask_names,
            project_fold_normalization=project_fold_normalization,
            project_min_significant_gene_value=project_min_significant_gene_value,
            project_max_consistency_fold_factor=project_max_consistency_fold_factor,
            project_candidates_count=project_candidates_count,
            project_min_usage_weight=project_min_usage_weight,
            reproducible=reproducible,
        )
        name = None
        type_ignored_mask_names = [f"misfit_gene_of_{type_name}"]

        if not _detect_type_misfit_genes(
            type_name=type_name,
            type_included_qdata=type_included_qdata,
            project_max_projection_fold_factor=project_max_projection_fold_factor,
            misfit_min_metacells_fraction=misfit_min_metacells_fraction,
            project_abs_folds=project_abs_folds,
            misfit_gene_of_type=misfit_gene_of_type,
        ):
            break

    tl.compute_similar_query_metacells(
        type_included_qdata,
        max_projection_fold_factor=project_max_projection_fold_factor,
        max_dissimilar_genes=project_max_dissimilar_genes,
        essential_genes_property=f"essential_gene_of_{type_name}",
        min_similar_essential_genes=None
        if min_essential_genes_of_type is None
        else min_essential_genes_of_type[type_name],
        abs_folds=project_abs_folds,
    )

    _convey_query_type_to_common_data(
        type_weights=type_weights,
        weights=weights,
        projected_folds=projected_folds,
        type_included_qdata=type_included_qdata,
        similar=similar,
        dissimilar_genes_count=dissimilar_genes_count,
        ignored_gene_of_type=ignored_gene_of_type,
    )

    if top_level_parallel:
        weights = sp.csr_matrix(weights)

    return (
        similar,
        dissimilar_genes_count,
        weights,
        projected_folds,
        misfit_gene_of_type,
        ignored_gene_of_type,
    )


def _convey_query_type_to_common_data(
    *,
    type_weights: ut.CompressedMatrix,
    weights: ut.NumpyMatrix,
    projected_folds: ut.NumpyMatrix,
    type_included_qdata: AnnData,
    similar: ut.NumpyVector,
    dissimilar_genes_count: ut.NumpyVector,
    ignored_gene_of_type: ut.NumpyVector,
) -> None:
    common_gene_index_of_type_included_qdata = ut.get_v_numpy(type_included_qdata, "common_gene_index_of_qdata")
    ignored_gene_of_type[common_gene_index_of_type_included_qdata] = False

    full_cell_index_of_type_qdata = ut.get_o_numpy(type_included_qdata, "full_metacell_index_of_qdata")
    similar[full_cell_index_of_type_qdata] = ut.get_o_numpy(type_included_qdata, "similar")
    ignored_gene_of_type[:] = True
    common_gene_index_of_type_qdata = ut.get_v_numpy(type_included_qdata, "common_gene_index_of_qdata")
    ignored_gene_of_type[common_gene_index_of_type_qdata] = False
    dissimilar_genes_count[full_cell_index_of_type_qdata] = ut.get_o_numpy(
        type_included_qdata, "dissimilar_genes_count", formatter=ut.mask_description
    )

    weights[full_cell_index_of_type_qdata, :] = ut.to_numpy_matrix(type_weights)
    type_projected_fold = ut.to_numpy_matrix(ut.get_vo_proper(type_included_qdata, "projected_fold"))
    projected_folds[
        full_cell_index_of_type_qdata[:, None], common_gene_index_of_type_included_qdata[None, :]
    ] = type_projected_fold


@ut.logged()
@ut.timed_call()
def _common_type_data(
    *,
    type_name: str,
    common_qdata: AnnData,
    query_total_common_umis: ut.NumpyVector,
) -> Tuple[AnnData, ut.NumpyVector]:
    type_of_query_metacells = ut.get_o_numpy(common_qdata, "projected_type")
    query_type_mask = type_of_query_metacells == type_name
    ut.log_calc("query cells mask", query_type_mask)
    type_common_qdata = ut.slice(
        common_qdata, obs=query_type_mask, name=f".{type_name}", track_var="full_metacell_index_of_qdata"
    )
    query_type_total_common_umis = query_total_common_umis[query_type_mask]
    return type_common_qdata, query_type_total_common_umis


@ut.logged()
@ut.timed_call()
def _compute_type_projection(
    *,
    what: str,
    name: Optional[str],
    type_name: str,
    type_included_adata: AnnData,
    type_included_qdata: AnnData,
    atlas_total_common_umis: ut.NumpyVector,
    query_type_total_common_umis: ut.NumpyVector,
    type_ignored_mask_names: List[str],
    project_fold_normalization: float,
    project_min_significant_gene_value: float,
    project_max_consistency_fold_factor: float,
    project_candidates_count: int,
    project_min_usage_weight: float,
    reproducible: bool,
) -> Tuple[AnnData, AnnData, ut.CompressedMatrix]:
    tl.combine_masks(type_included_qdata, type_ignored_mask_names, to=f"ignored_gene_of_{type_name}")
    type_included_genes_mask = ~ut.get_v_numpy(type_included_qdata, f"ignored_gene_of_{type_name}")
    if not np.any(type_included_genes_mask):
        type_included_genes_mask[0] = True

    type_included_adata = ut.slice(
        type_included_adata, name=name or ut.get_name(type_included_adata), vars=type_included_genes_mask
    )
    type_included_qdata = ut.slice(
        type_included_qdata, name=name or ut.get_name(type_included_qdata), vars=type_included_genes_mask
    )

    type_weights = tl.project_query_onto_atlas(
        what,
        adata=type_included_adata,
        qdata=type_included_qdata,
        atlas_total_umis=atlas_total_common_umis,
        query_total_umis=query_type_total_common_umis,
        fold_normalization=project_fold_normalization,
        min_significant_gene_value=project_min_significant_gene_value,
        max_consistency_fold_factor=project_max_consistency_fold_factor,
        candidates_count=project_candidates_count,
        min_usage_weight=project_min_usage_weight,
        reproducible=reproducible,
    )

    tl.compute_query_projection(
        adata=type_included_adata,
        qdata=type_included_qdata,
        weights=type_weights,
        atlas_total_umis=atlas_total_common_umis,
        query_total_umis=query_type_total_common_umis,
    )

    tl.compute_projected_fold_factors(
        type_included_qdata,
        what,
        total_umis=query_type_total_common_umis,
        fold_normalization=project_fold_normalization,
        min_significant_gene_value=project_min_significant_gene_value,
    )

    return type_included_adata, type_included_qdata, type_weights


def _detect_type_misfit_genes(
    *,
    type_name: str,
    type_included_qdata: AnnData,
    project_max_projection_fold_factor: float,
    misfit_min_metacells_fraction: float,
    project_abs_folds: bool,
    misfit_gene_of_type: ut.NumpyVector,
) -> bool:
    tl.find_misfit_genes(
        type_included_qdata,
        max_projection_fold_factor=project_max_projection_fold_factor,
        min_metacells_fraction=misfit_min_metacells_fraction,
        abs_folds=project_abs_folds,
        to_property_name=f"misfit_gene_of_{type_name}",
    )

    new_type_misfit_genes_mask = ut.get_v_numpy(type_included_qdata, f"misfit_gene_of_{type_name}")
    if not np.any(new_type_misfit_genes_mask):
        return False

    common_gene_index_of_type_included_qdata = ut.get_v_numpy(type_included_qdata, "common_gene_index_of_qdata")
    misfit_gene_of_type[common_gene_index_of_type_included_qdata] = new_type_misfit_genes_mask
    return True


@ut.logged()
@ut.timed_call()
def _changed_projected_types(
    *,
    common_adata: AnnData,
    common_qdata: AnnData,
    weights: ut.NumpyMatrix,
    atlas_type_property_name: str,
    taboo_types: List[Set[str]],
) -> bool:
    old_type_of_query_metacells = ut.get_o_numpy(common_qdata, "projected_type").copy()
    for metacell_index, old_type in enumerate(old_type_of_query_metacells):
        taboo_types[metacell_index].add(old_type)

    tl.project_atlas_to_query(
        adata=common_adata,
        qdata=common_qdata,
        weights=weights,
        property_name=atlas_type_property_name,
        to_property_name="projected_type",
    )

    new_type_of_query_metacells = ut.get_o_numpy(common_qdata, "projected_type")
    has_changed = False
    for metacell_index, new_type in enumerate(new_type_of_query_metacells):
        if new_type not in taboo_types[metacell_index]:
            old_type = old_type_of_query_metacells[metacell_index]
            ut.log_calc(f"metacell: {metacell_index} changed from: {old_type} to", new_type)
            taboo_types[metacell_index].add(new_type)
            has_changed = True

    ut.log_return("has_changed", has_changed)
    return has_changed


def _correct_correlated_genes(
    *,
    what: str,
    qdata: AnnData,
    common_qdata: AnnData,
    included_qdata: AnnData,
    project_corrections: bool,
    project_min_corrected_gene_correlation: float,
    project_min_corrected_gene_factor: float,
    reproducible: bool,
) -> bool:
    projected_gene_columns = ut.to_numpy_matrix(ut.get_vo_proper(included_qdata, "projected", layout="column_major"))
    projected_gene_rows = projected_gene_columns.transpose()

    observed_gene_rows = ut.to_numpy_matrix(ut.get_vo_proper(included_qdata, what, layout="column_major")).transpose()
    included_gene_correlations = ut.pairs_corrcoef_rows(
        projected_gene_rows, observed_gene_rows, reproducible=reproducible
    )
    assert len(included_gene_correlations) == included_qdata.n_vars
    ut.set_v_data(included_qdata, "projected_correlation", included_gene_correlations)

    correlated_included_genes_mask = included_gene_correlations >= project_min_corrected_gene_correlation
    ut.set_v_data(included_qdata, "correlated_gene", correlated_included_genes_mask)

    included_gene_indices = ut.get_v_numpy(included_qdata, "full_gene_index_of_qdata")

    gene_correlations = ut.get_v_numpy(qdata, "projected_correlation").copy()
    gene_correlations[included_gene_indices] = included_gene_correlations
    ut.set_v_data(qdata, "projected_correlation", gene_correlations)

    correlated_genes_mask = ut.get_v_numpy(qdata, "correlated_gene").copy()
    correlated_genes_mask[included_gene_indices] |= correlated_included_genes_mask
    ut.set_v_data(qdata, "correlated_gene", correlated_genes_mask)

    if not np.any(correlated_included_genes_mask):
        return False

    correlated_included_gene_indices = np.where(correlated_included_genes_mask)[0]
    correlated_included_projected_gene_rows = projected_gene_rows[correlated_included_gene_indices, :]
    correlated_included_observed_gene_rows = observed_gene_rows[correlated_included_gene_indices, :]
    correlated_included_projected_gene_totals = ut.sum_per(correlated_included_projected_gene_rows, per="row")
    correlated_included_observed_gene_totals = ut.sum_per(correlated_included_observed_gene_rows, per="row")
    correlated_included_gene_correction_factors = (
        correlated_included_projected_gene_totals / correlated_included_observed_gene_totals
    )

    correlated_included_genes_corrected_mask = (
        correlated_included_gene_correction_factors < (1.0 / (1 + project_min_corrected_gene_factor))
    ) | (correlated_included_gene_correction_factors > (1 + project_min_corrected_gene_factor))

    if ut.has_data(included_qdata, "atlas_significant_gene"):
        included_genes_atlas_significant_mask = ut.get_v_numpy(included_qdata, "atlas_significant_gene")
        correlated_included_genes_atlas_significant_mask = included_genes_atlas_significant_mask[
            correlated_included_gene_indices
        ]
        correlated_included_genes_corrected_mask &= correlated_included_genes_atlas_significant_mask

    ut.log_calc("correlated_included_genes_corrected_mask", correlated_included_genes_corrected_mask)
    if not np.any(correlated_included_genes_corrected_mask):
        return False

    corrected_gene_factors = correlated_included_gene_correction_factors[correlated_included_genes_corrected_mask]
    corrected_included_gene_indices = correlated_included_gene_indices[correlated_included_genes_corrected_mask]

    included_genes_correction_factors = ut.get_v_numpy(included_qdata, "correction_factor").copy()
    included_genes_correction_factors[corrected_included_gene_indices] *= corrected_gene_factors
    ut.set_v_data(included_qdata, "correction_factor", included_genes_correction_factors)

    if project_corrections:
        included_corrected_data = ut.to_numpy_matrix(
            ut.get_vo_proper(included_qdata, what, layout="column_major")
        ).copy()
        included_corrected_data[:, corrected_included_gene_indices] *= corrected_gene_factors[None, :]
        ut.set_vo_data(included_qdata, what, included_corrected_data)

    if id(included_qdata) != id(common_qdata):
        common_gene_index_of_included_qdata = ut.get_v_numpy(included_qdata, "common_gene_index_of_qdata")
        corrected_common_gene_indices = common_gene_index_of_included_qdata[corrected_included_gene_indices]

        common_genes_correction_factors = ut.get_v_numpy(common_qdata, "correction_factor").copy()
        common_genes_correction_factors[corrected_common_gene_indices] *= corrected_gene_factors
        ut.set_v_data(common_qdata, "correction_factor", common_genes_correction_factors)

        if project_corrections:
            common_corrected_data = ut.to_numpy_matrix(
                ut.get_vo_proper(common_qdata, what, layout="column_major")
            ).copy()
            common_corrected_data[:, corrected_common_gene_indices] *= corrected_gene_factors[None, :]
            ut.set_vo_data(common_qdata, what, common_corrected_data)

    return project_corrections


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def outliers_projection_pipeline(
    what: str = "__x__",
    *,
    adata: AnnData,
    odata: AnnData,
    fold_normalization: float = pr.outliers_fold_normalization,
    min_gene_outliers_fold_factor: float = pr.min_gene_outliers_fold_factor,
    min_entry_outliers_fold_factor: float = pr.min_entry_outliers_fold_factor,
    abs_folds: bool = pr.outliers_abs_folds,
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
       (default: {fold_normalization}), ``min_gene_outliers_fold_factor`` (default: {min_gene_outliers_fold_factor}),
       ``min_entry_outliers_fold_factor`` (default: {min_entry_outliers_fold_factor}) and ``abs_folds`` (default:
       {abs_folds}).
    """
    if list(odata.var_names) != list(adata.var_names):
        atlas_genes_list = list(adata.var_names)
        query_genes_list = list(odata.var_names)
        common_genes_list = list(sorted(set(atlas_genes_list) & set(query_genes_list)))
        atlas_gene_indices = np.array([atlas_genes_list.index(gene) for gene in common_genes_list])
        query_gene_indices = np.array([query_genes_list.index(gene) for gene in common_genes_list])
        common_adata = ut.slice(adata, name=".common", vars=atlas_gene_indices)
        common_odata = ut.slice(
            odata,
            name=".common",
            vars=query_gene_indices,
            track_var="full_gene_index_of_odata",
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
        fold_normalization=fold_normalization,
        min_gene_outliers_fold_factor=min_gene_outliers_fold_factor,
        min_entry_outliers_fold_factor=min_entry_outliers_fold_factor,
        abs_folds=abs_folds,
    )

    if list(odata.var_names) != list(adata.var_names):
        atlas_most_similar = ut.get_o_numpy(common_odata, "atlas_most_similar")
        ut.set_o_data(odata, "atlas_most_similar", atlas_most_similar)

        common_gene_indices = ut.get_v_numpy(common_odata, "full_gene_index_of_odata")
        common_folds = ut.get_vo_proper(common_odata, "atlas_most_similar_fold")
        atlas_most_similar_fold = sp.csr_matrix(odata.shape, dtype="float32")
        atlas_most_similar_fold[:, common_gene_indices] = common_folds
        ut.set_vo_data(odata, "atlas_most_similar_fold", atlas_most_similar_fold)


@ut.timed_call()
@ut.logged()
def _compute_dissimilar_residuals_projection(
    *,
    what: str,
    common_adata: AnnData,
    common_qdata: AnnData,
    weights: ut.NumpyMatrix,
    atlas_total_common_umis: ut.NumpyVector,
    query_total_common_umis: ut.NumpyVector,
    project_fold_normalization: float,
    project_candidates_count: int,
    project_min_significant_gene_value: float,
    project_min_usage_weight: float,
    project_max_consistency_fold_factor: float,
    project_max_projection_fold_factor: float,
    project_max_dissimilar_genes: int,
    min_essential_genes_of_type: Optional[Dict[str, Optional[float]]],
    project_abs_folds: bool,
    atlas_type_property_name: str,
    top_level_parallel: bool,
    reproducible: bool,
) -> ut.NumpyMatrix:
    secondary_type = [""] * common_qdata.n_obs
    dissimilar_mask = ~ut.get_o_numpy(common_qdata, "similar")
    if not np.any(dissimilar_mask):
        ut.set_o_data(common_qdata, "projected_secondary_type", np.array(secondary_type))
        return weights

    dissimilar_qdata = ut.slice(common_qdata, obs=dissimilar_mask, name=".dissimilar")
    dissimilar_total_common_umis = query_total_common_umis[dissimilar_mask]

    @ut.timed_call("single_metacell_residuals")
    def _single_metacell_residuals(
        dissimilar_metacell_index: int,
    ) -> Tuple[ut.NumpyVector, ut.NumpyVector, ut.NumpyVector, str, str, bool]:
        return _compute_single_metacell_residuals(
            dissimilar_metacell_index=dissimilar_metacell_index,
            what=what,
            common_adata=common_adata,
            dissimilar_qdata=dissimilar_qdata,
            atlas_total_common_umis=atlas_total_common_umis,
            dissimilar_total_common_umis=dissimilar_total_common_umis,
            project_fold_normalization=project_fold_normalization,
            project_candidates_count=project_candidates_count,
            project_min_significant_gene_value=project_min_significant_gene_value,
            project_min_usage_weight=project_min_usage_weight,
            project_max_consistency_fold_factor=project_max_consistency_fold_factor,
            project_max_projection_fold_factor=project_max_projection_fold_factor,
            project_max_dissimilar_genes=project_max_dissimilar_genes,
            min_essential_genes_of_type=min_essential_genes_of_type,
            project_abs_folds=project_abs_folds,
            atlas_type_property_name=atlas_type_property_name,
            reproducible=reproducible,
        )

    if top_level_parallel:
        results = ut.parallel_map(_single_metacell_residuals, dissimilar_qdata.n_obs)
    else:
        results = []
        for dissimilar_metacell_index in range(dissimilar_qdata.n_obs):
            results.append(_single_metacell_residuals(dissimilar_metacell_index))

    primary_type = ut.get_o_numpy(common_qdata, "projected_type").copy()
    projected_folds = ut.to_numpy_matrix(ut.get_vo_proper(common_qdata, "projected_fold"), copy=True)
    similar = ut.get_o_numpy(common_qdata, "similar").copy()

    for (
        dissimilar_metacell_index,
        (
            metacell_weights,
            metacell_gene_indices,
            metacell_projected_folds,
            metacell_primary_type,
            metacell_secondary_type,
            metacell_similar,
        ),
    ) in enumerate(results):
        metacell_index = ut.get_o_numpy(dissimilar_qdata, "full_metacell_index_of_qdata")[dissimilar_metacell_index]
        ut.log_calc("metacell_index", metacell_index)
        weights[metacell_index, :] = metacell_weights
        projected_folds[metacell_index, :] = 0.0
        projected_folds[metacell_index, metacell_gene_indices] = metacell_projected_folds
        primary_type[metacell_index] = metacell_primary_type
        secondary_type[metacell_index] = metacell_secondary_type
        similar[metacell_index] = metacell_similar

    ut.set_vo_data(common_qdata, "projected_fold", projected_folds)
    ut.set_o_data(common_qdata, "projected_type", primary_type)
    ut.set_o_data(common_qdata, "projected_secondary_type", np.array(secondary_type))
    ut.set_o_data(common_qdata, "similar", similar)

    tl.compute_query_projection(
        adata=common_adata,
        qdata=common_qdata,
        weights=weights,
        atlas_total_umis=atlas_total_common_umis,
        query_total_umis=query_total_common_umis,
    )

    return weights


@ut.timed_call()
@ut.logged()
def _compute_single_metacell_residuals(  # pylint: disable=too-many-statements
    *,
    dissimilar_metacell_index: int,
    what: str,
    common_adata: AnnData,
    dissimilar_qdata: AnnData,
    atlas_total_common_umis: ut.NumpyVector,
    dissimilar_total_common_umis: ut.NumpyVector,
    project_fold_normalization: float,
    project_candidates_count: int,
    project_min_significant_gene_value: float,
    project_min_usage_weight: float,
    project_max_consistency_fold_factor: float,
    project_max_projection_fold_factor: float,
    project_max_dissimilar_genes: int,
    min_essential_genes_of_type: Optional[Dict[str, Optional[float]]],
    project_abs_folds: bool,
    atlas_type_property_name: str,
    reproducible: bool,
) -> Tuple[ut.NumpyVector, ut.NumpyVector, ut.NumpyVector, str, str, bool]:
    primary_type = ut.get_o_numpy(dissimilar_qdata, "projected_type")[dissimilar_metacell_index]
    secondary_type = ""

    repeat = 0
    while True:
        repeat += 1
        ut.log_calc("residuals repeat", repeat)
        ut.log_calc("primary_type", primary_type)
        ut.log_calc("secondary_type", secondary_type)

        ignored_genes_mask = ut.get_v_numpy(dissimilar_qdata, f"ignored_gene_of_{primary_type}")
        if secondary_type != "":
            ignored_genes_mask = ignored_genes_mask | ut.get_v_numpy(
                dissimilar_qdata, f"ignored_gene_of_{secondary_type}"
            )

        included_adata = ut.slice(common_adata, vars=~ignored_genes_mask, name=".included")

        metacell_included_qdata = ut.slice(
            dissimilar_qdata, obs=[dissimilar_metacell_index], vars=~ignored_genes_mask, name=".single"
        )
        metacell_total_common_umis = dissimilar_total_common_umis[
            dissimilar_metacell_index : dissimilar_metacell_index + 1
        ]

        second_anchor_indices: List[int] = []

        weights = tl.project_query_onto_atlas(
            what,
            adata=included_adata,
            qdata=metacell_included_qdata,
            atlas_total_umis=atlas_total_common_umis,
            query_total_umis=metacell_total_common_umis,
            fold_normalization=project_fold_normalization,
            min_significant_gene_value=project_min_significant_gene_value,
            max_consistency_fold_factor=project_max_consistency_fold_factor,
            candidates_count=project_candidates_count,
            min_usage_weight=project_min_usage_weight,
            reproducible=reproducible,
            second_anchor_indices=second_anchor_indices,
        )

        first_anchor_weights = weights.copy()
        first_anchor_weights[:, second_anchor_indices] = 0.0

        if len(second_anchor_indices) == 0:
            new_secondary_type = ""
        else:
            second_anchor_weights = weights - first_anchor_weights  # type: ignore

            tl.project_atlas_to_query(
                adata=included_adata,
                qdata=metacell_included_qdata,
                weights=second_anchor_weights,
                property_name=atlas_type_property_name,
                to_property_name="projected_secondary_type",
            )
            new_secondary_type = ut.get_o_numpy(metacell_included_qdata, "projected_secondary_type")[0]

        if np.sum(first_anchor_weights.data) == 0:
            new_primary_type = new_secondary_type
            new_secondary_type = ""
        else:
            tl.project_atlas_to_query(
                adata=included_adata,
                qdata=metacell_included_qdata,
                weights=first_anchor_weights,
                property_name=atlas_type_property_name,
                to_property_name="projected_type",
            )
            new_primary_type = ut.get_o_numpy(metacell_included_qdata, "projected_type")[0]

        if repeat > 2 or (new_primary_type == primary_type and new_secondary_type == secondary_type):
            break

        primary_type = new_primary_type
        secondary_type = new_secondary_type

    tl.compute_query_projection(
        adata=included_adata,
        qdata=metacell_included_qdata,
        weights=weights,
        atlas_total_umis=atlas_total_common_umis,
        query_total_umis=metacell_total_common_umis,
    )

    tl.compute_projected_fold_factors(
        metacell_included_qdata,
        what,
        total_umis=metacell_total_common_umis,
        fold_normalization=project_fold_normalization,
        min_significant_gene_value=project_min_significant_gene_value,
    )

    min_similar_essential_genes: Optional[float] = None
    essential_genes_property: Optional[List[str]] = None
    if min_essential_genes_of_type is not None:
        min_similar_essential_genes = None
        essential_genes_property = []
        for type_name in (primary_type, secondary_type):
            if type_name == "" or min_essential_genes_of_type[type_name] is None:
                continue
            if min_similar_essential_genes is None:
                min_similar_essential_genes = min_essential_genes_of_type[type_name]
            else:
                min_similar_essential_genes += min_essential_genes_of_type[type_name]  # type: ignore
            essential_genes_property.append(f"essential_gene_of_{type_name}")

    tl.compute_similar_query_metacells(
        metacell_included_qdata,
        max_projection_fold_factor=project_max_projection_fold_factor,
        max_dissimilar_genes=project_max_dissimilar_genes,
        essential_genes_property=essential_genes_property,
        min_similar_essential_genes=min_similar_essential_genes,
        abs_folds=project_abs_folds,
    )

    weights_vector = ut.to_numpy_vector(weights)
    gene_indices = ut.get_v_numpy(metacell_included_qdata, "common_gene_index_of_qdata")
    projected_folds = ut.to_numpy_vector(ut.get_vo_proper(metacell_included_qdata, "projected_fold"))
    similar = ut.get_o_numpy(metacell_included_qdata, "similar")[0]

    ut.log_return("weights", weights_vector)
    ut.log_return("gene_indices", gene_indices)
    ut.log_return("projected_folds", projected_folds)
    ut.log_return("primary_type", primary_type)
    ut.log_return("secondary_type", secondary_type)
    ut.log_return("similar", similar)

    assert secondary_type is not None
    return weights_vector, gene_indices, projected_folds, primary_type, secondary_type, similar
