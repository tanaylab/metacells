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
    ignore_atlas_forbidden_genes: bool = pr.ignore_atlas_forbidden_genes,
    ignore_query_forbidden_genes: bool = pr.ignore_query_forbidden_genes,
    systematic_low_gene_quantile: float = pr.systematic_low_gene_quantile,
    systematic_high_gene_quantile: float = pr.systematic_high_gene_quantile,
    biased_min_metacells_fraction: float = pr.biased_min_metacells_fraction,
    project_fold_normalization: float = pr.project_fold_normalization,
    project_candidates_count: int = pr.project_candidates_count,
    project_min_significant_gene_value: float = pr.project_min_significant_gene_value,
    project_min_usage_weight: float = pr.project_min_usage_weight,
    project_max_consistency_fold_factor: float = pr.project_max_consistency_fold_factor,
    project_max_projection_fold_factor: float = pr.project_max_projection_fold_factor,
    project_max_dissimilar_genes: int = pr.project_max_dissimilar_genes,
    min_entry_project_fold_factor: float = pr.min_entry_project_fold_factor,
    project_abs_folds: bool = pr.project_abs_folds,
    ignored_gene_names_of_type: Optional[Dict[str, Collection[str]]] = None,
    ignored_gene_patterns_of_type: Optional[Dict[str, Collection[str]]] = None,
    atlas_type_property_name: str = "type",
    project_min_corrected_gene_correlation: float = pr.project_min_corrected_gene_correlation,
    project_min_corrected_gene_factor: float = pr.project_min_corrected_gene_factor,
    project_max_uncorrelated_gene_correlation: float = pr.project_max_uncorrelated_gene_correlation,
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

        ``uncorrelated_gene``
            A boolean mask of genes which were ignored because they were too uncorrelated between the query and the
            atlas.

        ``systematic_gene``
            A boolean mask indicating whether the gene is systematically higher or lower in the query compared to the
            atlas.

        ``biased_gene``
            A boolean mask indicating whether the gene has a strong bias in the query compared to the atlas.

        ``manually_ignored_gene``
            A boolean mask indicating whether the gene was manually ignored (by its name).

        ``atlas_gene``
            A boolean mask indicating whether the gene exists in the atlas.

        ``atlas_significant_gene`` (``ignore_atlas_insignificant_genes`` requires an atlas ``significant_gene`` mask)
            A boolean mask indicating whether the gene is considered significant in the atlas.

        ``atlas_forbidden_gene`` (``ignore_atlas_forbidden_genes`` requires an atlas has ``forbidden_gene`` mask)
            A boolean mask indicating whether the gene was forbidden from being a feature in the atlas (and hence
            ignored).

        ``ignored_gene``
            A boolean mask indicating whether the gene was ignored by the projection (for any reason).

        ``systematic_gene_of_<type>``
            For each query metacell type, a boolean mask indicating whether the gene is systematically higher or lower
            in the query metacells of this type compared to the atlas metacells of this type.

        ``biased_gene_of_<type>``
            For each query metacell type, a boolean mask indicating whether the gene has a strong bias in the query
            metacells of this type compared to the atlas metacells of this type.

        ``manually_ignored_gene_of_<type>``
            For each query metacell type, a boolean mask indicating whether the gene was manually ignored (by its name)
            when computing the projection for metacells of this type.

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

        ``dissimilar_genes_count``
            The number of genes whose fold factor between the query metacell and its projection on the atlas is above
            the threshold.

        ``projected_type``
            The type assigned to each query metacell by its projection on the atlas. This should not be taken as gospel,
            even for "similar" metacells, as there may still be significant systematic differences between the query and
            the atlas as captured by the various per-gene annotations.

        ``projected_secondary_type``
            If the query metacell is "similar" to its projection to a single point in the atlas, this will be the empty
            string. Otherwise, we try to project it using two distinct points in the atlas, in which case this would
            contain the type associated with that second point. Thus, if a query metacell contains doublet cells, it
            might end up being "similar" but to a combination of two types in the atlas. For a truly novel query
            metacell type, this will also end up being "dissimilar", and the secondary type may or may not be helpful.
            Also, even if the query metacell ends up being "similar" to a combination of two atlas points, this can be
            either due to the query metacell containing true doublet cells, or possibly due to the atlas having
            metacells that span some gradient where the query has only one metacell that covers a "too large" range of
            this gradient (in this case the secondary type might even end up being the same as the main projected type).
            In either case, having a non-empty value here doesn't guarantee this is a metacell of doublets.

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

    1. Invoke :py:func:`metacells.tools.project.find_systematic_genes` using ``systematic_low_gene_quantile`` (default:
       {systematic_low_gene_quantile}) and ``systematic_high_gene_quantile`` (default: {systematic_high_gene_quantile}).

    2. Compute a mask of ignored genes, containing any genes named in ``ignored_gene_names`` or that match any of the
       ``ignored_gene_patterns``. If ``ignore_atlas_insignificant_genes`` (default:
       ``ignore_atlas_insignificant_genes``), ignore genes the atlas did not mark as significant (and store the
       ``atlas_significant_gene`` mask). If ``ignore_query_insignificant_genes`` (default:
       ``ignore_query_insignificant_genes``), ignore genes the query did not mark as significant. If
       ``ignore_atlas_forbidden_genes`` (default: {ignore_atlas_forbidden_genes}), also ignore the ``forbidden_gene`` of
       the atlas (and store them as an ``atlas_forbidden_gene`` mask). If ``ignore_query_forbidden_genes`` (default:
       {ignore_query_forbidden_genes}), also ignore the ``forbidden_gene`` of the query. All these genes are ignored by
       the following code.

    3. Invoke :py:func:`metacells.tools.project.project_query_onto_atlas` and
       :py:func:`metacells.tools.project.compute_query_projection` to project each query metacell onto the atlas, using
       the ``project_fold_normalization`` (default: {project_fold_normalization}),
       ``project_min_significant_gene_value`` (default: {project_min_significant_gene_value}),
       ``project_candidates_count`` (default: {project_candidates_count}), ``project_min_usage_weight`` (default:
       {project_min_usage_weight}), ``project_abs_folds`` (default: {project_abs_folds}) and ``reproducible``.

    4. Invoke :py:func:`metacells.tools.project.compute_significant_projected_fold_factors` to compute the significant
       fold factors between the query and its projection, using the ``project_fold_normalization`` (default:
       {project_fold_normalization}), ``project_max_projection_fold_factor`` (default:
       {project_max_projection_fold_factor}), ``min_entry_project_fold_factor`` (default:
       {min_entry_project_fold_factor}), ``project_min_significant_gene_value`` (default:
       {project_min_significant_gene_value}) and ``project_abs_folds`` (default: {project_abs_folds}).

    5. Invoke :py:func:`metacells.tools.project.find_biased_genes` using the ``project_max_projection_fold_factor``
        (default: {project_max_projection_fold_factor}), ``biased_min_metacells_fraction`` (default:
        {biased_min_metacells_fraction}), and ``project_abs_folds`` (default: {project_abs_folds}). If any such genes
        are found, add them to the ignored genes and repeat steps 3-4.

    6. Invoke :py:func:`metacells.tools.quality.compute_similar_query_metacells` to annotate the query metacells
       similar to their projection on the atlas using ``project_max_projection_fold_factor`` (default:
       {project_max_projection_fold_factor}) and the ``project_max_dissimilar_genes`` (default:
       {project_max_dissimilar_genes}).

    7. Invoke :py:func:`metacells.tools.project.project_atlas_to_query` to assign a projected type to each of the
       query metacells based on the ``atlas_type_property_name`` (default: {atlas_type_property_name}).

    For each type of query metacells:

    If ``top_level_parallel`` (default: ``top_level_parallel``), do this in parallel. This seems to work better than
    doing this serially and using parallelism within each type. However this is still inefficient compared to using both
    types of parallelism at once, which the code currently can't do without non-trivial coding (this would have been
    trivial in Julia...).

    8. Invoke :py:func:`metacells.tools.project.find_systematic_genes` to compute the systematic genes using only
       metacells of this type in both the query and the atlas. Use these to create a list of type-specific ignored
       genes, instead of the global systematic genes list.

    9. Invoke :py:func:`metacells.tools.project.project_query_onto_atlas` and
       :py:func:`metacells.tools.project.compute_query_projection` to project each query metacell onto the atlas, using
       the type-specific ignored genes in addition to the global ignored genes. Note that this projection is not
       restricted to using atlas metacells of the specific type.

    10. Invoke :py:func:`metacells.tools.project.compute_significant_projected_fold_factors` to compute the significant
       fold factors between the query and its projection.

    .. todo::
        Implement a more clever typed projection algorithm that maximizes parallelism both across types and within each
        type. Which would have been trivial in a language like Julia with true multi-threading w/o a GIL and a global
        work-stealing scheduler.

    11. Invoke :py:func:`metacells.tools.project.find_biased_genes` to identify type-specific biased genes. If any such
       genes are found, add them to the type-specific ignored genes (instead of the global biased genes list) and repeat
       steps 9-10.

    And then:

    12. If steps 8-11 changed the type assigned to a query metacell (an extremely rate occurrence), repeat them (but do
       these steps no more than 3 times).

    13. Correlate the expression levels of each gene between the query and projection. If this is less than
       ``project_max_uncorrelated_gene_correlation`` (default: {project_max_uncorrelated_gene_correlation}), then mark
       the gene as ignored. If this is at least ``project_min_corrected_gene_correlation`` (default:
       {project_min_corrected_gene_correlation}), compute the ratio between the mean expression of the gene in the
       projection and the query. If this is at most 1/(1+``project_min_corrected_gene_factor``) or at least
       (1+``project_min_corrected_gene_factor``) (default: {project_min_corrected_gene_factor}), then multiply the
       gene's value by this factor so its level would match the atlas. If any genes were ignored or corrected, then
       start over from step 2 ignoring the uncorrelated genes and using the corrected gene expression levels (but do
       these steps no more than 3 times).

    14. Invoke :py:func:`metacells.tools.quality.compute_similar_query_metacells` to validate the updated projection.

    15. If ``renormalize_query_by_atlas`` (default: {renormalize_query_by_atlas}), then invoke
       :py:func:`metacells.tools.project.renormalize_query_by_atlas` using the ``renormalize_var_annotations``,
       ``renormalize_layers`` and ``renormalize_varp_annotations``, if any, to add an ``ATLASNORM`` pseudo-gene so that
       the fraction out of the total UMIs in the query of the genes common to the atlas would be the same on average as
       their fraction out of the total UMIs in the atlas.
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

    common_adata, common_qdata = _common_data(
        adata=adata,
        qdata=qdata,
        project_max_projection_fold_factor=project_max_projection_fold_factor,
        project_max_dissimilar_genes=project_max_dissimilar_genes,
        ignore_atlas_insignificant_genes=ignore_atlas_insignificant_genes,
        ignore_atlas_forbidden_genes=ignore_atlas_forbidden_genes,
    )

    atlas_total_common_umis = ut.get_o_numpy(common_adata, what, sum=True)

    repeat = 0
    while True:
        repeat += 1
        ut.log_calc("correlation repeat", repeat)

        query_total_common_umis = ut.get_o_numpy(common_qdata, what, sum=True)

        _compute_preliminary_projection(
            what=what,
            common_adata=common_adata,
            common_qdata=common_qdata,
            atlas_total_common_umis=atlas_total_common_umis,
            query_total_common_umis=query_total_common_umis,
            systematic_low_gene_quantile=systematic_low_gene_quantile,
            systematic_high_gene_quantile=systematic_high_gene_quantile,
            ignore_atlas_insignificant_genes=ignore_atlas_insignificant_genes,
            ignore_query_insignificant_genes=ignore_query_insignificant_genes,
            ignore_atlas_forbidden_genes=ignore_atlas_forbidden_genes,
            ignore_query_forbidden_genes=ignore_query_forbidden_genes,
            project_fold_normalization=project_fold_normalization,
            project_min_significant_gene_value=project_min_significant_gene_value,
            project_max_consistency_fold_factor=project_max_consistency_fold_factor,
            project_candidates_count=project_candidates_count,
            project_min_usage_weight=project_min_usage_weight,
            project_max_projection_fold_factor=project_max_projection_fold_factor,
            min_entry_project_fold_factor=min_entry_project_fold_factor,
            biased_min_metacells_fraction=biased_min_metacells_fraction,
            project_abs_folds=project_abs_folds,
            atlas_type_property_name=atlas_type_property_name,
            reproducible=reproducible,
        )

        weights = _compute_per_type_projection(
            what=what,
            common_adata=common_adata,
            common_qdata=common_qdata,
            atlas_total_common_umis=atlas_total_common_umis,
            query_total_common_umis=query_total_common_umis,
            systematic_low_gene_quantile=systematic_low_gene_quantile,
            systematic_high_gene_quantile=systematic_high_gene_quantile,
            biased_min_metacells_fraction=biased_min_metacells_fraction,
            project_fold_normalization=project_fold_normalization,
            project_candidates_count=project_candidates_count,
            project_min_significant_gene_value=project_min_significant_gene_value,
            project_min_usage_weight=project_min_usage_weight,
            project_max_consistency_fold_factor=project_max_consistency_fold_factor,
            project_max_projection_fold_factor=project_max_projection_fold_factor,
            project_max_dissimilar_genes=project_max_dissimilar_genes,
            min_entry_project_fold_factor=min_entry_project_fold_factor,
            project_abs_folds=project_abs_folds,
            atlas_type_property_name=atlas_type_property_name,
            top_level_parallel=top_level_parallel,
            reproducible=reproducible,
        )

        if repeat > 2 or not _correct_correlated_genes(
            what=what,
            common_qdata=common_qdata,
            project_max_uncorrelated_gene_correlation=project_max_uncorrelated_gene_correlation,
            project_min_corrected_gene_correlation=project_min_corrected_gene_correlation,
            project_min_corrected_gene_factor=project_min_corrected_gene_factor,
            reproducible=reproducible,
        ):
            break

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
        min_entry_project_fold_factor=min_entry_project_fold_factor,
        project_abs_folds=project_abs_folds,
        atlas_type_property_name=atlas_type_property_name,
        top_level_parallel=top_level_parallel,
        reproducible=reproducible,
    )

    _convey_query_common_to_full_data(what=what, qdata=qdata, common_qdata=common_qdata)

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
    ignore_atlas_insignificant_genes: bool,
    ignore_atlas_forbidden_genes: bool,
) -> Tuple[AnnData, AnnData]:
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

    ut.set_o_data(common_qdata, "common_cell_index_of_qdata", np.arange(common_qdata.n_obs))
    ut.set_v_data(common_qdata, "common_gene_index_of_qdata", np.arange(common_qdata.n_vars))
    ut.set_v_data(common_qdata, "biased_gene", np.full(common_qdata.n_vars, False))
    ut.set_v_data(common_qdata, "uncorrelated_gene", np.zeros(common_qdata.n_vars, dtype="bool"))
    ut.set_v_data(common_qdata, "correction_factor", np.full(common_qdata.n_vars, 1.0, dtype="float32"))
    ut.set_v_data(common_qdata, "atlas_gene", np.full(common_qdata.n_vars, True))

    if ignore_atlas_insignificant_genes:
        atlas_significant_mask = ut.get_v_numpy(common_adata, "significant_gene")
        ut.set_v_data(common_qdata, "atlas_significant_gene", atlas_significant_mask)

    if ignore_atlas_forbidden_genes:
        atlas_forbiden_mask = ut.get_v_numpy(common_adata, "forbidden_gene")
        ut.set_v_data(common_qdata, "atlas_forbidden_gene", atlas_forbiden_mask)

    return common_adata, common_qdata


@ut.logged()
@ut.timed_call()
def _convey_query_common_to_full_data(
    *,
    what: str,
    qdata: AnnData,
    common_qdata: AnnData,
) -> None:
    if id(qdata) == id(common_qdata):
        for data_name in ["full_metacell_index_of_qdata", "common_cell_index_of_qdata"]:
            del qdata.obs[data_name]

        for data_name in ["full_gene_index_of_qdata", "common_gene_index_of_qdata"]:
            del qdata.var[data_name]

        return

    assert common_qdata.n_obs == qdata.n_obs

    for data_name in ["similar", "projected_type", "projected_secondary_type"]:
        ut.set_o_data(qdata, data_name, ut.get_o_numpy(common_qdata, data_name))

    dissimilar_genes_count = ut.get_o_numpy(common_qdata, "dissimilar_genes_count")
    ut.log_calc("max dissimilar_genes_count", np.max(dissimilar_genes_count))
    ut.set_o_data(qdata, "dissimilar_genes_count", dissimilar_genes_count, formatter=ut.mask_description)

    full_gene_index_of_common_qdata = ut.get_v_numpy(common_qdata, "full_gene_index_of_qdata")

    for (data_name, dtype, default) in [
        ("atlas_gene", "bool", False),
        ("atlas_forbidden_gene", "bool", False),
        ("atlas_significant_gene", "bool", False),
        ("biased_gene", "bool", False),
        ("systematic_gene", "bool", False),
        ("ignored_gene", "bool", True),
        ("projected_correlation", "float32", 0.0),
        ("correlated_gene", "bool", False),
        ("uncorrelated_gene", "bool", False),
        ("correction_factor", "float32", 1.0),
    ]:
        if ut.has_data(common_qdata, data_name):
            full_data = np.full(qdata.n_vars, default, dtype=dtype)
            full_data[full_gene_index_of_common_qdata] = ut.get_v_numpy(common_qdata, data_name)
            ut.set_v_data(qdata, data_name, full_data)

    for type_name in np.unique(ut.get_o_numpy(qdata, "projected_type")):
        for data_name in ["systematic_gene", "ignored_gene", "biased_gene"]:
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
    full_projected_fold[:, full_gene_index_of_common_qdata] = ut.to_numpy_matrix(common_projected_fold)
    ut.set_vo_data(qdata, "projected_fold", sp.csr_matrix(full_projected_fold))

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
        forbidden_gene=False,
        gene_deviant_votes=0,
        high_relative_variance_gene=False,
        high_total_gene=False,
        noisy_lonely_gene=False,
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
        atlas_forbidden_gene=False,
        atlas_gene=False,
        atlas_significant_gene=False,
        biased_gene=False,
        correction_factor=1.0,
        correlated_gene=False,
        ignored_gene=False,
        manually_ignored_gene=False,
        projected_correlation=0.0,
        systematic_gene=False,
        uncorrelated_gene=False,
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
        var_annotations[f"systematic_gene_of_{type_name}"] = False
        var_annotations[f"biased_gene_of_{type_name}"] = False
        var_annotations[f"manually_ignored_gene_of_{type_name}"] = False
        var_annotations[f"ignored_gene_of_{type_name}"] = False

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
    common_adata: AnnData,
    common_qdata: AnnData,
    atlas_total_common_umis: ut.NumpyVector,
    query_total_common_umis: ut.NumpyVector,
    systematic_low_gene_quantile: float,
    systematic_high_gene_quantile: float,
    ignore_atlas_insignificant_genes: bool,
    ignore_query_insignificant_genes: bool,
    ignore_atlas_forbidden_genes: bool,
    ignore_query_forbidden_genes: bool,
    project_fold_normalization: float,
    project_min_significant_gene_value: float,
    project_max_consistency_fold_factor: float,
    project_candidates_count: int,
    project_min_usage_weight: float,
    project_max_projection_fold_factor: float,
    min_entry_project_fold_factor: float,
    biased_min_metacells_fraction: float,
    atlas_type_property_name: str,
    project_abs_folds: bool,
    reproducible: bool,
) -> None:
    ignored_mask_names = _initial_ignored_mask_names(
        what=what,
        common_adata=common_adata,
        common_qdata=common_qdata,
        atlas_total_common_umis=atlas_total_common_umis,
        query_total_common_umis=query_total_common_umis,
        ignore_atlas_insignificant_genes=ignore_atlas_insignificant_genes,
        ignore_query_insignificant_genes=ignore_query_insignificant_genes,
        ignore_atlas_forbidden_genes=ignore_atlas_forbidden_genes,
        ignore_query_forbidden_genes=ignore_query_forbidden_genes,
        systematic_low_gene_quantile=systematic_low_gene_quantile,
        systematic_high_gene_quantile=systematic_high_gene_quantile,
    )

    included_adata = common_adata
    included_qdata = common_qdata

    repeat = 0
    while True:
        repeat += 1
        ut.log_calc("biased repeat", repeat)

        weights, included_adata, included_qdata = _compute_global_projection(
            what=what,
            included_adata=included_adata,
            included_qdata=included_qdata,
            atlas_total_common_umis=atlas_total_common_umis,
            query_total_common_umis=query_total_common_umis,
            ignored_mask_names=ignored_mask_names,
            project_fold_normalization=project_fold_normalization,
            project_min_significant_gene_value=project_min_significant_gene_value,
            project_max_consistency_fold_factor=project_max_consistency_fold_factor,
            project_candidates_count=project_candidates_count,
            project_min_usage_weight=project_min_usage_weight,
            project_max_projection_fold_factor=project_max_projection_fold_factor,
            min_entry_project_fold_factor=min_entry_project_fold_factor,
            project_abs_folds=project_abs_folds,
            reproducible=reproducible,
        )

        if not _detect_global_biased_genes(
            weights=weights,
            common_adata=common_adata,
            common_qdata=common_qdata,
            included_qdata=included_qdata,
            project_max_projection_fold_factor=project_max_projection_fold_factor,
            biased_min_metacells_fraction=biased_min_metacells_fraction,
            atlas_type_property_name=atlas_type_property_name,
            project_abs_folds=project_abs_folds,
        ):
            break

        ignored_mask_names = ["biased_gene"]

    _convey_query_included_to_common_data(common_qdata=common_qdata, included_qdata=included_qdata)


def _convey_query_included_to_common_data(
    *,
    common_qdata: AnnData,
    included_qdata: AnnData,
) -> None:
    common_gene_index_of_included_qdata = ut.get_v_numpy(included_qdata, "common_gene_index_of_qdata")
    ignored_genes_mask = np.full(common_qdata.n_vars, True)
    ignored_genes_mask[common_gene_index_of_included_qdata] = False
    ut.set_v_data(common_qdata, "ignored_gene", ignored_genes_mask)

    projected_types = ut.get_o_numpy(included_qdata, "projected_type")
    ut.set_o_data(common_qdata, "projected_type", projected_types)


@ut.logged()
@ut.timed_call()
def _initial_ignored_mask_names(
    *,
    what: str,
    common_adata: AnnData,
    common_qdata: AnnData,
    atlas_total_common_umis: ut.NumpyVector,
    query_total_common_umis: ut.NumpyVector,
    ignore_atlas_insignificant_genes: bool,
    ignore_query_insignificant_genes: bool,
    ignore_atlas_forbidden_genes: bool,
    ignore_query_forbidden_genes: bool,
    systematic_low_gene_quantile: float,
    systematic_high_gene_quantile: float,
) -> List[str]:
    ignored_mask_names = ["|uncorrelated_gene", "|manually_ignored_gene?"]

    tl.find_systematic_genes(
        what,
        adata=common_adata,
        qdata=common_qdata,
        atlas_total_umis=atlas_total_common_umis,
        query_total_umis=query_total_common_umis,
        low_gene_quantile=systematic_low_gene_quantile,
        high_gene_quantile=systematic_high_gene_quantile,
    )
    ignored_mask_names.append("|systematic_gene")

    if ignore_atlas_insignificant_genes:
        ignored_mask_names.append("|~atlas_significant_gene")

    if ignore_query_insignificant_genes:
        ignored_mask_names.append("|~significant_gene")

    if ignore_atlas_forbidden_genes:
        ignored_mask_names.append("|atlas_forbidden_gene")

    if ignore_query_forbidden_genes:
        ignored_mask_names.append("|forbidden_gene")

    return ignored_mask_names


@ut.logged()
@ut.timed_call()
def _compute_global_projection(
    *,
    what: str,
    included_adata: AnnData,
    included_qdata: AnnData,
    atlas_total_common_umis: ut.NumpyVector,
    query_total_common_umis: ut.NumpyVector,
    ignored_mask_names: List[str],
    project_fold_normalization: float,
    project_min_significant_gene_value: float,
    project_max_consistency_fold_factor: float,
    project_candidates_count: int,
    project_min_usage_weight: float,
    project_max_projection_fold_factor: float,
    min_entry_project_fold_factor: float,
    project_abs_folds: bool,
    reproducible: bool,
) -> Tuple[ut.CompressedMatrix, AnnData, AnnData]:
    tl.combine_masks(included_qdata, ignored_mask_names, to="ignored_gene")
    included_genes_mask = ~ut.get_v_numpy(included_qdata, "ignored_gene")

    included_adata = ut.slice(included_adata, name=".included", vars=included_genes_mask)
    included_qdata = ut.slice(included_qdata, name=".included", vars=included_genes_mask)

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

    tl.compute_significant_projected_fold_factors(
        included_qdata,
        what,
        total_umis=query_total_common_umis,
        fold_normalization=project_fold_normalization,
        min_gene_fold_factor=project_max_projection_fold_factor,
        min_entry_fold_factor=min_entry_project_fold_factor,
        min_significant_gene_value=project_min_significant_gene_value,
        abs_folds=project_abs_folds,
    )

    return weights, included_adata, included_qdata


def _detect_global_biased_genes(
    *,
    weights: ut.CompressedMatrix,
    common_adata: AnnData,
    common_qdata: AnnData,
    included_qdata: AnnData,
    project_max_projection_fold_factor: float,
    biased_min_metacells_fraction: float,
    atlas_type_property_name: str,
    project_abs_folds: bool,
) -> bool:
    tl.project_atlas_to_query(
        adata=common_adata,
        qdata=included_qdata,
        weights=weights,
        property_name=atlas_type_property_name,
        to_property_name="projected_type",
    )

    tl.find_biased_genes(
        included_qdata,
        max_projection_fold_factor=project_max_projection_fold_factor,
        min_metacells_fraction=biased_min_metacells_fraction,
        abs_folds=project_abs_folds,
    )

    included_biased_genes_mask = ut.get_v_numpy(included_qdata, "biased_gene")
    if not np.any(included_biased_genes_mask):
        return False

    common_biased_genes_mask = ut.get_v_numpy(common_qdata, "biased_gene").copy()
    common_gene_index_of_included_qdata = ut.get_v_numpy(included_qdata, "common_gene_index_of_qdata")
    common_biased_genes_mask[common_gene_index_of_included_qdata] = included_biased_genes_mask
    ut.set_v_data(common_qdata, "biased_gene", common_biased_genes_mask)

    return True


@ut.logged()
@ut.timed_call()
def _compute_per_type_projection(  # pylint: disable=too-many-statements
    *,
    what: str,
    common_adata: AnnData,
    common_qdata: AnnData,
    atlas_total_common_umis: ut.NumpyVector,
    query_total_common_umis: ut.NumpyVector,
    systematic_low_gene_quantile: float,
    systematic_high_gene_quantile: float,
    biased_min_metacells_fraction: float,
    project_fold_normalization: float,
    project_candidates_count: int,
    project_min_significant_gene_value: float,
    project_min_usage_weight: float,
    project_max_consistency_fold_factor: float,
    project_max_projection_fold_factor: float,
    project_max_dissimilar_genes: int,
    min_entry_project_fold_factor: float,
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
            for data_name in ["systematic_gene", "biased_gene", "ignored_gene"]:
                full_name = f"{data_name}_of_{type_name}"
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
        systematic_gene_of_types = np.zeros((len(query_unique_types), common_qdata.n_vars), dtype="bool")
        biased_gene_of_types = np.zeros((len(query_unique_types), common_qdata.n_vars), dtype="bool")
        ignored_gene_of_types = np.zeros((len(query_unique_types), common_qdata.n_vars), dtype="bool")

        @ut.timed_call("single_type_projection")
        def _single_type_projection(
            type_index: int,
        ) -> Tuple[
            ut.NumpyVector,
            ut.NumpyVector,
            ut.ProperMatrix,
            ut.ProperMatrix,
            ut.NumpyVector,
            ut.NumpyVector,
            ut.NumpyVector,
        ]:
            return _compute_single_type_projection(
                what=what,
                type_name=query_unique_types[type_index],
                common_adata=common_adata,
                common_qdata=common_qdata,
                atlas_total_common_umis=atlas_total_common_umis,
                query_total_common_umis=query_total_common_umis,
                systematic_low_gene_quantile=systematic_low_gene_quantile,
                systematic_high_gene_quantile=systematic_high_gene_quantile,
                biased_min_metacells_fraction=biased_min_metacells_fraction,
                project_fold_normalization=project_fold_normalization,
                project_candidates_count=project_candidates_count,
                project_min_significant_gene_value=project_min_significant_gene_value,
                project_min_usage_weight=project_min_usage_weight,
                project_max_consistency_fold_factor=project_max_consistency_fold_factor,
                project_max_projection_fold_factor=project_max_projection_fold_factor,
                project_max_dissimilar_genes=project_max_dissimilar_genes,
                min_entry_project_fold_factor=min_entry_project_fold_factor,
                project_abs_folds=project_abs_folds,
                atlas_type_property_name=atlas_type_property_name,
                top_level_parallel=top_level_parallel,
                reproducible=reproducible,
            )

        def _collect_type_result(
            type_index: int,
            similar_of_type: ut.NumpyVector,
            dissimilar_genes_count_of_type: ut.NumpyVector,
            weights_of_type: ut.ProperMatrix,
            projected_folds_of_type: ut.ProperMatrix,
            systematic_gene_of_type: ut.NumpyVector,
            biased_gene_of_type: ut.NumpyVector,
            ignored_gene_of_type: ut.NumpyVector,
        ) -> None:
            nonlocal similar
            similar |= similar_of_type
            nonlocal dissimilar_genes_count
            dissimilar_genes_count += dissimilar_genes_count_of_type
            nonlocal weights
            weights += weights_of_type  # type: ignore
            nonlocal projected_folds
            projected_folds += projected_folds_of_type  # type: ignore
            nonlocal systematic_gene_of_types
            systematic_gene_of_types[type_index, :] = systematic_gene_of_type
            nonlocal biased_gene_of_types
            biased_gene_of_types[type_index, :] = biased_gene_of_type
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
                common_qdata,
                f"systematic_gene_of_{type_name}",
                ut.to_numpy_vector(systematic_gene_of_types[type_index, :]),
            )
            ut.set_v_data(
                common_qdata, f"biased_gene_of_{type_name}", ut.to_numpy_vector(biased_gene_of_types[type_index, :])
            )
            ut.set_v_data(
                common_qdata, f"ignored_gene_of_{type_name}", ut.to_numpy_vector(ignored_gene_of_types[type_index, :])
            )

        for type_name in set(atlas_unique_types) - set(query_unique_types):
            ut.set_v_data(common_qdata, f"systematic_gene_of_{type_name}", np.zeros(common_qdata.n_vars, dtype="bool"))
            ut.set_v_data(common_qdata, f"biased_gene_of_{type_name}", np.zeros(common_qdata.n_vars, dtype="bool"))
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
    systematic_low_gene_quantile: float,
    systematic_high_gene_quantile: float,
    biased_min_metacells_fraction: float,
    project_fold_normalization: float,
    project_candidates_count: int,
    project_min_significant_gene_value: float,
    project_min_usage_weight: float,
    project_max_consistency_fold_factor: float,
    project_max_projection_fold_factor: float,
    project_max_dissimilar_genes: int,
    min_entry_project_fold_factor: float,
    project_abs_folds: bool,
    atlas_type_property_name: str,
    top_level_parallel: bool,
    reproducible: bool,
) -> Tuple[
    ut.NumpyVector,
    ut.NumpyVector,
    ut.NumpyMatrix,
    ut.NumpyMatrix,
    ut.NumpyVector,
    ut.NumpyVector,
    ut.NumpyVector,
]:
    similar = np.zeros(common_qdata.n_obs, dtype="bool")
    dissimilar_genes_count = np.zeros(common_qdata.n_obs, dtype="int32")
    weights = np.zeros((common_qdata.n_obs, common_adata.n_obs), dtype="float32")
    projected_folds = np.zeros(common_qdata.shape, dtype="float32")
    systematic_gene_of_type = np.zeros(common_qdata.n_vars, dtype="bool")
    biased_gene_of_type = np.zeros(common_qdata.n_vars, dtype="bool")
    ignored_gene_of_type = np.zeros(common_qdata.n_vars, dtype="bool")

    (
        type_common_adata,
        type_common_qdata,
        atlas_type_total_common_umis,
        query_type_total_common_umis,
    ) = _common_type_data(
        type_name=type_name,
        common_adata=common_adata,
        common_qdata=common_qdata,
        atlas_total_common_umis=atlas_total_common_umis,
        query_total_common_umis=query_total_common_umis,
        atlas_type_property_name=atlas_type_property_name,
    )

    type_ignored_mask_names = _initial_type_ignored_mask_names(
        what=what,
        type_name=type_name,
        type_common_adata=type_common_adata,
        type_common_qdata=type_common_qdata,
        atlas_type_total_common_umis=atlas_type_total_common_umis,
        query_type_total_common_umis=query_type_total_common_umis,
        systematic_low_gene_quantile=systematic_low_gene_quantile,
        systematic_high_gene_quantile=systematic_high_gene_quantile,
        systematic_gene_of_type=systematic_gene_of_type,
    )

    type_included_adata = common_adata
    type_included_qdata = type_common_qdata

    repeat = 0
    while True:
        repeat += 1
        ut.log_calc(f"{type_name} biased repeat", repeat)

        type_included_adata, type_included_qdata, type_weights = _compute_type_projection(
            what=what,
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
            project_max_projection_fold_factor=project_max_projection_fold_factor,
            min_entry_project_fold_factor=min_entry_project_fold_factor,
            project_abs_folds=project_abs_folds,
            reproducible=reproducible,
        )
        if not _detect_type_biased_genes(
            type_name=type_name,
            type_included_qdata=type_included_qdata,
            project_max_projection_fold_factor=project_max_projection_fold_factor,
            biased_min_metacells_fraction=biased_min_metacells_fraction,
            project_abs_folds=project_abs_folds,
            biased_gene_of_type=biased_gene_of_type,
        ):
            break
        type_ignored_mask_names = [f"biased_gene_of_{type_name}"]

    tl.compute_similar_query_metacells(
        type_included_qdata,
        max_projection_fold_factor=project_max_projection_fold_factor,
        max_dissimilar_genes=project_max_dissimilar_genes,
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
        projected_folds = sp.csr_matrix(projected_folds)

    return (
        similar,
        dissimilar_genes_count,
        weights,
        projected_folds,
        systematic_gene_of_type,
        biased_gene_of_type,
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
    common_adata: AnnData,
    common_qdata: AnnData,
    atlas_total_common_umis: ut.NumpyVector,
    query_total_common_umis: ut.NumpyVector,
    atlas_type_property_name: str,
) -> Tuple[AnnData, AnnData, ut.NumpyVector, ut.NumpyVector]:
    type_of_atlas_metacells = ut.get_o_numpy(common_adata, atlas_type_property_name)
    type_of_query_metacells = ut.get_o_numpy(common_qdata, "projected_type")

    atlas_type_mask = type_of_atlas_metacells == type_name
    query_type_mask = type_of_query_metacells == type_name

    ut.log_calc("atlas cells mask", atlas_type_mask)
    ut.log_calc("query cells mask", query_type_mask)

    type_common_adata = ut.slice(common_adata, obs=atlas_type_mask, name=f".{type_name}")
    type_common_qdata = ut.slice(
        common_qdata, obs=query_type_mask, name=f".{type_name}", track_var="full_metacell_index_of_qdata"
    )

    atlas_type_total_common_umis = atlas_total_common_umis[atlas_type_mask]
    query_type_total_common_umis = query_total_common_umis[query_type_mask]

    return type_common_adata, type_common_qdata, atlas_type_total_common_umis, query_type_total_common_umis


@ut.logged()
@ut.timed_call()
def _initial_type_ignored_mask_names(
    *,
    what: str,
    type_name: str,
    type_common_adata: AnnData,
    type_common_qdata: AnnData,
    atlas_type_total_common_umis: ut.NumpyVector,
    query_type_total_common_umis: ut.NumpyVector,
    systematic_low_gene_quantile: float,
    systematic_high_gene_quantile: float,
    systematic_gene_of_type: ut.NumpyVector,
) -> List[str]:
    type_ignored_mask_names = ["|ignored_gene", f"|manually_ignored_gene_of_{type_name}?"]

    tl.find_systematic_genes(
        what,
        adata=type_common_adata,
        qdata=type_common_qdata,
        atlas_total_umis=atlas_type_total_common_umis,
        query_total_umis=query_type_total_common_umis,
        low_gene_quantile=systematic_low_gene_quantile,
        high_gene_quantile=systematic_high_gene_quantile,
        to_property_name=f"systematic_gene_of_{type_name}",
    )
    type_ignored_mask_names.append(f"|systematic_gene_of_{type_name}")

    systematic_gene_of_type[:] = ut.get_v_numpy(type_common_qdata, f"systematic_gene_of_{type_name}")

    return type_ignored_mask_names


@ut.logged()
@ut.timed_call()
def _compute_type_projection(
    *,
    what: str,
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
    project_max_projection_fold_factor: float,
    min_entry_project_fold_factor: float,
    project_abs_folds: bool,
    reproducible: bool,
) -> Tuple[AnnData, AnnData, ut.CompressedMatrix]:
    tl.combine_masks(type_included_qdata, type_ignored_mask_names, to=f"ignored_gene_of_{type_name}")
    type_included_genes_mask = ~ut.get_v_numpy(type_included_qdata, f"ignored_gene_of_{type_name}")
    type_included_adata = ut.slice(type_included_adata, name=".included", vars=type_included_genes_mask)
    type_included_qdata = ut.slice(type_included_qdata, name=".included", vars=type_included_genes_mask)

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

    tl.compute_significant_projected_fold_factors(
        type_included_qdata,
        what,
        total_umis=query_type_total_common_umis,
        fold_normalization=project_fold_normalization,
        min_gene_fold_factor=project_max_projection_fold_factor,
        min_entry_fold_factor=min_entry_project_fold_factor,
        min_significant_gene_value=project_min_significant_gene_value,
        abs_folds=project_abs_folds,
    )

    return type_included_adata, type_included_qdata, type_weights


def _detect_type_biased_genes(
    *,
    type_name: str,
    type_included_qdata: AnnData,
    project_max_projection_fold_factor: float,
    biased_min_metacells_fraction: float,
    project_abs_folds: bool,
    biased_gene_of_type: ut.NumpyVector,
) -> bool:
    tl.find_biased_genes(
        type_included_qdata,
        max_projection_fold_factor=project_max_projection_fold_factor,
        min_metacells_fraction=biased_min_metacells_fraction,
        abs_folds=project_abs_folds,
        to_property_name=f"biased_gene_of_{type_name}",
    )

    new_type_biased_genes_mask = ut.get_v_numpy(type_included_qdata, f"biased_gene_of_{type_name}")
    if not np.any(new_type_biased_genes_mask):
        return False

    common_gene_index_of_type_included_qdata = ut.get_v_numpy(type_included_qdata, "common_gene_index_of_qdata")
    biased_gene_of_type[common_gene_index_of_type_included_qdata] = new_type_biased_genes_mask
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
    common_qdata: AnnData,
    project_max_uncorrelated_gene_correlation: float,
    project_min_corrected_gene_correlation: float,
    project_min_corrected_gene_factor: float,
    reproducible: bool,
) -> bool:
    projected_gene_columns = ut.to_numpy_matrix(ut.get_vo_proper(common_qdata, "projected", layout="column_major"))
    projected_gene_rows = projected_gene_columns.transpose()

    observed_gene_columns = ut.to_numpy_matrix(ut.get_vo_proper(common_qdata, what, layout="column_major"))
    observed_gene_rows = observed_gene_columns.transpose()

    common_gene_correlations = ut.pairs_corrcoef_rows(
        projected_gene_rows, observed_gene_rows, reproducible=reproducible
    )
    assert len(common_gene_correlations) == common_qdata.n_vars
    ut.set_v_data(common_qdata, "projected_correlation", common_gene_correlations)

    prev_common_uncorrelated_genes_mask = ut.get_v_numpy(common_qdata, "uncorrelated_gene")

    did_correct = False

    new_common_uncorrelated_genes_mask = common_gene_correlations <= project_max_uncorrelated_gene_correlation
    add_common_uncorrelated_genes_mask = new_common_uncorrelated_genes_mask & ~prev_common_uncorrelated_genes_mask
    ut.log_calc("add_common_uncorrelated_genes_mask", add_common_uncorrelated_genes_mask)
    if np.any(add_common_uncorrelated_genes_mask):
        new_common_uncorrelated_genes_mask |= prev_common_uncorrelated_genes_mask
        ut.set_v_data(common_qdata, "uncorrelated_gene", new_common_uncorrelated_genes_mask)
        did_correct = True

    correlated_common_genes_mask = common_gene_correlations >= project_min_corrected_gene_correlation
    ut.set_v_data(common_qdata, "correlated_gene", correlated_common_genes_mask)

    if np.any(correlated_common_genes_mask):
        correlated_common_gene_indices = np.where(correlated_common_genes_mask)[0]
        projected_correlated_gene_rows = projected_gene_rows[correlated_common_gene_indices, :]
        observed_correlated_gene_rows = observed_gene_rows[correlated_common_gene_indices, :]
        projected_correlated_genes_totals = ut.sum_per(projected_correlated_gene_rows, per="row")
        observed_correlated_genes_totals = ut.sum_per(observed_correlated_gene_rows, per="row")
        correlated_common_genes_correction_factors = (
            projected_correlated_genes_totals / observed_correlated_genes_totals
        )

        corrected_genes_mask = (
            correlated_common_genes_correction_factors < (1.0 / (1 + project_min_corrected_gene_factor))
        ) | (correlated_common_genes_correction_factors > (1 + project_min_corrected_gene_factor))

        if ut.has_data(common_qdata, "atlas_significant_gene"):
            atlas_significant_mask = ut.get_v_numpy(common_qdata, "atlas_significant_gene")
            corrected_genes_mask &= atlas_significant_mask[correlated_common_gene_indices]

        ut.log_calc("corrected_genes_mask", corrected_genes_mask)
        if np.any(corrected_genes_mask):
            corrected_common_gene_indices = correlated_common_gene_indices[corrected_genes_mask]
            corrected_gene_factors = correlated_common_genes_correction_factors[corrected_genes_mask]
            prev_common_genes_correction_factors = ut.get_v_numpy(common_qdata, "correction_factor")
            new_common_genes_correction_factors = prev_common_genes_correction_factors.copy()
            new_common_genes_correction_factors[corrected_common_gene_indices] *= corrected_gene_factors
            ut.set_v_data(common_qdata, "correction_factor", new_common_genes_correction_factors)
            corrected_data = observed_gene_columns.copy()
            corrected_data[:, corrected_common_gene_indices] *= corrected_gene_factors[None, :]
            ut.set_vo_data(common_qdata, what, corrected_data)
            did_correct = True

    return did_correct


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
    min_entry_project_fold_factor: float,
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
            min_entry_project_fold_factor=min_entry_project_fold_factor,
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

    ut.set_vo_data(common_qdata, "projected_fold", sp.csr_matrix(projected_folds))
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
def _compute_single_metacell_residuals(
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
    min_entry_project_fold_factor: float,
    project_abs_folds: bool,
    atlas_type_property_name: str,
    reproducible: bool,
) -> Tuple[ut.NumpyVector, ut.NumpyVector, ut.NumpyVector, str, str, bool]:
    primary_type = ut.get_o_numpy(dissimilar_qdata, "projected_type")[dissimilar_metacell_index]
    secondary_type = ""

    repeat = 0
    while True:
        repeat += 1
        ut.log_calc("types repeat", repeat)
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

        tl.compute_query_projection(
            adata=included_adata,
            qdata=metacell_included_qdata,
            weights=weights,
            atlas_total_umis=atlas_total_common_umis,
            query_total_umis=metacell_total_common_umis,
        )

        first_anchor_weights = weights.copy()
        first_anchor_weights[:, second_anchor_indices] = 0.0

        tl.project_atlas_to_query(
            adata=included_adata,
            qdata=metacell_included_qdata,
            weights=first_anchor_weights,
            property_name=atlas_type_property_name,
            to_property_name="projected_type",
        )
        new_primary_type = ut.get_o_numpy(metacell_included_qdata, "projected_type")[0]

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

    tl.compute_significant_projected_fold_factors(
        metacell_included_qdata,
        what,
        total_umis=metacell_total_common_umis,
        fold_normalization=project_fold_normalization,
        min_gene_fold_factor=project_max_projection_fold_factor,
        min_entry_fold_factor=min_entry_project_fold_factor,
        min_significant_gene_value=project_min_significant_gene_value,
        abs_folds=project_abs_folds,
    )

    tl.compute_similar_query_metacells(
        metacell_included_qdata,
        max_projection_fold_factor=project_max_projection_fold_factor,
        max_dissimilar_genes=project_max_dissimilar_genes,
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
