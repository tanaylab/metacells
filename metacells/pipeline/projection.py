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
from metacells import __version__  # pylint: disable=cyclic-import

__all__ = [
    "projection_pipeline",
    "outliers_projection_pipeline",
    "write_projection_weights",
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
    project_log_data: bool = pr.project_log_data,
    project_fold_regularization: float = pr.project_fold_regularization,
    project_candidates_count: int = pr.project_candidates_count,
    project_min_candidates_fraction: float = pr.project_min_candidates_fraction,
    project_min_significant_gene_umis: int = pr.project_min_significant_gene_umis,
    project_min_usage_weight: float = pr.project_min_usage_weight,
    project_filter_ranges: bool = pr.project_filter_ranges,
    project_ignore_range_quantile: float = pr.project_ignore_range_quantile,
    project_ignore_range_min_overlap_fraction: float = pr.project_ignore_range_min_overlap_fraction,
    project_min_query_markers_fraction: float = pr.project_min_query_markers_fraction,
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
    containing such a matrix, containing the fraction of each gene in each cell. The atlas should also contain a
    ``type`` per-observation (metacell) annotation.

    The ``qdata`` may include per-gene masks, ``ignored_gene`` and ``ignored_gene_of_<type>``, which force the code
    below to ignore the marked genes from either the preliminary projection or the following refined type-specific
    projections.

    **Returns**

    A matrix whose rows are query metacells and columns are atlas metacells, where each entry is the weight of the atlas
    metacell in the projection of the query metacells. The sum of weights in each row (that is, for a single query
    metacell) is 1. The weighted sum of the atlas metacells using these weights is the "projected" image of the query
    metacell onto the atlas.

    Variable (Gene) Annotations
        ``atlas_gene``
            A mask of the query genes that also exist in the atlas. We match genes by their name; if projecting query
            data from a different technology, we expect the caller to modify the query gene names to match the atlas
            before projecting it.

        ``atlas_lateral_gene``, ``atlas_noisy_gene``, ``atlas_marker_gene``, ``essential_gene_of_<type>``
            Copied from the atlas to the query (``False`` for non-``atlas_gene``).

        ``projected_noisy_gene``
            The mask of the genes that were considered "noisy" when computing the projection. By default this is the
            union of the noisy atlas and query genes.

        ``correction_factor`` (if ``project_corrections``)
            If projecting a query on an atlas with different technologies (e.g., 10X v3 to 10X v2), an automatically
            computed factor we multiplied the query gene fractions by to compensate for the systematic difference
            between the technologies (1.0 for uncorrected genes and 0.0 for non-``atlas_gene``).

        ``fitted_gene_of_<type>``
            For each type, the genes that were projected well from the query to the atlas for most cells of that
            type; any ``atlas_gene`` outside this mask failed to project well from the query to the atlas for most
            metacells of this type. For non-``atlas_gene`` this is set to ``False``.

        ``misfit_gene_of_<type>``
            For each query metacell type, a boolean mask indicating whether the gene has a strong bias in the query
            metacells of this type compared to the atlas metacells of this type.

        ``ignored_gene_of_<type>``
            For each query metacell type, a boolean mask indicating whether the gene was ignored by the projection (for
            any reason) when computing the projection for metacells of this type.

    Observation (Cell) Annotations
        ``total_atlas_umis``
            The total UMIs of the ``atlas_gene`` in each query metacell. This is used in the analysis as described for
            ``total_umis`` above, that is, to ensure comparing expression levels will ignore cases where the total
            number of UMIs of both compared gene profiles is too low to make a reliable determination. In such cases we
            take the fold factor to be 0.

        ``projected_type``
            For each query metacell, the best atlas ``type`` we can assign to it based on its projection. Note this does
            not indicate that the query metacell is "truly" of this type; to make this determination one needs to look
            at the quality control data below.

        ``projected_secondary_type``
            In some cases, a query metacell may fail to project well to a single region of the atlas, but does project
            well to a combination of two distinct atlas regions. This may be due to the query metacell containing
            doublets, of a mixture of cells which match different atlas regions (e.g. due to sparsity of data in the
            query data set). Either way, if this happens, we place here the type that best describes the secondary
            region the query metacell was projected to; otherwise this would be the empty string. Note that the
            ``weights`` matrix above does not distinguish between the regions.

        ``projected_correlation`` per query metacell
            The correlation between between the ``corrected_fraction`` and the ``projected_fraction`` for only the
            ``fitted_gene`` expression levels of each query metacell. This serves as a very rough estimator for the
            quality of the projection for this query metacell (e.g. can be used to compute R^2 values).

            In general we expect high correlation (more than 0.9 in most metacells) since we restricted the
            ``fitted_gene`` mask only to genes we projected well.

        ``similar`` mask per query metacell
            A conservative determination of whether the query metacell is "similar" to its projection on the atlas. This
            is based on whether the number of ``misfit_gene`` for the query metacell is low enough (by default, up to 3
            genes), and also that at least 75% of the ``essential_gene`` of the query metacell were not ``misfit_gene``.
            Note that this explicitly allows for a ``projected_secondary_type``, that is, a metacell of doublets will be
            "similar" to the atlas, but a metacell of a novel state missing from the atlas will be "dissimilar".

            The final determination of whether to accept the projection is, as always, up to the analyst, based on prior
            biological knowledge, the context of the collection of the query (and atlas) data sets, etc. The analyst
            need not (indeed, *should not*) blindly accept the ``similar`` determination without examining the rest of
            the quality control data listed above.

    Observation-Variable (Cell-Gene) Annotations
        ``corrected_fraction`` per gene per query metacell
            For each ``atlas_gene``, its fraction in each query metacell, out of only the atlas genes. This may be
            further corrected (see below) if projecting between different scRNA-seq technologies (e.g. 10X v2 and 10X
            v3). For non-``atlas_gene`` this is 0.

        ``projected_fraction`` per gene per query metacell
            For each ``atlas_gene``, its fraction in its projection on the atlas. This projection is computed as a
            weighted average of some atlas metacells (see below), which are all sufficiently close to each other (in
            terms of gene expression), so averaging them is reasonable to capture the fact the query metacell may be
            along some position on some gradient that isn't an exact match for any specific atlas metacell. For
            non-``atlas_gene`` this is 0.

        ``fitted`` mask per gene per query metacell
            For each ``atlas_gene`` for each query metacell, whether the gene was expected to be projected well, based
            on the query metacell ``projected_type`` (and the ``projected_secondary_type``, if any). For
            non-``atlas_gene`` this is set to ``False``. This does not guarantee the gene was actually projected well.

        ``misfit``
            For each ``atlas_gene`` for each query metacell, whether the ``corrected_fraction`` of the gene was
            significantly different from the ``projected_fractions`` (that is, whether the gene was not projected well
            for this metacell). For non-``atlas_gene`` this is set to ``False``, to make it easier to identify
            problematic genes.

            This is expected to be rare for ``fitted_gene`` and common for the rest of the ``atlas_gene``. If too many
            ``fitted_gene`` are also ``misfit_gene``, then one should be suspicious whether the query metacell is
            "truly" of the ``projected_type``.

        ``essential``
            Which of the ``atlas_gene`` were also listed in the ``essential_gene_of_<type>`` for the ``projected_type``
            (and also the ``projected_secondary_type``, if any) of each query metacell.

            If an ``essential_gene`` is also a ``misfit_gene``, then one should be very suspicious whether the query
            metacell is "truly" of the ``projected_type``.

        ``projected_fold`` per gene per query metacell
            The fold factor between the ``corrected_fraction`` and the ``projected_fraction`` (0 for
            non-``atlas_gene``). If the absolute value of this is high (3 for 8x ratio) then the gene was not projected
            well for this metacell. This will be 0 for non-``atlas_gene``.

            It is expected this would have low values for most ``fitted_gene`` and high values for the rest of the
            ``atlas_gene``, but specific values will vary from one query metacell to another. This allows the analyst to
            make fine-grained determination about the quality of the projection, and/or identify quantitative
            differences between the query and the atlas (e.g., when studying perturbed systems such as knockouts or
            disease models).

    **Computation Parameters**

    0. Find the subset of genes that exist in both the query and the atlas. All computations will be done on this common
       subset. Normalize the fractions of the fitted gene fractions to sum to 1 in each metacell.

    Compute preliminary projection:

    1. Compute a mask of fitted genes, ignoring any genes included in ``ignored_gene``. If ``only_atlas_marker_genes``
       (default: ``only_atlas_marker_genes``), ignore any non-``marker_gene`` of the atlas. If
       ``only_query_marker_genes`` (default: ``only_query_marker_genes``), ignore any non-``marker_gene`` of the query.
       If ``ignore_atlas_lateral_genes`` (default: {ignore_atlas_lateral_genes}), ignore the ``lateral_gene`` of the
       atlas. If ``ignore_query_lateral_genes`` (default: {ignore_query_lateral_genes}), ignore the ``lateral_gene`` of
       the atlas. Normalize the fractions of the fitted gene fractions to sum to 1 in each metacell.

    2. Invoke :py:func:`metacells.tools.project.compute_projection_weights`
       to project each query metacell onto the atlas, using the ``project_log_data`` (default: {project_log_data}),
       ``project_fold_regularization`` (default: {project_fold_regularization}), ``project_min_significant_gene_umis``
       (default: {project_min_significant_gene_umis}), ``project_max_consistency_fold_factor`` (default:
       {project_max_consistency_fold_factor}), ``project_candidates_count`` (default: {project_candidates_count}),
       ``project_min_usage_weight`` (default: {project_min_usage_weight}), and ``reproducible``.

    3. If ``project_corrections`` (default: {project_corrections}: Correlate the expression levels of each gene between
       the query and projection. If this is at least ``project_min_corrected_gene_correlation`` (default:
       {project_min_corrected_gene_correlation}), compute the ratio between the mean expression of the gene in the
       projection and the query. If this is at most 1/(1+``project_min_corrected_gene_factor``) or at least
       (1+``project_min_corrected_gene_factor``) (default: {project_min_corrected_gene_factor}), then multiply the
       gene's value by this factor so its level would match the atlas. As usual, ignore genes which do not have at least
       ``project_min_significant_gene_umis``. If any genes were corrected, then repeat steps 2-3 (but do these steps no
       more than 3 times).

    4. If ``project_filter_ranges`` (default: {project_filter_ranges}): Compute for each gene its expression range
       (lowest and highest ``project_ignore_range_quantile`` (default: {project_ignore_range_quantile}) in both the
       projected and the corrected values. Compute the overlap between these ranges (shared range divided by the query
       range). If this is less than ``project_ignore_range_min_overlap_fraction`` (default:
       {project_ignore_range_min_overlap_fraction}), then ignore the gene. If any genes were ignored, repeat steps 2-4
       (but do this step no more than 3 times).

    5. Invoke :py:func:`metacells.tools.project.convey_atlas_to_query` to assign a projected type to each of the
       query metacells based on the ``atlas_type_property_name`` (default: {atlas_type_property_name}).

    Then, for each type of query metacells:

    If ``top_level_parallel`` (default: ``top_level_parallel``), do this in parallel. This seems to work better than
    doing this serially and using parallelism within each type. However this is still inefficient compared to using both
    types of parallelism at once, which the code currently can't do without non-trivial coding (this would have been
    trivial in Julia...).

    6. Further reduce the mask of type-specific fitted genes by ignoring any genes in ``ignored_gene_of_<type>``, if
       this annotation exists in the query. Normalize the sum of the fitted gene fractions to 1 in each metacell.

    7. Invoke :py:func:`metacells.tools.project.compute_projection_weights` to project each query metacell of the type
       onto the atlas. Note that even though we only look at query metacells (tentatively) assigned the type, their
       projection on the atlas may use metacells of any type.

    8. Invoke :py:func:`metacells.tools.quality.compute_projected_folds` to compute the significant fold factors
       between the query and its projection.

    9. Identify type-specific misfit genes whose fold factor is above ``project_max_projection_fold_factor``. If
       ``consider_atlas_noisy_genes`` and/or ``consider_query_noisy_genes``, then any gene listed in either is allowed
       an additional ``project_max_projection_noisy_fold_factor``. If any gene has such a high fold factor in at least
       ``misfit_min_metacells_fraction``, remove it from the fitted genes mask and repeat steps 6-9.

    10. Invoke :py:func:`metacells.tools.quality.compute_similar_query_metacells` to verify which query metacells
        ended up being sufficiently similar to their projection, using ``project_max_consistency_fold_factor`` (default:
        {project_max_consistency_fold_factor}), ``project_max_projection_noisy_fold_factor`` (default:
        {project_max_projection_noisy_fold_factor}), ``project_max_misfit_genes`` (default: {project_max_misfit_genes}),
        and if needed, the the ``essential_gene_of_<type>``. Also compute the correlation between the corrected
        and projected gene fractions for each metacell.

    And then:

    11. Invoke :py:func:`metacells.tools.project.convey_atlas_to_query` to assign an updated projected type to each of
        the query metacells based on the ``atlas_type_property_name`` (default: {atlas_type_property_name}). If this
        changed the type assigned to any query metacell, repeat steps 6-11 (but do this step no more than 3 times).

    For each query metacell that ended up being dissimilar, try to project them as a combination of two atlas regions:

    12. Reduce the list of fitted genes for the metacell based on the ``ignored_gene_of_<type>`` for both the primary
        (initially from the above) query metacell type and the secondary query metacell type (initially empty).
        Normalize the sum of the gene fractions in the metacell to 1.

    13. Invoke :py:func:`metacells.tools.project.compute_projection_weights` just for this metacell, allowing the
        projection to use a secondary location in the atlas based on the residuals of the atlas metacells relative to
        the primary query projection.

    14. Invoke :py:func:`metacells.tools.project.convey_atlas_to_query` twice, once for the weights of the primary
        location and once for the weights of the secondary location, to obtain a primary and secondary type for
        the query metacell. If these have changed, repeat steps 13-14 (but do these steps no more than 3 times; note
        will will always do them twice as the 1st run will generate some non-empty secondary type).

    15. Invoke :py:func:`metacells.tools.quality.compute_projected_folds` and
        :py:func:`metacells.tools.quality.compute_similar_query_metacells` to update the projection and evaluation of
        the query metacell. If it is now similar, then use the results for the metacell; otherwise, keep the original
        results as they were at the end of step 10.
    """
    assert project_min_corrected_gene_factor >= 0
    use_essential_genes = project_min_essential_genes_fraction is not None and _has_any_essential_genes(adata)

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

    common_adata = _initialize_common_adata(adata, what, atlas_common_gene_indices)
    common_qdata = _initialize_common_qdata(qdata, what, query_common_gene_indices)

    query_marker_genes_mask = ut.get_v_numpy(common_qdata, "marker_gene")
    min_fitted_query_marker_genes = np.sum(query_marker_genes_mask) * project_min_query_markers_fraction
    ut.log_calc("min_fitted_query_marker_genes", min_fitted_query_marker_genes)

    min_essential_genes_of_type = _min_essential_genes_of_type(
        adata=adata,
        common_adata=common_adata,
        type_names=type_names,
        project_min_essential_genes_fraction=project_min_essential_genes_fraction,
    )

    preliminary_fitted_genes_mask = _preliminary_fitted_genes_mask(
        common_adata=common_adata,
        common_qdata=common_qdata,
        only_atlas_marker_genes=only_atlas_marker_genes,
        only_query_marker_genes=only_query_marker_genes,
        ignore_atlas_lateral_genes=ignore_atlas_lateral_genes,
        ignore_query_lateral_genes=ignore_query_lateral_genes,
        min_significant_gene_umis=project_min_significant_gene_umis,
    )

    _compute_preliminary_projection(
        common_adata=common_adata,
        common_qdata=common_qdata,
        preliminary_fitted_genes_mask=preliminary_fitted_genes_mask,
        project_log_data=project_log_data,
        project_fold_regularization=project_fold_regularization,
        project_min_significant_gene_umis=project_min_significant_gene_umis,
        project_filter_ranges=project_filter_ranges,
        project_ignore_range_quantile=project_ignore_range_quantile,
        project_ignore_range_min_overlap_fraction=project_ignore_range_min_overlap_fraction,
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
        common_adata=common_adata,
        common_qdata=common_qdata,
        type_names=type_names,
        preliminary_fitted_genes_mask=preliminary_fitted_genes_mask,
        misfit_min_metacells_fraction=misfit_min_metacells_fraction,
        project_log_data=project_log_data,
        project_fold_regularization=project_fold_regularization,
        project_candidates_count=project_candidates_count,
        project_min_significant_gene_umis=project_min_significant_gene_umis,
        project_min_usage_weight=project_min_usage_weight,
        project_max_consistency_fold_factor=project_max_consistency_fold_factor,
        project_max_projection_fold_factor=project_max_projection_fold_factor,
        project_max_projection_noisy_fold_factor=project_max_projection_noisy_fold_factor,
        project_max_misfit_genes=project_max_misfit_genes,
        min_fitted_query_marker_genes=min_fitted_query_marker_genes,
        min_essential_genes_of_type=min_essential_genes_of_type,
        atlas_type_property_name=atlas_type_property_name,
        top_level_parallel=top_level_parallel,
        reproducible=reproducible,
    )

    _compute_dissimilar_residuals_projection(
        weights_per_atlas_per_query_metacell=weights,
        common_adata=common_adata,
        common_qdata=common_qdata,
        project_log_data=project_log_data,
        project_fold_regularization=project_fold_regularization,
        project_candidates_count=project_candidates_count,
        project_min_candidates_fraction=project_min_candidates_fraction,
        project_min_significant_gene_umis=project_min_significant_gene_umis,
        project_min_usage_weight=project_min_usage_weight,
        project_max_consistency_fold_factor=project_max_consistency_fold_factor,
        project_max_projection_fold_factor=project_max_projection_fold_factor,
        project_max_projection_noisy_fold_factor=project_max_projection_noisy_fold_factor,
        project_max_misfit_genes=project_max_misfit_genes,
        min_fitted_query_marker_genes=min_fitted_query_marker_genes,
        min_essential_genes_of_type=min_essential_genes_of_type,
        atlas_type_property_name=atlas_type_property_name,
        top_level_parallel=top_level_parallel,
        reproducible=reproducible,
    )

    tl.compute_projected_fractions(
        adata=common_adata,
        qdata=common_qdata,
        log_data=project_log_data,
        fold_regularization=project_fold_regularization,
        weights=weights,
    )

    _common_data_to_full(
        qdata=qdata,
        common_qdata=common_qdata,
        project_corrections=project_corrections,
        type_names=type_names,
        use_essential_genes=use_essential_genes,
    )

    ut.set_m_data(qdata, "projection_algorithm", f"metacells.{__version__}")
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


def _has_any_essential_genes(adata: AnnData) -> bool:
    for property_name in adata.var.keys():
        if property_name.startswith("essential_gene_of_"):
            return True
    return False


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


def _initialize_common_adata(adata: AnnData, what: str, atlas_common_gene_indices: ut.NumpyVector) -> AnnData:
    common_adata = ut.slice(adata, name=".common", vars=atlas_common_gene_indices, top_level=False)

    atlas_total_common_umis = ut.get_o_numpy(common_adata, "total_umis", sum=True)
    ut.set_o_data(common_adata, "total_umis", atlas_total_common_umis)

    _normalize_corrected_fractions(common_adata, what)

    return common_adata


def _initialize_common_qdata(
    qdata: AnnData,
    what: str,
    query_common_gene_indices: ut.NumpyVector,
) -> AnnData:
    common_qdata = ut.slice(
        qdata,
        name=".common",
        vars=query_common_gene_indices,
        track_var="full_gene_index_of_qdata",
        top_level=False,
    )

    query_total_common_umis = ut.get_o_numpy(common_qdata, "total_umis", sum=True)
    ut.set_o_data(common_qdata, "total_umis", query_total_common_umis)
    ut.set_o_data(qdata, "total_atlas_umis", query_total_common_umis)

    _normalize_corrected_fractions(common_qdata, what)

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


def _preliminary_fitted_genes_mask(
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
    preliminary_fitted_genes_mask = total_umis_per_gene >= min_significant_gene_umis
    ut.log_calc("total_umis_mask", preliminary_fitted_genes_mask)

    if only_atlas_marker_genes and ut.has_data(common_adata, "marker_gene"):
        preliminary_fitted_genes_mask &= ut.get_v_numpy(common_adata, "marker_gene")

    if only_query_marker_genes and ut.has_data(common_qdata, "marker_gene"):
        preliminary_fitted_genes_mask &= ut.get_v_numpy(common_qdata, "marker_gene")

    if ignore_atlas_lateral_genes and ut.has_data(common_adata, "lateral_gene"):
        preliminary_fitted_genes_mask &= ~ut.get_v_numpy(common_adata, "lateral_gene")

    if ignore_query_lateral_genes and ut.has_data(common_qdata, "lateral_gene"):
        preliminary_fitted_genes_mask &= ~ut.get_v_numpy(common_qdata, "lateral_gene")

    if ut.has_data(common_qdata, "ignored_gene"):
        preliminary_fitted_genes_mask &= ~ut.get_v_numpy(common_qdata, "ignored_gene")

    return preliminary_fitted_genes_mask


def _common_data_to_full(
    *,
    qdata: AnnData,
    common_qdata: AnnData,
    project_corrections: bool,
    use_essential_genes: bool,
    type_names: List[str],
) -> None:
    if use_essential_genes:
        primary_type_per_metacell = ut.get_o_numpy(common_qdata, "projected_type")
        secondary_type_per_metacell = ut.get_o_numpy(common_qdata, "projected_secondary_type")
        essential_per_gene_per_metacell = np.zeros(qdata.shape, dtype="bool")
        essential_gene_per_type = {
            type_name: ut.get_v_numpy(qdata, f"essential_gene_of_{type_name}")
            for type_name in type_names
            if type_name != "Outliers"
        }
        for metacell_index in range(qdata.n_obs):
            primary_type_of_metacell = primary_type_per_metacell[metacell_index]
            if primary_type_of_metacell != "Outliers":
                essential_per_gene_per_metacell[metacell_index, :] = essential_gene_per_type[primary_type_of_metacell]

            secondary_type_of_metacell = secondary_type_per_metacell[metacell_index]
            if secondary_type_of_metacell not in ("", "Outliers", primary_type_of_metacell):
                essential_per_gene_per_metacell[metacell_index, :] |= essential_gene_per_type[
                    secondary_type_of_metacell
                ]

        ut.set_vo_data(qdata, "essential", sp.csr_matrix(essential_per_gene_per_metacell))

    full_gene_index_of_common_qdata = ut.get_v_numpy(common_qdata, "full_gene_index_of_qdata")

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

    property_names = [f"fitted_gene_of_{type_name}" for type_name in type_names]
    if project_corrections:
        property_names.append("correction_factor")

    for property_name in property_names:
        data_per_common_gene = ut.get_v_numpy(common_qdata, property_name)
        data_per_full_gene = np.zeros(qdata.n_vars, dtype=data_per_common_gene.dtype)
        data_per_full_gene[full_gene_index_of_common_qdata] = data_per_common_gene
        ut.set_v_data(qdata, property_name, data_per_full_gene)

    for property_name, formatter in (
        ("projected_type", None),
        ("projected_secondary_type", None),
        ("projected_correlation", ut.sizes_description),
        ("similar", None),
    ):
        data_per_metacell = ut.get_o_numpy(common_qdata, property_name, formatter=formatter)
        ut.set_o_data(qdata, property_name, data_per_metacell, formatter=formatter)


@ut.logged()
@ut.timed_call()
def _compute_preliminary_projection(
    *,
    common_adata: AnnData,
    common_qdata: AnnData,
    preliminary_fitted_genes_mask: ut.NumpyVector,
    project_log_data: bool,
    project_fold_regularization: float,
    project_min_significant_gene_umis: float,
    project_filter_ranges: bool,
    project_ignore_range_quantile: float,
    project_ignore_range_min_overlap_fraction: float,
    project_max_consistency_fold_factor: float,
    project_candidates_count: int,
    project_min_usage_weight: float,
    project_corrections: bool,
    project_min_corrected_gene_correlation: float,
    project_min_corrected_gene_factor: float,
    atlas_type_property_name: str,
    reproducible: bool,
) -> None:
    correction_factor_per_gene = np.full(common_qdata.n_vars, 1.0, dtype="float32")

    repeat = 0
    while True:
        repeat += 1
        ut.log_calc("preliminary repeat", repeat)

        fitted_adata, weights = _compute_correction_factors(
            common_adata=common_adata,
            common_qdata=common_qdata,
            correction_factor_per_gene=correction_factor_per_gene,
            preliminary_fitted_genes_mask=preliminary_fitted_genes_mask,
            project_log_data=project_log_data,
            project_fold_regularization=project_fold_regularization,
            project_min_significant_gene_umis=project_min_significant_gene_umis,
            project_max_consistency_fold_factor=project_max_consistency_fold_factor,
            project_candidates_count=project_candidates_count,
            project_min_usage_weight=project_min_usage_weight,
            project_corrections=project_corrections,
            project_min_corrected_gene_correlation=project_min_corrected_gene_correlation,
            project_min_corrected_gene_factor=project_min_corrected_gene_factor,
            reproducible=reproducible,
        )

        if (
            repeat > 2
            or not project_filter_ranges
            or not _filter_range_genes(
                common_qdata=common_qdata,
                project_fold_regularization=project_fold_regularization,
                project_ignore_range_quantile=project_ignore_range_quantile,
                project_ignore_range_min_overlap_fraction=project_ignore_range_min_overlap_fraction,
                preliminary_fitted_genes_mask=preliminary_fitted_genes_mask,
            )
        ):
            ut.log_calc("preliminary last repeat", repeat)
            break

    if project_corrections:
        ut.set_v_data(common_qdata, "correction_factor", correction_factor_per_gene)

    tl.convey_atlas_to_query(
        adata=fitted_adata,
        qdata=common_qdata,
        weights=weights,
        property_name=atlas_type_property_name,
        to_property_name="projected_type",
    )


@ut.logged()
@ut.timed_call()
def _compute_correction_factors(
    *,
    common_adata: AnnData,
    common_qdata: AnnData,
    correction_factor_per_gene: ut.NumpyVector,
    preliminary_fitted_genes_mask: ut.NumpyVector,
    project_log_data: bool,
    project_fold_regularization: float,
    project_min_significant_gene_umis: float,
    project_max_consistency_fold_factor: float,
    project_candidates_count: int,
    project_min_usage_weight: float,
    project_corrections: bool,
    project_min_corrected_gene_correlation: float,
    project_min_corrected_gene_factor: float,
    reproducible: bool,
) -> Tuple[AnnData, ut.ProperMatrix]:
    repeat = 0
    while True:
        repeat += 1
        ut.log_calc("corrections repeat", repeat)

        fitted_adata = ut.slice(common_adata, name=".fitted", vars=preliminary_fitted_genes_mask, top_level=False)

        fitted_qdata = ut.slice(
            common_qdata,
            name=".fitted",
            vars=preliminary_fitted_genes_mask,
            track_var="common_gene_index_of_qdata",
            top_level=False,
        )

        weights = tl.compute_projection_weights(
            adata=fitted_adata,
            qdata=fitted_qdata,
            log_data=project_log_data,
            fold_regularization=project_fold_regularization,
            min_significant_gene_umis=project_min_significant_gene_umis,
            max_consistency_fold_factor=project_max_consistency_fold_factor,
            candidates_count=project_candidates_count,
            min_usage_weight=project_min_usage_weight,
            reproducible=reproducible,
        )

        tl.compute_projected_fractions(
            adata=common_adata,
            qdata=common_qdata,
            log_data=project_log_data,
            fold_regularization=project_fold_regularization,
            weights=weights,
        )

        if (
            repeat > 2
            or not project_corrections
            or not _correct_correlated_genes(
                common_adata=common_adata,
                common_qdata=common_qdata,
                preliminary_fitted_genes_mask=preliminary_fitted_genes_mask,
                project_min_corrected_gene_correlation=project_min_corrected_gene_correlation,
                project_min_corrected_gene_factor=project_min_corrected_gene_factor,
                correction_factor_per_gene=correction_factor_per_gene,
                reproducible=reproducible,
            )
        ):
            ut.log_calc("corrections last repeat", repeat)
            return fitted_adata, weights


def _correct_correlated_genes(
    *,
    common_adata: AnnData,
    common_qdata: AnnData,
    preliminary_fitted_genes_mask: ut.NumpyVector,
    project_min_corrected_gene_correlation: float,
    project_min_corrected_gene_factor: float,
    correction_factor_per_gene: ut.NumpyVector,
    reproducible: bool,
) -> bool:
    corrected_fractions_per_gene_per_metacell = ut.to_numpy_matrix(
        ut.get_vo_proper(common_qdata, "corrected_fraction", layout="column_major")
    )

    projected_fractions_per_gene_per_metacell = ut.to_numpy_matrix(
        ut.get_vo_proper(common_qdata, "projected_fraction", layout="column_major")
    )

    preliminary_fitted_genes_indices = np.where(preliminary_fitted_genes_mask)[0]
    corrected_fractions_per_fitted_gene_per_metacell = corrected_fractions_per_gene_per_metacell[
        :, preliminary_fitted_genes_indices
    ]
    projected_fractions_per_fitted_gene_per_metacell = projected_fractions_per_gene_per_metacell[
        :, preliminary_fitted_genes_indices
    ]

    correlation_per_fitted_gene = np.full(len(preliminary_fitted_genes_indices), -2, dtype="float32")
    correlation_per_fitted_gene = ut.pairs_corrcoef_rows(
        corrected_fractions_per_fitted_gene_per_metacell.transpose(),
        projected_fractions_per_fitted_gene_per_metacell.transpose(),
        reproducible=reproducible,
    )

    correlated_fitted_genes_mask = correlation_per_fitted_gene >= project_min_corrected_gene_correlation
    ut.log_calc("correlated_fitted_genes_mask", correlated_fitted_genes_mask)
    if not np.any(correlated_fitted_genes_mask):
        return False

    total_corrected_fractions_per_fitted_gene = ut.sum_per(
        corrected_fractions_per_fitted_gene_per_metacell, per="column"
    )
    total_projected_fractions_per_fitted_gene = ut.sum_per(
        projected_fractions_per_fitted_gene_per_metacell, per="column"
    )

    zero_fitted_genes_mask = (total_projected_fractions_per_fitted_gene == 0) | (
        total_corrected_fractions_per_fitted_gene == 0
    )
    ut.log_calc("zero_fitted_genes_mask", zero_fitted_genes_mask)
    total_corrected_fractions_per_fitted_gene[zero_fitted_genes_mask] = 1.0
    total_projected_fractions_per_fitted_gene[zero_fitted_genes_mask] = 1.0

    current_correction_factor_per_fitted_gene = (
        total_projected_fractions_per_fitted_gene / total_corrected_fractions_per_fitted_gene
    )

    high_factor = 1 + project_min_corrected_gene_factor
    low_factor = 1.0 / high_factor
    factor_fitted_genes_mask = (current_correction_factor_per_fitted_gene <= low_factor) | (
        current_correction_factor_per_fitted_gene >= high_factor
    )
    factor_genes_mask = np.zeros(common_adata.n_vars, dtype="bool")
    factor_genes_mask[preliminary_fitted_genes_indices] = factor_fitted_genes_mask
    ut.log_calc("factor_fitted_genes_mask", factor_fitted_genes_mask)
    if not np.any(factor_fitted_genes_mask):
        return False

    corrected_fitted_genes_mask = correlated_fitted_genes_mask & factor_fitted_genes_mask
    ut.log_calc("corrected_fitted_genes_mask", corrected_fitted_genes_mask)
    if not np.any(corrected_fitted_genes_mask):
        return False

    corrected_genes_mask = np.zeros(common_adata.n_vars, dtype="bool")
    corrected_genes_mask[preliminary_fitted_genes_indices] = corrected_fitted_genes_mask
    ut.log_calc("corrected_genes_mask", corrected_genes_mask)

    correction_factor_per_corrected_gene = current_correction_factor_per_fitted_gene[corrected_fitted_genes_mask]
    correction_factor_per_gene[corrected_genes_mask] *= correction_factor_per_corrected_gene

    corrected_fractions_per_gene_per_metacell[:, corrected_genes_mask] *= correction_factor_per_corrected_gene[
        np.newaxis, :
    ]
    corrected_fractions_per_gene_per_metacell = ut.fraction_by(  # type: ignore
        ut.to_layout(corrected_fractions_per_gene_per_metacell, layout="row_major"), by="row"
    )
    ut.set_vo_data(common_qdata, "corrected_fraction", sp.csr_matrix(corrected_fractions_per_gene_per_metacell))

    return True


@ut.logged()
@ut.timed_call()
def _filter_range_genes(
    *,
    common_qdata: AnnData,
    project_fold_regularization: float,
    project_ignore_range_quantile: float,
    project_ignore_range_min_overlap_fraction: float,
    preliminary_fitted_genes_mask: ut.NumpyVector,
) -> bool:
    corrected_fractions_per_gene_per_metacell = ut.to_numpy_matrix(
        ut.get_vo_proper(common_qdata, "corrected_fraction", layout="column_major")
    )
    projected_fractions_per_gene_per_metacell = ut.to_numpy_matrix(
        ut.get_vo_proper(common_qdata, "projected_fraction", layout="column_major")
    )

    preliminary_fitted_genes_indices = np.where(preliminary_fitted_genes_mask)[0]

    corrected_fractions_per_fitted_gene_per_metacell = corrected_fractions_per_gene_per_metacell[
        :, preliminary_fitted_genes_indices
    ]
    projected_fractions_per_fitted_gene_per_metacell = projected_fractions_per_gene_per_metacell[
        :, preliminary_fitted_genes_indices
    ]

    corrected_log_fractions_per_fitted_gene_per_metacell = (
        corrected_fractions_per_fitted_gene_per_metacell + project_fold_regularization
    )
    projected_log_fractions_per_fitted_gene_per_metacell = (
        projected_fractions_per_fitted_gene_per_metacell + project_fold_regularization
    )

    np.log2(
        corrected_log_fractions_per_fitted_gene_per_metacell, out=corrected_log_fractions_per_fitted_gene_per_metacell
    )
    np.log2(
        projected_log_fractions_per_fitted_gene_per_metacell, out=projected_log_fractions_per_fitted_gene_per_metacell
    )

    low_corrected_log_fractions_per_fitted_gene = ut.quantile_per(
        corrected_log_fractions_per_fitted_gene_per_metacell, project_ignore_range_quantile, per="column"
    )
    low_projected_log_fractions_per_fitted_gene = ut.quantile_per(
        projected_log_fractions_per_fitted_gene_per_metacell, project_ignore_range_quantile, per="column"
    )

    high_corrected_log_fractions_per_fitted_gene = ut.quantile_per(
        corrected_log_fractions_per_fitted_gene_per_metacell, 1.0 - project_ignore_range_quantile, per="column"
    )
    high_projected_log_fractions_per_fitted_gene = ut.quantile_per(
        projected_log_fractions_per_fitted_gene_per_metacell, 1.0 - project_ignore_range_quantile, per="column"
    )

    low_common_log_fractions_per_fitted_gene = np.maximum(
        low_corrected_log_fractions_per_fitted_gene, low_projected_log_fractions_per_fitted_gene
    )
    high_common_log_fractions_per_fitted_gene = np.minimum(
        high_corrected_log_fractions_per_fitted_gene, high_projected_log_fractions_per_fitted_gene
    )

    corrected_range_log_fractions_per_fitted_gene = (
        high_corrected_log_fractions_per_fitted_gene - low_corrected_log_fractions_per_fitted_gene
    )
    common_range_log_fractions_per_fitted_gene = (
        high_common_log_fractions_per_fitted_gene - low_common_log_fractions_per_fitted_gene
    )

    corrected_range_log_fractions_per_fitted_gene[corrected_range_log_fractions_per_fitted_gene == 0] = 1
    overlap_per_fitted_gene = common_range_log_fractions_per_fitted_gene / corrected_range_log_fractions_per_fitted_gene

    ignore_fitted_genes_mask = overlap_per_fitted_gene < project_ignore_range_min_overlap_fraction
    ut.log_calc("ignore_fitted_genes_mask", ignore_fitted_genes_mask)
    if not np.any(ignore_fitted_genes_mask):
        return False

    ignore_gene_indices = preliminary_fitted_genes_indices[ignore_fitted_genes_mask]
    preliminary_fitted_genes_mask[ignore_gene_indices] = False
    ut.log_calc("preliminary_fitted_genes_mask", preliminary_fitted_genes_mask)

    return True


@ut.logged()
@ut.timed_call()
def _compute_per_type_projection(
    *,
    common_adata: AnnData,
    common_qdata: AnnData,
    type_names: List[str],
    preliminary_fitted_genes_mask: ut.NumpyVector,
    misfit_min_metacells_fraction: float,
    project_log_data: bool,
    project_fold_regularization: float,
    project_candidates_count: int,
    project_min_significant_gene_umis: float,
    project_min_usage_weight: float,
    project_max_consistency_fold_factor: float,
    project_max_projection_fold_factor: float,
    project_max_projection_noisy_fold_factor: float,
    project_max_misfit_genes: int,
    min_fitted_query_marker_genes: float,
    min_essential_genes_of_type: Dict[Tuple[str, str], Optional[int]],
    atlas_type_property_name: str,
    top_level_parallel: bool,
    reproducible: bool,
) -> ut.NumpyMatrix:
    old_types_per_metacell: List[Set[str]] = []
    for _metacell_index in range(common_qdata.n_obs):
        old_types_per_metacell.append(set())

    fitted_genes_mask_per_type = _initial_fitted_genes_mask_per_type(
        common_qdata=common_qdata, type_names=type_names, preliminary_fitted_genes_mask=preliminary_fitted_genes_mask
    )

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
        query_type_names = list(np.unique(type_of_query_metacells))

        misfit_per_gene_per_metacell[:, :] = False
        projected_fold_per_gene_per_metacell[:, :] = 0.0

        @ut.timed_call("single_type_projection")
        def _single_type_projection(
            type_index: int,
        ) -> Dict[str, Any]:
            return _compute_single_type_projection(
                type_name=query_type_names[type_index],
                common_adata=common_adata,
                common_qdata=common_qdata,
                fitted_genes_mask_per_type=fitted_genes_mask_per_type,
                misfit_min_metacells_fraction=misfit_min_metacells_fraction,
                project_log_data=project_log_data,
                project_fold_regularization=project_fold_regularization,
                project_candidates_count=project_candidates_count,
                project_min_significant_gene_umis=project_min_significant_gene_umis,
                project_min_usage_weight=project_min_usage_weight,
                project_max_consistency_fold_factor=project_max_consistency_fold_factor,
                project_max_projection_fold_factor=project_max_projection_fold_factor,
                project_max_projection_noisy_fold_factor=project_max_projection_noisy_fold_factor,
                project_max_misfit_genes=project_max_misfit_genes,
                min_fitted_query_marker_genes=min_fitted_query_marker_genes,
                min_essential_genes_of_type=min_essential_genes_of_type,
                top_level_parallel=top_level_parallel,
                reproducible=reproducible,
            )

        @ut.logged()
        def _collect_single_type_result(
            type_index: int,
            *,
            query_metacell_indices_of_type: ut.NumpyVector,
            fitted_genes_indices_of_type: ut.NumpyVector,
            similar_per_metacell_of_type: ut.NumpyVector,
            misfit_per_gene_per_metacell_of_type: ut.ProperMatrix,
            projected_correlation_per_metacell_of_type: ut.NumpyVector,
            projected_fold_per_gene_per_metacell_of_type: ut.ProperMatrix,
            weights_per_atlas_per_query_metacell_of_type: ut.ProperMatrix,
        ) -> None:
            fitted_genes_mask_per_type[query_type_names[type_index]][:] = False
            fitted_genes_mask_per_type[query_type_names[type_index]][fitted_genes_indices_of_type] = True
            similar_per_metacell[query_metacell_indices_of_type] = similar_per_metacell_of_type
            projected_correlation_per_metacell[
                query_metacell_indices_of_type
            ] = projected_correlation_per_metacell_of_type
            misfit_per_gene_per_metacell[query_metacell_indices_of_type, :] = ut.to_numpy_matrix(
                misfit_per_gene_per_metacell_of_type
            )
            projected_fold_per_gene_per_metacell[query_metacell_indices_of_type, :] = ut.to_numpy_matrix(
                projected_fold_per_gene_per_metacell_of_type
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
            ut.log_calc("types last repeat", repeat)
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

    tl.compute_projected_fractions(
        adata=common_adata,
        qdata=common_qdata,
        log_data=project_log_data,
        fold_regularization=project_fold_regularization,
        weights=weights_per_atlas_per_query_metacell,
    )

    return weights_per_atlas_per_query_metacell


def _initial_fitted_genes_mask_per_type(
    common_qdata: AnnData,
    type_names: List[str],
    preliminary_fitted_genes_mask: ut.NumpyVector,
) -> Dict[str, ut.NumpyVector]:
    fitted_genes_mask_per_type: Dict[str, ut.NumpyVector] = {}
    for type_name in type_names:
        fitted_genes_mask_of_type = preliminary_fitted_genes_mask.copy()
        property_name = f"ignored_gene_of_{type_name}"
        if ut.has_data(common_qdata, property_name):
            ignored_gene_mask_of_type = ut.get_v_numpy(common_qdata, property_name)
            fitted_genes_mask_of_type &= ~ignored_gene_mask_of_type
        fitted_genes_mask_per_type[type_name] = fitted_genes_mask_of_type
    return fitted_genes_mask_per_type


@ut.logged()
@ut.timed_call()
def _compute_single_type_projection(
    *,
    type_name: str,
    common_adata: AnnData,
    common_qdata: AnnData,
    fitted_genes_mask_per_type: Dict[str, ut.NumpyVector],
    misfit_min_metacells_fraction: float,
    project_log_data: bool,
    project_fold_regularization: float,
    project_candidates_count: int,
    project_min_significant_gene_umis: float,
    project_min_usage_weight: float,
    project_max_consistency_fold_factor: float,
    project_max_projection_fold_factor: float,
    project_max_projection_noisy_fold_factor: float,
    project_max_misfit_genes: int,
    min_fitted_query_marker_genes: float,
    min_essential_genes_of_type: Dict[Tuple[str, str], Optional[int]],
    top_level_parallel: bool,
    reproducible: bool,
) -> Dict[str, Any]:
    projected_type_per_metacell = ut.get_o_numpy(common_qdata, "projected_type")
    query_metacell_mask_of_type = projected_type_per_metacell == type_name
    ut.log_calc("query_metacell_mask_of_type", query_metacell_mask_of_type)
    assert np.any(query_metacell_mask_of_type)
    type_common_qdata = ut.slice(common_qdata, name=f".{type_name}", obs=query_metacell_mask_of_type, top_level=False)
    corrected_fractions = ut.get_vo_proper(type_common_qdata, "corrected_fraction")

    fitted_genes_mask_of_type = fitted_genes_mask_per_type[type_name]
    assert np.any(fitted_genes_mask_of_type)

    repeat = 0
    while True:
        repeat += 1
        ut.log_calc(f"{type_name} misfit repeat", repeat)

        type_fitted_adata = ut.slice(
            common_adata, name=f".{type_name}.fitted", vars=fitted_genes_mask_of_type, top_level=False
        )

        type_fitted_qdata = ut.slice(
            type_common_qdata,
            name=".fitted",
            vars=fitted_genes_mask_of_type,
            track_var="common_gene_index_of_qdata",
            top_level=False,
        )

        type_weights = tl.compute_projection_weights(
            adata=type_fitted_adata,
            qdata=type_fitted_qdata,
            log_data=project_log_data,
            fold_regularization=project_fold_regularization,
            min_significant_gene_umis=project_min_significant_gene_umis,
            max_consistency_fold_factor=project_max_consistency_fold_factor,
            candidates_count=project_candidates_count,
            min_usage_weight=project_min_usage_weight,
            reproducible=reproducible,
        )

        tl.compute_projected_fractions(
            adata=common_adata,
            qdata=type_common_qdata,
            log_data=project_log_data,
            fold_regularization=project_fold_regularization,
            weights=type_weights,
        )
        projected_fractions = ut.get_vo_proper(type_common_qdata, "projected_fraction")

        tl.compute_projected_folds(
            type_common_qdata,
            fold_regularization=project_fold_regularization,
            min_significant_gene_umis=project_min_significant_gene_umis,
        )
        projected_fold_per_gene_per_metacell_of_type = ut.get_vo_proper(type_common_qdata, "projected_fold")

        if not _detect_type_misfit_genes(
            type_common_qdata=type_common_qdata,
            projected_fold_per_gene_per_metacell_of_type=projected_fold_per_gene_per_metacell_of_type,
            max_projection_fold_factor=project_max_projection_fold_factor,
            max_projection_noisy_fold_factor=project_max_projection_noisy_fold_factor,
            misfit_min_metacells_fraction=misfit_min_metacells_fraction,
            fitted_genes_mask_of_type=fitted_genes_mask_of_type,
        ):
            ut.log_calc(f"{type_name} misfit last repeat", repeat)
            break

    tl.compute_similar_query_metacells(
        type_common_qdata,
        max_projection_fold_factor=project_max_projection_fold_factor,
        max_projection_noisy_fold_factor=project_max_projection_noisy_fold_factor,
        max_misfit_genes=project_max_misfit_genes,
        min_fitted_query_marker_genes=min_fitted_query_marker_genes,
        essential_genes_property=f"essential_gene_of_{type_name}",
        min_essential_genes=min_essential_genes_of_type[(type_name, type_name)],
        fitted_genes_mask=fitted_genes_mask_of_type,
    )
    similar_per_metacell_of_type = ut.get_o_numpy(type_common_qdata, "similar")
    misfit_per_gene_per_metacell_of_type = ut.get_vo_proper(type_common_qdata, "misfit")

    fitted_corrected_fractions = corrected_fractions[:, fitted_genes_mask_of_type]
    fitted_projected_fractions = projected_fractions[:, fitted_genes_mask_of_type]
    projected_correlation_per_metacell_of_type = ut.pairs_corrcoef_rows(
        ut.to_layout(ut.to_numpy_matrix(fitted_corrected_fractions), layout="row_major"),
        ut.to_layout(ut.to_numpy_matrix(fitted_projected_fractions), layout="row_major"),
        reproducible=reproducible,
    )

    if top_level_parallel:
        if not isinstance(misfit_per_gene_per_metacell_of_type, sp.csr_matrix):
            misfit_per_gene_per_metacell_of_type = sp.csr_matrix(misfit_per_gene_per_metacell_of_type)
        if not isinstance(projected_fold_per_gene_per_metacell_of_type, sp.csr_matrix):
            projected_fold_per_gene_per_metacell_of_type = sp.csr_matrix(projected_fold_per_gene_per_metacell_of_type)
        if not isinstance(type_weights, sp.csr_matrix):
            type_weights = sp.csr_matrix(type_weights)

    ut.log_return("query_metacell_mask_of_type", query_metacell_mask_of_type)
    ut.log_return("fitted_genes_mask_of_type", fitted_genes_mask_of_type)
    ut.log_return("similar_per_metacell_of_type", similar_per_metacell_of_type)
    ut.log_return("misfit_per_gene_per_metacell_of_type", misfit_per_gene_per_metacell_of_type)
    ut.log_return("projected_correlation_per_metacell_of_type", projected_correlation_per_metacell_of_type)
    ut.log_return("projected_fold_per_gene_per_metacell_of_type", projected_fold_per_gene_per_metacell_of_type)
    ut.log_return("weights", type_weights)

    return {
        "query_metacell_indices_of_type": np.where(query_metacell_mask_of_type)[0],
        "fitted_genes_indices_of_type": np.where(fitted_genes_mask_of_type)[0],
        "similar_per_metacell_of_type": similar_per_metacell_of_type,
        "misfit_per_gene_per_metacell_of_type": misfit_per_gene_per_metacell_of_type,
        "projected_correlation_per_metacell_of_type": projected_correlation_per_metacell_of_type,
        "projected_fold_per_gene_per_metacell_of_type": projected_fold_per_gene_per_metacell_of_type,
        "weights_per_atlas_per_query_metacell_of_type": type_weights,
    }


def _detect_type_misfit_genes(
    *,
    type_common_qdata: AnnData,
    projected_fold_per_gene_per_metacell_of_type: ut.ProperMatrix,
    max_projection_fold_factor: float,
    max_projection_noisy_fold_factor: float,
    misfit_min_metacells_fraction: float,
    fitted_genes_mask_of_type: ut.NumpyVector,
) -> bool:
    assert max_projection_fold_factor >= 0
    assert max_projection_noisy_fold_factor >= 0
    assert 0 <= misfit_min_metacells_fraction <= 1

    if ut.has_data(type_common_qdata, "projected_noisy_gene"):
        max_projection_fold_factor_per_gene = np.full(
            type_common_qdata.n_vars, max_projection_fold_factor, dtype="float32"
        )
        noisy_per_gene = ut.get_v_numpy(type_common_qdata, "projected_noisy_gene")
        max_projection_fold_factor_per_gene[noisy_per_gene] += max_projection_noisy_fold_factor
        high_projection_fold_per_gene_per_metacell_of_type = (
            projected_fold_per_gene_per_metacell_of_type > max_projection_fold_factor_per_gene[np.newaxis, :]
        )
    else:
        high_projection_fold_per_gene_per_metacell_of_type = (
            ut.to_numpy_matrix(projected_fold_per_gene_per_metacell_of_type) > max_projection_fold_factor
        )
    ut.log_calc(
        "high_projection_fold_per_gene_per_metacell_of_type", high_projection_fold_per_gene_per_metacell_of_type
    )

    high_projection_metacells_per_gene = ut.sum_per(
        ut.to_layout(high_projection_fold_per_gene_per_metacell_of_type, layout="column_major"), per="column"
    )
    ut.log_calc(
        "high_projection_metacells_per_gene", high_projection_metacells_per_gene, formatter=ut.sizes_description
    )

    min_high_projection_metacells = type_common_qdata.n_obs * misfit_min_metacells_fraction
    ut.log_calc("min_high_projection_metacells", min_high_projection_metacells)

    high_projection_genes_mask = high_projection_metacells_per_gene >= min_high_projection_metacells
    type_misfit_genes_mask = fitted_genes_mask_of_type & high_projection_genes_mask
    ut.log_calc("type_misfit_genes_mask", type_misfit_genes_mask)
    if not np.any(type_misfit_genes_mask):
        return False

    fitted_genes_mask_of_type[type_misfit_genes_mask] = False
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
    common_adata: AnnData,
    common_qdata: AnnData,
    weights_per_atlas_per_query_metacell: ut.NumpyMatrix,
    project_log_data: bool,
    project_fold_regularization: float,
    project_candidates_count: int,
    project_min_candidates_fraction: float,
    project_min_significant_gene_umis: float,
    project_min_usage_weight: float,
    project_max_consistency_fold_factor: float,
    project_max_projection_fold_factor: float,
    project_max_projection_noisy_fold_factor: float,
    project_max_misfit_genes: int,
    min_fitted_query_marker_genes: float,
    min_essential_genes_of_type: Dict[Tuple[str, str], Optional[int]],
    atlas_type_property_name: str,
    top_level_parallel: bool,
    reproducible: bool,
) -> None:
    secondary_type = [""] * common_qdata.n_obs
    dissimilar_metacells_mask = ~ut.get_o_numpy(common_qdata, "similar")
    if not np.any(dissimilar_metacells_mask):
        ut.set_o_data(common_qdata, "projected_secondary_type", np.array(secondary_type))
        return
    dissimilar_metacell_indices = np.where(dissimilar_metacells_mask)[0]

    @ut.timed_call("single_metacell_residuals")
    def _single_metacell_residuals(
        dissimilar_metacell_position: int,
    ) -> Optional[Dict[str, Any]]:
        return _compute_single_metacell_residuals(
            dissimilar_metacell_index=dissimilar_metacell_indices[dissimilar_metacell_position],
            common_adata=common_adata,
            common_qdata=common_qdata,
            project_log_data=project_log_data,
            project_fold_regularization=project_fold_regularization,
            project_candidates_count=project_candidates_count,
            project_min_candidates_fraction=project_min_candidates_fraction,
            project_min_significant_gene_umis=project_min_significant_gene_umis,
            project_min_usage_weight=project_min_usage_weight,
            project_max_consistency_fold_factor=project_max_consistency_fold_factor,
            project_max_projection_fold_factor=project_max_projection_fold_factor,
            project_max_projection_noisy_fold_factor=project_max_projection_noisy_fold_factor,
            project_max_misfit_genes=project_max_misfit_genes,
            min_fitted_query_marker_genes=min_fitted_query_marker_genes,
            min_essential_genes_of_type=min_essential_genes_of_type,
            atlas_type_property_name=atlas_type_property_name,
            reproducible=reproducible,
        )

    if top_level_parallel:
        results = ut.parallel_map(_single_metacell_residuals, len(dissimilar_metacell_indices))
    else:
        results = [
            _single_metacell_residuals(dissimilar_metacell_position)
            for dissimilar_metacell_position in range(len(dissimilar_metacell_indices))
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
        projected_fold_per_gene: ut.NumpyVector,
        weights: ut.ProperMatrix,
    ) -> None:
        similar_per_metacell[dissimilar_metacell_index] = True
        primary_type_per_metacell[dissimilar_metacell_index] = primary_type
        secondary_type_per_metacell[dissimilar_metacell_index] = secondary_type
        projected_correlation_per_metacell[dissimilar_metacell_index] = projected_correlation

        fitted_per_gene_per_metacell[dissimilar_metacell_index, :] = fitted_genes_mask
        misfit_per_gene_per_metacell[dissimilar_metacell_index, :] = misfit_genes_mask
        projected_fold_per_gene_per_metacell[dissimilar_metacell_index, :] = projected_fold_per_gene
        weights_per_atlas_per_query_metacell[dissimilar_metacell_index, :] = weights

    for dissimilar_metacell_index, result in zip(dissimilar_metacell_indices, results):
        if result is not None:
            _collect_metacell_residuals(dissimilar_metacell_index, **result)

    ut.set_o_data(common_qdata, "similar", similar_per_metacell)
    ut.set_o_data(common_qdata, "projected_type", primary_type_per_metacell)
    ut.set_o_data(common_qdata, "projected_secondary_type", np.array(secondary_type_per_metacell))
    ut.set_o_data(common_qdata, "projected_correlation", projected_correlation_per_metacell)

    ut.set_vo_data(common_qdata, "fitted", fitted_per_gene_per_metacell)
    ut.set_vo_data(common_qdata, "misfit", misfit_per_gene_per_metacell)
    ut.set_vo_data(common_qdata, "projected_fold", projected_fold_per_gene_per_metacell)


@ut.timed_call()
@ut.logged()
def _compute_single_metacell_residuals(  # pylint: disable=too-many-statements
    *,
    dissimilar_metacell_index: int,
    common_adata: AnnData,
    common_qdata: AnnData,
    project_log_data: bool,
    project_fold_regularization: float,
    project_candidates_count: int,
    project_min_candidates_fraction: float,
    project_min_significant_gene_umis: float,
    project_min_usage_weight: float,
    project_max_consistency_fold_factor: float,
    project_max_projection_fold_factor: float,
    project_max_projection_noisy_fold_factor: float,
    project_max_misfit_genes: int,
    min_fitted_query_marker_genes: float,
    min_essential_genes_of_type: Dict[Tuple[str, str], Optional[int]],
    atlas_type_property_name: str,
    reproducible: bool,
) -> Optional[Dict[str, Any]]:
    dissimilar_qdata = ut.slice(
        common_qdata, name=f".dissimilar.{dissimilar_metacell_index}", obs=[dissimilar_metacell_index], top_level=False
    )
    corrected_fractions = ut.get_vo_proper(dissimilar_qdata, "corrected_fraction")

    primary_type = ut.get_o_numpy(dissimilar_qdata, "projected_type")[0]
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

        fitted_adata = ut.slice(
            common_adata,
            name=f".dissimilar.{dissimilar_metacell_index}.fitted",
            vars=fitted_genes_mask,
            top_level=False,
        )

        metacell_fitted_qdata = ut.slice(dissimilar_qdata, name=".fitted", vars=fitted_genes_mask, top_level=False)

        second_anchor_indices: List[int] = []
        weights = tl.compute_projection_weights(
            adata=fitted_adata,
            qdata=metacell_fitted_qdata,
            log_data=project_log_data,
            fold_regularization=project_fold_regularization,
            min_significant_gene_umis=project_min_significant_gene_umis,
            max_consistency_fold_factor=project_max_consistency_fold_factor,
            candidates_count=project_candidates_count,
            min_candidates_fraction=project_min_candidates_fraction,
            min_usage_weight=project_min_usage_weight,
            reproducible=reproducible,
            second_anchor_indices=second_anchor_indices,
        )
        tl.compute_projected_fractions(
            adata=common_adata,
            qdata=dissimilar_qdata,
            log_data=project_log_data,
            fold_regularization=project_fold_regularization,
            weights=weights,
        )
        projected_fractions = ut.get_vo_proper(dissimilar_qdata, "projected_fraction")

        first_anchor_weights = weights.copy()
        first_anchor_weights[:, second_anchor_indices] = 0.0

        if len(second_anchor_indices) == 0:
            new_secondary_type = ""
        else:
            second_anchor_weights = weights - first_anchor_weights  # type: ignore

            tl.convey_atlas_to_query(
                adata=common_adata,
                qdata=dissimilar_qdata,
                weights=second_anchor_weights,
                property_name=atlas_type_property_name,
                to_property_name="projected_secondary_type",
            )
            new_secondary_type = ut.get_o_numpy(dissimilar_qdata, "projected_secondary_type")[0]

        if np.sum(first_anchor_weights.data) == 0:
            new_primary_type = new_secondary_type
            new_secondary_type = ""
        else:
            tl.convey_atlas_to_query(
                adata=common_adata,
                qdata=dissimilar_qdata,
                weights=first_anchor_weights,
                property_name=atlas_type_property_name,
                to_property_name="projected_type",
            )
            new_primary_type = ut.get_o_numpy(dissimilar_qdata, "projected_type")[0]

        if repeat > 2 or (new_primary_type == primary_type and new_secondary_type == secondary_type):
            ut.log_calc("residuals last repeat", repeat)
            break

        primary_type = new_primary_type
        secondary_type = new_secondary_type

    projected_correlation = ut.pairs_corrcoef_rows(
        ut.to_numpy_matrix(corrected_fractions), ut.to_numpy_matrix(projected_fractions), reproducible=reproducible
    )[0]

    tl.compute_projected_folds(
        dissimilar_qdata,
        fold_regularization=project_fold_regularization,
        min_significant_gene_umis=project_min_significant_gene_umis,
    )
    projected_fold_per_gene = ut.to_numpy_vector(ut.get_vo_proper(dissimilar_qdata, "projected_fold"))

    essential_genes_properties = [f"essential_gene_of_{primary_type}"]
    if secondary_type == "":
        min_essential_genes = min_essential_genes_of_type[(primary_type, primary_type)]
    else:
        min_essential_genes = min_essential_genes_of_type[(primary_type, secondary_type)]
        essential_genes_properties.append(f"essential_gene_of_{secondary_type}")

    tl.compute_similar_query_metacells(
        dissimilar_qdata,
        max_projection_fold_factor=project_max_projection_fold_factor,
        max_projection_noisy_fold_factor=project_max_projection_noisy_fold_factor,
        max_misfit_genes=project_max_misfit_genes,
        essential_genes_property=essential_genes_properties,
        min_essential_genes=min_essential_genes,
        min_fitted_query_marker_genes=min_fitted_query_marker_genes,
        fitted_genes_mask=fitted_genes_mask,
    )

    similar = ut.get_o_numpy(dissimilar_qdata, "similar")[0]
    ut.log_return("similar", False)
    if not similar:
        return None

    misfit_genes_mask = ut.to_numpy_vector(ut.get_vo_proper(dissimilar_qdata, "misfit")[0, :])

    ut.log_return("primary_type", primary_type)
    ut.log_return("secondary_type", secondary_type)
    ut.log_return("fitted_genes_mask", fitted_genes_mask)
    ut.log_return("misfit_genes_mask", misfit_genes_mask)
    ut.log_return("projected_correlation", projected_correlation)
    ut.log_return("projected_fold_per_gene", projected_fold_per_gene)

    return {
        "primary_type": primary_type,
        "secondary_type": secondary_type,
        "fitted_genes_mask": fitted_genes_mask,
        "misfit_genes_mask": misfit_genes_mask,
        "projected_correlation": projected_correlation,
        "projected_fold_per_gene": projected_fold_per_gene,
        "weights": ut.to_numpy_vector(weights),
    }


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
    fold_regularization: float = pr.outliers_fold_regularization,
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

    1. Invoke :py:func:`metacells.tools.quality.compute_outliers_matches` using the ``fold_regularization``
       (default: {fold_regularization}) and ``reproducible``.

    2. Invoke :py:func:`metacells.tools.quality.compute_outliers_fold_factors` using the ``fold_regularization``
       (default: {fold_regularization}).
    """
    if list(odata.var_names) != list(adata.var_names):
        atlas_genes_list = list(adata.var_names)
        query_genes_list = list(odata.var_names)
        common_genes_list = list(sorted(set(atlas_genes_list) & set(query_genes_list)))
        atlas_common_gene_indices = np.array([atlas_genes_list.index(gene) for gene in common_genes_list])
        query_common_gene_indices = np.array([query_genes_list.index(gene) for gene in common_genes_list])
        common_adata = ut.slice(adata, name=".common", vars=atlas_common_gene_indices, top_level=False)
        common_odata = ut.slice(
            odata,
            name=".common",
            vars=query_common_gene_indices,
            top_level=False,
        )
    else:
        common_adata = adata
        common_odata = odata

    tl.compute_outliers_matches(
        what,
        adata=common_odata,
        gdata=common_adata,
        most_similar="atlas_most_similar",
        value_regularization=fold_regularization,
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


def write_projection_weights(path: str, adata: AnnData, qdata: AnnData, weights: ut.CompressedMatrix) -> None:
    """
    Write into the ``path`` the ``weights`` computed for the projection of the query ``qdata`` on the atlas ``adata``.

    Since the weights are (very) sparse, we just write them as a CSV file. This is also what ``MCView`` expect.
    """
    with open(path, "w", encoding="utf8") as file:
        file.write("query,atlas,weight\n")
        for query_index, atlas_index in zip(*weights.nonzero()):
            weight = weights[query_index, atlas_index]
            file.write(f"{qdata.obs_names[query_index]},{adata.obs_names[atlas_index]},{weight}\n")
