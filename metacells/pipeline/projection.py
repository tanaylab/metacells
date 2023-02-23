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

    2. Invoke :py:func:`metacells.tools.project.compute_projection`
       to project each query metacell onto the atlas, using the ``project_fold_normalization`` (default:
       {project_fold_normalization}), ``project_min_significant_gene_umis`` (default:
       {project_min_significant_gene_umis}), ``project_max_consistency_fold_factor`` (default:
       {project_max_consistency_fold_factor}), ``project_candidates_count`` (default: {project_candidates_count}),
       ``project_min_usage_weight`` (default: {project_min_usage_weight}), and ``reproducible``.

    3. If ``project_corrections``: Correlate the expression levels of each gene between the query and projection. If
       this is at least ``project_min_corrected_gene_correlation`` (default: {project_min_corrected_gene_correlation}),
       compute the ratio between the mean expression of the gene in the projection and the query. If this is at most
       1/(1+``project_min_corrected_gene_factor``) or at least (1+``project_min_corrected_gene_factor``) (default:
       {project_min_corrected_gene_factor}), then multiply the gene's value by this factor so its level would match the
       atlas. As usual, ignore genes which do not have at least ``project_min_significant_gene_umis``. If any genes were
       ignored or corrected, then repeat steps 1-4.

    4. Invoke :py:func:`metacells.tools.project.convey_atlas_to_query` to assign a projected type to each of the
       query metacells based on the ``atlas_type_property_name`` (default: {atlas_type_property_name}).

    Then, for each type of query metacells:

    If ``top_level_parallel`` (default: ``top_level_parallel``), do this in parallel. This seems to work better than
    doing this serially and using parallelism within each type. However this is still inefficient compared to using both
    types of parallelism at once, which the code currently can't do without non-trivial coding (this would have been
    trivial in Julia...).

    5. Further reduce the mask of type-specific fitted genes by ignoring any genes in ``ignored_gene_of_<type>``, if
       this annotation exists in the query. Normalize the sum of the fitted gene fractions to 1 in each metacell.

    6. Invoke :py:func:`metacells.tools.project.compute_projection` to project each query metacell of the type
       onto the atlas. Note that even though we only look at query metacells (tentatively) assigned the type, their
       projection on the atlas may use metacells of any type.

    7. Invoke :py:func:`metacells.tools.quality.compute_projected_folds` to compute the significant fold factors
       between the query and its projection.

    8. Identify type-specific misfit genes whose fold factor is above ``project_max_projection_fold_factor``. If
       ``consider_atlas_noisy_genes`` and/or ``consider_query_noisy_genes``, then any gene listed in either is allowed
       an additional ``project_max_projection_noisy_fold_factor``. If any gene has such a high fold factor in at least
       ``misfit_min_metacells_fraction``, remove it from the fitted genes mask and repeat steps 5-8.

    9. Invoke :py:func:`metacells.tools.quality.compute_similar_query_metacells` to verify which query metacells
        ended up being sufficiently similar to their projection, using ``project_max_consistency_fold_factor`` (default:
        {project_max_consistency_fold_factor}), ``project_max_projection_noisy_fold_factor`` (default:
        {project_max_projection_noisy_fold_factor}), ``project_max_misfit_genes`` (default: {project_max_misfit_genes}),
        and if needed, the the ``essential_gene_of_<type>``. Also compute the correlation between the corrected
        and projected gene fractions for each metacell.

    And then:

    10. Invoke :py:func:`metacells.tools.project.convey_atlas_to_query` to assign an updated projected type to each of
        the query metacells based on the ``atlas_type_property_name`` (default: {atlas_type_property_name}). If this
        changed the type assigned to any query metacell, repeat steps 5-10 (but do these steps no more than 3 times).

    For each query metacell that ended up being dissimilar, try to project them as a combination of two atlas regions:

    11. Reduce the list of fitted genes for the metacell based on the ``ignored_gene_of_<type>`` for both the primary
        (initially from the above) query metacell type and the secondary query metacell type (initially empty).
        Normalize the sum of the gene fractions in the metacell to 1.

    12. Invoke :py:func:`metacells.tools.project.compute_projection` just for this metacell, allowing the projection to
        use a secondary location in the atlas based on the residuals of the atlas metacells relative to the primary
        query projection.

    13. Invoke :py:func:`metacells.tools.project.convey_atlas_to_query` twice, once for the weights of the primary
        location and once for the weights of the secondary location, to obtain a primary and secondary type for
        the query metacell. If these have changed, repeat steps 12-14 (but do these steps no more than 3 times; note
        will will always do them twice as the 1st run will generate some non-empty secondary type).

    14. Invoke :py:func:`metacells.tools.quality.compute_projected_folds` and
        :py:func:`metacells.tools.quality.compute_similar_query_metacells` to update the projection and evaluation of
        the query metacell. If it is now similar, then use the results for the metacell; otherwise, keep the original
        results as they were at the end of step 9.
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

    if ut.has_data(common_qdata, "ignored_gene"):
        initial_fitted_genes_mask &= ~ut.get_v_numpy(common_qdata, "ignored_gene")

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

    fitted_genes_mask_per_type = _initial_fitted_genes_mask_per_type(
        common_qdata=common_qdata, type_names=type_names, initial_fitted_genes_mask=initial_fitted_genes_mask
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


def _initial_fitted_genes_mask_per_type(
    common_qdata: AnnData,
    type_names: List[str],
    initial_fitted_genes_mask: ut.NumpyVector,
) -> Dict[str, ut.NumpyVector]:
    fitted_genes_mask_per_type: Dict[str, ut.NumpyVector] = {}
    for type_name in type_names:
        fitted_genes_mask_of_type = initial_fitted_genes_mask.copy()
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
