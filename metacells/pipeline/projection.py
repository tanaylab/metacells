"""
Projection
----------
"""

from re import Pattern
from typing import Collection
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import scipy.sparse as sp  # type: ignore
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    "direct_projection_pipeline",
    "typed_projection_pipeline",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def direct_projection_pipeline(  # pylint: disable=too-many-statements
    what: str = "__x__",
    *,
    adata: AnnData,
    qdata: AnnData,
    ignored_gene_names: Optional[Collection[str]] = None,
    ignored_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    ignore_atlas_insignificant_genes: bool = pr.ignore_atlas_insignificant_genes,
    ignore_atlas_forbidden_genes: bool = pr.ignore_atlas_forbidden_genes,
    ignore_query_forbidden_genes: bool = pr.ignore_query_forbidden_genes,
    systematic_low_gene_quantile: float = pr.systematic_low_gene_quantile,
    systematic_high_gene_quantile: float = pr.systematic_high_gene_quantile,
    biased_min_metacells_fraction: float = pr.biased_min_metacells_fraction,
    project_fold_normalization: float = pr.project_fold_normalization,
    project_candidates_count: int = pr.project_candidates_count,
    project_min_significant_gene_value: float = pr.project_min_significant_gene_value,
    project_min_usage_weight: float = pr.project_min_usage_weight,
    project_min_consistency_weight: float = pr.project_min_consistency_weight,
    project_min_total_consistency_weight: float = pr.project_min_total_consistency_weight,
    project_max_consistency_fold_factor: float = pr.project_max_consistency_fold_factor,
    project_max_inconsistent_genes: int = pr.project_max_inconsistent_genes,
    project_max_projection_fold_factor: float = pr.project_max_projection_fold_factor,
    min_entry_project_fold_factor: float = pr.min_entry_project_fold_factor,
    min_entry_project_consistency_fold_factor: float = pr.min_entry_project_consistency_fold_factor,
    project_abs_folds: bool = pr.project_abs_folds,
) -> ut.CompressedMatrix:
    """
    Complete pipeline for projecting query metacells onto an atlas of metacells for the ``what`` (default: {what}) data.

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

        ``biased_gene``
            A boolean mask indicating whether the gene has a strong bias in the query compared to the atlas.

        ``manually_ignored_gene``
            A boolean mask indicating whether the gene was manually ignored (by its name).

        ``atlas_forbidden_gene`` (if ``ignore_atlas_forbidden_genes`` and the atlas has a ``forbidden_gene`` mask)
            A boolean mask indicating whether the gene was forbidden from being a feature in the atlas (and hence
            ignored).

        ``atlas_significant_gene`` (if ``ignore_atlas_insignificant_genes`` and the atlas has ``significant_gene`` mask)
            A boolean mask indicating whether the gene is considered significant in the atlas.

        ``ignored_gene``
            A boolean mask indicating whether the gene was ignored by the projection (for any reason).

    Observation (Cell) Annotations
        ``similar``
            A boolean mask indicating whether the query metacell is similar to its projection onto the atlas. If
            ``False`` the metacells is said to be "dissimilar", which may indicate the query contains cell states that
            do not appear in the atlas.

    Observation-Variable (Cell-Gene) Annotations
        ``projected``
            A matrix of UMIs where the sum of UMIs for each query metacell is the same as the sum of ``what`` UMIs,
            describing the "projected" image of the query metacell onto the atlas.

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
       ``ignore_atlas_insignificant_genes``), ignore genes the atlas did not mark as significant. If
       ``ignore_atlas_forbidden_genes`` (default: {ignore_atlas_forbidden_genes}), also ignore the ``forbidden_gene`` of
       the atlas (and store them as a ``atlas_forbidden_gene`` mask). If ``ignore_query_forbidden_genes`` (default:
       {ignore_query_forbidden_genes}), also ignore the ``forbidden_gene`` of the query. All these genes are ignored by
       the following code.

    3. Invoke :py:func:`metacells.tools.project.project_query_onto_atlas` and
       :py:func:`metacells.tools.project.compute_query_projection` to project each query metacell onto the atlas, using
       the ``project_fold_normalization`` (default: {project_fold_normalization}),
       ``project_min_significant_gene_value`` (default: {project_min_significant_gene_value}),
       ``project_candidates_count`` (default: {project_candidates_count}), ``project_min_usage_weight`` (default:
       {project_min_usage_weight}), and ``project_abs_folds`` (default: {project_abs_folds}).

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

    6. Invoke :py:func:`metacells.tools.quality.compute_significant_projected_consistency_factors` using the projection
       weights, ``project_min_consistency_weight`` (default: {project_min_consistency_weight}),
       ``project_min_total_consistency_weight`` (default: {project_min_total_consistency_weight}),
       ``project_fold_normalization`` (default: {project_fold_normalization}), ``project_min_significant_gene_value``
       (default: {project_min_significant_gene_value}), ``project_max_consistency_fold_factor`` (default:
       {project_max_consistency_fold_factor}), and ``min_entry_project_consistency_fold_factor`` (default:
       {min_entry_project_consistency_fold_factor}).

    7. Invoke :py:func:`metacells.tools.quality.compute_similar_query_metacells` to annotate the query metacells
       similar to their projection on the atlas using ``project_max_projection_fold_factor`` (default:
       {project_max_projection_fold_factor}), ``project_max_consistency_fold_factor`` (default:
       {project_max_consistency_fold_factor}), and ``project_max_inconsistent_genes`` (default:
       {project_max_inconsistent_genes}).
    """
    ignored_mask_names = ["|systematic_gene"]

    if ignored_gene_names is not None or ignored_gene_patterns is not None:
        tl.find_named_genes(qdata, to="manually_ignored_gene", names=ignored_gene_names, patterns=ignored_gene_patterns)
        ignored_mask_names.append("|manually_ignored_gene")

    if list(qdata.var_names) != list(adata.var_names):
        query_genes_list = list(qdata.var_names)
        atlas_genes_list = list(adata.var_names)
        common_genes_list = list(sorted(set(qdata.var_names) & set(adata.var_names)))
        query_gene_indices = np.array([query_genes_list.index(gene) for gene in common_genes_list])
        atlas_gene_indices = np.array([atlas_genes_list.index(gene) for gene in common_genes_list])
        common_qdata = ut.slice(qdata, name=".common", vars=query_gene_indices, track_var="full_index")
        common_adata = ut.slice(adata, name=".common", vars=atlas_gene_indices, track_var="full_index")
    else:
        common_adata = adata
        common_qdata = qdata

    assert list(common_qdata.var_names) == list(common_adata.var_names)
    full_gene_index_of_common_qdata = ut.get_v_numpy(common_qdata, "full_index")

    common_genes_mask = np.full(qdata.n_vars, False)
    common_genes_mask[full_gene_index_of_common_qdata] = True
    ut.set_v_data(qdata, "atlas_gene", common_genes_mask)

    atlas_total_umis = ut.get_o_numpy(common_adata, what, sum=True)
    query_total_umis = ut.get_o_numpy(common_qdata, what, sum=True)

    tl.find_systematic_genes(
        what,
        adata=common_adata,
        qdata=common_qdata,
        atlas_total_umis=atlas_total_umis,
        query_total_umis=query_total_umis,
        low_gene_quantile=systematic_low_gene_quantile,
        high_gene_quantile=systematic_high_gene_quantile,
    )

    if ignore_atlas_forbidden_genes and ut.has_data(common_adata, "forbidden_gene"):
        atlas_forbiden_mask = ut.get_v_numpy(common_adata, "forbidden_gene")
        ut.set_v_data(common_qdata, "atlas_forbidden_gene", atlas_forbiden_mask)
        ignored_mask_names.append("|atlas_forbidden_gene")

    if ignore_atlas_insignificant_genes and ut.has_data(common_adata, "significant_gene"):
        atlas_significant_mask = ut.get_v_numpy(common_adata, "significant_gene")
        ut.set_v_data(common_qdata, "atlas_significant_gene", atlas_significant_mask)
        ignored_mask_names.append("|~atlas_significant_gene")

    if ignore_query_forbidden_genes and ut.has_data(common_qdata, "forbidden_gene"):
        ignored_mask_names.append("|forbidden_gene")

    for mask_name in ignored_mask_names:
        while mask_name[0] in "|&~":
            mask_name = mask_name[1:]
        if mask_name == "forbidden_gene":
            continue
        full_mask = np.zeros(qdata.n_vars, dtype="bool")
        full_mask[full_gene_index_of_common_qdata] = ut.get_v_numpy(common_qdata, mask_name)
        ut.set_v_data(qdata, mask_name, full_mask)

    full_biased_genes_mask = np.zeros(qdata.n_vars, dtype="bool")

    full_gene_index_of_included_qdata = full_gene_index_of_common_qdata
    included_adata = common_adata
    included_qdata = common_qdata

    while True:

        tl.combine_masks(included_qdata, ignored_mask_names, to="ignored_gene")
        included_genes_mask = ~ut.get_v_numpy(included_qdata, "ignored_gene")
        included_qdata = ut.slice(included_qdata, name=".included", vars=included_genes_mask)
        included_adata = ut.slice(included_adata, name=".included", vars=included_genes_mask)
        full_gene_index_of_included_qdata = ut.get_v_numpy(included_qdata, "full_index")

        weights = tl.project_query_onto_atlas(
            what,
            adata=included_adata,
            qdata=included_qdata,
            atlas_total_umis=atlas_total_umis,
            query_total_umis=query_total_umis,
            fold_normalization=project_fold_normalization,
            min_significant_gene_value=project_min_significant_gene_value,
            candidates_count=project_candidates_count,
            min_usage_weight=project_min_usage_weight,
            abs_folds=project_abs_folds,
        )

        tl.compute_query_projection(
            adata=included_adata,
            qdata=included_qdata,
            weights=weights,
            atlas_total_umis=atlas_total_umis,
            query_total_umis=query_total_umis,
        )

        tl.compute_significant_projected_fold_factors(
            included_qdata,
            what,
            total_umis=query_total_umis,
            fold_normalization=project_fold_normalization,
            min_gene_fold_factor=project_max_projection_fold_factor,
            min_entry_fold_factor=min_entry_project_fold_factor,
            min_significant_gene_value=project_min_significant_gene_value,
            abs_folds=project_abs_folds,
        )

        if ignored_mask_names == ["biased_gene"]:
            break

        tl.find_biased_genes(
            included_qdata,
            max_projection_fold_factor=project_max_projection_fold_factor,
            min_metacells_fraction=biased_min_metacells_fraction,
            abs_folds=project_abs_folds,
        )

        biased_genes_mask = ut.get_v_numpy(included_qdata, "biased_gene")
        if np.sum(biased_genes_mask) == 0:
            break

        full_biased_genes_mask[full_gene_index_of_included_qdata] = biased_genes_mask
        ignored_mask_names = ["biased_gene"]

    tl.compute_significant_projected_consistency_factors(
        what,
        adata=included_adata,
        qdata=included_qdata,
        weights=weights,
        atlas_total_umis=atlas_total_umis,
        min_consistency_weight=project_min_consistency_weight,
        min_total_consistency_weight=project_min_total_consistency_weight,
        fold_normalization=project_fold_normalization,
        min_significant_gene_value=project_min_significant_gene_value,
        min_gene_fold_factor=project_max_consistency_fold_factor,
        min_entry_fold_factor=min_entry_project_consistency_fold_factor,
    )

    tl.compute_similar_query_metacells(
        included_qdata,
        max_projection_fold_factor=project_max_projection_fold_factor,
        max_consistency_fold_factor=project_max_consistency_fold_factor,
        max_inconsistent_genes=project_max_inconsistent_genes,
    )

    ignored_genes_mask = np.full(qdata.n_vars, True)
    ignored_genes_mask[full_gene_index_of_included_qdata] = False
    ut.set_v_data(qdata, "ignored_gene", ignored_genes_mask)

    ut.set_v_data(qdata, "biased_gene", full_biased_genes_mask)

    ut.set_o_data(qdata, "similar", ut.get_o_numpy(included_qdata, "similar"))

    tl.compute_query_projection(
        adata=common_adata,
        qdata=common_qdata,
        weights=weights,
        atlas_total_umis=atlas_total_umis,
        query_total_umis=query_total_umis,
    )

    projected = ut.get_vo_proper(common_qdata, "projected")
    full_projected = np.zeros(qdata.shape, dtype="float32")
    full_projected[:, full_gene_index_of_common_qdata] = projected
    ut.set_vo_data(qdata, "projected", full_projected)

    projected_fold = ut.get_vo_proper(included_qdata, "projected_fold")
    full_projected_fold = sp.csr_matrix(qdata.shape, dtype="float32")
    full_projected_fold[:, full_gene_index_of_included_qdata] = projected_fold
    ut.set_vo_data(qdata, "projected_fold", full_projected_fold)

    consistency_fold = ut.get_vo_proper(included_qdata, "consistency_fold")
    full_consistency_fold = sp.csr_matrix(qdata.shape, dtype="float32")
    full_consistency_fold[:, full_gene_index_of_included_qdata] = consistency_fold
    ut.set_vo_data(qdata, "consistency_fold", full_consistency_fold)

    ut.set_m_data(qdata, "project_max_projection_fold_factor", project_max_projection_fold_factor)
    ut.set_m_data(qdata, "project_max_consistency_fold_factor", project_max_consistency_fold_factor)

    return weights


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def typed_projection_pipeline(  # pylint: disable=too-many-statements
    what: str = "__x__",
    *,
    adata: AnnData,
    qdata: AnnData,
    ignored_gene_names: Optional[Collection[str]] = None,
    ignored_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    ignore_atlas_insignificant_genes: bool = pr.ignore_atlas_insignificant_genes,
    ignore_atlas_forbidden_genes: bool = pr.ignore_atlas_forbidden_genes,
    ignore_query_forbidden_genes: bool = pr.ignore_query_forbidden_genes,
    systematic_low_gene_quantile: float = pr.systematic_low_gene_quantile,
    systematic_high_gene_quantile: float = pr.systematic_high_gene_quantile,
    biased_min_metacells_fraction: float = pr.biased_min_metacells_fraction,
    project_fold_normalization: float = pr.project_fold_normalization,
    project_candidates_count: int = pr.project_candidates_count,
    project_min_significant_gene_value: float = pr.project_min_significant_gene_value,
    project_min_usage_weight: float = pr.project_min_usage_weight,
    project_min_consistency_weight: float = pr.project_min_consistency_weight,
    project_min_total_consistency_weight: float = pr.project_min_total_consistency_weight,
    project_max_consistency_fold_factor: float = pr.project_max_consistency_fold_factor,
    project_max_inconsistent_genes: int = pr.project_max_inconsistent_genes,
    project_max_projection_fold_factor: float = pr.project_max_projection_fold_factor,
    min_entry_project_fold_factor: float = pr.min_entry_project_fold_factor,
    min_entry_project_consistency_fold_factor: float = pr.min_entry_project_consistency_fold_factor,
    project_abs_folds: bool = pr.project_abs_folds,
    ignored_gene_names_of_type: Optional[Dict[str, Collection[str]]] = None,
    ignored_gene_patterns_of_type: Optional[Dict[str, Collection[str]]] = None,
    type_property_name: str = "type",
) -> ut.CompressedMatrix:
    """
    Similar to :py:func:`direct_projection_pipeline`, but perform a second phase depending on the ``type`` annotations
    of the atlas metacells.

    Sets the following additional annotations in ``qdata``:

    Observation (Metacell) Annotations
        ``projected_type``
            A type assigned to each query metacell based on the types of the atlas metacells it was projected to.

    Variable (Gene) Annotations
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

    **Computation Parameters**

    1. Invokes :py:func:`direct_projection_pipeline` with all the parameters except for ``typed_ignored_gene_names``,
       ``typed_ignored_gene_patterns`` and ``type_property_name``.

    2. Invoke :py:func:`metacells.tools.project.project_atlas_to_query` to assign a projected type to each of the
       query metacells based on the ``type_property_name`` (default: {type_property_name}) of the atlas metacells.

    For each type of query metacells:

    3. Invoke :py:func:`metacells.tools.project.find_systematic_genes` to compute the systematic genes using only
       metacells of this type in both the query and the atlas. Use these to create a list of type-specific ignored
       genes, instead of the global systematic genes list.

    4. Invoke :py:func:`metacells.tools.project.project_query_onto_atlas` and
       :py:func:`metacells.tools.project.compute_query_projection` to project each query metacell onto the atlas, using
       the type-specific ignored genes in addition to the global ignored genes. Note that this projection is not
       restricted to using atlas metacells of the specific type.

    5. Invoke :py:func:`metacells.tools.project.compute_significant_projected_fold_factors` to compute the significant
       fold factors between the query and its projection.

    6. Invoke :py:func:`metacells.tools.project.find_biased_genes` to identify type-specific biased genes. If any
       such genes are found, add them to the type-specific ignored genes (instead of the global biased genes list)
       and repeat steps 4-5.

    7. Invoke :py:func:`metacells.tools.quality.compute_significant_projected_consistency_factors` and
       :py:func:`metacells.tools.quality.compute_similar_query_metacells` to validate the updated projection.
    """
    if ignored_gene_names_of_type is None:
        ignored_gene_names_of_type = {}
    if ignored_gene_patterns_of_type is None:
        ignored_gene_patterns_of_type = {}

    weights = direct_projection_pipeline(
        what,
        adata=adata,
        qdata=qdata,
        ignored_gene_names=ignored_gene_names,
        ignored_gene_patterns=ignored_gene_patterns,
        ignore_atlas_insignificant_genes=ignore_atlas_insignificant_genes,
        ignore_atlas_forbidden_genes=ignore_atlas_forbidden_genes,
        ignore_query_forbidden_genes=ignore_query_forbidden_genes,
        systematic_low_gene_quantile=systematic_low_gene_quantile,
        systematic_high_gene_quantile=systematic_high_gene_quantile,
        biased_min_metacells_fraction=biased_min_metacells_fraction,
        project_fold_normalization=project_fold_normalization,
        project_candidates_count=project_candidates_count,
        project_min_significant_gene_value=project_min_significant_gene_value,
        project_min_usage_weight=project_min_usage_weight,
        project_min_consistency_weight=project_min_consistency_weight,
        project_min_total_consistency_weight=project_min_total_consistency_weight,
        project_max_consistency_fold_factor=project_max_consistency_fold_factor,
        project_max_inconsistent_genes=project_max_inconsistent_genes,
        project_max_projection_fold_factor=project_max_projection_fold_factor,
        min_entry_project_fold_factor=project_max_projection_fold_factor,
        min_entry_project_consistency_fold_factor=project_max_projection_fold_factor,
        project_abs_folds=project_abs_folds,
    )

    tl.project_atlas_to_query(
        adata=adata,
        qdata=qdata,
        weights=weights,
        property_name=type_property_name,
        to_property_name="projected_type",
    )

    if list(qdata.var_names) != list(adata.var_names):
        query_genes_list = list(qdata.var_names)
        atlas_genes_list = list(adata.var_names)
        common_genes_list = list(sorted(set(qdata.var_names) & set(adata.var_names)))
        query_gene_indices = np.array([query_genes_list.index(gene) for gene in common_genes_list])
        atlas_gene_indices = np.array([atlas_genes_list.index(gene) for gene in common_genes_list])
        common_qdata = ut.slice(qdata, name=".common", vars=query_gene_indices, track_var="full_index")
        common_adata = ut.slice(adata, name=".common", vars=atlas_gene_indices, track_var="full_index")
    else:
        common_adata = adata
        common_qdata = qdata

    assert list(common_qdata.var_names) == list(common_adata.var_names)

    full_gene_index_of_common_qdata = ut.get_v_numpy(common_qdata, "full_index")

    atlas_total_umis = ut.get_o_numpy(common_adata, what, sum=True)
    query_total_umis = ut.get_o_numpy(common_qdata, what, sum=True)

    type_of_atlas_metacells = ut.get_o_numpy(common_adata, type_property_name)
    type_of_query_metacells = ut.get_o_numpy(common_qdata, "projected_type")
    unique_types = np.unique(type_of_query_metacells)

    full_weights = sp.csr_matrix((qdata.n_obs, adata.n_obs), dtype="float32")
    full_projected_fold = np.zeros(qdata.shape, dtype="float32")
    full_consistency_fold = np.zeros(qdata.shape, dtype="float32")
    full_similar = np.zeros(qdata.n_obs, dtype="bool")

    for type_name in unique_types:
        atlas_type_mask = type_of_atlas_metacells == type_name
        query_type_mask = type_of_query_metacells == type_name
        full_cell_index_of_type_qdata = np.where(query_type_mask)[0]

        ut.log_calc("type_name", type_name)
        ut.log_calc("query cells mask", query_type_mask)
        ut.log_calc("atlas cells mask", atlas_type_mask)

        atlas_type_total_umis = atlas_total_umis[atlas_type_mask]
        query_type_total_umis = query_total_umis[query_type_mask]

        type_common_adata = ut.slice(common_adata, obs=atlas_type_mask, name=f".{type_name}")
        type_common_qdata = ut.slice(common_qdata, obs=query_type_mask, name=f".{type_name}")

        tl.find_systematic_genes(
            what,
            adata=type_common_adata,
            qdata=type_common_qdata,
            atlas_total_umis=atlas_type_total_umis,
            query_total_umis=query_type_total_umis,
            low_gene_quantile=systematic_low_gene_quantile,
            high_gene_quantile=systematic_high_gene_quantile,
        )

        type_ignored_mask_names = ["|ignored_gene"]

        type_systematic_genes_mask = ut.get_v_numpy(type_common_qdata, "systematic_gene")
        ut.set_v_data(type_common_qdata, f"systematic_gene_of_{type_name}", type_systematic_genes_mask)
        type_ignored_mask_names.append(f"|systematic_gene_of_{type_name}")

        if type_name in ignored_gene_names_of_type or type_name in ignored_gene_patterns_of_type:
            tl.find_named_genes(
                type_common_qdata,
                to=f"manually_ignored_gene_of_{type_name}",
                names=ignored_gene_names_of_type.get(type_name),
                patterns=ignored_gene_patterns_of_type.get(type_name),
            )
            type_ignored_mask_names.append(f"|manually_ignored_gene_of_{type_name}")

        for mask_name in type_ignored_mask_names:
            while mask_name[0] in "|&~":
                mask_name = mask_name[1:]
            if mask_name == "ignored_gene":
                continue
            full_mask = np.zeros(qdata.n_vars, dtype="bool")
            full_mask[full_gene_index_of_common_qdata] = ut.get_v_numpy(type_common_qdata, mask_name)
            ut.set_v_data(qdata, mask_name, full_mask)

        full_type_biased_genes_mask = np.zeros(qdata.n_vars, dtype="bool")

        type_included_adata = common_adata
        type_included_qdata = type_common_qdata
        full_gene_index_of_type_included_qdata = full_gene_index_of_common_qdata

        while True:

            tl.combine_masks(type_included_qdata, type_ignored_mask_names, to=f"ignored_gene_of_{type_name}")
            type_included_genes_mask = ~ut.get_v_numpy(type_included_qdata, f"ignored_gene_of_{type_name}")
            type_included_qdata = ut.slice(type_included_qdata, name=".included", vars=type_included_genes_mask)
            type_included_adata = ut.slice(type_included_adata, name=".included", vars=type_included_genes_mask)
            full_gene_index_of_type_included_qdata = ut.get_v_numpy(type_included_qdata, "full_index")

            type_weights = tl.project_query_onto_atlas(
                what,
                adata=type_included_adata,
                qdata=type_included_qdata,
                atlas_total_umis=atlas_total_umis,
                query_total_umis=query_type_total_umis,
                fold_normalization=project_fold_normalization,
                min_significant_gene_value=project_min_significant_gene_value,
                candidates_count=project_candidates_count,
                min_usage_weight=project_min_usage_weight,
                abs_folds=project_abs_folds,
            )

            tl.compute_query_projection(
                adata=type_included_adata,
                qdata=type_included_qdata,
                weights=type_weights,
                atlas_total_umis=atlas_total_umis,
                query_total_umis=query_type_total_umis,
            )

            tl.compute_significant_projected_fold_factors(
                type_included_qdata,
                what,
                total_umis=query_type_total_umis,
                fold_normalization=project_fold_normalization,
                min_gene_fold_factor=project_max_projection_fold_factor,
                min_entry_fold_factor=min_entry_project_fold_factor,
                min_significant_gene_value=project_min_significant_gene_value,
                abs_folds=project_abs_folds,
            )

            if type_ignored_mask_names == ["biased_gene"]:
                break

            tl.find_biased_genes(
                type_included_qdata,
                max_projection_fold_factor=project_max_projection_fold_factor,
                min_metacells_fraction=biased_min_metacells_fraction,
                abs_folds=project_abs_folds,
            )

            type_biased_genes_mask = ut.get_v_numpy(type_included_qdata, "biased_gene")
            if np.sum(type_biased_genes_mask) == 0:
                break

            full_type_biased_genes_mask[full_gene_index_of_type_included_qdata] = type_biased_genes_mask
            type_ignored_mask_names = ["biased_gene"]

        tl.compute_significant_projected_consistency_factors(
            what,
            adata=type_included_adata,
            qdata=type_included_qdata,
            weights=type_weights,
            atlas_total_umis=atlas_total_umis,
            min_consistency_weight=project_min_consistency_weight,
            min_total_consistency_weight=project_min_total_consistency_weight,
            fold_normalization=project_fold_normalization,
            min_significant_gene_value=project_min_significant_gene_value,
            min_gene_fold_factor=project_max_consistency_fold_factor,
            min_entry_fold_factor=min_entry_project_consistency_fold_factor,
        )

        tl.compute_similar_query_metacells(
            type_included_qdata,
            max_projection_fold_factor=project_max_projection_fold_factor,
            max_consistency_fold_factor=project_max_consistency_fold_factor,
            max_inconsistent_genes=project_max_inconsistent_genes,
        )

        full_type_ignored_genes_mask = np.full(qdata.n_vars, True)
        full_type_ignored_genes_mask[full_gene_index_of_type_included_qdata] = False
        ut.set_v_data(qdata, f"ignored_gene_of_{type_name}", full_type_ignored_genes_mask)

        ut.set_v_data(qdata, f"biased_gene_of_{type_name}", full_type_biased_genes_mask)

        full_weights[full_cell_index_of_type_qdata, :] = type_weights
        full_similar[full_cell_index_of_type_qdata] = ut.get_o_numpy(type_included_qdata, "similar")
        type_projected_fold = ut.to_numpy_matrix(ut.get_vo_proper(type_included_qdata, "projected_fold"))
        full_projected_fold[
            full_cell_index_of_type_qdata[:, None], full_gene_index_of_type_included_qdata[None, :]
        ] = type_projected_fold
        type_consistency_fold = ut.to_numpy_matrix(ut.get_vo_proper(type_included_qdata, "consistency_fold"))
        full_consistency_fold[
            full_cell_index_of_type_qdata[:, None], full_gene_index_of_type_included_qdata[None, :]
        ] = type_consistency_fold

    ut.set_o_data(qdata, "similar", full_similar)
    ut.set_vo_data(qdata, "projected_fold", sp.csr_matrix(full_projected_fold))
    ut.set_vo_data(qdata, "consistency_fold", sp.csr_matrix(full_consistency_fold))

    tl.compute_query_projection(
        adata=common_adata,
        qdata=common_qdata,
        weights=weights,
        atlas_total_umis=atlas_total_umis,
        query_total_umis=query_total_umis,
    )

    projected = ut.get_vo_proper(common_qdata, "projected")
    full_projected = np.zeros(qdata.shape, dtype="float32")
    full_projected[:, full_gene_index_of_common_qdata] = projected
    ut.set_vo_data(qdata, "projected", full_projected)

    return full_weights
