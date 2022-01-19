"""
Projection
----------
"""

from re import Pattern
from typing import Collection
from typing import Optional
from typing import Union

import numpy as np
import scipy.sparse as sp  # type: ignore
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    "projection_pipeline",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def projection_pipeline(  # pylint: disable=too-many-statements
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
        full_common_qdata = ut.slice(qdata, name=".common", vars=query_gene_indices, track_var="full_index")
        full_common_adata = ut.slice(adata, name=".common", vars=atlas_gene_indices, track_var="full_index")
    else:
        full_common_adata = adata
        full_common_qdata = qdata

    assert list(full_common_qdata.var_names) == list(full_common_adata.var_names)

    atlas_total_umis = ut.get_o_numpy(full_common_adata, what, sum=True)
    query_total_umis = ut.get_o_numpy(full_common_qdata, what, sum=True)

    tl.find_systematic_genes(
        what,
        adata=full_common_adata,
        qdata=full_common_qdata,
        atlas_total_umis=atlas_total_umis,
        query_total_umis=query_total_umis,
        low_gene_quantile=systematic_low_gene_quantile,
        high_gene_quantile=systematic_high_gene_quantile,
    )

    if ignore_atlas_forbidden_genes and ut.has_data(full_common_adata, "forbidden_gene"):
        atlas_forbiden_mask = ut.get_v_numpy(full_common_adata, "forbidden_gene")
        ut.set_v_data(full_common_qdata, "atlas_forbidden_gene", atlas_forbiden_mask)
        ignored_mask_names.append("|atlas_forbidden_gene")

    if ignore_atlas_insignificant_genes and ut.has_data(full_common_adata, "significant_gene"):
        atlas_significant_mask = ut.get_v_numpy(full_common_adata, "significant_gene")
        ut.set_v_data(full_common_qdata, "atlas_significant_gene", atlas_significant_mask)
        ignored_mask_names.append("|~atlas_significant_gene")

    if ignore_query_forbidden_genes and ut.has_data(full_common_qdata, "forbidden_gene"):
        ignored_mask_names.append("|forbidden_gene")

    full_index_of_full_common_qdata = ut.get_v_numpy(full_common_qdata, "full_index")

    for mask_name in ignored_mask_names:
        while mask_name[0] in "|&~":
            mask_name = mask_name[1:]
        if mask_name == "forbidden_gene":
            continue
        full_mask = np.zeros(qdata.n_vars, dtype="bool")
        full_mask[full_index_of_full_common_qdata] = ut.get_v_numpy(full_common_qdata, mask_name)
        ut.set_v_data(qdata, mask_name, full_mask)

    full_biased_genes_mask = np.zeros(qdata.n_vars, dtype="bool")

    full_index_of_common_qdata = full_index_of_full_common_qdata
    common_adata = full_common_adata
    common_qdata = full_common_qdata

    while True:

        tl.combine_masks(common_qdata, ignored_mask_names, to="ignored_gene")
        ignored_genes_mask = ut.get_v_numpy(common_qdata, "ignored_gene")
        included_genes_mask = ~ignored_genes_mask
        common_qdata = ut.slice(common_qdata, name=".included", vars=included_genes_mask)
        common_adata = ut.slice(common_adata, name=".included", vars=included_genes_mask)
        full_index_of_common_qdata = ut.get_v_numpy(common_qdata, "full_index")

        weights = tl.project_query_onto_atlas(
            what,
            adata=common_adata,
            qdata=common_qdata,
            atlas_total_umis=atlas_total_umis,
            query_total_umis=query_total_umis,
            fold_normalization=project_fold_normalization,
            min_significant_gene_value=project_min_significant_gene_value,
            candidates_count=project_candidates_count,
            min_usage_weight=project_min_usage_weight,
            abs_folds=project_abs_folds,
        )

        tl.compute_query_projection(
            adata=common_adata,
            qdata=common_qdata,
            weights=weights,
            atlas_total_umis=atlas_total_umis,
            query_total_umis=query_total_umis,
        )

        tl.compute_significant_projected_fold_factors(
            common_qdata,
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
            common_qdata,
            max_projection_fold_factor=project_max_projection_fold_factor,
            min_metacells_fraction=biased_min_metacells_fraction,
            abs_folds=project_abs_folds,
        )

        biased_genes_mask = ut.get_v_numpy(common_qdata, "biased_gene")
        if np.sum(biased_genes_mask) == 0:
            break

        full_biased_genes_mask[full_index_of_common_qdata] = biased_genes_mask
        ignored_mask_names = ["biased_gene"]

    tl.compute_significant_projected_consistency_factors(
        what,
        adata=common_adata,
        qdata=common_qdata,
        weights=weights,
        atlas_total_umis=atlas_total_umis,
        min_consistency_weight=project_min_consistency_weight,
        fold_normalization=project_fold_normalization,
        min_significant_gene_value=project_min_significant_gene_value,
        min_gene_fold_factor=project_max_consistency_fold_factor,
        min_entry_fold_factor=min_entry_project_consistency_fold_factor,
    )

    tl.compute_similar_query_metacells(
        common_qdata,
        max_projection_fold_factor=project_max_projection_fold_factor,
        max_consistency_fold_factor=project_max_consistency_fold_factor,
        max_inconsistent_genes=project_max_inconsistent_genes,
    )

    common_genes_mask = np.full(qdata.n_vars, False)
    common_genes_mask[full_index_of_full_common_qdata] = True
    ut.set_v_data(qdata, "atlas_gene", common_genes_mask)

    ignored_genes_mask = np.full(qdata.n_vars, True)
    ignored_genes_mask[full_index_of_common_qdata] = False
    ut.set_v_data(qdata, "ignored_gene", ignored_genes_mask)

    ut.set_v_data(qdata, "biased_gene", full_biased_genes_mask)

    ut.set_o_data(qdata, "similar", ut.get_o_numpy(common_qdata, "similar"))

    tl.compute_query_projection(
        adata=full_common_adata,
        qdata=full_common_qdata,
        weights=weights,
        atlas_total_umis=atlas_total_umis,
        query_total_umis=query_total_umis,
    )

    projected = ut.get_vo_proper(full_common_qdata, "projected")
    full_projected = np.zeros(qdata.shape, dtype="float32")
    full_projected[:, full_index_of_full_common_qdata] = projected
    ut.set_vo_data(qdata, "projected", full_projected)

    projected_fold = ut.get_vo_proper(common_qdata, "projected_fold")
    full_projected_fold = sp.csr_matrix(qdata.shape, dtype="float32")
    full_projected_fold[:, full_index_of_common_qdata] = projected_fold
    ut.set_vo_data(qdata, "projected_fold", full_projected_fold)

    consistency_fold = ut.get_vo_proper(common_qdata, "consistency_fold")
    full_consistency_fold = sp.csr_matrix(qdata.shape, dtype="float32")
    full_consistency_fold[:, full_index_of_common_qdata] = consistency_fold
    ut.set_vo_data(qdata, "consistency_fold", full_consistency_fold)

    ut.set_m_data(qdata, "project_max_projection_fold_factor", project_max_projection_fold_factor)
    ut.set_m_data(qdata, "project_max_consistency_fold_factor", project_max_consistency_fold_factor)

    return weights
