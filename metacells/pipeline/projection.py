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
def projection_pipeline(
    what: str = "__x__",
    *,
    adata: AnnData,
    qdata: AnnData,
    ignored_gene_names: Optional[Collection[str]] = None,
    ignored_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    ignore_atlas_forbidden_genes: bool = pr.ignore_atlas_forbidden_genes,
    ignore_query_forbidden_genes: bool = pr.ignore_query_forbidden_genes,
    systematic_low_gene_quantile: float = pr.systematic_low_gene_quantile,
    systematic_high_gene_quantile: float = pr.systematic_high_gene_quantile,
    systematic_min_low_gene_fraction: float = pr.systematic_min_low_gene_fraction,
    biased_min_metacells_fraction: float = pr.biased_min_metacells_fraction,
    project_fold_normalization: float = pr.project_fold_normalization,
    project_candidates_count: int = pr.project_candidates_count,
    project_min_significant_gene_value: float = pr.project_min_significant_gene_value,
    project_min_usage_weight: float = pr.project_min_usage_weight,
    project_min_consistency_weight: float = pr.project_min_consistency_weight,
    project_max_consistency_fold: float = pr.project_max_consistency_fold,
    project_max_inconsistent_genes: int = pr.project_max_inconsistent_genes,
    project_max_projection_fold: float = pr.project_max_projection_fold,
    min_entry_project_fold_factor: float = pr.min_entry_project_fold_factor,
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

        ``ignored_gene``
            A boolean mask indicating whether the gene was ignored by the projection (for any reason).

    Observation (Cell) Annotations
        ``charted``
            A boolean mask indicating whether the query metacell is meaningfully projected onto the atlas. If ``False``
            the metacells is said to be "uncharted", which may indicate the query contains cell states that do not
            appear in the atlas.

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
       {systematic_low_gene_quantile}), ``systematic_high_gene_quantile`` (default: {systematic_high_gene_quantile}),
       and ``systematic_min_low_gene_fraction`` (default: {systematic_min_low_gene_fraction}).

    2. Compute a mask of ignored genes, containing any genes named in ``ignored_gene_names`` or that match any of the
       ``ignored_gene_patterns``. If ``ignore_atlas_forbidden_genes`` (default: {ignore_atlas_forbidden_genes}), also
       ignore the ``forbidden_gene`` of the atlas (and store them as a ``atlas_forbidden_gene`` mask). If
       ``ignore_query_forbidden_genes`` (default: {ignore_query_forbidden_genes}), also ignore the ``forbidden_gene`` of
       the query. All these genes are ignored by the following code.

    3. Invoke :py:func:`metacells.tools.project.project_query_onto_atlas` to project each query metacell onto
       the atlas, using the ``project_fold_normalization`` (default: {project_fold_normalization}),
       ``project_candidates_count`` (default: {project_candidates_count}), ``project_min_significant_gene_value``
       (default: {project_min_significant_gene_value}), ``project_min_usage_weight`` (default:
       {project_min_usage_weight}), ``project_min_consistency_weight`` (default: {project_min_consistency_weight}),
       ``project_max_consistency_fold`` (default: {project_max_consistency_fold}), ``project_max_inconsistent_genes``
       (default: {project_max_inconsistent_genes}), and ``project_max_projection_fold`` (default:
       {project_max_projection_fold}).

    4. Invoke :py:func:`metacells.tools.quality.compute_project_fold_factors` to collect the fold factors of the
       query metacells relative to their projection on the atlas, using the ``project_fold_normalization`` (default:
       {project_fold_normalization}), the ``project_max_projection_fold`` (default: {project_max_projection_fold}), and
       the ``min_entry_project_fold_factor`` (default: {min_entry_project_fold_factor}).

    5. Invoke :py:func:`metacells.tools.project.find_biased_genes` using the ``project_max_projection_fold`` and the
       ``biased_min_metacells_fraction`` (default: {biased_min_metacells_fraction}). If any biased genes were found,
       repeat steps 2-4 while including the biased genes in the ignored genes list.
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

    tl.find_systematic_genes(
        what,
        adata=common_adata,
        qdata=common_qdata,
        low_gene_quantile=systematic_low_gene_quantile,
        high_gene_quantile=systematic_high_gene_quantile,
        min_low_gene_fraction=systematic_min_low_gene_fraction,
    )

    if ignore_atlas_forbidden_genes and ut.has_data(common_adata, "forbidden_gene"):
        atlas_forbiden_mask = ut.get_v_numpy(common_adata, "forbidden_gene")
        ut.set_v_data(common_qdata, "atlas_forbidden_gene", atlas_forbiden_mask)
        ignored_mask_names.append("|atlas_forbidden_gene")

    if ignore_query_forbidden_genes and ut.has_data(common_qdata, "forbidden_gene"):
        ignored_mask_names.append("|forbidden_gene")

    full_index_of_common_qdata = ut.get_v_numpy(common_qdata, "full_index")

    for mask_name in ignored_mask_names:
        mask_name = mask_name[1:]
        if mask_name == "forbidden_gene":
            continue
        full_mask = np.zeros(qdata.n_vars, dtype="bool")
        full_mask[full_index_of_common_qdata] = ut.get_v_numpy(common_qdata, mask_name)
        ut.set_v_data(qdata, mask_name, full_mask)

    full_biased_genes_mask = np.zeros(qdata.n_vars, dtype="bool")

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
            project_fold_normalization=project_fold_normalization,
            project_candidates_count=project_candidates_count,
            project_min_significant_gene_value=project_min_significant_gene_value,
            project_min_usage_weight=project_min_usage_weight,
            project_min_consistency_weight=project_min_consistency_weight,
            project_max_consistency_fold=project_max_consistency_fold,
            project_max_inconsistent_genes=project_max_inconsistent_genes,
            project_max_projection_fold=project_max_projection_fold,
        )

        tl.compute_projected_fold_factors(
            common_qdata,
            what,
            fold_normalization=project_fold_normalization,
            min_gene_fold_factor=project_max_projection_fold,
            min_entry_fold_factor=min_entry_project_fold_factor,
        )

        tl.find_biased_genes(
            common_qdata,
            what,
            project_max_projection_fold=project_max_projection_fold,
            project_min_significant_gene_value=project_min_significant_gene_value,
            min_metacells_fraction=biased_min_metacells_fraction
        )

        if ignored_mask_names == [ "biased_gene" ]:
            break

        biased_genes_mask = ut.get_v_numpy(common_qdata, "biased_gene")
        if np.sum(biased_genes_mask) == 0:
            break

        full_biased_genes_mask[full_index_of_common_qdata] = biased_genes_mask
        ignored_mask_names = [ "biased_gene" ]

    ignored_genes_mask = np.full(qdata.n_vars, True)
    ignored_genes_mask[full_index_of_common_qdata] = False
    ut.set_v_data(qdata, "ignored_gene", ignored_genes_mask)

    ut.set_v_data(qdata, "biased_gene", full_biased_genes_mask)

    ut.set_o_data(qdata, "charted", ut.get_o_numpy(common_qdata, "charted"))

    projected = ut.get_vo_proper(common_qdata, "projected")
    projected_fold = ut.get_vo_proper(common_qdata, "projected_fold")
    significant_projected_fold = ut.get_vo_proper(common_qdata, "significant_projected_fold")

    full_index_of_common_qdata = ut.get_v_numpy(common_qdata, "full_index")

    full_projected = np.zeros(qdata.shape, dtype="float32")
    full_projected_fold = sp.csr_matrix(qdata.shape, dtype="float32")
    full_significant_projected_fold = sp.csr_matrix(qdata.shape, dtype="bool")

    full_projected[:, full_index_of_common_qdata] = projected
    full_projected_fold[:, full_index_of_common_qdata] = projected_fold
    full_significant_projected_fold[:, full_index_of_common_qdata] = significant_projected_fold

    ut.set_vo_data(qdata, "projected", full_projected)
    ut.set_vo_data(qdata, "projected_fold", full_projected_fold)
    ut.set_vo_data(qdata, "significant_projected_fold", full_significant_projected_fold)

    ut.set_m_data(qdata, "project_max_projection_fold", project_max_projection_fold)

    return weights
