"""
Projection
----------
"""

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
    systematic_low_gene_quantile: float = pr.systematic_low_gene_quantile,
    systematic_high_gene_quantile: float = pr.systematic_high_gene_quantile,
    systematic_min_low_gene_fraction: float = pr.systematic_min_low_gene_fraction,
    project_fold_normalization: float = pr.project_fold_normalization,
    project_candidates_count: int = pr.project_candidates_count,
    project_min_significant_gene_value: float = pr.project_min_significant_gene_value,
    project_min_usage_weight: float = pr.project_min_usage_weight,
    project_min_consistency_weight: float = pr.project_min_consistency_weight,
    project_max_consistency_fold: float = pr.project_max_consistency_fold,
    project_max_inconsistent_genes: int = pr.project_max_inconsistent_genes,
    project_max_projection_fold: float = pr.project_max_projection_fold,
    min_gene_project_fold_factor: float = pr.min_gene_project_fold_factor,
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
        ``systematic``
            A boolean mask indicating whether the gene is systematically higher in the query compared to the atlas.

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

    2. Invoke :py:func:`metacells.tools.project.project_query_onto_atlas` to project each query metacell onto
       the atlas, using the ``project_fold_normalization`` (default: {project_fold_normalization}),
       ``project_candidates_count`` (default: {project_candidates_count}), ``project_min_significant_gene_value``
       (default: {project_min_significant_gene_value}), ``project_min_usage_weight`` (default:
       {project_min_usage_weight}), ``project_min_consistency_weight`` (default: {project_min_consistency_weight}),
       ``project_max_consistency_fold`` (default: {project_max_consistency_fold}), ``project_max_inconsistent_genes``
       (default: {project_max_inconsistent_genes}), and ``project_max_projection_fold`` (default:
       {project_max_projection_fold}).

    3. Invoke :py:func:`metacells.tools.quality.compute_project_fold_factors` to collect the fold factors of the
       query metacells relative to their projection on the atlas, using
       the ``project_fold_normalization`` (default: {project_fold_normalization}),
       the ``min_gene_project_fold_factor`` (default: {min_gene_project_fold_factor}),
       and the ``min_entry_project_fold_factor`` (default: {min_entry_project_fold_factor}).
    """
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

    systematic_mask = ut.get_v_numpy(common_qdata, "systematic")
    full_index_of_common_qdata = ut.get_v_numpy(common_qdata, "full_index")
    full_systematic_mask = np.zeros(qdata.n_vars, dtype="bool")
    full_systematic_mask[full_index_of_common_qdata] = systematic_mask

    ut.set_v_data(qdata, "systematic", full_systematic_mask)
    non_systematic_mask = ~systematic_mask
    common_qdata = ut.slice(common_qdata, name=".non_systematic", vars=non_systematic_mask)
    common_adata = ut.slice(common_adata, name=".non_systematic", vars=non_systematic_mask)

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
        min_gene_fold_factor=min_gene_project_fold_factor,
        min_entry_fold_factor=min_entry_project_fold_factor,
    )

    ut.set_o_data(qdata, "charted", ut.get_o_numpy(common_qdata, "charted"))

    projected = ut.get_vo_proper(common_qdata, "projected")
    projected_fold = ut.get_vo_proper(common_qdata, "projected_fold")

    full_index_of_common_qdata = ut.get_v_numpy(common_qdata, "full_index")

    full_projected = np.zeros(qdata.shape, dtype="float32")
    full_projected_fold = sp.csr_matrix(qdata.shape, dtype="float32")

    full_projected[:, full_index_of_common_qdata] = projected
    full_projected_fold[:, full_index_of_common_qdata] = projected_fold

    ut.set_vo_data(qdata, "projected", full_projected)
    ut.set_vo_data(qdata, "projected_fold", full_projected_fold)

    return weights
