"""
Related Genes
-------------
"""

from typing import List
from typing import Tuple
from typing import Union

import fastcluster as fc  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import scipy.sparse as sp  # type: ignore
import scipy.spatial.distance as scd  # type: ignore
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    "relate_to_lateral_genes",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def relate_to_lateral_genes(  # pylint: disable=too-many-statements
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    max_sampled_cells: int = pr.related_max_sampled_cells,
    genes_similarity_method: str = pr.related_genes_similarity_method,
    genes_cluster_method: str = pr.related_genes_cluster_method,
    max_genes_of_modules: int = pr.related_max_genes_of_modules,
    min_mean_gene_fraction: float = pr.related_min_mean_gene_fraction,
    min_gene_correlation: float = pr.related_min_gene_correlation,
    random_seed: int = 0,
) -> None:
    """
    Detect coarse relations between genes and lateral genes based on ``what`` (default: {what}) data.

    This is a quick-and-dirty way to look for genes highly correlated with lateral genes.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where ``what`` is a
    per-variable-per-observation matrix or the name of a per-variable-per-observation annotation containing such a
    matrix. The data should contain a ``lateral_gene`` mask containing some known-to-be lateral genes.

    **Returns**

    Variable-pair (Gene) Annotations
        ``lateral_genes_similarity``
            The similarity between each genes related to lateral genes.

    Variable (Gene) Annotations
        ``lateral_genes_module``
            The index of the related to lateral gene module for each gene.

    **Computation Parameters**

    1. If we have more than ``max_sampled_cells`` (default: {max_sampled_cells}), pick this number
       of random cells from the data using the ``random_seed``.

    2. Start by marking as "related genes" all the genes listed in ``lateral_gene``.

    3. Pick as candidates genes whose mean fraction in the population is at least the (very low)
       ``min_mean_gene_fraction`` (default: {min_mean_gene_fraction}) (and that are not lateral genes).

    4. Compute the correlation between the related genes and the candidate genes. Move from the candidate genes to the
       lateral genes any genes whose correlation is at least ``min_gene_correlation`` (default: {min_gene_correlation}).

    5. Repeat step 4 until no additional related genes are found.

    6. Compute the similarity between the all related genes using
       :py:func:`metacells.tools.similarity.compute_var_var_similarity` using the ``genes_similarity_method`` (default:
       {genes_similarity_method}).

    7. Create a hierarchical clustering of the candidate genes using the ``genes_cluster_method``
       (default: {genes_cluster_method}).

    8. Identify gene modules in the hierarchical clustering by bottom-up combining genes as long as we have at most
       ``max_genes_of_modules`` (default: {max_genes_of_modules}). Note that this may leave modules with a smaller
       number of genes, and modules that do not contain any lateral genes.
    """
    if max_sampled_cells < adata.n_obs:
        np.random.seed(random_seed)
        cell_indices = np.random.choice(np.arange(adata.n_obs), size=max_sampled_cells, replace=False)
        sdata = ut.slice(adata, obs=cell_indices, name=".sampled", top_level=False)
    else:
        sdata = ut.copy_adata(adata, top_level=False)

    umis_per_cell_per_gene = ut.get_vo_proper(sdata, what)
    fraction_per_gene_per_cell = ut.fraction_by(umis_per_cell_per_gene, by="row")
    fraction_per_gene_per_cell = ut.to_layout(fraction_per_gene_per_cell, layout="column_major")

    mean_fraction_per_gene = ut.mean_per(fraction_per_gene_per_cell, per="column")
    candidate_genes_mask = mean_fraction_per_gene >= min_mean_gene_fraction
    ut.log_calc("candidate_genes_mask", candidate_genes_mask)

    related_genes_mask = np.zeros(sdata.n_vars, dtype="bool")
    additional_genes_mask = ut.get_v_numpy(sdata, "lateral_gene").copy()

    while np.any(additional_genes_mask):
        related_genes_mask |= additional_genes_mask
        ut.log_calc("related_genes_mask", related_genes_mask)

        candidate_genes_mask &= ~additional_genes_mask
        ut.log_calc("candidate_genes_mask", candidate_genes_mask)
        if not np.any(candidate_genes_mask):
            break

        fraction_per_candidate_gene_per_cell = ut.to_numpy_matrix(fraction_per_gene_per_cell[:, candidate_genes_mask])
        fraction_per_additional_gene_per_cell = ut.to_numpy_matrix(fraction_per_gene_per_cell[:, additional_genes_mask])

        fraction_per_cell_per_candidate_gene = np.transpose(fraction_per_candidate_gene_per_cell)
        fraction_per_cell_per_additional_gene = np.transpose(fraction_per_additional_gene_per_cell)

        fraction_per_cell_per_candidate_gene = ut.to_layout(fraction_per_cell_per_candidate_gene, layout="row_major")
        fraction_per_cell_per_additional_gene = ut.to_layout(fraction_per_cell_per_additional_gene, layout="row_major")

        correlation_per_related_per_candidate = ut.cross_corrcoef_rows(
            fraction_per_cell_per_candidate_gene,
            fraction_per_cell_per_additional_gene,
            reproducible=random_seed != 0,
        )
        max_correlation_per_candidate = ut.max_per(correlation_per_related_per_candidate, per="row")
        assert len(max_correlation_per_candidate) == np.sum(candidate_genes_mask)

        additional_genes_mask = np.zeros(sdata.n_vars, dtype="bool")
        additional_genes_mask[candidate_genes_mask] = max_correlation_per_candidate >= min_gene_correlation
        ut.log_calc("additional_genes_mask", additional_genes_mask)

    name = (ut.get_name(sdata) or "cells") + ".related"
    var_names = sdata.var_names[related_genes_mask]
    sdata = AnnData(fraction_per_gene_per_cell[:, related_genes_mask])
    sdata.var_names = var_names
    ut.set_name(sdata, name)

    frame = tl.compute_var_var_similarity(
        sdata, what, method=genes_similarity_method, reproducible=(random_seed != 0), inplace=False
    )
    assert frame is not None
    similarity = ut.to_layout(ut.to_numpy_matrix(frame), layout="row_major")
    ut.log_calc("similarity", similarity)

    linkage = _cluster_genes(similarity, genes_cluster_method)
    ut.log_calc("linkage")
    clusters = _linkage_to_clusters(linkage, max_genes_of_modules, sdata.n_vars)
    ut.log_calc("cluster")

    cluster_of_genes = pd.Series(np.full(adata.n_vars, -1, dtype="int32"), index=adata.var_names)
    for cluster_index, gene_indices in enumerate(clusters):
        cluster_of_genes[sdata.var_names[gene_indices]] = cluster_index

    ut.set_v_data(adata, "lateral_genes_module", cluster_of_genes, formatter=ut.groups_description)

    related_gene_indices = np.where(related_genes_mask)[0]
    data = similarity.flatten(order="C")
    rows = np.repeat(related_gene_indices, len(related_gene_indices))
    columns = np.tile(related_gene_indices, len(related_gene_indices))
    full_similarity = sp.csr_matrix((data, (rows, columns)), shape=(adata.n_vars, adata.n_vars))

    ut.set_vv_data(adata, "lateral_genes_similarity", full_similarity)


# TODO: Replicated in metacell.tools.rare
@ut.timed_call()
def _cluster_genes(
    similarities_between_candidate_genes: ut.NumpyMatrix,
    genes_cluster_method: str,
) -> List[Tuple[int, int]]:
    with ut.timed_step("scipy.pdist"):
        ut.timed_parameters(size=similarities_between_candidate_genes.shape[0])
        distances = scd.pdist(similarities_between_candidate_genes)

    with ut.timed_step("scipy.linkage"):
        ut.timed_parameters(size=distances.shape[0], method=genes_cluster_method)
        linkage = fc.linkage(distances, method=genes_cluster_method)

    return linkage


@ut.timed_call()
def _linkage_to_clusters(
    linkage: List[Tuple[int, int]],
    max_entries_of_modules: int,
    entries_count: int,
) -> List[List[int]]:
    entries_of_cluster = {index: [index] for index in range(entries_count)}

    for link_index, link_data in enumerate(linkage):
        link_index += entries_count

        left_index = int(link_data[0])
        right_index = int(link_data[1])

        left_entries = entries_of_cluster.get(left_index)
        right_entries = entries_of_cluster.get(right_index)

        if (
            left_entries is None
            or right_entries is None
            or len(left_entries) + len(right_entries) > max_entries_of_modules
        ):
            continue

        entries_of_cluster[link_index] = sorted(left_entries + right_entries)
        del entries_of_cluster[left_index]
        del entries_of_cluster[right_index]

    return list(entries_of_cluster.values())
