"""
Related Genes
-------------
"""

from re import Pattern
from typing import Collection
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd  # type: ignore
import scipy.cluster.hierarchy as sch  # type: ignore
import scipy.sparse as sp  # type: ignore
import scipy.spatial.distance as scd  # type: ignore
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

from .feature import extract_feature_data

__all__ = [
    "relate_genes",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def relate_genes(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    max_sampled_cells: int = pr.related_max_sampled_cells,
    downsample_min_samples: float = pr.related_downsample_min_samples,
    downsample_min_cell_quantile: float = pr.related_downsample_min_cell_quantile,
    downsample_max_cell_quantile: float = pr.related_downsample_max_cell_quantile,
    min_gene_relative_variance: float = pr.related_min_gene_relative_variance,
    min_gene_total: int = pr.related_min_gene_total,
    min_gene_top3: int = pr.related_min_gene_top3,
    forbidden_gene_names: Optional[Collection[str]] = None,
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    genes_similarity_method: str = pr.related_genes_similarity_method,
    genes_cluster_method: str = pr.related_genes_cluster_method,
    min_genes_of_modules: int = pr.related_min_genes_of_modules,
    random_seed: int = 0,
) -> None:
    """
    Detect coarse relations between genes based on ``what`` (default: {what}) data.

    This is a quick-and-dirty way to group genes together and shouldn't only be used as a starting
    point for more precise forms of gene relationship analysis.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    **Returns**

    Variable-pair (Gene) Annotations
        ``related_genes_similarity``
            The similarity between each two related genes.

    Variable (Gene) Annotations
        ``related_genes_module``
            The index of the gene module for each gene.

    **Computation Parameters**

    1. If we have more than ``max_sampled_cells`` (default: {max_sampled_cells}), pick this number
       of random cells from the data using the ``random_seed``.

    2. Pick candidate genes using :py:func:`metacells.pipeline.feature.extract_feature_data`.

    3. Compute the similarity between the feature genes using
       :py:func:`metacells.tools.similarity.compute_var_var_similarity` using the
       ``genes_similarity_method`` (default: {genes_similarity_method}).

    4. Create a hierarchical clustering of the candidate genes using the ``genes_cluster_method``
       (default: {genes_cluster_method}).

    5. Identify gene modules in the hierarchical clustering which contain at least
       ``min_genes_of_modules`` genes.
    """
    if max_sampled_cells < adata.n_obs:
        np.random.seed(random_seed)
        cell_indices = np.random.choice(np.arange(adata.n_obs), size=max_sampled_cells, replace=False)
        sdata = ut.slice(adata, obs=cell_indices, name=".sampled", top_level=False)
    else:
        sdata = ut.copy_adata(adata, top_level=False)

    fdata = extract_feature_data(
        sdata,
        what,
        top_level=False,
        downsample_min_samples=downsample_min_samples,
        downsample_min_cell_quantile=downsample_min_cell_quantile,
        downsample_max_cell_quantile=downsample_max_cell_quantile,
        min_gene_relative_variance=min_gene_relative_variance,
        min_gene_total=min_gene_total,
        min_gene_top3=min_gene_top3,
        forbidden_gene_names=forbidden_gene_names,
        forbidden_gene_patterns=forbidden_gene_patterns,
        random_seed=random_seed,
    )
    assert fdata is not None

    frame = tl.compute_var_var_similarity(
        fdata, what, method=genes_similarity_method, reproducible=(random_seed != 0), inplace=False
    )
    assert frame is not None
    similarity = ut.to_layout(ut.to_numpy_matrix(frame), layout="row_major")

    linkage = _cluster_genes(similarity, genes_cluster_method)
    clusters = _linkage_to_clusters(linkage, min_genes_of_modules, fdata.n_vars)

    cluster_of_genes = pd.Series(np.full(adata.n_vars, -1, dtype="int32"), index=adata.var_names)
    for cluster_index, gene_indices in enumerate(clusters):
        cluster_of_genes[fdata.var_names[gene_indices]] = cluster_index

    ut.set_v_data(adata, "related_genes_module", cluster_of_genes, formatter=ut.groups_description)

    feature_gene_indices = ut.get_v_numpy(fdata, "full_gene_index")
    data = similarity.flatten(order="C")
    rows = np.repeat(feature_gene_indices, len(feature_gene_indices))
    cols = np.tile(feature_gene_indices, len(feature_gene_indices))
    full_similarity = sp.csr_matrix((data, (rows, cols)), shape=(adata.n_vars, adata.n_vars))

    ut.set_vv_data(adata, "related_genes_similarity", full_similarity)


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
        linkage = sch.linkage(distances, method=genes_cluster_method)

    return linkage


@ut.timed_call()
def _linkage_to_clusters(
    linkage: List[Tuple[int, int]],
    min_entries_of_modules: int,
    entries_count: int,
) -> List[List[int]]:
    entries_of_cluster = {index: [index] for index in range(entries_count)}

    for link_index, link_data in enumerate(linkage):
        link_index += entries_count

        left_index = int(link_data[0])
        right_index = int(link_data[1])

        left_entries = entries_of_cluster.get(left_index)
        right_entries = entries_of_cluster.get(right_index)

        if left_entries is None or len(left_entries) > min_entries_of_modules:
            continue

        if right_entries is None or len(right_entries) > min_entries_of_modules:
            continue

        entries_of_cluster[link_index] = sorted(left_entries + right_entries)
        del entries_of_cluster[left_index]
        del entries_of_cluster[right_index]

    return list(entries_of_cluster.values())
