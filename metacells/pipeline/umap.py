"""
UMAP
----

It is useful to project the metacells data to a 2D scatter plot (where each point is a metacell).
The steps provided here are expected to yield a reasonable such projection, though as always you
might need to tweak the parameters or even the overall flow for specific data sets.
"""

from typing import Tuple
from typing import Union

import igraph as ig  # type: ignore
import numpy as np
from anndata import AnnData  # type: ignore
from scipy import sparse  # type: ignore

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    "compute_knn_by_features",
    "compute_umap_by_features",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_knn_by_features(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    max_top_feature_genes: int = pr.max_top_feature_genes,
    similarity_value_normalization: float = pr.umap_similarity_value_normalization,
    similarity_log_data: bool = pr.umap_similarity_log_data,
    similarity_method: str = pr.umap_similarity_method,
    logistics_location: float = pr.logistics_location,
    logistics_slope: float = pr.logistics_slope,
    k: int,
    balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    reproducible: bool = pr.reproducible,
) -> ut.PandasFrame:
    """
    Compute KNN graph between metacells based on feature genes.

    If ``reproducible`` (default: {reproducible}) is ``True``, a slower (still parallel) but
    reproducible algorithm will be used to compute pearson correlations.

    **Input**

    Annotated ``adata`` where each observation is a metacells and the variables are genes,
    are genes, where ``what`` is a per-variable-per-observation matrix or the name of a
    per-variable-per-observation annotation containing such a matrix.

    **Returns**

    Sets the following in ``adata``:

    Observations-Pair (Metacells) Annotations
        ``obs_outgoing_weights``
            A sparse square matrix where each non-zero entry is the weight of an edge between a pair
            of cells or genes, where the sum of the weights of the outgoing edges for each element
            is 1 (there is always at least one such edge).

    Also return a pandas data frame of the similarities between the observations (metacells).

    **Computation Parameters**

    1. Invoke :py:func:`metacells.tools.high.find_top_feature_genes` using ``max_top_feature_genes``
       (default: {max_top_feature_genes}) to pick the feature genes to use to compute similarities
       between the metacells.

    2. Compute the fractions of each gene in each cell, and add the
       ``similarity_value_normalization`` (default: {similarity_value_normalization}) to
       it.

    3. If ``similarity_log_data`` (default: {similarity_log_data}), invoke the
       :py:func:`metacells.utilities.computation.log_data` function to compute the log (base 2) of
       the data.

    4. Invoke :py:func:`metacells.tools.similarity.compute_obs_obs_similarity` using
       ``similarity_method`` (default: {similarity_method}), ``logistics_location`` (default:
       {logistics_slope}) and ``logistics_slope`` (default: {logistics_slope}) and convert this
       to distances.

    5. Invoke :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph` using the distances,
       ``k`` (no default!), ``balanced_ranks_factor`` (default: {balanced_ranks_factor}),
       ``incoming_degree_factor`` (default: {incoming_degree_factor}), ``outgoing_degree_factor``
       (default: {outgoing_degree_factor}) to compute a "skeleton" graph to overlay on top of the
       UMAP graph.
    """
    tl.find_top_feature_genes(adata, max_genes=max_top_feature_genes)

    all_data = ut.get_vo_proper(adata, what, layout="row_major")
    all_fractions = ut.fraction_by(all_data, by="row")

    top_feature_genes_mask = ut.get_v_numpy(adata, "top_feature_gene")

    top_feature_genes_fractions = all_fractions[:, top_feature_genes_mask]
    top_feature_genes_fractions = ut.to_layout(top_feature_genes_fractions, layout="row_major")
    top_feature_genes_fractions = ut.to_numpy_matrix(top_feature_genes_fractions)

    top_feature_genes_fractions += similarity_value_normalization

    if similarity_log_data:
        top_feature_genes_fractions = ut.log_data(top_feature_genes_fractions, base=2)

    tdata = ut.slice(adata, vars=top_feature_genes_mask)
    similarities = tl.compute_obs_obs_similarity(
        tdata,
        top_feature_genes_fractions,
        method=similarity_method,
        reproducible=reproducible,
        logistics_location=logistics_location,
        logistics_slope=logistics_slope,
        inplace=False,
    )
    assert similarities is not None

    tl.compute_obs_obs_knn_graph(
        adata,
        similarities,
        k=k,
        balanced_ranks_factor=balanced_ranks_factor,
        incoming_degree_factor=incoming_degree_factor,
        outgoing_degree_factor=outgoing_degree_factor,
    )

    return similarities


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_umap_by_features(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    max_top_feature_genes: int = pr.max_top_feature_genes,
    similarity_value_normalization: float = pr.umap_similarity_value_normalization,
    similarity_log_data: bool = pr.umap_similarity_log_data,
    similarity_method: str = pr.umap_similarity_method,
    logistics_location: float = pr.logistics_location,
    logistics_slope: float = pr.logistics_slope,
    skeleton_k: int = pr.skeleton_k,
    balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    umap_k: int = pr.umap_k,
    dimensions: int = 2,
    min_dist: float = pr.umap_min_dist,
    spread: float = pr.umap_spread,
    random_seed: int = pr.random_seed,
) -> None:
    """
    Compute a UMAP projection of the (meta)cells.

    **Input**

    Annotated ``adata`` where each observation is a metacells and the variables are genes,
    are genes, where ``what`` is a per-variable-per-observation matrix or the name of a
    per-variable-per-observation annotation containing such a matrix.

    **Returns**

    Sets the following annotations in ``adata``:

    Variable (Gene) Annotations
        ``top_feature_gene``
            A boolean mask of the top feature genes used to compute similarities between the
            metacells.

    Observation-Observation (Metacell-Metacell) Annotations
        ``umap_distances``
            A sparse symmetric matrix of the graph of distances between the metacells.

    Observation (Metacell) Annotations
        ``umap_x``, ``umap_y``
            The X and Y coordinates of each metacell in the UMAP projection.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.pipeline.umap.compute_knn_by_features` using
       ``max_top_feature_genes`` (default: {max_top_feature_genes}),
       ``similarity_value_normalization`` (default: {similarity_value_normalization}),
       ``similarity_log_data`` (default: {similarity_log_data}), ``similarity_method`` (default:
       {similarity_method}), ``logistics_location`` (default: {logistics_location}),
       ``logistics_slope`` (default: {logistics_slope}), ``skeleton_k`` (default: {skeleton_k}),
       ``balanced_ranks_factor`` (default: {balanced_ranks_factor}), ``incoming_degree_factor``
       (default: {incoming_degree_factor}), ``outgoing_degree_factor`` (default:
       {outgoing_degree_factor}) to compute a "skeleton" graph to overlay on top of the UMAP graph.

    2. Invoke :py:func:`metacells.tools.layout.umap_by_distances` using the distances, ``umap_k``
       (default: {umap_k}), ``min_dist`` (default: {min_dist}), ``spread`` (default: {spread}),
       dimensions (default: {dimensions})

    .. note::

        Keep in mind that the KNN graph used by UMAP (controlled by ``umap_k``) is *not* identical to the KNN graph we
        compute (controlled by ``skeleton_k``). By default, we choose ``skeleton_k < umap_k``, as the purpose of the
        skeleton KNN is to highlight the "strong" structure of the data; in practice this strong skeleton is highly
        compatible with the structure used by UMAP, so it serves it purpose reasonably well. It would have been nice to
        make these compatible, but UMAP is not friendly towards dictating a KNN graph from the outside.
    """
    similarities = compute_knn_by_features(
        adata,
        what,
        max_top_feature_genes=max_top_feature_genes,
        similarity_value_normalization=similarity_value_normalization,
        similarity_log_data=similarity_log_data,
        similarity_method=similarity_method,
        logistics_location=logistics_location,
        logistics_slope=logistics_slope,
        k=skeleton_k,
        balanced_ranks_factor=balanced_ranks_factor,
        incoming_degree_factor=incoming_degree_factor,
        outgoing_degree_factor=outgoing_degree_factor,
        reproducible=(random_seed != 0),
    )

    distances = ut.to_numpy_matrix(similarities)
    distances *= -1
    distances += 1
    np.fill_diagonal(distances, 0.0)
    distances = sparse.csr_matrix(distances)

    ut.set_oo_data(adata, "umap_distances", distances)

    tl.umap_by_distances(
        adata, distances, k=umap_k, dimensions=dimensions, min_dist=min_dist, spread=spread, random_seed=random_seed
    )


def _build_igraph(edge_weights: ut.Matrix) -> Tuple[ig.Graph, ut.NumpyVector]:
    edge_weights = ut.to_proper_matrix(edge_weights)
    assert edge_weights.shape[0] == edge_weights.shape[1]
    size = edge_weights.shape[0]

    sources, targets = edge_weights.nonzero()
    weights_array = ut.to_numpy_vector(edge_weights[sources, targets]).astype("float64")

    graph = ig.Graph(directed=True)
    graph.add_vertices(size)
    graph.add_edges(list(zip(sources, targets)))
    graph.es["weight"] = weights_array

    return graph, weights_array
