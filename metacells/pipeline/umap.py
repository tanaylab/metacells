"""
UMAP
----

It is useful to project the metacells data to a 2D scatter plot (where each point is a metacell).
The steps provided here are expected to yield a reasonable such projection, though as always you
might need to tweak the parameters or even the overall flow for specific data sets.
"""

from re import Pattern
from typing import Collection
from typing import Optional
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
    "compute_knn_by_markers",
    "compute_umap_by_markers",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_knn_by_markers(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    marker_gene_names: Optional[Collection[str]] = None,
    marker_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    max_marker_genes: Optional[int] = pr.umap_max_marker_genes,
    ignore_lateral_genes: bool = pr.umap_ignore_lateral_genes,
    similarity_value_regularization: float = pr.umap_similarity_value_regularization,
    similarity_log_data: bool = pr.umap_similarity_log_data,
    similarity_method: str = pr.umap_similarity_method,
    logistics_location: float = pr.logistics_location,
    logistics_slope: float = pr.logistics_slope,
    k: int,
    balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    min_outgoing_degree: int = pr.markers_knn_min_outgoing_degree,
    reproducible: bool,
) -> ut.PandasFrame:
    """
    Compute KNN graph between metacells based on marker genes.

    If ``reproducible`` is ``True``, a slower (still parallel) but
    reproducible algorithm will be used to compute Pearson correlations.

    **Input**

    Annotated ``adata`` where each observation is a metacells and the variables are genes, are genes, where ``what`` is
    a per-variable-per-observation matrix or the name of a per-variable-per-observation annotation containing such a
    matrix. Should contain a ``marker_gene`` mask unless explicitly specifying the marker genes.

    **Returns**

    Sets the following in ``adata``:

    Observations-Pair (Metacells) Annotations
        ``obs_outgoing_weights``
            A sparse square matrix where each non-zero entry is the weight of an edge between a pair
            of cells or genes, where the sum of the weights of the outgoing edges for each element
            is 1 (there is always at least one such edge).

    Also return a pandas data frame of the similarities between the observations (metacells).

    **Computation Parameters**

    1. If ``marker_gene_names`` and/or ``marker_gene_patterns`` were specified, use the matching genes.
       Otherwise, use the ``marker_gene`` mask.

    2. If ``ignore_lateral_genes`` (default: {ignore_lateral_genes}), then remove any genes marked as lateral
       from the mask.

    3. If ``max_marker_genes`` (default: {max_marker_genes}) is not ``None``, then pick this number
       of marker genes with the highest variance.

    4. Compute the fractions of each ``marker_gene`` in each cell, and add the
       ``similarity_value_regularization`` (default: {similarity_value_regularization}) to it.

    5. If ``similarity_log_data`` (default: {similarity_log_data}), invoke the
       :py:func:`metacells.utilities.computation.log_data` function to compute the log (base 2) of
       the data.

    6. Invoke :py:func:`metacells.tools.similarity.compute_obs_obs_similarity` using
       ``similarity_method`` (default: {similarity_method}), ``logistics_location`` (default:
       {logistics_slope}) and ``logistics_slope`` (default: {logistics_slope}) and convert this
       to distances.

    7. Invoke :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph` using the distances,
       ``k`` (no default!), ``balanced_ranks_factor`` (default: {balanced_ranks_factor}),
       ``incoming_degree_factor`` (default: {incoming_degree_factor}), ``outgoing_degree_factor``
       (default: {outgoing_degree_factor}) and ``min_outgoing_degree`` (default: {min_outgoing_degree})
       to compute a "skeleton" graph to overlay on top of the UMAP graph.
    """
    assert max_marker_genes is None or max_marker_genes > 0
    if marker_gene_names is None and marker_gene_patterns is None:
        marker_genes_mask = ut.get_v_numpy(adata, "marker_gene")
    else:
        marker_genes_series = tl.find_named_genes(adata, names=marker_gene_names, patterns=marker_gene_patterns)
        assert marker_genes_series is not None
        marker_genes_mask = ut.to_numpy_vector(marker_genes_series)

    if ignore_lateral_genes:
        lateral_genes_mask = ut.get_v_numpy(adata, "lateral_gene")
        marker_genes_mask = marker_genes_mask & ~lateral_genes_mask
        ut.log_calc("marker_genes_mask", marker_genes_mask)

    all_fractions = ut.get_vo_proper(adata, what, layout="row_major")

    index_per_marker_gene = np.where(marker_genes_mask)[0]
    fraction_per_metacell_per_marker_gene = ut.to_numpy_matrix(all_fractions[:, index_per_marker_gene])

    if max_marker_genes is not None and max_marker_genes < len(index_per_marker_gene):
        fraction_per_metacell_per_marker_gene = ut.to_layout(
            fraction_per_metacell_per_marker_gene, layout="column_major"
        )
        variance_per_marker_gene = ut.variance_per(fraction_per_metacell_per_marker_gene, per="column")
        variance_per_marker_gene *= -1
        chosen_positions = np.argpartition(variance_per_marker_gene, max_marker_genes)[:max_marker_genes]
        ut.log_calc("chosen_positions", chosen_positions)
        index_per_marker_gene = index_per_marker_gene[chosen_positions]
        fraction_per_metacell_per_marker_gene = fraction_per_metacell_per_marker_gene[:, chosen_positions]

    fraction_per_metacell_per_marker_gene = ut.to_layout(fraction_per_metacell_per_marker_gene, layout="row_major")
    fraction_per_metacell_per_marker_gene += similarity_value_regularization

    if similarity_log_data:
        fraction_per_metacell_per_marker_gene = ut.log_data(fraction_per_metacell_per_marker_gene, base=2)

    tdata = ut.slice(adata, vars=index_per_marker_gene)
    similarities = tl.compute_obs_obs_similarity(
        tdata,
        fraction_per_metacell_per_marker_gene,
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
        min_outgoing_degree=min_outgoing_degree,
    )

    return similarities


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_umap_by_markers(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    marker_gene_names: Optional[Collection[str]] = None,
    marker_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    max_marker_genes: Optional[int] = pr.umap_max_marker_genes,
    ignore_lateral_genes: bool = pr.umap_ignore_lateral_genes,
    similarity_value_regularization: float = pr.umap_similarity_value_regularization,
    similarity_log_data: bool = pr.umap_similarity_log_data,
    similarity_method: str = pr.umap_similarity_method,
    logistics_location: float = pr.logistics_location,
    logistics_slope: float = pr.logistics_slope,
    skeleton_k: int = pr.skeleton_k,
    balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    min_outgoing_degree: int = pr.markers_knn_min_outgoing_degree,
    umap_k: int = pr.umap_k,
    dimensions: int = 2,
    min_dist: float = pr.umap_min_dist,
    spread: float = pr.umap_spread,
    random_seed: int,
) -> None:
    """
    Compute a UMAP projection of the (meta)cells.

    **Input**

    Annotated ``adata`` where each observation is a metacells and the variables are genes, are genes, where ``what`` is
    a per-variable-per-observation matrix or the name of a per-variable-per-observation annotation containing such a
    matrix. Should contain a ``marker_gene`` mask unless explicitly specifying the marker genes.

    **Returns**

    Sets the following annotations in ``adata``:

    Observation-Observation (Metacell-Metacell) Annotations
        ``umap_distances``
            A sparse symmetric matrix of the graph of distances between the metacells.

    Observation (Metacell) Annotations
        ``x``, ``y`` (if ``dimensions`` is 2)
            The X and Y coordinates of each metacell in the UMAP projection.

        ``u``, ``v``, ``w`` (if ``dimensions`` is 3)
            The U, V, W coordinates of each metacell in the UMAP projection.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.pipeline.umap.compute_knn_by_markers` using
       ``marker_gene_names`` (default: {marker_gene_names}), ``marker_gene_patterns`` (default: {marker_gene_patterns}),
       ``max_marker_genes`` (default: {max_marker_genes}), ``ignore_lateral_genes`` (default: {ignore_lateral_genes}),
       ``similarity_value_regularization`` (default: {similarity_value_regularization}), ``similarity_log_data``
       (default: {similarity_log_data}), ``similarity_method`` (default: {similarity_method}), ``logistics_location``
       (default: {logistics_location}), ``logistics_slope`` (default: {logistics_slope}), ``skeleton_k`` (default:
       {skeleton_k}), ``balanced_ranks_factor`` (default: {balanced_ranks_factor}), ``incoming_degree_factor`` (default:
       {incoming_degree_factor}) and ``outgoing_degree_factor`` (default: {outgoing_degree_factor}) and
       ``min_outgoing_degree`` (default: {min_outgoing_degree}) to compute a "skeleton" graph to overlay on top of the
       UMAP graph. Specify a non-zero ``random_seed`` to make this reproducible.

    2. Invoke :py:func:`metacells.tools.layout.umap_by_distances` using the distances, ``umap_k``
       (default: {umap_k}), ``min_dist`` (default: {min_dist}), ``spread`` (default: {spread}),
       dimensions (default: {dimensions}) and ``random_seed``.

    .. note::

        Keep in mind that the KNN graph used by UMAP (controlled by ``umap_k``) is *not* identical to the KNN graph we
        compute (controlled by ``skeleton_k``). By default, we choose ``skeleton_k < umap_k``, as the purpose of the
        skeleton KNN is to highlight the "strong" structure of the data; in practice this strong skeleton is highly
        compatible with the structure used by UMAP, so it serves it purpose reasonably well. It would have been nice to
        make these compatible, but UMAP is not friendly towards dictating a KNN graph from the outside.
    """
    similarities = compute_knn_by_markers(
        adata,
        what,
        marker_gene_names=marker_gene_names,
        marker_gene_patterns=marker_gene_patterns,
        max_marker_genes=max_marker_genes,
        ignore_lateral_genes=ignore_lateral_genes,
        similarity_value_regularization=similarity_value_regularization,
        similarity_log_data=similarity_log_data,
        similarity_method=similarity_method,
        logistics_location=logistics_location,
        logistics_slope=logistics_slope,
        k=skeleton_k,
        balanced_ranks_factor=balanced_ranks_factor,
        incoming_degree_factor=incoming_degree_factor,
        outgoing_degree_factor=outgoing_degree_factor,
        min_outgoing_degree=min_outgoing_degree,
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
