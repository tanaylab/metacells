"""
Split
-----
"""

from re import Pattern
from typing import Collection
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

from .direct import compute_direct_metacells

__all__ = [
    "split_groups",
    "compute_groups_self_consistency",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def split_groups(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    group: str = "metacell",
    feature_downsample_min_samples: int = pr.feature_downsample_min_samples,
    feature_downsample_min_cell_quantile: float = pr.feature_downsample_min_cell_quantile,
    feature_downsample_max_cell_quantile: float = pr.feature_downsample_max_cell_quantile,
    feature_min_gene_total: Optional[int] = None,
    feature_min_gene_top3: Optional[int] = None,
    feature_min_gene_relative_variance: Optional[float] = pr.feature_min_gene_relative_variance,
    forbidden_gene_names: Optional[Collection[str]] = None,
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    cells_similarity_value_normalization: float = pr.cells_similarity_value_normalization,
    cells_similarity_log_data: bool = pr.cells_similarity_log_data,
    cells_similarity_method: str = pr.cells_similarity_method,
    max_cell_size: Optional[float] = pr.max_cell_size,
    max_cell_size_factor: Optional[float] = pr.max_cell_size_factor,
    knn_balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    knn_incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    knn_outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    min_seed_size_quantile: float = pr.min_seed_size_quantile,
    max_seed_size_quantile: float = pr.max_seed_size_quantile,
    candidates_cooldown_pass: float = pr.cooldown_pass,
    candidates_cooldown_node: float = pr.cooldown_node,
    random_seed: int = pr.random_seed,
) -> None:
    """
    Split each metacell into two parts using ``what`` (default: {what}) data.

    This creates a new partition of cells into half-metacells, which can used to
    :py:func:`compute_groups_self_consistency`.

    **Input**

    The input annotated ``adata`` is expected to contain a per-observation annotation named
    ``group`` (default: {group}) which identifies the group (metacells) each observation (cell)
    belongs to.

    **Returns**

    Sets the following annotations in ``adata``:

    Observation (Cell) Annotations
        ``half_<group>``
            The index of the half-group each cell belongs to. This is ``-1`` for ungrouped cells.
            Indices 0 to the number of groups are the first (low) halves; from the number of groups
            to twice that are the second (low) halves.

    **Computation Parameters**

    1. For each group (metacell), invoke
       :py:func:`metacells.pipeline.direct.compute_direct_metacells` on the observations (cells)
       included in the group, forcing the creation of two half-groups that cover all the group's
       cells. The parameters are passed to this call as-is, setting ``must_complete_cover`` to
       ``True`` (that is, disabling outliers detection), and disabling restrictions on the
       half-group sizes.
    """
    group_of_cells = ut.get_o_numpy(adata, group)
    groups_count = np.max(group_of_cells) + 1
    half_groups_of_cells = np.full(adata.n_obs, -1, dtype="int32")

    @ut.timed_call("split_group")
    def split_group(group_index: int) -> Tuple[ut.NumpyVector, ut.NumpyVector]:
        group_cells_mask = group_of_cells == group_index
        assert np.any(group_cells_mask)
        name = f".{group}-{group_index}/{groups_count}"
        gdata = ut.slice(adata, name=name, top_level=False, obs=group_cells_mask, track_obs="complete_cell_index")
        target_metacell_size = (gdata.n_obs + 1) // 2
        compute_direct_metacells(
            gdata,
            what,
            feature_downsample_min_samples=feature_downsample_min_samples,
            feature_downsample_min_cell_quantile=feature_downsample_min_cell_quantile,
            feature_downsample_max_cell_quantile=feature_downsample_max_cell_quantile,
            feature_min_gene_total=feature_min_gene_total,
            feature_min_gene_top3=feature_min_gene_top3,
            feature_min_gene_relative_variance=feature_min_gene_relative_variance,
            forbidden_gene_names=forbidden_gene_names,
            forbidden_gene_patterns=forbidden_gene_patterns,
            cells_similarity_value_normalization=cells_similarity_value_normalization,
            cells_similarity_log_data=cells_similarity_log_data,
            cells_similarity_method=cells_similarity_method,
            target_metacell_size=target_metacell_size,
            max_cell_size=max_cell_size,
            max_cell_size_factor=max_cell_size_factor,
            cell_sizes=None,
            knn_k=target_metacell_size,
            min_knn_k=target_metacell_size,
            knn_balanced_ranks_factor=knn_balanced_ranks_factor,
            knn_incoming_degree_factor=knn_incoming_degree_factor,
            knn_outgoing_degree_factor=knn_outgoing_degree_factor,
            min_seed_size_quantile=min_seed_size_quantile,
            max_seed_size_quantile=max_seed_size_quantile,
            candidates_cooldown_pass=candidates_cooldown_pass,
            candidates_cooldown_node=candidates_cooldown_node,
            candidates_min_split_size_factor=None,
            candidates_max_merge_size_factor=None,
            candidates_min_metacell_cells=1,
            must_complete_cover=True,
            random_seed=random_seed,
        )
        direct_groups = ut.get_o_numpy(gdata, "metacell")
        zero_count = np.sum(direct_groups == 0)
        one_count = np.sum(direct_groups == 1)
        ut.log_calc(f"group: {group_index} size: {len(direct_groups)} " f"split into: {zero_count} + {one_count}")
        assert zero_count + one_count == len(direct_groups)
        assert zero_count > 0
        assert one_count > 0
        return (group_cells_mask, group_index + groups_count * direct_groups)

    for (group_cells_mask, group_cells_halves) in ut.parallel_map(split_group, groups_count):
        half_groups_of_cells[group_cells_mask] = group_cells_halves

    ut.set_o_data(adata, f"half_{group}", half_groups_of_cells, formatter=ut.groups_description)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_groups_self_consistency(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    group: str = "metacell",
    genes_mask: Optional[ut.NumpyVector] = None,
    self_similarity_log_data: bool = pr.self_similarity_log_data,
    self_similarity_value_normalization: float = pr.self_similarity_value_normalization,
    self_similarity_method: str = pr.self_similarity_method,
    reproducible: bool = pr.reproducible,
    logistics_location: float = pr.logistics_location,
    logistics_slope: float = pr.logistics_slope,
) -> ut.NumpyVector:
    """
    Compute the self consistency (similarity between two halves) of some groups.

    **Input**

    The input annotated ``adata`` is expected to contain a per-observation annotation named
    ``group`` (default: {group}) which identifies the group (metacells) each observation (cell)
    belongs to, and ``half_<group>`` which identifies the half-group each observation belongs
    to (e.g. as computed by :py:func:`split_groups`). Specifically, the indices of the halves
    of group index ``i`` are ``i`` and ``i + groups_count``.

    **Returns**

    A Numpy vector holding, for each group, the similarity between its two halves.

    **Computation Parameters**

    1. For each group, compute the sum of values in each half and normalize it to fractions (sum of 1).

    2. If ``genes_mask`` is specified, select only the genes specified in it. Note the sum of the
       fractions of the genes of each group in the result will be less than or equal to 1.

    3. If ``self_similarity_log_data`` (default: {self_similarity_log_data}), log2 the values using
       ``self_similarity_value_normalization`` (default: {self_similarity_value_normalization}).

    4. Compute the ``self_similarity_method`` (default: {self_similarity_method}) between the two
       halves. If this is the ``logistics`` similarity, then this will use ``logistics_location``
       (default: {logistics_location}) and ``logistics_slope`` (default: {logistics_slope}). If this
       is ``pearson``, and if ``reproducible`` (default: {reproducible}) is ``True``, a slower
       (still parallel) but reproducible algorithm will be used to compute Pearson correlations.
    """
    hdata = tl.group_obs_data(adata, what, groups=f"half_{group}", name=".halves")
    assert hdata is not None

    sum_of_halves = ut.get_o_numpy(hdata, f"{what}|sum")
    halves_values = ut.to_numpy_matrix(ut.get_vo_proper(hdata, what, layout="row_major"))
    halves_data = ut.mustbe_numpy_matrix(ut.scale_by(halves_values, sum_of_halves, by="row"))

    if self_similarity_value_normalization > 0:
        halves_data += self_similarity_value_normalization

    if self_similarity_log_data:
        halves_data = ut.log_data(halves_data, base=2)

    if genes_mask is not None:
        halves_data = halves_data[:, genes_mask]

    assert hdata.n_obs % 2 == 0
    groups_count = hdata.n_obs // 2
    low_half_indices = np.arange(groups_count)
    high_half_indices = low_half_indices + groups_count

    low_half_data = halves_data[low_half_indices, :]
    high_half_data = halves_data[high_half_indices, :]

    assert self_similarity_method in ("logistics", "pearson")
    if self_similarity_method == "logistics":
        similarity = ut.pairs_logistics_rows(
            low_half_data, high_half_data, location=logistics_location, slope=logistics_slope
        )
        similarity *= -1
        similarity += 1
    else:
        similarity = ut.pairs_corrcoef_rows(low_half_data, high_half_data, reproducible=reproducible)

    return similarity
