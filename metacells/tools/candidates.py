"""
Candidates
----------
"""

import sys
from math import ceil
from math import floor
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.utilities as ut

if "sphinx" not in sys.argv[0]:
    import metacells.extensions as xt  # type: ignore

__all__ = [
    "compute_candidate_metacells",
    "choose_seeds",
    "optimize_partitions",
    "score_partitions",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_candidate_metacells(  # pylint: disable=too-many-statements
    adata: AnnData,
    what: Union[str, ut.Matrix] = "obs_outgoing_weights",
    *,
    target_metacell_size: int = pr.target_metacell_size,
    min_metacell_size: int = pr.min_metacell_size,
    target_metacell_umis: int = pr.target_metacell_umis,
    cell_umis: Optional[ut.NumpyVector] = pr.cell_umis,
    min_seed_size_quantile: float = pr.min_seed_size_quantile,
    max_seed_size_quantile: float = pr.max_seed_size_quantile,
    cooldown_pass: float = pr.cooldown_pass,
    cooldown_node: float = pr.cooldown_node,
    cooldown_phase: float = pr.cooldown_phase,
    increase_phase: float = 1.01,
    min_split_size_factor: float = pr.candidates_min_split_size_factor,
    max_merge_size_factor: float = pr.candidates_max_merge_size_factor,
    max_split_min_cut_strength: float = pr.max_split_min_cut_strength,
    min_cut_seed_cells: int = pr.min_cut_seed_cells,
    must_complete_cover: bool = False,
    random_seed: int,
    inplace: bool = True,
) -> Optional[ut.PandasSeries]:
    """
    Assign observations (cells) to (raw, candidate) metacells based on ``what`` data. (a weighted
    directed graph).

    These candidate metacells typically go through additional vetting (e.g. deviant detection and
    dissolving too-small metacells) to obtain the final metacells.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-observation-per-observation matrix where each row is the outgoing weights from
    each observation to the rest, or just the name of a per-observation-per-observation annotation
    containing such a matrix. Typically this matrix will be sparse for efficient processing.

    **Returns**

    Observation (Cell) Annotations
        ``candidate``
            The integer index of the (raw, candidate) metacell each cell belongs to. The metacells
            are in no particular order.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the variable names).

    **Computation Parameters**

    0. If ``cell_umis`` is not specified, use the sum of the ``what`` data for each cell.

    1. We are trying to create metacells of size ``target_metacell_size`` cells and ``target_metacell_umis`` UMIs each.
       Compute the UMIs of the metacells by summing the ``cell_umis``.

    2. We start with some an assignment of cells to seeds using :py:func:`choose_seeds` using ``min_seed_size_quantile``
       (default: {min_seed_size_quantile}) and ``max_seed_size_quantile`` (default: {max_seed_size_quantile}) to compute
       them, picking a number of seeds such that the average metacell size would match the target.

    3. We optimize the seeds using :py:func:`optimize_partitions` to obtain initial communities by
       maximizing the "stability" of the solution (probability of starting at a random node and
       moving either forward or backward in the graph and staying within the same metacell, divided
       by the probability of staying in the metacell if the edges connected random nodes). We pass
       it the ``cooldown_pass`` {cooldown_pass}) and ``cooldown_node`` (default: {cooldown_node}).

    4. If ``min_split_size_factor`` (default: {min_split_size_factor}) is specified, split to two each community whose
       size is partition method on each community whose size is at least
       ``target_metacell_size * min_split_size_factor`` or whose UMIs are at least
       ``target_metacell_umis * min_split_size_factor``, as long as half of the community is at least the
       ``min_metacell_size`` (default: {min_metacell_size}). Then, re-optimize the solution (resulting in an additional
       metacells). Every time we re-optimize, we multiply 1 - ``cooldown_pass`` by 1 - ``cooldown_phase`` (default:
       {cooldown_phase}).

    5. Using ``max_split_min_cut_strength`` (default: {max_split_min_cut_strength}), if
       the minimal cut of a candidate is lower, split it into two. If one of the partitions is
       smaller than ``min_cut_seed_cells``, then mark the cells in it as outliers, or if
       ``must_complete_cover`` is ``True``, skip the cut altogether.

    6. Using ``max_merge_size_factor`` (default: {max_merge_size_factor}) and ``min_metacell_size``
       (default: {min_metacell_size}), make outliers of cells of a community whose size is at most
       ``target_metacell_size * max_merge_size_factor`` and whose UMIs are at most
       ``target_metacell_umis * max_merge_size_factor``, or that contain less cells than ``min_metacell_size``. Again,
       re-optimize, which will assign these cells to other metacells (resulting on one less metacell). We again apply
       the ``cooldown_phase`` every time we re-optimize.

    7. Repeat the above steps until all metacells candidates are in the acceptable size range.
    """
    assert 0.0 < max_merge_size_factor < min_split_size_factor
    assert target_metacell_size > 0
    assert target_metacell_umis > 0
    assert 0.0 < cooldown_pass < 1.0
    assert 0.0 <= cooldown_node <= 1.0
    assert 0.0 < cooldown_phase <= 1.0

    edge_weights = ut.get_oo_proper(adata, what, layout="row_major")
    assert edge_weights.shape[0] == edge_weights.shape[1]
    size = edge_weights.shape[0]

    outgoing_edge_weights = ut.mustbe_compressed_matrix(edge_weights)

    assert ut.is_layout(outgoing_edge_weights, "row_major")
    incoming_edge_weights = ut.mustbe_compressed_matrix(ut.to_layout(outgoing_edge_weights, layout="column_major"))
    assert ut.is_layout(incoming_edge_weights, "column_major")

    assert outgoing_edge_weights.data.dtype == "float32"
    assert outgoing_edge_weights.indices.dtype == "int32"
    assert outgoing_edge_weights.indptr.dtype == "int32"
    assert incoming_edge_weights.data.dtype == "float32"
    assert incoming_edge_weights.indices.dtype == "int32"
    assert incoming_edge_weights.indptr.dtype == "int32"

    if cell_umis is None:
        cell_umis = ut.get_o_numpy(adata, what, sum=True, formatter=ut.sizes_description).astype("float32")
    else:
        assert cell_umis.dtype == "float32"
    assert isinstance(cell_umis, ut.NumpyVector)

    lowest_metacell_size = min_metacell_size
    min_metacell_umis = int(floor(target_metacell_umis * max_merge_size_factor))
    ut.log_calc("min_metacell_umis", min_metacell_umis)
    max_metacell_umis = int(ceil(target_metacell_umis * min_split_size_factor)) - 1
    ut.log_calc("max_metacell_umis", max_metacell_umis)

    min_metacell_size = max(min_metacell_size, int(floor(target_metacell_size * max_merge_size_factor)))
    ut.log_calc("min_metacell_size", min_metacell_size)
    max_metacell_size = max(int(ceil(target_metacell_size * min_split_size_factor)) - 1, min_metacell_size + 1)
    ut.log_calc("max_metacell_size", max_metacell_size)

    total_umis = sum(cell_umis)
    ut.log_calc("total_umis", total_umis)

    initial_seeds_count = _seeds_count_for(
        total_size=size,
        total_umis=total_umis,
        min_metacell_size=min_metacell_size,
        max_metacell_size=max_metacell_size,
        min_metacell_umis=min_metacell_umis,
        max_metacell_umis=max_metacell_umis,
    )

    community_of_nodes = np.full(size, -1, dtype="int32")
    initial_seeds_count = _choose_seeds(
        outgoing_edge_weights=outgoing_edge_weights,
        incoming_edge_weights=incoming_edge_weights,
        seed_of_cells=community_of_nodes,
        max_seeds_count=initial_seeds_count,
        min_seed_size_quantile=min_seed_size_quantile,
        max_seed_size_quantile=max_seed_size_quantile,
        random_seed=random_seed,
    )

    ut.set_o_data(adata, "seed", community_of_nodes, formatter=ut.groups_description)
    community_of_nodes = community_of_nodes.copy()

    np.random.seed(random_seed)

    cold_temperature = 1 - cooldown_pass

    old_score = 1e9
    old_communities = community_of_nodes
    old_small_nodes_count = len(community_of_nodes)
    atomic_candidates: Set[Tuple[int, ...]] = set()
    hot_communities = list(range(initial_seeds_count))

    while True:
        cold_temperature, score = _reduce_communities(
            outgoing_edge_weights=outgoing_edge_weights,
            incoming_edge_weights=incoming_edge_weights,
            community_of_nodes=community_of_nodes,
            hot_communities=hot_communities,
            target_metacell_umis=target_metacell_umis,
            node_umis=cell_umis,
            min_metacell_umis=min_metacell_umis,
            max_metacell_umis=max_metacell_umis,
            target_metacell_size=target_metacell_size,
            min_metacell_size=min_metacell_size,
            max_metacell_size=max_metacell_size,
            max_split_min_cut_strength=max_split_min_cut_strength,
            min_cut_seed_cells=min_cut_seed_cells,
            must_complete_cover=must_complete_cover,
            min_seed_size_quantile=min_seed_size_quantile,
            max_seed_size_quantile=max_seed_size_quantile,
            random_seed=random_seed,
            cooldown_pass=cooldown_pass,
            cooldown_node=cooldown_node,
            cooldown_phase=cooldown_phase,
            increase_phase=increase_phase,
            cold_temperature=cold_temperature,
            atomic_candidates=atomic_candidates,
        )

        small_communities, small_nodes_count, small_nodes_umis = _find_small_communities(
            community_of_nodes=community_of_nodes,
            node_umis=cell_umis,
            min_metacell_umis=min_metacell_umis,
            max_metacell_umis=max_metacell_umis,
            min_metacell_size=min_metacell_size,
            max_metacell_size=max_metacell_size,
            lowest_metacell_size=lowest_metacell_size,
        )

        small_communities_count = len(small_communities)
        ut.log_calc("small communities count", small_communities_count)
        if small_communities_count < 2:
            break

        if (old_small_nodes_count, old_score) <= (small_nodes_count, score):
            ut.logger().debug("is not better, revert")
            community_of_nodes = old_communities
            score = old_score
            ut.log_calc("communities", community_of_nodes, formatter=ut.groups_description)
            ut.log_calc("score", score)
            break

        old_score = score
        old_communities = community_of_nodes.copy()
        old_small_nodes_count = small_nodes_count

        kept_communities_count = _cancel_communities(
            community_of_nodes=community_of_nodes, cancelled_communities=small_communities
        )

        small_communities_seeds = min(
            small_communities_count - 1,
            _seeds_count_for(
                total_size=small_nodes_count,
                total_umis=small_nodes_umis,
                min_metacell_size=min_metacell_size,
                max_metacell_size=max_metacell_size,
                min_metacell_umis=min_metacell_umis,
                max_metacell_umis=max_metacell_umis,
            ),
        )
        ut.log_calc("small communities seeds", small_communities_seeds)

        max_seeds_count = kept_communities_count + small_communities_seeds
        seeds_count = _choose_seeds(
            outgoing_edge_weights=outgoing_edge_weights,
            incoming_edge_weights=incoming_edge_weights,
            seed_of_cells=community_of_nodes,
            max_seeds_count=max_seeds_count,
            min_seed_size_quantile=min_seed_size_quantile,
            max_seed_size_quantile=max_seed_size_quantile,
            random_seed=random_seed,
        )

        hot_communities = list(range(kept_communities_count, seeds_count))

    assert np.all(community_of_nodes >= 0)

    if inplace:
        ut.set_o_data(adata, "candidate", community_of_nodes, formatter=ut.groups_description)
        return None

    ut.log_return("candidate", community_of_nodes, formatter=ut.groups_description)
    return ut.to_pandas_series(community_of_nodes, index=adata.obs_names)


@ut.logged()
def _seeds_count_for(
    *,
    total_umis: float,
    total_size: float,
    min_metacell_size: float,
    max_metacell_size: float,
    min_metacell_umis: float,
    max_metacell_umis: float,
) -> int:
    min_seeds_count_by_umis = int(ceil(total_umis / max_metacell_umis))
    ut.log_calc("min_seeds_count_by_umis", min_seeds_count_by_umis)
    max_seeds_count_by_umis = int(ceil(total_umis / min_metacell_umis))
    ut.log_calc("max_seeds_count_by_umis", max_seeds_count_by_umis)

    min_seeds_count_by_size = int(ceil(total_size / max_metacell_size))
    ut.log_calc("min_seeds_count_by_size", min_seeds_count_by_size)
    max_seeds_count_by_size = int(ceil(total_size / min_metacell_size))
    ut.log_calc("max_seeds_count_by_size", max_seeds_count_by_size)

    if max_seeds_count_by_size < min_seeds_count_by_umis:
        seeds_count = int(round((max_seeds_count_by_size + min_seeds_count_by_umis) / 2))
    elif max_seeds_count_by_umis < min_seeds_count_by_size:
        seeds_count = int(round((max_seeds_count_by_umis + min_seeds_count_by_size) / 2))
    else:
        min_seeds_count = max(min_seeds_count_by_size, min_seeds_count_by_umis)
        max_seeds_count = min(max_seeds_count_by_size, max_seeds_count_by_umis)
        seeds_count = int(round((min_seeds_count + max_seeds_count) / 2))
    ut.log_return("seeds_count", seeds_count)
    return seeds_count


def _reduce_communities(
    outgoing_edge_weights: ut.CompressedMatrix,
    incoming_edge_weights: ut.CompressedMatrix,
    community_of_nodes: ut.NumpyVector,
    hot_communities: List[int],
    target_metacell_umis: float,
    node_umis: ut.NumpyVector,
    min_metacell_umis: float,
    max_metacell_umis: float,
    target_metacell_size: float,
    min_metacell_size: float,
    max_metacell_size: float,
    max_split_min_cut_strength: float,
    min_cut_seed_cells: int,
    must_complete_cover: bool,
    min_seed_size_quantile: float,
    max_seed_size_quantile: float,
    random_seed: int,
    cooldown_pass: float,
    cooldown_node: float,
    cooldown_phase: float,
    increase_phase: float,
    cold_temperature: float,
    atomic_candidates: Set[Tuple[int, ...]],
) -> Tuple[float, float]:
    np.random.seed(random_seed)
    while True:
        ut.log_calc("cold_temperature", cold_temperature)
        score = _optimize_partitions(
            outgoing_edge_weights=outgoing_edge_weights,
            incoming_edge_weights=incoming_edge_weights,
            community_of_nodes=community_of_nodes,
            hot_communities=hot_communities,
            node_umis=node_umis,
            low_partition_umis=min_metacell_umis,
            target_partition_umis=target_metacell_umis,
            high_partition_umis=max_metacell_umis,
            low_partition_size=min_metacell_size,
            target_partition_size=target_metacell_size,
            high_partition_size=max_metacell_size,
            random_seed=random_seed,
            cooldown_pass=cooldown_pass,
            cooldown_node=cooldown_node,
            cold_temperature=cold_temperature,
        )

        cold_temperature = cold_temperature * (1 - cooldown_phase)
        max_metacell_size *= increase_phase
        ut.log_calc("communities", community_of_nodes, formatter=ut.groups_description)
        ut.log_calc("max_metacell_size", max_metacell_size)

        split_communities_count, cut_communities_count, hot_communities = _cut_split_communities(
            outgoing_edge_weights=outgoing_edge_weights,
            community_of_nodes=community_of_nodes,
            target_metacell_umis=target_metacell_umis,
            node_umis=node_umis,
            min_metacell_umis=min_metacell_umis,
            max_metacell_umis=max_metacell_umis,
            target_metacell_size=target_metacell_size,
            min_metacell_size=min_metacell_size,
            max_metacell_size=max_metacell_size,
            max_split_min_cut_strength=max_split_min_cut_strength,
            min_cut_seed_cells=min_cut_seed_cells,
            must_complete_cover=must_complete_cover,
            atomic_candidates=atomic_candidates,
        )

        if split_communities_count + cut_communities_count == 0:
            return cold_temperature, score

        max_seeds_count = np.max(community_of_nodes) + 1
        if cut_communities_count > 0:
            hot_communities.append(max_seeds_count)
            max_seeds_count += 1

        _choose_seeds(
            outgoing_edge_weights=outgoing_edge_weights,
            incoming_edge_weights=incoming_edge_weights,
            seed_of_cells=community_of_nodes,
            max_seeds_count=max_seeds_count,
            min_seed_size_quantile=min_seed_size_quantile,
            max_seed_size_quantile=max_seed_size_quantile,
            random_seed=random_seed,
        )


def _cut_split_communities(  # pylint: disable=too-many-branches,too-many-statements
    *,
    outgoing_edge_weights: ut.CompressedMatrix,
    community_of_nodes: ut.NumpyVector,
    target_metacell_umis: float,
    node_umis: ut.NumpyVector,
    min_metacell_umis: float,
    max_metacell_umis: float,
    target_metacell_size: float,
    min_metacell_size: float,
    max_metacell_size: float,
    max_split_min_cut_strength: float,
    min_cut_seed_cells: int,
    must_complete_cover: bool,
    atomic_candidates: Set[Tuple[int, ...]],
) -> Tuple[int, int, List[int]]:
    communities_count = np.max(community_of_nodes) + 1
    assert communities_count > 0

    split_communities_count = 0
    cut_communities_count = 0
    next_new_community_index = communities_count

    hot_communities = []

    for community_index in range(communities_count):
        community_mask = community_of_nodes == community_index
        community_size = np.sum(community_mask)
        community_umis = np.sum(node_umis[community_mask])

        if (community_umis > max_metacell_umis and community_size >= 3 * min_metacell_size) or (
            community_size > max_metacell_size and community_umis >= 3 * min_metacell_umis
        ):
            community_indices = np.where(community_mask)[0]
            split_parts_count = max(
                2,
                min(
                    int(round(community_umis / target_metacell_umis)), int(round(community_size / target_metacell_size))
                ),
            )
            ut.logger().debug(
                "community: %s size: %s umis: %s is too large, split into %s",
                community_index,
                community_size,
                community_umis,
                split_parts_count,
            )
            all_parts_exist = False
            while not all_parts_exist:
                split_parts_assignment = np.random.randint(  # type: ignore
                    split_parts_count, size=len(community_indices)
                )
                all_parts_exist = True
                for split_index in range(split_parts_count):
                    if not np.any(split_parts_assignment == split_index):
                        all_parts_exist = False
                        break
            community_of_nodes[community_indices] = -1
            for split_index in range(split_parts_count):
                split_part_mask = split_parts_assignment == split_index
                assert np.any(split_part_mask)
                split_part_indices = community_indices[split_part_mask]
                if split_index == 0:
                    hot_communities.append(community_index)
                    community_of_nodes[split_part_indices[np.random.randint(len(split_part_indices))]] = community_index
                else:
                    hot_communities.append(next_new_community_index)
                    community_of_nodes[
                        split_part_indices[np.random.randint(len(split_part_indices))]
                    ] = next_new_community_index
                    next_new_community_index += 1
                    split_communities_count += 1
            continue

    if split_communities_count == 0:
        for community_index in range(communities_count):
            community_mask = community_of_nodes == community_index
            community_indices_key = tuple(np.where(community_mask)[0])
            if community_indices_key in atomic_candidates:
                continue

            action = _min_cut_community(
                outgoing_edge_weights=outgoing_edge_weights,
                community_of_nodes=community_of_nodes,
                cut_community_index=community_index,
                max_split_min_cut_strength=max_split_min_cut_strength,
                min_cut_seed_cells=min_cut_seed_cells,
                must_complete_cover=must_complete_cover,
                new_community_index=next_new_community_index,
            )

            if action == "unchanged":
                atomic_candidates.add(community_indices_key)
                continue

            hot_communities.append(community_index)
            ut.logger().debug(
                "community: %s nodes: %s size: %s was %s",
                community_index,
                np.sum(community_mask),
                community_umis,
                action,
            )

            if action == "split":
                hot_communities.append(next_new_community_index)
                split_communities_count += 1
                next_new_community_index += 1
            else:
                cut_communities_count += 1

    if split_communities_count + cut_communities_count > 0:
        ut.logger().debug("old communities: %s", communities_count)
        ut.logger().debug("cut communities: %s", cut_communities_count)
        ut.logger().debug("split communities: %s", split_communities_count)
        ut.logger().debug("hot communities: %s", len(hot_communities))
        ut.logger().debug("total communities: %s", communities_count + split_communities_count)
    else:
        ut.logger().debug("no communities were cut or split")

    return (split_communities_count, cut_communities_count, hot_communities)


def _min_cut_community(
    outgoing_edge_weights: ut.CompressedMatrix,
    community_of_nodes: ut.NumpyVector,
    cut_community_index: int,
    max_split_min_cut_strength: float,
    min_cut_seed_cells: int,
    must_complete_cover: bool,
    new_community_index: int,
) -> str:
    community_mask = community_of_nodes == cut_community_index
    if np.sum(community_mask) < 2:
        return "unchanged"
    community_edge_weights = outgoing_edge_weights[community_mask, :][:, community_mask]
    community_edge_weights += community_edge_weights.T

    cut, cut_strength = ut.min_cut(community_edge_weights)
    if cut_strength is None:
        return "unchanged"

    if cut_strength == 0:
        community_indices = np.where(community_mask)[0]
        if len(cut.partition[0]) < len(cut.partition[1]):
            small_partition = 0
        else:
            small_partition = 1
        small_community_indices = community_indices[cut.partition[small_partition]]

    if cut_strength > max_split_min_cut_strength:
        return "unchanged"

    ut.logger().debug(
        "min cut community: %s partitions: %s + %s = %s strength: %s",
        cut_community_index,
        len(cut.partition[0]),
        len(cut.partition[1]),
        community_edge_weights.shape[0],
        cut_strength,
    )

    community_indices = np.where(community_mask)[0]
    if len(cut.partition[0]) < len(cut.partition[1]):
        small_partition = 0
    else:
        small_partition = 1

    if len(cut.partition[small_partition]) >= min_cut_seed_cells:
        second_partition_indices = community_indices[cut.partition[1]]
        community_of_nodes[second_partition_indices] = new_community_index
        return "split"

    if must_complete_cover and cut_strength > 0:
        ut.logger().debug("give up on small cut: %s", len(cut.partition[small_partition]))
        return "unchanged"

    small_community_indices = community_indices[cut.partition[small_partition]]
    community_of_nodes[small_community_indices] = -1
    return "cut"


def _find_small_communities(
    *,
    community_of_nodes: ut.NumpyVector,
    node_umis: ut.NumpyVector,
    min_metacell_umis: float,
    max_metacell_umis: float,
    min_metacell_size: float,
    max_metacell_size: float,
    lowest_metacell_size: float,
) -> Tuple[Set[int], int, float]:
    communities_count = np.max(community_of_nodes) + 1
    assert communities_count > 0

    small_communities: Set[int] = set()
    small_nodes_count = 0
    small_nodes_umis = 0

    for community_index in range(communities_count):
        community_mask = community_of_nodes == community_index
        community_size = np.sum(community_mask)
        community_umis = np.sum(node_umis[community_mask])

        if community_size < lowest_metacell_size:
            ut.logger().debug("community: %s size: %s is tiny", community_index, community_size)
        elif community_size < min_metacell_size and 2 * community_umis < max_metacell_umis:
            ut.logger().debug(
                "community: %s umis: %s and size: %s is too few", community_index, community_umis, community_size
            )
        elif community_umis < min_metacell_umis and 2 * community_size < max_metacell_size:
            ut.logger().debug(
                "community: %s umis: %s and size: %s is too small", community_index, community_umis, community_size
            )
        else:
            continue

        small_communities.add(community_index)
        small_nodes_count += community_size
        small_nodes_umis += community_umis

    return (small_communities, small_nodes_count, small_nodes_umis)


def _cancel_communities(community_of_nodes: ut.NumpyVector, cancelled_communities: Set[int]) -> int:
    communities_count = np.max(community_of_nodes) + 1
    kept_communities_count = 0
    for community_index in range(communities_count):
        if community_index in cancelled_communities:
            community_of_nodes[community_of_nodes == community_index] = -1
            continue

        if community_index > kept_communities_count:
            community_of_nodes[community_of_nodes == community_index] = kept_communities_count

        kept_communities_count += 1

    assert kept_communities_count == communities_count - len(cancelled_communities)
    return kept_communities_count


@ut.logged()
@ut.timed_call()
def choose_seeds(
    *,
    edge_weights: ut.CompressedMatrix,
    seed_of_cells: Optional[ut.NumpyVector] = None,
    max_seeds_count: int,
    min_seed_size_quantile: float = pr.min_seed_size_quantile,
    max_seed_size_quantile: float = pr.max_seed_size_quantile,
    random_seed: int,
) -> ut.NumpyVector:
    """
    Choose initial assignment of cells to seeds based on the ``edge_weights``.

    Returns a vector assigning each node (cell) to a seed (initial community).

    If ``seed_of_cells`` is specified, it is expected to contain a vector of partial seeds. Only
    cells which have a negative seed will be assigned a new seed. New seeds will be created so that
    the total number of seeds will not exceed ``max_seeds_count``. The ``seed_of_cells`` will be
    modified in-place and returned.

    Otherwise, a new vector is created, initialized with ``-1`` (that is, no seed) for all nodes,
    filled as above, and returned.

    **Computation Parameters**

    1. We compute for each candidate node the number of nodes it is connected to (by an outgoing
       edge).

    2. We pick as a seed a random node whose number of connected nodes ("seed size") quantile is at
       least ``min_seed_size_quantile`` and at most ``max_seed_size_quantile``. This ensures we pick
       seeds that aren't too small or too large to get a good coverage of the population with a low
       number of seeds.

    3. We assign each of the connected nodes to their seed, and discount them from the number of
       connected nodes of the remaining unassigned nodes.

    4. We repeat this until we reach the target number of seeds.
    """
    size = edge_weights.shape[0]

    outgoing_edge_weights = ut.mustbe_compressed_matrix(edge_weights)
    assert ut.is_layout(outgoing_edge_weights, "row_major")

    incoming_edge_weights = ut.mustbe_compressed_matrix(ut.to_layout(outgoing_edge_weights, layout="column_major"))
    assert ut.is_layout(incoming_edge_weights, "column_major")

    if seed_of_cells is None:
        seed_of_cells = np.full(size, -1, dtype="int32")
    else:
        assert seed_of_cells.dtype == "int32"

    assert outgoing_edge_weights.shape == incoming_edge_weights.shape == (len(seed_of_cells), len(seed_of_cells))

    _choose_seeds(
        outgoing_edge_weights=outgoing_edge_weights,
        incoming_edge_weights=incoming_edge_weights,
        seed_of_cells=seed_of_cells,
        max_seeds_count=max_seeds_count,
        min_seed_size_quantile=min_seed_size_quantile,
        max_seed_size_quantile=max_seed_size_quantile,
        random_seed=random_seed,
    )

    return seed_of_cells


@ut.logged()
@ut.timed_call()
def _choose_seeds(
    *,
    outgoing_edge_weights: ut.CompressedMatrix,
    incoming_edge_weights: ut.CompressedMatrix,
    seed_of_cells: ut.NumpyVector,
    max_seeds_count: int,
    min_seed_size_quantile: float,
    max_seed_size_quantile: float,
    random_seed: int,
) -> int:
    ut.log_calc("partial seeds", seed_of_cells, formatter=ut.groups_description)

    seeds_count = xt.choose_seeds(
        outgoing_edge_weights.data,
        outgoing_edge_weights.indices,
        outgoing_edge_weights.indptr,
        incoming_edge_weights.data,
        incoming_edge_weights.indices,
        incoming_edge_weights.indptr,
        random_seed,
        max_seeds_count,
        min_seed_size_quantile,
        max_seed_size_quantile,
        seed_of_cells,
    )

    ut.log_calc("chosen seeds count", seeds_count)
    ut.log_calc("chosen seeds", seed_of_cells, formatter=ut.groups_description)
    assert np.min(seed_of_cells) == 0
    return seeds_count


@ut.logged()
@ut.timed_call()
def optimize_partitions(
    *,
    edge_weights: ut.CompressedMatrix,
    community_of_nodes: ut.NumpyVector,
    node_umis: ut.NumpyVector,
    low_partition_umis: int,
    target_partition_umis: int,
    high_partition_umis: int,
    low_partition_size: int,
    target_partition_size: int,
    high_partition_size: int,
    cooldown_pass: float = pr.cooldown_pass,
    cooldown_node: float = pr.cooldown_node,
    random_seed: int,
) -> float:
    """
    Optimize partition to candidate metacells (communities) using the ``edge_weights``.

    Returns the score of the optimized partition.

    This modifies the ``community_of_nodes`` in-place.

    The goal is to minimize the "stability" goal function which is defined to be the ratio between
    (1) the probability that, selecting a random node and either a random outgoing edge or a random
    incoming edge (biased by their weights), that the node connected to by that edge is in the same
    community (metacell) and (2) the probability that a random edge would lead to this same
    community (the fraction of its number of nodes out of the total).

    To maximize this, we repeatedly pass on a randomized permutation of the nodes, and for each
    node, move it to a random "better" community. When deciding if a community is better, we
    consider both (1) just the "local" product of the sum of the weights of incoming and outgoing edges
    between the node and the current and candidate communities and (2) the effect on the "global" goal
    function (considering the impact on this product for all other nodes connected to the current
    node).

    We define a notion of ``temperature`` (initially, 1 - ``cooldown_pass``, default:
    {cooldown_pass}) and we give a weight of ``temperature`` to the local score and
    (1 - ``temperature``) to the global score. When we move to the next node, we multiply the
    temperature by 1 - ``cooldown_pass``. If we did not move the node, we multiply its temperature
    by ``cooldown_node`` (default: {cooldown_node}). We skip looking at nodes which are colder from
    the global temperature to accelerate the algorithm. If we don't move any node, we reduce the
    global temperature below that of any cold node; if there are no such nodes, we reduce it to zero
    to perform a final hill-climbing phase.

    This simulated-annealing-like behavior helps the algorithm to escape local maximums, although of
    course no claim is made of achieving the global maximum of the goal function.
    """
    outgoing_edge_weights = ut.mustbe_compressed_matrix(edge_weights)
    assert ut.is_layout(outgoing_edge_weights, "row_major")
    assert 0 < low_partition_size < target_partition_size < high_partition_size
    assert 0 < low_partition_umis < target_partition_umis < high_partition_umis

    incoming_edge_weights = ut.mustbe_compressed_matrix(ut.to_layout(outgoing_edge_weights, layout="column_major"))
    communities_count = np.max(community_of_nodes) + 1
    assert communities_count > 0
    assert ut.is_layout(incoming_edge_weights, "column_major")
    return _optimize_partitions(
        outgoing_edge_weights=outgoing_edge_weights,
        incoming_edge_weights=incoming_edge_weights,
        random_seed=random_seed,
        node_umis=node_umis,
        target_partition_umis=target_partition_umis,
        low_partition_umis=low_partition_umis,
        high_partition_umis=high_partition_umis,
        target_partition_size=target_partition_size,
        low_partition_size=low_partition_size,
        high_partition_size=high_partition_size,
        cooldown_pass=cooldown_pass,
        cooldown_node=cooldown_node,
        community_of_nodes=community_of_nodes,
        hot_communities=list(range(communities_count)),
        cold_temperature=cooldown_pass,
    )


@ut.logged()
@ut.timed_call()
def _optimize_partitions(
    *,
    outgoing_edge_weights: ut.CompressedMatrix,
    incoming_edge_weights: ut.CompressedMatrix,
    community_of_nodes: ut.NumpyVector,
    hot_communities: List[int],
    node_umis: ut.NumpyVector,
    low_partition_umis: float,
    target_partition_umis: float,
    high_partition_umis: float,
    target_partition_size: float,
    low_partition_size: float,
    high_partition_size: float,
    cooldown_pass: float,
    cooldown_node: float,
    cold_temperature: float,
    random_seed: int,
) -> float:
    assert community_of_nodes.dtype == "int32"
    assert node_umis.dtype == "float32"

    communities_count = np.max(community_of_nodes) + 1
    hot_communities_mask = np.zeros(max(communities_count, 2), dtype="int8")
    hot_communities_mask[hot_communities] = 1

    score = xt.optimize_partitions(
        outgoing_edge_weights.data,
        outgoing_edge_weights.indices,
        outgoing_edge_weights.indptr,
        incoming_edge_weights.data,
        incoming_edge_weights.indices,
        incoming_edge_weights.indptr,
        random_seed,
        node_umis,
        low_partition_umis,
        target_partition_umis,
        high_partition_umis,
        low_partition_size,
        target_partition_size,
        high_partition_size,
        cooldown_pass,
        cooldown_node,
        community_of_nodes,
        hot_communities_mask,
        cold_temperature,
    )
    ut.log_calc("score", score)
    ut.log_calc("partitions", community_of_nodes, formatter=ut.groups_description)
    assert np.min(community_of_nodes) == 0
    return score


@ut.logged()
@ut.timed_call()
def score_partitions(
    *,
    node_umis: ut.NumpyVector,
    low_partition_umis: float,
    target_partition_umis: float,
    high_partition_umis: float,
    low_partition_size: int,
    target_partition_size: int,
    high_partition_size: int,
    edge_weights: ut.CompressedMatrix,
    partition_of_nodes: ut.NumpyVector,
    temperature: float,
    with_orphans: bool = True,
) -> None:
    """
    Compute the "stability" the "stability" goal function which is defined to be the ratio between
    (1) the probability that, selecting a random node and either a random outgoing edge or a random
    incoming edge (biased by their weights), that the node connected to by that edge is in the same
    community (metacell) and (2) the probability that a random edge would lead to this same
    community (the fraction of its number of nodes out of the total).

    If ``with_orphans`` is True (the default), outlier nodes are included in the computation. In
    general we add 1e-6 to the product of the incoming and outgoing weights so we can safely log it
    for efficient computation; thus orphans are given a very small (non-zero) weight so the overall
    score is not zeroed even when including them.
    """
    assert partition_of_nodes.dtype == "int32"
    assert node_umis.dtype == "float32"
    outgoing_edge_weights = ut.mustbe_compressed_matrix(edge_weights)
    assert ut.is_layout(outgoing_edge_weights, "row_major")

    incoming_edge_weights = ut.mustbe_compressed_matrix(ut.to_layout(outgoing_edge_weights, layout="column_major"))
    assert ut.is_layout(incoming_edge_weights, "column_major")

    hot_communities_mask = np.zeros(np.max(partition_of_nodes) + 1, dtype="int8")

    with ut.unfrozen(partition_of_nodes):
        with ut.timed_step(".score"):
            score = xt.score_partitions(
                outgoing_edge_weights.data,
                outgoing_edge_weights.indices,
                outgoing_edge_weights.indptr,
                incoming_edge_weights.data,
                incoming_edge_weights.indices,
                incoming_edge_weights.indptr,
                node_umis,
                low_partition_umis,
                target_partition_umis,
                high_partition_umis,
                low_partition_size,
                target_partition_size,
                high_partition_size,
                temperature,
                partition_of_nodes,
                hot_communities_mask,
                with_orphans,
            )

    ut.log_calc("score", score)
    return score
