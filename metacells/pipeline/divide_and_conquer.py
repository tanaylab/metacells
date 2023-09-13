"""
Divide and Conquer
------------------
"""

import gc
import logging
import os
from dataclasses import dataclass
from dataclasses import replace
from math import ceil
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import psutil  # type: ignore
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

from .collect import collect_metacells
from .direct import compute_direct_metacells

__all__ = [
    "set_max_parallel_piles",
    "get_max_parallel_piles",
    "guess_max_parallel_piles",
    "compute_target_pile_size",
    "divide_and_conquer_pipeline",
    "compute_divide_and_conquer_metacells",
]


MAX_PARALLEL_PILES = 0


def set_max_parallel_piles(max_parallel_piles: int) -> None:
    """
    Set the (maximal) number of piles to compute in parallel.

    By default, we use all the available hardware threads. Override this by setting the
    ``METACELLS_MAX_PARALLEL_PILES`` environment variable or by invoking this function from the main
    thread.

    A value of ``0`` will use all the available processors (see
    :py:func:`metacells.utilities.parallel.set_processors_count`). Otherwise, the value is the
    positive maximal number of processors to use in parallel for computing piles.

    It may be useful to restrict the number of parallel piles to restrict the total amount of memory
    used by the application, to keep it within the physical RAM available.
    """
    global MAX_PARALLEL_PILES
    MAX_PARALLEL_PILES = max_parallel_piles


set_max_parallel_piles(int(os.environ.get("METACELLS_MAX_PARALLEL_PILES", "0")))


def get_max_parallel_piles() -> int:
    """
    Return the maximal number of piles to compute in parallel.
    """
    return MAX_PARALLEL_PILES


@ut.expand_doc(percent=round((1 + pr.max_gbs) * 100))
def guess_max_parallel_piles(
    cells: AnnData,
    what: str = "__x__",
    *,
    max_gbs: float = pr.max_gbs,
    target_metacell_umis: int = pr.target_metacell_umis,
    cell_umis: Optional[ut.NumpyVector] = pr.cell_umis,
    target_metacell_size: int = pr.target_metacell_size,
    min_target_pile_size: int = pr.min_target_pile_size,
    max_target_pile_size: int = pr.max_target_pile_size,
    target_metacells_in_pile: int = pr.target_metacells_in_pile,
) -> int:
    """
    Try and guess a reasonable maximal number of piles to use for computing metacells for the
    specified ``cells`` using at most ``max_gbs`` of memory (default: {max_gbs}, that is
    {percent}% of all the machine has - if zero or negative, is relative to the machines memory).

    The amount of memory used depends on the target pile size, so give this function the same parameters as for
    :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline` so it will use the same automatically
    adjusted target pile size.

    .. note::

        This is only a best-effort guess. A too-low number would slow down the computation by using
        less processors than it could. A too-high number might cause the computation to crash,
        running out of memory. So: use with care, YMMV, keep an eye on memory usage and other
        applications running in parallel to the computation, and apply common sense.

    .. todo::

        Ideally the system would self tune (avoid spawning more parallel processes for computing
        piles when getting close to the memory limit). This is "not easy" to achieve using Python's
        parallel programming APIs.

    **Computation Parameters**

    1. Figure out the target pile size by invoking
       :py:func:`metacells.pipeline.divide_and_conquer.compute_target_pile_size` using the ``target_metacell_umis``
       (default: {target_metacell_umis}), ``cell_umis`` (default: {cell_umis}), ``target_metacell_size`` (default:
       {target_metacell_size}), ``min_target_pile_size`` (default: {min_target_pile_size}), ``max_target_pile_size``
       (default: {max_target_pile_size}) and ``target_metacells_in_pile`` (default: {target_metacells_in_pile}).

    2. Use ``psutil.virtual_memory`` to query the amount of memory in the system, and apply ``max_gbs`` (default:
       {max_gbs}) off the top (e.g., ``-0.1`` means reduce it by 10%).

    3. Guesstimate the number of piles that, if computed in parallel, will consume this amount of memory.
    """
    target_pile_size = compute_target_pile_size(
        cells,
        what,
        target_metacell_umis=target_metacell_umis,
        cell_umis=cell_umis,
        target_metacell_size=target_metacell_size,
        min_target_pile_size=min_target_pile_size,
        max_target_pile_size=max_target_pile_size,
        target_metacells_in_pile=target_metacells_in_pile,
    )

    cells_nnz = ut.nnz_matrix(ut.get_vo_proper(cells, what))
    piles_nnz = cells_nnz * target_pile_size / cells.n_obs
    parallel_processes = ut.get_processors_count()
    if max_gbs <= 0:
        assert max_gbs > -1
        max_gbs += 1
        max_gbs *= psutil.virtual_memory().total / (1024.0 * 1024.0 * 1024.0)
    parallel_piles = int((max_gbs - cells_nnz * 6.5e-8 - parallel_processes / 13) / (piles_nnz * 13e-8))
    return max(parallel_piles, 1)


@ut.logged()
@ut.expand_doc()
def compute_target_pile_size(
    adata: AnnData,
    what: str = "__x__",
    *,
    cell_umis: Optional[ut.NumpyVector] = pr.cell_umis,
    target_metacell_size: int = pr.target_metacell_size,
    target_metacell_umis: int = pr.target_metacell_umis,
    min_target_pile_size: int = pr.min_target_pile_size,
    max_target_pile_size: int = pr.max_target_pile_size,
    target_metacells_in_pile: int = pr.target_metacells_in_pile,
) -> int:
    """
    Compute how many cells should be in each divide-and-conquer pile.

    Larger piles are slower to process (O(N^2)) than smaller piles, but (up to a point) they allow the algorithm to
    produce better results. The ideal pile size is just big enough to allow for a sufficient number of metacells to be
    created; specifically, we'd like the number of metacells in each pile to be roughly the same as the number of cells
    in a metacell, which is natural for the divide and conquer algorithm.

    Note that the target pile size is just a target. It is directly used in the 1st phase of the divide-and-conquer
    algorithm, but the pile sizes of the 2nd phase are determined by clustering metacells together which inevitably
    cause the actual pile sizes to vary around this number.

    **Computation Parameters**

    0. If ``cell_umis`` is not specified, use the sum of the ``what`` data for each cell.

    1. The natural pile size is the product of the ``target_metacell_size`` (default: {target_metacell_size}) and
       the ``target_metacells_in_pile`` (default: {target_metacells_in_pile}).

    2. Also consider the mean number of cells needed to reach the ``target_metacell_umis`` (default:
       {target_metacell_umis}); this times the ``target_metacells_in_pile`` gives us another pile size, and if this one
       is larger than what we computed in the previous step, we use it instead.

    3. Clamp this value to be no lower than ``min_target_pile_size`` (default: {min_target_pile_size}) and no higher
       than ``max_target_pile_size`` (default: {max_target_pile_size}). That is, if you set both to the same value, this
       will force the target pile size value.
    """
    assert 0 < min_target_pile_size <= max_target_pile_size

    pile_size_by_cells = target_metacell_size * target_metacells_in_pile
    ut.log_calc("pile_size_by_cells", pile_size_by_cells)

    if cell_umis is None:
        cell_umis = ut.get_o_numpy(adata, what, sum=True, formatter=ut.sizes_description).astype("float32")
    else:
        assert cell_umis.dtype == "float32"
    assert isinstance(cell_umis, ut.NumpyVector)
    mean_cell_umis = np.mean(cell_umis)
    ut.log_calc("mean_cell_umis", mean_cell_umis)
    mean_metacell_size = target_metacell_umis / mean_cell_umis
    pile_size_by_umis = int(round(mean_metacell_size * target_metacells_in_pile))
    ut.log_calc("pile_size_by_umis", pile_size_by_umis)

    target_pile_size = max(pile_size_by_cells, pile_size_by_umis)
    target_pile_size = max(target_pile_size, min_target_pile_size)
    target_pile_size = min(target_pile_size, max_target_pile_size)
    ut.log_return("target_pile_size", target_pile_size)
    return target_pile_size


@dataclass
class DirectParameters:  # pylint: disable=too-many-instance-attributes
    """
    Parameters for direct computation of metacells in a single pile.
    """

    select_downsample_min_samples: int
    select_downsample_min_cell_quantile: float
    select_downsample_max_cell_quantile: float
    select_min_gene_total: Optional[int]
    select_min_gene_top3: Optional[int]
    select_min_gene_relative_variance: Optional[float]
    select_min_genes: int
    cells_similarity_value_regularization: float
    cells_similarity_log_data: bool
    cells_similarity_method: str
    target_metacell_umis: int
    cell_umis: ut.NumpyVector
    target_metacell_size: int
    min_metacell_size: int
    knn_k: Optional[int]
    knn_k_umis_quantile: float
    min_knn_k: Optional[int]
    knn_balanced_ranks_factor: float
    knn_incoming_degree_factor: float
    knn_outgoing_degree_factor: float
    knn_min_outgoing_degree: int
    min_seed_size_quantile: float
    max_seed_size_quantile: float
    candidates_cooldown_pass: float
    candidates_cooldown_node: float
    candidates_cooldown_phase: float
    knn_k_size_factor: float
    candidates_min_split_size_factor: float
    candidates_max_merge_size_factor: float
    candidates_max_split_min_cut_strength: Optional[float]
    candidates_min_cut_seed_cells: Optional[int]
    must_complete_cover: bool
    deviants_policy: str
    deviants_gap_skip_cells: int
    deviants_min_gene_fold_factor: float
    deviants_min_noisy_gene_fold_factor: float
    deviants_max_gene_fraction: Optional[float]
    deviants_max_cell_fraction: Optional[float]
    deviants_max_gap_cells_count: int
    deviants_max_gap_cells_fraction: float
    dissolve_min_robust_size_factor: Optional[float]
    dissolve_min_convincing_gene_fold_factor: float
    random_seed: int


@dataclass
class DacParameters:  # pylint: disable=too-many-instance-attributes
    """
    Parameters controlling divide-and-conquer algorithm.
    """

    quick_and_dirty: bool
    groups_similarity_log_data: bool
    groups_similarity_method: str
    min_target_pile_size: int
    max_target_pile_size: int
    target_metacells_in_pile: int
    target_pile_size: int
    piles_knn_k_size_factor: float
    piles_min_split_size_factor: float
    piles_min_robust_size_factor: float
    piles_max_merge_size_factor: float
    direct_parameters: DirectParameters


class SubsetResults:
    """
    The results of computing metacells for a subset of the cells.
    """

    def __init__(self, sdata: AnnData) -> None:
        #: The computed metacell index for each cell.
        self.metacell_indices = ut.get_o_numpy(sdata, "metacell", formatter=ut.groups_description).copy()

        #: The mask of the cells which were in dissolved metacells.
        self.dissolved = ut.get_o_numpy(sdata, "dissolved", formatter=ut.groups_description).copy()

        #: The full index of each cell.
        self.full_cell_indices = ut.get_o_numpy(sdata, "__full_cell_index__")

        #: The mask of the genes that were selected to compute metacells by for the subset.
        self.selected_mask = ut.get_v_numpy(sdata, "selected_gene")

    @ut.logged()
    def collect(
        self,
        *,
        adata: AnnData,
        counts: List[int],
        metacells_level: int,
        collected_mask: ut.NumpyVector,
    ) -> None:
        """
        Collect the results of the subset into the full data.
        """
        metacell_of_cells = ut.get_o_numpy(adata, "metacell", formatter=ut.groups_description).copy()
        dissolved_of_cells = ut.get_o_numpy(adata, "dissolved", formatter=ut.groups_description).copy()
        level_of_cells = ut.get_o_numpy(adata, "metacell_level").copy()

        if np.any(collected_mask):
            max_collected = np.max(metacell_of_cells[collected_mask])
        else:
            max_collected = -1
        assert max_collected + 1 == counts[0]

        metacell_of_subset = self.metacell_indices
        dissolved_of_subset = self.dissolved
        ut.log_calc("subset[metacell]", metacell_of_subset, formatter=ut.groups_description)
        ut.log_calc("subset[dissolved]", dissolved_of_subset)
        metacells_count = round(np.max(metacell_of_subset) + 1)
        metacell_of_subset[metacell_of_subset >= 0] += counts[0]
        counts[0] += metacells_count
        metacell_of_cells[self.full_cell_indices] = metacell_of_subset
        dissolved_of_cells[self.full_cell_indices] = dissolved_of_subset
        level_of_cells[self.full_cell_indices] = metacells_level
        ut.set_o_data(adata, "metacell", metacell_of_cells, formatter=ut.groups_description)
        ut.set_o_data(adata, "dissolved", dissolved_of_cells)
        ut.set_o_data(adata, "metacell_level", level_of_cells)

        assert not np.any(collected_mask[self.full_cell_indices])
        collected_mask[self.full_cell_indices] = True

        selected_genes_mask = ut.get_v_numpy(adata, "selected_gene")
        ut.set_v_data(adata, "selected_gene", selected_genes_mask | self.selected_mask)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def divide_and_conquer_pipeline(
    adata: AnnData,
    what: str = "__x__",
    *,
    rare_max_genes: int = pr.rare_max_genes,
    rare_max_gene_cell_fraction: float = pr.rare_max_gene_cell_fraction,
    rare_min_gene_maximum: int = pr.rare_min_gene_maximum,
    rare_genes_similarity_method: str = pr.rare_genes_similarity_method,
    rare_genes_cluster_method: str = pr.rare_genes_cluster_method,
    rare_min_genes_of_modules: int = pr.rare_min_genes_of_modules,
    rare_min_cells_of_modules: int = pr.rare_min_cells_of_modules,
    rare_min_module_correlation: float = pr.rare_min_module_correlation,
    rare_min_related_gene_fold_factor: float = pr.rare_min_related_gene_fold_factor,
    rare_max_related_gene_increase_factor: float = pr.rare_max_related_gene_increase_factor,
    rare_min_cell_module_total: int = pr.rare_min_cell_module_total,
    rare_max_cells_factor_of_random_pile: float = pr.rare_max_cells_factor_of_random_pile,
    rare_deviants_max_cell_fraction: Optional[float] = pr.rare_deviants_max_cell_fraction,
    rare_dissolve_min_robust_size_factor: Optional[float] = pr.rare_dissolve_min_robust_size_factor,
    rare_dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    quick_and_dirty: bool = pr.quick_and_dirty,
    select_downsample_min_samples: int = pr.select_downsample_min_samples,
    select_downsample_min_cell_quantile: float = pr.select_downsample_min_cell_quantile,
    select_downsample_max_cell_quantile: float = pr.select_downsample_max_cell_quantile,
    select_min_gene_total: Optional[int] = pr.select_min_gene_total,
    select_min_gene_top3: Optional[int] = pr.select_min_gene_top3,
    select_min_gene_relative_variance: Optional[float] = pr.select_min_gene_relative_variance,
    select_min_genes: int = pr.select_min_genes,
    cells_similarity_value_regularization: float = pr.cells_similarity_value_regularization,
    cells_similarity_log_data: bool = pr.cells_similarity_log_data,
    cells_similarity_method: str = pr.cells_similarity_method,
    groups_similarity_log_data: bool = pr.groups_similarity_log_data,
    groups_similarity_method: str = pr.groups_similarity_method,
    target_metacell_umis: int = pr.target_metacell_umis,
    cell_umis: Optional[ut.NumpyVector] = pr.cell_umis,
    target_metacell_size: int = pr.target_metacell_size,
    min_metacell_size: int = pr.min_metacell_size,
    target_metacells_in_pile: int = pr.target_metacells_in_pile,
    min_target_pile_size: int = pr.min_target_pile_size,
    max_target_pile_size: int = pr.max_target_pile_size,
    piles_knn_k_size_factor: float = pr.piles_knn_k_size_factor,
    piles_min_split_size_factor: float = pr.piles_min_split_size_factor,
    piles_min_robust_size_factor: float = pr.piles_min_robust_size_factor,
    piles_max_merge_size_factor: float = pr.piles_max_merge_size_factor,
    knn_k: Optional[int] = pr.knn_k,
    knn_k_umis_quantile: float = pr.knn_k_umis_quantile,
    min_knn_k: Optional[int] = pr.min_knn_k,
    knn_balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    knn_incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    knn_outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    knn_min_outgoing_degree: int = pr.knn_min_outgoing_degree,
    min_seed_size_quantile: float = pr.min_seed_size_quantile,
    max_seed_size_quantile: float = pr.max_seed_size_quantile,
    candidates_knn_k_size_factor: float = pr.candidates_knn_k_size_factor,
    candidates_cooldown_pass: float = pr.cooldown_pass,
    candidates_cooldown_node: float = pr.cooldown_node,
    candidates_cooldown_phase: float = pr.cooldown_phase,
    candidates_min_split_size_factor: float = pr.candidates_min_split_size_factor,
    candidates_max_merge_size_factor: float = pr.candidates_max_merge_size_factor,
    candidates_max_split_min_cut_strength: Optional[float] = pr.max_split_min_cut_strength,
    candidates_min_cut_seed_cells: int = pr.min_cut_seed_cells,
    must_complete_cover: bool = False,
    deviants_policy: str = pr.deviants_policy,
    deviants_gap_skip_cells: int = pr.deviants_gap_skip_cells,
    deviants_min_gene_fold_factor: float = pr.deviants_min_gene_fold_factor,
    deviants_min_noisy_gene_fold_factor: float = pr.deviants_min_noisy_gene_fold_factor,
    deviants_max_gene_fraction: Optional[float] = pr.deviants_max_gene_fraction,
    deviants_max_cell_fraction: Optional[float] = pr.deviants_max_cell_fraction,
    deviants_max_gap_cells_count: int = pr.deviants_max_gap_cells_count,
    deviants_max_gap_cells_fraction: float = pr.deviants_max_gap_cells_fraction,
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    random_seed: int,
) -> None:
    """
    Complete pipeline using divide-and-conquer to compute the metacells for the ``what`` (default: {what}) data.

    If a ``progress_bar`` is active, progress will be reported into the current (slice of) the progress bar. The
    detection of rare gene modules is not counted in the progress bar; instead it is logged at the ``INFO`` level.

    .. note::

        This is applicable to "any size" data. If the data is "small" (O(10,000)), it will revert to using the direct
        metacell computation (but will still by default first look for rare gene modules). If the data is "large" (up to
        O(10,000,000)), this will be much faster and will require much less memory than using the direct approach. The
        current implementation is not optimized for "huge" data (O(1,000,000,000)) - that is, it will work, and keep
        will use a limited amount of memory, but a faster implementation would distribute the computation across
        multiple servers.

    **Input**

    The presumably "clean" annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation annotation
    containing such a matrix.

    **Returns**

    Sets the following annotations in ``adata``:

    Everything set by ``compute_divide_and_conquer_metacells``; and in addition:

    Variable (Gene) Annotations
        ``rare_gene_module_<N>``
            A boolean mask for the genes in the rare gene module ``N``.

        ``rare_gene``
            A boolean mask for the genes in any of the rare gene modules.

    Observations (Cell) Annotations
        ``cells_rare_gene_module``
            The index of the rare gene module each cell expresses the most, or ``-1`` in the common case it does not
            express any rare genes module.

        ``rare_cell``
            A boolean mask for the (few) cells that express a rare gene module.

    **Computation Parameters**

    0. If ``cell_umis`` is not specified, use the sum of the ``what`` data for each cell.

    1. Invoke :py:func:`metacells.pipeline.divide_and_conquer.compute_target_pile_size` using
       ``target_metacell_umis`` (default: {target_metacell_umis}), ``cell_umis`` (default: {cell_umis}),
       ``target_metacell_size`` (default: {target_metacell_size}), ``min_target_pile_size`` (default:
       {min_target_pile_size}), ``max_target_pile_size`` (default: {max_target_pile_size}) and
       ``target_metacells_in_pile`` (default: {target_metacells_in_pile}).

    2. Invoke :py:func:`metacells.tools.rare.find_rare_gene_modules` to isolate cells expressing
       rare gene modules, using the
       ``rare_max_genes`` (default: {rare_max_genes}),
       ``rare_max_gene_cell_fraction`` (default: {rare_max_gene_cell_fraction}),
       ``rare_min_gene_maximum`` (default: {rare_min_gene_maximum}),
       ``rare_genes_similarity_method`` (default: {rare_genes_similarity_method}),
       ``rare_genes_cluster_method`` (default: {rare_genes_cluster_method}),
       ``rare_min_genes_of_modules`` (default: {rare_min_genes_of_modules}),
       ``rare_min_cells_of_modules`` (default: {rare_min_cells_of_modules}),
       ``rare_min_module_correlation`` (default: {rare_min_module_correlation}),
       ``rare_min_related_gene_fold_factor`` (default: {rare_min_related_gene_fold_factor})
       ``rare_max_related_gene_increase_factor`` (default: {rare_max_related_gene_increase_factor})
       ``rare_max_cells_factor_of_random_pile`` (default: {rare_max_cells_factor_of_random_pile})
       and
       ``rare_min_cell_module_total`` (default: {rare_min_cell_module_total}). Use a non-zero
       ``random_seed`` to make this reproducible.

    3. For each detected rare gene module, collect all cells that express the module, and apply
       :py:func:`metacells.pipeline.direct.compute_direct_metacells` to them.

    4. Collect all the outliers from the above together with the rest of the cells (which express no rare gene module)
       and apply the :py:func:`compute_divide_and_conquer_metacells` to the combined result. The annotations for
       rare-but-outlier cells (e.g. ``cell_directs``) will not reflect the work done when computing the rare gene module
       metacells which they were rejected from.
    """
    if cell_umis is None:
        cell_umis = ut.get_o_numpy(adata, what, sum=True, formatter=ut.sizes_description).astype("float32")
    else:
        assert cell_umis.dtype == "float32"
    assert isinstance(cell_umis, ut.NumpyVector)

    target_pile_size = compute_target_pile_size(
        adata,
        what,
        target_metacell_umis=target_metacell_umis,
        cell_umis=cell_umis,
        target_metacell_size=target_metacell_size,
        target_metacells_in_pile=target_metacells_in_pile,
        min_target_pile_size=min_target_pile_size,
        max_target_pile_size=max_target_pile_size,
    )

    dac_parameters = DacParameters(
        quick_and_dirty=quick_and_dirty,
        groups_similarity_log_data=groups_similarity_log_data,
        groups_similarity_method=groups_similarity_method,
        min_target_pile_size=min_target_pile_size,
        max_target_pile_size=max_target_pile_size,
        target_pile_size=target_pile_size,
        target_metacells_in_pile=target_metacells_in_pile,
        piles_knn_k_size_factor=piles_knn_k_size_factor,
        piles_min_split_size_factor=piles_min_split_size_factor,
        piles_min_robust_size_factor=piles_min_robust_size_factor,
        piles_max_merge_size_factor=piles_max_merge_size_factor,
        direct_parameters=DirectParameters(
            select_downsample_min_samples=select_downsample_min_samples,
            select_downsample_min_cell_quantile=select_downsample_min_cell_quantile,
            select_downsample_max_cell_quantile=select_downsample_max_cell_quantile,
            select_min_gene_total=select_min_gene_total,
            select_min_gene_top3=select_min_gene_top3,
            select_min_gene_relative_variance=select_min_gene_relative_variance,
            select_min_genes=select_min_genes,
            cells_similarity_value_regularization=cells_similarity_value_regularization,
            cells_similarity_log_data=cells_similarity_log_data,
            cells_similarity_method=cells_similarity_method,
            target_metacell_umis=target_metacell_umis,
            cell_umis=cell_umis,
            target_metacell_size=target_metacell_size,
            min_metacell_size=min_metacell_size,
            knn_k=knn_k,
            knn_k_umis_quantile=knn_k_umis_quantile,
            min_knn_k=min_knn_k,
            knn_balanced_ranks_factor=knn_balanced_ranks_factor,
            knn_incoming_degree_factor=knn_incoming_degree_factor,
            knn_outgoing_degree_factor=knn_outgoing_degree_factor,
            knn_min_outgoing_degree=knn_min_outgoing_degree,
            min_seed_size_quantile=min_seed_size_quantile,
            max_seed_size_quantile=max_seed_size_quantile,
            candidates_cooldown_pass=candidates_cooldown_pass,
            candidates_cooldown_node=candidates_cooldown_node,
            candidates_cooldown_phase=candidates_cooldown_phase,
            knn_k_size_factor=candidates_knn_k_size_factor,
            candidates_min_split_size_factor=candidates_min_split_size_factor,
            candidates_max_merge_size_factor=candidates_max_merge_size_factor,
            candidates_max_split_min_cut_strength=candidates_max_split_min_cut_strength,
            candidates_min_cut_seed_cells=candidates_min_cut_seed_cells,
            must_complete_cover=must_complete_cover,
            deviants_policy=deviants_policy,
            deviants_gap_skip_cells=deviants_gap_skip_cells,
            deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
            deviants_min_noisy_gene_fold_factor=deviants_min_noisy_gene_fold_factor,
            deviants_max_gene_fraction=deviants_max_gene_fraction,
            deviants_max_cell_fraction=deviants_max_cell_fraction,
            deviants_max_gap_cells_count=deviants_max_gap_cells_count,
            deviants_max_gap_cells_fraction=deviants_max_gap_cells_fraction,
            dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
            dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
            random_seed=random_seed,
        ),
    )

    counts = [0]
    _initialize_divide_and_conquer_results(adata)

    if ut.has_progress_bar():
        logger = ut.logger()
        log_level = logger.level
        logger.setLevel(logging.INFO)
        logger.info("Detect rare gene modules...")
        logger.setLevel(log_level)

    name = ut.get_name(adata)

    with ut.timed_step(".rare"):
        tl.find_rare_gene_modules(
            adata,
            what,
            max_genes=rare_max_genes,
            max_gene_cell_fraction=rare_max_gene_cell_fraction,
            min_gene_maximum=rare_min_gene_maximum,
            genes_similarity_method=rare_genes_similarity_method,
            genes_cluster_method=rare_genes_cluster_method,
            min_genes_of_modules=rare_min_genes_of_modules,
            min_cells_of_modules=rare_min_cells_of_modules,
            target_metacell_size=target_metacell_size,
            target_pile_size=target_pile_size,
            min_module_correlation=rare_min_module_correlation,
            min_related_gene_fold_factor=rare_min_related_gene_fold_factor,
            max_related_gene_increase_factor=rare_max_related_gene_increase_factor,
            max_cells_factor_of_random_pile=rare_max_cells_factor_of_random_pile,
            min_cell_module_total=rare_min_cell_module_total,
            reproducible=(random_seed != 0),
        )

        rare_module_of_cells = ut.get_o_numpy(adata, "cells_rare_gene_module", formatter=ut.groups_description)
        rare_cells_count = np.sum(rare_module_of_cells >= 0)
        common_cells_fraction: Optional[float] = None
        if rare_cells_count > 0:
            if ut.has_progress_bar():
                rare_cells_fraction: Optional[float] = rare_cells_count / adata.n_obs
                assert rare_cells_fraction is not None  # Stupid mypy
                common_cells_fraction = 1.0 - rare_cells_fraction
            else:
                rare_cells_fraction = None

            with ut.progress_bar_slice(rare_cells_fraction):
                # TODO: This assumes each rare gene module cells fit in a single pile.
                # If/when we move to a truly ludicrous number of cells, this will no longer be true.
                _compute_metacells_of_piles(
                    adata,
                    what,
                    pile_of_cells=rare_module_of_cells,
                    prefix="rare" if name is None else name + ".rare",
                    counts=counts,
                    collected_mask=np.zeros(adata.n_obs, dtype="bool"),
                    metacells_level=0,
                    direct_parameters=replace(
                        dac_parameters.direct_parameters,
                        deviants_max_cell_fraction=rare_deviants_max_cell_fraction,
                        dissolve_min_robust_size_factor=rare_dissolve_min_robust_size_factor,
                        dissolve_min_convincing_gene_fold_factor=rare_dissolve_min_convincing_gene_fold_factor,
                        must_complete_cover=False,
                    ),
                )

    rare_metacell_of_cells = ut.get_o_numpy(adata, "metacell", formatter=ut.groups_description)
    common_cells_mask = rare_metacell_of_cells < 0
    collected_mask = ~common_cells_mask

    with ut.timed_step(".common"):
        with ut.progress_bar_slice(common_cells_fraction):
            _compute_divide_and_conquer_subset(
                adata,
                what,
                prefix="common" if name is None else name + ".common",
                metacells_level=0,
                subset_mask=common_cells_mask,
                collected_mask=collected_mask,
                counts=counts,
                dac_parameters=dac_parameters,
                random_seed=random_seed,
            )

    if rare_cells_count > 0:
        selected_genes = ut.get_v_numpy(adata, "selected_gene")
        rare_genes = ut.get_v_numpy(adata, "rare_gene")
        ut.set_v_data(adata, "selected_gene", selected_genes | rare_genes)

    _finalize_divide_and_conquer_results(adata)
    assert np.all(collected_mask)


@ut.logged()
def compute_divide_and_conquer_metacells(
    adata: AnnData,
    what: str = "__x__",
    *,
    quick_and_dirty: bool = pr.quick_and_dirty,
    select_downsample_min_samples: int = pr.select_downsample_min_samples,
    select_downsample_min_cell_quantile: float = pr.select_downsample_min_cell_quantile,
    select_downsample_max_cell_quantile: float = pr.select_downsample_max_cell_quantile,
    select_min_gene_total: Optional[int] = pr.select_min_gene_total,
    select_min_gene_top3: Optional[int] = pr.select_min_gene_top3,
    select_min_gene_relative_variance: Optional[float] = pr.select_min_gene_relative_variance,
    select_min_genes: int = pr.select_min_genes,
    cells_similarity_value_regularization: float = pr.cells_similarity_value_regularization,
    cells_similarity_log_data: bool = pr.cells_similarity_log_data,
    cells_similarity_method: str = pr.cells_similarity_method,
    groups_similarity_log_data: bool = pr.groups_similarity_log_data,
    groups_similarity_method: str = pr.groups_similarity_method,
    target_metacell_umis: int = pr.target_metacell_umis,
    cell_umis: Optional[ut.NumpyVector] = pr.cell_umis,
    target_metacell_size: int = pr.target_metacell_size,
    min_metacell_size: int = pr.min_metacell_size,
    min_target_pile_size: int = pr.min_target_pile_size,
    max_target_pile_size: int = pr.max_target_pile_size,
    target_metacells_in_pile: int = pr.target_metacells_in_pile,
    piles_knn_k_size_factor: float = pr.piles_knn_k_size_factor,
    piles_min_split_size_factor: float = pr.piles_min_split_size_factor,
    piles_min_robust_size_factor: float = pr.piles_min_robust_size_factor,
    piles_max_merge_size_factor: float = pr.piles_max_merge_size_factor,
    knn_k: Optional[int] = pr.knn_k,
    knn_k_umis_quantile: float = pr.knn_k_umis_quantile,
    min_knn_k: Optional[int] = pr.min_knn_k,
    knn_balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    knn_incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    knn_outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    knn_min_outgoing_degree: int = pr.knn_min_outgoing_degree,
    min_seed_size_quantile: float = pr.min_seed_size_quantile,
    max_seed_size_quantile: float = pr.max_seed_size_quantile,
    candidates_cooldown_pass: float = pr.cooldown_pass,
    candidates_cooldown_node: float = pr.cooldown_node,
    candidates_cooldown_phase: float = pr.cooldown_phase,
    candidates_knn_k_size_factor: float = pr.candidates_knn_k_size_factor,
    candidates_min_split_size_factor: float = pr.candidates_min_split_size_factor,
    candidates_max_merge_size_factor: float = pr.candidates_max_merge_size_factor,
    candidates_max_split_min_cut_strength: Optional[float] = pr.max_split_min_cut_strength,
    candidates_min_cut_seed_cells: int = pr.min_cut_seed_cells,
    must_complete_cover: bool = False,
    deviants_policy: str = pr.deviants_policy,
    deviants_gap_skip_cells: int = pr.deviants_gap_skip_cells,
    deviants_min_gene_fold_factor: float = pr.deviants_min_gene_fold_factor,
    deviants_min_noisy_gene_fold_factor: float = pr.deviants_min_noisy_gene_fold_factor,
    deviants_max_gene_fraction: Optional[float] = pr.deviants_max_gene_fraction,
    deviants_max_cell_fraction: Optional[float] = pr.deviants_max_cell_fraction,
    deviants_max_gap_cells_count: int = pr.deviants_max_gap_cells_count,
    deviants_max_gap_cells_fraction: float = pr.deviants_max_gap_cells_fraction,
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    random_seed: int,
    _fdata: Optional[AnnData] = None,
    _counts: Optional[Dict[str, int]] = None,
) -> None:
    """
    Compute metacells for ``what`` (default: {what}) data using the divide-and-conquer method.

    This divides large data to smaller "piles" and directly computes metacells for each, then generates new piles of
    "similar" metacells and directly computes the final metacells from each such pile. Due to this divide-and-conquer
    approach, the total amount of memory required is bounded (by the pile size), and the amount of total CPU grows
    slowly with the problem size, allowing this method to be applied to very large data (millions of cells).

    If a progress bar is active, progress will be reported into the current (slice of) the progress bar.

    .. todo::

        The divide-and-conquer implementation is restricted for using multiple CPUs on a single shared-memory machine.
        At some problem size (probably around a billion cells), it would make sense to modify it to support distributed
        execution of sub-problems on different servers. This would only be an implementation change, rather than an
        algorithmic change.

    **Input**

    The presumably "clean" annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation annotation
    containing such a matrix.

    **Returns**

    Sets the following annotations in ``adata``:

    Observations (Cell) Annotations
        ``metacell``
            The integer index of the final metacell each cell belongs to. The metacells are in no particular order. This
            is ``-1`` for cells not included in any metacell (outliers, excluded).

        ``dissolve``
           A boolean mask of the cells which were in a dissolved metacell (in the last pile they were in).

    **Computation Parameters**

    0. If ``cell_umis`` is not specified, use the sum of the ``what`` data for each cell.

    1. Invoke :py:func:`metacells.pipeline.divide_and_conquer.compute_target_pile_size` using ``target_metacell_umis``
       (default: {target_metacell_umis}), ``cell_umis`` (default: {cell_umis}), ``target_metacell_size`` (default:
       {target_metacell_size}), ``min_target_pile_size`` (default: {min_target_pile_size}), ``max_target_pile_size``
       (default: {max_target_pile_size}) and ``target_metacells_in_pile`` (default: {target_metacells_in_pile}).

    2. If the data is smaller than the target_pile_size times the ``piles_min_split_size_factor`` (default:
       {piles_min_split_size_factor}), then just invoke :py:func:`metacells.pipeline.direct.compute_direct_metacells`
       using the parameters. Otherwise, perform the following steps.

    3. Phase 0 (preliminary): group the cells randomly into equal-sized piles of roughly the ``target_pile_size`` using
       the ``random_seed`` (default: {random_seed}) to allow making this replicable. Otherwise, use the groups of
       metacells computed in the preliminary phase to create the piles of the final phase.

    4. Compute metacells using the piles by invoking :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
       possibly in parallel.

    5. Collect the outliers from all the piles and recursively compute metacells for them. Continue to compute metacells
       for the outliers-of-outliers, etc. until all the outliers fit in a single pile; then compute metacells for this
       outlier-most cells with ``must_complete_cover=True``.

    6. Phase 0 to 1: Recursively run the algorithm grouping the preliminary metacells into groups-of-metacells, using
       ``piles_knn_k_size_factor`` (default: {piles_knn_k_size_factor}),
       ``piles_min_split_size_factor`` (default: {piles_min_split_size_factor}),
       ``piles_min_robust_size_factor`` (default: {piles_min_robust_size_factor}), and
       ``piles_max_merge_size_factor`` (default: {piles_max_merge_size_factor}).

    7. Phase 1 (final): group the cells into piles based on the groups-of-metacells. Compute metacells using the piles,
       by invoking :py:func:`metacells.pipeline.direct.compute_direct_metacells`, possibly in parallel.

    8. Collect the outliers from all the piles and compute metacells for them (possibly in parallel). If
       ``quick_and_dirty``, stop. Otherwise, continue to group all the cells in metacells by grouping the
       outliers-of-outliers similarly to step 5.

    9. Phase 1 to 2: Recursively run the algorithm grouping the outlier metacells into groups-of-metacells, using
       ``piles_knn_k_size_factor`` (default: {piles_knn_k_size_factor}),
       ``piles_min_split_size_factor`` (default: {piles_min_split_size_factor}),
       ``piles_min_robust_size_factor`` (default: {piles_min_robust_size_factor}), and
       ``piles_max_merge_size_factor`` (default: {piles_max_merge_size_factor}).

    10. Phase 2 (outliers): Group the cells into piles based on the groups-of-metacells. Compute metacells using the
        piles, by invoking :py:func:`metacells.pipeline.direct.compute_direct_metacells`, possibly in parallel. Stop
        here and accept all outliers as final outliers (that is, there are no level 2 metacells).
    """
    if cell_umis is None:
        cell_umis = ut.get_o_numpy(adata, what, sum=True, formatter=ut.sizes_description).astype("float32")
    else:
        assert cell_umis.dtype == "float32"
    assert isinstance(cell_umis, ut.NumpyVector)

    target_pile_size = compute_target_pile_size(
        adata=adata,
        target_metacell_umis=target_metacell_umis,
        cell_umis=cell_umis,
        target_metacell_size=target_metacell_size,
        min_target_pile_size=min_target_pile_size,
        max_target_pile_size=max_target_pile_size,
        target_metacells_in_pile=target_metacells_in_pile,
    )

    dac_parameters = DacParameters(
        quick_and_dirty=quick_and_dirty,
        groups_similarity_log_data=groups_similarity_log_data,
        groups_similarity_method=groups_similarity_method,
        min_target_pile_size=min_target_pile_size,
        max_target_pile_size=max_target_pile_size,
        target_pile_size=target_pile_size,
        target_metacells_in_pile=target_metacells_in_pile,
        piles_knn_k_size_factor=piles_knn_k_size_factor,
        piles_min_split_size_factor=piles_min_split_size_factor,
        piles_min_robust_size_factor=piles_min_robust_size_factor,
        piles_max_merge_size_factor=piles_max_merge_size_factor,
        direct_parameters=DirectParameters(
            select_downsample_min_samples=select_downsample_min_samples,
            select_downsample_min_cell_quantile=select_downsample_min_cell_quantile,
            select_downsample_max_cell_quantile=select_downsample_max_cell_quantile,
            select_min_gene_total=select_min_gene_total,
            select_min_gene_top3=select_min_gene_top3,
            select_min_gene_relative_variance=select_min_gene_relative_variance,
            select_min_genes=select_min_genes,
            cells_similarity_value_regularization=cells_similarity_value_regularization,
            cells_similarity_log_data=cells_similarity_log_data,
            cells_similarity_method=cells_similarity_method,
            target_metacell_umis=target_metacell_umis,
            cell_umis=cell_umis,
            target_metacell_size=target_metacell_size,
            min_metacell_size=min_metacell_size,
            knn_k=knn_k,
            knn_k_umis_quantile=knn_k_umis_quantile,
            min_knn_k=min_knn_k,
            knn_balanced_ranks_factor=knn_balanced_ranks_factor,
            knn_incoming_degree_factor=knn_incoming_degree_factor,
            knn_outgoing_degree_factor=knn_outgoing_degree_factor,
            knn_min_outgoing_degree=knn_min_outgoing_degree,
            min_seed_size_quantile=min_seed_size_quantile,
            max_seed_size_quantile=max_seed_size_quantile,
            candidates_cooldown_pass=candidates_cooldown_pass,
            candidates_cooldown_node=candidates_cooldown_node,
            candidates_cooldown_phase=candidates_cooldown_phase,
            knn_k_size_factor=candidates_knn_k_size_factor,
            candidates_min_split_size_factor=candidates_min_split_size_factor,
            candidates_max_merge_size_factor=candidates_max_merge_size_factor,
            candidates_max_split_min_cut_strength=candidates_max_split_min_cut_strength,
            candidates_min_cut_seed_cells=candidates_min_cut_seed_cells,
            must_complete_cover=must_complete_cover,
            deviants_policy=deviants_policy,
            deviants_gap_skip_cells=deviants_gap_skip_cells,
            deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
            deviants_min_noisy_gene_fold_factor=deviants_min_noisy_gene_fold_factor,
            deviants_max_gene_fraction=deviants_max_gene_fraction,
            deviants_max_cell_fraction=deviants_max_cell_fraction,
            deviants_max_gap_cells_count=deviants_max_gap_cells_count,
            deviants_max_gap_cells_fraction=deviants_max_gap_cells_fraction,
            dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
            dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
            random_seed=random_seed,
        ),
    )

    _initialize_divide_and_conquer_results(adata)

    collected_mask = np.zeros(adata.n_obs, dtype="bool")

    _compute_divide_and_conquer_subset(
        adata,
        what,
        prefix=ut.get_name(adata) or "full",
        subset_mask=np.full(adata.n_obs, True, dtype="bool"),
        collected_mask=collected_mask,
        metacells_level=0,
        counts=[0],
        dac_parameters=dac_parameters,
        random_seed=random_seed,
    )

    _finalize_divide_and_conquer_results(adata)
    assert np.all(collected_mask)


@ut.logged()
def _initialize_divide_and_conquer_results(adata: AnnData) -> None:
    ut.set_v_data(
        adata,
        "selected_gene",
        np.zeros(adata.n_vars, dtype="bool"),
        formatter=lambda _: "* -> False",
    )
    ut.incremental(adata, "v", "selected_gene")

    for name, value, dtype in (
        ("metacell", -1, "int32"),
        ("dissolved", False, "bool"),
        ("metacell_level", -1, "int32"),
    ):
        ut.incremental(adata, "o", name)
        ut.set_o_data(
            adata,
            name,
            np.full(adata.n_obs, value, dtype=dtype),
            formatter=lambda _: f"* -> {value}",  # pylint: disable=cell-var-from-loop
        )


@ut.logged()
def _reset_subset_results(adata: AnnData, subset_cells_mask: ut.NumpyVector) -> None:
    for name, value, formatter in (("metacell", -1, ut.groups_description),):
        values = ut.get_o_numpy(adata, name, formatter=formatter).copy()
        values[subset_cells_mask] = value
        ut.set_o_data(adata, name, values, formatter=formatter)


@ut.logged()
def _finalize_divide_and_conquer_results(adata: AnnData) -> None:
    ut.done_incrementals(adata)


# NOTE: Any change here must be reflected in _divide_and_conquer_times
@ut.logged()
def _compute_divide_and_conquer_subset(
    adata: AnnData,
    what: str,
    *,
    prefix: str,
    subset_mask: ut.NumpyVector,
    collected_mask: ut.NumpyVector,
    metacells_level: int,
    counts: List[int],
    dac_parameters: DacParameters,
    random_seed: int,
) -> None:
    if not np.any(subset_mask):
        return

    preliminary_pile_of_cells = np.full(adata.n_obs, -1, dtype="int32")
    preliminary_pile_of_cells[subset_mask] = ut.random_piles(
        np.sum(subset_mask),
        target_pile_size=dac_parameters.target_pile_size,
        random_seed=dac_parameters.direct_parameters.random_seed,
    )
    preliminary_piles_count = np.max(preliminary_pile_of_cells) + 1

    time_fractions = _times_to_fractions(
        _divide_and_conquer_times(
            dac_parameters=dac_parameters,
            piles_count=preliminary_piles_count,
            prefix=prefix,
            metacells_level=metacells_level,
        )
    )
    if ut.has_progress_bar():
        time_fractions.reverse()
    else:
        time_fractions = [None] * len(time_fractions)  # type: ignore

    must_cover_dac_parameters = replace(
        dac_parameters, direct_parameters=replace(dac_parameters.direct_parameters, must_complete_cover=True)
    )

    _reset_subset_results(adata, subset_mask)

    if preliminary_piles_count == 1:
        ut.log_calc(f"# {prefix}")
        with ut.progress_bar_slice(time_fractions.pop()):
            _compute_metacells_in_levels(
                adata,
                what,
                prefix=prefix,
                subset_mask=subset_mask,
                collected_mask=collected_mask,
                pile_of_cells=preliminary_pile_of_cells,
                counts=counts,
                metacells_level=metacells_level,
                max_metacells_level=1000,
                dac_parameters=dac_parameters,
            )
        return

    ut.log_calc(f"# {prefix}.preliminary")
    with ut.progress_bar_slice(time_fractions.pop()):
        _compute_metacells_in_levels(
            adata,
            what,
            prefix=prefix + ".preliminary",
            subset_mask=subset_mask,
            collected_mask=np.zeros(adata.n_obs, dtype="bool"),
            pile_of_cells=preliminary_pile_of_cells,
            counts=[0],
            metacells_level=metacells_level,
            max_metacells_level=1000,
            dac_parameters=must_cover_dac_parameters,
        )

    ut.log_calc(f"# {prefix}.groups")
    collect_time = time_fractions.pop()
    groups_time = time_fractions.pop()
    if collect_time is None or groups_time is None:
        total_time = None
    else:
        total_time = collect_time + groups_time
        collect_time /= total_time
        groups_time /= total_time

    with ut.progress_bar_slice(total_time):
        final_pile_of_cells = _compute_metacell_groups(
            adata,
            what,
            collect_time=collect_time,
            groups_time=groups_time,
            prefix=prefix + ".groups",
            subset_mask=subset_mask,
            dac_parameters=must_cover_dac_parameters,
            random_seed=random_seed,
        )

    if dac_parameters.quick_and_dirty:
        ut.log_calc(f"# {prefix}.final")
        with ut.progress_bar_slice(time_fractions.pop()):
            _compute_metacells_in_levels(
                adata,
                what,
                prefix=prefix + ".final",
                subset_mask=subset_mask,
                collected_mask=collected_mask,
                pile_of_cells=final_pile_of_cells,
                counts=counts,
                metacells_level=metacells_level,
                max_metacells_level=1000 if dac_parameters.direct_parameters.must_complete_cover else 1,
                dac_parameters=dac_parameters,
            )
            assert np.all(collected_mask[subset_mask])

    else:
        ut.log_calc(f"# {prefix}.final")
        with ut.progress_bar_slice(time_fractions.pop()):
            _compute_metacells_in_levels(
                adata,
                what,
                prefix=prefix + ".final",
                subset_mask=subset_mask,
                collected_mask=collected_mask,
                pile_of_cells=final_pile_of_cells,
                counts=counts,
                metacells_level=metacells_level,
                max_metacells_level=metacells_level,
                dac_parameters=must_cover_dac_parameters,
            )
            assert np.all(collected_mask[subset_mask])

        if metacells_level == 0 or dac_parameters.direct_parameters.must_complete_cover:
            metacell_of_cells = ut.get_o_numpy(adata, "metacell")
            outliers_mask = subset_mask & (metacell_of_cells < 0)
            collected_mask[outliers_mask] = False

            ut.log_calc(f"# {prefix}.outliers")
            with ut.progress_bar_slice(time_fractions.pop()):
                _compute_divide_and_conquer_subset(
                    adata,
                    what,
                    prefix=prefix + ".outliers",
                    subset_mask=outliers_mask,
                    collected_mask=collected_mask,
                    metacells_level=metacells_level + 1,
                    counts=counts,
                    dac_parameters=dac_parameters,
                    random_seed=random_seed,
                )
                assert np.all(collected_mask[subset_mask])


@ut.logged()
@ut.timed_call()
def _compute_metacells_in_levels(
    adata: AnnData,
    what: str,
    *,
    prefix: str,
    subset_mask: ut.NumpyVector,
    collected_mask: ut.NumpyVector,
    pile_of_cells: ut.NumpyVector,
    counts: List[int],
    metacells_level: int,
    max_metacells_level: int,
    dac_parameters: DacParameters,
) -> None:
    assert metacells_level <= max_metacells_level
    piles_count = np.max(pile_of_cells[subset_mask]) + 1
    assert piles_count > 0

    remaining_time_fraction = 1.0 if ut.has_progress_bar() else None
    outliers_fraction = max(dac_parameters.direct_parameters.deviants_max_cell_fraction or 0.0, 0.05)

    was_last = False
    while True:
        if metacells_level + 1 > max_metacells_level:
            is_last = True
        elif piles_count == 1:
            is_last = was_last
            was_last = True
        else:
            is_last = False

        time_fraction = remaining_time_fraction
        if remaining_time_fraction is not None and not is_last:
            time_fraction = remaining_time_fraction * (1.0 - outliers_fraction)
            remaining_time_fraction *= outliers_fraction

        with ut.progress_bar_slice(time_fraction):
            _compute_metacells_of_piles(
                adata,
                what,
                prefix=f"{prefix}.level.{metacells_level}",
                pile_of_cells=pile_of_cells,
                counts=counts,
                collected_mask=collected_mask,
                direct_parameters=dac_parameters.direct_parameters,
                metacells_level=metacells_level + 1,
            )
            assert np.all(collected_mask[subset_mask])

        if is_last:
            return

        metacells_level += 1
        metacell_of_cells = ut.get_o_numpy(adata, "metacell")
        subset_mask = subset_mask & (metacell_of_cells < 0)
        subset_count = int(np.sum(subset_mask))
        if subset_count == 0:
            return

        collected_mask[subset_mask] = False
        pile_of_cells = np.full(adata.n_obs, -1, dtype="int32")
        pile_of_cells[subset_mask] = ut.random_piles(
            subset_count,
            target_pile_size=dac_parameters.target_pile_size,
            random_seed=dac_parameters.direct_parameters.random_seed,
        )


@ut.logged()
@ut.timed_call()
def _compute_metacell_groups(
    adata: AnnData,
    what: str,
    *,
    collect_time: Optional[float],
    groups_time: Optional[float],
    prefix: str,
    subset_mask: ut.NumpyVector,
    dac_parameters: DacParameters,
    random_seed: int,
) -> ut.NumpyVector:
    metacell_of_cells = ut.get_o_numpy(adata, "metacell")
    assert not np.any(metacell_of_cells[subset_mask] < 0)

    with ut.progress_bar_slice(collect_time):
        sdata = ut.slice(
            adata,
            name=f"{prefix}.grouped",
            top_level=False,
            obs=subset_mask,
            track_obs="__full_cell_index__",
        )

        mdata = collect_metacells(
            sdata,
            what,
            groups=metacell_of_cells[subset_mask],
            name=prefix,
            _metacell_groups=True,
            top_level=False,
            random_seed=random_seed,
        )

    with ut.progress_bar_slice(groups_time):
        metacell_sizes = ut.get_o_numpy(mdata, "grouped").astype("float32")
        target_pile_size = compute_target_pile_size(
            mdata,
            what,
            target_metacell_umis=dac_parameters.target_pile_size,
            cell_umis=metacell_sizes,
            target_metacell_size=dac_parameters.target_metacells_in_pile,
            min_target_pile_size=dac_parameters.min_target_pile_size,
            max_target_pile_size=dac_parameters.max_target_pile_size,
            target_metacells_in_pile=dac_parameters.target_metacells_in_pile,
        )

        group_dac_parameters = replace(
            dac_parameters,
            target_pile_size=target_pile_size,
            direct_parameters=replace(
                dac_parameters.direct_parameters,
                target_metacell_umis=dac_parameters.target_pile_size,
                cell_umis=metacell_sizes,
                target_metacell_size=dac_parameters.target_metacells_in_pile,
                knn_k_size_factor=dac_parameters.piles_knn_k_size_factor,
                candidates_min_split_size_factor=dac_parameters.piles_min_split_size_factor,
                candidates_max_merge_size_factor=dac_parameters.piles_max_merge_size_factor,
                cells_similarity_log_data=dac_parameters.groups_similarity_log_data,
                cells_similarity_method=dac_parameters.groups_similarity_method,
                dissolve_min_robust_size_factor=dac_parameters.piles_min_robust_size_factor,
                must_complete_cover=True,
            ),
        )

        _initialize_divide_and_conquer_results(mdata)
        collected_mask = np.zeros(mdata.n_obs, dtype="bool")

        _compute_divide_and_conquer_subset(
            mdata,
            what,
            prefix=prefix,
            subset_mask=np.full(mdata.n_obs, True, dtype="bool"),
            collected_mask=collected_mask,
            metacells_level=0,
            counts=[0],
            dac_parameters=group_dac_parameters,
            random_seed=random_seed,
        )

        _finalize_divide_and_conquer_results(mdata)
        assert np.all(collected_mask)

        pile_of_metacells = ut.get_o_numpy(mdata, "metacell", formatter=ut.groups_description)
        assert not np.any(pile_of_metacells < 0)

        pile_of_cells = np.full(adata.n_obs, -1, dtype="int32")
        pile_of_cells[subset_mask] = pile_of_metacells[metacell_of_cells[subset_mask]]

    groups_selected_genes_mask = ut.get_v_numpy(mdata, "selected_gene")
    metacells_selected_genes_mask = ut.get_v_numpy(adata, "selected_gene")
    ut.set_v_data(adata, "selected_gene", groups_selected_genes_mask | metacells_selected_genes_mask)

    return pile_of_cells


@ut.logged()
@ut.timed_call()
def _compute_metacells_of_piles(
    adata: AnnData,
    what: str,
    *,
    prefix: str,
    pile_of_cells: ut.NumpyVector,
    counts: List[int],
    collected_mask: ut.NumpyVector,
    direct_parameters: DirectParameters,
    metacells_level: int,
) -> int:
    piles_count = np.max(pile_of_cells) + 1

    @ut.timed_call()
    def _compute_pile_metacells(pile_index: int) -> SubsetResults:
        pile_cells_mask = pile_of_cells == pile_index
        assert np.any(pile_cells_mask)
        pdata = ut.slice(
            adata,
            name=f"{prefix}.{pile_index}/{piles_count}" if piles_count > 1 else prefix,
            top_level=False,
            obs=pile_cells_mask,
            track_obs="__full_cell_index__",
        )
        parameters = direct_parameters.__dict__.copy()
        parameters["cell_umis"] = direct_parameters.cell_umis[pile_cells_mask]
        if piles_count > 1:
            parameters["must_complete_cover"] = False
        with ut.timed_step(".direct"):
            compute_direct_metacells(pdata, what, **parameters)
        subset_results = SubsetResults(pdata)
        pdata = None
        gc.collect()
        return subset_results

    with ut.timed_step(".prepare"):
        ut.get_vo_proper(adata, what, layout="row_major")

    with ut.timed_step(".compute"):
        gc.collect()
        ut.logger().debug("MAX_PARALLEL_PILES: %s", get_max_parallel_piles())
        subset_results_of_piles = ut.parallel_map(
            _compute_pile_metacells,
            piles_count,
            max_processors=get_max_parallel_piles(),
        )

    with ut.timed_step(".collect"):
        for pile_index, subset_results in enumerate(subset_results_of_piles):
            with ut.log_step(
                "- pile",
                pile_index,
                formatter=lambda pile_index: ut.progress_description(piles_count, pile_index, "pile"),
            ):
                subset_results.collect(
                    adata=adata,
                    counts=counts,
                    collected_mask=collected_mask,
                    metacells_level=metacells_level,
                )

    ut.log_calc("collected counts", counts)
    return piles_count


# NOTE: Any change here must be reflected in  _compute_divide_and_conquer_subset
def _divide_and_conquer_times(
    *,
    prefix: str,
    piles_count: int,
    metacells_level: int,
    dac_parameters: DacParameters,
) -> List[float]:
    if piles_count == 1:
        ut.log_calc(f"expected {prefix} single pile time", 1.0)
        return [1.0]

    must_cover_dac_parameters = replace(
        dac_parameters, direct_parameters=replace(dac_parameters.direct_parameters, must_complete_cover=True)
    )

    times = [
        np.sum(
            _levels_times(
                prefix=prefix + ".preliminary",
                piles_count=piles_count,
                metacells_level=metacells_level,
                max_metacells_level=1000,
                dac_parameters=must_cover_dac_parameters,
            )
        ),
        1.0,
        np.sum(
            _group_times(
                prefix=prefix + ".groups",
                piles_count=piles_count,
                dac_parameters=must_cover_dac_parameters,
            )
        ),
    ]

    if dac_parameters.quick_and_dirty:
        times += [
            _levels_times(
                prefix=prefix + ".final",
                piles_count=piles_count,
                metacells_level=metacells_level,
                max_metacells_level=1000
                if dac_parameters.direct_parameters.must_complete_cover
                else metacells_level + 1,
                dac_parameters=dac_parameters,
            )
        ]

    else:
        times += _levels_times(
            prefix=prefix + ".final",
            piles_count=piles_count,
            metacells_level=metacells_level,
            max_metacells_level=metacells_level,
            dac_parameters=dac_parameters,
        )

        if metacells_level == 0 or dac_parameters.direct_parameters.must_complete_cover:
            outliers_fraction = max(dac_parameters.direct_parameters.deviants_max_cell_fraction or 0.0, 0.05)
            outlier_piles_count = max(1, round(piles_count * outliers_fraction))
            times += [
                np.sum(
                    _divide_and_conquer_times(
                        prefix=prefix + ".outliers",
                        piles_count=outlier_piles_count,
                        metacells_level=metacells_level + 1,
                        dac_parameters=dac_parameters,
                    )
                )
            ]

    return times


def _levels_times(
    *,
    prefix: str,
    piles_count: int,
    metacells_level: int,
    max_metacells_level: int,
    dac_parameters: DacParameters,
) -> List[float]:
    assert metacells_level <= max_metacells_level
    assert piles_count > 1

    times: List[float] = []

    while True:
        if metacells_level > max_metacells_level:
            return times

        if piles_count == 1:
            ut.log_calc(f"expected {prefix}.level.{metacells_level} single pile time", 1.0)
            times.append(1.0)
            return times

        piles_time = _piles_time(piles_count)
        ut.log_calc(f"expected {prefix}.level.{metacells_level} {piles_count} piles time", piles_time)
        times.append(piles_time)

        outliers_fraction = max(dac_parameters.direct_parameters.deviants_max_cell_fraction or 0.0, 0.05)
        piles_count = max(1, round(piles_count * outliers_fraction))
        metacells_level += 1


def _group_times(
    *,
    prefix: str,
    piles_count: int,
    dac_parameters: DacParameters,
) -> List[float]:
    assert dac_parameters.direct_parameters.must_complete_cover
    expected_mean_pre_metacells_per_group = dac_parameters.target_metacells_in_pile
    expected_groups_target_pile_size = round(
        expected_mean_pre_metacells_per_group * dac_parameters.target_metacells_in_pile
    )
    expected_groups_target_pile_size = max(expected_groups_target_pile_size, dac_parameters.min_target_pile_size)
    expected_groups_target_pile_size = min(expected_groups_target_pile_size, dac_parameters.max_target_pile_size)
    expected_total_pre_metacells = piles_count * dac_parameters.target_metacells_in_pile
    expected_groups_piles_count = max(1, round(expected_total_pre_metacells / expected_groups_target_pile_size))

    return _divide_and_conquer_times(
        prefix=prefix,
        piles_count=expected_groups_piles_count,
        metacells_level=0,
        dac_parameters=dac_parameters,
    )


def _piles_time(piles_count: int) -> float:
    parallel_processors = ut.get_processors_count()
    if MAX_PARALLEL_PILES > 0:
        parallel_processors = min(parallel_processors, MAX_PARALLEL_PILES)
    return ceil(piles_count / min(parallel_processors, piles_count))


def _times_to_fractions(times: List[float]) -> List[float]:
    total_time = sum(times)
    return [time / total_time for time in times]
