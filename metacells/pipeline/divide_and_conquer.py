"""
Divide and Conquer
------------------
"""

import gc
import logging
import os
from math import ceil
from re import Pattern
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union

import numpy as np
import psutil  # type: ignore
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

from .direct import FeatureCorrection
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
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    target_metacell_size: float = pr.target_metacell_size,
    min_target_pile_size: int = pr.min_target_pile_size,
    max_target_pile_size: int = pr.max_target_pile_size,
    target_metacells_in_pile: int = pr.target_metacells_in_pile,
) -> int:
    """
    Try and guess a reasonable maximal number of piles to use for computing metacells for the
    specified ``cells`` using at most ``max_gbs`` of memory (default: {max_gbs}, that is
    {percent}% of all the machine has - if zero or negative, is relative to the machines memory).

    The amount of memory used depends on the target pile size, so give this function the same parameters as for
    :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline` or
    :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer` so it will use the same automatically adjusted
    target pile size.

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
       :py:func:`metacells.pipeline.divide_and_conquer.compute_target_pile_size` using the
       ``cell_sizes`` (default: {cell_sizes}), ``target_metacell_size`` (default: {target_metacell_size}),
       ``min_target_pile_size`` (default: {min_target_pile_size}), ``max_target_pile_size`` (default:
       {max_target_pile_size}) and ``target_metacells_in_pile`` (default: {target_metacells_in_pile}).
    """
    target_pile_size = compute_target_pile_size(
        cells,
        cell_sizes=cell_sizes,
        target_metacell_size=target_metacell_size,
        min_target_pile_size=min_target_pile_size,
        max_target_pile_size=max_target_pile_size,
        target_metacells_in_pile=target_metacells_in_pile,
    )

    cells_nnz = ut.nnz_matrix(ut.get_vo_proper(cells, what))
    pile_nnz = cells_nnz * target_pile_size / cells.n_obs
    parallel_processes = ut.get_processors_count()
    if max_gbs <= 0:
        assert max_gbs > -1
        max_gbs += 1
        max_gbs *= psutil.virtual_memory().total / (1024.0 * 1024.0 * 1024.0)
    parallel_piles = int((max_gbs - cells_nnz * 6.5e-8 - parallel_processes / 13) / (pile_nnz * 13e-8))
    return max(parallel_piles, 1)


class ResultAnnotation(NamedTuple):
    """
    A per-gene or per-cell annotation used to report results.
    """

    #: The annotation name.
    name: str

    #: The default per-cell value before we collect results.
    default: Any

    #: The data type to use.
    dtype: str

    #: How to log the value.
    formatter: Optional[Callable[[Any], str]]


GENE_ANNOTATIONS = [
    ResultAnnotation(name="pre_high_total_gene", default=0, dtype="int32", formatter=ut.mask_description),
    ResultAnnotation(name="high_total_gene", default=0, dtype="int32", formatter=ut.mask_description),
    ResultAnnotation(name="pre_high_relative_variance_gene", default=0, dtype="int32", formatter=ut.mask_description),
    ResultAnnotation(name="high_relative_variance_gene", default=0, dtype="int32", formatter=ut.mask_description),
    ResultAnnotation(name="forbidden_gene", default=False, dtype="bool", formatter=None),
    ResultAnnotation(name="pre_feature_gene", default=0, dtype="int32", formatter=ut.mask_description),
    ResultAnnotation(name="feature_gene", default=0, dtype="int32", formatter=ut.mask_description),
    ResultAnnotation(name="pre_gene_deviant_votes", default=0, dtype="int32", formatter=ut.mask_description),
    ResultAnnotation(name="gene_deviant_votes", default=0, dtype="int32", formatter=ut.mask_description),
]


CELL_ANNOTATIONS = [
    ResultAnnotation(name="pre_cell_directs", default=0, dtype="int32", formatter=ut.sizes_description),
    ResultAnnotation(name="cell_directs", default=0, dtype="int32", formatter=ut.sizes_description),
    ResultAnnotation(name="pre_pile", default=-1, dtype="int32", formatter=ut.groups_description),
    ResultAnnotation(name="pile", default=0, dtype="int32", formatter=ut.groups_description),
    ResultAnnotation(name="pre_candidate", default=-1, dtype="int32", formatter=ut.groups_description),
    ResultAnnotation(name="candidate", default=-1, dtype="int32", formatter=ut.groups_description),
    ResultAnnotation(name="pre_cell_deviant_votes", default=0, dtype="int32", formatter=ut.mask_description),
    ResultAnnotation(name="cell_deviant_votes", default=0, dtype="int32", formatter=ut.mask_description),
    ResultAnnotation(name="pre_dissolved", default=False, dtype="bool", formatter=ut.mask_description),
    ResultAnnotation(name="dissolved", default=False, dtype="bool", formatter=ut.mask_description),
    ResultAnnotation(name="pre_metacell", default=-1, dtype="int32", formatter=ut.groups_description),
    ResultAnnotation(name="metacell", default=-1, dtype="int32", formatter=ut.groups_description),
    ResultAnnotation(name="outlier", default=True, dtype="bool", formatter=ut.mask_description),
]


class SubsetResults:
    """
    Results of computing metacells for a subset of the cells.

    This data is the only thing that is transmitted from sub-processes computing metacells for
    piles. Everything else done in such sub-processes is lost when the
    :py:func:`metacells.utilities.parallel.parallel_map` is completed.
    """

    def __init__(
        self,
        adata: AnnData,
        *,
        is_direct: bool,
        pre_target: Optional[str],
        final_target: str,
    ) -> None:  #
        """
        Extract the results from ``adata`` which is a subset of the "complete" (clean) data.

        This assumes the data contains a ``complete_cell_index`` per-observation (cell) annotation
        mapping the subset cells back to the complete cell indices.
        """
        assert pre_target in (None, "preliminary")
        assert final_target in ("rare", "preliminary", "final")
        assert pre_target != final_target
        assert not (is_direct and pre_target is not None)

        #: Whether we are collecting annotations from the results of a
        #: :py:func:`metacells.pipeline.direct.compute_direct_metacells`, that is, whether there are
        #: no preliminary annotations in the data.
        self.is_direct = is_direct

        #: Where to collect the preliminary annotations (either ``None`` or ``preliminary``).
        self.pre_target = pre_target

        #: Where to collect the final annotations (either ``preliminary`` or ``rare``/``final``).
        self.final_target = final_target

        if is_direct:
            #: The number of :py:func:`metacells.pipeline.direct.compute_direct_metacells`
            #: invocations in the preliminary phase.
            self.pre_directs = 0

            #: The number of :py:func:`metacells.pipeline.direct.compute_direct_metacells`
            #: invocations in the final phase.
            self.directs = 1

        else:
            self.pre_directs = ut.get_m_data(adata, "pre_directs")
            self.directs = ut.get_m_data(adata, "directs")

        #: The per-gene data.
        #:
        #: This must cover all the genes of the "complete" (clean) data. It must contain a
        #: ``feature_gene`` column, and optionally the ``high_total_gene``,
        #: ``high_relative_variance_gene``, ``forbidden_gene`` and ``gene_deviant_votes`` columns.
        self.genes_frame = ut.to_pandas_frame(index=range(adata.n_vars))

        for gene_annotation in GENE_ANNOTATIONS:
            if self.pre_target is None and gene_annotation.name.startswith("pre_"):
                continue

            self.genes_frame[gene_annotation.name] = ut.get_v_numpy(adata, gene_annotation.name)

        cell_indices = ut.get_o_numpy(adata, "complete_cell_index")

        #: The per-cell data.
        #:
        #: The index contains the indices of the cells in the "complete" (clean) data. It must
        #: contain a ``metacell`` column, and optionally the ``candidate``, ``cell_deviant_votes``,
        #: ``dissolved`` and ``outlier`` columns.
        self.cells_frame = ut.to_pandas_frame(index=cell_indices)

        for cell_annotation in CELL_ANNOTATIONS:
            if (self.pre_target is None and cell_annotation.name.startswith("pre_")) or (
                self.is_direct and cell_annotation.name in ("cell_directs", "pile")
            ):
                continue

            self.cells_frame[cell_annotation.name] = ut.get_o_numpy(
                adata, cell_annotation.name, formatter=cell_annotation.formatter
            )

    def collect(  # pylint: disable=too-many-branches,too-many-statements
        self,
        adata: AnnData,
        counts: Dict[str, int],
        *,
        rare: bool = False,
    ) -> None:
        """
        Collect the results of the subset into the specified ``adata``.

        Use the ``counts`` to ensure all identifiers (piles, metacells) are unique in the end result.
        """

        if self.pre_target is not None:
            assert self.pre_target == "preliminary"
            _modify_value(
                adata, ut.get_m_data, ut.set_m_data, "pre_directs", lambda directs: directs + self.pre_directs
            )

        if self.final_target == "preliminary":
            directs_target_name = "pre_directs"
        else:
            directs_target_name = "directs"
        _modify_value(adata, ut.get_m_data, ut.set_m_data, directs_target_name, lambda directs: directs + self.directs)

        target_name: Optional[str]

        for gene_annotation in GENE_ANNOTATIONS:
            if gene_annotation.name.startswith("pre_"):
                if self.pre_target is None:
                    continue
                assert self.pre_target == "preliminary"
                target_name = gene_annotation.name

            else:
                if self.final_target == "preliminary" and gene_annotation.name != "forbidden_gene":
                    target_name = "pre_" + gene_annotation.name
                else:
                    target_name = gene_annotation.name

            if target_name is None:
                continue

            # pylint: disable=cell-var-from-loop
            new_genes_value = ut.to_numpy_vector(self.genes_frame[gene_annotation.name])

            if gene_annotation.dtype == "bool":
                _modify_value(
                    adata,
                    ut.get_v_numpy,
                    ut.set_v_data,
                    target_name,
                    lambda old_genes_value: old_genes_value | new_genes_value,
                    formatter=gene_annotation.formatter,
                )
            else:
                _modify_value(
                    adata,
                    ut.get_v_numpy,
                    ut.set_v_data,
                    target_name,
                    lambda old_genes_value: old_genes_value + new_genes_value.astype("int32"),
                    formatter=gene_annotation.formatter,
                )
            # pylint: enable=cell-var-from-loop

        cell_indices = self.cells_frame.index.values

        for cell_annotation in CELL_ANNOTATIONS:
            source_name = cell_annotation.name
            if source_name.startswith("pre_"):
                if self.pre_target is not None:
                    assert self.pre_target == "preliminary"
                    target_name = source_name
                elif rare:
                    target_name = source_name
                    source_name = source_name[4:]
                else:
                    target_name = None
            else:
                if self.final_target == "preliminary":
                    if source_name == "outlier":
                        continue
                    target_name = "pre_" + source_name
                else:
                    target_name = source_name

            if target_name is None:
                continue

            if "cell_directs" in source_name:
                # pylint: disable=cell-var-from-loop
                if self.is_direct:
                    assert source_name == "cell_directs"
                    new_value: Union[int, ut.NumpyVector] = 1
                else:
                    new_value = ut.to_numpy_vector(self.cells_frame[source_name])

                _modify_value_subset(
                    adata,
                    ut.get_o_numpy,
                    ut.set_o_data,
                    target_name,
                    cell_indices,
                    lambda old_cells_value: old_cells_value + new_value,
                    formatter=cell_annotation.formatter,
                )

                continue
                # pylint: enable=cell-var-from-loop

            # pylint: disable=cell-var-from-loop
            if self.is_direct and "pile" in source_name:
                new_cells_value = np.zeros(len(self.cells_frame.index), dtype="int32")
            else:
                new_cells_value = ut.to_numpy_vector(self.cells_frame[source_name], copy=rare)

            count = counts.get(target_name)
            if count is not None:
                new_cells_value[new_cells_value >= 0] += count
                counts[target_name] = max(count, np.max(new_cells_value) + 1)

            _modify_value_subset(
                adata,
                ut.get_o_numpy,
                ut.set_o_data,
                target_name,
                cell_indices,
                lambda _: new_cells_value,
                formatter=cell_annotation.formatter,
            )

            # pylint: enable=cell-var-from-loop


@ut.timed_call()
def _initialize_results(
    adata: AnnData,
) -> None:
    ut.incremental(adata, "m", "pre_directs")
    ut.set_m_data(adata, "pre_directs", 0)

    ut.incremental(adata, "m", "directs")
    ut.set_m_data(adata, "directs", 0)

    # pylint: disable=cell-var-from-loop
    for gene_annotation in GENE_ANNOTATIONS:
        ut.incremental(adata, "v", gene_annotation.name, formatter=gene_annotation.formatter)
        ut.set_v_data(
            adata,
            gene_annotation.name,
            np.full(adata.n_vars, gene_annotation.default, dtype=gene_annotation.dtype),
            formatter=lambda _: f"* <- {gene_annotation.default}",
        )
    # pylint: enable=cell-var-from-loop

    # pylint: disable=cell-var-from-loop
    for cell_annotation in CELL_ANNOTATIONS:
        ut.incremental(adata, "o", cell_annotation.name, formatter=cell_annotation.formatter)
        ut.set_o_data(
            adata,
            cell_annotation.name,
            np.full(adata.n_obs, cell_annotation.default, dtype=cell_annotation.dtype),
            formatter=lambda _: f"* <- {cell_annotation.default}",
        )
    # pylint: enable=cell-var-from-loop


@ut.logged()
@ut.timed_call()
def _patch_direct_results(
    adata: AnnData,
) -> None:
    ut.set_m_data(adata, "pre_directs", 0)
    ut.set_m_data(adata, "directs", 1)

    # pylint: disable=cell-var-from-loop
    for gene_annotation in GENE_ANNOTATIONS:
        if not ut.has_data(adata, gene_annotation.name):
            ut.set_v_data(
                adata,
                gene_annotation.name,
                np.full(adata.n_vars, gene_annotation.default, dtype=gene_annotation.dtype),
                formatter=lambda _: f"* <- {gene_annotation.default}",
            )
            continue

        value = ut.get_v_numpy(adata, gene_annotation.name, formatter=gene_annotation.formatter)
        if str(value.dtype) == gene_annotation.dtype:
            continue

        value = value.astype(gene_annotation.dtype)
        ut.set_v_data(adata, gene_annotation.name, value, formatter=gene_annotation.formatter)
    # pylint: enable=cell-var-from-loop

    # pylint: disable=cell-var-from-loop
    for cell_annotation in CELL_ANNOTATIONS:
        if ut.has_data(adata, cell_annotation.name):
            continue
        default = cell_annotation.default
        ut.set_o_data(
            adata,
            cell_annotation.name,
            np.full(adata.n_obs, default, dtype=cell_annotation.dtype),
            formatter=lambda _: f"* <- {default}",
        )
    # pylint: enable=cell-var-from-loop


@ut.logged()
@ut.expand_doc()
def compute_target_pile_size(
    adata: AnnData,
    *,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    target_metacell_size: float = pr.target_metacell_size,
    min_target_pile_size: int = pr.min_target_pile_size,
    max_target_pile_size: int = pr.max_target_pile_size,
    target_metacells_in_pile: int = pr.target_metacells_in_pile,
) -> int:
    """
    Compute how many cells should be in each divide-and-conquer pile.

    Larger piles are slower to process (O(N^2)) than smaller piles, but (up to a point) they allow the algorithm to
    produce better results. The ideal pile size is just big enough to allow for a sufficient number of metacells to be
    created.

    For normal 10X data with a mean (clean genes, clean cells) UMIs count of ~1600 UMIs, and aiming at 100 metacells per
    pile with a mean of 160K UMIs in each metacell, this works out to a target pile size of 10K cells, which by default
    we make the minimal pile size. If the cells are more sparse (e.g., ~800 UMIs) then the pile size will increase to
    ~20K cells which will slow down the algorithm but will improve quality. By default we will not allow the pile size
    to go above 30K as the computation cost starts to be excessive and it isn't clear that one can meaningfully evaluate
    quality for cells with such low number of UMIs.

    Note that the target pile size is just a target. It is directly used in the 1st phase of the divide-and-conquer
    algorithm, but the pile sizes of the 2nd phase are determined by clustering metacells together which inevitably
    cause the actual pile sizes to vary around this number.

    **Computation Parameters**

    0. If no ``cell_sizes`` are specified, return the minimal target pile size.

    1. Pick an ideal pile size such that on average, each pile will give rise to ``target_metacells_in_pile`` (default:
       {target_metacells_in_pile}) such that the size of each one is ``target_metacell_size`` (default:
       {target_metacell_size}) using the specified ``cell_sizes`` (by default, the sum of the data).

    2. Clamp this value to be no lower than ``min_target_pile_size`` (default: {min_target_pile_size}) and no higher
       than ``max_target_pile_size`` (default: {max_target_pile_size}). That is, if you set both to the same value, this
       will force the target pile size value.
    """
    assert 0 < min_target_pile_size <= max_target_pile_size

    if cell_sizes is None or min_target_pile_size == max_target_pile_size:
        ut.log_return("target_pile_size", min_target_pile_size)
        return min_target_pile_size

    cell_sizes = ut.get_o_numpy(adata, cell_sizes, formatter=ut.sizes_description)
    mean_cell_size = np.mean(cell_sizes)
    ut.log_calc("mean_cell_size", mean_cell_size)
    mean_cells_per_metacell = target_metacell_size / mean_cell_size
    ut.log_calc("mean_cells_per_metacell", mean_cells_per_metacell)
    target_pile_size = int(round(mean_cells_per_metacell * target_metacells_in_pile))
    ut.log_calc("raw target_pile_size", target_pile_size)
    target_pile_size = max(target_pile_size, min_target_pile_size)
    target_pile_size = min(target_pile_size, max_target_pile_size)
    ut.log_return("target_pile_size", target_pile_size)
    return target_pile_size


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def divide_and_conquer_pipeline(  # pylint: disable=too-many-branches,too-many-statements
    adata: AnnData,
    what: str = "__x__",
    *,
    rare_max_gene_cell_fraction: float = pr.rare_max_gene_cell_fraction,
    rare_min_gene_maximum: int = pr.rare_min_gene_maximum,
    rare_genes_similarity_method: str = pr.rare_genes_similarity_method,
    rare_genes_cluster_method: str = pr.rare_genes_cluster_method,
    rare_min_genes_of_modules: int = pr.rare_min_genes_of_modules,
    rare_min_cells_of_modules: int = pr.rare_min_cells_of_modules,
    rare_min_modules_size_factor: float = pr.rare_min_modules_size_factor,
    rare_min_module_correlation: float = pr.rare_min_module_correlation,
    rare_min_related_gene_fold_factor: float = pr.rare_min_related_gene_fold_factor,
    rare_max_related_gene_increase_factor: float = pr.rare_max_related_gene_increase_factor,
    rare_min_cell_module_total: int = pr.rare_min_cell_module_total,
    rare_max_cells_factor_of_random_pile: float = pr.rare_max_cells_factor_of_random_pile,
    rare_dissolve_min_robust_size_factor: Optional[float] = pr.rare_dissolve_min_robust_size_factor,
    rare_dissolve_min_convincing_size_factor: Optional[float] = pr.rare_dissolve_min_convincing_size_factor,
    rare_dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    feature_downsample_min_samples: int = pr.feature_downsample_min_samples,
    feature_downsample_min_cell_quantile: float = pr.feature_downsample_min_cell_quantile,
    feature_downsample_max_cell_quantile: float = pr.feature_downsample_max_cell_quantile,
    feature_min_gene_total: Optional[int] = pr.feature_min_gene_total,
    feature_min_gene_top3: Optional[int] = pr.feature_min_gene_top3,
    feature_min_gene_relative_variance: Optional[float] = pr.feature_min_gene_relative_variance,
    feature_gene_names: Optional[Collection[str]] = None,
    feature_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    forbidden_gene_names: Optional[Collection[str]] = None,
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    feature_correction: Optional[FeatureCorrection] = None,
    cells_similarity_value_normalization: float = pr.cells_similarity_value_normalization,
    cells_similarity_log_data: bool = pr.cells_similarity_log_data,
    cells_similarity_method: str = pr.cells_similarity_method,
    groups_similarity_log_data: bool = pr.groups_similarity_log_data,
    groups_similarity_method: str = pr.groups_similarity_method,
    min_target_pile_size: int = pr.min_target_pile_size,
    max_target_pile_size: int = pr.max_target_pile_size,
    target_metacells_in_pile: int = pr.target_metacells_in_pile,
    max_cell_size: Optional[float] = pr.max_cell_size,
    max_cell_size_factor: Optional[float] = pr.max_cell_size_factor,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    pile_min_split_size_factor: float = pr.pile_min_split_size_factor,
    pile_min_robust_size_factor: float = pr.pile_min_robust_size_factor,
    pile_max_merge_size_factor: float = pr.pile_max_merge_size_factor,
    target_metacell_size: float = pr.target_metacell_size,
    knn_k: Optional[int] = pr.knn_k,
    min_knn_k: Optional[int] = pr.min_knn_k,
    knn_balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    knn_incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    knn_outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    min_seed_size_quantile: float = pr.min_seed_size_quantile,
    max_seed_size_quantile: float = pr.max_seed_size_quantile,
    candidates_cooldown_pass: float = pr.cooldown_pass,
    candidates_cooldown_node: float = pr.cooldown_node,
    candidates_cooldown_phase: float = pr.cooldown_phase,
    candidates_min_split_size_factor: Optional[float] = pr.candidates_min_split_size_factor,
    candidates_max_merge_size_factor: Optional[float] = pr.candidates_max_merge_size_factor,
    candidates_min_metacell_cells: Optional[int] = pr.candidates_min_metacell_cells,
    candidates_max_split_min_cut_strength: Optional[float] = pr.max_split_min_cut_strength,
    candidates_min_cut_seed_cells: Optional[int] = pr.min_cut_seed_cells,
    must_complete_cover: bool = False,
    final_max_outliers_levels: Optional[int] = pr.final_max_outliers_levels,
    deviants_min_gene_fold_factor: float = pr.deviants_min_gene_fold_factor,
    deviants_abs_folds: bool = pr.deviants_abs_folds,
    deviants_max_gene_fraction: Optional[float] = pr.deviants_max_gene_fraction,
    deviants_max_cell_fraction: Optional[float] = pr.deviants_max_cell_fraction,
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    dissolve_min_metacell_cells: int = pr.dissolve_min_metacell_cells,
    random_seed: int = pr.random_seed,
) -> None:
    """
    Complete pipeline using divide-and-conquer to compute the metacells for the ``what`` (default:
    {what}) data.

    If a progress bar is active, progress will be reported into the current (slice of) the progress bar. Finding rate
    genes is not reported into the progress bar; instead it is logged at the ``INFO`` level.

    .. note::

        This is applicable to "any size" data. If the data is "small" (O(10,000)), it will revert to
        using the direct metacell computation (but will still by default first look for rare gene
        modules). If the data is "large" (up to O(10,000,000)), this will be much faster and will
        require much less memory than using the direct approach. The current implementation is not
        optimized for "huge" data (O(1,000,000,000)) - that is, it will work, and keep will use a
        limited amount of memory, but a faster implementation would distribute the computation
        across multiple servers.

    **Input**

    The presumably "clean" annotated ``adata``, where the observations are cells and the variables
    are genes, where ``what`` is a per-variable-per-observation matrix or the name of a
    per-variable-per-observation annotation containing such a matrix.

    **Returns**

    Sets the following annotations in ``adata``:

    Unstructured Annotations
        ``pre_directs``, ``directs``
            The number of times we invoked
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` for computing the
            preliminary and final metacells. If we end up directly computing the metacells, the
            preliminary value will be zero, and the final value will be one. This can be used to
            normalize the ``pre_<name>`` and ``<name>`` properties below to fractions
            (probabilities).

    Variable (Gene) Annotations
        ``rare_gene_module_<N>``
            A boolean mask for the genes in the rare gene module ``N``.

        ``rare_gene``
            A boolean mask for the genes in any of the rare gene modules.

        ``pre_high_total_gene``, ``high_total_gene``
            The number of times the gene was marked as having a high expression level when computing
            the preliminary and final metacells. This is zero for non-"clean" genes.

        ``pre_high_relative_variance_gene``, ``high_relative_variance_gene``
            The number of times the gene was marked as having a high normalized variance relative to
            other genes with a similar expression level when when computing the preliminary and
            final metacells. This is zero for non-"clean" genes.

        ``forbidden_gene``
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name. This is ``False`` for non-"clean" genes.

        ``pre_feature_gene``, ``feature_gene``
            The number of times the gene was used as a feature when computing the preliminary and
            final metacells. If we end up directly computing the metacells, the preliminary value
            will be all-zero, and the final value will be one for feature genes, zero otherwise.

        ``pre_gene_deviant_votes``, ``gene_deviant_votes``
            The total number of cells each gene marked as deviant (if zero, the gene did not mark
            any cell as deviant) when computing the preliminary and final metacells. This will be
            zero for non-"feature" genes.

    Observations (Cell) Annotations
        ``cells_rare_gene_module``
            The index of the rare gene module each cell expresses the most, or ``-1`` in the common
            case it does not express any rare genes module.

        ``rare_cell``
            A boolean mask for the (few) cells that express a rare gene module.

        ``pre_cell_directs``, ``cell_directs``
            The number of times we invoked
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` to try and group this cell
            when computing the preliminary and final metacells. If we end up directly computing the
            metacells, the preliminary value will be zero. Otherwise this will be at least one, with
            higher values for cells which we tried to group again in the outlier pile(s).

        ``pre_pile``, ``pile``
            The index of the pile the cell was in in the last invocation of
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` when computing the
            preliminary and final metacells. This is ``-1`` for non-"clean" cells. The preliminary
            value is likewise ``-1`` if we end up directly computing the metacells.

        ``pre_candidate``, ``candidate``
            The index of the candidate metacell each cell was assigned to to in the last grouping
            attempt when computing the preliminary and final metacells. This is ``-1`` for
            non-"clean" cells.

        ``pre_cell_deviant_votes``, ``cell_deviant_votes``
            The number of genes that were the reason the cell was marked as deviant in the last
            grouping attempt when computing the preliminary and final metacells (if zero, the cell
            is not deviant). This is zero for non-"clean" cells.

        ``pre_dissolved``, ``dissolved``
            A boolean mask of the cells contained in a dissolved metacell in the last grouping
            attempt when computing the preliminary and final metacells. This is ``False`` for
            non-"clean" cells.

        ``pre_metacell``, ``metacell``
            The integer index of the preliminary and final metacell each cell belongs to. The
            metacells are in no particular order. This is ``-1`` for outlier cells and ``-2`` for
            non-"clean" cells.

        ``outlier``
            A boolean mask of the cells not assigned to any final metacell. We assign every
            ("clean") cell to a preliminary metacell so there is no point in providing a
            ``pre_outlier`` mask (it would be identical to the ``clean_cell`` mask).

    **Computation Parameters**

    0. Invoke :py:func:`metacells.pipeline.divide_and_conquer.compute_target_pile_size` using ``target_metacell_size``
       (default: {target_metacell_size}), ``min_target_pile_size`` (default: {min_target_pile_size}),
       ``max_target_pile_size`` (default: {max_target_pile_size}) and ``target_metacells_in_pile`` (default:
       {target_metacells_in_pile}).

    1. Invoke :py:func:`metacells.tools.rare.find_rare_gene_modules` to isolate cells expressing
       rare gene modules, using the
       ``forbidden_gene_names``, ``forbidden_gene_patterns``,
       ``rare_max_gene_cell_fraction`` (default: {rare_max_gene_cell_fraction}),
       ``rare_min_gene_maximum`` (default: {rare_min_gene_maximum}),
       ``rare_genes_similarity_method`` (default: {rare_genes_similarity_method}),
       ``rare_genes_cluster_method`` (default: {rare_genes_cluster_method}),
       ``rare_min_genes_of_modules`` (default: {rare_min_genes_of_modules}),
       ``rare_min_cells_of_modules`` (default: {rare_min_cells_of_modules}),
       ``rare_min_modules_size_factor`` (default: {rare_min_modules_size_factor}),
       ``rare_min_module_correlation`` (default: {rare_min_module_correlation}),
       ``rare_min_related_gene_fold_factor`` (default: {rare_min_related_gene_fold_factor})
       ``rare_max_related_gene_increase_factor`` (default: {rare_max_related_gene_increase_factor})
       ``rare_max_cells_factor_of_random_pile`` (default: {rare_max_cells_factor_of_random_pile})
       and
       ``rare_min_cell_module_total`` (default: {rare_min_cell_module_total}).

    2. For each detected rare gene module, collect all cells that express the module, and invoke
       :py:func:`metacells.pipeline.direct.compute_direct_metacells` to compute metacells for them, using the rest of
       the parameters. Here we allow using a lower ``rare_dissolve_min_robust_size_factor`` (default:
       {rare_dissolve_min_robust_size_factor}), ``rare_dissolve_min_convincing_size_factor`` (default:
       {rare_dissolve_min_convincing_size_factor}) and ``rare_dissolve_min_convincing_size_factor`` (default:
       {rare_dissolve_min_convincing_size_factor}) because even "weak" metacells detected here are known be relatively
       "convincing" based on them being different from the overall population.

       .. todo::

            Technically we should invoke :py:func:`compute_divide_and_conquer_metacells` for each group of cells that
            express a rare gene module. However, even for very large data sets (O(1,000,000) cells) the number of such
            cells is expected to be small (O(1,000) cells) so directly computing metacells for them seems reasonable,
            and it was easier to (ab)use the "compute metacells in parallel for piles" here, rather than create a new
            mechanism.

    3. Collect all cells that either did not express any rare gene module, or did but were considered an outlier in the
       previous step, and invoke :py:func:`compute_divide_and_conquer_metacells` to compute metacells for them, using
       the rest of the parameters.

    4. Combine the results from the previous two steps.
    """
    target_pile_size = compute_target_pile_size(
        adata,
        cell_sizes=cell_sizes,
        target_metacell_size=target_metacell_size,
        min_target_pile_size=min_target_pile_size,
        max_target_pile_size=max_target_pile_size,
        target_metacells_in_pile=target_metacells_in_pile,
    )

    counts = dict(pre_pile=0, pile=0, pre_candidate=0, candidate=0, pre_metacell=0, metacell=0)

    did_apply_subset = False
    normal_cells_mask = np.full(adata.n_obs, True, dtype="bool")

    try:
        if ut.has_progress_bar():
            logger = ut.logger()
            log_level = logger.level
            logger.setLevel(logging.INFO)
            logger.info("Compute metacells for rare gene modules...")
            logger.setLevel(log_level)

        with ut.timed_step(".rare"):
            tl.find_rare_gene_modules(
                adata,
                what,
                forbidden_gene_names=forbidden_gene_names,
                forbidden_gene_patterns=forbidden_gene_patterns,
                max_gene_cell_fraction=rare_max_gene_cell_fraction,
                min_gene_maximum=rare_min_gene_maximum,
                genes_similarity_method=rare_genes_similarity_method,
                genes_cluster_method=rare_genes_cluster_method,
                min_genes_of_modules=rare_min_genes_of_modules,
                min_cells_of_modules=rare_min_cells_of_modules,
                target_metacell_size=target_metacell_size,
                min_modules_size_factor=rare_min_modules_size_factor,
                min_module_correlation=rare_min_module_correlation,
                min_related_gene_fold_factor=rare_min_related_gene_fold_factor,
                max_related_gene_increase_factor=rare_max_related_gene_increase_factor,
                target_pile_size=target_pile_size,
                max_cells_factor_of_random_pile=rare_max_cells_factor_of_random_pile,
                min_cell_module_total=rare_min_cell_module_total,
                reproducible=(random_seed != 0),
            )

            rare_module_of_cells = ut.get_o_numpy(adata, "cells_rare_gene_module", formatter=ut.groups_description)
            rare_modules_count = np.max(rare_module_of_cells) + 1
            if rare_modules_count > 0:
                subset_results = _run_parallel_piles(
                    adata,
                    what,
                    phase="rare",
                    piles_count=rare_modules_count,
                    pile_of_cells=rare_module_of_cells,
                    feature_downsample_min_samples=feature_downsample_min_samples,
                    feature_downsample_min_cell_quantile=feature_downsample_min_cell_quantile,
                    feature_downsample_max_cell_quantile=feature_downsample_max_cell_quantile,
                    feature_min_gene_total=feature_min_gene_total,
                    feature_min_gene_top3=feature_min_gene_top3,
                    feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                    feature_gene_names=feature_gene_names,
                    feature_gene_patterns=feature_gene_patterns,
                    forbidden_gene_names=forbidden_gene_names,
                    forbidden_gene_patterns=forbidden_gene_patterns,
                    feature_correction=feature_correction,
                    cells_similarity_value_normalization=cells_similarity_value_normalization,
                    cells_similarity_log_data=cells_similarity_log_data,
                    cells_similarity_method=cells_similarity_method,
                    target_metacell_size=target_metacell_size,
                    max_cell_size=max_cell_size,
                    max_cell_size_factor=max_cell_size_factor,
                    cell_sizes=cell_sizes,
                    knn_k=knn_k,
                    min_knn_k=min_knn_k,
                    knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                    knn_incoming_degree_factor=knn_incoming_degree_factor,
                    knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                    min_seed_size_quantile=min_seed_size_quantile,
                    max_seed_size_quantile=max_seed_size_quantile,
                    candidates_cooldown_pass=candidates_cooldown_pass,
                    candidates_cooldown_node=candidates_cooldown_node,
                    candidates_cooldown_phase=candidates_cooldown_phase,
                    candidates_min_split_size_factor=candidates_min_split_size_factor,
                    candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                    candidates_min_metacell_cells=candidates_min_metacell_cells,
                    candidates_max_split_min_cut_strength=candidates_max_split_min_cut_strength,
                    candidates_min_cut_seed_cells=candidates_min_cut_seed_cells,
                    must_complete_cover=False,
                    deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                    deviants_abs_folds=deviants_abs_folds,
                    deviants_max_gene_fraction=deviants_max_gene_fraction,
                    deviants_max_cell_fraction=deviants_max_cell_fraction,
                    dissolve_min_robust_size_factor=rare_dissolve_min_robust_size_factor,
                    dissolve_min_convincing_size_factor=rare_dissolve_min_convincing_size_factor,
                    dissolve_min_convincing_gene_fold_factor=rare_dissolve_min_convincing_gene_fold_factor,
                    dissolve_min_metacell_cells=dissolve_min_metacell_cells,
                    random_seed=random_seed,
                    hide_from_progress_bar=True,
                )

                with ut.timed_step(".collect_piles"):
                    ut.log_calc("collect rare modules results:")
                    for rare_index, rare_results in enumerate(subset_results):
                        with ut.log_step(
                            "- module",
                            rare_index,
                            formatter=lambda rare_index: ut.progress_description(
                                len(subset_results), rare_index, "module"
                            ),
                        ):
                            rare_cells_mask = rare_results.cells_frame["metacell"] >= 0
                            ut.log_calc("rare_grouped_cells", rare_cells_mask)
                            if not np.any(rare_cells_mask):
                                continue

                            if not did_apply_subset:
                                did_apply_subset = True
                                _initialize_results(adata)

                            cell_indices = rare_results.cells_frame.index.values[rare_cells_mask]
                            normal_cells_mask[cell_indices] = False

                            rare_results.collect(adata, counts, rare=True)

        if did_apply_subset:
            cdata = ut.slice(
                adata, name=".common", top_level=False, obs=normal_cells_mask, track_obs="complete_cell_index"
            )
        else:
            cdata = adata

        if ut.has_progress_bar():
            logger = ut.logger()
            log_level = logger.level
            logger.setLevel(logging.INFO)
            logger.info("Compute common metacells...")
            logger.setLevel(log_level)

        is_divide_and_conquer = compute_divide_and_conquer_metacells(
            cdata,
            what,
            feature_downsample_min_samples=feature_downsample_min_samples,
            feature_downsample_min_cell_quantile=feature_downsample_min_cell_quantile,
            feature_downsample_max_cell_quantile=feature_downsample_max_cell_quantile,
            feature_min_gene_total=feature_min_gene_total,
            feature_min_gene_top3=feature_min_gene_top3,
            feature_min_gene_relative_variance=feature_min_gene_relative_variance,
            feature_gene_names=feature_gene_names,
            feature_gene_patterns=feature_gene_patterns,
            forbidden_gene_names=forbidden_gene_names,
            forbidden_gene_patterns=forbidden_gene_patterns,
            feature_correction=feature_correction,
            cells_similarity_value_normalization=cells_similarity_value_normalization,
            cells_similarity_log_data=cells_similarity_log_data,
            cells_similarity_method=cells_similarity_method,
            groups_similarity_log_data=groups_similarity_log_data,
            groups_similarity_method=groups_similarity_method,
            min_target_pile_size=min_target_pile_size,
            max_target_pile_size=max_target_pile_size,
            target_metacells_in_pile=target_metacells_in_pile,
            pile_min_split_size_factor=pile_min_split_size_factor,
            pile_min_robust_size_factor=pile_min_robust_size_factor,
            pile_max_merge_size_factor=pile_max_merge_size_factor,
            target_metacell_size=target_metacell_size,
            max_cell_size=max_cell_size,
            max_cell_size_factor=max_cell_size_factor,
            cell_sizes=cell_sizes,
            knn_k=knn_k,
            min_knn_k=min_knn_k,
            knn_balanced_ranks_factor=knn_balanced_ranks_factor,
            knn_incoming_degree_factor=knn_incoming_degree_factor,
            knn_outgoing_degree_factor=knn_outgoing_degree_factor,
            min_seed_size_quantile=min_seed_size_quantile,
            max_seed_size_quantile=max_seed_size_quantile,
            candidates_cooldown_pass=candidates_cooldown_pass,
            candidates_cooldown_node=candidates_cooldown_node,
            candidates_cooldown_phase=candidates_cooldown_phase,
            candidates_min_split_size_factor=candidates_min_split_size_factor,
            candidates_max_merge_size_factor=candidates_max_merge_size_factor,
            candidates_min_metacell_cells=candidates_min_metacell_cells,
            candidates_max_split_min_cut_strength=candidates_max_split_min_cut_strength,
            candidates_min_cut_seed_cells=candidates_min_cut_seed_cells,
            must_complete_cover=must_complete_cover,
            final_max_outliers_levels=final_max_outliers_levels,
            deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
            deviants_abs_folds=deviants_abs_folds,
            deviants_max_gene_fraction=deviants_max_gene_fraction,
            deviants_max_cell_fraction=deviants_max_cell_fraction,
            dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
            dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
            dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
            dissolve_min_metacell_cells=dissolve_min_metacell_cells,
            random_seed=random_seed,
        )

        if did_apply_subset:
            with ut.log_step("collect common cells results:"):
                common_results = SubsetResults(
                    cdata,
                    is_direct=not is_divide_and_conquer,
                    pre_target="preliminary" if is_divide_and_conquer else None,
                    final_target="final",
                )
                common_results.collect(adata, counts)

        if did_apply_subset:
            ut.done_incrementals(adata)
    finally:
        if did_apply_subset:
            ut.cancel_incrementals(adata)


@ut.logged()
def compute_divide_and_conquer_metacells(
    adata: AnnData,
    what: str = "__x__",
    *,
    feature_downsample_min_samples: int = pr.feature_downsample_min_samples,
    feature_downsample_min_cell_quantile: float = pr.feature_downsample_min_cell_quantile,
    feature_downsample_max_cell_quantile: float = pr.feature_downsample_max_cell_quantile,
    feature_min_gene_total: Optional[int] = pr.feature_min_gene_total,
    feature_min_gene_top3: Optional[int] = pr.feature_min_gene_top3,
    feature_min_gene_relative_variance: Optional[float] = pr.feature_min_gene_relative_variance,
    feature_gene_names: Optional[Collection[str]] = None,
    feature_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    forbidden_gene_names: Optional[Collection[str]] = None,
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    feature_correction: Optional[FeatureCorrection] = None,
    cells_similarity_value_normalization: float = pr.cells_similarity_value_normalization,
    cells_similarity_log_data: bool = pr.cells_similarity_log_data,
    cells_similarity_method: str = pr.cells_similarity_method,
    groups_similarity_log_data: bool = pr.groups_similarity_log_data,
    groups_similarity_method: str = pr.groups_similarity_method,
    min_target_pile_size: int = pr.min_target_pile_size,
    max_target_pile_size: int = pr.max_target_pile_size,
    target_metacells_in_pile: int = pr.target_metacells_in_pile,
    max_cell_size: Optional[float] = pr.max_cell_size,
    max_cell_size_factor: Optional[float] = pr.max_cell_size_factor,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    pile_min_split_size_factor: float = pr.pile_min_split_size_factor,
    pile_min_robust_size_factor: float = pr.pile_min_robust_size_factor,
    pile_max_merge_size_factor: float = pr.pile_max_merge_size_factor,
    target_metacell_size: float = pr.target_metacell_size,
    knn_k: Optional[int] = pr.knn_k,
    min_knn_k: Optional[int] = pr.min_knn_k,
    knn_balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    knn_incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    knn_outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    min_seed_size_quantile: float = pr.min_seed_size_quantile,
    max_seed_size_quantile: float = pr.max_seed_size_quantile,
    candidates_cooldown_pass: float = pr.cooldown_pass,
    candidates_cooldown_node: float = pr.cooldown_node,
    candidates_cooldown_phase: float = pr.cooldown_phase,
    candidates_min_split_size_factor: Optional[float] = pr.candidates_min_split_size_factor,
    candidates_max_merge_size_factor: Optional[float] = pr.candidates_max_merge_size_factor,
    candidates_min_metacell_cells: Optional[int] = pr.min_metacell_cells,
    candidates_max_split_min_cut_strength: Optional[float] = pr.max_split_min_cut_strength,
    candidates_min_cut_seed_cells: Optional[int] = pr.min_cut_seed_cells,
    must_complete_cover: bool = False,
    final_max_outliers_levels: Optional[int] = pr.final_max_outliers_levels,
    deviants_min_gene_fold_factor: float = pr.deviants_min_gene_fold_factor,
    deviants_abs_folds: bool = pr.deviants_abs_folds,
    deviants_max_gene_fraction: Optional[float] = pr.deviants_max_gene_fraction,
    deviants_max_cell_fraction: Optional[float] = pr.deviants_max_cell_fraction,
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    dissolve_min_metacell_cells: int = pr.dissolve_min_metacell_cells,
    random_seed: int = pr.random_seed,
) -> bool:
    """
    Compute metacells for ``what`` (default: {what}) data using the divide-and-conquer method.

    This divides large data to smaller "piles" and directly computes metacells for each, then
    generates new piles of "similar" metacells and directly computes the final metacells from each
    such pile. Due to this divide-and-conquer approach, the total amount of memory required is
    bounded (by the pile size), and the amount of total CPU grows slowly with the problem size,
    allowing this method to be applied to very large data (millions of cells).

    If a progress bar is active, progress will be reported into the current (slice of) the progress bar.

    .. todo::

        The divide-and-conquer implementation is restricted for using multiple CPUs on a single
        shared-memory machine. At some problem size (probably around a billion cells), it would make
        sense to modify it to support distributed execution of sub-problems on different servers.
        This would only be an implementation change, rather than an algorithmic change.

    **Input**

    The presumably "clean" annotated ``adata``, where the observations are cells and the variables
    are genes, where ``what`` is a per-variable-per-observation matrix or the name of a
    per-variable-per-observation annotation containing such a matrix.

    **Returns**

    ``True`` if actually used divide-and-conquer, ``False`` if used the direct algorithm.

    Sets the following annotations in ``adata``:

    Structured Annotations
        ``pre_directs``, ``directs``
            The number of times we invoked
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` for computing the
            preliminary and final metacells. If we end up directly computing the metacells, the
            preliminary value will be zero, and the final value will be one. This can be used to
            normalize the ``pre_<name>`` and ``<name>`` properties below to fractions
            (probabilities).

    Variable (Gene) Annotations
        ``pre_high_total_gene``, ``high_total_gene``
            The number of times the gene was marked as having a high expression level when computing
            the preliminary and final metacells. This is zero for non-"clean" genes.

        ``pre_high_relative_variance_gene``, ``high_relative_variance_gene``
            The number of times the gene was marked as having a high normalized variance relative to
            other genes with a similar expression level when when computing the preliminary and
            final metacells. This is zero for non-"clean" genes.

        ``forbidden_gene``
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name. This is ``False`` for non-"clean" genes.

        ``pre_feature_gene``, ``feature_gene``
            The number of times the gene was used as a feature when computing the preliminary and
            final metacells. If we end up directly computing the metacells, the preliminary value
            will be all-zero, and the final value will be one for feature genes, zero otherwise.

        ``pre_gene_deviant_votes``, ``gene_deviant_votes``
            The total number of cells each gene marked as deviant (if zero, the gene did not mark
            any cell as deviant) when computing the preliminary and final metacells. This will be
            zero for non-"feature" genes.

    Observations (Cell) Annotations
        ``pre_cell_directs``, ``cell_directs``
            The number of times we invoked
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` to try and group this cell
            when computing the preliminary and final metacells. If we end up directly computing the
            metacells, the preliminary value will be zero. Otherwise this will be at least one, with
            higher values for cells which we tried to group again in the outlier pile(s).

        ``pre_pile``, ``pile``
            The index of the pile the cell was in in the last invocation of
            :py:func:`metacells.pipeline.direct.compute_direct_metacells` when computing the
            preliminary and final metacells. This is ``-1`` for non-"clean" cells. The preliminary
            value is likewise ``-1`` if we end up directly computing the metacells.

        ``pre_candidate``, ``candidate``
            The index of the candidate metacell each cell was assigned to to in the last grouping
            attempt when computing the preliminary and final metacells. This is ``-1`` for
            non-"clean" cells.

        ``pre_cell_deviant_votes``, ``cell_deviant_votes``
            The number of genes that were the reason the cell was marked as deviant in the last
            grouping attempt when computing the preliminary and final metacells (if zero, the cell
            is not deviant). This is zero for non-"clean" cells.

        ``pre_dissolved``, ``dissolved``
            A boolean mask of the cells contained in a dissolved metacell in the last grouping
            attempt when computing the preliminary and final metacells. This is ``False`` for
            non-"clean" cells.

        ``pre_metacell``, ``metacell``
            The integer index of the preliminary and final metacell each cell belongs to. The
            metacells are in no particular order. This is ``-1`` for outlier cells and ``-2`` for
            non-"clean" cells.

        ``outlier``
            A boolean mask of the cells not assigned to any final metacell. We assign every
            ("clean") cell to a preliminary metacell so there is no point in providing a
            ``pre_outlier`` mask (it would be identical to the ``clean_cell`` mask).

    **Computation Parameters**

    0. Invoke :py:func:`metacells.pipeline.divide_and_conquer.compute_target_pile_size` using ``target_metacell_size``
       (default: {target_metacell_size}), ``min_target_pile_size`` (default: {min_target_pile_size}),
       ``max_target_pile_size`` (default: {max_target_pile_size}) and ``target_metacells_in_pile`` (default:
       {target_metacells_in_pile}).

    1. If the data is smaller than the target_pile_size times the ``pile_min_split_size_factor`` (default:
       {pile_min_split_size_factor}), then just invoke :py:func:`metacells.pipeline.direct.compute_direct_metacells`
       using the parameters, patch the results to contain the expected annotations from a divide-and-conquer call, and
       return. Otherwise, perform the following steps.

    3. Group the cells randomly into equal-sized piles of roughly the ``target_pile_size`` using the ``random_seed``
       (default: {random_seed}) to allow making this replicable.

    4. Compute preliminary metacells using the random piles and the parameters. Here we set ``must_complete_cover`` to
       ``True`` to ensure each cell is placed in some preliminary metacell so all cells will be in some pile in the next
       phase.

    5. Invoke :py:func:`metacells.tools.group.group_obs_data` to sum the cells into preliminary metacells.

    6. Invoke :py:func:`compute_divide_and_conquer_metacells` using the parameters to group the
       preliminary metacells into piles. We use here ``pile_min_split_size_factor`` (default:
       {pile_min_split_size_factor}), ``pile_min_robust_size_factor`` (default:
       {pile_min_robust_size_factor}) and ``pile_max_merge_size_factor`` (default:
       {pile_max_merge_size_factor}), so control over the pile size (number of cells) is separate
       from control over the metacell size (number of UMIs).

    7. Compute the final metacells using the preliminary metacell piles, using using ``final_max_outliers_levels`` to
       prevent forcing outlier cells to be placed in low-quality metacells.
    """
    target_pile_size = compute_target_pile_size(
        adata,
        cell_sizes=cell_sizes,
        target_metacell_size=target_metacell_size,
        min_target_pile_size=min_target_pile_size,
        max_target_pile_size=max_target_pile_size,
        target_metacells_in_pile=target_metacells_in_pile,
    )

    random_pile_of_cells = ut.random_piles(adata.n_obs, target_pile_size=target_pile_size, random_seed=random_seed)
    piles_count = np.max(random_pile_of_cells) + 1
    ut.log_calc("piles", piles_count)

    expected_total_metacells = piles_count * target_metacells_in_pile
    expected_total_groups = expected_total_metacells / target_metacells_in_pile
    expected_total_group_piles = expected_total_groups / target_metacells_in_pile
    expected_group_piles_per_pile = expected_total_group_piles / piles_count

    if piles_count < 2:
        with ut.timed_step(".direct"):
            compute_direct_metacells(
                adata,
                what,
                feature_downsample_min_samples=feature_downsample_min_samples,
                feature_downsample_min_cell_quantile=feature_downsample_min_cell_quantile,
                feature_downsample_max_cell_quantile=feature_downsample_max_cell_quantile,
                feature_min_gene_total=feature_min_gene_total,
                feature_min_gene_top3=feature_min_gene_top3,
                feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                feature_gene_names=feature_gene_names,
                feature_gene_patterns=feature_gene_patterns,
                forbidden_gene_names=forbidden_gene_names,
                forbidden_gene_patterns=forbidden_gene_patterns,
                feature_correction=feature_correction,
                cells_similarity_value_normalization=cells_similarity_value_normalization,
                cells_similarity_log_data=cells_similarity_log_data,
                cells_similarity_method=cells_similarity_method,
                target_metacell_size=target_metacell_size,
                max_cell_size=max_cell_size,
                max_cell_size_factor=max_cell_size_factor,
                cell_sizes=cell_sizes,
                knn_k=knn_k,
                min_knn_k=min_knn_k,
                knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                knn_incoming_degree_factor=knn_incoming_degree_factor,
                knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                min_seed_size_quantile=min_seed_size_quantile,
                max_seed_size_quantile=max_seed_size_quantile,
                candidates_cooldown_pass=candidates_cooldown_pass,
                candidates_cooldown_node=candidates_cooldown_node,
                candidates_cooldown_phase=candidates_cooldown_phase,
                candidates_min_split_size_factor=candidates_min_split_size_factor,
                candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                candidates_min_metacell_cells=candidates_min_metacell_cells,
                candidates_max_split_min_cut_strength=candidates_max_split_min_cut_strength,
                candidates_min_cut_seed_cells=candidates_min_cut_seed_cells,
                must_complete_cover=must_complete_cover,
                deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                deviants_abs_folds=deviants_abs_folds,
                deviants_max_gene_fraction=deviants_max_gene_fraction,
                deviants_max_cell_fraction=deviants_max_cell_fraction,
                dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                dissolve_min_metacell_cells=dissolve_min_metacell_cells,
                random_seed=random_seed,
            )

            _patch_direct_results(adata)

        return False

    if not ut.has_progress_bar():
        preliminary_progress_fraction: Optional[float] = None
        groups_progress_fraction: Optional[float] = None
        final_progress_fraction: Optional[float] = None
    else:
        preliminary_progress_fraction, groups_progress_fraction, final_progress_fraction = _progress_fractions(
            _divide_and_conquer_times(
                piles_count=len(random_pile_of_cells),
                deviants_max_cell_fraction=deviants_max_cell_fraction,
                final_max_outliers_levels=final_max_outliers_levels,
                expected_group_piles_per_pile=expected_group_piles_per_pile,
            )
        )

    _initialize_results(adata)
    try:
        with ut.timed_step(".preliminary_metacells"):
            with ut.progress_bar_slice(preliminary_progress_fraction):
                compute_piled_metacells(
                    adata,
                    what,
                    phase="preliminary",
                    pile_of_cells=random_pile_of_cells,
                    feature_downsample_min_samples=feature_downsample_min_samples,
                    feature_downsample_min_cell_quantile=feature_downsample_min_cell_quantile,
                    feature_downsample_max_cell_quantile=feature_downsample_max_cell_quantile,
                    feature_min_gene_total=feature_min_gene_total,
                    feature_min_gene_top3=feature_min_gene_top3,
                    feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                    feature_gene_names=feature_gene_names,
                    feature_gene_patterns=feature_gene_patterns,
                    forbidden_gene_names=forbidden_gene_names,
                    forbidden_gene_patterns=forbidden_gene_patterns,
                    feature_correction=feature_correction,
                    cells_similarity_value_normalization=cells_similarity_value_normalization,
                    cells_similarity_log_data=cells_similarity_log_data,
                    cells_similarity_method=cells_similarity_method,
                    groups_similarity_log_data=groups_similarity_log_data,
                    groups_similarity_method=groups_similarity_method,
                    min_target_pile_size=min_target_pile_size,
                    max_target_pile_size=max_target_pile_size,
                    target_metacells_in_pile=target_metacells_in_pile,
                    pile_min_split_size_factor=pile_min_split_size_factor,
                    pile_min_robust_size_factor=pile_min_robust_size_factor,
                    pile_max_merge_size_factor=pile_max_merge_size_factor,
                    max_cell_size=max_cell_size,
                    max_cell_size_factor=max_cell_size_factor,
                    cell_sizes=cell_sizes,
                    target_metacell_size=target_metacell_size,
                    knn_k=knn_k,
                    min_knn_k=min_knn_k,
                    knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                    knn_incoming_degree_factor=knn_incoming_degree_factor,
                    knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                    min_seed_size_quantile=min_seed_size_quantile,
                    max_seed_size_quantile=max_seed_size_quantile,
                    candidates_cooldown_pass=candidates_cooldown_pass,
                    candidates_cooldown_node=candidates_cooldown_node,
                    candidates_cooldown_phase=candidates_cooldown_phase,
                    candidates_min_split_size_factor=candidates_min_split_size_factor,
                    candidates_max_merge_size_factor=None,
                    candidates_min_metacell_cells=None,
                    candidates_max_split_min_cut_strength=candidates_max_split_min_cut_strength,
                    candidates_min_cut_seed_cells=candidates_min_cut_seed_cells,
                    must_complete_cover=True,
                    final_max_outliers_levels=None,
                    deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                    deviants_abs_folds=deviants_abs_folds,
                    deviants_max_gene_fraction=deviants_max_gene_fraction,
                    deviants_max_cell_fraction=deviants_max_cell_fraction,
                    dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                    dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                    dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                    dissolve_min_metacell_cells=dissolve_min_metacell_cells,
                    random_seed=random_seed,
                    expected_group_piles_per_pile=expected_group_piles_per_pile,
                )

        with ut.timed_step(".metacell_piles"):
            mdata = tl.group_obs_data(adata, what, groups="pre_metacell", name=".preliminary_metacells")
            if mdata is None:
                raise ValueError("Empty metacells data, giving up")

            with ut.progress_bar_slice(groups_progress_fraction):
                compute_divide_and_conquer_metacells(
                    mdata,
                    feature_downsample_min_samples=feature_downsample_min_samples,
                    feature_downsample_min_cell_quantile=feature_downsample_min_cell_quantile,
                    feature_downsample_max_cell_quantile=feature_downsample_max_cell_quantile,
                    feature_min_gene_total=feature_min_gene_total,
                    feature_min_gene_top3=feature_min_gene_top3,
                    feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                    feature_gene_names=feature_gene_names,
                    feature_gene_patterns=feature_gene_patterns,
                    forbidden_gene_names=forbidden_gene_names,
                    forbidden_gene_patterns=forbidden_gene_patterns,
                    feature_correction=None,
                    cells_similarity_value_normalization=cells_similarity_value_normalization,
                    cells_similarity_log_data=groups_similarity_log_data,
                    cells_similarity_method=groups_similarity_method,
                    groups_similarity_log_data=groups_similarity_log_data,
                    groups_similarity_method=groups_similarity_method,
                    min_target_pile_size=min_target_pile_size,
                    max_target_pile_size=max_target_pile_size,
                    target_metacells_in_pile=target_metacells_in_pile,
                    pile_min_split_size_factor=pile_min_split_size_factor,
                    pile_min_robust_size_factor=pile_min_robust_size_factor,
                    pile_max_merge_size_factor=pile_max_merge_size_factor,
                    target_metacell_size=target_pile_size,
                    max_cell_size=None,
                    max_cell_size_factor=None,
                    cell_sizes="grouped",
                    knn_k=knn_k,
                    min_knn_k=min_knn_k,
                    knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                    knn_incoming_degree_factor=knn_incoming_degree_factor,
                    knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                    min_seed_size_quantile=min_seed_size_quantile,
                    max_seed_size_quantile=max_seed_size_quantile,
                    candidates_cooldown_pass=candidates_cooldown_pass,
                    candidates_cooldown_node=candidates_cooldown_node,
                    candidates_cooldown_phase=candidates_cooldown_phase,
                    candidates_min_split_size_factor=pile_min_split_size_factor,
                    candidates_max_merge_size_factor=None,
                    candidates_min_metacell_cells=None,
                    candidates_max_split_min_cut_strength=candidates_max_split_min_cut_strength,
                    candidates_min_cut_seed_cells=candidates_min_cut_seed_cells,
                    must_complete_cover=True,
                    final_max_outliers_levels=None,
                    deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                    deviants_abs_folds=deviants_abs_folds,
                    deviants_max_gene_fraction=deviants_max_gene_fraction,
                    deviants_max_cell_fraction=deviants_max_cell_fraction,
                    dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                    dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                    dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                    dissolve_min_metacell_cells=dissolve_min_metacell_cells,
                    random_seed=random_seed,
                )
            preliminary_metacell_of_cells = ut.get_o_numpy(adata, "pre_metacell", formatter=ut.groups_description)

            pile_of_preliminary_metacells = ut.get_o_numpy(mdata, "metacell", formatter=ut.groups_description)

            preliminary_pile_of_cells = ut.group_piles(preliminary_metacell_of_cells, pile_of_preliminary_metacells)

        with ut.timed_step(".final_metacells"):
            with ut.progress_bar_slice(final_progress_fraction):
                compute_piled_metacells(
                    adata,
                    what,
                    phase="final",
                    pile_of_cells=preliminary_pile_of_cells,
                    feature_downsample_min_samples=feature_downsample_min_samples,
                    feature_downsample_min_cell_quantile=feature_downsample_min_cell_quantile,
                    feature_downsample_max_cell_quantile=feature_downsample_max_cell_quantile,
                    feature_min_gene_total=feature_min_gene_total,
                    feature_min_gene_top3=feature_min_gene_top3,
                    feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                    feature_gene_names=feature_gene_names,
                    feature_gene_patterns=feature_gene_patterns,
                    forbidden_gene_names=forbidden_gene_names,
                    forbidden_gene_patterns=forbidden_gene_patterns,
                    feature_correction=feature_correction,
                    cells_similarity_value_normalization=cells_similarity_value_normalization,
                    cells_similarity_log_data=cells_similarity_log_data,
                    cells_similarity_method=cells_similarity_method,
                    groups_similarity_log_data=groups_similarity_log_data,
                    groups_similarity_method=groups_similarity_method,
                    min_target_pile_size=min_target_pile_size,
                    max_target_pile_size=max_target_pile_size,
                    target_metacells_in_pile=target_metacells_in_pile,
                    pile_min_split_size_factor=pile_min_split_size_factor,
                    pile_min_robust_size_factor=pile_min_robust_size_factor,
                    pile_max_merge_size_factor=pile_max_merge_size_factor,
                    target_metacell_size=target_metacell_size,
                    max_cell_size=max_cell_size,
                    max_cell_size_factor=max_cell_size_factor,
                    cell_sizes=cell_sizes,
                    knn_k=knn_k,
                    min_knn_k=min_knn_k,
                    knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                    knn_incoming_degree_factor=knn_incoming_degree_factor,
                    knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                    min_seed_size_quantile=min_seed_size_quantile,
                    max_seed_size_quantile=max_seed_size_quantile,
                    candidates_cooldown_pass=candidates_cooldown_pass,
                    candidates_cooldown_node=candidates_cooldown_node,
                    candidates_cooldown_phase=candidates_cooldown_phase,
                    candidates_min_split_size_factor=candidates_min_split_size_factor,
                    candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                    candidates_min_metacell_cells=candidates_min_metacell_cells,
                    candidates_max_split_min_cut_strength=candidates_max_split_min_cut_strength,
                    candidates_min_cut_seed_cells=candidates_min_cut_seed_cells,
                    must_complete_cover=must_complete_cover,
                    final_max_outliers_levels=final_max_outliers_levels,
                    deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                    deviants_abs_folds=deviants_abs_folds,
                    deviants_max_gene_fraction=deviants_max_gene_fraction,
                    deviants_max_cell_fraction=deviants_max_cell_fraction,
                    dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                    dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                    dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                    dissolve_min_metacell_cells=dissolve_min_metacell_cells,
                    random_seed=random_seed,
                    expected_group_piles_per_pile=expected_group_piles_per_pile,
                )
        ut.done_incrementals(adata)

        return True
    finally:
        ut.cancel_incrementals(adata)


@ut.logged(pile_of_cells=ut.groups_description)
def compute_piled_metacells(
    adata: AnnData,
    what: str,
    *,
    phase: str,
    pile_of_cells: ut.NumpyVector,
    feature_downsample_min_samples: int = pr.feature_downsample_min_samples,
    feature_downsample_min_cell_quantile: float = pr.feature_downsample_min_cell_quantile,
    feature_downsample_max_cell_quantile: float = pr.feature_downsample_max_cell_quantile,
    feature_min_gene_total: Optional[int],
    feature_min_gene_top3: Optional[int],
    feature_min_gene_relative_variance: Optional[float],
    feature_gene_names: Optional[Collection[str]] = None,
    feature_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    forbidden_gene_names: Optional[Collection[str]],
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]],
    feature_correction: Optional[FeatureCorrection],
    cells_similarity_value_normalization: float,
    cells_similarity_log_data: bool,
    cells_similarity_method: str,
    groups_similarity_log_data: bool,
    groups_similarity_method: str,
    min_target_pile_size: int = pr.min_target_pile_size,
    max_target_pile_size: int = pr.max_target_pile_size,
    target_metacells_in_pile: int = pr.target_metacells_in_pile,
    pile_min_split_size_factor: float,
    pile_min_robust_size_factor: float,
    pile_max_merge_size_factor: float,
    target_metacell_size: float,
    max_cell_size: Optional[float] = pr.max_cell_size,
    max_cell_size_factor: Optional[float] = pr.max_cell_size_factor,
    cell_sizes: Optional[Union[str, ut.Vector]],
    knn_k: Optional[int],
    min_knn_k: Optional[int],
    knn_balanced_ranks_factor: float,
    knn_incoming_degree_factor: float,
    knn_outgoing_degree_factor: float,
    min_seed_size_quantile: float = pr.min_seed_size_quantile,
    max_seed_size_quantile: float = pr.max_seed_size_quantile,
    candidates_cooldown_pass: float = pr.cooldown_pass,
    candidates_cooldown_node: float = pr.cooldown_node,
    candidates_cooldown_phase: float = pr.cooldown_phase,
    candidates_min_split_size_factor: Optional[float],
    candidates_max_merge_size_factor: Optional[float],
    candidates_min_metacell_cells: Optional[int] = pr.min_metacell_cells,
    candidates_max_split_min_cut_strength: Optional[float] = pr.max_split_min_cut_strength,
    candidates_min_cut_seed_cells: Optional[int] = pr.min_cut_seed_cells,
    must_complete_cover: bool,
    final_max_outliers_levels: Optional[int],
    deviants_min_gene_fold_factor: float,
    deviants_abs_folds: bool = pr.deviants_abs_folds,
    deviants_max_gene_fraction: Optional[float],
    deviants_max_cell_fraction: Optional[float],
    dissolve_min_robust_size_factor: Optional[float],
    dissolve_min_convincing_size_factor: Optional[float],
    dissolve_min_convincing_gene_fold_factor: float,
    dissolve_min_metacell_cells: int,
    random_seed: int,
    expected_group_piles_per_pile: float,
) -> None:
    """
    Compute metacells separately in each pile.
    """

    piles_count = np.max(pile_of_cells) + 1
    assert piles_count > 0

    if not ut.has_progress_bar():
        piles_progress_fraction: Optional[float] = None
        outliers_progress_fraction: Optional[float] = None
    else:
        piles_progress_fraction, outliers_progress_fraction = _progress_fractions(
            _phase_times(
                piles_count=piles_count,
                deviants_max_cell_fraction=deviants_max_cell_fraction,
                final_max_outliers_levels=final_max_outliers_levels,
                expected_group_piles_per_pile=expected_group_piles_per_pile,
            )
        )

    with ut.progress_bar_slice(piles_progress_fraction):
        subset_results = _run_parallel_piles(
            adata,
            what,
            phase=phase,
            piles_count=piles_count,
            pile_of_cells=pile_of_cells,
            feature_downsample_min_samples=feature_downsample_min_samples,
            feature_downsample_min_cell_quantile=feature_downsample_min_cell_quantile,
            feature_downsample_max_cell_quantile=feature_downsample_max_cell_quantile,
            feature_min_gene_total=feature_min_gene_total,
            feature_min_gene_top3=feature_min_gene_top3,
            feature_min_gene_relative_variance=feature_min_gene_relative_variance,
            feature_gene_names=feature_gene_names,
            feature_gene_patterns=feature_gene_patterns,
            forbidden_gene_names=forbidden_gene_names,
            forbidden_gene_patterns=forbidden_gene_patterns,
            feature_correction=feature_correction,
            cells_similarity_value_normalization=cells_similarity_value_normalization,
            cells_similarity_log_data=cells_similarity_log_data,
            cells_similarity_method=cells_similarity_method,
            target_metacell_size=target_metacell_size,
            max_cell_size=max_cell_size,
            max_cell_size_factor=max_cell_size_factor,
            cell_sizes=cell_sizes,
            knn_k=knn_k,
            min_knn_k=min_knn_k,
            knn_balanced_ranks_factor=knn_balanced_ranks_factor,
            knn_incoming_degree_factor=knn_incoming_degree_factor,
            knn_outgoing_degree_factor=knn_outgoing_degree_factor,
            min_seed_size_quantile=min_seed_size_quantile,
            max_seed_size_quantile=max_seed_size_quantile,
            candidates_cooldown_pass=candidates_cooldown_pass,
            candidates_cooldown_node=candidates_cooldown_node,
            candidates_cooldown_phase=candidates_cooldown_phase,
            candidates_min_split_size_factor=candidates_min_split_size_factor,
            candidates_max_merge_size_factor=candidates_max_merge_size_factor,
            candidates_min_metacell_cells=candidates_min_metacell_cells,
            candidates_max_split_min_cut_strength=candidates_max_split_min_cut_strength,
            candidates_min_cut_seed_cells=candidates_min_cut_seed_cells,
            must_complete_cover=must_complete_cover and piles_count == 1,
            deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
            deviants_abs_folds=deviants_abs_folds,
            deviants_max_gene_fraction=deviants_max_gene_fraction,
            deviants_max_cell_fraction=deviants_max_cell_fraction,
            dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
            dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
            dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
            dissolve_min_metacell_cells=dissolve_min_metacell_cells,
            random_seed=random_seed,
        )

        assert len(subset_results) == piles_count

        counts = dict(pre_pile=0, pile=0, pre_candidate=0, candidate=0, pre_metacell=0, metacell=0)

        with ut.timed_step(".collect_piles"):
            ut.log_calc("collect piles results:")
            for pile_index, pile_results in enumerate(subset_results):
                with ut.log_step(
                    "- pile",
                    pile_index,
                    formatter=lambda pile_index: ut.progress_description(piles_count, pile_index, "pile"),
                ):
                    pile_results.collect(adata, counts)

    with ut.progress_bar_slice(outliers_progress_fraction):
        assert phase in ("preliminary", "final")
        if phase == "preliminary":
            metacell_annotation = "pre_metacell"
        else:
            metacell_annotation = "metacell"

        metacell_of_cells = ut.get_o_numpy(adata, metacell_annotation, formatter=ut.groups_description)

        outlier_of_cells = metacell_of_cells < 0
        if (
            piles_count == 1
            or not np.any(outlier_of_cells)
            or (not must_complete_cover and final_max_outliers_levels is not None and final_max_outliers_levels <= 0)
        ):
            return

        with ut.timed_step(".outliers"):
            if final_max_outliers_levels is not None:
                final_max_outliers_levels = final_max_outliers_levels - 1
            name = f".{phase}.outliers"
            odata = ut.slice(adata, name=name, top_level=False, obs=outlier_of_cells, track_obs="complete_cell_index")

            is_divide_and_conquer = compute_divide_and_conquer_metacells(
                odata,
                what,
                feature_downsample_min_samples=feature_downsample_min_samples,
                feature_downsample_min_cell_quantile=feature_downsample_min_cell_quantile,
                feature_downsample_max_cell_quantile=feature_downsample_max_cell_quantile,
                feature_min_gene_total=feature_min_gene_total,
                feature_min_gene_top3=feature_min_gene_top3,
                feature_min_gene_relative_variance=feature_min_gene_relative_variance,
                feature_gene_names=feature_gene_names,
                feature_gene_patterns=feature_gene_patterns,
                forbidden_gene_names=forbidden_gene_names,
                forbidden_gene_patterns=forbidden_gene_patterns,
                feature_correction=feature_correction,
                cells_similarity_value_normalization=cells_similarity_value_normalization,
                cells_similarity_log_data=cells_similarity_log_data,
                cells_similarity_method=cells_similarity_method,
                groups_similarity_log_data=groups_similarity_log_data,
                groups_similarity_method=groups_similarity_method,
                min_target_pile_size=min_target_pile_size,
                max_target_pile_size=max_target_pile_size,
                target_metacells_in_pile=target_metacells_in_pile,
                pile_min_split_size_factor=pile_min_split_size_factor,
                pile_min_robust_size_factor=pile_min_robust_size_factor,
                pile_max_merge_size_factor=pile_max_merge_size_factor,
                target_metacell_size=target_metacell_size,
                max_cell_size=max_cell_size,
                max_cell_size_factor=max_cell_size_factor,
                knn_k=knn_k,
                min_knn_k=min_knn_k,
                knn_balanced_ranks_factor=knn_balanced_ranks_factor,
                knn_incoming_degree_factor=knn_incoming_degree_factor,
                knn_outgoing_degree_factor=knn_outgoing_degree_factor,
                min_seed_size_quantile=min_seed_size_quantile,
                max_seed_size_quantile=max_seed_size_quantile,
                candidates_cooldown_pass=candidates_cooldown_pass,
                candidates_cooldown_node=candidates_cooldown_node,
                candidates_cooldown_phase=candidates_cooldown_phase,
                candidates_min_split_size_factor=candidates_min_split_size_factor,
                candidates_max_merge_size_factor=candidates_max_merge_size_factor,
                candidates_min_metacell_cells=candidates_min_metacell_cells,
                candidates_max_split_min_cut_strength=candidates_max_split_min_cut_strength,
                candidates_min_cut_seed_cells=candidates_min_cut_seed_cells,
                must_complete_cover=must_complete_cover,
                final_max_outliers_levels=final_max_outliers_levels,
                deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
                deviants_abs_folds=deviants_abs_folds,
                deviants_max_gene_fraction=deviants_max_gene_fraction,
                deviants_max_cell_fraction=deviants_max_cell_fraction,
                dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
                dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
                dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
                dissolve_min_metacell_cells=dissolve_min_metacell_cells,
                cell_sizes=cell_sizes,
                random_seed=random_seed,
            )

            with ut.log_step("collect outliers results:"):
                outliers_results = SubsetResults(
                    odata, is_direct=not is_divide_and_conquer, pre_target=None, final_target=phase
                )
                outliers_results.collect(adata, counts)


def _run_parallel_piles(
    adata: AnnData,
    what: str,
    *,
    phase: str,
    piles_count: int,
    pile_of_cells: ut.NumpyVector,
    feature_downsample_min_samples: int = pr.feature_downsample_min_samples,
    feature_downsample_min_cell_quantile: float = pr.feature_downsample_min_cell_quantile,
    feature_downsample_max_cell_quantile: float = pr.feature_downsample_max_cell_quantile,
    feature_min_gene_total: Optional[int],
    feature_min_gene_top3: Optional[int],
    feature_min_gene_relative_variance: Optional[float],
    feature_gene_names: Optional[Collection[str]] = None,
    feature_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    forbidden_gene_names: Optional[Collection[str]],
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]],
    feature_correction: Optional[FeatureCorrection],
    cells_similarity_value_normalization: float,
    cells_similarity_log_data: bool,
    cells_similarity_method: str,
    target_metacell_size: float,
    max_cell_size: Optional[float] = pr.max_cell_size,
    max_cell_size_factor: Optional[float] = pr.max_cell_size_factor,
    cell_sizes: Optional[Union[str, ut.Vector]],
    knn_k: Optional[int],
    min_knn_k: Optional[int],
    knn_balanced_ranks_factor: float,
    knn_incoming_degree_factor: float,
    knn_outgoing_degree_factor: float,
    min_seed_size_quantile: float = pr.min_seed_size_quantile,
    max_seed_size_quantile: float = pr.max_seed_size_quantile,
    candidates_cooldown_pass: float = pr.cooldown_pass,
    candidates_cooldown_node: float = pr.cooldown_node,
    candidates_cooldown_phase: float = pr.cooldown_phase,
    candidates_min_split_size_factor: Optional[float],
    candidates_max_merge_size_factor: Optional[float],
    candidates_min_metacell_cells: Optional[int],
    candidates_max_split_min_cut_strength: Optional[float] = pr.max_split_min_cut_strength,
    candidates_min_cut_seed_cells: Optional[int] = pr.min_cut_seed_cells,
    must_complete_cover: bool,
    deviants_min_gene_fold_factor: float,
    deviants_abs_folds: bool = pr.deviants_abs_folds,
    deviants_max_gene_fraction: Optional[float],
    deviants_max_cell_fraction: Optional[float],
    dissolve_min_robust_size_factor: Optional[float],
    dissolve_min_convincing_size_factor: Optional[float],
    dissolve_min_convincing_gene_fold_factor: float,
    dissolve_min_metacell_cells: int,
    random_seed: int,
    hide_from_progress_bar: bool = False,
) -> List[SubsetResults]:
    def _compute_pile_metacells(pile_index: int) -> SubsetResults:
        pile_cells_mask = pile_of_cells == pile_index
        assert np.any(pile_cells_mask)
        name = f".{phase}.pile-{pile_index}/{piles_count}"
        pdata = ut.slice(adata, name=name, top_level=False, obs=pile_cells_mask, track_obs="complete_cell_index")
        compute_direct_metacells(
            pdata,
            what,
            feature_downsample_min_samples=feature_downsample_min_samples,
            feature_downsample_min_cell_quantile=feature_downsample_min_cell_quantile,
            feature_downsample_max_cell_quantile=feature_downsample_max_cell_quantile,
            feature_min_gene_total=feature_min_gene_total,
            feature_min_gene_top3=feature_min_gene_top3,
            feature_min_gene_relative_variance=feature_min_gene_relative_variance,
            feature_gene_names=feature_gene_names,
            feature_gene_patterns=feature_gene_patterns,
            forbidden_gene_names=forbidden_gene_names,
            forbidden_gene_patterns=forbidden_gene_patterns,
            feature_correction=feature_correction,
            cells_similarity_value_normalization=cells_similarity_value_normalization,
            cells_similarity_log_data=cells_similarity_log_data,
            cells_similarity_method=cells_similarity_method,
            target_metacell_size=target_metacell_size,
            max_cell_size=max_cell_size,
            max_cell_size_factor=max_cell_size_factor,
            cell_sizes=cell_sizes,
            knn_k=knn_k,
            min_knn_k=min_knn_k,
            knn_balanced_ranks_factor=knn_balanced_ranks_factor,
            knn_incoming_degree_factor=knn_incoming_degree_factor,
            knn_outgoing_degree_factor=knn_outgoing_degree_factor,
            min_seed_size_quantile=min_seed_size_quantile,
            max_seed_size_quantile=max_seed_size_quantile,
            candidates_cooldown_pass=candidates_cooldown_pass,
            candidates_cooldown_node=candidates_cooldown_node,
            candidates_cooldown_phase=candidates_cooldown_phase,
            candidates_min_split_size_factor=candidates_min_split_size_factor,
            candidates_max_merge_size_factor=candidates_max_merge_size_factor,
            candidates_min_metacell_cells=candidates_min_metacell_cells,
            candidates_max_split_min_cut_strength=candidates_max_split_min_cut_strength,
            candidates_min_cut_seed_cells=candidates_min_cut_seed_cells,
            must_complete_cover=must_complete_cover,
            deviants_min_gene_fold_factor=deviants_min_gene_fold_factor,
            deviants_abs_folds=deviants_abs_folds,
            deviants_max_gene_fraction=deviants_max_gene_fraction,
            deviants_max_cell_fraction=deviants_max_cell_fraction,
            dissolve_min_robust_size_factor=dissolve_min_robust_size_factor,
            dissolve_min_convincing_size_factor=dissolve_min_convincing_size_factor,
            dissolve_min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
            dissolve_min_metacell_cells=dissolve_min_metacell_cells,
            random_seed=random_seed,
        )
        return SubsetResults(pdata, is_direct=True, pre_target=None, final_target=phase)

    @ut.timed_call("compute_pile_metacells")
    def compute_pile_metacells(pile_index: int) -> SubsetResults:
        results = _compute_pile_metacells(pile_index)
        gc.collect()
        return results

    with ut.timed_step(".prepare"):
        ut.get_vo_proper(adata, what, layout="row_major")

    with ut.timed_step(".piles"):
        gc.collect()
        ut.logger().debug("MAX_PARALLEL_PILES: %s", get_max_parallel_piles())
        return ut.parallel_map(
            compute_pile_metacells,
            piles_count,
            max_processors=get_max_parallel_piles(),
            hide_from_progress_bar=hide_from_progress_bar,
        )


LogValue = Optional[Callable[[Any], Optional[str]]]
try:
    from mypy_extensions import NamedArg

    Setter = Callable[[AnnData, str, Any, NamedArg(LogValue, "formatter")], None]  # noqa: F821
except ModuleNotFoundError:
    pass


def _modify_value(
    adata: AnnData,
    getter: Callable[[AnnData, str], Any],
    setter: "Setter",
    name: str,
    modifier: Callable[[Any], Any],
    formatter: LogValue = None,
) -> None:
    old_value = getter(adata, name)
    new_value = modifier(old_value)
    assert new_value is not None
    setter(adata, name, new_value, formatter=formatter)


def _modify_value_subset(
    adata: AnnData,
    getter: Callable[[AnnData, str], ut.NumpyVector],
    setter: "Setter",
    name: str,
    indices: ut.NumpyVector,
    modifier: Callable[[ut.NumpyVector], ut.NumpyVector],
    formatter: LogValue = None,
) -> None:
    old_values = getter(adata, name)
    ut.unfreeze(old_values)
    old_values[indices] = modifier(old_values[indices])
    ut.freeze(old_values)
    setter(adata, name, old_values, formatter=formatter)


def _divide_and_conquer_times(
    piles_count: int,
    deviants_max_cell_fraction: Optional[float],
    final_max_outliers_levels: Optional[int],
    expected_group_piles_per_pile: float,
) -> List[float]:
    assert piles_count > 0
    if piles_count == 1:
        return [1.0, 0.0, 0.0]

    if deviants_max_cell_fraction is None:
        deviants_max_cell_fraction = 0.5

    preliminary_time = sum(
        _phase_times(
            piles_count=piles_count,
            deviants_max_cell_fraction=deviants_max_cell_fraction,
            final_max_outliers_levels=None,
            expected_group_piles_per_pile=expected_group_piles_per_pile,
        )
    )

    groups_piles_count = ceil(piles_count * expected_group_piles_per_pile)
    groups_time = sum(
        _divide_and_conquer_times(
            groups_piles_count,
            deviants_max_cell_fraction=deviants_max_cell_fraction,
            final_max_outliers_levels=None,
            expected_group_piles_per_pile=expected_group_piles_per_pile,
        )
    )

    final_time = sum(
        _phase_times(
            piles_count=piles_count,
            deviants_max_cell_fraction=deviants_max_cell_fraction,
            final_max_outliers_levels=final_max_outliers_levels,
            expected_group_piles_per_pile=expected_group_piles_per_pile,
        )
    )

    return [preliminary_time, groups_time, final_time]


def _phase_times(
    *,
    piles_count: int,
    deviants_max_cell_fraction: Optional[float],
    final_max_outliers_levels: Optional[int],
    expected_group_piles_per_pile: float,
) -> List[float]:
    assert piles_count > 0
    if piles_count == 1:
        return [1.0, 0.0]

    if deviants_max_cell_fraction is None:
        deviants_max_cell_fraction = 0.5

    scale = ut.get_processors_count()
    if MAX_PARALLEL_PILES > 0:
        scale = min(scale, MAX_PARALLEL_PILES)

    piles_time = max(piles_count / scale, 1.0)

    if final_max_outliers_levels is not None:
        if final_max_outliers_levels <= 0:
            return [piles_time, 0.0]

    outlier_piles_count = ceil(piles_count * deviants_max_cell_fraction)
    if final_max_outliers_levels is not None:
        final_max_outliers_levels -= 1

    outliers_time = sum(
        _divide_and_conquer_times(
            piles_count=outlier_piles_count,
            deviants_max_cell_fraction=deviants_max_cell_fraction,
            final_max_outliers_levels=final_max_outliers_levels,
            expected_group_piles_per_pile=expected_group_piles_per_pile,
        )
    )

    return [piles_time, outliers_time]


def _progress_fractions(times: List[float]) -> List[float]:
    total_time = sum(times)
    return [time / total_time for time in times]
