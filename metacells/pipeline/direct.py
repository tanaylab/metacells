"""
Direct
------
"""

from typing import Callable
from typing import Optional
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

from .select import extract_selected_data

__all__ = [
    "compute_direct_metacells",
]

try:
    from mypy_extensions import NamedArg

    #: A function to correct the extracted selected values before computing cell-cell similarity.
    #:
    #: This function takes the following named arguments:
    #:
    #: ``adata`` - The full annotated data of the cells.
    #:
    #: ``sdata`` - The selected genes annotated data of the cells (a slice of ``adata``). This has a
    #: ``downsample_samples`` unstructured annotation with the target number of samples per cell.
    #:
    #: ``downsampled`` - A dense matrix of the downsampled UMIs of the selected genes of the cells (a copy of the
    #: ``downsampled`` layer of ``sdata``).
    #:
    #: The function is free to modify ``downsampled`` as it sees fit to perform any type of correction (typically, some
    #: form of batch correction). The data type of ``downsampled`` is ``float32`` so the corrected values need not be
    #: integers (even though their initial values will be).
    #:
    #: Note that this is only invoked for metacells-of-cells by
    #: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline` or
    #: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`, and is not invoked for
    #: computing groups of metacelles. Therefore the function can access any metadata of the original input.
    #:
    #: See :py:func:`compute_direct_metacells`.
    FeatureCorrection = Callable[
        [
            NamedArg(AnnData, "adata"),  # noqa: F821
            NamedArg(AnnData, "sdata"),  # noqa: F821
            NamedArg(ut.NumpyMatrix, "downsampled"),  # noqa: F821
        ],
        None,
    ]

except ModuleNotFoundError:
    FeatureCorrection = Callable  # type: ignore


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_direct_metacells(  # pylint: disable=too-many-branches,too-many-statements
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    select_downsample_min_samples: int = pr.select_downsample_min_samples,
    select_downsample_min_cell_quantile: float = pr.select_downsample_min_cell_quantile,
    select_downsample_max_cell_quantile: float = pr.select_downsample_max_cell_quantile,
    select_min_gene_total: Optional[int] = pr.select_min_gene_total,
    select_min_gene_top3: Optional[int] = pr.select_min_gene_top3,
    select_min_gene_relative_variance: Optional[float] = pr.select_min_gene_relative_variance,
    select_correction: Optional[FeatureCorrection] = None,
    cells_similarity_value_regularization: float = pr.cells_similarity_value_regularization,
    cells_similarity_log_data: bool = pr.cells_similarity_log_data,
    cells_similarity_method: str = pr.cells_similarity_method,
    target_metacell_size: float = pr.target_metacell_size,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    knn_k: Optional[int] = pr.knn_k,
    knn_k_size_factor: float = pr.candidates_knn_k_size_factor,
    knn_k_size_quantile: float = pr.knn_k_size_quantile,
    min_knn_k: Optional[int] = pr.min_knn_k,
    knn_balanced_ranks_factor: float = pr.knn_balanced_ranks_factor,
    knn_incoming_degree_factor: float = pr.knn_incoming_degree_factor,
    knn_outgoing_degree_factor: float = pr.knn_outgoing_degree_factor,
    min_seed_size_quantile: float = pr.min_seed_size_quantile,
    max_seed_size_quantile: float = pr.max_seed_size_quantile,
    candidates_cooldown_pass: float = pr.cooldown_pass,
    candidates_cooldown_node: float = pr.cooldown_node,
    candidates_cooldown_phase: float = pr.cooldown_phase,
    candidates_min_split_size_factor: float = pr.candidates_min_split_size_factor,
    candidates_max_merge_size_factor: float = pr.candidates_max_merge_size_factor,
    candidates_min_metacell_cells: Optional[int] = pr.candidates_min_metacell_cells,
    candidates_max_split_min_cut_strength: Optional[float] = pr.max_split_min_cut_strength,
    candidates_min_cut_seed_cells: int = pr.min_cut_seed_cells,
    must_complete_cover: bool = False,
    deviants_policy: str = pr.deviants_policy,
    deviants_min_gene_fold_factor: float = pr.deviants_min_gene_fold_factor,
    deviants_gap_skip_cells: int = pr.deviants_gap_skip_cells,
    deviants_min_noisy_gene_fold_factor: float = pr.deviants_min_noisy_gene_fold_factor,
    deviants_max_gene_fraction: float = pr.deviants_max_gene_fraction,
    deviants_max_cell_fraction: Optional[float] = pr.deviants_max_cell_fraction,
    deviants_max_gap_cells_count: int = pr.deviants_max_gap_cells_count,
    deviants_max_gap_cells_fraction: float = pr.deviants_max_gap_cells_fraction,
    dissolve_min_robust_size_factor: Optional[float] = pr.dissolve_min_robust_size_factor,
    dissolve_min_convincing_size_factor: Optional[float] = pr.dissolve_min_convincing_size_factor,
    dissolve_min_convincing_gene_fold_factor: float = pr.dissolve_min_convincing_gene_fold_factor,
    dissolve_min_metacell_cells: int = pr.dissolve_min_metacell_cells,
    random_seed: int,
) -> None:
    """
    Directly compute metacells using ``what`` (default: {what}) data.

    This directly computes the metacells on the whole data. Like any method that directly looks at
    the whole data at once, the amount of CPU and memory needed becomes unreasonable when the data
    size grows. Above O(10,000) you are much better off using the divide-and-conquer method.

    .. note::

        The current implementation is naive in that it computes the full dense N^2 correlation
        matrix, and only then extracts the sparse graph out of it. We actually need two copies where
        each requires 4 bytes per entry, so for O(100,000) cells, we have storage of
        O(100,000,000,000). In addition, the implementation is serial for the graph clustering
        phases.

        It is possible to mitigate this by fusing the correlations phase and the graph generation
        phase, parallelizing the result, and also (somehow) parallelizing the graph clustering
        phase. This might increase the "reasonable" size for the direct approach to O(100,000).

        We have decided not to invest in this direction since it won't allow us to push the size to
        O(1,000,000) and above. Instead we provide the divide-and-conquer method, which easily
        scales to O(1,000,000) on a single multi-core server, and to "unlimited" size if we further
        enhance the implementation to use a distributed compute cluster of such servers.

    .. todo::

        Should :py:func:`compute_direct_metacells` avoid computing the graph and partition it for a
        very small number of cells?

    **Input**

    The presumably "clean" annotated ``adata``, where the observations are cells and the variables
    are genes, where ``what`` is a per-variable-per-observation matrix or the name of a
    per-variable-per-observation annotation containing such a matrix.

    **Returns**

    Sets the following annotations in ``adata``:

    Unstructured Annotations
        ``downsample_samples``
            The target total number of samples in each downsampled cell.

    Observation-Variable (Cell-Gene) Annotations:
        ``downsampled``
            The downsampled data where the total number of samples in each cell is at most ``downsample_samples``.

    Variable (Gene) Annotations
        ``high_total_gene``
            A boolean mask of genes with "high" expression level (unless a ``select_gene`` mask exists).

        ``high_relative_variance_gene``
            A boolean mask of genes with "high" normalized variance, relative to other genes with a similar expression
            level (unless a ``select_gene`` mask exists).

        ``selected_gene``
            A boolean mask of the actually selected genes.

    Observation (Cell) Annotations
        ``metacell``
            The integer index of the metacell each cell belongs to. The metacells are in no
            particular order. Cells with no metacell assignment ("outliers") are given a metacell
            index of ``-1``.

    **Computation Parameters**

    1. Invoke :py:func:`metacells.pipeline.select.extract_selected_data` to extract "select" data
       from the clean data, using the
       ``select_downsample_min_samples`` (default: {select_downsample_min_samples}),
       ``select_downsample_min_cell_quantile`` (default: {select_downsample_min_cell_quantile}),
       ``select_downsample_max_cell_quantile`` (default: {select_downsample_max_cell_quantile}),
       ``select_min_gene_total`` (default: {select_min_gene_total}),
       ``select_min_gene_top3`` (default: {select_min_gene_top3}),
       ``select_min_gene_relative_variance`` (default: {select_min_gene_relative_variance}) and
       ``random_seed``.

    2. Apply the ``select_correction`` function, if any, to modify the downsampled selects data.

    3. Compute the fractions of each variable in each cell, and add the
       ``cells_similarity_value_regularization`` (default: {cells_similarity_value_regularization}) to
       it.

    4. If ``cells_similarity_log_data`` (default: {cells_similarity_log_data}), invoke the
       :py:func:`metacells.utilities.computation.log_data` function to compute the log (base 2) of
       the data.

    5. Invoke :py:func:`metacells.tools.similarity.compute_obs_obs_similarity` to compute the
       similarity between each pair of cells, using the
       ``cells_similarity_method`` (default: {cells_similarity_method}).

    6. Invoke :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph` to compute a K-Nearest-Neighbors graph,
       using the ``knn_balanced_ranks_factor`` (default: {knn_balanced_ranks_factor}), ``knn_incoming_degree_factor``
       (default: {knn_incoming_degree_factor}) and ``knn_outgoing_degree_factor`` (default:
       {knn_outgoing_degree_factor}). If ``knn_k`` (default: {knn_k}) is not specified, then it is chosen to be the
       ``knn_k_size_factor`` (default: {knn_k_size_factor} times the median number of cells required to reach the target
       metacell size, or the ``knn_k_size_quantile`` (default: {knn_k_size_quantile}) the number of cells required, or
       ``min_knn_k`` (default: {min_knn_k}), whichever is largest.

    7. Invoke :py:func:`metacells.tools.candidates.compute_candidate_metacells` to compute
       the candidate metacells, using the
       ``min_seed_size_quantile`` (default: {min_seed_size_quantile}),
       ``max_seed_size_quantile`` (default: {max_seed_size_quantile}),
       ``candidates_cooldown_pass`` (default: {candidates_cooldown_pass}),
       ``candidates_cooldown_node`` (default: {candidates_cooldown_node}),
       ``candidates_cooldown_phase`` (default: {candidates_cooldown_phase}),
       ``candidates_min_split_size_factor`` (default: {candidates_min_split_size_factor}),
       ``candidates_max_merge_size_factor`` (default: {candidates_max_merge_size_factor}),
       ``candidates_min_metacell_cells`` (default: {candidates_min_metacell_cells}),
       and
       ``random_seed``. This tries to build metacells of the
       ``target_metacell_size`` (default: {target_metacell_size})
       using the effective cell sizes.

    8. Unless ``must_complete_cover`` (default: {must_complete_cover}), invoke
       :py:func:`metacells.tools.deviants.find_deviant_cells` to remove deviants from the candidate
       metacells, using the
       ``deviants_policy`` (default: {deviants_policy}),
       ``deviants_min_gene_fold_factor`` (default: {deviants_min_gene_fold_factor}),
       ``deviants_gap_skip_cells`` (default: {deviants_gap_skip_cells}),
       ``deviants_min_noisy_gene_fold_factor`` (default: {deviants_min_noisy_gene_fold_factor}),
       ``deviants_max_gene_fraction`` (default: {deviants_max_gene_fraction})
       and
       ``deviants_max_cell_fraction`` (default: {deviants_max_cell_fraction}).

    9. Unless ``must_complete_cover`` (default: {must_complete_cover}), invoke
       :py:func:`metacells.tools.dissolve.dissolve_metacells` to dissolve small unconvincing
       metacells, using the same
       ``target_metacell_size`` (default: {target_metacell_size}),
       and the effective cell sizes
       and the
       ``dissolve_min_robust_size_factor`` (default: {dissolve_min_robust_size_factor}),
       ``dissolve_min_convincing_size_factor`` (default: {dissolve_min_convincing_size_factor}),
       ``dissolve_min_convincing_gene_fold_factor`` (default: {dissolve_min_convincing_size_factor})
       and
       ``dissolve_min_metacell_cells`` (default: ``dissolve_min_metacell_cells``).
    """
    assert (
        target_metacell_size < 1000 or cell_sizes is not None
    ), f"target_metacell_size: {target_metacell_size} seems to be in UMIs, should be in cells"

    cell_sizes = ut.maybe_o_numpy(adata, cell_sizes, formatter=ut.sizes_description)
    if cell_sizes is not None:
        ut.log_calc("cell_sizes", cell_sizes, formatter=ut.sizes_description)
        is_small = np.sum(cell_sizes) <= target_metacell_size * candidates_min_split_size_factor
    else:
        is_small = adata.n_obs <= target_metacell_size * candidates_min_split_size_factor
    ut.log_calc("is_small", is_small)

    if is_small:
        candidate_of_cells = np.zeros(adata.n_obs, dtype="int32")
        ut.set_v_data(adata, "selected_gene", np.zeros(adata.n_vars, dtype="bool"))

    else:
        sdata = extract_selected_data(
            adata,
            what,
            top_level=False,
            downsample_min_samples=select_downsample_min_samples,
            downsample_min_cell_quantile=select_downsample_min_cell_quantile,
            downsample_max_cell_quantile=select_downsample_max_cell_quantile,
            min_gene_relative_variance=select_min_gene_relative_variance,
            min_gene_total=select_min_gene_total,
            min_gene_top3=select_min_gene_top3,
            random_seed=random_seed,
        )

        data = ut.get_vo_proper(sdata, "downsampled", layout="row_major")
        data = ut.to_numpy_matrix(data, copy=True)

        if select_correction is not None:
            select_correction(adata=adata, sdata=sdata, downsampled=data)

        if cells_similarity_value_regularization > 0:
            data += cells_similarity_value_regularization

        if cells_similarity_log_data:
            data = ut.log_data(data, base=2)

        if knn_k is None:
            if cell_sizes is None:
                median_cell_size = 1.0
                quantile_cell_size = 1.0
            else:
                median_cell_size = float(np.median(cell_sizes))
                quantile_cell_size = np.quantile(cell_sizes, knn_k_size_quantile)
            knn_k_of_median = int(round(target_metacell_size / median_cell_size))
            knn_k_median = int(round(knn_k_size_factor * target_metacell_size / median_cell_size))
            knn_k_quantile = int(round(target_metacell_size / quantile_cell_size))
            ut.log_calc("median_cell_size", median_cell_size)
            ut.log_calc("knn_k by median_cell_size", knn_k_median)
            ut.log_calc("quantile_cell_size", quantile_cell_size)
            ut.log_calc("knn_k by quantile_cell_size", knn_k_quantile)
            knn_k = max(
                int(round(knn_k_size_factor * target_metacell_size / median_cell_size)),
                int(round(target_metacell_size / quantile_cell_size)),
                min_knn_k or 0,
            )

        if knn_k == 0:
            ut.log_calc("knn_k: 0 (too small, trying a single metacell)")
            ut.set_o_data(sdata, "candidate", np.full(sdata.n_obs, 0, dtype="int32"), formatter=lambda _: "* <- 0")
        elif knn_k_of_median >= sdata.n_obs:
            ut.log_calc(f"knn_k of median: {knn_k_of_median} (too large, trying a single metacell)")
            ut.set_o_data(sdata, "candidate", np.full(sdata.n_obs, 0, dtype="int32"), formatter=lambda _: "* <- 0")

        else:
            ut.log_calc("knn_k", knn_k)

            tl.compute_obs_obs_similarity(sdata, data, method=cells_similarity_method, reproducible=random_seed != 0)

            tl.compute_obs_obs_knn_graph(
                sdata,
                k=knn_k,
                balanced_ranks_factor=knn_balanced_ranks_factor,
                incoming_degree_factor=knn_incoming_degree_factor,
                outgoing_degree_factor=knn_outgoing_degree_factor,
            )

            tl.compute_candidate_metacells(
                sdata,
                target_metacell_size=target_metacell_size,
                cell_sizes=cell_sizes,
                min_seed_size_quantile=min_seed_size_quantile,
                max_seed_size_quantile=max_seed_size_quantile,
                cooldown_pass=candidates_cooldown_pass,
                cooldown_node=candidates_cooldown_node,
                cooldown_phase=candidates_cooldown_phase,
                min_split_size_factor=candidates_min_split_size_factor,
                max_merge_size_factor=candidates_max_merge_size_factor,
                min_metacell_cells=candidates_min_metacell_cells,
                max_split_min_cut_strength=candidates_max_split_min_cut_strength,
                min_cut_seed_cells=candidates_min_cut_seed_cells,
                must_complete_cover=must_complete_cover,
                random_seed=random_seed,
            )

        candidate_of_cells = ut.get_o_numpy(sdata, "candidate", formatter=ut.groups_description)

    if must_complete_cover:
        assert np.min(candidate_of_cells) == 0
        ut.set_o_data(adata, "metacell", candidate_of_cells, formatter=ut.groups_description)

    else:
        deviants = tl.find_deviant_cells(
            adata,
            candidates=candidate_of_cells,
            policy=deviants_policy,
            min_gene_fold_factor=deviants_min_gene_fold_factor,
            gap_skip_cells=deviants_gap_skip_cells,
            min_noisy_gene_fold_factor=deviants_min_noisy_gene_fold_factor,
            max_gene_fraction=deviants_max_gene_fraction,
            max_cell_fraction=deviants_max_cell_fraction,
            max_gap_cells_count=deviants_max_gap_cells_count,
            max_gap_cells_fraction=deviants_max_gap_cells_fraction,
        )

        tl.dissolve_metacells(
            adata,
            candidates=candidate_of_cells,
            deviants=deviants,
            target_metacell_size=target_metacell_size,
            cell_sizes=cell_sizes,
            min_robust_size_factor=dissolve_min_robust_size_factor,
            min_convincing_size_factor=dissolve_min_convincing_size_factor,
            min_convincing_gene_fold_factor=dissolve_min_convincing_gene_fold_factor,
            min_metacell_cells=dissolve_min_metacell_cells,
        )
