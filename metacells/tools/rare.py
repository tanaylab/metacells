"""
Rare
----
"""

from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
import scipy.cluster.hierarchy as sch  # type: ignore
import scipy.spatial.distance as scd  # type: ignore
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.utilities as ut

from .similarity import compute_var_var_similarity

__all__ = [
    "find_rare_gene_modules",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def find_rare_gene_modules(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    max_genes: int = pr.rare_max_genes,
    max_gene_cell_fraction: float = pr.rare_max_gene_cell_fraction,
    min_gene_maximum: int = pr.rare_min_gene_maximum,
    genes_similarity_method: str = pr.rare_genes_similarity_method,
    genes_cluster_method: str = pr.rare_genes_cluster_method,
    min_genes_of_modules: int = pr.rare_min_genes_of_modules,
    min_cells_of_modules: int = pr.rare_min_cells_of_modules,
    target_metacell_size: float = pr.target_metacell_size,
    target_pile_size: int = pr.min_target_pile_size,
    max_cells_factor_of_random_pile: float = pr.rare_max_cells_factor_of_random_pile,
    min_module_correlation: float = pr.rare_min_module_correlation,
    min_related_gene_fold_factor: float = pr.rare_min_related_gene_fold_factor,
    max_related_gene_increase_factor: float = pr.rare_max_related_gene_increase_factor,
    min_cell_module_total: int = pr.rare_min_cell_module_total,
    inplace: bool = True,
    reproducible: bool,
) -> Optional[Tuple[ut.PandasFrame, ut.PandasFrame]]:
    """
    Detect rare genes modules based on ``what`` (default: {what}) data.

    Rare gene modules include genes which are weakly and rarely expressed, yet are highly correlated
    with each other, allowing for robust detection. Global analysis algorithms (such as metacells)
    tend to ignore or at least discount such genes.

    It is therefore useful to explicitly identify, in a pre-processing step, the few cells which
    express such rare gene modules. Once identified, these cells can be exempt from the global
    algorithm, or the global algorithm can be tweaked in some way to pay extra attention to them.

    If ``reproducible`` is ``True``, a slower (still parallel) but reproducible algorithm will be used to compute
    pearson correlations.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    Obeys (ignores the genes of) the ``lateral_gene`` per-gene (variable) annotation, if any.

    **Returns**

    Observation (Cell) Annotations
        ``cells_rare_gene_module``
            The index of the rare gene module each cell expresses the most, or ``-1`` in the common
            case it does not express any rare genes module.

        ``rare_cell``
            A boolean mask for the (few) cells that express a rare gene module.

    Variable (Gene) Annotations
        ``rare_gene``
            A boolean mask for the genes in any of the rare gene modules.

        ``rare_gene_module``
            The index of the rare gene module a gene belongs to (-1 for non-rare genes).

    If ``inplace``, these are written to to the data, and the function returns ``None``. Otherwise
    they are returned as tuple containing two data frames.

    **Computation Parameters**

    1. Pick as candidates all genes that are expressed in at most ``max_gene_cell_fraction``
       (default: {max_gene_cell_fraction}) of the cells, and whose maximal value in a cell is at least
       ``min_gene_maximum`` (default: {min_gene_maximum}). If a ``lateral_gene`` masks exist, exclude them from the
       candidates. Out of the candidates, pick at most ``max_genes`` (default {max_genes}) which are expressed in the
       least cells.

    2. Compute the similarity between the genes using
       :py:func:`metacells.tools.similarity.compute_var_var_similarity` using the
       ``genes_similarity_method`` (default: {genes_similarity_method}).

    3. Create a hierarchical clustering of the candidate genes using the ``genes_cluster_method``
       (default: {genes_cluster_method}).

    4. Identify gene modules in the hierarchical clustering which contain at least
       ``min_genes_of_modules`` genes (default: {min_genes_of_modules}), with an average gene-gene
       cross-correlation of at least ``min_module_correlation`` (default:
       {min_module_correlation}).

    5. Consider cells expressing of any of the genes in the gene module. If the expected number of
       such cells in each random pile of size ``target_pile_size`` (default: {target_pile_size}), whose total number of
       UMIs of the rare gene module is at least ``min_cell_module_total`` (default: {min_cell_module_total}), is more
       than the ``max_cells_factor_of_random_pile`` (default: {max_cells_factor_of_random_pile}) as a fraction of the
       target metacells size, then discard the rare gene module as not that rare after all.

    6. Add to the gene module all genes whose fraction in cells expressing any of the genes in the
       rare gene module is at least 2^``min_related_gene_fold_factor`` (default:
       {min_related_gene_fold_factor}) times their fraction in the rest of the population, as long
       as their maximal value in one of the expressing cells is at least ``min_gene_maximum``,
       as long as this doesn't add more than ``max_related_gene_increase_factor`` times the original
       number of cells to the rare gene module, and as long as they are not listed in the ``lateral_gene`` masks. If a
       gene is above the threshold for multiple gene modules, associate it with the gene module for which its fold
       factor is higher.

    7. Associate cells with the rare gene module if they contain at least ``min_cell_module_total``
       (default: {min_cell_module_total}) UMIs of the expanded rare gene module. If a cell meets the
       above threshold for several rare gene modules, it is associated with the one for which it
       contains more UMIs.
    """
    assert min_cells_of_modules > 0
    assert min_genes_of_modules > 0

    max_cells_of_random_pile = target_metacell_size * max_cells_factor_of_random_pile
    ut.log_calc("max_cells_of_random_pile", max_cells_of_random_pile)

    allowed_genes_mask = np.full(adata.n_vars, True, dtype="bool")

    if ut.has_data(adata, "lateral_gene"):
        allowed_genes_mask &= ~ut.get_v_numpy(adata, "lateral_gene")

    ut.log_calc("allowed_genes_mask", allowed_genes_mask)

    rare_module_of_cells = np.full(adata.n_obs, -1, dtype="int32")
    list_of_rare_gene_indices_of_modules: List[List[int]] = []

    umis_per_gene_per_cell_by_rows = ut.get_vo_proper(adata, what, layout="row_major")
    ut.log_calc("umis_per_gene_per_cell_by_rows", umis_per_gene_per_cell_by_rows)
    umis_per_gene_per_cell_by_columns = ut.get_vo_proper(adata, what, layout="column_major")
    ut.log_calc("umis_per_gene_per_cell_by_columns", umis_per_gene_per_cell_by_columns)

    candidates = _pick_candidates(
        adata_of_all_genes_of_all_cells=adata,
        what=what,
        max_genes=max_genes,
        max_gene_cell_fraction=max_gene_cell_fraction,
        min_gene_maximum=min_gene_maximum,
        min_genes_of_modules=min_genes_of_modules,
        allowed_genes_mask=allowed_genes_mask,
    )
    if candidates is None:
        return _results(
            adata=adata,
            rare_module_of_cells=rare_module_of_cells,
            list_of_rare_gene_indices_of_modules=list_of_rare_gene_indices_of_modules,
            inplace=inplace,
        )
    candidate_data, candidate_genes_indices = candidates

    similarities_between_candidate_genes = _genes_similarity(
        candidate_data=candidate_data, what=what, method=genes_similarity_method, reproducible=reproducible
    )

    linkage = _cluster_genes(
        similarities_between_candidate_genes=similarities_between_candidate_genes,
        genes_cluster_method=genes_cluster_method,
    )

    rare_gene_indices_of_modules = _identify_genes(
        candidate_genes_indices=candidate_genes_indices,
        similarities_between_candidate_genes=similarities_between_candidate_genes,
        linkage=linkage,
        min_module_correlation=min_module_correlation,
    )

    max_cells_of_modules = int(max_cells_of_random_pile * adata.n_obs / target_pile_size)
    ut.log_calc("max_cells_of_modules", max_cells_of_modules)

    related_gene_indices_of_modules = _related_genes(
        adata_of_all_genes_of_all_cells=adata,
        umis_per_gene_per_cell_by_columns=umis_per_gene_per_cell_by_columns,
        umis_per_gene_per_cell_by_rows=umis_per_gene_per_cell_by_rows,
        rare_gene_indices_of_modules=rare_gene_indices_of_modules,
        allowed_genes_mask=allowed_genes_mask,
        min_genes_of_modules=min_genes_of_modules,
        min_cells_of_modules=min_cells_of_modules,
        max_cells_of_modules=max_cells_of_modules,
        min_cell_module_total=min_cell_module_total,
        min_gene_maximum=min_gene_maximum,
        min_related_gene_fold_factor=min_related_gene_fold_factor,
        max_related_gene_increase_factor=max_related_gene_increase_factor,
    )

    _identify_cells(
        adata_of_all_genes_of_all_cells=adata,
        umis_per_gene_per_cell_by_columns=umis_per_gene_per_cell_by_columns,
        related_gene_indices_of_modules=related_gene_indices_of_modules,
        min_cells_of_modules=min_cells_of_modules,
        max_cells_of_modules=max_cells_of_modules,
        min_cell_module_total=min_cell_module_total,
        rare_module_of_cells=rare_module_of_cells,
    )

    list_of_rare_gene_indices_of_modules = _compress_modules(
        adata_of_all_genes_of_all_cells=adata,
        min_cells_of_modules=min_cells_of_modules,
        max_cells_of_modules=max_cells_of_modules,
        related_gene_indices_of_modules=related_gene_indices_of_modules,
        rare_module_of_cells=rare_module_of_cells,
    )

    return _results(
        adata=adata,
        rare_module_of_cells=rare_module_of_cells,
        list_of_rare_gene_indices_of_modules=list_of_rare_gene_indices_of_modules,
        inplace=inplace,
    )


@ut.timed_call()
def _pick_candidates(
    *,
    adata_of_all_genes_of_all_cells: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    max_genes: int,
    max_gene_cell_fraction: float,
    min_gene_maximum: int,
    min_genes_of_modules: int,
    allowed_genes_mask: ut.NumpyVector,
) -> Optional[Tuple[AnnData, ut.NumpyVector]]:
    data = ut.get_vo_proper(adata_of_all_genes_of_all_cells, what, layout="column_major")
    nnz_cells_of_genes = ut.nnz_per(data, per="column")

    nnz_cell_fraction_of_genes = nnz_cells_of_genes / adata_of_all_genes_of_all_cells.n_obs
    nnz_cell_fraction_mask_of_genes = nnz_cell_fraction_of_genes <= max_gene_cell_fraction

    max_umis_of_genes = ut.max_per(data, per="column")
    max_umis_mask_of_genes = max_umis_of_genes >= min_gene_maximum

    candidates_mask_of_genes = max_umis_mask_of_genes & nnz_cell_fraction_mask_of_genes & allowed_genes_mask
    ut.log_calc("candidate_genes", candidates_mask_of_genes)

    candidate_genes_indices = np.where(candidates_mask_of_genes)[0]
    candidate_genes_count = candidate_genes_indices.size

    if candidate_genes_count > max_genes:
        nnz_cell_fraction_of_candidates = nnz_cell_fraction_of_genes[candidate_genes_indices]
        partitioned_genes_indices = np.argpartition(nnz_cell_fraction_of_candidates, max_genes)
        candidate_genes_indices = candidate_genes_indices[partitioned_genes_indices[0:max_genes]]
        candidate_genes_count = candidate_genes_indices.size

    if candidate_genes_count < min_genes_of_modules:
        return None

    candidate_data = ut.slice(
        adata_of_all_genes_of_all_cells, name=".candidate_genes", vars=candidate_genes_indices, top_level=False
    )
    return candidate_data, candidate_genes_indices


@ut.timed_call()
def _genes_similarity(
    *,
    candidate_data: AnnData,
    what: Union[str, ut.Matrix],
    method: str,
    reproducible: bool,
) -> ut.NumpyMatrix:
    similarity = compute_var_var_similarity(
        candidate_data, what, method=method, reproducible=reproducible, inplace=False
    )
    assert similarity is not None
    return ut.to_numpy_matrix(similarity, only_extract=True)


# TODO: Replicated in metacell.pipeline.related_genes
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
def _identify_genes(
    *,
    candidate_genes_indices: ut.NumpyVector,
    similarities_between_candidate_genes: ut.NumpyMatrix,
    min_module_correlation: float,
    linkage: List[Tuple[int, int]],
) -> List[List[int]]:
    candidate_genes_count = candidate_genes_indices.size
    np.fill_diagonal(similarities_between_candidate_genes, None)
    combined_candidate_indices = {index: [index] for index in range(candidate_genes_count)}

    for link_index, link_data in enumerate(linkage):
        link_index += candidate_genes_count

        left_index = int(link_data[0])
        right_index = int(link_data[1])

        left_combined_candidates = combined_candidate_indices.get(left_index)
        right_combined_candidates = combined_candidate_indices.get(right_index)
        if not left_combined_candidates or not right_combined_candidates:
            continue

        link_combined_candidates = sorted(left_combined_candidates + right_combined_candidates)
        assert link_combined_candidates
        link_similarities = similarities_between_candidate_genes[link_combined_candidates, :][  #
            :, link_combined_candidates
        ]
        average_link_similarity = np.nanmean(link_similarities)
        if average_link_similarity < min_module_correlation:
            continue

        combined_candidate_indices[link_index] = link_combined_candidates
        del combined_candidate_indices[left_index]
        del combined_candidate_indices[right_index]

    return [
        list(candidate_genes_indices[candidate_indices]) for candidate_indices in combined_candidate_indices.values()
    ]


@ut.timed_call()
def _related_genes(  # pylint: disable=too-many-statements,too-many-branches
    *,
    adata_of_all_genes_of_all_cells: AnnData,
    umis_per_gene_per_cell_by_rows: ut.ProperMatrix,
    umis_per_gene_per_cell_by_columns: ut.ProperMatrix,
    rare_gene_indices_of_modules: List[List[int]],
    allowed_genes_mask: ut.NumpyVector,
    min_genes_of_modules: int,
    min_gene_maximum: int,
    min_cells_of_modules: int,
    max_cells_of_modules: int,
    min_cell_module_total: int,
    min_related_gene_fold_factor: float,
    max_related_gene_increase_factor: float,
) -> List[List[int]]:
    total_umis_per_gene = ut.sum_per(umis_per_gene_per_cell_by_columns, per="column")
    ut.log_calc("total_umis_per_gene", total_umis_per_gene)

    ut.log_calc("genes for modules:")
    modules_count = 0
    related_gene_indices_of_modules: List[List[int]] = []

    rare_gene_indices_of_any: Set[int] = set()
    for rare_gene_indices_of_module in rare_gene_indices_of_modules:
        if len(rare_gene_indices_of_module) >= min_genes_of_modules:
            rare_gene_indices_of_any.update(list(rare_gene_indices_of_module))

    for rare_gene_indices_of_module in rare_gene_indices_of_modules:
        if len(rare_gene_indices_of_module) < min_genes_of_modules:
            continue

        module_index = modules_count
        modules_count += 1

        with ut.log_step("- module", module_index):
            ut.log_calc(
                "rare_gene_names", sorted(adata_of_all_genes_of_all_cells.var_names[rare_gene_indices_of_module])
            )

            umis_per_module_gene_per_cell_by_columns = umis_per_gene_per_cell_by_columns[:, rare_gene_indices_of_module]
            ut.log_calc("umis_per_module_gene_per_cell_by_columns", umis_per_module_gene_per_cell_by_columns)
            assert ut.is_layout(umis_per_module_gene_per_cell_by_columns, "column_major")

            umis_per_module_gene_per_cell_by_rows = ut.to_layout(
                umis_per_module_gene_per_cell_by_columns, layout="row_major"
            )
            ut.log_calc("umis_per_module_gene_per_cell_by_rows", umis_per_module_gene_per_cell_by_rows)

            total_module_umis_per_cell = ut.sum_per(umis_per_module_gene_per_cell_by_rows, per="row")
            ut.log_calc("total_module_umis_per_cell", total_module_umis_per_cell)

            mask_of_expressed_cells = total_module_umis_per_cell > 0
            expressed_cells_count = np.sum(mask_of_expressed_cells)

            if expressed_cells_count > max_cells_of_modules:
                if ut.logging_calc():
                    ut.log_calc("expressed_cells", ut.mask_description(mask_of_expressed_cells) + " (too many)")
                continue

            if expressed_cells_count < min_cells_of_modules:
                if ut.logging_calc():
                    ut.log_calc("expressed_cells", ut.mask_description(mask_of_expressed_cells) + " (too few)")
                continue

            ut.log_calc("expressed_cells", mask_of_expressed_cells)

            umis_per_gene_per_expressed_cell_by_rows = umis_per_gene_per_cell_by_rows[mask_of_expressed_cells, :]
            ut.log_calc("umis_per_gene_per_expressed_cell_by_rows", umis_per_gene_per_expressed_cell_by_rows)
            assert ut.is_layout(umis_per_gene_per_expressed_cell_by_rows, "row_major")

            umis_per_gene_per_expressed_cell_by_columns = ut.to_layout(
                umis_per_gene_per_expressed_cell_by_rows, layout="column_major"
            )
            ut.log_calc("umis_per_module_gene_per_cell_by_columns", umis_per_module_gene_per_cell_by_columns)

            max_expressed_umis_per_gene = ut.max_per(umis_per_gene_per_expressed_cell_by_columns, per="column")
            ut.log_calc("max_expressed_umis_per_gene", max_expressed_umis_per_gene)

            total_expressed_umis_per_gene = ut.sum_per(umis_per_gene_per_expressed_cell_by_columns, per="column")
            expressed_fraction_per_gene = total_expressed_umis_per_gene / sum(total_expressed_umis_per_gene)
            ut.log_calc("expressed_fraction_per_gene", expressed_fraction_per_gene)

            total_background_umis_per_gene = total_umis_per_gene - total_expressed_umis_per_gene
            background_fraction_per_gene = total_background_umis_per_gene / sum(total_background_umis_per_gene)
            ut.log_calc("background_fraction_per_gene", background_fraction_per_gene)

            mask_of_related_genes = (
                allowed_genes_mask
                & (max_expressed_umis_per_gene >= min_gene_maximum)
                & (expressed_fraction_per_gene >= background_fraction_per_gene * (2**min_related_gene_fold_factor))
            )
            ut.log_calc("mask_of_related_genes", mask_of_related_genes)

            related_gene_indices = np.where(mask_of_related_genes)[0]
            assert np.all(mask_of_related_genes[rare_gene_indices_of_module])

            mask_of_strong_base_cells = total_module_umis_per_cell >= min_cell_module_total
            count_of_strong_base_cells = np.sum(mask_of_strong_base_cells)

            if ut.logging_calc():
                ut.log_calc(
                    "candidate_gene_names", sorted(adata_of_all_genes_of_all_cells.var_names[related_gene_indices])
                )
                ut.log_calc("mask_of_strong_base_cells", mask_of_strong_base_cells)

            related_gene_indices_of_module = list(rare_gene_indices_of_module)
            for gene_index in related_gene_indices:
                if gene_index in rare_gene_indices_of_module:
                    continue

                if gene_index in rare_gene_indices_of_any:
                    ut.log_calc(
                        f"- candidate gene {adata_of_all_genes_of_all_cells.var_names[gene_index]} "
                        f"belongs to another module"
                    )
                    continue

                umis_of_related_gene_per_cell = ut.to_numpy_vector(
                    umis_per_gene_per_cell_by_columns[:, gene_index], copy=True
                )
                umis_of_related_gene_per_cell += total_module_umis_per_cell
                mask_of_strong_related_cells = umis_of_related_gene_per_cell >= min_cell_module_total
                count_of_strong_related_cells = np.sum(mask_of_strong_related_cells)
                ut.log_calc(
                    f"- candidate gene {adata_of_all_genes_of_all_cells.var_names[gene_index]} "
                    f"strong cells: {count_of_strong_related_cells} "
                    f"factor: {count_of_strong_related_cells / count_of_strong_base_cells}"
                )
                if count_of_strong_related_cells > max_related_gene_increase_factor * count_of_strong_base_cells:
                    continue

                related_gene_indices_of_module.append(gene_index)

            related_gene_indices_of_modules.append(related_gene_indices_of_module)

    if ut.logging_calc():
        ut.log_calc("related genes for modules:")
        for module_index, related_gene_indices_of_module in enumerate(related_gene_indices_of_modules):
            ut.log_calc(
                f"- module {module_index} related_gene_names",
                sorted(adata_of_all_genes_of_all_cells.var_names[related_gene_indices_of_module]),
            )

    return related_gene_indices_of_modules


@ut.timed_call()
def _identify_cells(
    *,
    adata_of_all_genes_of_all_cells: AnnData,
    umis_per_gene_per_cell_by_columns: ut.ProperMatrix,
    related_gene_indices_of_modules: List[List[int]],
    min_cell_module_total: int,
    min_cells_of_modules: int,
    max_cells_of_modules: int,
    rare_module_of_cells: ut.NumpyVector,
) -> None:
    max_strength_of_cells = np.zeros(adata_of_all_genes_of_all_cells.n_obs)

    ut.log_calc("cells for modules:")
    modules_count = len(related_gene_indices_of_modules)
    for module_index, related_gene_indices_of_module in enumerate(related_gene_indices_of_modules):
        if len(related_gene_indices_of_module) == 0:
            continue

        with ut.log_step(
            "- module",
            module_index,
            formatter=lambda module_index: ut.progress_description(modules_count, module_index, "module"),
        ):
            umis_per_related_gene_per_cell_by_columns = umis_per_gene_per_cell_by_columns[
                :, related_gene_indices_of_module
            ]
            ut.log_calc("umis_per_related_gene_per_cell_by_columns", umis_per_related_gene_per_cell_by_columns)

            umis_per_related_gene_per_cell_by_rows = ut.to_layout(
                umis_per_related_gene_per_cell_by_columns, layout="row_major"
            )
            ut.log_calc("umis_per_related_gene_per_cell_by_rows", umis_per_related_gene_per_cell_by_rows)

            total_related_umis_per_cell = ut.sum_per(umis_per_related_gene_per_cell_by_rows, per="row")

            mask_of_strong_cells_of_module = total_related_umis_per_cell >= min_cell_module_total

            median_strength_of_module = np.median(total_related_umis_per_cell[mask_of_strong_cells_of_module])  #
            strong_cells_count = np.sum(mask_of_strong_cells_of_module)

            if strong_cells_count > max_cells_of_modules:
                if ut.logging_calc():
                    ut.log_calc("strong_cells", ut.mask_description(mask_of_strong_cells_of_module) + " (too many)")  #
                related_gene_indices_of_module.clear()
                continue

            if strong_cells_count < min_cells_of_modules:
                if ut.logging_calc():
                    ut.log_calc("strong_cells", ut.mask_description(mask_of_strong_cells_of_module) + " (too few)")  #
                related_gene_indices_of_module.clear()
                continue

            ut.log_calc("strong_cells", mask_of_strong_cells_of_module)

            strength_of_all_cells = total_related_umis_per_cell / median_strength_of_module
            mask_of_strong_cells_of_module &= strength_of_all_cells >= max_strength_of_cells
            max_strength_of_cells[mask_of_strong_cells_of_module] = strength_of_all_cells[
                mask_of_strong_cells_of_module
            ]

            rare_module_of_cells[mask_of_strong_cells_of_module] = module_index


@ut.timed_call()
def _compress_modules(
    *,
    adata_of_all_genes_of_all_cells: AnnData,
    min_cells_of_modules: int,
    max_cells_of_modules: int,
    related_gene_indices_of_modules: List[List[int]],
    rare_module_of_cells: ut.NumpyVector,
) -> List[List[int]]:
    list_of_rare_gene_indices_of_modules: List[List[int]] = []
    list_of_names_of_genes_of_modules: List[List[str]] = []

    cell_counts_of_modules: List[int] = []

    ut.log_calc("compress modules:")
    modules_count = len(related_gene_indices_of_modules)
    for module_index, gene_indices_of_module in enumerate(related_gene_indices_of_modules):
        if len(gene_indices_of_module) == 0:
            continue

        with ut.log_step(
            "- module",
            module_index,
            formatter=lambda module_index: ut.progress_description(modules_count, module_index, "module"),
        ):
            module_cells_mask = rare_module_of_cells == module_index
            module_cells_count = np.sum(module_cells_mask)

            if module_cells_count < min_cells_of_modules:
                if ut.logging_calc():
                    ut.log_calc("cells", str(module_cells_count) + " (too few)")
                rare_module_of_cells[module_cells_mask] = -1
                continue

            if module_cells_count > max_cells_of_modules:
                if ut.logging_calc():
                    ut.log_calc("cells", str(module_cells_count) + " (too many)")
                rare_module_of_cells[module_cells_mask] = -1
                continue

            ut.log_calc("cells", module_cells_count)

            next_module_index = len(list_of_rare_gene_indices_of_modules)
            if module_index != next_module_index:
                ut.log_calc("is reindexed to", next_module_index)
                rare_module_of_cells[module_cells_mask] = next_module_index
                module_index = next_module_index

            next_module_index += 1
            list_of_rare_gene_indices_of_modules.append(gene_indices_of_module)

            if ut.logging_calc():
                cell_counts_of_modules.append(np.sum(module_cells_mask))
            list_of_names_of_genes_of_modules.append(  #
                sorted(adata_of_all_genes_of_all_cells.var_names[gene_indices_of_module])
            )

    if ut.logging_calc():
        ut.log_calc("final modules:")
        for module_index, (module_cells_count, module_gene_names) in enumerate(
            zip(cell_counts_of_modules, list_of_names_of_genes_of_modules)
        ):
            ut.log_calc(f"- module: {module_index} cells: {module_cells_count} genes: {module_gene_names}")  #

    return list_of_rare_gene_indices_of_modules


def _results(
    *,
    adata: AnnData,
    rare_module_of_cells: ut.NumpyVector,
    list_of_rare_gene_indices_of_modules: List[List[int]],
    inplace: bool,
) -> Optional[Tuple[ut.PandasFrame, ut.PandasFrame]]:
    assert np.max(rare_module_of_cells) == len(list_of_rare_gene_indices_of_modules) - 1

    if not inplace:
        var_metrics = ut.to_pandas_frame(index=adata.var_names)

    rare_gene_mask = np.zeros(adata.n_vars, dtype="bool")
    rare_gene_modules = np.full(adata.n_vars, -1, dtype="int32")
    for module_index, rare_gene_indices_of_module in enumerate(list_of_rare_gene_indices_of_modules):
        rare_gene_mask[rare_gene_indices_of_module] = True
        rare_gene_modules[rare_gene_indices_of_module] = module_index

    if inplace:
        ut.set_v_data(adata, "rare_gene", rare_gene_mask)
        ut.set_v_data(adata, "rare_gene_module", rare_gene_modules, formatter=ut.groups_description)
    else:
        var_metrics["rare_gene"] = rare_gene_mask
        ut.log_return("rare_gene", rare_gene_mask)
        var_metrics["rare_gene_module"] = rare_gene_modules
        ut.log_return("rare_gene_module", rare_gene_modules, formatter=ut.groups_description)

    if inplace:
        ut.set_o_data(adata, "cells_rare_gene_module", rare_module_of_cells, formatter=ut.groups_description)
        ut.set_o_data(adata, "rare_cell", rare_module_of_cells >= 0)
        return None

    obs_metrics = ut.to_pandas_frame(index=adata.obs_names)
    ut.log_return("cells_rare_gene_module", rare_module_of_cells, formatter=ut.groups_description)
    ut.log_return("rare_cell", rare_module_of_cells >= 0)

    return obs_metrics, var_metrics
