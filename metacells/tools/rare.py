'''
Rare
----
'''

from re import Pattern
from typing import Collection, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.cluster.hierarchy as sch  # type: ignore
import scipy.spatial.distance as scd  # type: ignore
from anndata import AnnData

import metacells.parameters as pr
import metacells.utilities as ut

from .named import find_named_genes
from .similarity import compute_var_var_similarity

__all__ = [
    'find_rare_gene_modules',
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def find_rare_gene_modules(
    adata: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    *,
    max_gene_cell_fraction: float = pr.rare_max_gene_cell_fraction,
    min_gene_maximum: int = pr.rare_min_gene_maximum,
    similarity_of: Optional[str] = None,
    genes_similarity_method: str = pr.rare_genes_similarity_method,
    genes_cluster_method: str = pr.rare_genes_cluster_method,
    forbidden_gene_names: Optional[Collection[str]] = None,
    forbidden_gene_patterns: Optional[Collection[Union[str, Pattern]]] = None,
    min_genes_of_modules: int = pr.rare_min_genes_of_modules,
    min_cells_of_modules: int = pr.rare_min_cells_of_modules,
    target_pile_size: int = pr.target_pile_size,
    max_cells_of_random_pile: int = pr.rare_max_cells_of_random_pile,
    target_metacell_size: int = pr.target_metacell_size,
    min_modules_size_factor: float = pr.rare_min_modules_size_factor,
    min_module_correlation: float = pr.rare_min_module_correlation,
    min_related_gene_fold_factor: float = pr.rare_min_related_gene_fold_factor,
    min_cell_module_total: int = pr.rare_min_cell_module_total,
    inplace: bool = True,
) -> Optional[Tuple[ut.PandasFrame, ut.PandasFrame, ut.NumpyVector]]:
    '''
    Detect rare genes modules based ``of`` some data (by default, the focus).

    Rare gene modules include genes which are weakly and rarely expressed, yet are highly correlated
    with each other, allowing for robust detection. Global analysis algorithms (such as metacells)
    tend to ignore or at least discount such genes.

    It is therefore useful to explicitly identify, in a pre-processing step, the few cells which
    express such rare gene modules. Once identified, these cells can be exempt from the global
    algorithm, or the global algorithm can be tweaked in some way to pay extra attention to them.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes.

    **Returns**

    Observation (Cell) Annotations
        ``cells_rare_gene_module``
            The index of the rare gene module each cell expresses the most, or ``-1`` in the common
            case it does not express any rare genes module.

        ``rare_cells``
            A boolean mask for the (few) cells that express a rare gene module.

    Variable (Gene) Annotations
        ``genes_rare_gene_modules``
            The indices of the rare gene module(s) each gene belongs to, where every non-zero bit
            indicates the gene participates in the matching rare gene module.

        ``rare_genes``
            A boolean mask for the genes in any of the rare gene modules.

    Unstructured Annotations
        ``rare_gene_modules``
            An array of rare gene modules, where every entry is the array of the names of the genes
            of the module.

    If ``inplace``, these are written to to the data, and the function returns ``None``. Otherwise
    they are returned as tuple containing two data frames and an array.

    **Computation Parameters**

    1. Pick as candidates all genes that are expressed in at most than ``max_gene_cell_fraction``
       (default: {max_gene_cell_fraction}) of the cells, and whose maximal value in a cell is at
       least ``min_gene_maximum`` (default: {min_gene_maximum}), as long as they do not match the
       ``forbidden_gene_names`` or the ``forbidden_gene_patterns``.

    2. Compute the similarity between the genes using
       :py:func:`metacells.tools.similarity.compute_var_var_similarity` using the
       ``genes_similarity_method`` (default: {genes_similarity_method}). If ``similarity_of`` is
       specified (default: {similarity_of}), use this data for computing the similarity in exactly
       the way you prefer (e.g., to correlate the log values).

    3. Create a hierarchical clustering of the candidate genes using the ``genes_cluster_method``
       (default: {genes_cluster_method}).

    4. Identify gene modules in the hierarchical clustering which contain at least
       ``min_genes_of_modules`` genes (default: {min_genes_of_modules}), with an average gene-gene
       cross-correlation of at least ``min_module_correlation`` (default:
       {min_module_correlation}).

    5. Consider cells expressing of any of the genes in the gene module. If the expected number of
       such cells in each random pile of size ``target_pile_size`` (default: {target_pile_size}),
       whose total number of UMIs of the rare gene module is at least ``min_cell_module_total``
       (default: {min_cell_module_total}), is more than the ``max_cells_of_random_pile`` (default:
       {max_cells_of_random_pile}), then discard the rare gene module as not that rare after all.

    6. Add to the gene module all genes whose fraction in cells expressing any of the genes in the
       rare gene module is at least 2^``min_related_gene_fold_factor`` (default:
       {min_related_gene_fold_factor}) times their fraction in the rest of the population, as long
       as their maximal value in one of the expressing cells is at least ``min_gene_maximum``, as
       long as they do not match the ``forbidden_gene_names`` or the ``forbidden_gene_patterns``. If
       a gene is above the threshold for multiple gene modules, associate it with the gene module
       for which its fold factor is higher.

    7. Associate cells with the rare gene module if they contain at least ``min_cell_module_total``
       (default: {min_cell_module_total}) UMIs of the expanded rare gene module. If a cell meets the
       above threshold for several rare gene modules, it is associated with the one for which it
       contains more UMIs.

    8. Discard modules which have less than ``min_cells_of_modules`` (default:
       {min_cells_of_modules}) cells or whose total UMIs are less than the ``target_metacell_size``
       (default: {target_metacell_size}) times the ``min_modules_size_factor`` (default:
       {min_modules_size_factor}).
    '''
    assert min_cells_of_modules > 0
    assert min_genes_of_modules > 0

    forbidden_genes_mask = \
        find_named_genes(adata, names=forbidden_gene_names,
                         patterns=forbidden_gene_patterns)
    assert forbidden_genes_mask is not None

    allowed_genes_mask = ~forbidden_genes_mask.values
    ut.log_calc('allowed_genes_mask', allowed_genes_mask)

    rare_module_of_cells = np.full(adata.n_obs, -1, dtype='int32')
    list_of_names_of_genes_of_modules: List[ut.NumpyVector] = []

    candidates = \
        _pick_candidates(adata_of_all_genes_of_all_cells=adata,
                         what=what,
                         max_gene_cell_fraction=max_gene_cell_fraction,
                         min_gene_maximum=min_gene_maximum,
                         min_genes_of_modules=min_genes_of_modules,
                         allowed_genes_mask=allowed_genes_mask)
    if candidates is None:
        return _results(adata=adata,
                        rare_module_of_cells=rare_module_of_cells,
                        list_of_names_of_genes_of_modules=list_of_names_of_genes_of_modules,
                        inplace=inplace)
    candidate_data, candidate_genes_indices = candidates

    similarities_between_candidate_genes = \
        _genes_similarity(candidate_data=candidate_data,
                          what=similarity_of or what,
                          method=genes_similarity_method)

    linkage = \
        _cluster_genes(similarities_between_candidate_genes=similarities_between_candidate_genes,
                       genes_cluster_method=genes_cluster_method)

    rare_gene_indices_of_modules = \
        _identify_genes(candidate_genes_indices=candidate_genes_indices,
                        similarities_between_candidate_genes=similarities_between_candidate_genes,
                        linkage=linkage,
                        min_module_correlation=min_module_correlation)

    max_cells_of_modules = \
        int(max_cells_of_random_pile * adata.n_obs / target_pile_size)
    ut.log_calc('max_cells_of_modules', max_cells_of_modules)

    related_gene_indices_of_modules = \
        _related_genes(adata_of_all_genes_of_all_cells=adata,
                       what=what,
                       rare_gene_indices_of_modules=rare_gene_indices_of_modules,
                       allowed_genes_mask=allowed_genes_mask,
                       min_genes_of_modules=min_genes_of_modules,
                       min_cells_of_modules=min_cells_of_modules,
                       max_cells_of_modules=max_cells_of_modules,
                       min_gene_maximum=min_gene_maximum,
                       min_related_gene_fold_factor=min_related_gene_fold_factor)

    _identify_cells(adata_of_all_genes_of_all_cells=adata,
                    what=what,
                    related_gene_indices_of_modules=related_gene_indices_of_modules,
                    min_cells_of_modules=min_cells_of_modules,
                    max_cells_of_modules=max_cells_of_modules,
                    min_cell_module_total=min_cell_module_total,
                    rare_module_of_cells=rare_module_of_cells)

    _compress_modules(adata_of_all_genes_of_all_cells=adata,
                      what=what,
                      min_cells_of_modules=min_cells_of_modules,
                      max_cells_of_modules=max_cells_of_modules,
                      target_metacell_size=target_metacell_size,
                      min_modules_size_factor=min_modules_size_factor,
                      related_gene_indices_of_modules=related_gene_indices_of_modules,
                      rare_module_of_cells=rare_module_of_cells,
                      list_of_names_of_genes_of_modules=list_of_names_of_genes_of_modules)

    return _results(adata=adata,
                    rare_module_of_cells=rare_module_of_cells,
                    list_of_names_of_genes_of_modules=list_of_names_of_genes_of_modules,
                    inplace=inplace)


@ut.timed_call()
def _pick_candidates(
    *,
    adata_of_all_genes_of_all_cells: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    max_gene_cell_fraction: float,
    min_gene_maximum: int,
    min_genes_of_modules: int,
    allowed_genes_mask: ut.NumpyVector,
) -> Optional[Tuple[AnnData, ut.NumpyVector]]:
    data = ut.get_vo_proper(adata_of_all_genes_of_all_cells, what,
                            layout='column_major')
    nnz_cells_of_genes = ut.nnz_per(data, per='column')

    nnz_cell_fraction_of_genes = nnz_cells_of_genes / \
        adata_of_all_genes_of_all_cells.n_obs
    nnz_cell_fraction_mask_of_genes = \
        nnz_cell_fraction_of_genes <= max_gene_cell_fraction

    max_umis_of_genes = ut.max_per(data, per='column')
    max_umis_mask_of_genes = max_umis_of_genes >= min_gene_maximum

    candidates_mask_of_genes = \
        max_umis_mask_of_genes & nnz_cell_fraction_mask_of_genes & allowed_genes_mask
    ut.log_calc('candidate_genes', candidates_mask_of_genes)

    candidate_genes_indices = np.where(candidates_mask_of_genes)[0]
    candidate_genes_count = candidate_genes_indices.size
    if candidate_genes_count < min_genes_of_modules:
        return None

    candidate_data = \
        ut.slice(adata_of_all_genes_of_all_cells, name='.candidate_genes',
                 vars=candidate_genes_indices, top_level=False)
    return candidate_data, candidate_genes_indices


@ut.timed_call()
def _genes_similarity(
    *,
    candidate_data: AnnData,
    what: Union[str, ut.Matrix],
    method: str,
) -> ut.NumpyMatrix:
    similarity = \
        compute_var_var_similarity(candidate_data, what,
                                   method=method, inplace=False)
    assert similarity is not None
    return ut.to_numpy_matrix(similarity, only_extract=True)


@ut.timed_call()
def _cluster_genes(
    similarities_between_candidate_genes: ut.NumpyMatrix,
    genes_cluster_method: str,
) -> List[Tuple[int, int]]:
    with ut.timed_step('scipy.pdist'):
        ut.timed_parameters(size=similarities_between_candidate_genes.shape[0])
        distances = scd.pdist(similarities_between_candidate_genes)

    with ut.timed_step('scipy.linkage'):
        ut.timed_parameters(size=distances.shape[0],
                            method=genes_cluster_method)
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
    combined_candidate_indices = \
        {index: [index] for index in range(candidate_genes_count)}

    for link_index, link_data in enumerate(linkage):
        link_index += candidate_genes_count

        left_index = int(link_data[0])
        right_index = int(link_data[1])

        left_combined_candidates = \
            combined_candidate_indices.get(left_index)
        right_combined_candidates = \
            combined_candidate_indices.get(right_index)
        if not left_combined_candidates or not right_combined_candidates:
            continue

        link_combined_candidates = \
            sorted(left_combined_candidates + right_combined_candidates)
        assert link_combined_candidates
        link_similarities = \
            similarities_between_candidate_genes[link_combined_candidates,  #
                                                 :][:,
                                                    link_combined_candidates]
        average_link_similarity = np.nanmean(link_similarities)
        if average_link_similarity < min_module_correlation:
            continue

        combined_candidate_indices[link_index] = link_combined_candidates
        del combined_candidate_indices[left_index]
        del combined_candidate_indices[right_index]

    return [candidate_genes_indices[candidate_indices]
            for candidate_indices
            in combined_candidate_indices.values()]


@ut.timed_call()
def _related_genes(  # pylint: disable=too-many-branches,too-many-statements
    *,
    adata_of_all_genes_of_all_cells: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    rare_gene_indices_of_modules: List[List[int]],
    allowed_genes_mask: ut.NumpyVector,
    min_genes_of_modules: int,
    min_gene_maximum: int,
    min_cells_of_modules: int,
    max_cells_of_modules: int,
    min_related_gene_fold_factor: float,
) -> List[List[int]]:
    total_all_cells_umis_of_all_genes = \
        ut.get_v_numpy(adata_of_all_genes_of_all_cells, what, sum=True)

    related_data_of_genes: Dict[int, Tuple[bool, float, float, int]] = {}

    ut.log_calc('genes for modules:')
    modules_count = 0
    for rare_gene_indices_of_module in rare_gene_indices_of_modules:
        if len(rare_gene_indices_of_module) < min_genes_of_modules:
            continue

        module_index = modules_count
        modules_count += 1

        with ut.log_step('- module', module_index):
            ut.log_calc('rare_gene_names',
                        sorted(adata_of_all_genes_of_all_cells.var_names[rare_gene_indices_of_module]))

            adata_of_module_genes_of_all_cells = \
                ut.slice(adata_of_all_genes_of_all_cells,
                         name=f'.module{module_index}.rare_genes',
                         vars=rare_gene_indices_of_module,
                         top_level=False)

            total_module_genes_umis_of_all_cells = \
                ut.get_o_numpy(adata_of_module_genes_of_all_cells,
                               what, sum=True)

            mask_of_expressed_cells = total_module_genes_umis_of_all_cells > 0

            expressed_cells_count = np.sum(mask_of_expressed_cells)

            if expressed_cells_count > max_cells_of_modules:
                if ut.logging_calc():
                    ut.log_calc('expressed_cells',
                                ut.mask_description(mask_of_expressed_cells)
                                + ' (too many)')
                continue

            if expressed_cells_count < min_cells_of_modules:
                if ut.logging_calc():
                    ut.log_calc('expressed_cells',
                                ut.mask_description(mask_of_expressed_cells)
                                + ' (too few)')
                continue

            ut.log_calc('expressed_cells', mask_of_expressed_cells)

            adata_of_all_genes_of_expressed_cells_of_module = \
                ut.slice(adata_of_all_genes_of_all_cells,
                         name=f'.module{module_index}.rare_cells',
                         obs=mask_of_expressed_cells,
                         top_level=False)

            total_expressed_cells_umis_of_all_genes = \
                ut.get_v_numpy(adata_of_all_genes_of_expressed_cells_of_module,
                               what, sum=True)

            data = ut.get_vo_proper(adata_of_all_genes_of_expressed_cells_of_module, what,
                                    layout='column_major')
            max_expressed_cells_umis_of_all_genes = \
                ut.max_per(data, per='column')

            total_background_cells_umis_of_all_genes = \
                total_all_cells_umis_of_all_genes - total_expressed_cells_umis_of_all_genes

            expressed_cells_fraction_of_all_genes = \
                total_expressed_cells_umis_of_all_genes \
                / sum(total_expressed_cells_umis_of_all_genes)

            background_cells_fraction_of_all_genes = \
                total_background_cells_umis_of_all_genes \
                / sum(total_background_cells_umis_of_all_genes)

            mask_of_related_genes = \
                allowed_genes_mask \
                & (max_expressed_cells_umis_of_all_genes >= min_gene_maximum) \
                & (expressed_cells_fraction_of_all_genes
                   >= background_cells_fraction_of_all_genes
                   * (2 ** min_related_gene_fold_factor))

            related_gene_indices = np.where(mask_of_related_genes)[0]
            assert np.all(mask_of_related_genes[rare_gene_indices_of_module])

            if ut.logging_calc():
                ut.log_calc('candidate_gene_names',
                            sorted(adata_of_all_genes_of_all_cells.var_names[related_gene_indices]))

            for gene_index in related_gene_indices:
                current_gene_data = \
                    related_data_of_genes.get(gene_index,
                                              (False, 0.0, 0.0, module_index))

                if gene_index in rare_gene_indices_of_module:
                    gene_data = (True, 0.0, 0.0, module_index)
                    assert not current_gene_data[0]
                elif background_cells_fraction_of_all_genes[gene_index] == 0:
                    gene_data = \
                        (False,
                         expressed_cells_fraction_of_all_genes[gene_index],
                         0.0,
                         module_index)
                else:
                    gene_data = \
                        (False,
                         0.0,
                         expressed_cells_fraction_of_all_genes[gene_index]
                         / background_cells_fraction_of_all_genes[gene_index],
                         module_index)

                if gene_data > current_gene_data:
                    related_data_of_genes[gene_index] = gene_data

    related_gene_indices_of_modules: List[List[int]] = [[]] * modules_count

    ut.log_calc('related genes for modules:')
    for module_index in range(modules_count):
        related_gene_indices_of_module = [gene_index
                                          for gene_index, gene_data
                                          in related_data_of_genes.items()
                                          if gene_data[3] == module_index]
        related_gene_indices_of_modules[module_index] = related_gene_indices_of_module
        if ut.logging_calc():
            with ut.log_step('- module', module_index,
                             formatter=lambda module_index:
                             ut.progress_description(modules_count,
                                                     module_index, 'module')):
                ut.log_calc('related_gene_names',
                            sorted(adata_of_all_genes_of_all_cells.var_names[related_gene_indices_of_module]))

    return related_gene_indices_of_modules


@ut.timed_call()
def _identify_cells(
    *,
    adata_of_all_genes_of_all_cells: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    related_gene_indices_of_modules: List[List[int]],
    min_cell_module_total: int,
    min_cells_of_modules: int,
    max_cells_of_modules: int,
    rare_module_of_cells: ut.NumpyVector,
) -> None:
    max_strength_of_cells = np.zeros(adata_of_all_genes_of_all_cells.n_obs)

    ut.log_calc('cells for modules:')
    modules_count = len(related_gene_indices_of_modules)
    for module_index, related_gene_indices_of_module \
            in enumerate(related_gene_indices_of_modules):
        if len(related_gene_indices_of_module) == 0:
            continue

        with ut.log_step('- module', module_index,
                         formatter=lambda module_index:
                         ut.progress_description(modules_count,
                                                 module_index, 'module')):
            adata_of_related_genes_of_all_cells = \
                ut.slice(adata_of_all_genes_of_all_cells,
                         name=f'.module{module_index}.related_genes',
                         vars=related_gene_indices_of_module,
                         top_level=False)
            total_related_genes_of_all_cells = \
                ut.get_o_numpy(adata_of_related_genes_of_all_cells,
                               what, sum=True)

            mask_of_strong_cells_of_module = \
                total_related_genes_of_all_cells >= min_cell_module_total

            strong_cells_count = np.sum(mask_of_strong_cells_of_module)

            if strong_cells_count > max_cells_of_modules:
                if ut.logging_calc():
                    ut.log_calc('strong_cells',
                                ut.mask_description(  #
                                    mask_of_strong_cells_of_module)
                                + ' (too many)')
                related_gene_indices_of_module.clear()
                continue

            if strong_cells_count < min_cells_of_modules:
                if ut.logging_calc():
                    ut.log_calc('strong_cells',
                                ut.mask_description(  #
                                    mask_of_strong_cells_of_module)
                                + ' (too few)')
                related_gene_indices_of_module.clear()
                continue

            ut.log_calc('strong_cells', mask_of_strong_cells_of_module)

            mask_of_strong_cells_of_module &= \
                total_related_genes_of_all_cells >= max_strength_of_cells
            rare_module_of_cells[mask_of_strong_cells_of_module] = module_index


@ut.timed_call()
def _compress_modules(
    *,
    adata_of_all_genes_of_all_cells: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    min_cells_of_modules: int,
    max_cells_of_modules: int,
    target_metacell_size: int,
    min_modules_size_factor: float,
    related_gene_indices_of_modules: List[List[int]],
    rare_module_of_cells: ut.NumpyVector,
    list_of_names_of_genes_of_modules: List[ut.NumpyVector],
) -> None:
    min_umis_of_modules = target_metacell_size * min_modules_size_factor
    ut.log_calc('min_umis_of_modules', min_umis_of_modules)

    total_all_genes_of_all_cells = \
        ut.get_o_numpy(adata_of_all_genes_of_all_cells, what, sum=True)

    cell_counts_of_modules: List[int] = []

    ut.log_calc('compress modules:')
    next_module_index = 0
    modules_count = len(related_gene_indices_of_modules)
    for module_index, gene_indices_of_module in enumerate(related_gene_indices_of_modules):
        if len(gene_indices_of_module) == 0:
            continue

        with ut.log_step('- module', module_index,
                         formatter=lambda module_index:
                         ut.progress_description(modules_count,
                                                 module_index, 'module')):
            module_cells_mask = rare_module_of_cells == module_index
            module_cells_count = np.sum(module_cells_mask)
            module_umis_count = \
                np.sum(total_all_genes_of_all_cells[module_cells_mask])

            if module_cells_count < min_cells_of_modules:
                if ut.logging_calc():
                    ut.log_calc('cells',
                                str(module_cells_count) + ' (too few)')
                rare_module_of_cells[module_cells_mask] = -1
                continue

            if module_cells_count > max_cells_of_modules:
                if ut.logging_calc():
                    ut.log_calc('cells',
                                str(module_cells_count) + ' (too many)')
                rare_module_of_cells[module_cells_mask] = -1
                continue

            ut.log_calc('cells', module_cells_count)

            if module_umis_count < min_umis_of_modules:
                if ut.logging_calc():
                    ut.log_calc('UMIs', str(module_umis_count) + ' (too few)')
                rare_module_of_cells[module_cells_mask] = -1
                continue

            ut.log_calc('UMIs', module_umis_count)

            if module_index != next_module_index:
                ut.log_calc('is reindexed to', next_module_index)
                rare_module_of_cells[module_cells_mask] = next_module_index
                module_index = next_module_index

            next_module_index += 1

            if ut.logging_calc():
                cell_counts_of_modules.append(np.sum(module_cells_mask))
            list_of_names_of_genes_of_modules.append(np.array(  #
                sorted(adata_of_all_genes_of_all_cells.var_names[gene_indices_of_module])))

    if ut.logging_calc():
        ut.log_calc('final modules:')
        modules_count = len(list_of_names_of_genes_of_modules)
        for module_index, (module_cells_count, module_gene_names) \
                in enumerate(zip(cell_counts_of_modules, list_of_names_of_genes_of_modules)):
            with ut.log_step('- module', module_index,
                             formatter=lambda module_index:
                             ut.progress_description(modules_count,
                                                     module_index, 'module')):
                ut.log_calc('cells', module_cells_count)
                ut.log_calc('genes', module_gene_names)


def _results(
    *,
    adata: AnnData,
    rare_module_of_cells: ut.NumpyVector,
    list_of_names_of_genes_of_modules: List[ut.NumpyVector],
    inplace: bool
) -> Optional[Tuple[ut.PandasFrame, ut.PandasFrame, ut.NumpyVector]]:
    assert np.max(rare_module_of_cells) \
        == len(list_of_names_of_genes_of_modules) - 1

    array_of_names_of_genes_of_modules = \
        np.array(list_of_names_of_genes_of_modules, dtype='object')

    rare_module_of_genes = np.full(adata.n_vars, -1, dtype='int32')
    genes_series = \
        ut.to_pandas_series(rare_module_of_genes, index=adata.var_names)
    for module_index, list_of_names_of_genes_of_module in enumerate(list_of_names_of_genes_of_modules):
        genes_series[list_of_names_of_genes_of_module] = module_index

    if inplace:
        ut.set_m_data(adata, 'rare_gene_modules',
                      array_of_names_of_genes_of_modules,
                      formatter=lambda matrix:
                      '[ ' + ', '.join([f'[ {", ".join(list(array))} ]'
                                        for array in matrix]) + ' ]')
        ut.set_v_data(adata, 'genes_rare_gene_module', rare_module_of_genes,
                      formatter=ut.groups_description)
        ut.set_v_data(adata, 'rare_gene', rare_module_of_genes >= 0)
        ut.set_o_data(adata, 'cells_rare_gene_module', rare_module_of_cells,
                      formatter=ut.groups_description)
        ut.set_o_data(adata, 'rare_cell', rare_module_of_cells >= 0)
        return None

    obs_metrics = ut.to_pandas_frame(index=adata.obs_names)
    var_metrics = ut.to_pandas_frame(index=adata.var_names)

    obs_metrics['cells_rare_gene_module'] = rare_module_of_cells
    obs_metrics['rare_cell'] = rare_module_of_cells >= 0
    var_metrics['genes_rare_gene_module'] = rare_module_of_genes
    var_metrics['rare_gene'] = rare_module_of_genes >= 0

    ut.log_return('rare_gene_modules', len(list_of_names_of_genes_of_modules))
    ut.log_return('genes_rare_gene_module', rare_module_of_genes,
                  formatter=ut.groups_description)
    ut.log_return('rare_genes', rare_module_of_genes >= 0)
    ut.log_return('cells_rare_gene_module', rare_module_of_cells,
                  formatter=ut.groups_description)
    ut.log_return('rare_cells', rare_module_of_cells >= 0)

    return obs_metrics, var_metrics, array_of_names_of_genes_of_modules
