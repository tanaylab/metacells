'''
Detect rare genes modules.
'''

from typing import List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy.cluster.hierarchy as sch  # type: ignore
import scipy.spatial.distance as scd  # type: ignore
from anndata import AnnData

import metacells.utilities as ut

from .similarity import compute_var_var_similarity

__all__ = [
    'find_rare_genes_modules',
]


@ut.timed_call()
@ut.expand_doc()
def find_rare_genes_modules(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    adata: AnnData,
    of: Optional[str] = None,
    *,
    maximal_fraction_of_cells_of_genes: float = 1e-3,
    minimal_max_umis_of_genes: int = 6,
    log_similarity: bool = False,
    log_similarity_base: Optional[float] = None,
    log_similarity_normalization: float = -1,
    repeated_similarity: bool = True,
    cluster_method_of_genes: str = 'ward',
    minimal_size_of_modules: int = 4,
    minimal_correlation_of_modules: float = 0.1,
    minimal_umis_of_module_of_cells: int = 6,
    inplace: bool = True,
    intermediate: bool = True,
) -> Optional[Tuple[ut.PandasFrame, ut.PandasFrame, np.ndarray]]:
    '''
    Detect rare genes modules based ``of`` some data (by default, the focus).

    Rare gene modules include genes which are weakly and rarely expressed, yet are highly correlated
    with each other, allowing for robust detection. Global analysis algorithms (such as metacells)
    tend to ignore or at least discount such genes.

    It is therefore useful to explicitly identify, in a pre-processing step, the few cells which
    express such rare gene modules. Once identified, these cells can be exempt from the global
    algorithm, or the global algorithm can be tweaked in some way to pay extra attention to them.

    **Input**

    A :py:func:`metacells.utilities.preparation.prepare`-ed annotated ``adata``, where the
    observations are cells and the variables are genes.

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Returns**

    Observation (Cell) Annotations
        ``cells_rare_gene_module``
            The index of the rare gene module each cell expresses the most, or ``-1`` in the common
            case it does not express any rare genes module.

        ``rare_cells``
            A boolean mask for the (few) cells that express a rare gene module.

    Variable (Gene) Annotations
        ``genes_rare_gene_module``
            The index of the rare gene module each gene belongs to, or ``-1`` in the common case it
            does not belong to any rare genes module.

        ``rare_genes``
            A boolean mask for the genes in any of the rare gene modules.

    Unstructured Annotations
        ``rare_gene_modules``
            An array of rare gene modules, where every entry is the array of the names of the genes
            of the module.

    If ``inplace``, these are written to to the data, and the function returns ``None``. Otherwise
    they are returned as tuple containing two data frames and an array.

    **Computation Parameters**

    1. Pick as candidates all genes that are expressed in more than
       ``maximal_fraction_of_cells_of_genes`` (default: {maximal_fraction_of_cells_of_genes}), and
       whose maximal number of UMIs in a cell is at least ``minimal_max_umis_of_genes``
       (default: {minimal_max_umis_of_genes}).

    2. Compute the similarity between the genes using
       :py:func:`metacells.tools.similarity.compute_var_var_similarity`. Pass it ``log_similarity``
       (default: {log_similarity}), ``log_similarity_base`` (default: {log_similarity_base}), and
       ``log_similarity_normalization`` (default: {log_similarity_normalization}) as well as
       ``repeated_similarity`` (default: {repeated_similarity}).

    3. Create a hierarchical clustering of the candidate genes using the ``cluster_method_of_genes``
       (default: {cluster_method_of_genes}).

    4. Identify gene modules in the hierarchical clustering which contain at least
       ``minimal_size_of_modules`` genes (default: {minimal_size_of_modules}), with an average
       gene-gene cross-correlation of at least ``minimal_correlation_of_modules`` (default:
       {minimal_correlation_of_modules}).

    5. Associate cells with a gene module if they contain at least
       ``minimal_umis_of_module_of_cells`` (default: {minimal_umis_of_module_of_cells}) UMIs from
       the module. In the very rare case a cell contains at least the minimal number of UMIs for
       multiple gene modules, associate it with the gene module which is most enriched (relative to
       the least enriched cell associated with the gene module). Gene modules with no associated
       cells are discarded.
    '''

    with ut.focus_on(ut.get_vo_data, adata, of, intermediate=intermediate):
        cells_count = adata.n_obs
        genes_count = adata.n_vars

        rare_module_of_cells = np.full(cells_count, -1)
        rare_module_of_genes = np.full(genes_count, -1)
        list_of_names_of_genes_of_modules: List[np.ndarray] = []

        total_umis_of_cells = ut.get_per_obs(adata, ut.sum_per).proper
        max_umis_of_genes = ut.get_per_var(adata, ut.max_per).proper
        nnz_cells_of_genes = ut.get_per_var(adata, ut.nnz_per).proper

        def results() -> Optional[Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]]:
            array_of_names_of_genes_of_modules = \
                np.array(list_of_names_of_genes_of_modules, dtype='object')

            if inplace:
                adata.obs['cells_rare_gene_module'] = rare_module_of_cells
                adata.obs['rare_cells'] = rare_module_of_cells >= 0
                adata.var['genes_rare_gene_module'] = rare_module_of_genes
                adata.var['rare_genes'] = rare_module_of_genes >= 0
                adata.uns['rare_gene_modules'] = array_of_names_of_genes_of_modules
                return None

            obs_metrics = pd.DataFrame(index=adata.obs_names)
            var_metrics = pd.DataFrame(index=adata.var_names)

            obs_metrics['cells_rare_gene_module'] = rare_module_of_cells
            obs_metrics['rare_cells'] = rare_module_of_cells >= 0
            var_metrics['genes_rare_gene_module'] = rare_module_of_genes
            var_metrics['rare_genes'] = rare_module_of_genes >= 0

            return obs_metrics, var_metrics, array_of_names_of_genes_of_modules

        with ut.timed_step('.pick_candidate_genes'):
            nnz_cells_fraction_of_genes = nnz_cells_of_genes / cells_count

            nnz_cells_fraction_mask_of_genes = \
                nnz_cells_fraction_of_genes <= maximal_fraction_of_cells_of_genes

            max_umis_mask_of_genes = max_umis_of_genes >= minimal_max_umis_of_genes

            candidates_mask_of_genes = \
                max_umis_mask_of_genes & nnz_cells_fraction_mask_of_genes

            candidate_genes_indices = np.where(candidates_mask_of_genes)[0]

            candidate_genes_count = candidate_genes_indices.size
            if candidate_genes_count < minimal_size_of_modules:
                return results()

            candidate_data = ut.slice(adata, vars=candidate_genes_indices)

        with ut.timed_step('.similarity'):
            similarity = \
                compute_var_var_similarity(candidate_data,
                                           repeated=repeated_similarity,
                                           log=log_similarity,
                                           log_base=log_similarity_base,
                                           log_normalization=log_similarity_normalization,
                                           inplace=False)
            assert similarity is not None
            similarities_between_candidate_genes = \
                ut.to_proper_matrix(similarity)

        with ut.timed_step('.cluster'):
            with ut.timed_step('pdist'):
                ut.timed_parameters(  #
                    size=similarities_between_candidate_genes.shape[0])
                distances = scd.pdist(similarities_between_candidate_genes)
            with ut.timed_step('linkage'):
                ut.timed_parameters(size=distances.shape[0],
                                    method=cluster_method_of_genes)
                linkage = \
                    sch.linkage(distances, method=cluster_method_of_genes)

        with ut.timed_step('.identify_genes'):
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
                    sorted(left_combined_candidates
                           + right_combined_candidates)
                assert link_combined_candidates
                link_similarities = \
                    similarities_between_candidate_genes[link_combined_candidates,  #
                                                         :][:,  #
                                                            link_combined_candidates]
                average_link_similarity = np.nanmean(link_similarities)
                if average_link_similarity < minimal_correlation_of_modules:
                    continue

                combined_candidate_indices[link_index] = link_combined_candidates
                del combined_candidate_indices[left_index]
                del combined_candidate_indices[right_index]

        with ut.timed_step('.identify_cells'):
            maximal_strength_of_cells = np.zeros(cells_count)
            gene_indices_of_modules: List[np.ndarray] = []
            candidate_umis = \
                ut.get_vo_data(candidate_data, of, layout='column_major')

            for link_index, module_candidate_indices in combined_candidate_indices.items():
                if len(module_candidate_indices) < minimal_size_of_modules:
                    continue

                total_umis_of_module_of_cells = \
                    ut.to_dense_vector(candidate_umis[:,
                                                      module_candidate_indices].sum(axis=1))
                assert total_umis_of_module_of_cells.size == cells_count

                minimal_total_umis_of_module_mask_of_cells = \
                    total_umis_of_module_of_cells >= minimal_umis_of_module_of_cells
                strong_cell_indices = \
                    np.where(minimal_total_umis_of_module_mask_of_cells)[0]
                if strong_cell_indices.size == 0:
                    continue

                total_umis_of_module_of_strong_cells = \
                    total_umis_of_module_of_cells[strong_cell_indices]
                total_umis_of_strong_cells = total_umis_of_cells[strong_cell_indices]
                fraction_of_module_of_strong_cells = \
                    total_umis_of_module_of_strong_cells / total_umis_of_strong_cells

                minimal_strong_fraction = fraction_of_module_of_strong_cells.min()
                assert minimal_strong_fraction > 0
                module_strength_of_strong_cells = \
                    fraction_of_module_of_strong_cells / minimal_strong_fraction

                maximal_strength_of_strong_cells = \
                    maximal_strength_of_cells[strong_cell_indices]
                stronger_cells_mask = module_strength_of_strong_cells \
                    > maximal_strength_of_strong_cells

                stronger_cell_indices = strong_cell_indices[stronger_cells_mask]
                if stronger_cell_indices.size == 0:
                    continue

                maximal_strength_of_cells[stronger_cell_indices] = \
                    maximal_strength_of_strong_cells[stronger_cells_mask]
                module_index = len(gene_indices_of_modules)
                module_gene_indices = \
                    candidate_genes_indices[module_candidate_indices]
                gene_indices_of_modules.append(module_gene_indices)
                rare_module_of_genes[module_gene_indices] = module_index
                rare_module_of_cells[strong_cell_indices] = module_index

            if len(gene_indices_of_modules) == 0:
                return results()

        with ut.timed_step('.results'):
            for raw_module_index, module_gene_indices \
                    in enumerate(gene_indices_of_modules):
                module_cell_indices = \
                    np.where(rare_module_of_cells == raw_module_index)[0]
                if module_cell_indices.size == 0:
                    continue

                module_index = len(list_of_names_of_genes_of_modules)

                if raw_module_index != module_index:
                    rare_module_of_cells[module_cell_indices] = module_index
                    rare_module_of_genes[module_gene_indices] = module_index

                names_of_genes_of_module = \
                    np.array(adata.var_names[module_gene_indices])
                list_of_names_of_genes_of_modules.append(  #
                    names_of_genes_of_module)

            assert len(list_of_names_of_genes_of_modules) > 0

            return results()
