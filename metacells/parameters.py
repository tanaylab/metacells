'''
Defaults
--------
'''

from typing import Optional, Union
import metacells.utilities.typing as utt
import metacells.utilities.partition as utp


#: The generic random seed. The default of ``0`` makes for a different result each time the code
#: is run. For replicable results, provide a non-zero value. Used by too many functions to list
#: here.
random_seed: int = 0

#: The generic minimal "significant" gene fraction. See
#: :py:const:`noisy_lonely_min_gene_fraction`,
#: :py:const:`rare_max_gene_cell_fraction`,
#: :py:const:`feature_min_gene_fraction`
#: and
#: :py:func:`metacells.tools.high.find_high_fraction_genes`.
significant_gene_fraction: float = 1e-5

#: The generic minimal "significant" gene normalized variance. See
#: :py:func:`metacells.tools.high.find_high_normalized_variance_genes`
#: and
#: :py:const:`noisy_lonely_min_gene_normalized_variance`.
significant_gene_normalized_variance: float = 2.5

#: The generic minimal "significant" gene relative variance. See
#: :py:func:`metacells.tools.high.find_high_relative_variance_genes`.
significant_gene_relative_variance: float = 0.1

#: The generic minimal "significant" gene similarity. See
#: :py:const:`noisy_lonely_max_gene_similarity`
#: and
#: :py:const:`rare_min_module_correlation`.
significant_gene_similarity: float = 0.15

#: The generic "significant" fold factor. See
#: :py:const:`deviants_min_gene_fold_factor`
#: and
#: :py:const:`dissolve_min_convincing_gene_fold_factor`.
significant_gene_fold_factor: float = 3.0

#: The generic minimal value (number of UMIs) we can say is "significant" given the technical noise.
#: See
#: py:const:`rare_min_gene_maximum`,
#: py:const:`rare_min_cell_module_total`
#: and
#: py:const:`cells_similarity_log_normalization`.
significant_value: int = 7

#: The generic quantile of the cells total size to use for downsampling the cells for some
#: purpose. See
#: :py:const:`noisy_lonely_downsample_cell_quantile`,
#: :py:const:`feature_downsample_cell_quantile`,
#: :py:const:`excess_downsample_cell_quantile`
#: and
#: :py:func:`metacells.tools.downsample.downsample_cells`.
downsample_cell_quantile: float = 0.05

#: The generic target number of observations (cells) in a pile, allowing us to
#: directly compute groups (metacells) for it. See
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
target_pile_size: int = 10000

#: The generic target total metacell size. See
#: :py:const:`candidates_target_metacell_size`,
#: :py:const:`dissolve_target_metacell_size`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
target_metacell_size: int = 160000

#: The genetic size of each cell for computing each metacell's size. See
#: :py:const:`candidates_cell_sizes`,
#: :py:const:`dissolve_cell_sizes`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`
#: and
#: :py:func:`metacells.pipeline.direct.direct_pipeline`,
cell_sizes: Union[str, utt.Vector] = '<of>|sum_per_obs'

#: The generic maximal group size factor, above which we should split it. See
#: :py:const:`pile_min_split_size_factor`
#: and
#: :py:const:`candidates_min_split_size_factor`.
min_split_size_factor: float = 2.0

#: The generic minimal group size factor, below which we should consider dissolving it. See
#: :py:const:`pile_min_robust_size_factor`
#: and
#: :py:const:`dissolve_min_robust_size_factor`.
min_robust_size_factor: float = 0.5

#: The generic maximal group size factor, below which we should merge it. See
#: :py:const:`pile_max_merge_size_factor`
#: and
#: :py:const:`candidates_max_merge_size_factor`.
max_merge_size_factor: float = 0.25

#: The minimal size factor of a pile, above which we can split it. See
#: :py:const:`min_split_size_factor`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
pile_min_split_size_factor: float = min_split_size_factor

#: The generic minimal pile size factor, below which we should consider dissolving it. See
#: :py:const:`min_robust_size_factor`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
pile_min_robust_size_factor: float = min_robust_size_factor

#: The maximal size factor of a file, below which we can merge it. See
#: :py:const:`max_merge_size_factor`
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
pile_max_merge_size_factor: float = max_merge_size_factor

#: The minimal total value for a cell to be considered "properly sampled". See
#: :py:func:`metacells.tools.properly_sampled.find_properly_sampled_cells`,
#: :py:func:`metacells.pipeline.clean.extract_clean_data`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
properly_sampled_min_cell_total: int = 800

#: The maximal total value for a cell to be considered "properly sampled". See
#: :py:func:`metacells.tools.properly_sampled.find_properly_sampled_cells`,
#: :py:func:`metacells.pipeline.clean.extract_clean_data`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
properly_sampled_max_cell_total: Optional[int] = None

#: The minimal total value for a gene to be considered "properly sampled". See
#: :py:func:`metacells.tools.properly_sampled.find_properly_sampled_genes`,
#: :py:func:`metacells.pipeline.clean.extract_clean_data`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
properly_sampled_min_gene_total: int = 1

#: The number of randomly selected cells to use for computing "noisy lonely" genes. See
#: :py:func:`metacells.tools.noisy_lonely.find_noisy_lonely_genes`,
#: :py:func:`metacells.pipeline.clean.extract_clean_data`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
noisy_lonely_max_sampled_cells: int = 10000

#: The quantile of the cells total size to use for downsampling the cells for computing "noisy
#: lonely" genes. See
#: :py:const:`downsample_cell_quantile`,
#: :py:func:`metacells.tools.noisy_lonely.find_noisy_lonely_genes`.
#: :py:func:`metacells.pipeline.clean.extract_clean_data`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
noisy_lonely_downsample_cell_quantile: float = downsample_cell_quantile

#: The minimal overall expression of a gene to be considered "noisy". See
#: :py:const:`significant_gene_fraction`,
#: :py:func:`metacells.tools.noisy_lonely.find_noisy_lonely_genes`,
#: :py:func:`metacells.pipeline.clean.extract_clean_data`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
noisy_lonely_min_gene_fraction: float = significant_gene_fraction

#: The minimal normalized variance of a gene to be considered "noisy". See
#: :py:const:`significant_gene_normalized_variance`,
#: :py:func:`metacells.tools.noisy_lonely.find_noisy_lonely_genes`,
#: :py:func:`metacells.pipeline.clean.extract_clean_data`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
noisy_lonely_min_gene_normalized_variance: float = significant_gene_normalized_variance

#: The maximal similarity between a gene and another gene to be considered "lonely". See
#: :py:const:`significant_gene_similarity`,
#: :py:func:`metacells.tools.noisy_lonely.find_noisy_lonely_genes`,
#: :py:func:`metacells.pipeline.clean.extract_clean_data`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
noisy_lonely_max_gene_similarity: float = significant_gene_similarity

#: The maximal expression of a gene (in any cell) to be considered "rare". See
#: :py:func:`metacells.tools.rare.find_rare_genes_modules`.
#: Note this is different from the lower
#: :py:const:`significant_gene_fraction`
#: which applies to the mean expression of the gene across the cells.
rare_max_gene_cell_fraction: float = 1e-3

#: The minimal maximum-across-all-cells value of a gene to be considered as a candidate for rare
#: gene modules. See
#: :py:const:`significant_value` #: and
#: :py:func:`metacells.tools.rare.find_rare_genes_modules`.
rare_min_gene_maximum: int = significant_value

#: Whether to compute repeated gene-cgene similarity for computing the rare gene modules. See
#: :py:func:`metacells.tools.rare.find_rare_genes_modules`.
rare_repeated_similarity: bool = True

#: The hierarchical clustering method to use for computing the rare gene modules. See
#: :py:func:`metacells.tools.rare.find_rare_genes_modules`.
rare_genes_cluster_method: str = 'ward'

#: The minimal number of genes in a rare gene module. See
#: :py:func:`metacells.tools.rare.find_rare_genes_modules`.
rare_min_size_of_modules: int = 4

#: The minimal average correlation between the genes in a rare gene module. See
#: :py:func:`metacells.tools.rare.find_rare_genes_modules`.
#: Note this is different from the higher
#: :py:const:`significant_gene_similarity`
#: which applies to the maximal correlation between some pairs of genes.
rare_min_module_correlation: float = 0.1

#: The minimal total value of the genes of a rare gene module for considering a cell as
#: expressing it. See
#: :py:const:`significant_value`
#: and
#: :py:func:`metacells.tools.rare.find_rare_genes_modules`.
rare_min_cell_module_total: int = significant_value

#: The quantile of the cells total size to use for downsampling the cells for computing
#: "feature" genes. See
#: :py:const:`downsample_cell_quantile`,
#: :py:func:`metacells.tools.downsample.downsample_cells`,
#: :py:func:`metacells.pipeline.direct.extract_feature_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
feature_downsample_cell_quantile: float = downsample_cell_quantile

#: The minimal mean fraction of a gene to be considered a "feature". See
#: :py:const:`significant_gene_fraction`,
#: :py:func:`metacells.tools.high.find_high_fraction_genes`,
#: :py:func:`metacells.pipeline.direct.extract_feature_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
feature_min_gene_fraction: float = significant_gene_fraction

#: The minimal relative variance of a gene to be considered a "feature". See
#: :py:func:`metacells.tools.high.find_high_relative_variance_genes`,
#: :py:func:`metacells.pipeline.direct.extract_feature_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
feature_min_gene_relative_variance: float = 0.1

#: Whether to compute cell-cell similarity using the log (base 2) of the data. See
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
cells_similarity_log_data: bool = True

#: The normalization factor to use if/when computing the log (base 2) of the data for directly
#: computing the metacells. See
#: :py:const:`significant_value`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
cells_similarity_log_normalization: float = 1/significant_value

#: Whether to compute repeated cell-cell similarity for directly computing the metacells. See
#: :py:func:`metacells.tools.similarity.compute_obs_obs_similarity`,
#: :py:func:`metacells.pipeline.direct.extract_feature_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
cells_repeated_similarity: bool = True

#: The target K for building the K-Nearest-Neighbors graph. See
#: :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph`,
#: :py:func:`metacells.tools.knn_graph.compute_var_var_knn_graph`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
knn_k: Optional[int] = None

#: The factor of K of edges to keep when computing the balanced edge weights for the
#: K-Nearest-Neighbors graph. See
#: :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph`,
#: :py:func:`metacells.tools.knn_graph.compute_var_var_knn_graph`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
knn_balanced_ranks_factor: float = 4.0

#: The factor of K of edges to keep when pruning the incoming edges of the K-Nearest-Neighbors
#: graph. See
#: :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph`,
#: :py:func:`metacells.tools.knn_graph.compute_var_var_knn_graph`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
knn_incoming_degree_factor: float = 3.0

#: The factor of K of edges to keep when pruning the outgoing edges of the K-Nearest-Neighbors
#: graph. See
#: :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph`,
#: :py:func:`metacells.tools.knn_graph.compute_var_var_knn_graph`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
knn_outgoing_degree_factor: float = 1.0

#: The partition method to use for clustering the nodes of the K-Nearest-Neighbors graph. See
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
candidates_partition_method: 'utp.PartitionMethod' = utp.leiden_bounded_surprise

#: The target total cluster size for clustering the nodes of the K-Nearest-Neighbors graph. See
#: :py:const:`target_metacell_size`,
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
candidates_target_metacell_size: int = target_metacell_size

#: The size of each node for clustering the nodes of the K-Nearest-Neighbors graph. See
#: :py:const:`cell_sizes`
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
candidates_cell_sizes: Union[str, utt.Vector] = cell_sizes

#: The minimal size factor of clusters to split when clustering the nodes of the
#: K-Nearest-Neighbors graph. See
#: :py:const:`min_split_size_factor`,
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
candidates_min_split_size_factor: float = min_split_size_factor

#: The maximal size factor of clusters to merge when clustering the nodes of the
#: K-Nearest-Neighbors graph. See
#: :py:const:`max_merge_size_factor`,
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
candidates_max_merge_size_factor: float = max_merge_size_factor

#: The maximal number of times to recursively collect and attempt to group outliers when computing
#: the final metacells in the divide and conquer algorithm. See
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`.
max_outliers_levels: Optional[int] = 1

#: The minimal fold factor for a gene to indicate a cell is "deviant". See
#: :py:const:`significant_gene_fold_factor`,
#: :py:func:`metacells.tools.deviants.find_deviant_cells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
deviants_min_gene_fold_factor: float = significant_gene_fold_factor

#: The maximal fraction of genes to use to indicate cell are "deviants". See
#: :py:func:`metacells.tools.deviants.find_deviant_cells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
deviants_max_gene_fraction: float = 0.03

#: The maximal fraction of cells to mark as "deviants". See
#: :py:const:`final_deviants_max_cell_fraction`,
#: :py:func:`metacells.tools.deviants.find_deviant_cells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
deviants_max_cell_fraction: Optional[float] = 0.25

#: The maximal fraction of cells to mark as "deviants" when computing the final metacells in divide
#: and conquer algorithm. See
#: :py:const:`deviants_max_cell_fraction`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
final_deviants_max_cell_fraction: Optional[float] = None

#: The target total metacell size for dissolving too-small metacells. See
#: :py:const:`target_metacell_size`
#: :py:func:`metacells.tools.dissolve.dissolve_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
dissolve_target_metacell_size: int = target_metacell_size

#: The size of each cell for for dissolving too-small metacells. See
#: :py:const:`cell_sizes`,
#: :py:func:`metacells.tools.dissolve.dissolve_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
dissolve_cell_sizes: Union[str, utt.Vector] = cell_sizes

#: The minimal size factor for a metacell to be considered "robust". See
#: :py:const:`min_robust_size_factor`
#: :py:func:`metacells.tools.dissolve.dissolve_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
dissolve_min_robust_size_factor: float = min_robust_size_factor

#: The minimal size factor for a metacell to be kept if it is "convincing". See
#: :py:const:`max_merge_size_factor`
#: :py:func:`metacells.tools.dissolve.dissolve_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
dissolve_min_convincing_size_factor: float = max_merge_size_factor

#: The minimal fold factor of a gene in a metacell to make it "convincing". See
#: :py:const:`significant_gene_fold_factor`,
#: :py:func:`metacells.tools.dissolve.dissolve_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.direct.direct_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
dissolve_min_convincing_gene_fold_factor: float = significant_gene_fold_factor

#: The quantile of the cells total size to use for downsampling the cells for computing "excess"
#: R^2 for the genes. See
#: :py:const:`downsample_cell_quantile`
#: and
#: :py:func:`metacells.tools.excess.compute_excess_r2`.
excess_downsample_cell_quantile: float = downsample_cell_quantile

#: The minimal total value of a gene in a metacell to allow computing "excess" R^2 for it. See
#: :py:func:`metacells.tools.excess.compute_excess_r2`.
excess_min_gene_total: float = 10

#: The rank of the "top" gene-gene similarity to use for computing "excess" R^2 for each gene.
#: See :py:func:`metacells.tools.excess.compute_excess_r2`.
excess_top_gene_rank: int = 5

#: The number of times to shuffle the genes for averaging the baseline technical R^2 for each
#: gene. See :py:func:`metacells.tools.excess.compute_excess_r2`.
excess_shuffles_count: int = 10
