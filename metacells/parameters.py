'''
Defaults
--------
'''

from math import sqrt
from typing import Optional, Union
import metacells.utilities.typing as utt

#: The generic random seed. The default of ``0`` makes for a different result each time the code is
#: run. For reproducible results, provide a non-zero value, and also see
#: :py:func:`metacells.parameters.reproducible`. Used by too many functions to list here.
random_seed: int = 0

#: Whether to make the results reproducible, possibly at the cost of some slowdown. For reproducible
#: results, specify a ``True`` values, and also see :py:func:`metacells.parameters.random_seed`.
#: Used by too many functions to list here.
reproducible: bool = False

#: The generic minimal "significant" gene fraction. See
#: :py:func:`metacells.tools.high.find_high_fraction_genes`.
significant_gene_fraction: float = 1e-5

#: The generic minimal "significant" gene normalized variance. See
#: :py:func:`metacells.tools.high.find_high_normalized_variance_genes`.
significant_gene_normalized_variance: float = 2**2.5

#: The generic minimal "significant" gene relative variance. See
#: :py:func:`metacells.tools.high.find_high_relative_variance_genes`.
significant_gene_relative_variance: float = 0.1

#: The generic minimal "significant" gene similarity. See
#: :py:const:`noisy_lonely_max_gene_similarity`
#: and
#: :py:const:`rare_min_module_correlation`.
significant_gene_similarity: float = 0.1

#: The generic "significant" fold factor. See
#: :py:const:`deviants_min_gene_fold_factor`
#: and
#: :py:const:`dissolve_min_convincing_gene_fold_factor`.
significant_gene_fold_factor: float = 3.0

#: The generic minimal value (number of UMIs) we can say is "significant" given the technical noise.
#: See
#: py:const:`rare_min_gene_maximum`,
#: py:const:`rare_min_cell_module_total`,
#: and
#: py:const:`cells_similarity_value_normalization`.
significant_value: int = 7

#: The generic minimal samples to use for downsampling the cells for some
#: purpose. See
#: :py:const:`noisy_lonely_downsample_min_samples`,
#: :py:const:`feature_downsample_min_samples`,
#: and
#: :py:func:`metacells.tools.downsample.downsample_cells`.
downsample_min_samples: int = 750

#: The generic minimal quantile of the cells total size to use for downsampling the cells for some
#: purpose. See
#: :py:const:`noisy_lonely_downsample_min_cell_quantile`,
#: :py:const:`feature_downsample_min_cell_quantile`,
#: and
#: :py:func:`metacells.tools.downsample.downsample_cells`.
downsample_min_cell_quantile: float = 0.05

#: The generic maximal quantile of the cells total size to use for downsampling the cells for some
#: purpose. See
#: :py:const:`noisy_lonely_downsample_max_cell_quantile`,
#: :py:const:`feature_downsample_max_cell_quantile`,
#: and
#: :py:func:`metacells.tools.downsample.downsample_cells`.
downsample_max_cell_quantile: float = 0.5

#: The window size to use to compute relative variance. See
#: :py:func:`metacells.utilities.computation.relative_variance_per`
#: and
#: :py:func:`metacells.tools.high.find_high_relative_variance_genes`.
relative_variance_window_size: int = 100

#: The method to use to compute similarities. See
#: :py:func:`metacells.tools.similarity.compute_obs_obs_similarity`,
#: and
#: :py:func:`metacells.tools.similarity.compute_var_var_similarity`.
similarity_method: str = 'pearson'

#: The default location for the logistics function. See
#: :py:func:`metacells.pipeline.umap.compute_umap_by_features`,
#: :py:func:`metacells.tools.similarity.compute_obs_obs_similarity`,
#: :py:func:`metacells.tools.similarity.compute_var_var_similarity`.
#: and
#: :py:func:`metacells.utilities.computation.logistics`.
logistics_location: float = 0.8

#: The default scale for the logistics function. See
#: :py:func:`metacells.pipeline.umap.compute_umap_by_features`,
#: :py:func:`metacells.tools.similarity.compute_obs_obs_similarity`,
#: :py:func:`metacells.tools.similarity.compute_var_var_similarity`.
#: and
#: :py:func:`metacells.utilities.computation.logistics`.
logistics_scale: float = 0.5

#: The generic target number of observations (cells) in a pile, allowing us to
#: directly compute groups (metacells) for it. See
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
target_pile_size: int = 10000

#: The generic target total metacell size. See
#: :py:const:`candidates_target_metacell_size`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
target_metacell_size: float = 160000

#: The maximal cell size (total UMIs) to use. See
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`
#: and
#: py:func:`metacells.pipeline.collect.collect_metacells`.
max_cell_size: Optional[float] = None

#: The maximal cell size as a factor of the median cell size. See
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`
#: and
#: py:func:`metacells.pipeline.collect.collect_metacells`.
max_cell_size_factor: Optional[float] = 2.0

#: The genetic size of each cell for computing each metacell's size. See
#: :py:const:`candidates_cell_sizes`,
#: :py:const:`dissolve_cell_sizes`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`
#: and
#: :py:func:`metacells.pipeline.collect.collect_metacells`.
cell_sizes: Union[str, utt.Vector] = '__x__|sum'

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
#: :py:const:`rare_min_modules_size_factor`,
#: :py:const:`pile_max_merge_size_factor`,
#: :py:const:`candidates_max_merge_size_factor`,
#: and
#: :py:const:`dissolve_min_convincing_size_factor`.
max_merge_size_factor: float = 0.25

#: The minimal number of cells in a metacell, below which we would merge it. See
#: :py:const:`rare_min_cells_of_modules`,
#: :py:func:`candidates_min_metacell_cells`,
#: :py:func:`dissolve_min_metacell_cells`,
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`
#: and
#: :py:func:`metacells.tools.dissolve.dissolve_metacells`.
min_metacell_cells: int = 12

#: The minimal size factor of a pile, above which we can split it. See
#: :py:const:`min_split_size_factor`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
pile_min_split_size_factor: float = 1.25

#: The minimal pile size factor, below which we should consider dissolving it. See
#: :py:const:`min_robust_size_factor`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
pile_min_robust_size_factor: float = min_robust_size_factor

#: The maximal size factor of a pile, below which we should merge it. See
#: :py:const:`min_robust_size_factor`,
#: :py:const:`max_merge_size_factor`,
#: :py:const:`dissolve_min_convincing_size_factor`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
pile_max_merge_size_factor: float = max_merge_size_factor

#: The minimal total value for a cell to be considered "properly sampled". See
#: :py:func:`metacells.tools.properly_sampled.find_properly_sampled_cells`
#: and
#: :py:func:`metacells.pipeline.clean.extract_clean_data`.
#:
#: .. note::
#:
#:    There's no "reasonable" default value here. This must be tailored to the data.
properly_sampled_min_cell_total: Optional[int] = None

#: The maximal total value for a cell to be considered "properly sampled". See
#: :py:func:`metacells.tools.properly_sampled.find_properly_sampled_cells`
#: and
#: :py:func:`metacells.pipeline.clean.extract_clean_data`.
#:
#: .. note::
#:
#:    There's no "reasonable" default value here. This must be tailored to the data.
properly_sampled_max_cell_total: Optional[int] = None

#: The minimal total value for a gene to be considered "properly sampled". See
#: :py:func:`metacells.tools.properly_sampled.find_properly_sampled_genes`
#: and
#: :py:func:`metacells.pipeline.clean.extract_clean_data`.
properly_sampled_min_gene_total: int = 1

#: The maximal fraction of excluded gene UMIs from a cell for it to be considered
#: "properly_sampled". See
#: :py:func:`metacells.tools.properly_sampled.find_properly_sampled_cells`
#: and
#: :py:func:`metacells.pipeline.clean.extract_clean_data`.
#:
#: .. note::
#:
#:    There's no "reasonable" default value here. This must be tailored to the data.
properly_sampled_max_excluded_genes_fraction: Optional[float] = None

#: The number of randomly selected cells to use for computing related genes. See
#: :py:func:`metacells.pipeline.related_genes.relate_genes`.
related_max_sampled_cells: int = 10000

#: How to compute gene-gene similarity for computing the related genes. See
#: :py:func:`metacells.pipeline.related_genes.relate_genes`.
related_genes_similarity_method: str = 'repeated_pearson'

#: The hierarchical clustering method to use for computing the related genes. See
#: :py:func:`metacells.pipeline.related_genes.relate_genes`.
related_genes_cluster_method: str = 'ward'

#: The minimal number of genes in a related gene module. See
#: :py:func:`metacells.pipeline.related_genes.relate_genes`.
related_min_genes_of_modules: int = 16

#: The minimal samples to use for downsampling the cells for computing related genes. See
#: :py:const:`downsample_min_samples`,
#: :py:func:`metacells.tools.downsample.downsample_cells`,
#: :py:func:`metacells.pipeline.feature.extract_feature_data`,
#: and
#: :py:func:`metacells.pipeline.related_genes.relate_genes`.
related_downsample_min_samples: int = downsample_min_samples

#: The minimal quantile of the cells total size to use for downsampling the cells for computing
#: "feature" genes. See
#: :py:const:`downsample_min_cell_quantile`,
#: :py:func:`metacells.tools.downsample.downsample_cells`,
#: :py:func:`metacells.pipeline.feature.extract_feature_data`,
#: and
#: :py:func:`metacells.pipeline.related_genes.relate_genes`.
related_downsample_min_cell_quantile: float = downsample_min_cell_quantile

#: The maximal quantile of the cells total size to use for downsampling the cells for computing
#: "feature" genes. See
#: :py:const:`downsample_max_cell_quantile`,
#: :py:func:`metacells.tools.downsample.downsample_cells`,
#: :py:func:`metacells.pipeline.feature.extract_feature_data`,
#: and
#: :py:func:`metacells.pipeline.related_genes.relate_genes`.
related_downsample_max_cell_quantile: float = downsample_max_cell_quantile

#: The minimal relative variance of a gene to be considered a "feature". See
#: :py:func:`metacells.tools.high.find_high_relative_variance_genes`,
#: :py:func:`metacells.pipeline.feature.extract_feature_data`,
#: and
#: :py:func:`metacells.pipeline.related_genes.relate_genes`.
related_min_gene_relative_variance: float = 0.1

#: The minimal number of downsampled UMIs of a gene to be considered a "feature". See
#: :py:func:`metacells.tools.high.find_high_total_genes`,
#: :py:func:`metacells.pipeline.feature.extract_feature_data`,
#: and
#: :py:func:`metacells.pipeline.related_genes.relate_genes`.
related_min_gene_total: int = 50

#: The minimal number of the top-3rd downsampled UMIs of a gene to be considered a "feature". See
#: :py:func:`metacells.tools.high.find_high_topN_genes`,
#: :py:func:`metacells.pipeline.feature.extract_feature_data`,
#: and
#: :py:func:`metacells.pipeline.related_genes.relate_genes`.
related_min_gene_top3: int = 1

#: The number of randomly selected cells to use for computing "noisy lonely" genes. See
#: :py:func:`metacells.tools.noisy_lonely.find_noisy_lonely_genes`
#: and
#: :py:func:`metacells.pipeline.clean.extract_clean_data`.
noisy_lonely_max_sampled_cells: int = 10000

#: The minimal samples to use for downsampling the cells for computing "noisy lonely" genes. See
#: :py:const:`downsample_min_cell_quantile`,
#: :py:func:`metacells.tools.noisy_lonely.find_noisy_lonely_genes`,
#: and
#: :py:func:`metacells.pipeline.clean.extract_clean_data`.
noisy_lonely_downsample_min_samples: int = downsample_min_samples

#: The minimal quantile of the cells total size to use for downsampling the cells for computing
#: "noisy lonely" genes. See
#: :py:const:`downsample_min_cell_quantile`,
#: :py:func:`metacells.tools.noisy_lonely.find_noisy_lonely_genes`,
#: and
#: :py:func:`metacells.pipeline.clean.extract_clean_data`.
noisy_lonely_downsample_min_cell_quantile: float = downsample_min_cell_quantile

#: The maximal quantile of the cells total size to use for downsampling the cells for computing
#: "noisy lonely" genes. See
#: :py:const:`downsample_min_cell_quantile`,
#: :py:func:`metacells.tools.noisy_lonely.find_noisy_lonely_genes`,
#: and
#: :py:func:`metacells.pipeline.clean.extract_clean_data`.
noisy_lonely_downsample_max_cell_quantile: float = downsample_max_cell_quantile

#: The minimal total UMIs in the downsamples selected cells of a gene to be considered when
#: computing "lonely" genes. See
#: :py:func:`metacells.tools.noisy_lonely.find_noisy_lonely_genes` and
#: :py:func:`metacells.tools.high.find_high_total_genes`.
noisy_lonely_min_gene_total: int = 100

#: The minimal normalized variance of a gene to be considered "noisy". See
#: :py:const:`significant_gene_normalized_variance`,
#: :py:func:`metacells.tools.noisy_lonely.find_noisy_lonely_genes`
#: and
#: :py:func:`metacells.pipeline.clean.extract_clean_data`.
noisy_lonely_min_gene_normalized_variance: float = significant_gene_normalized_variance

#: The maximal similarity between a gene and another gene to be considered "lonely". See
#: :py:const:`significant_gene_similarity`,
#: :py:func:`metacells.tools.noisy_lonely.find_noisy_lonely_genes`
#: and
#: :py:func:`metacells.pipeline.clean.extract_clean_data`.
noisy_lonely_max_gene_similarity: float = significant_gene_similarity

#: The maximal fraction of the cells where a gene is expressed to be considered "rare". See
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_max_gene_cell_fraction: float = 1e-3

#: The minimal maximum-across-all-cells value of a gene to be considered as a candidate for rare
#: gene modules. See
#: :py:const:`significant_value`,
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_min_gene_maximum: int = significant_value

#: How to compute gene-gene similarity for computing the rare gene modules. See
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_genes_similarity_method: str = 'repeated_pearson'

#: The hierarchical clustering method to use for computing the rare gene modules. See
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_genes_cluster_method: str = 'ward'

#: The minimal number of genes in a rare gene module. See
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_min_genes_of_modules: int = 4

#: The minimal number of cells in a rare gene module. See
#: :py:const:`min_metacell_cells`,
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_min_cells_of_modules: int = min_metacell_cells

#: The maximal mean number of cells in a random pile for a rare gene module to be considered rare.
#: See
#: :py:const:`min_metacell_cells`,
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_max_cells_of_random_pile: int = min_metacell_cells * 4

#: The minimal total UMIs of all the cells in a rare gene module (as a fraction
#: of the :py:const:`target_metacell_size`). See
#: :py:const:`max_merge_size_factor`,
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_min_modules_size_factor: float = 0

#: The minimal average correlation between the genes in a rare gene module. See
#: :py:func:`metacells.parameters.significant_gene_similarity`,
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_min_module_correlation: float = significant_gene_similarity

#: The minimal fold factor between rare cells and the rest of the population for a gene to be
#: considered related to the rare gene module. See
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_min_related_gene_fold_factor: float = 7

#: The maximal ratio of total cells to include as a result of adding a related gene to a rare gene
#: module. See
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_max_related_gene_increase_factor: float = 4.0

#: The minimal number of UMIs of a rare gene module in a cell to be considered as expressing the
#: rare behavior. See
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_min_cell_module_total: int = int((significant_value + 1) / 2)

#: The minimal samples to use for downsampling the cells for computing "feature" genes. See
#: :py:const:`downsample_min_samples`,
#: :py:func:`metacells.tools.downsample.downsample_cells`,
#: :py:func:`metacells.pipeline.feature.extract_feature_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
feature_downsample_min_samples: int = downsample_min_samples

#: The minimal quantile of the cells total size to use for downsampling the cells for computing
#: "feature" genes. See
#: :py:const:`downsample_min_cell_quantile`,
#: :py:func:`metacells.tools.downsample.downsample_cells`,
#: :py:func:`metacells.pipeline.feature.extract_feature_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
feature_downsample_min_cell_quantile: float = downsample_min_cell_quantile

#: The maximal quantile of the cells total size to use for downsampling the cells for computing
#: "feature" genes. See
#: :py:const:`downsample_max_cell_quantile`,
#: :py:func:`metacells.tools.downsample.downsample_cells`,
#: :py:func:`metacells.pipeline.feature.extract_feature_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
feature_downsample_max_cell_quantile: float = downsample_max_cell_quantile

#: The minimal relative variance of a gene to be considered a "feature". See
#: :py:func:`metacells.tools.high.find_high_relative_variance_genes`,
#: :py:func:`metacells.pipeline.feature.extract_feature_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
feature_min_gene_relative_variance: float = 0.1

#: The minimal number of downsampled UMIs of a gene to be considered a "feature". See
#: :py:func:`metacells.tools.high.find_high_total_genes`,
#: :py:func:`metacells.pipeline.feature.extract_feature_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
feature_min_gene_total: int = 50

#: The minimal number of the top-3rd downsampled UMIs of a gene to be considered a "feature". See
#: :py:func:`metacells.tools.high.find_high_topN_genes`,
#: :py:func:`metacells.pipeline.feature.extract_feature_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
feature_min_gene_top3: int = 4

#: Whether to compute cell-cell similarity using the log (base 2) of the data. See
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
cells_similarity_log_data: bool = True

#: The normalization factor to use if/when computing the fractions of the data for directly
#: computing the metacells. See
#: :py:const:`significant_value`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
cells_similarity_value_normalization: float = 1/significant_value

#: The method to use to compute cell-cell similarity. See
#: :py:func:`metacells.tools.similarity.compute_obs_obs_similarity`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
cells_similarity_method: str = similarity_method

#: The method to use to compute group-group similarity. See
#: :py:const:`cells_similarity_method`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
groups_similarity_method: str = similarity_method

#: The target K for building the K-Nearest-Neighbors graph. See
#: :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph`,
#: :py:func:`metacells.tools.knn_graph.compute_var_var_knn_graph`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
knn_k: Optional[int] = None

#: The minimal target K for building the K-Nearest-Neighbors graph. See
#: :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph`,
#: :py:func:`metacells.tools.knn_graph.compute_var_var_knn_graph`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
min_knn_k: Optional[int] = 30

#: The factor of K edge ranks to keep when computing the balanced ranks for the
#: K-Nearest-Neighbors graph. See
#: :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph`,
#: :py:func:`metacells.tools.knn_graph.compute_var_var_knn_graph`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_features`.
knn_balanced_ranks_factor: float = sqrt(10)

#: The factor of K of edges to keep when pruning the incoming edges of the K-Nearest-Neighbors
#: graph. See
#: :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph`,
#: :py:func:`metacells.tools.knn_graph.compute_var_var_knn_graph`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_features`.
knn_incoming_degree_factor: float = 3.0

#: The factor of K of edges to keep when pruning the outgoing edges of the K-Nearest-Neighbors
#: graph. See
#: :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph`,
#: :py:func:`metacells.tools.knn_graph.compute_var_var_knn_graph`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_features`.
knn_outgoing_degree_factor: float = 1.0

#: The minimal quantile of a seed to be selected. See
#: :py:func:`metacells.tools.candidates.choose_seeds`,
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
min_seed_size_quantile: float = 0.85

#: The maximal quantile of a seed to be selected. See
#: :py:func:`metacells.tools.candidates.choose_seeds`,
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
max_seed_size_quantile: float = 0.95

#: By how much (as a fraction) to cooldown the temperature after doing a pass on all the nodes. See
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells` and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
cooldown_pass: float = 0.02

#: By how much (as a fraction) to cooldown the node temperature after improving it. See
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells` and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
cooldown_node: float = 0.25

#: By how much (as a fraction) to reduce the cooldown each time we re-optimize a slightly modified
#: partition. See
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells` and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
cooldown_phase: float = 0.75

#: The target total cluster size for clustering the nodes of the K-Nearest-Neighbors graph. See
#: :py:const:`target_metacell_size`,
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
candidates_target_metacell_size: float = target_metacell_size

#: The size of each node for clustering the nodes of the K-Nearest-Neighbors graph. See
#: :py:const:`cell_sizes`
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
candidates_cell_sizes: Union[str, utt.Vector] = cell_sizes

#: The minimal size factor of clusters to split when clustering the nodes of the
#: K-Nearest-Neighbors graph. See
#: :py:const:`min_split_size_factor`,
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
candidates_min_split_size_factor: float = min_split_size_factor

#: The maximal size factor of clusters to merge when clustering the nodes of the
#: K-Nearest-Neighbors graph. See
#: :py:const:`max_merge_size_factor`,
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
candidates_max_merge_size_factor: float = max_merge_size_factor

#: The minimal number of cells in a metacell, below which we would merge it. See
#: :py:const:`min_metacell_cells`
#: and
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`.
candidates_min_metacell_cells: int = min_metacell_cells

#: The maximal number of times to recursively collect and attempt to group outliers when computing
#: the final metacells in the divide and conquer algorithm. See
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`.
final_max_outliers_levels: Optional[int] = 1

#: The minimal fold factor for a gene to indicate a cell is "deviant". See
#: :py:const:`significant_gene_fold_factor`,
#: :py:func:`metacells.tools.deviants.find_deviant_cells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
deviants_min_gene_fold_factor: float = significant_gene_fold_factor

#: The maximal fraction of genes to use to indicate cell are "deviants". See
#: :py:func:`metacells.tools.deviants.find_deviant_cells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
deviants_max_gene_fraction: float = 0.03

#: The maximal fraction of cells to mark as "deviants". See
#: :py:func:`metacells.tools.deviants.find_deviant_cells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
deviants_max_cell_fraction: Optional[float] = 0.25

#: The size of each cell for for dissolving too-small metacells. See
#: :py:const:`cell_sizes`,
#: :py:func:`metacells.tools.dissolve.dissolve_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
dissolve_cell_sizes: Union[str, utt.Vector] = cell_sizes

#: The minimal size factor for a metacell to be considered "robust". See
#: :py:const:`min_robust_size_factor`
#: :py:func:`metacells.tools.dissolve.dissolve_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
dissolve_min_robust_size_factor: float = min_robust_size_factor

#: The minimal number of cells in a metacell, below which we would dissolve it. See
#: :py:const:`min_metacell_cells`
#: and
#: :py:func:`metacells.tools.dissolve.dissolve_metacells`.
dissolve_min_metacell_cells: int = min_metacell_cells

#: The minimal size factor for a metacell to be considered "robust" when grouping rare gene module
#: cells. See
#: :py:const:`min_robust_size_factor`,
#: :py:const:`max_merge_size_factor`
#: :py:func:`metacells.tools.dissolve.dissolve_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_dissolve_min_robust_size_factor: float = max_merge_size_factor

#: The minimal size factor for a metacell to be kept if it is "convincing". See
#: :py:const:`max_merge_size_factor`
#: :py:func:`metacells.tools.dissolve.dissolve_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
dissolve_min_convincing_size_factor: Optional[float] = max_merge_size_factor

#: The minimal size factor for a metacell to be kept if it is "convincing" when grouping rare gene
#: module cells. See
#: :py:const:`max_merge_size_factor`
#: :py:func:`metacells.tools.dissolve.dissolve_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_dissolve_min_convincing_size_factor: Optional[float] = None

#: The minimal fold factor of a gene in a metacell to make it "convincing". See
#: :py:const:`significant_gene_fold_factor`,
#: :py:func:`metacells.tools.dissolve.dissolve_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
dissolve_min_convincing_gene_fold_factor: float = significant_gene_fold_factor

#: The number of most-distinct genes to collect for each cell. See
#: :py:func:`metacells.tools.distinct.find_distinct_genes`.
distinct_genes_count: int = 20

#: The normalization factor to use if/when computing the fractions of the data for UMAP.
#: See
#: :py:const:`metacells.parameters.target_metacell_size`
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_features`.
umap_similarity_value_normalization: float = 1/target_metacell_size

#: Whether to compute metacell-metacell similarity using the log (base 2) of the data for UMAP. See
#: :py:func:`metacells.pipeline.umap.compute_umap_by_features`.
umap_similarity_log_data: bool = True

#: The method to use to compute similarities for UMAP. See
#: :py:func:`metacells.tools.similarity.compute_obs_obs_similarity`,
#: :py:func:`metacells.tools.similarity.compute_var_var_similarity`,
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_features`.
umap_similarity_method: str = 'logistics_pearson'

#: The minimal UMAP point distance. See :py:const:`umap_spread` and
#: :py:func:`metacells.tools.layout.umap_by_distances`
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_features`.
umap_min_dist: float = 0.5

#: The minimal UMAP spread. This is automatically raised if the :py:const:`umap_min_dist` is higher.
#: See :py:func:`metacells.tools.layout.umap_by_distances`
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_features`.
umap_spread: float = 1.0

#: The UMAP KNN graph degree. See
#: :py:func:`metacells.tools.layout.umap_by_distances`
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_features`.
umap_k: int = 15

#: The UMAP KNN skeleton graph degree. See
#: :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph`,
#: :py:func:`metacells.tools.knn_graph.compute_var_var_knn_graph`,
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_features`.
skeleton_k: int = 4

#: The maximal number of top feature genes to pick.
#: See :py:func:`metacells.tools.high.find_top_feature_genes`
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_features`.
max_top_feature_genes: int = 1000

#: The value to add to gene fractions before applying the log function. See
#: See :py:func:`metacells.pipeline.umap.compute_umap_by_features`.
umap_fraction_normalization: float = 1e-5

#: The fraction of the UMAP plot area to cover with points. See
#: :py:func:`metacells.utilities.computation.cover_diameter`,
#: :py:func:`metacells.utilities.computation.cover_coordinates`
#: and
#: :py:func:`metacells.tools.layout.umap_by_distances`,
cover_fraction: float = 1/3.0

#: The noise to add to the UMAP plot area. See
#: :py:func:`metacells.utilities.computation.cover_coordinates`
#: and
#: :py:func:`metacells.tools.layout.umap_by_distances`,
noise_fraction: float = 0.1

#: The minimal total number of UMIs for a gene to compute meaningful quality statistics for it.
#: See :py:func:`metacells.tools.quality.compute_inner_normalized_variance`.
quality_min_gene_total: int = 40

#: The maximal amount of memory to use when guessing the number of parallel piles. If zero or
#: negative, is the fraction of the machine's total RAM to use as a safety buffer. See
#: :py:func:`metacells.pipeline.divide_and_conquer.guess_max_parallel_piles`.
max_gbs: float = -0.1
