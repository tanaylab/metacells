"""
Defaults
--------
"""

from math import sqrt
from typing import Optional

import metacells.utilities.typing as utt

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
#: :py:const:`bursty_lonely_max_gene_similarity`
#: and
#: :py:const:`rare_min_module_correlation`.
significant_gene_similarity: float = 0.1

#: The generic "significant" fold factor. See
#: :py:const:`deviants_min_gene_fold_factor`
#: :py:const:`rare_min_convincing_gene_fold_factor`,
#: and
#: :py:const:`dissolve_min_convincing_gene_fold_factor`.
significant_gene_fold_factor: float = 3.0

#: The generic additional "significant" fold factor for noisy genes, in addition to
#: :py:const:`significant_gene_fold_factor`. See
#: :py:const:`deviants_min_gene_fold_factor`
#: and
#: :py:const:`rare_min_convincing_gene_fold_factor`,
#: and
#: :py:const:`dissolve_min_convincing_gene_fold_factor`.
significant_noisy_gene_fold_factor: float = 2.0

#: The generic minimal value (number of UMIs) we can say is "significant" given the technical noise. See
#: :py:const:`rare_min_gene_maximum`,
#: :py:const:`rare_min_cell_module_total`,
#: and
#: :py:const:`cells_similarity_value_regularization`.
significant_value: int = 7

#: The generic minimal samples to use for downsampling the cells for some purpose. See
#: :py:const:`bursty_lonely_downsample_min_samples`,
#: :py:const:`select_downsample_min_samples`,
#: and
#: :py:func:`metacells.tools.downsample.downsample_cells`.
downsample_min_samples: int = 750

#: The generic minimal quantile of the cells total size to use for downsampling the cells for some
#: purpose. See
#: :py:const:`bursty_lonely_downsample_min_cell_quantile`,
#: :py:const:`select_downsample_min_cell_quantile`,
#: and
#: :py:func:`metacells.tools.downsample.downsample_cells`.
downsample_min_cell_quantile: float = 0.05

#: The generic maximal quantile of the cells total size to use for downsampling the cells for some
#: purpose. See
#: :py:const:`bursty_lonely_downsample_max_cell_quantile`,
#: :py:const:`select_downsample_max_cell_quantile`,
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
similarity_method: str = "abs_pearson"

#: The default location for the logistics function. See
#: :py:func:`metacells.pipeline.umap.compute_umap_by_markers`,
#: :py:func:`metacells.tools.similarity.compute_obs_obs_similarity`,
#: :py:func:`metacells.tools.similarity.compute_var_var_similarity`.
#: and
#: :py:func:`metacells.utilities.computation.logistics`.
logistics_location: float = 0.8

#: The default slope for the logistics function. See
#: :py:func:`metacells.pipeline.umap.compute_umap_by_markers`,
#: :py:func:`metacells.tools.similarity.compute_obs_obs_similarity`,
#: :py:func:`metacells.tools.similarity.compute_var_var_similarity`.
#: and
#: :py:func:`metacells.utilities.computation.logistics`.
logistics_slope: float = 0.5

#: The minimal target number of observations (cells) in a pile, allowing us to
#: directly compute groups (metacells) for it. See
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
min_target_pile_size: int = 8000

#: The maximal target number of observations (cells) in a pile, allowing us to
#: directly compute groups (metacells) for it. See
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
max_target_pile_size: int = 24000

#: The target number of metacells computed in each pile. See
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
target_metacells_in_pile: int = 100

#: The generic target total metacell size (in cells). See
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
target_metacell_size: int = 48

#: The generic target total metacell total UMIs. See
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
target_metacell_umis: int = 160000

#: Whether to use geometrical mean to compute metacell gene fractions.
#: See :py:func:`metacells.pipeline.collect.collect_metacells`.
metacell_geo_mean: bool = True

#: The number of UMIs to use for regularization when computing metacell gene fractions geometrical mean.
#: See :py:func:`metacells.pipeline.collect.collect_metacells`.
metacell_umis_regularization: float = 1 / 16.0

#: The quantile of the cell sizes to use for computing the number of zero-valued cells.
#: See
#: :py:func:`metacells.pipeline.collect.collect_metacells`.
zeros_cell_size_quantile: float = 0.1

#: The number of UMIs in each cell for computing each metacell's total UMIs. See
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`
#: and
#: :py:func:`metacells.pipeline.collect.collect_metacells`.
cell_umis: Optional[utt.NumpyVector] = None

#: The generic maximal group size factor, above which we should split it. See
#: :py:const:`piles_min_split_size_factor`
#: and
#: :py:const:`candidates_min_split_size_factor`.
min_split_size_factor: float = 2.0

#: The generic minimal group size factor, below which we should consider dissolving it. See
#: :py:const:`piles_min_robust_size_factor`
#: and
#: :py:const:`dissolve_min_robust_size_factor`.
min_robust_size_factor: float = 0.5

#: The generic maximal group size factor, below which we should merge it. See
#: :py:const:`piles_max_merge_size_factor`,
#: :py:const:`candidates_max_merge_size_factor`.
max_merge_size_factor: float = 0.5

#: The minimal number of cells in a metacell, below which we would merge it. See
#: :py:const:`rare_min_cells_of_modules`,
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`
#: and
#: :py:func:`metacells.tools.dissolve.dissolve_metacells`.
min_metacell_size: int = 12

#: The maximal strength of a min-cut of a metacell that will cause it to be split. See
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`.
max_split_min_cut_strength: float = 0.1

#: The minimal number of cells to keep as a seed when cutting a metacell. See
#: :py:func:`metacells.tools.candidates.compute_candidate_metacells`.
min_cut_seed_cells: int = 7

#: The minimal size factor of a pile, above which we can split it. See
#: :py:const:`min_split_size_factor`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
piles_min_split_size_factor: float = 1.5

#: The minimal pile size factor, below which we should consider dissolving it. See
#: :py:const:`min_robust_size_factor`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
piles_min_robust_size_factor: float = 0.3

#: The maximal size factor of a pile, below which we should merge it. See
#: :py:const:`min_robust_size_factor`,
#: :py:const:`max_merge_size_factor`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
piles_max_merge_size_factor: float = 0.15

#: The minimal total value for a cell to be considered "properly sampled". See
#: :py:func:`metacells.tools.properly_sampled.find_properly_sampled_cells`
#: and
#: :py:func:`metacells.pipeline.exclude.extract_clean_data`.
#:
#: .. note::
#:
#:    There's no "reasonable" default value here. This must be tailored to the data.
properly_sampled_min_cell_total: Optional[int] = None

#: The maximal total value for a cell to be considered "properly sampled". See
#: :py:func:`metacells.tools.properly_sampled.find_properly_sampled_cells`
#: and
#: :py:func:`metacells.pipeline.exclude.extract_clean_data`.
#:
#: .. note::
#:
#:    There's no "reasonable" default value here. This must be tailored to the data.
properly_sampled_max_cell_total: Optional[int] = None

#: The minimal total value for a gene to be considered "properly sampled". See
#: :py:func:`metacells.tools.properly_sampled.find_properly_sampled_genes`
#: and
#: :py:func:`metacells.pipeline.exclude.extract_clean_data`.
properly_sampled_min_gene_total: int = 1

#: The maximal fraction of excluded gene UMIs from a cell for it to be considered
#: "properly_sampled". See
#: :py:func:`metacells.tools.properly_sampled.find_properly_sampled_cells`
#: and
#: :py:func:`metacells.pipeline.exclude.extract_clean_data`.
#:
#: .. note::
#:
#:    There's no "reasonable" default value here. This must be tailored to the data.
properly_sampled_max_excluded_genes_fraction: Optional[float] = None

#: The number of randomly selected cells to use for computing related genes. See
#: :py:func:`metacells.pipeline.related_genes.relate_to_lateral_genes`.
related_max_sampled_cells: int = 10000

#: How to compute gene-gene similarity for computing the related genes. See
#: :py:func:`metacells.pipeline.related_genes.relate_to_lateral_genes`.
related_genes_similarity_method: str = "abs_pearson"

#: The hierarchical clustering method to use for computing the related genes. See
#: :py:func:`metacells.pipeline.related_genes.relate_to_lateral_genes`.
related_genes_cluster_method: str = "ward"

#: The maximal number of genes in a related gene module. See
#: :py:func:`metacells.pipeline.related_genes.relate_to_lateral_genes`.
related_max_genes_of_modules: int = 64

#: The minimal mean fraction of a related gene. See
#: :py:func:`metacells.pipeline.related_genes.relate_to_lateral_genes`.
related_min_mean_gene_fraction: float = 1e-5

#: The minimal correlation of a related gene to the base gene(s). See
#: :py:func:`metacells.pipeline.related_genes.relate_to_lateral_genes`.
related_min_gene_correlation: float = 0.1

#: The number of randomly selected cells to use for computing "bursty lonely" genes. See
#: :py:func:`metacells.tools.bursty_lonely.find_bursty_lonely_genes`
#: and
#: :py:func:`metacells.pipeline.exclude.extract_clean_data`.
bursty_lonely_max_sampled_cells: int = 10000

#: The minimal samples to use for downsampling the cells for computing "bursty lonely" genes. See
#: :py:const:`downsample_min_cell_quantile`,
#: :py:func:`metacells.tools.bursty_lonely.find_bursty_lonely_genes`,
#: and
#: :py:func:`metacells.pipeline.exclude.extract_clean_data`.
bursty_lonely_downsample_min_samples: int = downsample_min_samples

#: The minimal quantile of the cells total size to use for downsampling the cells for computing
#: "bursty lonely" genes. See
#: :py:const:`downsample_min_cell_quantile`,
#: :py:func:`metacells.tools.bursty_lonely.find_bursty_lonely_genes`,
#: and
#: :py:func:`metacells.pipeline.exclude.extract_clean_data`.
bursty_lonely_downsample_min_cell_quantile: float = downsample_min_cell_quantile

#: The maximal quantile of the cells total size to use for downsampling the cells for computing
#: "bursty lonely" genes. See
#: :py:const:`downsample_min_cell_quantile`,
#: :py:func:`metacells.tools.bursty_lonely.find_bursty_lonely_genes`,
#: and
#: :py:func:`metacells.pipeline.exclude.extract_clean_data`.
bursty_lonely_downsample_max_cell_quantile: float = downsample_max_cell_quantile

#: The minimal total UMIs in the downsamples selected cells of a gene to be considered when
#: computing "lonely" genes. See
#: :py:func:`metacells.tools.bursty_lonely.find_bursty_lonely_genes` and
#: :py:func:`metacells.tools.high.find_high_total_genes`.
bursty_lonely_min_gene_total: int = 100

#: The minimal normalized variance of a gene to be considered "bursty". See
#: :py:const:`significant_gene_normalized_variance`,
#: :py:func:`metacells.tools.bursty_lonely.find_bursty_lonely_genes`
#: and
#: :py:func:`metacells.pipeline.exclude.extract_clean_data`.
bursty_lonely_min_gene_normalized_variance: float = significant_gene_normalized_variance

#: The maximal similarity between a gene and another gene to be considered "lonely". See
#: :py:const:`significant_gene_similarity`,
#: :py:func:`metacells.tools.bursty_lonely.find_bursty_lonely_genes`
#: and
#: :py:func:`metacells.pipeline.exclude.extract_clean_data`.
bursty_lonely_max_gene_similarity: float = significant_gene_similarity

#: The maximal number of candidate rare genes. See
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_max_genes: int = 500

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
rare_genes_similarity_method: str = "repeated_pearson"

#: The hierarchical clustering method to use for computing the rare gene modules. See
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_genes_cluster_method: str = "ward"

#: The minimal number of genes in a rare gene module. See
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_min_genes_of_modules: int = 4

#: The minimal number of cells in a rare gene module. See
#: :py:const:`min_metacell_size`,
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_min_cells_of_modules: int = min_metacell_size

#: The maximal mean number of cells (as a fraction of the target metacell size) in a random pile for a rare gene module
#: to be considered rare. See
#: :py:const:`min_metacell_size`,
#: :py:func:`metacells.tools.rare.find_rare_gene_modules`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_max_cells_factor_of_random_pile: float = 0.5

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

#: The maximal fraction of cells to mark as "deviants" in rare gene module piles. See
#: :py:func:`metacells.tools.deviants.find_deviant_cells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_deviants_max_cell_fraction: Optional[float] = 0.25

#: Whether to compute metacells more quickly with lower quality. See
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
quick_and_dirty: bool = False

#: The minimal samples to use for downsampling the cells for computing "select" genes. See
#: :py:const:`downsample_min_samples`,
#: :py:func:`metacells.tools.downsample.downsample_cells`,
#: :py:func:`metacells.pipeline.select.extract_selected_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
select_downsample_min_samples: int = downsample_min_samples

#: The minimal quantile of the cells total size to use for downsampling the cells for computing
#: "select" genes. See
#: :py:const:`downsample_min_cell_quantile`,
#: :py:func:`metacells.tools.downsample.downsample_cells`,
#: :py:func:`metacells.pipeline.select.extract_selected_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
select_downsample_min_cell_quantile: float = downsample_min_cell_quantile

#: The maximal quantile of the cells total size to use for downsampling the cells for computing
#: "select" genes. See
#: :py:const:`downsample_max_cell_quantile`,
#: :py:func:`metacells.tools.downsample.downsample_cells`,
#: :py:func:`metacells.pipeline.select.extract_selected_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
select_downsample_max_cell_quantile: float = downsample_max_cell_quantile

#: The minimal relative variance of a gene to be considered a "select". See
#: :py:func:`metacells.tools.high.find_high_relative_variance_genes`,
#: :py:func:`metacells.pipeline.select.extract_selected_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
select_min_gene_relative_variance: Optional[float] = 0.1

#: The minimal number of downsampled UMIs of a gene to be "select". See
#: :py:func:`metacells.tools.high.find_high_total_genes`,
#: :py:func:`metacells.pipeline.select.extract_selected_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
select_min_gene_total: Optional[int] = 50

#: The minimal number of the top-3rd downsampled UMIs of a gene to be "select". See
#: :py:func:`metacells.tools.high.find_high_topN_genes`,
#: :py:func:`metacells.pipeline.select.extract_selected_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
select_min_gene_top3: Optional[int] = 4

#: The minimal number of "select" genes. See
#: :py:func:`metacells.pipeline.select.extract_selected_data`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
select_min_genes: int = 100

#: Whether to compute cell-cell similarity using the log (base 2) of the data. See
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
cells_similarity_log_data: bool = True

#: The regularization factor to use if/when computing the fractions of the data for directly
#: computing the metacells. See
#: :py:const:`significant_value`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
cells_similarity_value_regularization: float = 1 / significant_value

#: The method to use to compute cell-cell similarity. See
#: :py:func:`metacells.tools.similarity.compute_obs_obs_similarity`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
cells_similarity_method: str = similarity_method

#: Whether to compute group-group similarity using the log (base 2) of the data. See
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
groups_similarity_log_data: bool = True

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

#: The size of the default K for building the K-Nearest-Neighbors graph when computing metacells, multiplied
#: by the median number of cells needed to reach the :py:const:`target_metacell_size`. See
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`,
#: and
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`.
candidates_knn_k_size_factor: int = 2

#: The size of the default K for building the K-Nearest-Neighbors graph when computing groups of metacells, multiplied
#: by the median number of cells needed to reach the :py:const:`target_metacell_size`. See
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`,
#: and
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`.
piles_knn_k_size_factor: int = 3

#: The size of the default K for building the K-Nearest-Neighbors graph, based
#: on the number of cells needed to reach the :py:const:`target_metacell_umis` for a quantile of the cells. See
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`,
#: and
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`.
knn_k_umis_quantile: float = 0.1

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
#: :py:func:`metacells.pipeline.umap.compute_umap_by_markers`.
knn_balanced_ranks_factor: float = sqrt(10)

#: The factor of K of edges to keep when pruning the incoming edges of the K-Nearest-Neighbors
#: graph. See
#: :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph`,
#: :py:func:`metacells.tools.knn_graph.compute_var_var_knn_graph`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_markers`.
knn_incoming_degree_factor: float = 3.0

#: The factor of K of edges to keep when pruning the outgoing edges of the K-Nearest-Neighbors
#: graph. See
#: :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph`,
#: :py:func:`metacells.tools.knn_graph.compute_var_var_knn_graph`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_markers`.
knn_outgoing_degree_factor: float = 1.0

#: The minimal outgoing degree of nodes in the KNN graph. See
#: :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph`,
#: :py:func:`metacells.tools.knn_graph.compute_var_var_knn_graph`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`
#: :py:func:`metacells.pipeline.umap.compute_knn_by_markers`
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_markers`.
knn_min_outgoing_degree: int = 2

#: The minimal outgoing degree of nodes in the markers KNN graph. See
#: :py:func:`metacells.pipeline.umap.compute_knn_by_markers` and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_markers`.
markers_knn_min_outgoing_degree: int = 1

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

#: The policy to use for deciding which cell is "deviant". See
#: :py:func:`metacells.tools.deviants.find_deviant_cells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
deviants_policy: str = "gaps"

#: If using the ``gaps`` deviants policy, how many cells to skip ahead when computing gap sizes
#: (``0``, ``1``, ``2``). See
#: :py:func:`metacells.tools.deviants.find_deviant_cells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
deviants_gap_skip_cells: int = 1

#: Do not mark deviants by a gene in a metacell if it causes more than this number of cells to become deviant
#: (unless the count is no more than :py:const:`deviants_max_gap_cells_fraction`). See
#: :py:func:`metacells.tools.deviants.find_deviant_cells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
deviants_max_gap_cells_count: int = 3

#: Do not mark deviants by a gene in a metacell if it causes more than this fraction of cells to become deviant
#: (unless the count is no more than :py:const:`deviants_max_gap_cells_count`). See
#: :py:func:`metacells.tools.deviants.find_deviant_cells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
deviants_max_gap_cells_fraction: float = 0.1

#: Use this quantile to pick the minimal cell size for the regularization factor when
# computing deviants. See
#: :py:func:`metacells.tools.deviants.find_deviant_cells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
deviant_cells_regularization_quantile: float = 0.25

#: The minimal fold factor for a gene to indicate a cell is "deviant". See
#: :py:const:`significant_gene_fold_factor`,
#: :py:func:`metacells.tools.deviants.find_deviant_cells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
deviants_min_gene_fold_factor: float = significant_gene_fold_factor

#: The minimal number of UMIs in the gene for two cells for using it as a certificate for a gap between
#: the expression level of the gene in the cells for computinng deviants. See
#: :py:const:`significant_gene_fold_factor`,
#: :py:func:`metacells.tools.deviants.find_deviant_cells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
deviants_min_compare_umis: int = 8

#: The minimal additional fold factor for a noisy gene to indicate a cell is "deviant",
# in addition to :py:const:`deviants_min_gene_fold_factor`. See
#: :py:const:`significant_gene_fold_factor`,
#: :py:func:`metacells.tools.deviants.find_deviant_cells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
deviants_min_noisy_gene_fold_factor: float = significant_noisy_gene_fold_factor

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

#: The minimal size factor for a metacell to be considered "robust". See
#: :py:const:`min_robust_size_factor`
#: :py:func:`metacells.tools.dissolve.dissolve_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
dissolve_min_robust_size_factor: float = min_robust_size_factor

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

#: The minimal fold factor of a gene in a rare metacell to make it "convincing". See
#: module cells. See
#: :py:const:`max_merge_size_factor`
#: :py:func:`metacells.tools.dissolve.dissolve_metacells`,
#: :py:func:`metacells.pipeline.direct.compute_direct_metacells`,
#: :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`
#: and
#: :py:func:`metacells.pipeline.divide_and_conquer.divide_and_conquer_pipeline`.
rare_min_convincing_gene_fold_factor: Optional[float] = None

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

#: The maximal number of marker genes to use for UMAP.
#: See
#: :py:func:`metacells.pipeline.umap.compute_knn_by_markers` and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_markers`.
umap_max_marker_genes: int = 1000

#: Whether to ignore lateral genes when computing the UMAP.
#: See
#: :py:func:`metacells.pipeline.umap.compute_knn_by_markers` and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_markers`.
umap_ignore_lateral_genes: bool = True

#: The regularization factor to use if/when computing the fractions of the data for UMAP.
#: See
#: :py:const:`metacells.parameters.significant_gene_fraction`
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_markers`.
umap_similarity_value_regularization: float = significant_gene_fraction

#: Whether to compute metacell-metacell similarity using the log (base 2) of the data for UMAP. See
#: :py:func:`metacells.pipeline.umap.compute_umap_by_markers`.
umap_similarity_log_data: bool = True

#: The method to use to compute similarities for UMAP. See
#: :py:func:`metacells.tools.similarity.compute_obs_obs_similarity`,
#: :py:func:`metacells.tools.similarity.compute_var_var_similarity`,
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_markers`.
umap_similarity_method: str = "logistics_abs_pearson"

#: The minimal UMAP point distance. See :py:const:`umap_spread` and
#: :py:func:`metacells.tools.layout.umap_by_distances`
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_markers`.
umap_min_dist: float = 0.5

#: The minimal UMAP spread. This is automatically raised if the :py:const:`umap_min_dist` is higher.
#: See :py:func:`metacells.tools.layout.umap_by_distances`
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_markers`.
umap_spread: float = 1.0

#: The UMAP KNN graph degree. See
#: :py:func:`metacells.tools.layout.umap_by_distances`
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_markers`.
umap_k: int = 15

#: The UMAP KNN skeleton graph degree. See
#: :py:func:`metacells.tools.knn_graph.compute_obs_obs_knn_graph`,
#: :py:func:`metacells.tools.knn_graph.compute_var_var_knn_graph`,
#: and
#: :py:func:`metacells.pipeline.umap.compute_umap_by_markers`.
skeleton_k: int = 4

#: The value to add to gene fractions before applying the log function. See
#: See :py:func:`metacells.pipeline.umap.compute_umap_by_markers`.
umap_fraction_regularization: float = 1e-5

#: The fraction of the UMAP plot area to cover with points. See
#: :py:func:`metacells.utilities.computation.cover_diameter`,
#: :py:func:`metacells.utilities.computation.cover_coordinates`
#: and
#: :py:func:`metacells.tools.layout.umap_by_distances`,
spread_cover_fraction: float = 1 / 3.0

#: The noise to add to the UMAP plot area. See
#: :py:func:`metacells.utilities.computation.cover_coordinates`
#: and
#: :py:func:`metacells.tools.layout.umap_by_distances`,
spread_noise_fraction: float = 0.1

#: The minimal total number of UMIs for a gene to compute meaningful quality statistics for it.
#: See :py:func:`metacells.tools.quality.compute_stdev_logs`,
#: :py:func:`metacells.tools.quality.compute_inner_folds`, and
#: :py:func:`metacells.tools.quality.compute_outliers_fold_factors`.
quality_min_gene_total: int = 40

#: The maximal amount of memory to use when guessing the number of parallel piles. If zero or
#: negative, is the fraction of the machine's total RAM to use as a safety buffer. See
#: :py:func:`metacells.pipeline.divide_and_conquer.guess_max_parallel_piles`.
max_gbs: float = -0.1

#: Whether the projection will ignore genes where the ranges of the corrected vs. the projected expression is too low.
#: See :py:func:`metacells.pipeline.projection.projection_pipeline`.
project_filter_ranges: bool = True

#: The quantile to use (on both low and high ends) to compute the range of expression of a corrected of a projected
#: gene. See :py:func:`metacells.pipeline.projection.projection_pipeline`.
project_ignore_range_quantile: float = 0.02

#: The minimal overlap (shared range divided by query range) for genes to keep projecting. See
#: :py:func:`metacells.pipeline.projection.projection_pipeline`.
project_ignore_range_min_overlap_fraction: float = 0.5

#: The minimal fraction of the query marker genes that are fitted for a query metacell to be considered "similar" to the
#: atlas. See :py:func:`metacells.pipeline.projection.projection_pipeline`.
project_min_query_markers_fraction: float = 1 / 3

#: The regularization factor to use when computing fold factors for projecting a query onto an atlas. See
#: :py:func:`metacells.tools.project.compute_projection_weights`.
project_fold_regularization: float = 1e-5

#: The minimal number of UMIs for a gene to be a potential cause to mark a metacell as dissimilar. See
#: :py:func:`metacells.tools.project.compute_projection_weights`.
project_min_significant_gene_umis: int = 40

#: The number of atlas candidates to consider when projecting a query onto an atlas. See
#: :py:func:`metacells.tools.project.compute_projection_weights`.
project_candidates_count: int = 50

#: The minimal number of atlas candidates to use even if they fail the consistency check as a fraction of
#: :py:const:`project_candidates_count`. See
#: :py:func:`metacells.tools.project.compute_projection_weights`.
project_min_candidates_fraction: float = 1.0 / 3.0

#: The minimal weight of an atlas metacell used for the projection of a query metacell. See
#: :py:func:`metacells.tools.project.compute_projection_weights`.
project_min_usage_weight: float = 1e-5

#: The maximal fold factor of genes between the projection and the query metacell. See
#: :py:func:`metacells.tools.project.compute_projection_weights`.
project_max_projection_fold_factor: float = significant_gene_fold_factor

#: The maximal additional fold factor of noisy genes between the projection and the query metacell,
# in addition to :py:const:`project_max_projection_fold_factor`. See
#: :py:func:`metacells.tools.project.compute_projection_weights`.
project_max_projection_noisy_fold_factor: float = significant_noisy_gene_fold_factor

#: The maximal fold factor of genes between the atlas metacells used for the projection of a query metacell. See
#: :py:func:`metacells.tools.project.compute_projection_weights`.
project_max_consistency_fold_factor: float = significant_gene_fold_factor - 1.0

#: Whether to compute projectio the log (base 2) of the data. See
#: :py:func:`metacells.tools.project.compute_projection_weights`.
project_log_data: bool = True

#: The regularization factor to use when computing log of fractions for finding the most similar group for outliers. See
#: :py:func:`metacells.tools.quality.compute_outliers_matches`.
outliers_fold_regularization: float = 1e-5

#: Whether to ignore the lateral genes of the atlas when computing projections. See
#: :py:func:`metacells.pipeline.projection.projection_pipeline`.
ignore_atlas_lateral_genes: bool = True

#: Whether to ignore the noisy genes of the atlas when computing projections. See
#: :py:func:`metacells.pipeline.projection.projection_pipeline`.
consider_atlas_noisy_genes: bool = True

#: Whether to ignore the non-marker genes of the atlas when computing projections. See
#: :py:func:`metacells.pipeline.projection.projection_pipeline`.
only_atlas_marker_genes: bool = True

#: Whether to ignore the non-marker genes of the query when computing projections. See
#: :py:func:`metacells.pipeline.projection.projection_pipeline`.
only_query_marker_genes: bool = False

#: Whether to ignore the lateral genes of the query when computing projections. See
#: :py:func:`metacells.pipeline.projection.projection_pipeline`.
ignore_query_lateral_genes: bool = True

#: Whether to ignore the noisy genes of the query when computing projections. See
#: :py:func:`metacells.pipeline.projection.projection_pipeline`.
consider_query_noisy_genes: bool = True

#: The minimal fraction of metacells where a gene has a high projection fold factor to mark the gene as "misfit".
#: See :py:func:`metacells.pipeline.projection.projection_pipeline`.
misfit_min_metacells_fraction: float = 0.5

#: The minimal fold between the maximal and minimal gene expression in metacells to be a "marker".
#: See :py:func:`metacells.tools.high.find_metacells_marker_genes`.
min_marker_metacells_gene_range_fold_factor: float = 2.0

#: The regularization factor to use after computing the fractions of the data for
#: computing metacell gene range folds. See
#: :py:func:`metacells.tools.high.find_metacells_marker_genes`.
metacells_gene_range_regularization: float = 1e-5

#: The minimal maximal gene expression in metacells to be a "marker".
#: See :py:func:`metacells.tools.high.find_metacells_marker_genes`.
min_marker_max_metacells_gene_fraction: float = 1e-4

#: Whether to renormalize the query to account for missing atlas genes when computing projections.
#: :py:func:`metacells.pipeline.projection.projection_pipeline`.
project_renormalize_query: bool = False

#: Whether to compute linear corrections for genes between the query and the atlas.
#: See :py:func:`metacells.pipeline.projection.projection_pipeline`.
project_corrections: bool = False

#: The minimal correlation between observed and projected genes for considering linear correction of the query gene
#: value. See :py:func:`metacells.pipeline.projection.projection_pipeline`.
project_min_corrected_gene_correlation: float = 0.8

#: The minimal strength of the correction between the mean query and projected mean value (for correlated genes).
#: See :py:func:`metacells.pipeline.projection.projection_pipeline`.
project_min_corrected_gene_factor: float = 0.15

#: The maximal number of deviant genes allowed for saying a query is similar to the projection in the atlas.
#: See :py:func:`metacells.tools.quality.compute_similar_query_metacells`
# and :py:func:`metacells.pipeline.projection.projection_pipeline`.
project_max_misfit_genes: int = 3

#: The quantile of each gene's normalized variance across the metacells to use for the overall gene's variability.
#: See :py:func:`metacells.tools.quality.compute_type_genes_normalized_variances`.
type_gene_normalized_variance_quantile: float = 0.95

#: Minimal fraction of atlas essential genes required for saying a query is similar to the projection in the
#: atlas.
#: See :py:func:`metacells.tools.quality.compute_similar_query_metacells`
# and :py:func:`metacells.pipeline.projection.projection_pipeline`.
project_min_essential_genes_fraction: float = 0.75
