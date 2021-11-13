install.packages('anndata')
install.packages('chameleon')
install.packages('dplyr')
install.packages('hdf5r')
install.packages('pheatmap')
install.packages('pracma')
install.packages('Seurat')
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
remotes::install_github("mojaveazure/seurat-disk")
