Vignette
========

This vignette demonstrates step-by-step use of the metacells package to analyze scRNA-seq data.
It is expected that these steps will be modified when analyzing other data sets, and points
where such modifications are likely will be called out.

The following instructions assume a Unix system. If you are unfortunate enough to run on Windows,
the easy way to follow them is to install WSL2 (Windows System for Linux 2), and the hard way is to
convert commands to the equivalent Windows operations.

Preparation
-----------

* Install the metacells package (e.g. ``pip install metacells``).

* Create a work directory somewhere and ``cd`` to it.

Data Set
--------

The metacells package is built around the `scanpy https://pypi.org/project/scanpy/`_ framework. In
particular it uses `anndata https://pypi.org/project/anndata/`_ to hold the analyzed data, and uses
`.h5ad` files to persist this data on disk. You can access also these files directly from R using
several packages, most notably `anndata https://cran.r-project.org/web/packages/anndata/index.html`.

You can convert data from various "standard" scRNA data formats into a `.h5ad` file using any of the
functions available in the scanpy and/or anndata packages. Note that converting textual data to this
format takes a "non-trivial" amount of time for large data sets. Mercifully, this is a one-time
operation. Less excusable is the fact that none of the above packages memory-map the `.h5ad` files
so reading large files will still take a noticeable amount of time for no good reason.

For the purposes of this vignette, we'll use a ~160K cells data set which is a unification of
several batches of PBMC scRNA data from `TenX
<https://support.10xgenomics.com/single-cell-gene-expression/datasets>`_, specifically from the
"Single Cell 3' Paper: Zheng et al. 2017" datasets. Since 10x do not provide stable links to their
data sets, and to avoid the long time it would take to convert their textual format files to `.h5ad`
files, simply download the `PBMC 163K
<http://www.wisdom.weizmann.ac.il/~atanay/metac_data/pbmc163k.h5ad.gz>`_ data file to your work
directory (using ``wget``, ``curl``, your browser's download function, etc.).

To save network bandwidth (download time), this file is compressed, so you will need to run ``gunzip
pbmc163k.h5ad.gz`` to extract the ``pbmc163k.h5ad`` data to analyze.

Cleaning The Data
-----------------

The first step in processing the data is to extract a "clean" subset of it for further analysis. The
exact details might vary depending on your specific data set's origins. Still, the metacells package
supports a basic 2-phase procedure which should be useful in many cases.

Phase 1: Clean genes
....................

There are several reasons to exclude genes from the "clean" data:

- Genes whose inclusion in the data is detrimental to the analysis. The poster child for such genes
  are mitochondrial genes, but other genes might interfere with the analysis to such a degree that
  requires them to be completely excluded from the data.

  Typically we create a file called `excluded_genes.txt` which contains one gene name per line,
  and read it by wr
