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

* Download the `PBMC 163K
  <http://www.wisdom.weizmann.ac.il/~atanay/metac_data/pbmc163k.h5ad.gz>`_ data file to the work
  directory (use ``wget``, ``curl``, your browser's download function, whatever).

  This data is a unification of the following TenX sample data sets:

  * `Fresh 68k PBMCs (Donor A) <https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/fresh_68k_pbmc_donor_a>`_

  * `CD19+ B Cells <https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/b_cells>`_

  * `CD14+ Monocytes <https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/cd14_monocytes>`_

  * `CD34+ Cells <https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/cd34>`_

  * `CD4+ Helper T Cells <https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/cd4_t_helper>`_

  * `CD56+ Natural Killer Cells <https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/cd56_nk>`_

  * `CD8+ Cytotoxic T cells <https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/cytotoxic_t>`_

  * `CD4+/CD45RO+ Memory T Cells <https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/memory_t>`_

  * `CD8+/CD45RA+ Naive Cytotoxic T Cells <https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/naive_cytotoxic>`_

  * TODO: ``Naive T``

  * TODO: ``Regular T``

* ``gunzip pbmc163k.h5ad.gz`` to extract the ``pbmc163k.h5ad`` data to analyze.
