Metacells 0.9.0-dev.1 - Single-cell RNA Sequencing Analysis
===========================================================

.. image:: https://readthedocs.org/projects/metacells/badge/?version=latest
    :target: https://metacells.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

The metacells package implements the improved metacell algorithm [1]_ for single-cell RNA sequencing (scRNA-seq) data
analysis within the ``scipy https://www.scipy.org/`` framework. The original metacell algorithm [2]_ was implemented in
R. The python package contains various algorithmic improvements and is scalable for larger data sets (millions of
cells).

Metacell Analysis
-----------------

Naively, scRNA_seq data is a set of cell profiles, where for each one, for each gene, we get a count of the mRNA
molecules that existed in the cell for that gene. This serves as an indicator of how "expressed" or "active" the gene
is.

As in any real world technology, the raw data may suffer from technical artifacts (counting the molecules of two cells
in one profile, counting the molecules from a ruptured cells, counting only the molecules from the cell nucleus, etc.).
This requires pruning the raw data to exclude such artifacts.

The current technology scRNA-seq data is also very sparse (typically <<10% the RNA molecules are counted). This
introduces large sampling variance on top of the original signal, which itself contains significant inherent biological
noise.

Analyzing scRNA-seq data therefore requires processing the profiles in bulk. Classically, this has been done by directly
clustering the cells using various methods.

In contrast, the metacell approach groups together profiles of the "same" biological state into groups of cells of the
"same" biological state, with the *minimal* number of profiles needed for computing robust statistics (in particular,
mean gene expression). Each such group is a single "metacell".

By summing profiles of cells of the "same" state together, each metacell greatly reduces the sampling variance, and
provides a more robust estimation of the transcription state. Note a metacell is *not* a cell type (multiple metacells
may belong to the same "type", or even have the "same" state, if the data sufficiently over-samples this state). Also, a
metacell is *not* a parametric model of the cell state. It is merely a more robust description of some cell state.

The metacells should therefore be further analyzed as if they were cells, using additional methods to classify cell
types, detect cell trajectories and/or lineage, build parametric models for cell behavior, etc. Using metacells as input
for such analysis techniques should benefit both from the more robust, less noisy input; and also from the (~100-fold)
reduction in the number of cells to analyze when dealing with large data (e.g. analyzing millions of individual cells).

A common use case is taking a new data set and using an existing atlas with annotations (in particular, "type"
annotations) to provide initial annotations for the new data set. As of version 0.9 this capability is provided
by this package.

Metacell projection provides a quantitative "projected" genes profile for each query metacell in the atlas, together
with a "corrected" one for the same subset of genes shared between the query and the atlas. Actual correction is
optional, to be used only if there are technological differences between the data sets, e.g. 10X v2 vs. 10X v3. This
allows performing a quantitative comparison between the projected and corrected gene expression profiles for determining
whether the query metacell is a novel state that does not exist in the atlas, or, if it does match an atlas state,
analyze any differences it may still have. This serves both for quality control and for quantitative analysis of
perturbed systems (e.g. knockouts or disease models) in comparison to a baseline atlas.

Terminology and Results Format
------------------------------

.. note::

   Version 0.9 **breaks compatibility** with version 0.8 when it comes to some APIs and the names and semantics of the
   result annotations. See below for the description of updated results (and how they differ from 0.8). The new format
   is meant to improve the usability of the system in downstream analysis pipelines. For convenience we also list here
   the results of the new projection pipeline.

   In the upcoming version 0.10 we will migrate from using ``AnnData`` to using ``daf`` to represent the data (``h5ad``
   files will still be supported, either directly through an adapter or via a conversion import process). This will
   again unavoidingly break API compatibility, but will provide many advantages over the restricted ``AnnData`` APIs. We
   apologize for the inconvenience.

Metacells Computation
.....................

In theory, the only inputs required for metacell analysis are cell gene profiles with a UMIs count per gene per cell. In
practice, a key part of the analysis is specifying lists of genes for special treatment. We use the following
terminology for these lists:

``excluded_gene``, ``excluded_cell`` masks
    Excluded genes (and/or cells) are totally ignored by the algorithm (e.g. mytochondrial genes, cells with too few
    UMIs). Currently the 1st step of the processing must be to create a "clean" data set which lacks the excluded genes
    and cells.

``select_gene`` mask
    When computing metacells, we only consider the similarity between cells using a select subset of the genes. When we
    employ the divide-and-conquer algorithm this happens several times with different subsets being selected in
    different steps, so this mask is only a result of the simple direct (single-pile) algorithm. In general it shouldn't
    be used for downstream analysis of the results.

``lateral_gene`` mask
    Lateral genes are forbidden from being selected for computing cells similarity (e.g., cell cycle genes). In version
    0.8 these were called "forbidden" genes. Lateral genes are still counted towards the total UMIs count when computing
    gene expression levels for cells similarity. In addition, lateral genes are still used to compute deviant (outlier)
    cells. That is, each computed metacell should still have a consistent gene expression level even for lateral genes.

    The motivation is that we don't want the algorithm to even try to create metacells based on these genes. Since these
    genes may be very strong (again, cell cycle), they would overcome the cell-type genes we are interested in,
    resulting in for example an "M-state" metacell which combines cells from several (similar) cell types.

    Deciding on the "right" list of lateral genes is crucial for creating high-quality metacells. We rely on the analyst
    to provide this list based on prior biological knowledge. To support this supervised task, we provide the
    ``relate_genes`` pipeline for identifying genes closely related to known lateral genes, so they can be added to the
    list.

``noisy_gene`` mask
    Noisy genes are not only forbidden from being selected for computing cells similarity, but are also ignored when
    computing deviant (outlier) cells. Noisy genes are still counted towards the total UMIs count when computing gene
    expression level for cell similarity. That is, we don't expect cells in the same metacell to have the same
    expression level for such genes.

    The motivation is that some genes are inherently bursty and therefore cause many cells which are otherwise a good
    match for their metacell to be marked as deviant (outliers). An indication for this is by examining the
    ``deviant_fold`` matrix (see below).

    Deciding on the "right" list of noisy genes is again crucial for creating high-quality metacells (and minimizing the
    fraction of outlier cells). Again we rely on the analyst here, but we also provide the ``find_bursty_lonely_genes``
    function as a way to identify such troublesome genes. In version 0.8 this was called ``find_noisy_lonely_genes`` and
    the genes were excluded.

Having determined the inputs and possibly tweaking the hyper-parameters (favorite ones are the ``target_metacell_size``
which by default is 160K UMIs; this may be reduced for small data sets and may be increased for larger data sets), one
typically runs ``divide_and_conquer_pipeline`` to obtain the following:

``metacell`` (index) vs. ``metacell_name`` (string) per cell
    The result of computing metacells for a set of cells with the above assigns each cell a metacell index. We also give
    each metacell a name of the format ``M<index>.<checksum>`` where the checksum reflects the cells grouped into the
    metacell. This protects the analyst from mistakenly applying metadata assigned to metacells from an old computation
    to different newly computed metacells.

    The provided functions for conveying annotations from per-cell to per-metacell all currently use the metacell
    integer indices (this will change when we switch to ``daf``). The metacell string names are safer to use, especially
    when slicing the data.

Having computed the metacells, the next step is to run ``collect_metacells`` to create a new ``AnnData`` object for them
(when using ``daf``, they will be created in the same dataset for easier analysis), which will contain all the per-gene
metadata, and also:

``X`` per gene per metacell
    Once the metacells have been computed (typically using ``divide_and_conquer_pipeline``), we can collect the gene
    expression levels profile for each one. The main motivation for computing metacells is that they allow for a robust
    estimation of the gene expression level, and we therefore by default compute a matrix of gene fractions (which sum
    to 1) in each metacell, rather than providing a UMIs count for each. This simplifies the further analysis of the
    computed metacells.

    Note that the expression level of noisy genes is less reliable, as we do not guarantee the cells in each metacell
    have a consistent expression level for such genes. Our estimator therefore uses a normal weighted mean for most
    genes and a normalized geometric mean for the noisy gene.

``total_umis`` per metacell
    We still provide the effective total UMIs count for each metacell. This is important to ensure that analysis is
    based on a sufficient number of UMIs. Specifically, comparing expression levels of lowly-expressed genes will yield
    wildly inaccurate results unless a sufficient number of effective UMIs are involved. The functions provided here for
    computing fold factors (log base 2 of the ratio) and related comparisons automatically ignore cases when the sum of
    the effective UMIs of the compared values is below some threshold (by considering the effective fold factor to be
    0).

If using ``divide_and_conquer_pipeline``, the following are also computed (but not by the simple
``compute_divide_and_conquer_metacells``:

``rare_gene_module_<N>`` mask (for N = 0, ...)
    A mask of the genes combined into each of the detected "rare gene modules". This is done in (expensive)
    pre-processing before the full divide-and-conquer algorithm to increase the sensitivity of the method.

``rare_gene`` mask
    A mask of all the genes in all the rare gene modules, for convenience.

``rare_gene_module`` per cell or metacell
    The index of the rare gene module each cell or metacell expresses (or negative for the common case it expresses none
    of them).

``rare_cell``, ``rare_metacell`` mask
    A mask of all the cells or metacells expressing any of the rare gene modules, for convenience.

In theory one is free to go use the metacells for further analysis, but it is prudent to perform quality control first.
One obvious measure is the number of outlier cells (with a negative metacell index and a metacell name of ``Outliers``).
In addition, one should compute and look at the following (an easy way to compute all of them at once is to call
``compute_for_mcview``, this will change in the future):

``inner_fold`` (computed by ``compute_inner_fold_factors``)
    For each gene, in each metacell, hold the fold factor between the maximal and minimal gene expression level of the
    gene in the cells of the metacells. This uses the same (strong) normalization factor we use when computing deviant
    (outlier) cells, so ideally you will not see any fold factors above 3 (8x). Such fold factors may still exist when
    we cap the fraction of deviant (outlier) cells (by default we cap it at 25%), however this should be rare for good
    quality data sets. If the same genes have a high fold factor in many cells, you should consider marking them as
    noisy genes (you can double check this by looking at the most similar fold, see below).

``most_similar``, ``most_similar_name`` per cell (computed by ``compute_outliers_most_similar``)
    For each outlier cell (whose metacell index is ``-1`` and metacell name is ``Outliers``), the index and name of the
    metacell which is the "most similar" (has highest correlation) with the cell.

``most_similar_fold`` per gene per cell (computed by ``compute_outliers_fold_factors``)
    For each outlier cell, for each gene, hold the fold factor between the expression level of the gene in the cell and
    in the most similar metacell to that cell. This uses the same (strong) normalization factor we use when computing
    deviant (outlier) cells, so you should see some (non-excluded, non-noisy) genes with a fold factor above 3 (8x)
    which justify why we haven't merged that cell into a metacell. If there is a large number of outlier cells and a few
    genes have a high fold factor for many of them, you should consider marking these genes as noisy and recomputing the
    metacells.

``marker_gene`` mask (computed by ``find_metacells_marker_genes``)
    Given the computed metacells, we can identify genes that have a sufficient number of effective UMIs (in some
    metacells) and also have a wide range of expressions (between different metacells). These genes serve as markers for
    identifying the "type" of the metacell (or, more generally, the "gene programs" that are active in each metacell).

    Typically analysis groups the marker genes into "gene modules" (or, more generally, "gene programs"), and then use
    the notion of "type X expresses gene module/program Y". As of version 0.9, collecting such gene modules (or
    programs) is left to the analyst with little or no direct support in this package.

``x``, ``y`` per metacell (computed by ``compute_umap_by_markers``)
    A common and generally effective way to visualize the computed metacells is to project them to a 2D view. Currently
    we do this by giving UMAP a distance metric between metacells based on a logistic function based on the expression
    levels of the marker genes. In version 0.8 this was based on picking (some of) the selected genes.

    This view is good for quality control. If it forces "unrelated" cell types together, this might mean that more genes
    should be made lateral or marked as noisy (or even excluded). Or maybe the data contains a metacell of doublets, or
    a metacell mixing cells from different types, if too many genes were marked as lateral or noisy (it takes very few
    of these "connector" metacells to mess UMAP up).

    Also, one shouldn't read too much from the 2D layout, as by definition it can't express the "true" structure of the
    data. Looking at specific gene-gene plots gives much more robust insight into the actual differences between the
    metacell types.

``u``, ``v``, ``w`` per metacell (computed by ``compute_umap_by_markers`` with ``dimentions=3``)
    These are computed similar to ``x`` and ``y``, but project the metacells to a 3D space. One way to kickstart
    analysis of brand-new metacells data is to use K-means to group them into some number of clusters (which wouldn't
    map exactly to "types" but would be a start). To give a reasonable initial color to each such cluster, we project
    the metacells to a 3D space and (using the ``chameleon`` R package) map that to RGB, so that "similar" colors are
    given to "similar" clusters. Using a 3D space allows the projection to better capture the "true" structure of the
    data, but of course it is still only an approximation.

Metacells Projection
....................

For the use case of projecting metacells we use the following terminology:

``atlas``
    A set of metacells with associated metadata, most importantly a ``type`` annotation per metacell. In addition, the
    atlas may provide an ``essential_gene_of_<type>`` mask for each such type. For a query metacell to successfully
    project to a given type will require that the expression of the type's essential genes matches the atlas. We
    also use the metadata listed above (specifically, ``lateral_gene``, ``noisy_gene`` and ``marker_gene``).

``query``
    A set of metacells with minimal associated metadata, specifically without a ``type``. This may optionally contain
    its own ``lateral_gene``, ``noisy_gene`` and/or ``marker_gene`` annotations.

Given these two input data sets, the ``projection_pipeline`` computes the following (inside the query ``AnnData``
object):

``atlas_gene`` mask
    A mask of the query genes that also exist in the atlas. We match genes by their name; if projecting query data from
    a different technology, we expect the caller to modify the query gene names to match the atlas before projecting
    it.

``atlas_lateral_gene``, ``atlas_noisy_gene``, ``atlas_marker_gene``, ``essential_gene_of_<type>`` masks
    These masks are copied from the atlas to the query (restricting them to the common ``atlas_gene`` subset).

``corrected_fractions`` per gene per query metacell
    For each ``atlas_gene``, its fraction in each query metacell, out of all the atlas genes. This may be further
    corrected (see below) if projecting between different scRNA-seq technologies (e.g. 10X v2 and 10X v3). For
    non-``atlas_gene`` this is 0.

``projected_fractions`` per gene per query metacell
    For each ``atlas_gene``, its fraction in its projection on the atlas. This projection is computed as a weighted
    average of some atlas metacells (see below), which are all sufficiently close to each other (in terms of gene
    expression), so averaging them is reasonable to capture the fact the query metacell may be along some position on
    some gradient that isn't an exact match for any single atlas metacell. For non-``atlas_gene`` this is 0.

``total_atlas_umis`` per query metacell
    The total effective UMIs of the ``atlas_gene`` in each query metacell. This is used in the analysis as described for
    ``total_umis`` above, that is, to ensure comparing expression levels will ignore cases where the total effective
    number of UMIs of both compared gene profiles is too low to make a reliable determination. In such cases we take the
    effective fold factor to be 0.

``weights`` per query metacell per atlas metacsll
    The weights used to compute the ``projected_fractions``. Due to ``AnnData`` limitations this is returned as a
    separate object, but in ``daf`` we may be able to do better.

In theory, this would be enough for looking at the query metacells and comparing them to the atlas, possibly projecting
metadata from the atlas to the query (e.g., the metacell type). In practice, there is significant amount of quality
control one needs to apply before accepting these results, which we compute as follows:

``correction_factor`` per ``atlas_gene``
    If projecting a query on an atlas with different technologies (e.g., 10X v3 to 10X v2), an automatically computed
    factor we multiplied the query gene fractions by to compensate for the systematic difference between the
    technologies (1.0 for uncorrected genes and non-``atlas_gene``).

``projected_type`` per query metacell
    For each query metacell, the best ``type`` we can assign to it based on its projection. Note this does not indicate
    that the query metacell is "truly" of this type; to make this determination one needs to look at the quality control
    data below.

``projected_secondary_type`` per query metacell
    In some cases, a query metacell may fail to project to a single region of the atlas, but instead does project well
    to a combination of two regions. This may be due to the query metacell containing doublets, of a mixture of cells
    which match different atlas regions (e.g. due to sparsity of data in the query data set). Either way, if this
    happens, we place here the type that best describes the secondary region the query metacell was projected to,
    otherwise this would be the empty string. Note that the ``weights`` matrix above does not distinguish between the
    regions.

``fitted_gene_of_<type>`` mask
    For each type, the genes that were in general successfully projected from the query to the atlas for that type. For
    non-``atlas_gene`` this is set to ``False``. This does not guarantee that a specific query metacell of that type
    successfully projected each of these genes, just that most of them did. Any ``atlas_gene`` outside this mask failed
    to project well from the query to the atlas for (most) metacells of this type. Whether this indicates that the query
    metacells weren't "truly" of the ``projected_type`` is a decision which only the analyst can make based on prior
    biological knowledge of the relevant genes.

``fitted_gene`` mask per gene per query metacell
    For each ``atlas_gene`` for each query metacell, whether the gene was expected to be well projected, based on the
    query metacell ``projected_type`` (and the ``projected_secondary_type``, if any). For non-``atlas_gene`` this is set
    to ``False``. This does not guarantee the gene was actually well projected.

``misfit_gene`` mask per gene per query metacell
    For each ``atlas_gene`` for each query metacell, whether the ``corrected_fractions`` of the gene was significantly
    different from the ``projected_fractions``. This is expected to be rate for ``fitted_gene`` and common for the rest
    of the ``atlas_gene``. For non-``atlas_gene`` this is set to ``False``. For most (but not all) metacells and genes,
    this would be ``False`` for ``fitted_gene`` and ``True`` for the rest of the ``atlas_gene``.

``essential_gene`` mask per gene per query metacell
    Which of the ``atlas_gene`` were listed in the ``essential_gene_of_<type>`` for the ``projected_type`` (and also the
    ``projected_secondary_type``, if any) of each query metacell. If an ``essential_gene`` is also a ``misfit_gene``,
    then one should be very suspicious whether the query metacell is "truly" of the ``projected_type``.

``projection_correlation`` per query metacell
    The correlation between between the ``corrected_fractions`` and the ``projected_fractions`` for only the
    ``fitted_gene`` expression levels of each query metacell. This serves as a very rough estimator for the quality of
    the projection for this query metacell (e.g. can be used to compute R^2 values).

``projected_fold`` per gene per query metacell
    The fold factor between the ``corrected_fractions`` and the ``projected_fractions`` (0 for non-``atlas_gene``). If
    the absolute value of this is high (3 for 8x ratio) then the gene was not well mapped for this metacell. It is
    expected this would have low values for most fitted genes and high values for the rest, but specific values will
    vary from one query metacell to another. This allows the analyst to make fine-grained determination about the
    quality of the projection, and/or identify quantitative differences between the query and the atlas (e.g., when
    studying perturbed systems such as knockouts or disease models).

``similar`` mask per query metacell
    A conservative determination of whether the query metacell is "similar" to its projection on the atlas. This is
    based on whether the number of ``misfit_gene`` for the query metacell is low enough (by default, up to 3 genes), and
    also that at least 75% of the ``essential_gene`` of the query metacell were not ``misfit_gene``. Note that this
    explicitly allows for a ``projected_secondary_type``, that is, a metacell of doublets will be "similar" to the
    atlas.

    The final determination is, as always, up to the analyst, based on prior biological knowledge, the context of the
    collection of the query (and atlas) data sets, etc. The analyst need not (indeed, should not) blindly accept this
    determination without examining the rest of the quality control data listed above.

Installation
------------

In short: ``pip install metacells``. Note that ``metacells`` requires many "heavy" dependencies, most notably ``numpy``,
``pandas``, ``scipy``, ``scanpy``, which ``pip`` should automatically install for you. If you are running inside a
``conda`` environment, you might prefer to use it to first install these dependencies, instead of having ``pip`` install
them from ``PyPI``.

Note that ``metacells`` only runs natively on Linux and MacOS. To run it on a Windows computer, you must activate
`Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl>`_ and install ``metacells`` within it.

The metacells package contains extensions written in C++. The ``metacells`` distribution provides pre-compiled Python
wheels for both Linux and MacOS, so installing it using ``pip`` should not require a C++ compilation step.

Note that for X86 CPUs, these pre-compiled wheels were built to use AVX2 (Haswell/Excavator CPUs or newer), and will not
work on older CPUs which are limited to SSE. Also, these wheels will not make use of any newer instructions (such as
AVX512), even if available. While these wheels may not the perfect match for the machine you are running on, they are
expected to work well for most machines.

To see the native capabilities of your machine, you can ``grep flags /proc/cpuinfo | head -1`` which will give you a
long list of supported CPU features in an arbitrary order, which may include ``sse``, ``avx2``, ``avx512``, etc. You can
therefore simply ``grep avx2 /proc/cpuinfo | head -1`` to test whether AVX2 is/not supported by your machine.

You can avoid installing the pre-compiled wheel by running ``pip install metacells --install-option='--native'``. This
will force ``pip`` to compile the C++ extensions locally on your machine, optimizing for its native capabilities,
whatever these may be. This will take much longer but may give you faster results (note: the results will **not** be
exactly the same as when running the precompiled wheel due to differences in floating-point rounding). Also, this
requires you to have a C++ compiler which supports C++14 installed (either ``g++`` or ``clang``). Installing a C++
compiler depends on your specific system (using ``conda`` may make this less painful).

Vignettes
---------

The `generated documentation <https://metacells.readthedocs.io/en/latest>`_ contains the following vignettes:
`Basic Metacells Vignette <https://metacells.readthedocs.io/en/latest/Metacells_Vignette.html>`_,
`Manual Analysis Vignette <https://metacells.readthedocs.io/en/latest/Manual_Analysis.html>`_,
and `Seurat Analysis Vignette <https://metacells.readthedocs.io/en/latest/Seurat_Analysis.html>`_.

You can also access their very latest version in the
`Github repository <https://github.com/tanaylab/metacells/tree/master/docs>`_.

References
----------

Please cite the references appropriately in case they are used:

.. [1] Ben-Kiki, O., Bercovich, A., Lifshitz, A. et al. Metacell-2: a divide-and-conquer metacell algorithm for scalable
   scRNA-seq analysis. Genome Biol 23, 100 (2022). https://doi.org/10.1186/s13059-022-02667-1

.. [2] Baran, Y., Bercovich, A., Sebe-Pedros, A. et al. MetaCell: analysis of single-cell RNA-seq data using K-nn graph
   partitions. Genome Biol 20, 206 (2019). `10.1186/s13059-019-1812-2 <https://doi.org/10.1186/s13059-019-1812-2>`_

License (MIT)
-------------

Copyright © 2020, 2021, 2022 Weizmann Institute of Science

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
