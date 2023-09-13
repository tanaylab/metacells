Metacells 0.9.3 - Single-cell RNA Sequencing Analysis
=====================================================

.. image:: https://readthedocs.org/projects/metacells/badge/?version=latest
    :target: https://metacells.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

The metacells package implements the improved metacell algorithm [1]_ for single-cell RNA sequencing (scRNA-seq) data
analysis within the `scipy <https://www.scipy.org/>`_ framework. The original metacell algorithm [2]_ was implemented in
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

**NOTE**: Version 0.9 **breaks compatibility** with version 0.8 when it comes to some APIs and the names and semantics
of the result annotations. See below for the description of updated results (and how they differ from version 0.8). The
new format is meant to improve the usability of the system in downstream analysis pipelines. For convenience we also
list here the results of the new projection pipeline added in version 0.9.*. Versions 0.9.1 and 0.9.2 contain some bug
fixes. The latest version 0.9.3 allows specifying target UMIs for the metacells, in addition to the target size in
cells, and adaptively tries to satisfy both. This should produce better-sized metacells "out of the box" compared to the
0.9.[0-2] versions.

If you have existing metacell data that was computed using version 0.8 (the current published version you will get
from using ``pip install metacells``, you can use the provided
`conversion script <https://github.com/tanaylab/metacells/blob/master/bin/convert_0.8_to_0.9.py>`_
script to migrate your data to the format described below, while preserving any additional annotations you may have
created for your data (e.g. metacells type annotations). The script will not modify your existing data files, so you can
examine the results and tweak them if necessary.

In the upcoming version 0.10 we will migrate from using ``AnnData`` to using ``daf`` to represent the data (``h5ad``
files will still be supported, either directly through an adapter or via a conversion process). This will again
unavoidingly break API compatibility, but will provide many advantages over the restricted ``AnnData`` APIs.

We apologize for the inconvenience.

Metacells Computation
.....................

In theory, the only inputs required for metacell analysis are cell gene profiles with a UMIs count per gene per cell. In
practice, a key part of the analysis is specifying lists of genes for special treatment. We use the following
terminology for these lists:

``excluded_gene``, ``excluded_cell`` masks
    Excluded genes (and/or cells) are totally ignored by the algorithm (e.g. mytochondrial genes, cells with too few
    UMIs).

    Deciding on the "right" list of excluded genes (and cells) is crucial for creating high-quality metacells. We rely
    on the analyst to provide this list based on prior biological knowledge. To support this supervised task, we provide
    the ``excluded_genes`` and ``exclude_cells`` functions which implement "reasonable" strategies for detecting some
    (not all) of the genes and cells to exclude. For example, these will exclude any genes found by
    ``find_bursty_lonely_genes``, (called ``find_noisy_lonely_genes`` in v0.8). Additional considerations might be to
    use ``relate_genes`` to (manually) exclude genes that are highly correlated with known-to-need-to-be-excluded genes,
    or exclude any cells that are marked as doublets, etc.

    Currently the 1st step of the processing must be to create a "clean" data set which lacks the excluded genes and
    cells (e.g. using ``extract_clean_data``). When we switch to ``daf`` we'll just stay with the original data set and
    apply the exclusion masks to the rest of the algorithm.

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
    Noisy genes are given more freedom when computing deviant (outlier) cells. That is, we don't expect the expression
    level of such genes in the cells in the same metacell to be as consistent as we do for regular (non-noisy) genes.
    Note this isn't related to the question of whether the gene is lateral of not. That is, a gee maybe lateral, noisy,
    both, or neither.

    The motivation is that some genes are inherently bursty and therefore cause many cells which are otherwise a good
    match for their metacell to be marked as deviant (outliers). An indication for this is by examining the
    ``deviant_fold`` matrix (see below).

    Deciding on the "right" list of noisy genes is again crucial for creating high-quality metacells (and minimizing the
    fraction of outlier cells). Again we rely on the analyst here,

Having determined the inputs and possibly tweaking the hyper-parameters (a favorite one is the ``target_metacell_size``,
which by default is 160K UMIs; this may be reduced for small data sets and may be increased for larger data sets), one
typically runs ``divide_and_conquer_pipeline`` to obtain the following:

``metacell`` (index) vs. ``metacell_name`` (string) per cell
    The result of computing metacells for a set of cells with the above assigns each cell a metacell index. We also give
    each metacell a name of the format ``M<index>.<checksum>`` where the checksum reflects the cells grouped into the
    metacell. This protects the analyst from mistakenly applying metadata assigned to metacells from an old computation
    to different newly computed metacells.

    We provide functions (``convey_obs_to_group``, ``convey_group_to_obs``) for conveying between per-cell and
    per-metacell annotations, which all currently use the metacell integer indices (this will change when we switch to
    ``daf``). The metacell string names are safer to use, especially when slicing the data.

``dissolve`` cells mask
    Whether the cell was in a candidate matecall that was dissolved due to being too small (too few cells and/or total
    UMIs). This may aid quality control when there are a large number of outliers; lowering the ``target_metacell_size``
    may help avoid this.

``selected_gene`` mask
    Whether each gene was ever selected to be used to compute the similarity between cells to compute the metacells.
    When using the divide-and-conquer algorithm, this mask is different for each pile (especially in the second phase
    when piles are homogeneous). This mask is the union of all the masks used in all the piles. It is useful for
    ensuring no should-be-lateral genes were selected as this would reduce the quality of the metacells. If such genes
    exist, add them to the ``lateral_gene`` mask and recompute the metacells.

Having computed the metacells, the next step is to run ``collect_metacells`` to create a new ``AnnData`` object for them
(when using ``daf``, they will be created in the same dataset for easier analysis), which will contain all the per-gene
metadata, and also:

``X`` per gene per metacell
    Once the metacells have been computed (typically using ``divide_and_conquer_pipeline``), we can collect the gene
    expression levels profile for each one. The main motivation for computing metacells is that they allow for a robust
    estimation of the gene expression level, and therefore we by default compute a matrix of gene fractions (which sum
    to 1) in each metacell, rather than providing a UMIs count for each. This simplifies the further analysis of the
    computed metacells (this is known as ``e_gc`` in the old R metacells package).

    Note that the expression level of noisy genes is less reliable, as we do not guarantee the cells in each metacell
    have a consistent expression level for such genes. Our estimator therefore uses a normal weighted mean for most
    genes and a normalized geometric mean for the noisy gene. Since the sizes of the cells collected into the same
    metacell may vary, our estimator also ensures one large cell doesn't dominate the results. That is, the computed
    fractions are *not* simply "sum of the gene UMIs in all cells divided by the sum of all gene UMIs in all cells".

``grouped`` per metacell
    The number of cells grouped into each metacell.

``total_umis`` per metacell, and per gene per metacell
    We still provide the total UMIs count for each each gene for each cell in each metacell, and the total UMIs in each
    metacell. Note that the estimated fraction of each gene in the metacell is *not* its total UMIs divided by the
    metacell's total UMIs; the actual estimator is more complex.

    The total UMIs are important to ensure that analysis is meaningful. For example, comparing expression levels of
    lowly-expressed genes in two metacells will yield wildly inaccurate results unless a sufficient number of UMIs were
    used (the sum of UMIs of the gene in both compared metacells). The functions provided here for computing fold
    factors (log base 2 of the ratio) and related comparisons automatically ignore cases when this sum is below some
    threshold (40) by considering the effective fold factor to be 0 (that is, "no difference").

``metacells_level`` per cell or metacell
    This is 0 for rare gene module metacells, 1 for metacells computed from the main piles in the 2nd divide-and-conquer
    phase and 2 for metacells computed for their outliers.

If using ``divide_and_conquer_pipeline``, the following are also computed (but not by the simple
``compute_divide_and_conquer_metacells``:

``rare_gene_module_<N>`` mask (for N = 0, ...)
    A mask of the genes combined into each of the detected "rare gene modules". This is done in (expensive)
    pre-processing before the full divide-and-conquer algorithm to increase the sensitivity of the method, by creating
    metacells computed only from cells that express each rare gene module.

``rare_gene`` mask
    A mask of all the genes in all the rare gene modules, for convenience.

``rare_gene_module`` per cell or metacell
    The index of the rare gene module each cell or metacell expresses (or negative for the common case it expresses none
    of them).

``rare_cell``, ``rare_metacell`` masks
    A mask of all the cells or metacells expressing any of the rare gene modules, for convenience.

In theory one is free to go use the metacells for further analysis, but it is prudent to perform quality control first.
One obvious measure is the number of outlier cells (with a negative metacell index and a metacell name of ``Outliers``).
In addition, one should compute and look at the following (an easy way to compute all of them at once is to call
``compute_for_mcview``, this will change in the future):

``most_similar``, ``most_similar_name`` per cell (computed by ``compute_outliers_most_similar``)
    For each outlier cell (whose metacell index is ``-1`` and metacell name is ``Outliers``), the index and name of the
    metacell which is the "most similar" to the cell (has highest correlation).

``deviant_fold`` per gene per cell (computed by ``compute_deviant_folds``)
    For each cell, for each gene, the ``deviant_fold`` holds the fold factor (log base 2) between the expression level
    of the gene in the cell and the metacell it belongs to (or the most similar metacell for outlier cells). This uses
    the same (strong) normalization factor we use when computing deviant (outlier) cells, so for outliers, you should
    see some (non-excluded, non-noisy) genes with a fold factor above 3 (8x), or some (non-excluded, noisy) genes with a
    fold factor above 5 (32x), which justify why we haven't merged that cell into a metacell; for cells grouped into
    metacells, you shouldn't see (many) such genes. If there is a large number of outlier cells and a few non-noisy
    genes have a high fold factor for many of them, you should consider marking these genes as noisy and recomputing the
    metacells. If they are already marked as noisy, you may want to completely exclude them.

``inner_fold`` per gene per metacell (computed by ``compute_inner_folds``)
    For each metacell, for each gene, the ``inner_fold`` is the strongest (highest absolute value) ``deviant_fold`` of
    any of the cells contained in the metacell. Both this and the ``inner_stdev_log`` below can be used for quality
    control over the consistency of the gene expression in the metacell.

``significant_inner_folds_count`` per gene
    For each gene, the number of metacells in which there's at least one cell with a high ``deviant_fold`` (that is,
    where the ``inner_fold`` is high). This helps in identifying troublesome genes, which can be then marked as noisy,
    lateral or even excluded, depending on their biological significance.

``inner_stdev_log`` per gene per metacell (computed by ``compute_inner_stdev_logs``)
    For each metacell, for each gene, the standard deviation of the log (base 2) of the fraction of the gene across the
    cells of the metacell. Ideally, the standard deviation should be ~1/3rd of the ``deviants_min_gene_fold_factor``
    (which is ``3`` by default), indicating that (all)most cells are within that maximal fold factor. In practice we may
    see higher values - the lower, the better. Both this and the ``inner_fold`` above can be used for quality control over the consistency of the gene expression in the metacell.

``marker_gene`` mask (computed by ``find_metacells_marker_genes``)
    Given the computed metacells, we can identify genes that have a sufficient number of effective UMIs (in some
    metacells) and also have a wide range of expressions (between different metacells). These genes serve as markers for
    identifying the "type" of the metacell (or, more generally, the "gene programs" that are active in each metacell).

    Typically analysis groups the marker genes into "gene modules" (or, more generally, "gene programs"), and then use
    the notion of "type X expresses the gene module/programs Y, Z, ...". As of version 0.9, collecting such gene modules
    (or programs) is left to the analyst with little or no direct support in this package, other than providing the rare
    gene modules (which by definition would apply only to a small subset of the metacells).

``x``, ``y`` per metacell (computed by ``compute_umap_by_markers``)
    A common and generally effective way to visualize the computed metacells is to project them to a 2D view. Currently
    we do this by giving UMAP a distance metric between metacells based on a logistic function based on the expression
    levels of the marker genes. In version 0.8 this was based on picking (some of) the selected genes.

    This view is good for quality control. If it forces "unrelated" cell types together, this might mean that more genes
    should be made lateral, or noisy, or even excluded; or maybe the data contains a metacell of doublets; or metacells
    mixing cells from different types, if too many genes were marked as lateral or noisy, or excluded. It takes a
    surprising small number of such doublet/mixture metacells to mess up the UMAP projection.

    Also, one shouldn't read too much from the 2D layout, as by definition it can't express the "true" structure of the
    data. Looking at specific gene-gene plots gives much more robust insight into the actual differences between the
    metacell types, identify doublets, etc.

``obs_outgoing_weights`` per metacell per metacell (also computed by ``compute_umap_by_markers``)
    The (sparse) matrix of weights of the graph used to generate the ``x`` and ``y`` 2D projection. This graph is *very*
    sparse, that is, has a very low degree for the nodes. It is meant to be used only in conjunction with the 2D
    coordinates for visualization, and should **not** be used by any downstream analysis to determine which metacells
    are "near" each other for any other purpose.

Metacells Projection
....................

For the use case of projecting metacells we use the following terminology:

``atlas``
    A set of metacells with associated metadata, most importantly a ``type`` annotation per metacell. In addition, the
    atlas may provide an ``essential_gene_of_<type>`` mask for each type. For a query metacell to successfully project
    to a given type will require that the query's expression of the type's essential genes matches the atlas. We also
    use the metadata listed above (specifically, ``lateral_gene``, ``noisy_gene`` and ``marker_gene``).

``query``
    A set of metacells with minimal associated metadata, specifically without a ``type``. This may optionally contain
    its own ``lateral_gene``, ``noisy_gene`` and/or even ``marker_gene`` annotations.

``ignored_gene`` mask, ``ignored_gene_of_<type>`` mask
    A set of genes to not even try to match between the query and the atlas. In general the projection matches only a
    subset of the genes (that are common to the atlas and the query). However, the analyst has the option to force
    additional genes to be ignored, either in general or only when projecting metacells of a specific type. Manually
    ignoring specific genes which are known not to match (e.g., due to the query being some experiment, e.g. a knockout
    or a disease model) can improve the quality of the projection for the genes which do match.

Given these two input data sets, the ``projection_pipeline`` computes the following (inside the query ``AnnData``
object):

``atlas_gene`` mask
    A mask of the query genes that also exist in the atlas. We match genes by their name; if projecting query data from
    a different technology, we expect the caller to modify the query gene names to match the atlas before projecting
    it.

``atlas_lateral_gene``, ``atlas_noisy_gene``, ``atlas_marker_gene``, ``essential_gene_of_<type>`` masks
    These masks are copied from the atlas to the query (restricting them to the common ``atlas_gene`` subset).

``projected_noisy_gene``
    The mask of the genes that were considered "noisy" when computing the projection. By default this is the union
    of the noisy atlas and query genes.

``corrected_fraction`` per gene per query metacell
    For each ``atlas_gene``, its fraction in each query metacell, out of only the atlas genes. This may be further
    corrected (see below) if projecting between different scRNA-seq technologies (e.g. 10X v2 and 10X v3). For
    non-``atlas_gene`` this is 0.

``projected_fraction`` per gene per query metacell
    For each ``atlas_gene``, its fraction in its projection on the atlas. This projection is computed as a weighted
    average of some atlas metacells (see below), which are all sufficiently close to each other (in terms of gene
    expression), so averaging them is reasonable to capture the fact the query metacell may be along some position on
    some gradient that isn't an exact match for any specific atlas metacell. For non-``atlas_gene`` this is 0.

``total_atlas_umis`` per query metacell
    The total UMIs of the ``atlas_gene`` in each query metacell. This is used in the analysis as described for
    ``total_umis`` above, that is, to ensure comparing expression levels will ignore cases where the total number of
    UMIs of both compared gene profiles is too low to make a reliable determination. In such cases we take the fold
    factor to be 0.

``weights`` per query metacell per atlas metacsll
    The weights used to compute the ``projected_fractions``. Due to ``AnnData`` limitations this is returned as a
    separate object, but in ``daf`` we should be able to store this directly into the query object.

In theory, this would be enough for looking at the query metacells and comparing them to the atlas, and to project
metadata from the atlas to the query (e.g., the metacell type) using ``convey_atlas_to_query``. In practice, there is
significant amount of quality control one needs to apply before accepting these results, which we compute as follows:

``correction_factor`` per gene
    If projecting a query on an atlas with different technologies (e.g., 10X v3 to 10X v2), an automatically computed
    factor we multiplied the query gene fractions by to compensate for the systematic difference between the
    technologies (1.0 for uncorrected genes and 0.0 for non-``atlas_gene``).

``projected_type`` per query metacell
    For each query metacell, the best atlas ``type`` we can assign to it based on its projection. Note this does not
    indicate that the query metacell is "truly" of this type; to make this determination one needs to look at the
    quality control data below.

``projected_secondary_type`` per query metacell
    In some cases, a query metacell may fail to project well to a single region of the atlas, but does project well to a
    combination of two distinct atlas regions. This may be due to the query metacell containing doublets, of a mixture
    of cells which match different atlas regions (e.g. due to sparsity of data in the query data set). Either way, if
    this happens, we place here the type that best describes the secondary region the query metacell was projected to;
    otherwise this would be the empty string. Note that the ``weights`` matrix above does not distinguish between the
    regions.

``fitted_gene_of_<type>`` mask
    For each type, the genes that were projected well from the query to the atlas for most cells of that type; any
    ``atlas_gene`` outside this mask failed to project well from the query to the atlas for most metacells of this type.
    For non-``atlas_gene`` this is set to ``False``.

    Whether failing to project well some of the ``atlas_gene`` for most metacells of some ``projected_type`` indicates
    that they aren't "truly" of that type is a decision which only the analyst can make based, on prior biological
    knowledge of the relevant genes.

``fitted`` mask per gene per query metacell
    For each ``atlas_gene`` for each query metacell, whether the gene was expected to be projected well, based on the
    query metacell ``projected_type`` (and the ``projected_secondary_type``, if any). For non-``atlas_gene`` this is set
    to ``False``. This does not guarantee the gene was actually projected well.

``misfit`` mask per gene per query metacell
    For each ``atlas_gene`` for each query metacell, whether the ``corrected_fraction`` of the gene was significantly
    different from the ``projected_fractions`` (that is, whether the gene was not projected well for this metacell). For
    non-``atlas_gene`` this is set to ``False``, to make it easier to identify problematic genes.

    This is expected to be rare for ``fitted`` genes and common for the rest of the ``atlas_gene``. If too many
    ``fitted`` genes are also ``misfit``, then one should be suspicious whether the query metacell is "truly" of the
    ``projected_type``.

``essential`` mask per gene per query metacell
    Which of the ``atlas_gene`` were also listed in the ``essential_gene_of_<type>`` for the ``projected_type`` (and
    also the ``projected_secondary_type``, if any) of each query metacell.

    If an ``essential`` gene is also a ``misfit`` gene, then one should be very suspicious whether the query metacell is
    "truly" of the ``projected_type``.

``projected_correlation`` per query metacell
    The correlation between between the ``corrected_fraction`` and the ``projected_fraction`` for only the ``fitted``
    genes expression levels of each query metacell. This serves as a very rough estimator for the quality of the
    projection for this query metacell (e.g. can be used to compute R^2 values).

    In general we expect high correlation (more than 0.9 in most metacells) since we restricted the ``fitted`` genes
    mask only to genes we projected well.

``projected_fold`` per gene per query metacell
    The fold factor between the ``corrected_fraction`` and the ``projected_fraction`` (0 for non-``atlas_gene``). If
    the absolute value of this is high (3 for 8x ratio) then the gene was not projected well for this metacell. This
    will be 0 for non-``atlas_gene``.

    It is expected this would have low values for most ``fitted`` genes and high values for the rest of the
    ``atlas_gene``, but specific values will vary from one query metacell to another. This allows the analyst to make
    fine-grained determination about the quality of the projection, and/or identify quantitative differences between the
    query and the atlas (e.g., when studying perturbed systems such as knockouts or disease models).

``similar`` mask per query metacell
    A conservative determination of whether the query metacell is "similar" to its projection on the atlas. This is
    based on whether the number of ``misfit`` for the query metacell is low enough (by default, up to 3 genes), and also
    that at least 75% of the ``essential`` genes of the query metacell were not ``misfit`` genes. Note that this
    explicitly allows for a ``projected_secondary_type``, that is, a metacell of doublets will be "similar" to the
    atlas, but a metacell of a novel state missing from the atlas will be "dissimilar".

    The final determination of whether to accept the projection is, as always, up to the analyst, based on prior
    biological knowledge, the context of the collection of the query (and atlas) data sets, etc. The analyst need not
    (indeed, *should not*) blindly accept the ``similar`` determination without examining the rest of the quality
    control data listed above.

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

You can avoid installing the pre-compiled wheel by running ``pip install metacells --no-binary :all:``. This will force
``pip`` to compile the C++ extensions locally on your machine, optimizing for its native capabilities, whatever these
may be. This will take much longer but may give you *somewhat* faster results (note: the results will **not** be exactly
the same as when running the precompiled wheel due to differences in floating-point rounding). Also, this requires you
to have a C++ compiler which supports C++14 installed (either ``g++`` or ``clang``). Installing a C++ compiler depends
on your specific system (using ``conda`` may make this less painful).

Vignettes
---------

The latest vignettes can be found `here <https://github.com/tanaylab/metacells-vignettes>`_.

References
----------

Please cite the references appropriately in case they are used:

.. [1] Ben-Kiki, O., Bercovich, A., Lifshitz, A. et al. Metacell-2: a divide-and-conquer metacell algorithm for scalable
   scRNA-seq analysis. Genome Biol 23, 100 (2022). https://doi.org/10.1186/s13059-022-02667-1

.. [2] Baran, Y., Bercovich, A., Sebe-Pedros, A. et al. MetaCell: analysis of single-cell RNA-seq data using K-nn graph
   partitions. Genome Biol 20, 206 (2019). `10.1186/s13059-019-1812-2 <https://doi.org/10.1186/s13059-019-1812-2>`_

License (MIT)
-------------

Copyright © 2020-2023 Weizmann Institute of Science

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
