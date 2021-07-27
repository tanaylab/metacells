Metacell Analysis
=================

Naively, scRNA_seq data is a set of cell profiles, where for each one, for each gene, we get a count
of the mRNA molecules that existed in the cell for that gene. This serves as an indicator of how
"expressed" or "active" the gene is.

As in any real world technology, the raw data may suffer from technical artifacts (counting the
molecules of two cells in one profile, counting the molecules from a ruptured cells, counting only
the molecules from the cell nucleus, etc.). This requires pruning the raw data to exclude such
artifacts.

The current technology scRNA-seq data is also very sparse (typically <<10% the RNA molecules are
counted). This introduces large sampling variance on top of the original signal, which itself
contains significant inherent biological noise.

Analyzing scRNA-seq data therefore requires processing the profiles in bulk. Classically, this has
been done by directly clustering the cells using various methods.

In contrast, the metacell approach groups together profiles of the "same" biological state into
groups of cells of the "same" biological state, with the *minimal* number of profiles needed for
computing robust statistics (in particular, mean gene expression). Each such group is a single
"metacell".

By summing profiles of cells of the "same" state together, each metacell greatly reduces the
sampling variance, and provides a more robust estimation of the transcription state. Note a metacell
is *not* a cell type (multiple metacells may belong to the same "type", or even have the "same"
state, if the data sufficiently over-samples this state). Also, a metacell is *not* a parametric
model of the cell state. It is merely a more robust description of some cell state.

The metacells should therefore be further analyzed as if they were cells, using additional methods
to classify cell types, detect cell trajectories and/or lineage, build parametric models for cell
behavior, etc. Using metacells as input for such analysis techniques should benefit both from the
more robust, less noisy input; and also from the (~100-fold) reduction in the number of cells to
analyze when dealing with large data (e.g. analyzing millions of individual cells).

Installation
============

In short: ``pip install metacells``. If you do not have ``sudo`` privileges, you might need to ``pip
install --user metacells``. Note that ``metacells`` requires many "heavy" dependencies, most notably
``numpy``, ``pandas``, ``scipy``, ``scanpy``, which ``pip`` should automatically install for you.

Note that ``metacells`` only runs natively on Linux and MacOS. To run it on a Windows computer, you
must activate `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl>`_ and
install ``metacells`` within it.

The metacells package contains extensions written in C++. The ``metacells`` distribution provides
pre-compiled Python wheels for both Linux and MacOS, so installing it using ``pip`` should not
require a C++ compilation step.

Note these pre-compiled wheels were built to use AVX2, and will not work on older CPUs which are
limited to SSE. Also, these wheels will not make use of any newer instructions (such as AVX512),
even if available. While these wheels may not the perfect match for the machine you are running on,
they are expected to work well for most machines.

To see the native capabilities of your machine, you can ``grep flags /proc/cpuinfo | head -1`` which
will give you a long list of supported CPU features in an arbitrary order, which may include
``sse``, ``avx2``, ``avx512``, etc. You can therefore simply ``grep avx2 /proc/cpuinfo | head -1``
to test whether AVX2 is/not supported by your machine.

You can avoid installing the pre-compiled wheel by running ``pip install metacells
--install-option='--native'``. This will force ``pip`` to compile the C++ extensions locally on your
machine, optimizing for its native capabilities, whatever these may be. However, this requires you
to have a C++ compiler installed (either ``g++`` or ``clang``), and it will take much longer to
complete the installation.

Vignettes
=========

The `generated documentation <https://metacells.readthedocs.io/en/latest>`_
contains the following vignettes:
`Basic Metacells Vignette <https://metacells.readthedocs.io/en/latest/Metacells_Vignette.html>`_,
`Manual Analysis Vignette <https://metacells.readthedocs.io/en/latest/Manual_Analysis.html>`_,
and
`Seurat Analysis Vignette <https://metacells.readthedocs.io/en/latest/Seurat_Analysis.html>`_.

You can also access their very latest version in the `Github repository
<https://github.com/tanaylab/metacells/tree/master/sphinx>`_.
