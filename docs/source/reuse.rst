Reuse
=====

In addition to the top-level analysis function described above, the metacells package provides
lower-level building-block functions which can be directly used to either modify the default
algorithm, or be reused as steps in non-metacell analysis pipelines.

Rare Genes Modules Detection
----------------------------

A rare genes module is a group of correlated genes which are too rare and too weakly expressed to be
captured by the divide-and-conquer metacell algorithm. The default algorithm therefore identifies
such gene modules, identifies cells expressing the gene module, and groups them separately from the
overall population. This increases the overall algorithm sensitivity in detecting rare cell types.

Usage
.....

**TODO**
