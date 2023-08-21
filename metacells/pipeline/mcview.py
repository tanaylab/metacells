"""
MCView
------

Compute metacell analysis in preparation for exporting the data to MCView.
"""

from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore

import metacells.tools as tl
import metacells.utilities as ut

from .umap import compute_umap_by_markers

__all__ = [
    "compute_for_mcview",
]


# pylint: disable=dangerous-default-value


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_for_mcview(
    what: Union[str, ut.Matrix] = "__x__",
    *,
    adata: AnnData,
    gdata: AnnData,
    group: Union[str, ut.Vector] = "metacell",
    find_metacells_marker_genes: Optional[Dict[str, Any]] = {},
    compute_umap_by_markers_2: Optional[Dict[str, Any]] = {},
    compute_umap_by_markers_3: Optional[Dict[str, Any]] = {},
    compute_outliers_matches: Optional[Dict[str, Any]] = {},
    compute_deviant_folds: Optional[Dict[str, Any]] = {},
    compute_inner_folds: Optional[Dict[str, Any]] = {},
    count_significant_inner_folds: Optional[Dict[str, Any]] = {},
    compute_stdev_logs: Optional[Dict[str, Any]] = {},
    compute_var_var_similarity: Optional[Dict[str, Any]] = {"top": 50, "bottom": 50},
    random_seed: int,
) -> None:
    """
    Compute metacell analysis in preparation for exporting the data to MCView.

    This simply invokes a series of tools (which you can also invoke independently), using the default parameters,
    except for the ``group`` (default: {group}) which allows applying this to any clustering (not only metacells) and
    the ``random_seed`` (non-zero for reproducibility).

    If specific tool parameters need to be specified, you can pass them as a dictionary using the specific tool name
    (e.g., ``compute_umap_by_markers_2 = dict(spread = 0.5)``. If this parameter is set to ``None``, running the tool
    is skipped.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    In addition, ``gdata`` is assumed to have one observation for each group, and use the same
    genes as ``adata``.

    **Returns**

    Sets many properties in ``gdata`` and some in ``adata``, see below.

    **Computation Parameters**

    0. Use the default parameters for each tool, unless a parameter with the same name provides specific parameter
       overrides for it. In addition, the special parameters ``group`` (default: {group}) and ``random_seed``,
       and the ``reproducible`` flag derived from it (true if the seed is not zero) are automatically
       passed to all relevant tools.

    1. Compute the "marker" metacell genes using :py:func:`metacells.tools.high.find_metacells_marker_genes`.

    2. Computes UMAP projections by invoking :py:func:`metacells.pipeline.umap.compute_umap_by_markers`. This is done
       twice, once with ``dimensions=2`` for visualization and once with ``dimensions=3`` to capture more of the
       manifold structure (used to automatically generate cluster colors). Therefore in this case there are two
       dictionary parameters ``compute_umap_by_markers_2`` and ``compute_umap_by_markers_3``.

    3. Compute for each outlier cell the "most similar" metacecell for it using
       :py:func:`metacells.tools.quality.compute_outliers_matches`.

    4. Compute for each gene and cell the fold factor between the gene's expression and that of the metacell
       it belongs to (or, for outliers, the one it is most similar to) using
       :py:func:`metacells.tools.quality.compute_deviant_folds`.

    5. Compute for each gene for each metacell the maximal of the above using
       :py:func:`metacells.tools.quality.compute_inner_folds`.

    6. Compute for each gene the number of metacells where the above is high using
       :py:func:`metacells.tools.quality.count_significant_inner_folds`.

    7. Compute for each gene for each metacell the standard deviation of the log (base 2) of the fractions of each gene
       across the cells of the metacell using :py:func:`metacells.tools.quality.compute_stdev_logs`.

    8. Compute the gene-gene (variable-variable) similarity matrix. Note by default this will use
       {compute_var_var_similarity} which aren't the normal defaults for ``compute_var_var_similarity``, in order to
       keep just the top correlated genes and bottom (anti-)correlated genes for each gene. Otherwise you will get a
       dense matrix of ~X0K by ~X0K entries, which typically isn't what you want.
    """
    reproducible = random_seed != 0

    ut.set_m_data(gdata, "mcview_format", "1.0")

    metacell_of_cells = ut.get_o_numpy(adata, group)
    ut.set_m_data(gdata, "outliers", np.sum(metacell_of_cells < 0))

    step_is_not_none = [
        find_metacells_marker_genes is not None,
        compute_umap_by_markers_3 is not None,
        compute_umap_by_markers_2 is not None,
        compute_outliers_matches is not None,
        compute_deviant_folds is not None,
        compute_inner_folds is not None,
        count_significant_inner_folds is not None,
        compute_stdev_logs is not None,
        compute_var_var_similarity is not None,
    ]
    steps_count = len([is_not_none for is_not_none in step_is_not_none if is_not_none])

    step_time = 1 / steps_count if ut.has_progress_bar() else None

    if find_metacells_marker_genes is not None:
        with ut.progress_bar_slice(step_time):
            tl.find_metacells_marker_genes(gdata, what, **find_metacells_marker_genes)

    if compute_umap_by_markers_3 is not None:
        with ut.progress_bar_slice(step_time):
            compute_umap_by_markers(gdata, what, dimensions=3, random_seed=random_seed, **compute_umap_by_markers_3)

    if compute_umap_by_markers_2 is not None:
        with ut.progress_bar_slice(step_time):
            compute_umap_by_markers(gdata, what, dimensions=2, random_seed=random_seed, **compute_umap_by_markers_2)

    if compute_outliers_matches is not None:
        with ut.progress_bar_slice(step_time):
            tl.compute_outliers_matches(
                what, adata=adata, gdata=gdata, group=group, reproducible=reproducible, **compute_outliers_matches
            )

    if compute_deviant_folds is not None:
        with ut.progress_bar_slice(step_time):
            tl.compute_deviant_folds(what, adata=adata, gdata=gdata, group=group, **compute_deviant_folds)

    if compute_inner_folds is not None:
        with ut.progress_bar_slice(step_time):
            tl.compute_inner_folds(adata=adata, gdata=gdata, group=group)

    if count_significant_inner_folds is not None:
        with ut.progress_bar_slice(step_time):
            tl.count_significant_inner_folds(gdata, **count_significant_inner_folds)

    if compute_stdev_logs is not None:
        with ut.progress_bar_slice(step_time):
            tl.compute_stdev_logs(what, adata=adata, gdata=gdata, group=group, **compute_stdev_logs)

    if compute_var_var_similarity is not None:
        with ut.progress_bar_slice(step_time):
            tl.compute_var_var_similarity(gdata, what, reproducible=reproducible, **compute_var_var_similarity)
