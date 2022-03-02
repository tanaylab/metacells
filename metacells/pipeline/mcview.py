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

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

from .umap import compute_umap_by_features

__all__ = [
    "compute_for_mcview",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_for_mcview(
    what: Union[str, ut.Matrix] = "__x__",
    *,
    adata: AnnData,
    gdata: AnnData,
    group: Union[str, ut.Vector] = "metacell",
    random_seed: int = pr.random_seed,
    compute_umap_by_features_2: Optional[Dict[str, Any]] = {},
    compute_umap_by_features_3: Optional[Dict[str, Any]] = {},
    compute_inner_fold_factors: Optional[Dict[str, Any]] = {},
    compute_outliers_matches: Optional[Dict[str, Any]] = {},
    compute_outliers_fold_factors: Optional[Dict[str, Any]] = {},
    compute_deviant_fold_factors: Optional[Dict[str, Any]] = {},
    compute_var_var_similarity: Optional[Dict[str, Any]] = dict(top=50, bottom=50),
    find_metacells_significant_genes: Optional[Dict[str, Any]] = {},
) -> Optional[AnnData]:
    """
    Compute metacell analysis in preparation for exporting the data to MCView.

    This simply invokes a series of tools (which you can also invoke independently), using the default parameters,
    except for the ``group`` (default: {group}) which allows applying this to any clustering (not only metacells) and
    the ``random_seed`` (default: {random_seed}) needed for reproducibility.

    If specific tool parameters need to be specified, you can pass them as a dictionary using the specific tool name
    (e.g., ``compute_umap_by_features_2 = dict(spread = 0.5)``. If this parameter is set to ``None``, running the tool
    is skipped.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    In addition, ``gdata`` is assumed to have one observation for each group, and use the same
    genes as ``adata``.

    **Returns**

    Sets many properties in ``gdata``, see below. If ``compute_outliers_fold_factors`` is not ``None``, return a new
    annotated data containing just the outliers with their ``most_similar_fold`` sparse layer. Otherwise, returns
    ``None``.

    **Computation Parameters**

    0. Use the default parameters for each tool, unless a parameter with the same name provides specific parameter
       overrides for it. In addition, the special parameters ``group`` (default: {group}) and ``random_seed`` (default:
       {random_seed}) and the ``reproducible`` flag derived from it (true if the seed is not zero) are automatically
       passed to all relevant tools.

    1. Computes UMAP projections by invoking :py:func:`metacells.pipeline.compute_umap_by_features`. This is done twice,
       once with ``dimensions=2`` for visualization and once with ``dimensions=3`` to capture more of the manifold
       structure (used to automatically generate cluster colors). Therefore in this case there are two dictionary
       parameters ``compute_umap_by_features_2`` and ``compute_umap_by_features_3``.

    2. Compute for each gene and for each metacell the fold factor between the metacell cells using
       :py:func:`metacells.tools.compute_inner_fold_factors`.

    3. Compute for each outlier cell the "most similar" metacecell for it using
       :py:func:`metacells.tools.compute_outliers_matches`.

    4. Compute for each metacell the fold factor between the metacell and the outliers most similar to it using
       :py:func:`metacells.tools.compute_deviant_fold_factors`.

    5. Compute the gene-gene (variable-variable) similarity matrix. Note by default this will use
       {compute_var_var_similarity} which aren't the normal defaults for ``compute_var_var_similarity``, in order to
       keep just the top correlated genes and bottom (anti-)correlated genes for each gene. Otherwise you will get a
       dense matrix of ~X0K by ~X0K entries, which typically isn't what you want.

    6. Compute the significant metacell genes using :py:func:`metacells.tools.find_metacells_significant_genes`.
    """
    reproducible = random_seed != 0
    if compute_umap_by_features_2 is not None:
        compute_umap_by_features(gdata, what, dimensions=2, random_seed=random_seed, **compute_umap_by_features_2)
    if compute_umap_by_features_3 is not None:
        compute_umap_by_features(gdata, what, dimensions=3, random_seed=random_seed, **compute_umap_by_features_3)
    if compute_outliers_matches is not None:
        tl.compute_outliers_matches(
            what, adata=adata, gdata=gdata, group=group, reproducible=reproducible, **compute_outliers_matches
        )
    if compute_inner_fold_factors is not None:
        tl.compute_inner_fold_factors(what, adata=adata, gdata=gdata, group=group, **compute_inner_fold_factors)
    if compute_deviant_fold_factors is not None:
        tl.compute_deviant_fold_factors(what, adata=adata, gdata=gdata, group=group, **compute_deviant_fold_factors)
    if compute_var_var_similarity is not None:
        tl.compute_var_var_similarity(gdata, what, **compute_var_var_similarity)
    if find_metacells_significant_genes is not None:
        tl.find_metacells_significant_genes(gdata, what, **find_metacells_significant_genes)

    if compute_outliers_fold_factors is None:
        return None
    group_of_cells = ut.get_o_numpy(adata, group)
    outliers_mask = group_of_cells < 0
    if np.sum(outliers_mask) == 0:
        return None
    odata = ut.slice(adata, name=".outliers", obs=outliers_mask)
    tl.compute_outliers_fold_factors(what, adata=odata, gdata=gdata, **compute_outliers_fold_factors)
    return odata
