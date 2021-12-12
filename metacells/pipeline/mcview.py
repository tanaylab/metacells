"""
MCView
------

Compute metacell analysis in preparation for exporting the data to MCView.
"""

from typing import Any
from typing import Dict
from typing import Union

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
    compute_umap_by_features_2: Dict[str, Any] = {},
    compute_umap_by_features_3: Dict[str, Any] = {},
    compute_inner_fold_factors: Dict[str, Any] = {},
    compute_outliers_matches: Dict[str, Any] = {},
    compute_deviant_fold_factors: Dict[str, Any] = {},
) -> None:
    """
    Compute metacell analysis in preparation for exporting the data to MCView.

    This simply invokes a series of tools (which you can also invoke independently), using the default parameters,
    except for the ``group`` (default: {group}) which allows applying this to any clustering (not only metacells) and
    the ``random_seed`` (default: {random_seed}) needed for reproducibility.

    If specific tool parameters need to be specified, you can pass them as a dictionary using the specific tool name
    (e.g., ``compute_umap_by_features_2 = dict(spread = 0.5)``.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    In addition, ``gdata`` is assumed to have one observation for each group, and use the same
    genes as ``adata``.

    **Returns**

    **Computation Parameters**

    0. Use the default parameters for each tool, unless a parameter with the same name provides specific parameter
       overrides for it. In addition, the special parameters ``group`` (default: {group}) and ``random_seed`` (default:
       {random_seed}) and the ``reproducible`` flag derived from it (true if the seed is not zero) are automatically
       passed to all relevant tools.

    1. Computes UMAP projections by invoking :py:func:`metacells.pipeline.compute_umap_by_features`. This is done twice,
       once with `dimensions=2` for visualization and once with `dimensions=3` to capture more of the manifold structure
       (used to automatically generate cluster colors). Therefore in this case there are two dictionary parameters
       ``compute_umap_by_features_2`` and ``compute_umap_by_features_3``.

    2. Compute for each gene and for each metacell the fold factor between the metacell cells using
       :py:func:`metacells.tools.compute_inner_fold_factors`.

    3. Compute for each outlier cell the "most similar" metacecell for it using
       :py:func:`metacells.tools.compute_outliers_matches`.

    4. Compute for each metacell the fold factor between the metacell and the outliers most similar to it using
       :py:func:`metacells.tools.compute_deviant_fold_factors`.
    """
    reproducible = random_seed != 0
    compute_umap_by_features(adata, what, dimensions=2, random_seed=random_seed, **compute_umap_by_features_2)
    compute_umap_by_features(adata, what, dimensions=3, random_seed=random_seed, **compute_umap_by_features_3)
    tl.compute_outliers_matches(
        what, adata=adata, gdata=gdata, group=group, reproducible=reproducible, **compute_outliers_matches
    )
    tl.compute_inner_fold_factors(what, adata=adata, gdata=gdata, group=group, **compute_inner_fold_factors)
    tl.compute_deviant_fold_factors(what, adata=adata, gdata=gdata, group=group, **compute_deviant_fold_factors)
