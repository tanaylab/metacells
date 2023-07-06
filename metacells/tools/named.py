"""
Named
-----
"""

from re import Pattern
from typing import Collection
from typing import Optional
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore

import metacells.utilities as ut

__all__ = [
    "find_named_genes",
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def find_named_genes(  # pylint: disable=too-many-branches
    adata: AnnData,
    *,
    name_property: Optional[str] = None,
    names: Optional[Collection[str]] = None,
    patterns: Optional[Collection[Union[str, Pattern]]] = None,
    to: Optional[str] = None,
    invert: bool = False,
    op: str = "set",
) -> Optional[ut.PandasSeries]:
    """
    Find genes by their (case-insensitive) name.

    This computes a mask of all the genes whose name appears in ``names`` or matches any of the ``patterns``. If
    ``invert`` (default: {invert}), invert the resulting mask.

    Depending on ``op``, this will ``set`` a (compute a brand new) mask, ``add`` the result to a mask (which must
    exist), or ``remove`` genes from a mask (which must exist).

    If ``name_property`` is specified the mask will be based on this per-variable (gene) property.

    If ``to`` (default: {to}) is specified, this is stored as a per-variable (gene) annotation with that name, and
    returns ``None``. This is useful to fill gene masks such as ``excluded_genes`` (genes which should be excluded from
    the rest of the processing), ``lateral_genes`` (genes which must not be selected for metacell computation) and
    ``noisy_genes`` (genes which are given more leeway when computing deviant cells).

    Otherwise, it returns it as a pandas series (indexed by the variable, that is gene, names).
    """
    assert op in ("set", "add", "remove")
    if op in ("add", "remove"):
        assert to is not None
        base_mask = ut.get_v_numpy(adata, to)
    else:
        base_mask = np.zeros(adata.n_vars, dtype="bool")

    if name_property is None:
        var_names = adata.var_names
    else:
        var_names = ut.get_v_numpy(adata, name_property)

    if names is None or len(names) == 0:
        names_mask = np.zeros(adata.n_vars, dtype="bool")
    else:
        lower_names_set = {name.lower() for name in names}
        names_mask = np.array([name.lower() in lower_names_set for name in var_names])  #

    if patterns is None or len(patterns) == 0:
        patterns_mask = np.zeros(adata.n_vars, dtype="bool")
    else:
        patterns_mask = ut.patterns_matches(patterns, var_names)

    genes_mask = names_mask | patterns_mask

    if invert:
        genes_mask = ~genes_mask

    if op == "add":
        result_mask = base_mask | genes_mask
    elif op == "remove":
        result_mask = base_mask & ~genes_mask
    else:
        assert op == "set"
        result_mask = genes_mask

    if to is not None:
        ut.set_v_data(adata, to, result_mask)
        return None

    ut.log_return("named_genes", result_mask)
    return ut.to_pandas_series(result_mask, index=adata.var_names)
