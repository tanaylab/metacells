"""
Group
-----
"""

from hashlib import shake_128
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from anndata import AnnData  # type: ignore

import metacells.utilities as ut

__all__ = [
    "group_obs_data",
    "group_obs_annotation",
]


@ut.logged()
@ut.timed_call()
def group_obs_data(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    groups: Union[str, ut.Vector],
    name: Optional[str] = None,
    prefix: Optional[str] = None,
) -> Optional[AnnData]:
    """
    Compute new data which has the ``what`` (default: {what}) sum of the observations (cells) for
    each group.

    For example, having computed a metacell index for each cell, compute the per-metacell data
    for further analysis.

    If ``groups`` is a string, it is expected to be the name of a per-observation vector annotation.
    Otherwise it should be a vector. The group indices should be integers, where negative values
    indicate "no group" and non-negative values indicate the index of the group to which each
    observation (cell) belongs to.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    **Returns**

    An annotated data where each observation is the sum of the group of original observations
    (cells). Observations with a negative group index are discarded. If all observations are
    discarded, return ``None``.

    The new data will contain only:

    * A single observation for each group. The name of each observation will be the optional ``prefix``
      (default: {prefix}), followed by the group's index, followed by ``.`` and a 2-digit checksum of
      the grouped members.

    * An ``X`` member holding the summed-per-group data.

    * A new ``grouped`` per-observation data which counts, for each group, the number
      of grouped observations summed into it.

    If ``name`` is not specified, the data will be unnamed. Otherwise, if it starts with a ``.``, it
    will be appended to the current name (if any). Otherwise, ``name`` is the new name.
    """
    group_of_cells = ut.get_o_numpy(adata, groups, formatter=ut.groups_description)

    data = ut.get_vo_proper(adata, what, layout="row_major")
    results = ut.sum_groups(data, group_of_cells, per="row")
    if results is None:
        return None
    summed_data, cell_counts = results

    gdata = AnnData(summed_data)
    gdata.var_names = adata.var_names
    gdata.obs_names = _obs_names(prefix or "", ut.to_numpy_vector(adata.obs_names), group_of_cells)

    ut.set_name(gdata, ut.get_name(adata))
    ut.set_name(gdata, name)

    ut.set_o_data(gdata, "grouped", cell_counts, formatter=ut.sizes_description)

    return gdata


# TODO: Replicated in metacells.pipeline.collect
def _obs_names(prefix: str, name_of_members: ut.NumpyVector, group_of_members: ut.NumpyVector) -> List[str]:
    groups_count = np.max(group_of_members) + 1
    name_of_groups: List[str] = []
    prefix = prefix or ""
    for group_index in range(groups_count):
        groups_mask = group_of_members == group_index
        assert np.any(groups_mask)
        hasher = shake_128()
        for member_name in name_of_members[groups_mask]:
            hasher.update(member_name.encode("utf8"))
        checksum = int(hasher.hexdigest(16), 16) % 10
        name_of_groups.append(f"{prefix}{group_index}.{checksum:02d}")
    return name_of_groups


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def group_obs_annotation(
    adata: AnnData,
    gdata: AnnData,
    *,
    groups: Union[str, ut.Vector],
    name: str,
    formatter: Optional[Callable[[Any], Any]] = None,
    method: str = "majority",
    min_value_fraction: float = 0.5,
    conflict: Optional[Any] = None,
    inplace: bool = True,
) -> Optional[pd.Series]:
    """
    Transfer per-observation data from the per-observation (cell) ``adata`` to the
    per-group-of-observations (metacells) ``gdata``.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, and the
    ``gdata`` containing the per-metacells summed data.

    **Returns**

    Observations (Cell) Annotations
        ``<name>``
            The per-group-observation annotation computed based on the per-observation annotation.

    If ``inplace`` (default: {inplace}), this is written to the ``gdata``, and the function returns
    ``None``. Otherwise this is returned as a pandas series (indexed by the group observation
    names).

    **Computation Parameters**

    1. Iterate on all the observations (groups, metacells) in ``gdata``.

    2. Consider all the cells whose ``groups`` annotation maps them into this group.

    3. Consider all the ``name`` annotation values of these cells.

    4. Compute an annotation value for the whole group of cells using the ``method``. Supported
       methods are:

       ``unique``
            All the values of all the cells in the group are expected to be the same, use this
            unique value for the whole groups.

       ``majority``
            Use the most common value across all cells in the group as the value for the whole
            group. If this value doesn't have at least ``min_value_fraction`` (default:
            {min_value_fraction}) of the cells, use the ``conflict`` (default: {conflict}) value
            instead.
    """
    group_of_cells = ut.get_o_numpy(adata, groups, formatter=ut.groups_description)
    values_of_cells = ut.get_o_numpy(adata, name, formatter=formatter)

    value_of_groups = np.empty(gdata.n_obs, dtype=values_of_cells.dtype)

    assert method in ("unique", "majority")

    if method == "unique":
        with ut.timed_step(".unique"):
            grouped_mask = group_of_cells >= 0
            value_of_groups[group_of_cells[grouped_mask]] = values_of_cells[grouped_mask]

    else:
        assert method == "majority"
        with ut.timed_step(".majority"):
            for group_index in range(gdata.n_obs):
                cells_mask = group_of_cells == group_index
                cells_count = np.sum(cells_mask)
                assert cells_count > 0
                values_of_cells_of_group = values_of_cells[cells_mask]
                unique_values_of_group, unique_counts_of_group = np.unique(values_of_cells_of_group, return_counts=True)
                majority_index = np.argmax(unique_counts_of_group)
                majority_count = unique_counts_of_group[majority_index]
                if majority_count / cells_count < min_value_fraction:
                    value_of_groups[group_index] = conflict
                else:
                    majority_value = unique_values_of_group[majority_index]
                    value_of_groups[group_index] = majority_value

    if inplace:
        ut.set_o_data(gdata, name, value_of_groups)
        return None

    return ut.to_pandas_series(value_of_groups, index=gdata.obs_names)
