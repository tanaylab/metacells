"""
Convey
------
"""

from typing import Any
from typing import Callable
from typing import Optional

import numpy as np
from anndata import AnnData  # type: ignore

import metacells.utilities as ut

__all__ = [
    "convey_group_to_obs",
    "convey_obs_to_obs",
    "convey_obs_to_group",
    "convey_obs_fractions_to_group",
    "convey_obs_obs_to_group_group",
]


@ut.logged()
@ut.timed_call()
def convey_group_to_obs(
    *,
    adata: AnnData,
    gdata: AnnData,
    group: str = "metacell",
    property_name: str,
    formatter: Optional[Callable[[Any], Any]] = None,
    to_property_name: Optional[str] = None,
    default: Any = None,
) -> None:
    """
    Project the value of a property from per-group data to per-observation data.

    The input annotated ``gdata`` is expected to contain a per-observation (group) annotation named
    ``property_name``. The input annotated ``adata`` is expected to contain a per-observation
    annotation named ``group`` which identifies the group each observation (cell) belongs to.

    This will generate a new per-observation (cell) annotation in ``adata``, named
    ``to_property_name`` (by default, the same as ``property_name``), containing the value of the
    property for the group it belongs to. If the ``group`` annotation contains a negative number
    instead of a valid group index, the ``default`` value is used.
    """
    if to_property_name is None:
        to_property_name = property_name

    group_of_obs = ut.get_o_numpy(adata, group, formatter=ut.groups_description)
    property_of_group = ut.get_o_numpy(gdata, property_name, formatter=formatter)
    property_of_obs = np.array([default if group < 0 else property_of_group[group] for group in group_of_obs])
    ut.set_o_data(adata, to_property_name, property_of_obs)


@ut.logged()
@ut.timed_call()
def convey_obs_to_obs(
    *,
    adata: AnnData,
    bdata: AnnData,
    property_name: str,
    formatter: Optional[Callable[[Any], Any]] = None,
    to_property_name: Optional[str] = None,
    default: Any = None,
) -> None:
    """
    Project the value of a property from one annotated data to another.

    The observation names are expected to be compatible between ``adata`` and ``bdata``. The
    annotated ``adata`` is expected to contain a per-observation (cell) annotation named
    ``property_name``.

    This will generate a new per-observation (cell) annotation in ``bdata``, named
    ``to_property_name`` (by default, the same as ``property_name``), containing the value of the
    observation with the same name in ``adata``. If no such observation exists, the ``default``
    value is used.
    """
    if to_property_name is None:
        to_property_name = property_name

    property_of_from = ut.get_o_numpy(adata, property_name, formatter=formatter)
    property_of_name = {name: property_of_from[index] for index, name in enumerate(adata.obs_names)}
    property_of_to = np.array([property_of_name.get(name, default) for name in bdata.obs_names])
    ut.set_o_data(bdata, to_property_name, property_of_to)


@ut.logged()
@ut.timed_call()
def convey_obs_to_group(
    *,
    adata: AnnData,
    gdata: AnnData,
    group: str = "metacell",
    property_name: str,
    formatter: Optional[Callable[[Any], Any]] = None,
    to_property_name: Optional[str] = None,
    method: Callable[[ut.Vector], Any] = ut.most_frequent,
) -> None:
    """
    Project the value of a property from per-observation data to per-group data.

    The input annotated ``adata`` is expected to contain a per-observation (cell) annotation named
    ``property_name`` and also a per-observation annotation named ``group`` which identifies the
    group each observation (cell) belongs to, which must be an integer.

    This will generate a new per-observation (group) annotation in ``gdata``, named
    ``to_property_name`` (by default, the same as ``property_name``), containing the aggregated
    value of the property of all the observations (cells) that belong to the group.

    The aggregation method (by default, :py:func:`metacells.utilities.computation.most_frequent`) is
    any function taking an array of values and returning a single value.
    """
    if to_property_name is None:
        to_property_name = property_name

    group_of_obs = ut.get_o_numpy(adata, group, formatter=ut.groups_description)
    property_of_obs = ut.get_o_numpy(adata, property_name, formatter=formatter)
    assert gdata.n_obs == (np.max(group_of_obs) + 1)
    property_of_group = np.array([method(property_of_obs[group_of_obs == group]) for group in range(gdata.n_obs)])
    ut.set_o_data(gdata, to_property_name, property_of_group)


@ut.logged()
@ut.timed_call()
def convey_obs_fractions_to_group(
    *,
    adata: AnnData,
    gdata: AnnData,
    group: str = "metacell",
    property_name: str,
    formatter: Optional[Callable[[Any], Any]] = None,
    to_property_name: Optional[str] = None,
) -> None:
    """
    Similar to ``convey_obs_to_group``, but create a per-metacell property for each value of the
    per-cell property, storing the fraction of cells of the metacell that had that value.

    The input annotated ``adata`` is expected to contain a per-observation (cell) annotation named
    ``property_name`` and also a per-observation annotation named ``group`` which identifies the
    group each observation (cell) belongs to, which must be an integer.

    This will generate multiple new per-observation (group) annotation in ``gdata``, named
    ``<to_property_name>_fraction_of_<value>`` (by default, the ``to_property_name`` is the same as ``property_name``),
    containing the fraction of the metacell cells containing the specific property value.
    """
    if to_property_name is None:
        to_property_name = property_name

    property_of_obs = ut.get_o_numpy(adata, property_name, formatter=formatter)
    unique_values = sorted(np.unique(property_of_obs))
    for value in unique_values:
        convey_obs_to_group(
            adata=adata,
            gdata=gdata,
            group=group,
            property_name=property_name,
            to_property_name=f"{to_property_name}_fraction_of_{value}",
            method=ut.fraction_of_grouped(value),
        )


@ut.logged()
@ut.timed_call()
def convey_obs_obs_to_group_group(
    *,
    adata: AnnData,
    gdata: AnnData,
    group: str = "metacell",
    property_name: str,
    formatter: Optional[Callable[[Any], Any]] = None,
    to_property_name: Optional[str] = None,
    method: Callable[[ut.Matrix], Any] = ut.nanmean_matrix,
) -> None:
    """
    Project the value of a property from per-observation-per-observation data to per-group-per-group
    data.

    The input annotated ``adata`` is expected to contain a per-observation-per-observation (cell)
    annotation named ``property_name`` and also a per-observation annotation named ``group`` which
    identifies the group each observation (cell) belongs to, which must be an integer.

    This will generate a new per-observation-per-observation (group) annotation in ``gdata``, named
    ``to_property_name`` (by default, the same as ``property_name``), containing the aggregated
    value of the property of all the observations (cells) that belong to the group.

    The aggregation method (by default, :py:func:`metacells.utilities.computation.nanmean_matrix`)
    is any function taking a matrix of values and returning a single value.
    """
    if to_property_name is None:
        to_property_name = property_name

    group_of_obs = ut.get_o_numpy(adata, group, formatter=ut.groups_description)
    property_of_obs_obs = ut.get_oo_proper(adata, property_name, formatter=formatter)
    assert gdata.n_obs == (np.max(group_of_obs) + 1)
    property_of_group_group = np.empty((gdata.n_obs, gdata.n_obs), dtype=ut.shaped_dtype(property_of_obs_obs))

    # TODO: This is a slow implementation.
    for row_group in range(gdata.n_obs):
        row_cells = np.where(group_of_obs == row_group)[0]
        assert len(row_cells) > 0

        for column_group in range(gdata.n_obs):
            column_cells = np.where(group_of_obs == column_group)[0]
            assert len(column_cells) > 0

            property_of_group_group[row_group, column_group] = method(
                property_of_obs_obs[row_cells, :][:, column_cells]
            )

    ut.set_oo_data(gdata, to_property_name, property_of_group_group)
