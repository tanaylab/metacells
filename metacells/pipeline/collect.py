"""
Collect
-------
"""

from typing import Optional
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    "collect_metacells",
]


@ut.logged()
@ut.timed_call()
def collect_metacells(
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    max_cell_size: Optional[float] = pr.max_cell_size,
    max_cell_size_factor: Optional[float] = pr.max_cell_size_factor,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    groups: Union[str, ut.Vector] = "metacell",
    name: str = "metacells",
    prefix: Optional[str] = "M",
    top_level: bool = True,
    random_seed: int,
) -> AnnData:
    """
    Collect computed metacells ``what`` (default: {what}) data.

    **Input**

    Annotated (presumably "clean") ``adata``, where the observations are cells and the variables are
    genes, and where ``what`` is a per-variable-per-observation matrix or the name of a
    per-variable-per-observation annotation containing such a matrix.

    **Returns**

    Annotated metacell data containing for each observation the sum of the data (by of the cells for
    each metacell, which contains the following annotations:

    Variable (Gene) Annotations
        ``*``
            Every per-gene annotations of ``adata`` is copied to the result.

    Observations (Cell) Annotations
        ``grouped``
            The number of ("clean") cells grouped into each metacell.

        ``metacell_level``
            Which level of processing generated the metacell (0 - from cells which were never
            outliers; 1 - from cells which were outliers in the 1st step, etc.).

    Also sets in the full ``adata``:

    Observations (Cell) Annotations
        ``metacell_name``
            The string name of the metacell, which is ``prefix`` (default: {prefix}) followed by the metacell index,
            followed by ``.checksum`` where the checksum is a 2 digits, reflecting the (names of the) set of cells
            grouped into each metacell. This keeps metacell names short, and provides protection against assigning
            per-metacell annotations from a different metacell computation to the new results (unless they are
            identical).

    **Computation Parameters**

    1. In each metacell, compute each cell's capped size by invoking
       :py:func:`metacells.utilities.computation.capped_sizes` using the ``max_cell_size`` (default: {max_cell_size}),
       ``max_cell_size_factor`` (default: {max_cell_size_factor}) and ``cell_sizes`` (default: {cell_sizes}).

    2. Downsample the cell's data using these capped, if needed, using the ``random_seed``.

    3. Invoke :py:func:`metacells.tools.group.group_obs_data` to sum the cells into
       metacells.

    4. Pass all relevant per-gene and per-cell annotations to the result.

    5. Set the ``metacell_name`` property of each cell in ``adata`` to the name of the group (metacell) it belongs to.
       Cells which do not belong to any metacell are assigned the metacell name ``Outliers``.
    """
    cell_sizes = ut.maybe_o_numpy(adata, cell_sizes, formatter=ut.sizes_description)
    cell_samples = _cell_samples(
        adata, max_cell_size=max_cell_size, max_cell_size_factor=max_cell_size_factor, cell_sizes=cell_sizes
    )

    if cell_samples is not None:
        data = ut.get_vo_proper(adata, what, layout="row_major")
        what = ut.downsample_matrix(data, per="row", samples=cell_samples, random_seed=random_seed)

    mdata = tl.group_obs_data(adata, what, groups=groups, name=name, prefix=prefix)
    assert mdata is not None
    if top_level:
        ut.top_level(mdata)

    metacell_names = np.array(["Outliers"] + list(mdata.obs_names), dtype="str")
    metacell_of_cells = ut.get_o_numpy(adata, groups, formatter=ut.groups_description).copy()
    metacell_of_cells[metacell_of_cells < 0] = -1
    metacell_of_cells += 1
    metacell_name_of_cells = metacell_names[metacell_of_cells]
    ut.set_o_data(adata, "metacell_name", metacell_name_of_cells)

    for annotation_name in adata.var.keys():
        value_per_gene = ut.get_v_numpy(adata, annotation_name)
        ut.set_v_data(mdata, annotation_name, value_per_gene)

    if ut.has_data(adata, "metacell_level"):
        tl.group_obs_annotation(
            adata, mdata, groups=groups, formatter=ut.sizes_description, name="metacell_level", method="unique"
        )

    return mdata


def _cell_samples(
    adata: AnnData,
    *,
    max_cell_size: Optional[float] = pr.max_cell_size,
    max_cell_size_factor: Optional[float] = pr.max_cell_size_factor,
    cell_sizes: Optional[ut.NumpyVector],
) -> Optional[ut.NumpyVector]:
    if cell_sizes is None or (max_cell_size is None and max_cell_size_factor is None):
        return None

    metacell_of_cells = ut.get_o_numpy(adata, "metacell")
    metacells_count = np.max(metacell_of_cells) + 1
    capped_sizes = cell_sizes.copy()

    for metacell in range(metacells_count):
        cell_indices = np.where(metacell_of_cells == metacell)[0]
        capped_sizes[cell_indices] = ut.capped_sizes(
            max_size=max_cell_size, max_size_factor=max_cell_size_factor, sizes=cell_sizes[cell_indices]
        )

    capped_sizes[capped_sizes >= cell_sizes] = -1
    return capped_sizes
