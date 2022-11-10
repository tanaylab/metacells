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
    name: str = "metacells",
    top_level: bool = True,
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
        ``excluded_gene``
            A mask of the genes which were excluded by name.

        ``clean_gene``
            A boolean mask of the clean genes.

        ``lateral_gene``
            A boolean mask of genes which are lateral from being chosen as "feature" genes based
            on their name. This is ``False`` for non-"clean" genes.

        ``bystander_gene``
            A boolean mask of genes which are lateral and are also ignored when computing deviant
            (outlier) cells. This is ``False`` for non-"clean" genes.

        If directly computing metecalls:

        ``feature``
            A boolean mask of the "feature" genes. This is ``False`` for non-"clean" genes.

        If using divide-and-conquer:

        ``pre_feature``, ``feature``
            The number of times the gene was used as a feature when computing the preliminary and
            final metacells. This is zero for non-"clean" genes.

    Observations (Cell) Annotations
        ``grouped``
            The number of ("clean") cells grouped into each metacell.

        ``pile``
            The index of the pile used to compute the metacell each cell was assigned to to. This is
            ``-1`` for non-"clean" cells.

        ``candidate``
            The index of the candidate metacell each cell was assigned to to. This is ``-1`` for
            non-"clean" cells.

    Also sets all relevant annotations in the full data based on their value in the clean data, with
    appropriate defaults for non-"clean" data.

    **Computation Parameters**

    1. In each metacell, compute each cell's scale factors by invoking
       :py:func:`metacells.utilities.computation.capped_sizes` using the ``max_cell_size`` (default: {max_cell_size}),
       ``max_cell_size_factor`` (default: {max_cell_size_factor}) and ``cell_sizes`` (default: {cell_sizes}).

    2. Scale the cell's data using these factors, if needed.

    3. Invoke :py:func:`metacells.tools.group.group_obs_data` to sum the cells into
       metacells.

    4. Pass all relevant per-gene and per-cell annotations to the result.
    """
    cell_sizes = ut.maybe_o_numpy(adata, cell_sizes, formatter=ut.sizes_description)
    cell_scale_factors = _cell_scale_factors(
        adata, max_cell_size=max_cell_size, max_cell_size_factor=max_cell_size_factor, cell_sizes=cell_sizes
    )

    if cell_scale_factors is not None:
        data = ut.get_vo_proper(adata, what, layout="row_major")
        what = ut.scale_by(data, cell_scale_factors, by="row")

    mdata = tl.group_obs_data(adata, what, groups="metacell", name=name)
    assert mdata is not None
    if top_level:
        ut.top_level(mdata)

    for annotation_name in (
        "excluded_gene",
        "clean_gene",
        "lateral_gene",
        "bystander_gene",
        "pre_feature_gene",
        "feature_gene",
    ):
        if not ut.has_data(adata, annotation_name):
            continue
        value_per_gene = ut.get_v_numpy(adata, annotation_name, formatter=ut.mask_description)
        ut.set_v_data(mdata, annotation_name, value_per_gene, formatter=ut.mask_description)

    for annotation_name in ("pile", "candidate"):
        if ut.has_data(adata, annotation_name):
            tl.group_obs_annotation(
                adata, mdata, groups="metacell", formatter=ut.groups_description, name=annotation_name, method="unique"
            )

    return mdata


def _cell_scale_factors(
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

    cell_scale_factors = capped_sizes
    cell_scale_factors /= cell_sizes
    assert np.min(cell_scale_factors) > 0.0
    assert np.max(cell_scale_factors) <= 1.0
    return cell_scale_factors
