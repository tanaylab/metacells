"""
Collect
-------
"""

from hashlib import shake_128
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut
from metacells import __version__

__all__ = [
    "collect_metacells",
]


@ut.logged()
@ut.timed_call()
def collect_metacells(  # pylint: disable=too-many-statements
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    max_cell_size_quantile: Optional[float] = pr.max_cell_size_quantile,
    max_cell_size_factor: Optional[float] = pr.max_cell_size_factor,
    max_cell_size_noisy_quantile: Optional[float] = pr.max_cell_size_noisy_quantile,
    max_cell_size_noisy_factor: Optional[float] = pr.max_cell_size_noisy_factor,
    groups: Union[str, ut.Vector] = "metacell",
    name: str = "metacells",
    prefix: Optional[str] = "M",
    top_level: bool = True,
    _metacell_groups: bool = False,
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

    Unstructured (Scalar)
        ``outliers``
            The number of outlier cells (which were not assigned to any metacell).

    Variable (Gene) Annotations
        ``*``
            Every per-gene annotations of ``adata`` is copied to the result.

        ``noisy_gene`` (optional)
            Genes which may have inconsistent expression levels in the cells of the same metacell.

    Observations (Cell) Annotations
        ``grouped``
            The number of ("clean") cells grouped into each metacell.

    Also sets in the full ``adata``:

    Observations-Variables (Cell-Gene) Annotations
        ``total_umis``
            The total of all the UMIs for all the cells grouped into each metacell.

    Observations (Cell) Annotations
        ``total_umis``
            The total of all the UMIs for all the cells grouped into each metacell.

        ``metacell_name``
            The string name of the metacell, which is ``prefix`` (default: {prefix}) followed by the metacell index,
            followed by ``.checksum`` where the checksum is a 2 digits, reflecting the (names of the) set of cells
            grouped into each metacell. This keeps metacell names short, and provides protection against assigning
            per-metacell annotations from a different metacell computation to the new results (unless they are
            identical).

    **Computation Parameters**

    1. Compute the total size of each metacell using the ``cell_sizes`` and the ``groups``.

    2. In each metacell, compute each cell's capped size by invoking
       :py:func:`metacells.utilities.computation.capped_sizes` using the ``max_cell_size_quantile`` (default:
       {max_cell_size_quantile}), ``max_cell_size_factor`` (default: {max_cell_size_factor}).

    3. Scale the cells UMIs using these caps, if needed.

    4. Invoke :py:func:`metacells.utilities.computation.sum_groups` to sum the scaled cell UMIs into metacells.

    5. If there are no ``noisy_gene``, skip steps 6-9.

    6. In each metacell, compute each cell's noisy capped size by invoking
       :py:func:`metacells.utilities.computation.capped_sizes` using the ``max_cell_size_noisy_quantile`` (default:
       {max_cell_size_noisy_quantile}), ``max_cell_size_noisy_factor`` (default: {max_cell_size_noisy_factor}).

    7. Scale the cells noisy UMIs using these caps, if needed.

    8. Compute a geo-mean with a normalization factor of 1 UMI for the (scaled) noisy UMIs of the cells in each
       metacell.

    9. Finally, scale the geo-mean according to the ratio between the capped size from step 2 and the capped noisy size
       from step 6, and use the result for the noisy gene UMIs instead of the results from step 5.

    10. Compute the fraction of each gene effective UMIs out of the total effective metacell UMIs.

    11. Pass all relevant per-gene and per-cell annotations to the result.

    12. Set the ``metacell_name`` property of each cell in ``adata`` to the name of the group (metacell) it belongs to.
        Cells which do not belong to any metacell are assigned the metacell name ``Outliers``.
    """
    metacell_of_cells = ut.get_o_numpy(adata, groups, formatter=ut.groups_description).copy()
    metacell_of_cells[metacell_of_cells < 0] = -1
    outliers_count = np.sum(metacell_of_cells < 0)

    raw_cell_umis = ut.get_vo_proper(adata, what, layout="row_major")
    raw_cell_sizes = ut.sum_per(raw_cell_umis, per="row")
    ut.log_calc("raw_cell_sizes", raw_cell_sizes, formatter=ut.sizes_description)

    if max_cell_size_quantile is not None and max_cell_size_factor is not None:
        effective_cell_sizes = ut.capped_sizes(
            max_size_quantile=max_cell_size_quantile, max_size_factor=max_cell_size_factor, sizes=raw_cell_sizes
        )
        normal_cell_factors = effective_cell_sizes / raw_cell_sizes
        effective_cell_umis = ut.scale_by(raw_cell_umis, normal_cell_factors, by="row")
        ut.log_calc("effective_cell_sizes", effective_cell_sizes, formatter=ut.sizes_description)
        ut.log_calc("normal_cell_factors", normal_cell_factors, formatter=ut.sizes_description)
    else:
        effective_cell_umis = raw_cell_umis

    if (
        max_cell_size_noisy_quantile is not None
        and max_cell_size_noisy_factor is not None
        and ut.has_data(adata, "noisy_gene")
    ):
        noisy_genes_mask = ut.get_v_numpy(adata, "noisy_gene")
        if np.any(noisy_genes_mask):
            raw_cell_noisy_umis = ut.to_numpy_matrix(raw_cell_umis[:, noisy_genes_mask])

            noisy_cell_sizes = ut.capped_sizes(
                max_size_quantile=max_cell_size_noisy_quantile,
                max_size_factor=max_cell_size_noisy_factor,
                sizes=raw_cell_sizes,
            )
            noisy_cell_factors = noisy_cell_sizes / raw_cell_sizes
            ut.log_calc("noisy_cell_sizes", noisy_cell_sizes, formatter=ut.sizes_description)
            ut.log_calc("noisy_cell_factors", noisy_cell_factors, formatter=ut.sizes_description)

            effective_cell_noisy_umis = ut.mustbe_numpy_matrix(
                ut.scale_by(raw_cell_noisy_umis, noisy_cell_factors, by="row")
            )
            effective_cell_noisy_umis += 1
            effective_cell_log2_noisy_umis = np.log2(effective_cell_noisy_umis, out=effective_cell_noisy_umis)

            results = ut.sum_groups(effective_cell_log2_noisy_umis, metacell_of_cells, per="row")
            assert results is not None
            effective_metacell_sum_log2_noisy_umis, cells_of_metacells = results

            scale_of_metacells = np.reciprocal(cells_of_metacells, out=cells_of_metacells)
            effective_metacell_mean_log2_noisy_umis = ut.mustbe_numpy_matrix(
                ut.scale_by(effective_metacell_sum_log2_noisy_umis, scale_of_metacells, by="row")
            )
            effective_metacell_noisy_umis = np.exp2(
                effective_metacell_mean_log2_noisy_umis, out=effective_metacell_mean_log2_noisy_umis
            )
            effective_metacell_noisy_umis -= 1

            noisy_scale_factors = effective_cell_sizes / noisy_cell_sizes
            ut.log_calc("noisy_scale_factors", noisy_scale_factors, formatter=ut.sizes_description)

            effective_metacell_noisy_umis *= noisy_scale_factors
            if effective_cell_umis is raw_cell_umis:
                effective_cell_umis = raw_cell_umis.copy()
            effective_cell_umis[:, noisy_genes_mask] = effective_metacell_noisy_umis

    results = ut.sum_groups(effective_cell_umis, metacell_of_cells, per="row")
    assert results is not None
    effective_metacell_umis, cells_of_metacells = results
    ut.log_calc("cells_of_metacells", cells_of_metacells, formatter=ut.sizes_description)

    if _metacell_groups:
        mdata = AnnData(effective_metacell_umis)
    else:
        effective_metacell_sizes = ut.sum_per(effective_metacell_umis, per="row")
        ut.log_calc("effective_metacell_sizes", effective_metacell_sizes, formatter=ut.sizes_description)
        effective_metacell_scale = np.reciprocal(effective_metacell_sizes, out=effective_metacell_sizes)
        ut.log_calc("effective_metacell_scale", effective_metacell_scale, formatter=ut.sizes_description)
        effective_metacell_fractions = ut.scale_by(effective_metacell_umis, effective_metacell_scale, by="row")
        mdata = AnnData(effective_metacell_fractions)

    if top_level:
        ut.top_level(mdata)

    mdata.var_names = adata.var_names
    mdata.obs_names = _obs_names(prefix or "", ut.to_numpy_vector(adata.obs_names), metacell_of_cells)

    ut.set_name(mdata, name)

    zero_results = ut.sum_groups(
        raw_cell_umis,
        metacell_of_cells,
        per="row",
        transform=lambda values: ut.to_numpy_matrix(values == 0).astype("int32"),  # type: ignore
    )
    assert zero_results is not None
    zero_metacell_umis, _cells_of_metacells = zero_results
    ut.set_vo_data(mdata, "zeros", zero_metacell_umis)

    raw_results = ut.sum_groups(raw_cell_umis, metacell_of_cells, per="row")
    assert raw_results is not None
    total_metacell_umis, _cells_of_metacells = raw_results
    ut.set_vo_data(mdata, "total_umis", total_metacell_umis)

    raw_metacell_sizes = _metacell_sizes(raw_cell_sizes, metacell_of_cells)
    ut.set_o_data(mdata, "total_umis", raw_metacell_sizes, formatter=ut.sizes_description)
    ut.set_o_data(mdata, "grouped", cells_of_metacells, formatter=ut.sizes_description)

    metacell_names = np.array(["Outliers"] + list(mdata.obs_names), dtype="str")
    metacell_of_cells += 1
    metacell_name_of_cells = metacell_names[metacell_of_cells]
    ut.set_o_data(adata, "metacell_name", metacell_name_of_cells)

    for annotation_name in adata.var.keys():
        value_per_gene = ut.get_v_numpy(adata, annotation_name)
        ut.set_v_data(mdata, annotation_name, value_per_gene)

    if isinstance(groups, str) and ut.has_data(adata, "metacells_level"):
        tl.convey_obs_to_group(adata=adata, gdata=mdata, group=groups, property_name="metacell_level")

    ut.set_m_data(mdata, "outliers", outliers_count)
    ut.set_m_data(mdata, "metacells_algorithm", f"metacells.{__version__}")

    return mdata


# TODO: Replicated in metacells.tools.group
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
        checksum = int(hasher.hexdigest(16), 16) % 100  # pylint: disable=too-many-function-args
        name_of_groups.append(f"{prefix}{group_index}.{checksum:02d}")
    return name_of_groups


def _metacell_sizes(
    cell_sizes: ut.NumpyVector,
    metacell_of_cells: ut.NumpyVector,
) -> ut.NumpyVector:
    metacells_count = np.max(metacell_of_cells) + 1
    metacell_sizes = np.empty(metacells_count, dtype="float32")

    for metacell_index in range(metacells_count):
        metacell_sizes[metacell_index] = np.sum(cell_sizes[metacell_of_cells == metacell_index])

    return metacell_sizes
