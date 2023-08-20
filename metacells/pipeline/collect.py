"""
Collect
-------
"""

from hashlib import shake_128
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import scipy.sparse as sp  # type: ignore
import scipy.stats as ss  # type: ignore
from anndata import AnnData  # type: ignore

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut
from metacells import __version__  # pylint: disable=cyclic-import

__all__ = [
    "collect_metacells",
]


@ut.logged()
@ut.timed_call()
def collect_metacells(  # pylint: disable=too-many-statements
    adata: AnnData,
    what: Union[str, ut.Matrix] = "__x__",
    *,
    metacell_geo_mean: bool = pr.metacell_geo_mean,
    metacell_umis_regularization: float = pr.metacell_umis_regularization,
    zeros_cell_size_quantile: float = pr.zeros_cell_size_quantile,
    groups: Union[str, ut.Vector] = "metacell",
    name: str = "metacells",
    prefix: Optional[str] = "M",
    top_level: bool = True,
    _metacell_groups: bool = False,
    random_seed: int,
) -> AnnData:
    """
    Collect computed metacells ``what`` (default: {what}) data.

    **Input**

    Annotated (presumably "clean") ``adata``, where the observations are cells and the variables are genes, and where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation annotation
    containing such a matrix.

    **Returns**

    Annotated metacell data containing for each observation (metacell) for each variable (gene) the fraction of the gene
    in the metacell, and the following annotations:

    Unstructured (Scalar)
        ``outliers``
            The number of outlier cells (which were not assigned to any metacell).

        ``metacells_algorithm``
            The version of the algorithm generating the metacell (based on the package version number).

    Variable (Gene) Annotations
        ``*``
            Every per-gene annotations of ``adata`` is copied to the result.

    Observations (Metacell) Annotations
        ``grouped``
            The number of ("clean") cells grouped into each metacell.

        ``total_umis``
            The total of all the UMIs of all the genes of all the cells grouped into the metacell.

    Observations-Variables (Metacell-Gene) Annotations
        ``total_umis``
            The total of all the UMIs of each genes in all the cells grouped into the metacell.

    Also sets in the full ``adata``:

    Observations (Cell) Annotations
        ``metacell_name``
            The string name of the metacell, which is ``prefix`` (default: {prefix}) followed by the metacell index,
            followed by ``.checksum`` where the checksum is a 2 digits, reflecting the (names of the) set of cells
            grouped into each metacell. This keeps metacell names short, and provides protection against assigning
            per-metacell annotations from a different metacell computation to the new results (unless they are
            identical).

    **Computation Parameters**

    For each metacell:

    3. If ``metacell_geo_mean`` (default: {metacell_geo_mean}),

        A. Compute the fraction of each gene out of each cell grouped into the metacell.

        B. For each cell, add to the fractions the ``metacell_umis_regularization`` divided by the total UMIs of the
           cell.

        C. For each gene, compute the weighted geomean of these fractions across all the cells, where the weight of each
           cell is the log of its total number of UMIs, and

        D. Subtract the geomean of the per-cell regularization so all-zero genes would have a zero fraction.

    4. Otherwise, for each metacell, sum the total UMIs of each gene across all cells, and divide it by the total UMIs
       of all genes in all cells.

    5. Normalize the per-gene fractions so their sum would be 1.0 in the metacell.

    6. Compute the ``zeros_cell_size_quantile`` of the total UMIs of the cells to pick the number of UMIs to use for
       zeros computation.

    7. For each gene in each cell, compute the probability it would be zero if we downsampled the cell to have this
       number of UMIs.

    8. Use these probabilities to decide whether the each gene is "effectively zero" in each cell.

    9. For each gene, count the number of cells in which it is "effectively zero".

    In addition:

    10. Pass all relevant per-gene annotations from ``adata`` to the result.

    11. Set the ``metacell_name`` property of each cell in ``adata`` to the name of the group (metacell) it belongs to.
        Cells which do not belong to any metacell are assigned the metacell name ``Outliers``.
    """
    metacell_of_cells = ut.get_o_numpy(adata, groups, formatter=ut.groups_description).copy()
    metacell_of_cells[metacell_of_cells < 0] = -1
    outliers_count = np.sum(metacell_of_cells < 0)
    metacells_count = np.max(metacell_of_cells) + 1
    ut.log_calc("outliers_count", outliers_count)
    ut.log_calc("metacells_count", metacells_count)

    umis_per_gene_per_cell = ut.get_vo_proper(adata, what, layout="row_major")
    umis_per_cell = ut.get_o_numpy(adata, what, sum=True).astype("float64")

    @ut.timed_call()
    def _collect_metacell(metacell_index: int) -> Dict[str, Any]:
        with ut.log_step(
            "- metacell",
            metacell_index,
            formatter=lambda metacell_index: ut.progress_description(metacells_count, metacell_index, "metacell"),
        ):
            if random_seed != 0:
                np.random.seed(random_seed + metacell_index)

            mask_per_cell = metacell_of_cells == metacell_index
            grouped_of_metacell = np.sum(mask_per_cell)
            ut.log_calc("grouped_of_metacell", grouped_of_metacell)
            assert grouped_of_metacell > 0

            umis_per_cell_of_metacell = umis_per_cell[mask_per_cell]
            total_umis_of_metacell = np.sum(umis_per_cell_of_metacell.astype("int64"))
            ut.log_calc("total_umis_of_metacell", total_umis_of_metacell)

            umis_per_gene_per_cell_of_metacell = ut.to_numpy_matrix(umis_per_gene_per_cell[mask_per_cell, :])

            zeros_downsample_umis = round(np.quantile(umis_per_cell_of_metacell, zeros_cell_size_quantile))
            ut.log_calc("zeros_downsample_umis", zeros_downsample_umis)

            downsampled_umis_per_gene_per_cell_of_metacell = ut.downsample_matrix(
                umis_per_gene_per_cell_of_metacell, per="row", samples=zeros_downsample_umis, random_seed=random_seed
            )
            zeros_per_gene_per_cell_of_metacell = downsampled_umis_per_gene_per_cell_of_metacell == 0
            zeros_per_gene_per_cell_of_metacell = ut.to_layout(
                zeros_per_gene_per_cell_of_metacell, layout="column_major"  # type: ignore
            )

            zeros_per_gene = ut.sum_per(zeros_per_gene_per_cell_of_metacell, per="column").astype("int32")
            ut.log_calc("zeros_per_gene", zeros_per_gene, formatter=ut.sizes_description)

            umis_per_gene_per_cell_of_metacell = ut.to_layout(umis_per_gene_per_cell_of_metacell, layout="column_major")
            umis_per_gene_of_metacell = ut.sum_per(umis_per_gene_per_cell_of_metacell, per="column").astype("float32")

            if not metacell_geo_mean:
                total_downsampled_umis_of_metacell = np.sum(
                    downsampled_umis_per_gene_per_cell_of_metacell  # type: ignore
                )

                downsampled_umis_per_gene_per_cell_of_metacell = ut.to_layout(
                    downsampled_umis_per_gene_per_cell_of_metacell, layout="column_major"
                )
                downsampled_umis_per_gene_of_metacell = ut.sum_per(
                    downsampled_umis_per_gene_per_cell_of_metacell, per="column"
                ).astype("float32")
                assert np.sum(downsampled_umis_per_gene_of_metacell) == total_downsampled_umis_of_metacell

                fraction_per_gene_of_metacell = (
                    downsampled_umis_per_gene_of_metacell / total_downsampled_umis_of_metacell
                )

            else:
                fraction_per_gene_per_cell_of_metacell = ut.to_layout(
                    umis_per_gene_per_cell_of_metacell / ut.to_numpy_matrix(umis_per_cell_of_metacell[:, np.newaxis]),
                    layout="column_major",
                )

                regularization_per_cell_of_metacell = metacell_umis_regularization / umis_per_cell_of_metacell
                regularization_of_metacell = ss.mstats.gmean(regularization_per_cell_of_metacell)

                log_fraction_per_gene_per_cell_of_metacell = np.log2(
                    fraction_per_gene_per_cell_of_metacell + regularization_per_cell_of_metacell[:, np.newaxis]
                )
                weight_per_cell_of_metacell = np.log2(umis_per_cell_of_metacell)
                weighted_log_fraction_per_gene_per_cell_of_metacell = ut.to_layout(
                    log_fraction_per_gene_per_cell_of_metacell * weight_per_cell_of_metacell[:, np.newaxis],
                    layout="column_major",
                )

                total_weighted_log_fraction_per_gene_of_metacell = ut.sum_per(
                    weighted_log_fraction_per_gene_per_cell_of_metacell, per="column"
                )
                total_weight_of_metacell = np.sum(weight_per_cell_of_metacell)

                mean_log_fraction_per_gene_of_metacell = (
                    total_weighted_log_fraction_per_gene_of_metacell / total_weight_of_metacell
                )
                fraction_per_gene_of_metacell = 2**mean_log_fraction_per_gene_of_metacell
                fraction_per_gene_of_metacell -= regularization_of_metacell
                fraction_per_gene_of_metacell[umis_per_gene_of_metacell == 0] = 0
                fraction_per_gene_of_metacell[fraction_per_gene_of_metacell < 0] = 0
                assert np.min(fraction_per_gene_of_metacell) >= 0
                fraction_per_gene_of_metacell /= np.sum(fraction_per_gene_of_metacell)

            fraction_per_gene_of_metacell = fraction_per_gene_of_metacell.astype("float32")
            if _metacell_groups:
                fraction_per_gene_of_metacell *= total_umis_of_metacell
                ut.log_calc(
                    "effective_umis_per_gene_of_metacell", fraction_per_gene_of_metacell, formatter=ut.sizes_description
                )
            else:
                ut.log_calc(
                    "fraction_per_gene_of_metacell", fraction_per_gene_of_metacell, formatter=ut.sizes_description
                )

            return {
                "grouped": grouped_of_metacell,
                "total_umis": total_umis_of_metacell,
                "umis_per_gene": umis_per_gene_of_metacell,
                "fraction_per_gene": fraction_per_gene_of_metacell,
                "zeros_downsample_umis": zeros_downsample_umis,
                "zeros_per_gene": zeros_per_gene,
            }

    results = ut.parallel_map(_collect_metacell, metacells_count)

    fraction_per_gene_per_metacell = sp.csr_matrix(np.vstack([result["fraction_per_gene"] for result in results]))
    assert str(fraction_per_gene_per_metacell.dtype) == "float32"
    assert fraction_per_gene_per_metacell.shape == (metacells_count, adata.n_vars)

    mdata = AnnData(fraction_per_gene_per_metacell)
    if top_level:
        ut.top_level(mdata)
    if name.startswith("."):
        name = ut.get_name(adata, "unnamed") + name  # type: ignore
    ut.set_name(mdata, name)
    mdata.var_names = adata.var_names
    mdata.obs_names = _obs_names(prefix or "", ut.to_numpy_vector(adata.obs_names), metacell_of_cells)

    grouped_per_metacell = np.array([result["grouped"] for result in results])
    ut.set_o_data(mdata, "grouped", grouped_per_metacell)

    total_umis_per_metacell = np.array([result["total_umis"] for result in results])
    ut.set_o_data(mdata, "total_umis", total_umis_per_metacell)

    umis_per_gene_per_metacell = np.vstack([result["umis_per_gene"] for result in results])
    assert str(umis_per_gene_per_metacell.dtype) == "float32"
    assert umis_per_gene_per_metacell.shape == (metacells_count, adata.n_vars)
    ut.set_vo_data(mdata, "total_umis", umis_per_gene_per_metacell)

    zeros_downsample_umis_per_metacell = np.array([result["zeros_downsample_umis"] for result in results])
    ut.set_o_data(mdata, "__zeros_downsample_umis", zeros_downsample_umis_per_metacell)

    zeros_per_gene_per_metacell = np.vstack([result["zeros_per_gene"] for result in results])
    assert str(zeros_per_gene_per_metacell.dtype) == "int32"
    assert zeros_per_gene_per_metacell.shape == (metacells_count, adata.n_vars)
    ut.set_vo_data(mdata, "zeros", zeros_per_gene_per_metacell)

    metacell_names = np.array(["Outliers"] + list(mdata.obs_names), dtype="str")
    metacell_of_cells += 1
    metacell_name_of_cells = metacell_names[metacell_of_cells]
    ut.set_o_data(adata, "metacell_name", metacell_name_of_cells)

    for annotation_name in adata.var.keys():
        value_per_gene = ut.get_v_numpy(adata, annotation_name)
        ut.set_v_data(mdata, annotation_name, value_per_gene)

    if isinstance(groups, str) and ut.has_data(adata, "metacells_level"):
        tl.convey_obs_to_group(adata=adata, gdata=mdata, group=groups, property_name="metacell_level")
    if isinstance(groups, str) and ut.has_data(adata, "rare_cell"):
        tl.convey_obs_to_group(
            adata=adata,
            gdata=mdata,
            group=groups,
            property_name="cells_rare_gene_module",
            to_property_name="metacells_rare_gene_module",
        )
        tl.convey_obs_to_group(
            adata=adata,
            gdata=mdata,
            group=groups,
            property_name="rare_cell",
            to_property_name="rare_metacell",
        )

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
        checksum = int(hasher.hexdigest(16), 16) % 100
        name_of_groups.append(f"{prefix}{group_index}.{checksum:02d}")
    return name_of_groups
