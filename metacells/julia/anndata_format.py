"""
AnnData Format
--------------
"""

from typing import AbstractSet
from typing import Mapping
from typing import Optional
from typing import Tuple

from daf import DafWriter
from daf import StorageScalar
from daf.julia_import import jl


def import_h5ads(
    *,
    destination: DafWriter,
    raw_cells_h5ad: Optional[str] = None,
    clean_cells_h5ad: str,
    metacells_h5ad: str,
    type_property: Optional[str] = None,
    rename_type: Optional[str] = "type",
    type_colors_csv: Optional[str] = None,
    type_properties: Optional[AbstractSet[str]] = None,
    properties_defaults: Optional[Mapping[str, Optional[Tuple[str, Optional[StorageScalar]]]]] = None,
) -> None:
    """
    Import an ``AnnData`` based metacells dataset into a ``Daf`` ``destination`` data set. See the Julia
    `documentation <https://tanaylab.github.io/Metacells.jl/v0.1.0/anndata_format.html#Metacells.AnnDataFormat.import_h5ad!>`__
    for details.
    """
    jl.Metacells.import_h5ads_b(
        destination=destination,
        raw_cells_h5ad=raw_cells_h5ad,
        clean_cells_h5ad=clean_cells_h5ad,
        metacells_h5ad=metacells_h5ad,
        type_property=type_property,
        rename_type=rename_type,
        type_colors_csv=type_colors_csv,
        type_properties=type_properties,
        properties_defaults=properties_defaults,
    )
