'''
Collect
-------
'''

from typing import Optional, Tuple, Union

import numpy as np
from anndata import AnnData

import metacells.parameters as pr
import metacells.tools as tl
import metacells.utilities as ut

__all__ = [
    'collect_metacells',
    'compute_effective_cell_sizes',
]


@ut.logged()
@ut.timed_call()
def collect_metacells(
    adata: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    *,
    max_cell_size: Optional[float] = pr.max_cell_size,
    max_cell_size_factor: Optional[float] = pr.max_cell_size_factor,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
    name: str = 'metacells',
    top_level: bool = True,
) -> AnnData:
    '''
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

        ``forbidden_gene``
            A boolean mask of genes which are forbidden from being chosen as "feature" genes based
            on their name. This is ``False`` for non-"clean" genes.

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

    1. Compute the cell's scale factors by invoking :py:func:`compute_effective_cell_sizes` using the
       ``max_cell_size`` (default: {max_cell_size}), ``max_cell_size_factor`` (default:
       {max_cell_size_factor}) and ``cell_sizes`` (default: {cell_sizes}).

    2. Scale the cell's data using these factors, if needed.

    3. Invoke :py:func:`metacells.tools.group.group_obs_data` to sum the cells into
       metacells.

    4. Pass all relevant per-gene and per-cell annotations to the result.
    '''
    _cell_sizes, _max_cell_size, cell_scale_factors = \
        compute_effective_cell_sizes(adata,
                                     max_cell_size=max_cell_size,
                                     max_cell_size_factor=max_cell_size_factor,
                                     cell_sizes=cell_sizes)

    if cell_scale_factors is not None:
        data = ut.get_vo_proper(adata, what, layout='row_major')
        what = ut.scale_by(data, cell_scale_factors, by='row')

    mdata = tl.group_obs_data(adata, what, groups='metacell', name=name)
    assert mdata is not None
    if top_level:
        ut.top_level(mdata)

    for annotation_name in ('excluded_gene',
                            'clean_gene',
                            'forbidden_gene',
                            'pre_feature_gene',
                            'feature_gene'):
        if not ut.has_data(adata, annotation_name):
            continue
        value_per_gene = ut.get_v_numpy(adata, annotation_name,
                                        formatter=ut.mask_description)
        ut.set_v_data(mdata, annotation_name, value_per_gene,
                      formatter=ut.mask_description)

    for annotation_name in ('pile', 'candidate'):
        if ut.has_data(adata, annotation_name):
            tl.group_obs_annotation(adata, mdata, groups='metacell',
                                    formatter=ut.groups_description,
                                    name=annotation_name, method='unique')

    return mdata


@ut.logged()
@ut.timed_call()
def compute_effective_cell_sizes(
    adata: AnnData,
    *,
    max_cell_size: Optional[float] = pr.max_cell_size,
    max_cell_size_factor: Optional[float] = pr.max_cell_size_factor,
    cell_sizes: Optional[Union[str, ut.Vector]] = pr.cell_sizes,
) -> Tuple[Optional[ut.NumpyVector], Optional[float], Optional[ut.NumpyVector]]:
    '''
    Compute the effective cell sizes for controlling metacell sizes, and the scale factor to apply to
    each cell (if needed).

    If ``max_cell_size`` (default: {max_cell_size}) is specified, use it as a cap on the size (total
    UMI values) of each cell. Otherwise, if ``max_cell_size_factor`` (default:
    {max_cell_size_factor}) is specified, use as a cap the median size times this factor. If both
    are ``None``, use no cap.

    If we have a cap, and there are cells whose total size is above the cap, then we'll need to
    scale them by some factor (less than 1.0).

    Returns the cell sizes, the maximal cell size, and the optional cell scale factors vector.
    '''
    effective_cell_sizes = \
        ut.maybe_o_numpy(adata, cell_sizes, formatter=ut.sizes_description)

    cell_scale_factors: Optional[ut.NumpyVector] = None

    if (max_cell_size is not None
            or max_cell_size_factor is not None) \
            and effective_cell_sizes is not None:
        if max_cell_size is None:
            assert max_cell_size_factor is not None
            ut.log_calc('median cell size', np.median(effective_cell_sizes))
            max_cell_size = \
                float(np.median(effective_cell_sizes)
                      * max_cell_size_factor)

        ut.log_calc('cap_cell_size', max_cell_size)
        cell_scale_factors = max_cell_size / effective_cell_sizes
        effective_cell_sizes[effective_cell_sizes
                             > max_cell_size] = max_cell_size
        assert cell_scale_factors is not None  # Dumb mypy.
        cell_scale_factors[cell_scale_factors > 1.0] = 1.0
        scaled_count = np.sum(cell_scale_factors < 1.0)
        ut.log_calc('scaled cells count', scaled_count)
        if scaled_count == 0:
            cell_scale_factors = None

    return (effective_cell_sizes, max_cell_size, cell_scale_factors)
