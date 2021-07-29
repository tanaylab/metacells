'''
Distincts
---------
'''

import sys
from typing import Optional, Tuple, Union

import numpy as np
from anndata import AnnData

import metacells.parameters as pr
import metacells.utilities as ut

if not 'sphinx' in sys.argv[0]:
    import metacells.extensions as xt  # type: ignore

__all__ = [
    'compute_distinct_folds',
    'find_distinct_genes',
    'compute_subset_distinct_genes',
]


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_distinct_folds(
    adata: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    *,
    normalization: float = 0,
    inplace: bool = True,
) -> Optional[ut.PandasFrame]:
    '''
    Compute for each observation (cell) and each variable (gene) how much is the ``what`` (default:
    {what}) value different from the overall population.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    **Returns**

    Per-Observation-Per-Variable (Cell-Gene) Annotations:
        ``distinct_ratio``
            For each gene in each cell, the log based 2 of the ratio between the fraction of the
            gene in the cell and the fraction of the gene in the overall population (sum of cells).

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as a pandas frame (indexed by the observation and
    distinct gene rank).

    **Computation Parameters**

    1. Compute, for each gene, the fraction of the gene's values out of the total sum of the values
       (that is, the mean fraction of the gene's expression in the population).

    2. Compute, for each cell, for each gene, the fraction of the gene's value out of the sum
       of the values in the cell (that is, the fraction of the gene's expression in the cell).

    3. Divide the two to the distinct ratio (that is, how much the gene's expression in the cell is
       different from the overall population), first adding the ``normalization`` (default:
       {normalization}) to both.

    4. Compute the log (base 2) of the result and use it as the fold factor.
    '''
    columns_data = ut.get_vo_proper(adata, what, layout='column_major')
    fractions_of_genes_in_data = ut.fraction_per(columns_data, per='column')
    fractions_of_genes_in_data += normalization

    total_umis_of_cells = ut.get_o_numpy(adata, what, sum=True)
    total_umis_of_cells[total_umis_of_cells == 0] = 1

    rows_data = ut.get_vo_proper(adata, what, layout='row_major')
    fraction_of_genes_in_cells = \
        ut.to_numpy_matrix(rows_data) / total_umis_of_cells[:, np.newaxis]
    fraction_of_genes_in_cells += normalization

    zeros_mask = fractions_of_genes_in_data <= 0
    fractions_of_genes_in_data[zeros_mask] = -1
    fraction_of_genes_in_cells[:, zeros_mask] = -1

    ratio_of_genes_in_cells = fraction_of_genes_in_cells
    ratio_of_genes_in_cells /= fractions_of_genes_in_data
    assert np.min(np.min(ratio_of_genes_in_cells)) > 0

    fold_of_genes_in_cells = \
        np.log2(ratio_of_genes_in_cells, out=ratio_of_genes_in_cells)

    if inplace:
        ut.set_vo_data(adata, 'distinct_fold', fold_of_genes_in_cells)
        return None

    return ut.to_pandas_frame(fold_of_genes_in_cells,
                              index=adata.obs_names, columns=adata.var_names)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def find_distinct_genes(
    adata: AnnData,
    what: Union[str, ut.Matrix] = 'distinct_fold',
    *,
    distinct_genes_count: int = pr.distinct_genes_count,
    inplace: bool = True,
) -> Optional[Tuple[ut.PandasFrame, ut.PandasFrame]]:
    '''
    Find for each observation (cell) the genes in which its ``what`` (default: {what}) value is most
    distinct from the general population. This is typically applied to the metacells data rather
    than to the cells data.

    **Input**

    Annotated ``adata``, where the observations are (mata)cells and the variables are genes,
    including a per-observation-per-variable annotated folds data, {what}), e.g. as computed by
    :py:func:`compute_distinct_folds`.

    **Returns**

    Observation-Any (Cell) Annotations
        ``cell_distinct_gene_indices``
            For each cell, the indices of its top ``distinct_genes_count`` genes.
        ``cell_distinct_gene_folds``
            For each cell, the fold factor of its top ``distinct_genes_count``.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. Otherwise this is returned as two pandas frames (indexed by the observation and
    distinct gene rank).

    **Computation Parameters**

    1. Fetch the previously computed per-observation-per-variable ``what`` data.

    2. Keep the ``distinct_genes_count`` (default: {distinct_genes_count}) top fold factors.
    '''
    assert 0 < distinct_genes_count < adata.n_vars

    distinct_gene_indices = \
        np.empty((adata.n_obs, distinct_genes_count), dtype='int32')
    distinct_gene_folds = \
        np.empty((adata.n_obs, distinct_genes_count), dtype='float32')

    fold_in_cells = ut.get_vo_proper(adata, what, layout='row_major')
    xt.top_distinct(distinct_gene_indices, distinct_gene_folds,
                    fold_in_cells, False)

    if inplace:
        ut.set_oa_data(adata, 'cell_distinct_gene_indices',
                       distinct_gene_indices)
        ut.set_oa_data(adata, 'cell_distinct_gene_folds', distinct_gene_folds)
        return None

    return ut.to_pandas_frame(distinct_gene_indices, index=adata.obs_names), \
        ut.to_pandas_frame(distinct_gene_folds, index=adata.obs_names)


@ut.logged()
@ut.timed_call()
@ut.expand_doc()
def compute_subset_distinct_genes(
    adata: AnnData,
    what: Union[str, ut.Matrix] = '__x__',
    *,
    prefix: Optional[str] = None,
    scale: Optional[Union[bool, str, ut.NumpyVector]],
    subset: Union[str, ut.NumpyVector],
    normalization: float,
) -> Optional[Tuple[ut.PandasSeries, ut.PandasSeries]]:
    '''
    Given a subset of the observations (cells), compute for each gene how distinct its ``what``
    (default: {what}) value is in the subset compared to the overall population.

    This is the area-under-curve of the receiver operating characteristic (AUROC) for the gene, that
    is, the probability that a randomly selected observation (cell) in the subset will have a higher
    value than a randomly selected observation (cell) outside the subset.

    **Input**

    Annotated ``adata``, where the observations are cells and the variables are genes, where
    ``what`` is a per-variable-per-observation matrix or the name of a per-variable-per-observation
    annotation containing such a matrix.

    **Returns**

    Variable (Gene) Annotations
        ``<prefix>_fold``
            Store the ratio of the expression of the gene in the subset as opposed to the rest of
            the population.
        ``<prefix>_auroc``
            Store the distinctiveness of the gene in the subset as opposed to the rest of the
            population.

    If ``prefix`` (default: {prefix}), is specified, this is written to the data. Otherwise this is
    returned as two pandas series (indexed by the gene names).

    **Computation Parameters**

    1. Use the ``subset`` to assign a boolean label to each observation (cell). The ``subset`` can
       be a vector of integer observation names, or a boolean mask, or the string name of a
       per-observation annotation containing the boolean mask.

    2. If ``scale`` is ``False``, use the data as-is. If it is ``True``, divide the data by the
       sum of each observation (cell). If it is a string, it should be the name of a per-observation
       annotation to use. Otherwise, it should be a vector of the scale factor for each observation
       (cell).

    3. Compute the fold ratios using the ``normalization`` (no default!) and the AUROC for each gene,
       for the scaled data based on this mask.
    '''
    if isinstance(subset, str):
        subset = ut.get_o_numpy(adata, subset)

    if subset.dtype != 'bool':
        mask: ut.NumpyVector = np.full(adata.n_obs, False)
        mask[subset] = True
        subset = mask

    scale_of_cells: Optional[ut.NumpyVector] = None
    if not isinstance(scale, bool):
        scale_of_cells = \
            ut.maybe_o_numpy(adata, scale, formatter=ut.sizes_description)
    elif scale:
        scale_of_cells = ut.get_o_numpy(adata, what, sum=True)
    else:
        scale_of_cells = None

    matrix = ut.get_vo_proper(adata, what, layout='column_major').transpose()
    fold_of_genes, auroc_of_genes = \
        ut.matrix_rows_folds_and_aurocs(matrix,
                                        columns_subset=subset,
                                        columns_scale=scale_of_cells,
                                        normalization=normalization)

    if prefix is not None:
        ut.set_v_data(adata, f'{prefix}_auroc', auroc_of_genes)
        ut.set_v_data(adata, f'{prefix}_fold', fold_of_genes)
        return None

    return (ut.to_pandas_series(fold_of_genes, index=adata.var_names),
            ut.to_pandas_series(auroc_of_genes, index=adata.var_names))
