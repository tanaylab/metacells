'''
Distincts
---------
'''

import logging
from typing import Optional, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.extensions as xt  # type: ignore
import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    'compute_distinct_folds',
    'find_distinct_genes',
    'compute_subset_distinct_genes',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def compute_distinct_folds(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    normalization: float = 0,
    inplace: bool = True,
) -> Optional[ut.PandasFrame]:
    '''
    Compute for each observation (cell) and each variable (gene) how much is the value
    different from the overall population.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are (mata)cells and the variables are genes.

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
    of, _ = ut.log_operation(LOG, adata, 'compute_distinct_folds', of)

    columns_data = ut.get_vo_proper(adata, of, layout='column_major')
    fractions_of_genes_in_data = ut.fraction_per(columns_data, per='column')
    fractions_of_genes_in_data += normalization

    rows_data = ut.get_vo_proper(adata, of, layout='row_major')
    total_umis_of_cells = ut.sum_per(rows_data, per='row')
    total_umis_of_cells = np.copy(total_umis_of_cells)
    total_umis_of_cells[total_umis_of_cells == 0] = 1
    fraction_of_genes_in_cells = \
        ut.to_dense_matrix(rows_data) / total_umis_of_cells[:, np.newaxis]
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

    return pd.DataFrame(fold_of_genes_in_cells, index=adata.obs_names, columns=adata.var_names)


@ut.timed_call()
@ut.expand_doc()
def find_distinct_genes(
    adata: AnnData,
    *,
    distinct_fold: str = 'distinct_fold',
    distinct_genes_count: int = pr.distinct_genes_count,
    inplace: bool = True,
) -> Optional[Tuple[ut.PandasFrame, ut.PandasFrame]]:
    '''
    Find for each observation (cell) the genes in which it is most distinct from the general
    population. This is typically applied to the metacells data rather than to the cells data.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are (mata)cells and the variables are genes, including a per-observation-per-variable annotated
    data named ``distinct_fold``, e.g. as computed by :py:func:`compute_distinct_folds`.

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

    1. Fetch the previously computed per-observation-per-variable `distinct_fold`` annotation
       (default: {distinct_fold}).

    2. Keep the ``distinct_genes_count`` (default: {distinct_genes_count}) top fold factors.
    '''
    ut.log_operation(LOG, adata, 'find_distinct_genes')
    assert 0 < distinct_genes_count < adata.n_vars

    distinct_gene_indices = \
        np.empty((adata.n_obs, distinct_genes_count), dtype='int32')
    distinct_gene_folds = \
        np.empty((adata.n_obs, distinct_genes_count), dtype='float32')

    fold_in_cells = ut.get_vo_proper(adata, distinct_fold, layout='row_major')
    xt.top_distinct(distinct_gene_indices, distinct_gene_folds,
                    fold_in_cells, False)

    if inplace:
        ut.set_oa_data(adata, 'cell_distinct_gene_indices',
                       distinct_gene_indices)
        ut.set_oa_data(adata, 'cell_distinct_gene_folds', distinct_gene_folds)
        return None

    return pd.DataFrame(distinct_gene_indices, index=adata.obs_names), \
        pd.DataFrame(distinct_gene_folds, index=adata.obs_names)


@ut.timed_call()
@ut.expand_doc()
def compute_subset_distinct_genes(
    adata: AnnData,
    *,
    of: Optional[str] = None,
    to: Optional[str] = None,
    normalize: Optional[Union[bool, str, ut.DenseVector]],
    subset: Union[str, ut.DenseVector],
    inplace: bool = True,
) -> Optional[ut.PandasSeries]:
    '''
    Given a subset of the observations (cells), compute for each gene how distinct it is
    in the subset compared to the overall population.

    This is the area-under-curve of the receiver operating characteristic (AUROC) for the gene,
    that is, the probability that a randomly selected observation (cell) in the subset will
    have a higher value ``of`` some data than a randomly selected observation (cell) outside
    the subset.

    **Input**

    A :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where the observations
    are (mata)cells and the variables are genes.

    **Returns**

    Variable (Gene) Annotations
        ``to``
            Store the distinctiveness of the gene in the subset as opposed to the rest of the
            population.

    If ``inplace`` (default: {inplace}), this is written to the data, and the function returns
    ``None``. This requires ``to`` to be specified. Otherwise this is returned as a pandas series
    (indexed by the gene names).

    **Computation Parameters**

    1. Use the ``subset`` to assign a boolean label to each observation (cell). The ``subset`` can
       be a vector of integer observation names, or a boolean mask, or the string name of a
       per-observation annotation containing the boolean mask.

    2. If ``normalize`` is ``False``, use the data as-is. If it is ``True``, divide the data by the
       sum of each observation (cell). If it is a string, it should be the name of a per-observation
       annotation to use. Otherwise, it should be a vector of the scale factor for each observation
       (cell).

    3. Compute the AUROC for each gene for the scaled data based on this mask.
    '''
    ut.log_operation(LOG, adata, 'compute_subset_distinct_genes')

    if isinstance(subset, str):
        subset = ut.get_o_dense(adata, subset)

    if subset.dtype != 'bool':
        mask: ut.DenseVector = np.full(adata.n_obs, False)
        mask[subset] = True
        subset = mask

    scale_of_cells: Optional[ut.DenseVector] = None
    if not isinstance(normalize, bool):
        ut.log_use(LOG, adata, normalize, name='normalize', per='o')
        if normalize is not None:
            scale_of_cells = ut.get_o_dense(adata, normalize)

    elif normalize:
        data = ut.get_vo_proper(adata, of, layout='row_major')
        scale_of_cells = ut.sum_per(data, per='row')
        LOG.debug('normalize: <sum>')

    else:
        LOG.debug('normalize: None')

    if scale_of_cells is not None:
        assert scale_of_cells.size == adata.n_obs

    matrix = ut.get_vo_proper(adata, of, layout='column_major').transpose()

    distinct_of_genes = ut.matrix_rows_auroc(matrix, subset, scale_of_cells)

    if inplace:
        assert to is not None
        ut.set_v_data(adata, to, distinct_of_genes)
        return None

    return pd.Series(distinct_of_genes, index=adata.var_names)
