'''
Results
-------
'''

import logging
from typing import Iterable, List, Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy.sparse as sparse  # type: ignore
from anndata import AnnData

import metacells.preprocessing as pp
import metacells.utilities as ut

__all__ = [
    'collect_result_metacells',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def collect_result_metacells(  # pylint: disable=too-many-statements
    adata: AnnData,
    cdata: Union[AnnData, Iterable[AnnData]],
    of: Optional[str] = None,
    *,
    name: str = 'METACELLS',
    tmp: bool = False,
    intermediate: bool = True,
) -> Optional[AnnData]:
    '''
    Collect the result metacells computed using the clean data.

    **Input**

    The full :py:func:`metacells.utilities.annotation.setup` annotated ``adata``, where
    the observations are cells and the variables are genes, and the clean data we
    have computed metacells for.

    **Returns**

    Annotated metacell data containing for each observation the sum ``of`` the data (by default,
    the focus) of the cells for each metacell, which contains the following annotations:

    Observations (Cell) Annotations
        ``cell_pile``
            The index of the (final) pile all the cells of each metacell were assigned to for the
            final invocation of :py:func:`metacells.pipeline.direct.compute_direct_metacells`.

    Variable-Any (Gene) Annotations
        ``gene_pile_feature``
            A sparse boolean 2D mask which is ``True`` for each feature gene for each pile (only one
            pile here). Provided for compatibility with
            :py:func:`metacells.pipeline.divide_and_conquer.compute_divide_and_conquer_metacells`.

    Sets the above as well as the following in the full data:

    Observations (Cell) Annotations
        ``metacell``
            The index of the metacell each cell belongs to. This is ``-1`` for outlier cells and
            ``-2`` for excluded cells.

    If ``intermediate`` (default: {intermediate}), keep all all the intermediate data (e.g. sums)
    for future reuse. Otherwise, discard it.

    **Computation Parameters**

    1. Iterate on all the ``cdata`` clean data, which we computed ``metacells`` (per-observation
        annotations) for.

    2. Combine the metacell indices from the clean data into the final metacells, using the
       ``base_indices`` pre-observation (cell) annotation to map the clean cell indices to the full
       cell indices. If multiple clean data assign a metacell index to the same cell, the last one
       wins. The clean metacell indices are shifted so that the result is unique metacell indices
       across the whole data.

       Cells which were "excluded", that is, not included in any of the clean data, are given the
       metacell index ``-2``. This distinguishes them from "outlier" cells which were not placed in
       any metacell in the clean data they were included in.

    3. Similarly, also combine the ``cell_pile`` indices from the clean data into the final pile
       indices. Cells which were "excluded" will have the pile index ``-1``.

    4. Combine the ``gene_pile_feature`` of all the clean data into a single sparse matrix
       specifying the feature genes used for processing each of the piles.

    5. Invoke :py:func:`metacells.preprocessing.group.group_obs_data` to sum the ``of`` data into a
       new metacells annotated data, using the ``name`` (default: {name}) and ``tmp`` (default:
       {tmp}).
    '''
    ut.log_pipeline_step(LOG, adata, 'collect_result_metacells')

    pile_of_all_cells = np.full(adata.n_obs, -1, dtype='int32')
    metacell_of_all_cells = np.full(adata.n_obs, -2, dtype='int32')

    if isinstance(cdata, AnnData):
        cdata = [cdata]

    clean_genes_count = -1
    feature_of_piles_of_genes_list: List[pd.Frame] = []
    piles_count = 0
    metacells_count = 0
    for cdatum in cdata:
        if clean_genes_count < 0:
            clean_genes_count = cdatum.n_vars
            base_index_of_clean_genes = ut.get_v_data(cdatum, 'var_base_index')
        else:
            assert cdatum.n_vars == clean_genes_count

        feature_of_piles_of_clean_genes: pd.DataFrame = \
            ut.get_va_data(cdatum, 'gene_pile_feature')
        if not isinstance(feature_of_piles_of_clean_genes, pd.DataFrame):
            feature_of_piles_of_clean_genes = \
                pd.DataFrame.sparse.from_spmatrix(  #
                    feature_of_piles_of_clean_genes)

        feature_of_piles_of_clean_genes.columns += piles_count
        feature_of_piles_of_genes_list.append(feature_of_piles_of_clean_genes)

        metacell_of_clean_cells = np.copy(ut.get_o_data(cdatum, 'metacell'))
        grouped_clean_cells_mask = metacell_of_clean_cells >= 0

        clean_metacells_count = np.max(metacell_of_clean_cells) + 1
        metacell_of_clean_cells[grouped_clean_cells_mask] += metacells_count
        metacells_count += clean_metacells_count

        clean_cell_indices = ut.get_o_data(cdatum, 'obs_base_index')
        metacell_of_all_cells[clean_cell_indices] = metacell_of_clean_cells

        pile_of_cells = ut.get_o_data(cdatum, 'cell_pile')
        pile_of_all_cells[clean_cell_indices] = pile_of_cells
        pile_of_all_cells[clean_cell_indices] += piles_count

        clean_piles_count = len(feature_of_piles_of_clean_genes.columns)
        piles_count += clean_piles_count

        if LOG.isEnabledFor(logging.DEBUG):
            clean_name = ut.get_name(cdatum)
            if clean_name is None:
                LOG.debug('  - collect metacells from clean data')
            else:
                LOG.debug('  - collect metacells from: %s', clean_name)

            LOG.debug('    piles: %s', clean_piles_count)
            LOG.debug('    metacells: %s', clean_metacells_count)
            LOG.debug('    cells: %s',
                      ut.mask_description(grouped_clean_cells_mask))
            LOG.debug('    outliers: %s',
                      ut.mask_description(~grouped_clean_cells_mask))

    assert clean_genes_count > 0
    assert piles_count == np.max(pile_of_all_cells) + 1
    feature_of_piles_of_clean_genes_frame = \
        pd.concat(feature_of_piles_of_genes_list, axis=1)

    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug('  collected piles: %s', piles_count)
        LOG.debug('  collected metacells: %s',
                  np.max(metacell_of_all_cells) + 1)
        LOG.debug('  collected cells: %s',
                  ut.mask_description(metacell_of_all_cells >= 0))
        LOG.debug('  collected outliers: %s',
                  ut.mask_description(metacell_of_all_cells == -1))
        LOG.debug('  collected excluded: %s',
                  ut.mask_description(metacell_of_all_cells == -2))

    ut.set_o_data(adata, 'cell_pile', pile_of_all_cells,
                  log_value=lambda: str(piles_count))
    ut.set_o_data(adata, 'metacell', metacell_of_all_cells,
                  log_value=lambda: str(np.max(metacell_of_all_cells) + 1))

    feature_of_piles_of_clean_genes = \
        feature_of_piles_of_clean_genes_frame.sparse.to_coo()
    assert feature_of_piles_of_clean_genes.shape[0] == clean_genes_count
    piles_count = feature_of_piles_of_clean_genes.shape[1]
    genes_count = adata.n_vars
    assert clean_genes_count <= genes_count

    data = feature_of_piles_of_clean_genes.data
    pile_indices = feature_of_piles_of_clean_genes.col
    clean_gene_indices = feature_of_piles_of_clean_genes.row
    gene_indices = base_index_of_clean_genes[clean_gene_indices]

    feature_of_piles_of_genes = \
        sparse.coo_matrix((data, (gene_indices, pile_indices)),
                          shape=(genes_count, piles_count))
    feature_of_piles_of_genes.has_canonical_format = True

    ut.set_va_data(adata, 'gene_pile_feature', feature_of_piles_of_genes)

    gdata = pp.group_obs_data(adata, of=of, groups='metacell',
                              name=name, tmp=tmp, intermediate=intermediate)
    if gdata is None:
        return gdata

    pp.group_obs_annotation(adata, gdata, groups='metacell',
                            name='cell_pile', policy='unique')

    ut.set_va_data(gdata, 'gene_pile_feature', feature_of_piles_of_genes)

    return gdata
