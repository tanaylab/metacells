'''
Layout
------
'''

import logging

import numpy as np
import scipy.sparse as sp  # type: ignore
import umap  # type: ignore
from anndata import AnnData

import metacells.parameters as pr
import metacells.utilities as ut

__all__ = [
    'umap_by_distances',
    'spread_coordinates',
]


LOG = logging.getLogger(__name__)


@ut.timed_call()
@ut.expand_doc()
def umap_by_distances(
    adata: AnnData,
    distances: str = 'umap_distances',
    *,
    prefix: str = 'umap',
    k: int = pr.umap_k,
    min_dist: float = pr.umap_min_dist,
) -> None:
    '''
    Compute layout for the observations using UMAP, based on a distances matrix.

    **Input**

    The input annotated ``adata`` is expected to contain a per-observation-per-observation property
    ``distances`` (default: {distances}), which describes the distance between each two observations
    (cells). The distances must be non-negative, symmetrical, and zero for self-distances (on the
    diagonal).

    **Returns**

    Sets the following annotations in ``adata``:

    Observation (Cell) Annotations
        ``<prefix>_x``, ``<prefix>_y``
            Coordinates for UMAP 2D projection of the observations.

    **Computation Parameters**

    1. Invoke UMAP to compute the layout using ``min_dist`` (default: {min_dist}) and ``k``
       (default: {k}).
    '''
    ut.log_operation(LOG, adata, 'umap_by_distances')

    distances_matrix = ut.get_oo_proper(adata, distances)

    # UMAP implementation dies when given a dense matrix.
    distances_csr = sp.csr_matrix(distances_matrix)

    # UMAP implementation doesn't know to reduce K by itself.
    n_neighbors = min(k, adata.n_obs - 2)

    try:
        coordinates = umap.UMAP(metric='precomputed',
                                n_neighbors=n_neighbors,
                                min_dist=min_dist).fit_transform(distances_csr)
    except ValueError:
        # UMAP implementation doesn't know how to handle too few edges.
        # However, it considers structural zeros as real edges.
        distances_matrix = distances_matrix + 1.0  # type: ignore
        np.fill_diagonal(distances_matrix, 0.0)
        distances_csr = sp.csr_matrix(distances_matrix)
        distances_csr.data -= 1.0
        coordinates = umap.UMAP(metric='precomputed',
                                n_neighbors=n_neighbors,
                                min_dist=min_dist).fit_transform(distances_csr)

    x_coordinates = ut.to_numpy_vector(coordinates[:, 0], copy=True)
    y_coordinates = ut.to_numpy_vector(coordinates[:, 1], copy=True)

    x_min = np.min(x_coordinates)
    y_min = np.min(y_coordinates)

    x_max = np.max(x_coordinates)
    y_max = np.max(y_coordinates)

    x_size = x_max - x_min
    y_size = y_max - y_min

    assert x_size > 0
    assert y_size > 0

    if y_size > x_size:
        x_coordinates, y_coordinates = y_coordinates, x_coordinates

    ut.set_o_data(adata, f'{prefix}_x', x_coordinates)
    ut.set_o_data(adata, f'{prefix}_y', y_coordinates)


@ut.timed_call()
@ut.expand_doc()
def spread_coordinates(
    adata: AnnData,
    *,
    prefix: str = 'umap',
    suffix: str = 'spread',
    cover_fraction: float = pr.cover_fraction,
    noise_fraction: float = pr.noise_fraction,
    random_seed: int = pr.random_seed,
) -> None:
    '''
    Move UMAP points so they cover some fraction of the plot area without overlapping.

    **Input**

    The input annotated ``adata`` is expected to contain the per-observation properties
    ``<prefix>_x`` and ``<prefix>_y`` (default prefix: {prefix}) which contain the UMAP coordinates.

    **Returns**

    Sets the following annotations in ``adata``:

    Observation (Cell) Annotations
        ``<prefix>_x_<suffix>``, ``<prefix>_y_<suffix>`` (default suffix: {suffix})
            The new coordinates which will be spread out so the points do not overlap and
            cover some fraction of the total plot area.

    **Computation Parameters**

    1. Move the points so they cover ``cover_fraction`` (default: {cover_fraction}) of the total
       plot area. Also add a noise of the ``noise_fraction`` (default: {noise_fraction}) of the
       minimal distance between the
       points, using the ``random_seed`` (default: {random_seed}).
    '''
    ut.log_operation(LOG, adata, 'spread_coordinates')

    LOG.info('cover_fraction: %s', cover_fraction)
    LOG.info('noise_fraction: %s', noise_fraction)

    assert 0 < cover_fraction < 1
    assert noise_fraction >= 0

    x_coordinates = ut.get_o_numpy(adata, f'{prefix}_x')
    y_coordinates = ut.get_o_numpy(adata, f'{prefix}_y')

    x_coordinates, y_coordinates = ut.cover_coordinates(x_coordinates, y_coordinates,
                                                        cover_fraction=cover_fraction,
                                                        noise_fraction=noise_fraction,
                                                        random_seed=random_seed)

    ut.set_o_data(adata, f'{prefix}_x_{suffix}', x_coordinates)
    ut.set_o_data(adata, f'{prefix}_y_{suffix}', y_coordinates)
