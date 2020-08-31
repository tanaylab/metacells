'''
Test the preprocessing functions.
'''

from glob import glob
from typing import Any, Dict, Tuple

import numpy as np  # type: ignore
import pytest  # type: ignore
import scanpy as sc  # type: ignore
import yaml
from anndata import AnnData

import metacells as mc

# pylint: disable=missing-docstring


LOADED: Dict[str, Tuple[AnnData, Dict[str, Any]]] = {}


def _load(path: str) -> Tuple[AnnData, Dict[str, Any]]:
    global LOADED
    if path in LOADED:
        return LOADED[path]

    with open(path[:-4] + 'yaml') as file:
        expected = yaml.safe_load(file)
    with mc.ut.timed_step('read'):
        adata = sc.read(path)

    mc.ut.canonize(adata.X)
    mc.ut.setup(adata, x_name='UMIs')

    LOADED[path] = (adata, expected)
    return adata, expected


@pytest.mark.parametrize('path', glob('../metacells-test-data/*.h5ad'))
def test_find_rare_genes_modules(path: str) -> None:
    adata, expected = _load(path)

    mc.tl.find_rare_genes_modules(adata)

    actual_rare_gene_modules = [
        list(module_gene_names) for module_gene_names
        in mc.ut.get_data(adata, 'rare_gene_modules')
    ]
    expected_rare_gene_modules = expected['rare_gene_modules']

    assert actual_rare_gene_modules == expected_rare_gene_modules


@pytest.mark.parametrize('path', glob('../metacells-test-data/*.h5ad'))
def test_find_noisy_lonely_genes(path: str) -> None:
    adata, expected = _load(path)

    adata = adata[range(20000), :].copy()
    mc.tl.find_high_fraction_genes(adata)
    mc.tl.find_high_normalized_variance_genes(adata)

    mc.pp.track_base_indices(adata)
    bdata = mc.pp.filter_data(adata, ['high_fraction_genes',
                                      'high_normalized_variance_genes'])
    assert bdata is not None

    mc.tl.compute_var_var_similarity(bdata, repeated=False)
    mc.tl.find_lonely_genes(bdata)

    actual_noisy_lonely_genes_mask = np.full(adata.n_vars, False)
    actual_noisy_lonely_genes_mask[mc.ut.get_v_data(bdata, 'var_base_index')] = \
        mc.ut.get_v_data(bdata, 'lonely_genes')

    actual_noisy_lonely_gene_names = \
        sorted(adata.var_names[actual_noisy_lonely_genes_mask])
    expected_noisy_lonely_gene_names = expected['noisy_lonely_genes']

    assert actual_noisy_lonely_gene_names == expected_noisy_lonely_gene_names
