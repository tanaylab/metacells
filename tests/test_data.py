'''
Test applying functions to real data.
'''

from glob import glob
from typing import Any, Dict, Tuple

import numpy as np  # type: ignore
import pytest  # type: ignore
import scanpy as sc  # type: ignore
import yaml
from anndata import AnnData

import metacells as mc

# pylint: disable=missing-function-docstring


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
def test_find_rare_gene_modules(path: str) -> None:
    adata, expected = _load(path)

    mc.tl.find_rare_gene_modules(adata)

    actual_rare_gene_modules = [
        list(module_gene_names) for module_gene_names
        in mc.ut.get_data(adata, 'rare_gene_modules')
    ]
    expected_rare_gene_modules = expected['rare_gene_modules']

    assert actual_rare_gene_modules == expected_rare_gene_modules


@pytest.mark.parametrize('path', glob('../metacells-test-data/*.h5ad'))
def test_direct_pipeline(path: str) -> None:
    adata, expected = _load(path)

    pdata = adata[range(6000), :].copy()

    cdata = mc.pl.extract_clean_data(pdata, random_seed=123456,
                                     **expected['clean_kwargs'])
    assert cdata is not None

    mc.pl.compute_direct_metacells(cdata, random_seed=123456,
                                   **expected['compute_kwargs'])

    mdata = mc.pl.collect_metacells(cdata)

    mc.tl.compute_excess_r2(cdata, random_seed=123456, mdata=mdata)

    assert np.allclose(expected['gene_max_top_r2'],
                       np.nanmean(mc.ut.get_v_data(cdata, 'gene_max_top_r2')))

    assert np.allclose(expected['gene_max_top_shuffled_r2'],
                       np.nanmean(mc.ut.get_v_data(cdata, 'gene_max_top_shuffled_r2')))

    assert np.allclose(expected['gene_max_excess_r2'],
                       np.nanmean(mc.ut.get_v_data(cdata, 'gene_max_excess_r2')))
