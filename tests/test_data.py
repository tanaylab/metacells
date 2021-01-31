'''
Test applying functions to real data.
'''

from glob import glob
from typing import Any, Dict, Tuple

import numpy as np  # type: ignore
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

    mc.ut.setup(adata, x_name='UMIs')

    LOADED[path] = (adata, expected)
    return adata, expected


def test_find_rare_gene_modules() -> None:
    for path in glob('../metacells-test-data/*.h5ad'):
        adata, expected = _load(path)

        mc.tl.find_rare_gene_modules(adata,
                                     **expected.get('find_rare_gene_modules', {}))

        actual_rare_gene_modules = [
            list(module_gene_names) for module_gene_names
            in mc.ut.get_data(adata, 'rare_gene_modules')
        ]
        expected_rare_gene_modules = expected['rare_gene_modules']

        assert actual_rare_gene_modules == expected_rare_gene_modules


def test_direct_pipeline() -> None:
    for path in glob('../metacells-test-data/*.h5ad'):
        adata, expected = _load(path)

        pdata = adata[range(6000), :].copy()

        mc.pl.analyze_clean_genes(pdata, random_seed=123456,
                                  **expected.get('analyze_clean_genes', {}))
        mc.pl.pick_clean_genes(pdata)
        mc.pl.analyze_clean_cells(pdata,
                                  **expected.get('analyze_clean_cells', {}))
        mc.pl.pick_clean_cells(pdata)
        cdata = mc.pl.extract_clean_data(pdata)
        assert cdata is not None

        mc.pl.compute_direct_metacells(cdata, random_seed=123456,
                                       **expected.get('compute_direct_metacells', {}))

        mdata = mc.pl.collect_metacells(cdata)

        mc.tl.compute_excess_r2(cdata, random_seed=123456,
                                compatible_size=None, mdata=mdata)

        expected_results = [expected['gene_max_top_r2'],
                            expected['gene_max_top_shuffled_r2'],
                            expected['gene_max_excess_r2']]

        actual_results = [np.nanmean(mc.pp.get_per_var(mdata, mc.ut.nanmax_per,  # type: ignore
                                                       'top_r2').dense),
                          np.nanmean(mc.pp.get_per_var(mdata, mc.ut.nanmax_per,  # type: ignore
                                                       'top_shuffled_r2').dense),
                          np.nanmean(mc.pp.get_per_var(mdata, mc.ut.nanmax_per,  # type: ignore
                                                       'excess_r2').dense)]

        assert np.allclose(expected_results, actual_results)
