'''
Test applying functions to real data.
'''

import logging
from glob import glob
from typing import Any, Dict, Tuple

import numpy as np
import scanpy as sc  # type: ignore
import yaml
from anndata import AnnData

import metacells as mc

# pylint: disable=missing-function-docstring

np.seterr(all='raise')
mc.ut.setup_logger(level=logging.WARN)
mc.ut.allow_inefficient_layout(False)
mc.ut.set_processors_count(4)

LOADED: Dict[str, Tuple[AnnData, Dict[str, Any]]] = {}


def _load(path: str) -> Tuple[AnnData, Dict[str, Any]]:
    global LOADED
    if path in LOADED:
        return LOADED[path]

    with open(path[:-4] + 'yaml') as file:
        expected = yaml.safe_load(file)
    with mc.ut.timed_step('read'):
        adata = sc.read(path)

    LOADED[path] = (adata, expected)
    mc.ut.set_name(adata, path.split('/')[-1][:-5])
    return adata, expected


def test_find_rare_gene_modules() -> None:
    for path in glob('../metacells-test-data/*.h5ad'):
        adata, expected = _load(path)

        mc.tl.find_rare_gene_modules(  #
            adata,
            forbidden_gene_names=expected['compute_direct_metacells']['forbidden_gene_names'],
            **expected.get('find_rare_gene_modules', {}))

        module_index = 0
        actual_rare_gene_modules = []
        while mc.ut.has_data(adata, f'rare_gene_module_{module_index}'):
            actual_rare_gene_modules.append(sorted(adata.var_names[  #
                mc.ut.get_v_numpy(adata, f'rare_gene_module_{module_index}')]))
            module_index += 1

        expected_rare_gene_modules = expected['rare_gene_modules']

        assert actual_rare_gene_modules == expected_rare_gene_modules


def test_direct_pipeline() -> None:
    for path in glob('../metacells-test-data/*.h5ad'):
        adata, expected = _load(path)

        mc.ut.log_calc('path', path)
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

        mc.tl.compute_inner_normalized_variance(adata=cdata,
                                                gdata=mdata,
                                                random_seed=123456)

        expected_results = expected['inner_normalized_variance']

        actual_results = \
            np.nanmean(mc.ut.get_vo_proper(mdata, 'inner_normalized_variance',
                                           layout='column_major'))

        # mc.ut.log_calc('PATH', path)
        # mc.ut.log_calc('EXPECT', expected_results)
        # mc.ut.log_calc('ACTUAL', actual_results)
        assert np.allclose([expected_results], [actual_results])
