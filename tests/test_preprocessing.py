'''
Test the preprocessing functions.
'''

from glob import glob

import pytest  # type: ignore
import scanpy as sc  # type: ignore
import yaml

import metacells as mc

# pylint: disable=missing-docstring


@pytest.mark.parametrize('data_path', glob('../metacells-test-data/*.h5ad'))
def test_find_rare_genes_modules(data_path):
    with open(data_path[:-4] + 'yaml') as file:
        expected = yaml.safe_load(file)
    adata = sc.read(data_path)
    adata.uns['value'] = 'UMI'
    _cells_data, _genes_data, modules_data = \
        mc.pp.find_rare_genes_modules(adata)
    actual_modules = [list(module) for module in modules_data]
    expected_modules = expected['rare_gene_modules']
    assert actual_modules == expected_modules
