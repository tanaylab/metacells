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
    with mc.ut.step('read'):
        adata = sc.read(data_path)
    mc.ut.canonize(adata.X)
    mc.ut.set_x_name(adata, 'UMIs')
    mc.pp.find_rare_genes_modules(adata)
    actual_modules = [
        list(module_gene_names) for module_gene_names
        in mc.ut.get_metadata(adata, 'rare_gene_modules')
    ]
    expected_modules = expected['rare_gene_modules']
    assert actual_modules == expected_modules
