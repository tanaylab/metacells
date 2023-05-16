# pylint: disable=invalid-name,logging-fstring-interpolation

import argparse as ap
import logging

import anndata as ad  # type: ignore
import numpy as np

import metacells as mc

parser = ap.ArgumentParser(
    prog="convert_old_metacells",
    description='Convert old metacells to new "Grand Rename" format.',
)

parser.add_argument("-ll", "--log-level", default="INFO")
parser.add_argument("-fc", "--full-cells-h5ad", required=False)
parser.add_argument("-oc", "--old-cells-h5ad", required=True)
parser.add_argument("-nc", "--new-cells-h5ad", required=True)
parser.add_argument("-om", "--old-metacells-h5ad", required=True)
parser.add_argument("-nm", "--new-metacells-h5ad", required=True)
parser.add_argument("-rs", "--random_seed", default="123456")
parser.add_argument("-un", "--unsafe-names", default=False, action="store_true")
parser.add_argument("-u,", "--umsafe-names", default=False, action="store_true")
parser.add_argument("-ma", "--metacells-algorithm", default="metacells.0.8.0")
parser.add_argument("-ku", "--keep-umap", default=False, action="store_true")

args = parser.parse_args()

log_level = getattr(logging, args.log_level.upper(), None)
if log_level is None:
    raise ValueError(f"Invalid log level: {args.log_level}")

np.seterr(all="raise")
LOG = mc.ut.setup_logger(level=log_level, long_level_names=False, time=True, name="convert_old_metacells")

full_cells_path = args.full_cells_h5ad
old_cells_path = args.old_cells_h5ad
new_cells_path = args.new_cells_h5ad
old_metacells_path = args.old_metacells_h5ad
new_metacells_path = args.new_metacells_h5ad
random_seed = int(args.random_seed)
unsafe_names = args.unsafe_names
umsafe_names = args.umsafe_names
assert not unsafe_names or not umsafe_names
metacells_algorithm = args.metacells_algorithm
keep_umap = args.keep_umap

assert old_cells_path != new_cells_path, f"Cowardly refuse to overwrite the cells file {old_cells_path}"
assert old_metacells_path != new_metacells_path, f"Cowardly refuse to overwrite the metacells file {old_metacells_path}"
assert old_cells_path != new_metacells_path, f"Old cells file {old_cells_path} is same as new metacells file"
assert old_metacells_path != new_cells_path, f"Old metacells file {old_metacells_path} is same as new cells file"

if full_cells_path is not None:
    LOG.info(f"Read {full_cells_path}...")
    fdata = ad.read_h5ad(full_cells_path)
else:
    fdata = None

LOG.info(f"Read {old_cells_path}...")
cdata = ad.read_h5ad(old_cells_path)

if fdata is not None:
    assert len(fdata.var_names) == len(cdata.var_names) and list(fdata.var_names) == list(
        cdata.var_names
    ), f"different genes in: {full_cells_path} and: {old_cells_path}"

LOG.info(f"Read {old_metacells_path}...")
old_mdata = ad.read_h5ad(old_metacells_path)
assert not mc.ut.has_data(old_mdata, "metacells_algorithm")

assert len(cdata.var_names) == len(old_mdata.var_names) and list(cdata.var_names) == list(
    old_mdata.var_names
), f"different genes in: {old_cells_path} and: {old_metacells_path}"

metacell_of_cells = mc.ut.get_o_numpy(cdata, "metacell")
metacells_count = np.max(metacell_of_cells) + 1

assert (
    old_mdata.n_obs == metacells_count
), f"Metacells file {old_metacells_path} has wrong number of metacells for cells file {old_cells_path}"

LOG.info("Rename gene properties...")
for old_name, new_name in [("forbidden_gene", "lateral_gene"), ("noisy_lonely_gene", "bursty_lonely_gene")]:
    if not mc.ut.has_data(cdata, old_name):
        continue
    LOG.info(f"* rename {old_name} to {new_name}")
    genes_mask = mc.ut.get_v_numpy(cdata, old_name)
    del cdata.var[old_name]
    mc.ut.set_v_data(cdata, new_name, genes_mask)

LOG.info("Purge old gene properties...")
for name in sorted(cdata.var.keys()):
    if name in ("forbidden_gene", "noisy_lonely_gene", "significant_gene") or name.startswith("pre_"):
        LOG.info(f"* purge {name}")
        del cdata.var[name]
        if name.startswith("pre_") and name[4:] in cdata.var:
            LOG.info(f"* purge {name[4:]}")
            del cdata.var[name[4:]]

LOG.info("Keep old gene properties...")
for name in sorted(cdata.var.keys()):
    LOG.info(f"* keep {name}")

LOG.info("Purge old cell properties...")
for name in sorted(cdata.obs.keys()):
    if name in ("cell_deviant_votes", "cell_directs", "clean_cell", "dissolved", "outlier", "pile") or name.startswith(
        "pre_"
    ):
        LOG.info(f"* purge {name}")
        del cdata.obs[name]
        if name != "pre_metacell" and name.startswith("pre_") and name[4:] in cdata.obs:
            LOG.info(f"* purge {name[4:]}")
            del cdata.obs[name[4:]]

LOG.info("Keep old cell properties...")
for name in sorted(cdata.obs.keys()):
    LOG.info(f"* keep {name}")

LOG.info("Collect metacells...")
new_mdata = mc.pl.collect_metacells(cdata, name=mc.ut.get_name(old_mdata) or "metacells", random_seed=random_seed)
assert new_mdata.shape == old_mdata.shape, f"NEW SHAPE {new_mdata.shape} != OLD SHAPE {old_mdata.shape}"
mc.ut.set_m_data(new_mdata, "metacells_algorithm", metacells_algorithm)

if unsafe_names:
    new_mdata.obs_names = [obs_name[1:-3] for obs_name in new_mdata.obs_names]
if umsafe_names:
    new_mdata.obs_names = [obs_name[:-3] for obs_name in new_mdata.obs_names]

LOG.info("Compute for MCView...")

if keep_umap:
    assert mc.ut.has_data(old_mdata, "umap_x")
    assert mc.ut.has_data(old_mdata, "umap_y")
    assert mc.ut.has_data(old_mdata, "umap_u")
    assert mc.ut.has_data(old_mdata, "umap_v")
    assert mc.ut.has_data(old_mdata, "umap_w")

mc.pl.compute_for_mcview(
    adata=cdata,
    gdata=new_mdata,
    compute_umap_by_markers_2=None if keep_umap else {},
    compute_umap_by_markers_3=None if keep_umap else {},
    random_seed=random_seed,
)

if keep_umap:
    obs_outgoing_weights = mc.ut.get_oo_proper(old_mdata, "obs_outgoing_weights")
    mc.ut.set_oo_data(new_mdata, "obs_outgoing_weights", obs_outgoing_weights)

LOG.info("Skip MC gene properties...")
for name in sorted(old_mdata.var.keys()):
    if mc.ut.has_data(new_mdata, name):
        continue
    if name in ("significant_gene", "forbidden_gene", "noisy_lonely_gene") or name.startswith("pre_"):
        LOG.info(f"* skip {name}")
        del old_mdata.var[name]
        if name.startswith("pre_") and name[4:] in cdata.var:
            LOG.info(f"* skip {name[4:]}")
            del old_mdata.var[name[4:]]

LOG.info("Copy MC gene properties...")
for name in sorted(old_mdata.var.keys()):
    if mc.ut.has_data(new_mdata, name):
        continue
    LOG.info(f"* copy {name}")
    data = mc.ut.get_v_numpy(old_mdata, name)
    mc.ut.set_v_data(new_mdata, name, data)

LOG.info("Copy MC metacell properties...")
for name in sorted(old_mdata.obs.keys()):
    if (
        name in ("significant_gene", "forbidden_gene", "pile", "candidate")
        or (name.startswith("umap_") and not keep_umap)
        or mc.ut.has_data(new_mdata, name)
    ):
        continue

    old_name = name
    if old_name.startswith("umap_"):
        new_name = name[5:]
    else:
        new_name = name

    LOG.info(f"* copy {name}")
    data = mc.ut.get_o_numpy(old_mdata, old_name)
    mc.ut.set_o_data(new_mdata, new_name, data)

if fdata is not None:
    LOG.info("Global MC properties...")
    excluded_cells = fdata.n_obs - cdata.n_obs
    excluded_genes = fdata.n_vars - cdata.n_vars
    mc.ut.set_m_data(new_mdata, "excluded_cells", excluded_cells)
    mc.ut.set_m_data(new_mdata, "excluded_genes", excluded_genes)

LOG.info(f"Write {new_cells_path}...")
cdata.write(new_cells_path)

LOG.info(f"Write {new_metacells_path}...")
new_mdata.write(new_metacells_path)

LOG.info("Done")
