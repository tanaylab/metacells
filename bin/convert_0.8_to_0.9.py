import anndata as ad
import argparse as ap
import logging
import metacells as mc
import numpy as np

np.seterr(all='raise')
LOG = mc.ut.setup_logger(
    level=logging.INFO,
    long_level_names=False,
    time=True,
    name="convert_old_metacells"
)

parser = ap.ArgumentParser(
    prog="convert_old_metacells",
    description="Convert old metacells to new \"Grand Rename\" format.",
)

parser.add_argument("-fc", "--full-cells-h5ad", required=False)
parser.add_argument("-oc", "--old-cells-h5ad", required=True)
parser.add_argument("-nc", "--new-cells-h5ad", required=True)
parser.add_argument("-om", "--old-metacells-h5ad", required=True)
parser.add_argument("-nm", "--new-metacells-h5ad", required=True)
parser.add_argument("-rs", "--random_seed", default="123456")

args = parser.parse_args()

full_cells_path = args.full_cells_h5ad
old_cells_path = args.old_cells_h5ad
new_cells_path = args.new_cells_h5ad
old_metacells_path = args.old_metacells_h5ad
new_metacells_path = args.new_metacells_h5ad
random_seed = int(args.random_seed)

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

LOG.info(f"Read {old_metacells_path}...")
old_mdata = ad.read_h5ad(old_metacells_path)

metacell_of_cells = mc.ut.get_o_numpy(cdata, "metacell")
metacells_count = np.max(metacell_of_cells) + 1

assert old_mdata.n_obs == metacells_count, \
    f"Metacells file {old_metacells_path} has wrong number of metacells for cells file {old_cells_path}"

LOG.info(f"Rename gene properties...")
for old_name, new_name in [
    ("forbidden_gene", "lateral_gene"),
    ("noisy_lonely_gene", "bursty_lonely_gene")
]:
    if not mc.ut.has_data(cdata, old_name):
        continue
    LOG.info(f"* rename {old_name} to {new_name}")
    genes_mask = mc.ut.get_v_numpy(cdata, old_name)
    del cdata.var[old_name]
    mc.ut.set_v_data(cdata, new_name, genes_mask)

LOG.info(f"Purge old gene properties...")
for name in sorted(cdata.var.keys()):
    if name in ("forbidden_gene", "noisy_lonely_gene", "significant_gene") \
    or name.startswith("pre_"):
        LOG.info(f"* purge {name}")
        del cdata.var[name]
        if name.startswith("pre_") and name[4:] in cdata.var:
            LOG.info(f"* purge {name[4:]}")
            del cdata.var[name[4:]]

LOG.info(f"Keep old gene properties...")
for name in sorted(cdata.var.keys()):
    LOG.info(f"* keep {name}")

LOG.info(f"Purge old cell properties...")
for name in sorted(cdata.obs.keys()):
    if name in ("cell_deviant_votes", "cell_directs", "clean_cell", "dissolved", "outlier", "pile") \
    or name.startswith("pre_"):
        LOG.info(f"* purge {name}")
        del cdata.obs[name]
        if name != "pre_metacell" and name.startswith("pre_") and name[4:] in cdata.obs:
            LOG.info(f"* purge {name[4:]}")
            del cdata.obs[name[4:]]

LOG.info(f"Keep old cell properties...")
for name in sorted(cdata.obs.keys()):
    LOG.info(f"* keep {name}")

LOG.info(f"Collect metacells...")
new_mdata = mc.pl.collect_metacells(cdata)
assert new_mdata.shape == old_mdata.shape

LOG.info(f"Compute for MCView...")
mc.pl.compute_for_mcview(adata=cdata, gdata=new_mdata, random_seed=random_seed)

LOG.info(f"Skip MC gene properties...")
for name in sorted(old_mdata.var.keys()):
    if mc.ut.has_data(new_mdata, name):
        continue
    if name in ("significant_gene", "forbidden_gene", "noisy_lonely_gene") \
    or name.startswith("pre_"):
        LOG.info(f"* skip {name}")
        del old_mdata.var[name]
        if name.startswith("pre_") and name[4:] in cdata.var:
            LOG.info(f"* skip {name[4:]}")
            del old_mdata.var[name[4:]]

LOG.info(f"Copy MC gene properties...")
for name in sorted(old_mdata.var.keys()):
    if mc.ut.has_data(new_mdata, name):
        continue
    LOG.info(f"* copy {name}")
    data = mc.ut.get_v_numpy(old_mdata, name)
    mc.ut.set_v_data(new_mdata, name, data)

LOG.info(f"Copy MC metacell properties...")
for name in sorted(old_mdata.obs.keys()):
    if name in ("significant_gene", "forbidden_gene", "pile", "candidate") \
    or name.startswith("umap_") \
    or mc.ut.has_data(new_mdata, name):
        continue
    LOG.info(f"* copy {name}")
    data = mc.ut.get_o_numpy(old_mdata, name)
    mc.ut.set_o_data(new_mdata, name, data)

if fdata is not None:
    LOG.info(f"Global MC properties...")
    excluded_cells = fdata.n_obs - cdata.n_obs
    excluded_genes = fdata.n_vars - cdata.n_vars
    mc.ut.set_m_data(new_mdata, "excluded_cells", excluded_cells)
    mc.ut.set_m_data(new_mdata, "excluded_genes", excluded_genes)

LOG.info(f"Write {new_cells_path}...")
cdata.write(new_cells_path)

LOG.info(f"Write {new_metacells_path}...")
new_mdata.write(new_metacells_path)

LOG.info(f"Done")
