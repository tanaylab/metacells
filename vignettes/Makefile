all: \
    ../sphinx/Metacells_Vignette.rst \
    ../sphinx/Manual_Analysis.rst \
    ../sphinx/Seurat_Analysis.rst

../sphinx/Metacells_Vignette.rst: Metacells_Vignette.rst
	sed 's/ [ ]*$$//;s/parsed-literal::/code::/;s:Metacells_Vignette_files/::;' \
	    Metacells_Vignette.rst > ../sphinx/Metacells_Vignette.rst
	cp Metacells_Vignette_files/*.svg ../sphinx

../sphinx/Manual_Analysis.rst: Manual_Analysis.rst
	sed 's/ [ ]*$$//;s/parsed-literal::/code::/;s:Manual_Analysis_files/::;' \
	    Manual_Analysis.rst > ../sphinx/Manual_Analysis.rst
	cp Manual_Analysis_files/*.svg ../sphinx

../sphinx/Seurat_Analysis.rst: Seurat_Analysis.rst
	sed 's/ [ ]*$$//;s/parsed-literal::/code::/;s:Seurat_Analysis_files/::;' \
	    Seurat_Analysis.rst > ../sphinx/Seurat_Analysis.rst
	cp Seurat_Analysis_files/*.svg ../sphinx

Metacells_Vignette.rst: Metacells_Vignette.ipynb pbmc163k.h5ad
	rm -rf Metacells_Vignette_files
	jupyter nbconvert --execute --to=rst Metacells_Vignette.ipynb

Manual_Analysis.rst: Manual_Analysis.ipynb Metacells_Vignette.rst
	rm -rf Manual_Analysis_files
	mv Manual_Analysis.ipynb Manual_Analysis.ipynb.original
	sed 's:image/png:image/svg+xml:' Manual_Analysis.ipynb.original > Manual_Analysis.ipynb
	jupyter nbconvert --execute --to=rst Manual_Analysis.ipynb
	mv Manual_Analysis.ipynb.original Manual_Analysis.ipynb

Seurat_Analysis.rst: Seurat_Analysis.ipynb Metacells_Vignette.rst
	rm -rf Seurat_Analysis_files
	mv Seurat_Analysis.ipynb Seurat_Analysis.ipynb.original
	sed 's:image/png:image/svg+xml:' Seurat_Analysis.ipynb.original > Seurat_Analysis.ipynb
	jupyter nbconvert --execute --to=rst Seurat_Analysis.ipynb
	mv Seurat_Analysis.ipynb.original Seurat_Analysis.ipynb

pbmc163k.h5ad:
	echo "curl http://www.wisdom.weizmann.ac.il/~atanay/metac_data/pbmc163k.h5ad.gz | gzcat -d > $?"
