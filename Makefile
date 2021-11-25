NAME = metacells

MAX_LINE_LENGTH = 120

ALL_SOURCE_FILES = $(shell git ls-files)

ALL_PACKAGE_FILES = $(filter-out vignettes/%, $(filter-out docs/%, $(ALL_SOURCE_FILES)))

PY_SOURCE_FILES = $(filter %.py, $(ALL_SOURCE_FILES))

PY_PACKAGE_FILES = $(filter-out vignettes/%, $(filter-out docs/%, $(PY_SOURCE_FILES)))

RST_SOURCE_FILES = $(filter %.rst, $(ALL_SOURCE_FILES))

DOCS_SOURCE_FILES = $(filter docs/%, $(ALL_SOURCE_FILES))

H_SOURCE_FILES = $(filter %.h, $(ALL_SOURCE_FILES))

CPP_SOURCE_FILES = $(filter %.cpp, $(ALL_SOURCE_FILES))

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help.replace('TODO-', 'TODO')))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-make clean-build clean-pyc clean-test clean-vignettes clean-docs  ## remove all build, test, coverage and Python artifacts

clean-make:
	rm -fr .make.*

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	rm -fr metacells/*.so

clean-pyc:
	find . -name .mypy_cache -exec rm -fr {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

clean-vignettes:
	rm -f vignettes/*.rst
	rm -fr vignettes/*_files
	rm -f vignettes/Manual_Analysis.ipynb vignettes/Seurat_Analysis.ipynb

clean-docs:
	rm -fr docs/_build

TODO = todo$()x

pc: is_dev $(TODO) history format smells dist pytest docs staged tox  ## check everything before commit

ci: history format smells dist docs tox  ## check everything in a CI server

history:  ## check to-be-done version is described in HISTORY.rst
	@version=`grep 'current_version =' setup.cfg | sed 's/.* //;s/.dev.*//;'`; \
	if grep -q "^$$version" HISTORY.rst; \
	then true; \
	else \
	    echo 'No entry in HISTORY.rst (run `make start_history`).'; \
	    false; \
	fi

staged:  ## check everything is staged for git commit
	@if git status . | grep -q 'Changes not staged\|Untracked files'; \
	then \
	    git status; \
	    echo 'There are unstaged changes (run `git add .`).'; \
	    false; \
	else true; \
	fi

format: trailingspaces linebreaks backticks fstrings isort black flake8 clang-format ## check code format

trailingspaces: .make.trailingspaces  ## check for trailing spaces

REAL_SOURCE_FILES = \
    $(filter-out %.png, \
    $(filter-out %.svg, \
    $(filter-out vignettes/%.rst, \
    $(filter-out docs/Manual_Analysis.rst, \
    $(filter-out docs/Metacells_Vignette.rst, \
    $(filter-out docs/Seurat_Analysis.rst, \
    $(ALL_SOURCE_FILES)))))))

# TODO: Remove setup.cfg exception when bumpversion is fixed.
SP_SOURCE_FILES = $(filter-out setup.cfg, $(REAL_SOURCE_FILES))

.make.trailingspaces: $(SP_SOURCE_FILES)
	@echo "trailingspaces"
	@if grep -Hn '\s$$' $(SP_SOURCE_FILES); \
	then \
	    echo 'Files contain trailing spaces (run `make reformat` or `make stripspaces`).'; \
	    false; \
	else true; \
	fi
	touch $@

linebreaks: .make.linebreaks  ## check line breaks in Python code

.make.linebreaks: $(PY_SOURCE_FILES)
	@echo "linebreaks"
	@if grep -Hn "[^=*][^][/<>\"'a-zA-Z0-9_,:()#}{.?!\\=\`+-]$$" $(PY_SOURCE_FILES) | grep -v -- '--$$\|import \*$$'; \
	then \
	    echo 'Files wrap lines after instead of before an operator (fix manually).'; \
	    false; \
	fi
	touch $@

backticks: .make.backticks  ## check usage of backticks in documentation

BT_SOURCE_FILES = \
    $(filter-out docs/Metacells_Vignette.rst, \
    $(filter-out docs/Manual_Analysis.rst, \
    $(filter-out docs/Seurat_Analysis.rst, \
    $(RST_SOURCE_FILES))))

.make.backticks: $(PY_SOURCE_FILES) $(BT_SOURCE_FILES)
	@echo "backticks"
	@OK=true; \
	for FILE in $(PY_SOURCE_FILES) $(BT_SOURCE_FILES); \
	do \
	    if sed 's/``\([^`]*\)``/\1/g;s/:`\([^`]*\)`/:\1/g;s/`\([^`]*\)`_/\1_/g' "$$FILE" \
	    | grep --label "$$FILE" -n -H '`' \
	    | sed 's//`/g' \
	    | grep '.'; \
	    then OK=false; \
	    fi; \
	done; \
	if $$OK; \
	then true; \
	else \
	    echo 'Documentation contains invalid ` markers (fix manually).'; \
	    false; \
	fi
	touch $@

fstrings: .make.fstrings  ## check f-strings in Python code

.make.fstrings: $(PY_SOURCE_FILES)
	@echo "fstrings"
	@if grep -Hn '^[^"]*\("\([^"]\|\\"\)*"[^"]*\)*[^f]"\([^"]\|\\"\)*{' $(PY_SOURCE_FILES); \
	then \
	    echo 'Strings appear to be f-strings, but are not (fix manually).'; \
	    false; \
	fi
	touch $@

isort: .make.isort  ## check imports with isort

.make.isort: $(PY_SOURCE_FILES)
	isort --line-length $(MAX_LINE_LENGTH) --force-single-line-imports --check $(NAME) tests
	touch $@

$(TODO): .make.$(TODO)  ## check there are no leftover TODO-X

.make.$(TODO): $(REAL_SOURCE_FILES)
	@echo 'grep -n -i $(TODO) `git ls-files | grep -v pybind11`'
	@if grep -n -i $(TODO) `git ls-files | grep -v pybind11`; \
	then \
	    echo "Files contain $(TODO) markers (fix manually)."; \
	    false; \
	else true; \
	fi
	touch $@

black: .make.black  ## check format with black

.make.black: $(PY_SOURCE_FILES)
	black --line-length $(MAX_LINE_LENGTH) --check $(NAME) tests
	touch $@

flake8: .make.flake8  ## check format with flake8

.make.flake8: $(PY_SOURCE_FILES)
	flake8 --max-line-length $(MAX_LINE_LENGTH) --ignore E203,F401,F403,W503 $(NAME) tests
	touch $@

clang-format: .make.clang-format  ## check format with clang-format

.make.clang-format: .clang-format $(H_SOURCE_FILES) $(CPP_SOURCE_FILES)
	@echo "clang-format"
	@if clang-format --dry-run -Werror $(H_SOURCE_FILES) $(CPP_SOURCE_FILES) 2> /dev/null; \
	then true; \
	else \
	    echo "clang-format would like to make changes"; \
	    false; \
	fi
	touch $@

reformat: stripspaces isortify blackify clang-reformat  ## reformat code

stripspaces:  # strip trailing spaces
	@echo stripspaces
	@for FILE in $$(grep -l '\s$$' $$(git ls-files | grep -v setup.cfg)); \
	do sed -i -s 's/\s\s*$$//' $$FILE; \
	done

isortify:  ## sort imports with isort
	isort --line-length $(MAX_LINE_LENGTH) --force-single-line-imports $(NAME) tests

blackify:  ## reformat with black
	black --line-length $(MAX_LINE_LENGTH) $(NAME) tests

clang-reformat: $(H_SOURCE_FILES) $(CPP_SOURCE_FILES)
	@echo "clang-format -i"
	@clang-format -i $(H_SOURCE_FILES) $(CPP_SOURCE_FILES)

smells: mypy pylint  ## check for code smells

pylint: .make.pylint  ## check code with pylint

.make.pylint: $(PY_SOURCE_FILES)
	pylint --max-line-length $(MAX_LINE_LENGTH) $(NAME) tests
	touch $@

mypy: .make.mypy  ## check code with mypy

.make.mypy: $(PY_SOURCE_FILES)
	mypy $(NAME) tests
	touch $@

pytest: .make.pytest  ## run tests on the active Python with pytest

.make.pytest: .make.build
	pytest -s --cov=$(NAME) --cov-report=html --cov-report=term --no-cov-on-fail tests
	touch $@

tox: .make.tox  ## run tests on a clean Python version with tox

.make.tox: $(PY_PACKAGE_FILES) $(H_SOURCE_FILES) $(CPP_SOURCE_FILES) tox.ini
	tox
	touch $@

.PHONY: docs
docs: .make.docs  ## generate HTML documentation

RST_GENERATED_FILES = \
    docs/Metacells_Vignette.rst \
    docs/Manual_Analysis.rst \
    docs/Seurat_Analysis.rst \
    docs/timing_script.rst

.make.docs: $(DOCS_SOURCE_FILES) $(PY_PACKAGE_FILES) $(RST_SOURCE_FILES) $(RST_GENERATED_FILES)
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	@echo "Results in docs/_build/html/index.html"
	touch $@

build: .make.build  ## build the C++ extensions

.make.build: $(PY_PACKAGE_FILES) $(H_SOURCE_FILES) $(CPP_SOURCE_FILES)
	python setup.py build_ext --inplace
	python setup.py build
	touch $@

docs/timing_script.rst: \
    docs/timing_script.part.1 \
    docs/timing_script.part.2 \
    docs/timing_script.part.3 \
    docs/timing_script.part.4 \
    metacells/scripts/timing.py
	( cat docs/timing_script.part.1 \
	; python metacells/scripts/timing.py --help 2>&1 \
	| sed 's/timing.py/metacells_timing.py/;s/^\(.\)/    \1/;s/`/``/g' \
	; cat docs/timing_script.part.2 \
	; python metacells/scripts/timing.py combine --help 2>&1 \
	| sed 's/timing.py/metacells_timing.py/;s/^\(.\)/    \1/;s/`/``/g' \
	; cat docs/timing_script.part.3 \
	; python metacells/scripts/timing.py sum --help 2>&1 \
	| sed 's/timing.py/metacells_timing.py/;s/^\(.\)/    \1/;s/`/``/g' \
	; cat docs/timing_script.part.4 \
	; python metacells/scripts/timing.py flame --help 2>&1 \
	| sed 's/timing.py/metacells_timing.py/;s/^\(.\)/    \1/;s/`/``/g' \
	; cat docs/timing_script.part.5 \
	) > $@


docs/Metacells_Vignette.rst: vignettes/Metacells_Vignette.rst
	sed 's/ [ ]*$$//;s/parsed-literal::/code::/;s:Metacells_Vignette_files/::;s/:: ipython3/:: python/;' \
	    vignettes/Metacells_Vignette.rst > docs/Metacells_Vignette.rst
	cp vignettes/Metacells_Vignette_files/*.svg docs

docs/Manual_Analysis.rst: vignettes/Manual_Analysis.rst
	sed 's/ [ ]*$$//;s/parsed-literal::/code::/;s:Manual_Analysis_files/::;s/:: ipython3/:: python/;' \
	    vignettes/Manual_Analysis.rst > docs/Manual_Analysis.rst
	cp vignettes/Manual_Analysis_files/*.svg docs

docs/Seurat_Analysis.rst: vignettes/Seurat_Analysis.rst
	sed 's/ [ ]*$$//;s/parsed-literal::/code::/;s:Seurat_Analysis_files/::;s/:: ipython3/:: python/;' \
	    vignettes/Seurat_Analysis.rst > docs/Seurat_Analysis.rst
	cp vignettes/Seurat_Analysis_files/*.svg docs

vignettes/Metacells_Vignette.rst: .make.build vignettes/Metacells_Vignette.ipynb vignettes/pbmc163k.h5ad
	rm -rf vignettes/Metacells_Vignette_files
	export PYTHONPATH=$$PWD && cd vignettes && jupyter nbconvert --execute --to=rst Metacells_Vignette.ipynb

vignettes/Manual_Analysis.rst: .make.build vignettes/Manual_Analysis.png.ipynb vignettes/Metacells_Vignette.rst
	rm -rf vignettes/Manual_Analysis_files
	sed 's:image/png:image/svg+xml:' vignettes/Manual_Analysis.png.ipynb > vignettes/Manual_Analysis.ipynb
	export PYTHONPATH=$$PWD && cd vignettes && jupyter nbconvert --execute --to=rst Manual_Analysis.ipynb

vignettes/Seurat_Analysis.rst: .make.build vignettes/Seurat_Analysis.png.ipynb vignettes/Metacells_Vignette.rst
	rm -rf vignettes/Seurat_Analysis_files
	sed 's:image/png:image/svg+xml:' vignettes/Seurat_Analysis.png.ipynb > vignettes/Seurat_Analysis.ipynb
	export PYTHONPATH=$$PWD && cd vignettes && jupyter nbconvert --execute --to=rst Seurat_Analysis.ipynb

vignettes/pbmc163k.h5ad:
	curl https://www.wisdom.weizmann.ac.il/~atanay/metac_data/pbmc163k.h5ad.gz | gunzip -c > vignettes/pbmc163k.h5ad

committed:  staged ## check everything is committed in git
	@if [ -z "$$(git status --short)" ]; \
	then true; \
	else \
	    git status; \
	    echo "There are uncommitted changes (run `git commit -m ...`)." \
	    false; \
	fi

install: committed clean  ## install the package into the active Python
	python setup.py install

dist: .make.dist  ## builds the release distribution package

.make.dist: staged $(ALL_PACKAGE_FILES)
	rm -fr dist/
	python setup.py sdist
	twine check dist/*
	touch $@

upload: committed is_not_dev .make.dist  ## upload the release distribution package
	twine upload dist/*

current_version:  # report the current version number
	@grep 'current_version =' setup.cfg

start_patch: committed  ## start working on the next patch version
	@if grep -q 'current_version.*dev' setup.cfg; \
	then \
	    read -p "Skip over releasing the current development version [n]? " answer; \
	    case "$$answer" in \
	        y|yes|Y|Yes|YES) bumpversion patch;; \
	        *) false;; \
	    esac; \
	else bumpversion patch; \
	fi
	@grep 'current_version =' setup.cfg

start_minor: committed  ## start working on the next minor version
	@if grep -q 'current_version.*dev' setup.cfg; \
	then \
	    read -p "Skip over releasing the current development version [n]? " answer; \
	    case "$$answer" in \
	        y|yes|Y|Yes|YES) bumpversion minor;; \
	        *) false ;; \
	    esac; \
	else bumpversion minor; \
	fi
	@grep 'current_version =' setup.cfg

start_major: committed  ## start working on the next major version
	@if grep -q 'current_version.*dev' setup.cfg; \
	then \
	    read -p "Skip over releasing the current development version [n]? " answer; \
	    case "$$answer" in \
	        y|yes|Y|Yes|YES) bumpversion major;; \
	        *) false ;; \
	    esac; \
	else bumpversion major; \
	fi
	@grep 'current_version =' setup.cfg

start_history: is_dev  ## append a history section for the development version
	@version=`grep 'current_version =' setup.cfg | sed 's/.* //;s/.dev.*//;'`; \
	if grep -q "^$$version (WIP)\$$" HISTORY.rst; \
	then true; \
	else \
	    echo >> HISTORY.rst; \
	    echo "$$version (WIP)" >> HISTORY.rst; \
	    echo "$$version" | sed 's/./-/g' >> HISTORY.rst; \
	    echo >> HISTORY.rst; \
	    echo "* ..." >> HISTORY.rst; \
	fi

bump_dev: committed is_dev  ## bump the development version indicator
	bumpversion dev
	@grep 'current_version =' setup.cfg

done_dev: committed done_history is_dev dist  ## remove the development version indicator
	bumpversion rel
	@grep 'current_version =' setup.cfg

done_history:  ## check to-be-done version is described in HISTORY.rst
	@version=`grep 'current_version =' setup.cfg | sed 's/.* //;s/.dev.*//;'`; \
	if grep -q "^$$version\$$" HISTORY.rst; \
	then true; \
	else \
	    echo "No finalized entry in HISTORY.rst (fix manually)."; \
	    false; \
	fi

is_not_dev:
	@if grep -q 'current_version.*dev' setup.cfg; \
	then \
	    echo "`grep 'current_version =' setup.cfg` is a development version."; \
	    false; \
	fi

is_dev:
	@if grep -q 'current_version.*dev' setup.cfg; \
	then true; \
	else \
	    echo "`grep 'current_version =' setup.cfg` is not a development version."; \
	    false; \
	fi

tags: $(PY_PACKAGE_FILES)  ## generate a tags file for vi
	ctags $(PY_PACKAGE_FILES)

flame: flame.html  ## generate a flame graph from a timing.csv file

flame.html: timing.csv
	python metacells/scripts/timing.py combine timing.csv \
	| python metacells/scripts/timing.py flame -s \
	| flameview.py --sizename 'Total Elapsed Time' --sortby size \
	> flame.html

sum:  ## summerize a timing.csv file
	python metacells/scripts/timing.py combine timing.csv \
	| python metacells/scripts/timing.py sum \
	| column -t -s,
