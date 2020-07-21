.PHONY: all for-commit reformat format isort rst unstaged todo mypy pylint build test tox docs
.PHONY: dist clean

all: for-commit

for-commit: reformat format isort rst unstaged todo mypy pylint test tox docs

reformat:
	autopep8 -i -r metacells tests setup.py
	clang-format -i metacells/extensions.cpp

format:
	@if grep -n '^[ ].*[(%<>/+-]$$\|[^ =][[]$$\|[^`]`[^`]' `git ls-files | grep '\.py$$'`; then false; fi

isort:
	isort metacells/**/*.py tests/*.py setup.py

unstaged:
	@if git status| grep 'not staged' > /dev/null; then git status; false; fi

todo:
	@if grep -i -n TODO''X `git ls-files | grep -v pybind11`; then false; fi

mypy:
	mypy metacells tests

pylint:
	pylint metacells tests

build:
	python setup.py build_ext --inplace

test: build
	pytest -s --cov=metacells tests

tox:
	tox

docs:
	sphinx-build docs/source docs/build

rst: README.rst LICENSE.rst

LICENSE.rst: docs/source/license.rst
	cp $? $@

README.rst: prefix.rst docs/source/intro.rst docs/source/install.rst references.rst LICENSE.rst
	cat $? > $@

dist:
	python setup.py sdist

clean:
	rm -rf `cat .gitignore`
