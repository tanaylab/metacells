.PHONY: all for-commit reformat format isort rst unstaged todo mypy build pylint test tox docs
.PHONY: dist clean

.PHONY: coverage flame sum

all: for-commit

for-commit: reformat format isort rst unstaged todo mypy pylint test tox docs

reformat: .clang-formatted
	autopep8 -i -r metacells tests setup.py

.clang-formatted: .clang-format metacells/extensions.cpp
	clang-format -i metacells/extensions.cpp
	@touch .clang-formatted

format:
	@if grep -n '^[ ].*[(%<>/+-]$$\|[^ =][[]$$\|[^:`]`[^:`][^:`]*`[^`]' `git ls-files | grep '\.py$$'`; then false; fi

isort:
	isort metacells/**/*.py tests/*.py setup.py

unstaged:
	@if git status| grep 'not staged' > /dev/null; then git status; false; fi

todo:
	@if grep -i -n 'TODO''X\|reveal_''type' `git ls-files | grep -v pybind11`; then false; fi

mypy:
	mypy metacells tests

build:
	python setup.py build_ext --inplace

pylint: build
	pylint metacells tests

test: build
	@rm -f timing.csv
	pytest -s --cov=metacells tests

tox:
	tox

docs:
	sphinx-build -W docs/source docs/build

rst: README.rst LICENSE.rst

LICENSE.rst: docs/source/license.rst
	cp $? $@

README.rst: prefix.rst docs/source/intro.rst docs/source/install.rst references.rst LICENSE.rst
	cat $? > $@

dist:
	python setup.py sdist

clean:
	rm -rf `cat .gitignore`

coverage: coverage/index.html

coverage/index.html: timing.csv
	rm -rf coverage
	coverage html -d coverage

flame: flame.html

flame.html: timing.csv
	python metacells/scripts/timing.py flame -s < timing.csv \
	| flameview.py --sizename 'Total Elapsed Time' --sortby size > flame.html

sum:
	python metacells/scripts/timing.py sum < timing.csv | column -t -s,
