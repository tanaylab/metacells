.PHONY: all for_commit reformat unstaged todo docs test rst

all: for_commit

for_commit: reformat rst unstaged todo test docs

reformat:
	autopep8 -i -r metacells tests

unstaged:
	@if git status| grep 'not staged' > /dev/null; then git status; false; fi

todo:
	@if grep -i TODO''X `git ls-files`; then false; fi

docs:
	sphinx-build docs/source docs/build

test:
	pytest --cov=metacells
	tox

rst: README.rst LICENSE.rst

LICENSE.rst: docs/source/license.rst
	cp $? $@

README.rst: prefix.rst docs/source/intro.rst docs/source/install.rst references.rst LICENSE.rst
	cat $? > $@
