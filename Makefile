PYTHON ?= python

.PHONY: check test

check:
	PYTHONPATH=src $(PYTHON) -m compileall -q src tests
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
