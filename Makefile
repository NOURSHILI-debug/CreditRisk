PYTHON ?= python

.PHONY: install lint test ci

install:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

lint:
	$(PYTHON) -m ruff check src tests

test:
	$(PYTHON) -m pytest -q

ci: lint test
