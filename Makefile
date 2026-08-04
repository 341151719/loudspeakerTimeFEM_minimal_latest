PYTHON ?= python3
CONFIG ?= configs/transient_70Hz_nonlinear_comsol_physical_abc.json
OUTDIR ?= runs/transient_70Hz_nonlinear_comsol_physical_abc
SCRATCH ?= /tmp/loudspeaker-time-fem-runs

.PHONY: inspect test self-test run validate manifest

inspect:
	PYTHONPATH=src $(PYTHON) cli.py inspect --config $(CONFIG)

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

self-test:
	PYTHONPATH=src $(PYTHON) self_test.py

run:
	PYTHONPATH=src $(PYTHON) cli.py run --config $(CONFIG) --outdir $(OUTDIR) --scratch-root $(SCRATCH)

validate:
	PYTHONPATH=src $(PYTHON) comsol_validation/compare_comsol_python.py --help
	PYTHONPATH=src $(PYTHON) comsol_validation/compare_comsol_convergence.py --help

manifest:
	PYTHONPATH=src $(PYTHON) tools/build_project_manifest.py
