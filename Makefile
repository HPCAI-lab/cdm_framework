# CDM pipeline — convenience targets
.PHONY: help install demo test clean

help:
	@echo "make install   Install runtime dependencies"
	@echo "make demo      Run the pipeline; writes stamped outputs to results/"
	@echo "make test      Run the test suite"
	@echo "make clean     Remove local run artifacts"

install:
	python -m pip install -r requirements.txt

demo:
	python -m cdm.pipeline --outdir results

test:
	python -m pytest -q

clean:
	rm -rf results out __pycache__ cdm/__pycache__ tests/__pycache__ .pytest_cache

pilot:
	python run_pilot.py --logs $(LOGS) --profile $(PROFILE) --outdir results_pilot

controls:
	python run_controls.py --outdir results_controls --reps 500 --sweep-reps 50

controls-quick:
	python run_controls.py --quick --outdir results_controls

ingest-dry:
	python -m cdm.ingest --format $(FORMAT) --input $(INPUT) --dry-run

PROFILE ?= hpc
.PHONY: pilot controls controls-quick ingest-dry
