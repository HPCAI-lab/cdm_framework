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
