#!/usr/bin/env python3
"""Convenience entry point for the CDM pipeline demonstration.

Equivalent to `python -m cdm.pipeline`. Run from the repository root:

    python run_demo.py --outdir results

All outputs are stamped SIMULATED. Nothing here is an empirical finding.
"""
from cdm.pipeline import main

if __name__ == "__main__":
    main()
