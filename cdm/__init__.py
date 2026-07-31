"""Cognitive Diversity Metrics (CDM) analysis pipeline.

A working implementation of the full CDM measurement-validation pipeline,
demonstrated on data from a declared generative model. See the project README
and docs/methodology.md for the integrity framing: structural parameters are
inputs, and every reported statistic emerges from them plus sampling noise.
"""

from cdm import model_core, agreement  # noqa: F401

__all__ = ["model_core", "agreement", "pipeline"]
__version__ = "1.0.0"
