"""Sanity tests for the CDM pipeline.

These check that the machinery behaves correctly on simulated data. They do not
assert empirical claims; the point is that the pipeline recovers known injected
structure and that the statistics are well-formed.
"""
import numpy as np

from cdm import model_core as mc
from cdm import agreement as ag
from cdm.pipeline import cronbach_alpha, run_pipeline


def test_simulate_shape():
    df, raters_df, ta_df = mc.simulate(n_per_group=10, seed=20260617)
    assert len(df) == 30                      # 3 disciplines x 10
    for dim in mc.DIMENSIONS:
        assert dim in df.columns


def test_weighted_kappa_perfect_agreement():
    a = [1, 2, 3, 4, 5, 3, 2]
    assert abs(ag.weighted_kappa(a, a) - 1.0) < 1e-9


def test_weighted_kappa_bounds():
    rng = np.random.default_rng(0)
    a = rng.integers(1, 6, size=200)
    b = rng.integers(1, 6, size=200)
    k = ag.weighted_kappa(a, b)
    assert -1.0 <= k <= 1.0


def test_cronbach_alpha_in_unit_interval():
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(200, 1))
    items = latent + rng.normal(scale=0.5, size=(200, 5))   # correlated items
    alpha = cronbach_alpha(items)
    assert 0.0 <= alpha <= 1.0


def test_pipeline_runs_end_to_end():
    df, raters_df, ta_df = mc.simulate(n_per_group=10, seed=20260617)
    out = []
    run_pipeline(df, raters_df, ta_df, out)
    report = "\n".join(out)
    assert "MANOVA" in report
    assert "SIMULATED" in report or "simulated" in report.lower()
