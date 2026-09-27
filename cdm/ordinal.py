"""Ordinal-appropriate estimation for five-point rubric items.

Maximum likelihood on five-category indicators treats an ordinal scale as
continuous and normally distributed. It is known to bias fit indices and
factor loadings downward, which matters here because the paper's headline
result is a fit index at small N. Some of the misfit reported there is
ordinality rather than identification, and separating the two is a small
contribution in its own right.

This module supplies the ordinal side: polychoric correlations estimated by
the conventional two-step method, ordinal alpha and omega computed from them,
and a lavaan script for the WLSMV re-estimation that Python's SEM packages do
not implement well.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats


def _thresholds(x: np.ndarray, n_cat: int = 5) -> np.ndarray:
    """Normal-ogive thresholds from the observed marginal distribution."""
    counts = np.array([(x == k).sum() for k in range(1, n_cat + 1)], float)
    props = np.cumsum(counts / counts.sum())[:-1]
    props = np.clip(props, 1e-6, 1 - 1e-6)
    return stats.norm.ppf(props)


def _bivariate_prob(rho: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cell probabilities of a bivariate normal under given thresholds."""
    a = np.concatenate([[-np.inf], a, [np.inf]])
    b = np.concatenate([[-np.inf], b, [np.inf]])
    cdf = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf
    grid = np.array([[cdf([ai, bj]) for bj in b] for ai in a])
    p = (grid[1:, 1:] - grid[:-1, 1:] - grid[1:, :-1] + grid[:-1, :-1])
    return np.clip(p, 1e-12, 1.0)


def polychoric(x: np.ndarray, y: np.ndarray, n_cat: int = 5) -> float:
    """Polychoric correlation between two ordinal variables.

    Two-step estimator: thresholds are fixed from each variable's marginal
    distribution, then the latent correlation is chosen to maximise the
    likelihood of the observed contingency table. Falls back to the Pearson
    correlation when a variable has no variance.
    """
    x = np.asarray(x, int)
    y = np.asarray(y, int)
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    a, b = _thresholds(x, n_cat), _thresholds(y, n_cat)
    obs = np.zeros((n_cat, n_cat))
    for i, j in zip(x - 1, y - 1):
        obs[i, j] += 1

    def neg_ll(r):
        r = float(np.clip(r, -0.999, 0.999))
        return -float((obs * np.log(_bivariate_prob(r, a, b))).sum())

    start = float(np.clip(np.corrcoef(x, y)[0, 1], -0.9, 0.9))
    res = optimize.minimize_scalar(neg_ll, bounds=(-0.999, 0.999), method="bounded",
                                   options={"xatol": 1e-4})
    return float(res.x) if res.success else start


def polychoric_matrix(df: pd.DataFrame, n_cat: int = 5) -> pd.DataFrame:
    """Polychoric correlation matrix over ordinal item columns.

    Cost is quadratic in the number of items and each cell runs a numerical
    optimisation, so twenty-five items take a few seconds rather than being
    instant.
    """
    cols = list(df.columns)
    k = len(cols)
    R = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            r = polychoric(df[cols[i]].values, df[cols[j]].values, n_cat)
            R[i, j] = R[j, i] = 0.0 if np.isnan(r) else r
    return pd.DataFrame(R, index=cols, columns=cols)


def ordinal_alpha(df: pd.DataFrame, n_cat: int = 5) -> float:
    """Cronbach's alpha computed on the polychoric matrix.

    Ordinal alpha estimates the reliability the scale would have if the
    underlying continuous variables were observed, and is the appropriate
    companion to a polychoric factor model. It is systematically higher than
    Cronbach's alpha on the same items; report both rather than substituting.
    """
    R = polychoric_matrix(df, n_cat).values
    k = R.shape[0]
    if k < 2:
        return np.nan
    off = (R.sum() - np.trace(R)) / (k * (k - 1))
    return float(k * off / (1 + (k - 1) * off))


def ordinal_omega(df: pd.DataFrame, n_cat: int = 5) -> float:
    """McDonald's omega from a one-factor solution on the polychoric matrix.

    Omega does not assume the equal loadings that alpha requires, so where the
    two diverge sharply the items are not tau-equivalent and alpha is the less
    trustworthy of the pair.
    """
    R = polychoric_matrix(df, n_cat).values
    vals, vecs = np.linalg.eigh(R)
    lam = vecs[:, -1] * np.sqrt(max(vals[-1], 0.0))
    if lam.sum() < 0:
        lam = -lam
    num = lam.sum() ** 2
    err = float(np.sum(np.clip(1.0 - lam**2, 0.0, None)))
    return float(num / (num + err)) if (num + err) > 0 else np.nan


def reliability_table(df: pd.DataFrame, dimensions: list[str],
                      items_per_dim: int = 5) -> pd.DataFrame:
    """Cronbach's alpha beside ordinal alpha and omega, per dimension.

    The gap between the first column and the other two is the ordinality
    correction. Reporting all three lets a reader see how much of the
    reliability estimate depends on treating a five-point scale as continuous.
    """
    from cdm.pipeline import cronbach_alpha
    rows = []
    for dim in dimensions:
        cols = [f"{dim}_i{i+1}" for i in range(items_per_dim)]
        sub = df[cols]
        rows.append(dict(dimension=dim,
                         cronbach_alpha=round(float(cronbach_alpha(sub.values)), 3),
                         ordinal_alpha=round(float(ordinal_alpha(sub)), 3),
                         ordinal_omega=round(float(ordinal_omega(sub)), 3)))
    return pd.DataFrame(rows)


LAVAAN_TEMPLATE = """# CDM confirmatory factor analysis, WLSMV on ordinal indicators.
# Generated by cdm.ordinal.write_lavaan_script. Run in R:
#   install.packages("lavaan"); source("cfa_wlsmv.R")
#
# WLSMV is the estimator the measurement literature recommends for five-point
# indicators. Python's SEM packages do not implement it reliably, which is why
# this step leaves Python.

library(lavaan)

data <- read.csv("{csv}")
items <- c({items})
data[items] <- lapply(data[items], ordered)

model <- '
{model}
'

fit <- cfa(model, data = data, ordered = items, estimator = "WLSMV")
summary(fit, fit.measures = TRUE, standardized = TRUE)

cat("\\n--- scaled fit indices (report these, not the naive ones) ---\\n")
print(fitMeasures(fit, c("chisq.scaled", "df.scaled", "pvalue.scaled",
                         "cfi.scaled", "tli.scaled", "rmsea.scaled",
                         "rmsea.ci.lower.scaled", "rmsea.ci.upper.scaled", "srmr")))

# One-factor alternative, for the comparison the dimensional claim depends on.
model1 <- 'G =~ {all_items}'
fit1 <- cfa(model1, data = data, ordered = items, estimator = "WLSMV")
cat("\\n--- five-factor vs one-factor ---\\n")
print(anova(fit1, fit))
"""


def write_lavaan_script(df: pd.DataFrame, dimensions: list[str], outdir: str | Path,
                        items_per_dim: int = 5) -> tuple[Path, Path]:
    """Write the item data and an R script that re-estimates the model with WLSMV."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cols = [f"{d}_i{i+1}" for d in dimensions for i in range(items_per_dim)]
    csv_path = outdir / "cfa_items.csv"
    df[cols].to_csv(csv_path, index=False)

    model = "\n".join(
        f"{d} =~ " + " + ".join(f"{d}_i{i+1}" for i in range(items_per_dim))
        for d in dimensions)
    script = LAVAAN_TEMPLATE.format(
        csv=csv_path.name,
        items=", ".join(f'"{c}"' for c in cols),
        model=model,
        all_items=" + ".join(cols))
    r_path = outdir / "cfa_wlsmv.R"
    r_path.write_text(script, encoding="utf-8")
    return csv_path, r_path
