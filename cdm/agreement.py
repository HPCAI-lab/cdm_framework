# Ordinal-appropriate inter-rater agreement statistics.
import numpy as np
from itertools import combinations

def weighted_kappa(a, b, n_cat=5):
    """Quadratic-weighted Cohen's kappa for two raters (categories 1..n_cat).
    Near-miss disagreements (4 vs 5) penalized far less than (1 vs 5)."""
    a = np.asarray(a, int) - 1; b = np.asarray(b, int) - 1
    O = np.zeros((n_cat, n_cat))
    for x, y in zip(a, b):
        O[x, y] += 1
    O /= O.sum()
    r = O.sum(1); c = O.sum(0)
    E = np.outer(r, c)
    idx = np.arange(n_cat)
    W = (idx[:, None] - idx[None, :])**2 / (n_cat - 1)**2     # quadratic disagreement weights
    denom = (W * E).sum()
    return 1 - (W * O).sum() / denom if denom > 0 else np.nan

def mean_pairwise_weighted_kappa(wide, n_cat=5):
    cols = list(wide.columns)
    ks = [weighted_kappa(wide[i].values, wide[j].values, n_cat)
          for i, j in combinations(cols, 2)]
    return float(np.mean(ks))

def krippendorff_interval(units_by_raters):
    """Krippendorff's alpha with the interval metric, complete reliability matrix.
    units_by_raters: (n_units x n_raters) array of category values."""
    X = np.asarray(units_by_raters, float)
    vals = np.unique(X)
    # coincidence matrix
    o = {}
    for row in X:
        mu = len(row)
        for p in range(mu):
            for q in range(mu):
                if p != q:
                    o[(row[p], row[q])] = o.get((row[p], row[q]), 0) + 1.0 / (mu - 1)
    nc = {v: sum(o.get((v, k), 0) for k in vals) for v in vals}
    n = sum(nc.values())
    d2 = lambda c, k: (c - k)**2
    Do = sum(o.get((c, k), 0) * d2(c, k) for c in vals for k in vals)
    De = sum(nc[c] * nc[k] * d2(c, k) for c in vals for k in vals) / (n - 1)
    return 1 - Do / De if De > 0 else np.nan
