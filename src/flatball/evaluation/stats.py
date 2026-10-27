"""Small-sample paired statistics used for the hypothesis verdicts.

The design uses five seeds, so paired t intervals are reported together with
the raw values.  Nothing here assumes independent point-level observations:
uncertainty is always computed across seeds, never across points.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


def t_interval(values: Sequence[float], confidence: float = 0.95) -> dict[str, float]:
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    n = int(array.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    mean = float(array.mean())
    if n == 1:
        sd = 0.0
        se = 0.0
        low = high = mean
    else:
        sd = float(array.std(ddof=1))
        se = sd / math.sqrt(n)
        critical = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
        low = mean - critical * se
        high = mean + critical * se
    return {"n": n, "mean": mean, "sd": sd, "se": se, "ci_low": float(low), "ci_high": float(high)}


def noninferiority_p(values: Sequence[float], margin: float) -> float:
    """One-sided test of H0: mean <= -margin against H1: mean > -margin."""
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    n = int(array.size)
    if n < 2:
        return float("nan")
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(n))
    if se == 0:
        return 0.0 if mean > -margin else 1.0
    statistic = (mean + margin) / se
    return float(1.0 - stats.t.cdf(statistic, df=n - 1))


def tost_p(values: Sequence[float], margin: float) -> float:
    """Two one-sided tests for equivalence within [-margin, +margin]."""
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    n = int(array.size)
    if n < 2:
        return float("nan")
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(n))
    if se == 0:
        return 0.0 if abs(mean) < margin else 1.0
    df = n - 1
    lower = (mean + margin) / se
    upper = (margin - mean) / se
    p_lower = float(1.0 - stats.t.cdf(lower, df=df))
    p_upper = float(stats.t.cdf(upper, df=df))
    return float(max(p_lower, p_upper))


def negative_spearman_test(rhos: Sequence[float]) -> dict[str, float]:
    """Summarise per-seed Spearman rho and test mean rho < 0 across seeds."""
    array = np.asarray([v for v in rhos if np.isfinite(v)], dtype=np.float64)
    n = int(array.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "p": float("nan")}
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if n > 1 else 0.0
    if n < 2:
        p = float("nan")
    elif np.allclose(array, array[0]):
        p = 0.0 if mean < 0 else 1.0
    else:
        p = float(stats.ttest_1samp(array, 0.0, alternative="less").pvalue)
    return {"n": n, "mean": mean, "sd": sd, "p": p}


def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    result = stats.spearmanr(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return float(result.statistic), float(result.pvalue)
