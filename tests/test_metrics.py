import numpy as np

from flatball.evaluation.metrics import auroc, choose_threshold, error_rates
from flatball.evaluation.stats import negative_spearman_test, noninferiority_p, t_interval, tost_p


def test_auroc_and_error_rates():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auroc(labels, scores) == 1.0
    fnr, fpr = error_rates(labels, scores, 0.5)
    assert fnr == 0.0
    assert fpr == 0.0
    threshold = choose_threshold(labels, scores, "youden")
    assert 0.3 <= threshold <= 0.7


def test_paired_interval_and_tost():
    diffs = [-0.01, -0.02, 0.0, -0.01, -0.03]
    estimate = t_interval(diffs)
    assert estimate["n"] == 5
    assert estimate["ci_low"] > -0.05
    assert noninferiority_p(diffs, 0.05) < 0.05
    assert tost_p(diffs, 0.05) < 0.05


def test_negative_spearman_summary():
    summary = negative_spearman_test([-0.9, -0.8, -0.7])
    assert summary["mean"] < 0
    assert summary["p"] < 0.05
