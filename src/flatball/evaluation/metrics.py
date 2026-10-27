"""Point-level detection metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def choose_threshold(labels: np.ndarray, scores: np.ndarray, method: str = "youden", fixed: float = 0.5) -> float:
    """Choose an operating point on the *plane validation* split."""
    if method == "fixed":
        return float(fixed)
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float(fixed)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    index = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[index])
    if not np.isfinite(threshold):
        threshold = float(np.max(scores)) + 1e-12
    return threshold


def error_rates(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    """Return (false-negative rate, false-positive rate) at a fixed threshold."""
    labels = np.asarray(labels).reshape(-1).astype(bool)
    predicted = np.asarray(scores, dtype=np.float64).reshape(-1) >= float(threshold)
    positives = int(np.count_nonzero(labels))
    negatives = int(labels.size - positives)
    false_negative = int(np.count_nonzero(~predicted & labels))
    false_positive = int(np.count_nonzero(predicted & ~labels))
    fnr = false_negative / positives if positives else float("nan")
    fpr = false_positive / negatives if negatives else float("nan")
    return float(fnr), float(fpr)
