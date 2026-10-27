"""Metrics, paired statistics, records and reporting."""

from .evaluate import evaluate_model
from .metrics import auroc, choose_threshold, error_rates
from .records import CSV_FIELDS, MetricRecord, read_csv, write_csv
from .reporting import summarize, write_summary
from .stats import negative_spearman_test, noninferiority_p, spearman, t_interval, tost_p

__all__ = [
    "CSV_FIELDS",
    "MetricRecord",
    "auroc",
    "choose_threshold",
    "error_rates",
    "evaluate_model",
    "negative_spearman_test",
    "noninferiority_p",
    "read_csv",
    "spearman",
    "summarize",
    "t_interval",
    "tost_p",
    "write_csv",
    "write_summary",
]
