"""Run a fitted model on the test domains and emit metric records."""

from __future__ import annotations

from ..config import Config
from ..data.conditions import Condition
from ..data.dataset import DatasetBundle
from ..models.base import PointClassifier
from .metrics import auroc, choose_threshold, error_rates
from .records import MetricRecord


def evaluate_model(
    experiment: str,
    model: PointClassifier,
    condition: Condition,
    bundle: DatasetBundle,
    seed: int,
    cfg: Config,
) -> list[MetricRecord]:
    val_scores = model.predict_scores(bundle.val)
    threshold = choose_threshold(
        bundle.val.labels,
        val_scores,
        method=cfg.evaluation.threshold,
        fixed=cfg.evaluation.fixed_threshold,
    )

    records: list[MetricRecord] = []
    for domain, split in (("plane", bundle.plane_test), ("sphere", bundle.sphere_test)):
        scores = model.predict_scores(split)
        auc = auroc(split.labels, scores)
        fnr, fpr = error_rates(split.labels, scores, threshold)
        records.append(
            MetricRecord(
                experiment=experiment,
                seed=int(seed),
                condition_id=condition.id,
                model=model.name,
                radius=float(condition.radius),
                r=float(condition.r),
                mu0=float(condition.mu0),
                sign=int(condition.sign),
                amplitude_mode=condition.amplitude_mode,
                radius_mode=condition.radius_mode,
                domain=domain,
                auc=float(auc),
                fnr=float(fnr),
                fpr=float(fpr),
                threshold=float(threshold),
                anomaly_fraction=float(split.anomaly_fraction),
                mean_regions=float(split.mean_regions),
                n_points=int(split.n_points),
            )
        )
    return records
