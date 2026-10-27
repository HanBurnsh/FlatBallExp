"""Gradient-boosted trees on the six handcrafted local features.

This is the primary model (model A).  It is deliberately a fixed measurement
instrument: no hyper-parameter search is performed.  XGBoost is used when
installed and requested; scikit-learn's histogram GBDT is the dependency-light
fallback with the same role.
"""

from __future__ import annotations

import importlib.util
import logging

import numpy as np

from ..config import ModelConfig
from ..data.dataset import DatasetSplit
from .base import PointClassifier

logger = logging.getLogger("flatball.models")


class GBDTClassifier(PointClassifier):
    name = "gbdt"

    def __init__(self, cfg: ModelConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = int(seed)
        self.backend = self._resolve_backend(cfg.gbdt_backend)
        self._model = None

    @staticmethod
    def _resolve_backend(requested: str) -> str:
        if requested == "sklearn":
            return "sklearn"
        has_xgb = importlib.util.find_spec("xgboost") is not None
        if requested == "xgboost":
            if not has_xgb:
                raise RuntimeError("xgboost backend requested but xgboost is not installed")
            return "xgboost"
        return "xgboost" if has_xgb else "sklearn"

    def fit(self, train: DatasetSplit, val: DatasetSplit) -> None:
        x_train = np.asarray(train.features, dtype=np.float32).reshape(-1, train.n_features)
        y_train = np.asarray(train.labels, dtype=np.uint8).reshape(-1)
        x_val = np.asarray(val.features, dtype=np.float32).reshape(-1, val.n_features)
        y_val = np.asarray(val.labels, dtype=np.uint8).reshape(-1)

        counts = np.bincount(y_train, minlength=2).astype(np.float64)
        if counts[0] == 0 or counts[1] == 0:
            raise ValueError("training split must contain both classes")
        weights = np.where(y_train == 1, len(y_train) / (2.0 * counts[1]), len(y_train) / (2.0 * counts[0]))

        if self.backend == "xgboost":
            from xgboost import XGBClassifier

            self._model = XGBClassifier(
                n_estimators=self.cfg.gbdt_max_iter,
                max_depth=6,
                learning_rate=self.cfg.gbdt_learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                eval_metric="auc",
                early_stopping_rounds=20,
                n_jobs=-1,
                random_state=self.seed,
                verbosity=0,
            )
            self._model.fit(
                x_train,
                y_train,
                sample_weight=weights,
                eval_set=[(x_val, y_val)],
                verbose=False,
            )
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            self._model = HistGradientBoostingClassifier(
                max_iter=self.cfg.gbdt_max_iter,
                learning_rate=self.cfg.gbdt_learning_rate,
                max_leaf_nodes=self.cfg.gbdt_max_leaf_nodes,
                early_stopping=False,
                random_state=self.seed,
            )
            self._model.fit(x_train, y_train, sample_weight=weights)

        logger.info(
            "fit gbdt backend=%s on %d points (%d positive)",
            self.backend,
            len(y_train),
            int(counts[1]),
        )

    def predict_scores(self, split: DatasetSplit) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model has not been fitted")
        x = np.asarray(split.features, dtype=np.float32).reshape(-1, split.n_features)
        scores = self._model.predict_proba(x)[:, 1]
        return np.asarray(scores, dtype=np.float64).reshape(split.labels.shape)
