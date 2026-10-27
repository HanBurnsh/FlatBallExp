"""Lightweight PointNet-style control model (model B).

The network consumes the *same six local features* as model A, never raw
coordinates.  Two shared per-point layers, a global max-pool and a per-point
classification head keep the comparison focused on the encoder, not on input
information.  PyTorch is optional; importing this module is safe without it.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import ModelConfig
from ..data.dataset import DatasetSplit
from .base import ModelUnavailableError, PointClassifier

logger = logging.getLogger("flatball.models")

try:  # optional dependency
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if torch is not None:

    class _PointNet(nn.Module):
        def __init__(self, in_dim: int, width: int = 64) -> None:
            super().__init__()
            self.embed = nn.Sequential(
                nn.Linear(in_dim, width),
                nn.ReLU(),
                nn.Linear(width, width * 2),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(width * 4, width * 2),
                nn.ReLU(),
                nn.Linear(width * 2, 1),
            )

        def forward(self, x):  # (B, N, F) -> (B, N)
            h = self.embed(x)
            pooled = torch.max(h, dim=1).values
            pooled = pooled[:, None, :].expand(-1, x.shape[1], -1)
            return self.head(torch.cat([h, pooled], dim=-1)).squeeze(-1)


class PointNetClassifier(PointClassifier):
    name = "pointnet"

    def __init__(self, cfg: ModelConfig, seed: int) -> None:
        if torch is None:  # pragma: no cover
            raise ModelUnavailableError(
                "PointNet requires PyTorch. Install it with: pip install -e '.[deep]'"
            )
        self.cfg = cfg
        self.seed = int(seed)
        self.device = self._resolve_device(cfg.device)
        self._model = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @staticmethod
    def _resolve_device(requested: str):
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; using CPU")
            return torch.device("cpu")
        return torch.device(requested)

    def _normalise(self, features: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._std is not None
        return ((np.asarray(features, dtype=np.float32) - self._mean) / self._std).astype(np.float32)

    def fit(self, train: DatasetSplit, val: DatasetSplit) -> None:
        torch.manual_seed(self.seed % (2**32))
        np.random.seed(self.seed % (2**32))
        flat = np.asarray(train.features, dtype=np.float32).reshape(-1, train.n_features)
        self._mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-8] = 1.0
        self._std = std
        del flat

        in_dim = train.n_features
        self._model = _PointNet(in_dim=in_dim).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.cfg.pointnet_lr)

        labels = np.asarray(train.labels, dtype=np.uint8)
        positives = float(labels.sum())
        negatives = float(labels.size - positives)
        if positives == 0:
            raise ValueError("training split has no positive points")
        pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=self.device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        rng = np.random.default_rng(self.seed)
        best_auc = -np.inf
        best_state: dict | None = None
        stale = 0
        batch = max(1, self.cfg.pointnet_batch_samples)
        total = train.n_samples

        for epoch in range(1, self.cfg.pointnet_epochs + 1):
            self._model.train()
            order = rng.permutation(total)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, total, batch):
                index = order[start : start + batch]
                x = self._normalise(train.features[index])
                y = np.asarray(train.labels[index], dtype=np.float32)
                xb = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
                yb = torch.from_numpy(np.ascontiguousarray(y)).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                batches += 1

            val_scores = self.predict_scores(val)
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(val.labels.reshape(-1), val_scores.reshape(-1)))
            logger.info(
                "pointnet epoch %d/%d loss=%.4f val_auroc=%.4f",
                epoch,
                self.cfg.pointnet_epochs,
                epoch_loss / max(batches, 1),
                auc,
            )
            if auc > best_auc + 1e-6:
                best_auc = auc
                best_state = {key: value.detach().cpu().clone() for key, value in self._model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.cfg.pointnet_patience:
                    logger.info("pointnet early stopping at epoch %d (best val AUROC %.4f)", epoch, best_auc)
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

    def predict_scores(self, split: DatasetSplit) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model has not been fitted")
        self._model.eval()
        scores = np.empty(split.labels.shape, dtype=np.float64)
        batch = max(1, self.cfg.pointnet_batch_samples)
        with torch.no_grad():
            for start in range(0, split.n_samples, batch):
                stop = min(start + batch, split.n_samples)
                x = self._normalise(split.features[start:stop])
                xb = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
                probabilities = torch.sigmoid(self._model(xb))
                scores[start:stop] = probabilities.detach().cpu().numpy()
        return scores
