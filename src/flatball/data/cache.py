"""On-disk cache for extracted point features.

Feature extraction is the expensive part of the experiment.  Cached splits are
keyed by a hash of geometry/feature/anomaly/split settings plus the condition
and random seed, so changing any protocol knob invalidates the cache.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .dataset import DatasetSplit

_SAFE = re.compile(r"[^A-Za-z0-9_.+-]+")


def safe_name(value: str) -> str:
    return _SAFE.sub("_", value).strip("_") or "unnamed"


class FeatureCache:
    def __init__(self, root: str | Path, fingerprint: str) -> None:
        self.base = Path(root) / fingerprint[:16]

    def path(self, condition_id: str, seed: int, split_name: str) -> Path:
        return self.base / safe_name(condition_id) / f"seed_{seed:04d}" / f"{split_name}.npz"

    def load(self, condition_id: str, seed: int, split_name: str) -> DatasetSplit | None:
        path = self.path(condition_id, seed, split_name)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                features = np.asarray(data["features"], dtype=np.float32).copy()
                labels = np.asarray(data["labels"], dtype=np.uint8).copy()
                meta = tuple(json.loads(str(data["meta"].item())))
            return DatasetSplit(name=split_name, features=features, labels=labels, meta=meta)
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def save(self, condition_id: str, seed: int, split: DatasetSplit) -> Path:
        path = self.path(condition_id, seed, split.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps(list(split.meta), ensure_ascii=False)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                features=split.features,
                labels=split.labels,
                meta=np.asarray(meta),
            )
        temporary.replace(path)
        return path
