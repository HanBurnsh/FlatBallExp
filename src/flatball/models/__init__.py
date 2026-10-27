"""Model registry."""

from __future__ import annotations

import importlib.util
from typing import Callable

from ..config import Config
from .base import ModelUnavailableError, PointClassifier
from .gbdt import GBDTClassifier
from .pointnet import PointNetClassifier

MODEL_NAMES: tuple[str, ...] = ("gbdt", "pointnet")


def model_available(name: str) -> bool:
    if name == "gbdt":
        return True
    if name == "pointnet":
        return importlib.util.find_spec("torch") is not None
    raise KeyError(f"unknown model: {name}")


_FACTORIES: dict[str, Callable[[Config, int], PointClassifier]] = {
    "gbdt": lambda cfg, seed: GBDTClassifier(cfg.model, seed),
    "pointnet": lambda cfg, seed: PointNetClassifier(cfg.model, seed),
}


def build_model(name: str, config: Config, seed: int) -> PointClassifier:
    if name not in _FACTORIES:
        raise KeyError(f"unknown model: {name}")
    if not model_available(name):
        raise ModelUnavailableError(f"model '{name}' is not available in this environment")
    return _FACTORIES[name](config, seed)


__all__ = [
    "GBDTClassifier",
    "MODEL_NAMES",
    "ModelUnavailableError",
    "PointClassifier",
    "PointNetClassifier",
    "build_model",
    "model_available",
]
