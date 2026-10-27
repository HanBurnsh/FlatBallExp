"""Build train/validation/test splits for one condition and seed."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..geometry.anomalies import inject_anomalies
from ..geometry.surfaces import PlaneSurface, SphereSurface, Surface
from .cache import FeatureCache
from .conditions import Condition
from .dataset import DatasetBundle, DatasetSplit
from .features import FEATURE_NAMES, FeatureExtractor

logger = logging.getLogger("flatball.data")

_SPLIT_CODES = {"train": 1, "val": 2, "plane_test": 3, "sphere_test": 4}


@dataclass(frozen=True)
class _SplitSpec:
    name: str
    kind: str
    n_normal: int
    n_anomaly: int

    @property
    def total(self) -> int:
        return self.n_normal + self.n_anomaly


class DatasetBuilder:
    """Deterministic dataset construction with optional feature caching."""

    def __init__(
        self,
        config: Config,
        condition: Condition,
        seed: int,
        cache: FeatureCache | None = None,
    ) -> None:
        self.config = config
        self.condition = condition
        self.seed = int(seed)
        self.cache = cache
        self.extractor = FeatureExtractor(config.features)
        self._surfaces: dict[str, Surface] = {}

    # ------------------------------------------------------------------ RNG
    def _rng(self, tag: int) -> np.random.Generator:
        entropy = [
            self.seed & 0xFFFFFFFF,
            self.condition.rng_key,
            int(tag) & 0xFFFFFFFF,
        ]
        return np.random.default_rng(np.random.SeedSequence(entropy))

    # -------------------------------------------------------------- surfaces
    def _surface(self, kind: str, split_name: str) -> Surface:
        key = f"{kind}:{split_name}"
        if key in self._surfaces:
            return self._surfaces[key]
        code = _SPLIT_CODES[split_name]
        if kind == "plane":
            surface: Surface = PlaneSurface(
                scale=self.condition.radius,
                n_points=self.config.geometry.n_points,
                area_factor=self.config.geometry.plane_area_factor,
                rng=self._rng(1000 + code),
            )
        elif kind == "sphere":
            surface = SphereSurface(
                radius=self.condition.radius,
                n_points=self.config.geometry.n_points,
                pool_factor=self.config.geometry.fps_pool_factor,
                rng=self._rng(2000 + code),
            )
        else:  # pragma: no cover - internal invariant
            raise ValueError(f"unknown surface kind: {kind}")
        self._surfaces[key] = surface
        return surface

    # ----------------------------------------------------------------- build
    def _specs(self) -> list[_SplitSpec]:
        splits = self.config.splits
        return [
            _SplitSpec("train", "plane", splits.train_normal, splits.train_anomaly),
            _SplitSpec("val", "plane", splits.val_normal, splits.val_anomaly),
            _SplitSpec("plane_test", "plane", splits.test_normal, splits.test_anomaly),
            _SplitSpec("sphere_test", "sphere", splits.test_normal, splits.test_anomaly),
        ]

    def build(self) -> DatasetBundle:
        built: dict[str, DatasetSplit] = {}
        for spec in self._specs():
            cached = None
            if self.cache is not None:
                cached = self.cache.load(self.condition.id, self.seed, spec.name)
            if cached is not None:
                logger.debug("cache hit %s / seed %d / %s", self.condition.id, self.seed, spec.name)
                built[spec.name] = cached
                continue

            logger.info(
                "building %s / seed %d / %s (%d samples x %d points)",
                self.condition.id,
                self.seed,
                spec.name,
                spec.total,
                self.config.geometry.n_points,
            )
            surface = self._surface(spec.kind, spec.name)
            split = self._build_split(spec, surface)
            if self.cache is not None:
                path = self.cache.save(self.condition.id, self.seed, split)
                logger.debug("cached %s", path)
            built[spec.name] = split

        return DatasetBundle(
            condition_id=self.condition.id,
            train=built["train"],
            val=built["val"],
            plane_test=built["plane_test"],
            sphere_test=built["sphere_test"],
        )

    def _build_split(self, spec: _SplitSpec, surface: Surface) -> DatasetSplit:
        cfg = self.config
        n = surface.n_points
        features = np.empty((spec.total, n, len(FEATURE_NAMES)), dtype=np.float32)
        labels = np.zeros((spec.total, n), dtype=np.uint8)
        meta: list[dict[str, object]] = []
        geodesic_radius = float(cfg.features.neighborhood_ratio * self.condition.radius)

        normal_features: np.ndarray | None = None
        if spec.n_normal > 0:
            normal_features = self.extractor.extract(surface.points, surface, geodesic_radius)

        flags = np.concatenate(
            [
                np.zeros(spec.n_normal, dtype=np.uint8),
                np.ones(spec.n_anomaly, dtype=np.uint8),
            ]
        )
        self._rng(3000 + _SPLIT_CODES[spec.name]).shuffle(flags)

        for i, flag in enumerate(flags):
            if flag == 0:
                assert normal_features is not None
                features[i] = normal_features
                meta.append(
                    {
                        "anomalous": False,
                        "n_regions": 0,
                        "fraction": 0.0,
                        "target_fraction": 0.0,
                    }
                )
                continue

            anomaly_rng = self._rng(10000 * _SPLIT_CODES[spec.name] + i)
            displaced, point_labels, plan = inject_anomalies(
                surface=surface,
                rng=anomaly_rng,
                mu0=self.condition.mu0,
                r=self.condition.r,
                sigma_ratio=cfg.anomaly.sigma_ratio,
                target_fraction=cfg.anomaly.target_fraction,
                max_regions=cfg.anomaly.max_regions,
                min_regions=cfg.anomaly.min_regions,
                center_separation_factor=cfg.anomaly.center_separation_factor,
                placement_attempts=cfg.anomaly.placement_attempts,
                allow_partial=cfg.anomaly.allow_partial,
            )
            features[i] = self.extractor.extract(displaced, surface, geodesic_radius)
            labels[i] = point_labels
            meta.append(
                {
                    "anomalous": True,
                    "n_regions": int(plan.n_regions),
                    "fraction": float(plan.fraction),
                    "target_fraction": float(plan.target_fraction),
                }
            )
            if (i + 1) % 100 == 0:
                logger.info("  anomaly sample %d / %d", i + 1, spec.total)

        return DatasetSplit(name=spec.name, features=features, labels=labels, meta=tuple(meta))
