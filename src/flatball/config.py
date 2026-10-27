"""Typed configuration objects for the flat-to-sphere transfer experiment.

The configuration is intentionally explicit: every experimental knob that can
change a metric lives in one of the frozen dataclasses below.  Config files are
plain JSON (TOML is accepted when the interpreter provides ``tomllib``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - depends on interpreter
    tomllib = None  # type: ignore[assignment]


class ConfigError(ValueError):
    """Raised when a configuration file is malformed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _check_keys(data: Mapping[str, Any], where: str, allowed: Sequence[str]) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError(f"unknown key(s) in {where}: {', '.join(unknown)}")


def _as_tuple_of_floats(value: Any, where: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{where} must be a list of numbers")
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where} must contain only numbers") from exc


@dataclass(frozen=True)
class GeometryConfig:
    """Surface sampling and local-neighbourhood geometry."""

    n_points: int = 10240
    fps_pool_factor: float = 1.5
    plane_area_factor: float = 4.0

    def __post_init__(self) -> None:
        _need(self.n_points >= 64, "geometry.n_points must be >= 64")
        _need(self.fps_pool_factor >= 1.0, "geometry.fps_pool_factor must be >= 1")
        _need(self.plane_area_factor > 0, "geometry.plane_area_factor must be > 0")


@dataclass(frozen=True)
class FeatureConfig:
    """Local PCA / quadratic-fit feature extraction."""

    neighborhood_ratio: float = 0.15
    min_neighbors: int = 8
    neighbor_slack: float = 1.8
    pca_ridge: float = 1e-8
    chunk_size: int = 2048

    def __post_init__(self) -> None:
        _need(0 < self.neighborhood_ratio < 0.5, "features.neighborhood_ratio must be in (0, 0.5)")
        _need(self.min_neighbors >= 6, "features.min_neighbors must be >= 6")
        _need(self.neighbor_slack >= 1.0, "features.neighbor_slack must be >= 1")
        _need(self.pca_ridge >= 0, "features.pca_ridge must be >= 0")
        _need(self.chunk_size >= 1, "features.chunk_size must be >= 1")


@dataclass(frozen=True)
class AnomalyConfig:
    """Normal-displacement anomaly protocol."""

    mu0: float = 0.1
    r_ratio: float = 0.1
    radius_mode: str = "relative"  # relative: r = r_ratio * R, fixed: r = r_ratio * R_ref
    sigma_ratio: float = 0.2
    target_fraction: tuple[float, float] = (0.08, 0.12)
    min_regions: int = 1
    max_regions: int = 96
    center_separation_factor: float = 2.0
    placement_attempts: int = 8192
    allow_partial: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_fraction", _as_tuple_of_floats(self.target_fraction, "anomaly.target_fraction")
        )
        _need(len(self.target_fraction) == 2, "anomaly.target_fraction must have two entries")
        lo, hi = self.target_fraction
        _need(self.mu0 != 0, "anomaly.mu0 must be non-zero")
        _need(0 < self.r_ratio <= 0.25, "anomaly.r_ratio must be in (0, 0.25]")
        _need(self.radius_mode in {"relative", "fixed"}, "anomaly.radius_mode must be 'relative' or 'fixed'")
        _need(self.sigma_ratio >= 0, "anomaly.sigma_ratio must be >= 0")
        _need(0 <= lo < hi <= 1, "anomaly.target_fraction must satisfy 0 <= low < high <= 1")
        _need(self.min_regions >= 1, "anomaly.min_regions must be >= 1")
        _need(self.max_regions >= self.min_regions, "anomaly.max_regions must be >= min_regions")
        _need(self.center_separation_factor >= 1, "anomaly.center_separation_factor must be >= 1")
        _need(self.placement_attempts >= 1, "anomaly.placement_attempts must be >= 1")


@dataclass(frozen=True)
class SplitConfig:
    """Sample counts.  Each anomaly sample contains 8--12% abnormal points."""

    train_normal: int = 500
    train_anomaly: int = 500
    val_normal: int = 100
    val_anomaly: int = 100
    test_normal: int = 100
    test_anomaly: int = 100

    def __post_init__(self) -> None:
        for name in dataclasses.fields(self):
            _need(getattr(self, name.name) >= 0, f"splits.{name.name} must be >= 0")
        _need(self.train_normal + self.train_anomaly > 0, "splits: training set is empty")
        _need(self.train_anomaly > 0, "splits.train_anomaly must be > 0")
        _need(self.val_anomaly > 0 and self.val_normal > 0, "splits: validation needs both classes")
        _need(
            self.test_anomaly > 0 and self.test_normal > 0,
            "splits: test sets need both classes for AUROC",
        )

    @property
    def train_total(self) -> int:
        return self.train_normal + self.train_anomaly

    @property
    def val_total(self) -> int:
        return self.val_normal + self.val_anomaly

    @property
    def test_total(self) -> int:
        return self.test_normal + self.test_anomaly


@dataclass(frozen=True)
class ModelConfig:
    """Fixed model hyper-parameters.  These are measurement instruments, not tuning knobs."""

    gbdt_backend: str = "auto"  # auto | sklearn | xgboost
    gbdt_max_iter: int = 200
    gbdt_learning_rate: float = 0.1
    gbdt_max_leaf_nodes: int = 31
    pointnet_epochs: int = 40
    pointnet_batch_samples: int = 8
    pointnet_lr: float = 1e-3
    pointnet_patience: int = 6
    device: str = "cpu"

    def __post_init__(self) -> None:
        _need(self.gbdt_backend in {"auto", "sklearn", "xgboost"}, "model.gbdt_backend is invalid")
        _need(self.gbdt_max_iter >= 1, "model.gbdt_max_iter must be >= 1")
        _need(self.gbdt_learning_rate > 0, "model.gbdt_learning_rate must be > 0")
        _need(self.gbdt_max_leaf_nodes >= 2, "model.gbdt_max_leaf_nodes must be >= 2")
        _need(self.pointnet_epochs >= 1, "model.pointnet_epochs must be >= 1")
        _need(self.pointnet_batch_samples >= 1, "model.pointnet_batch_samples must be >= 1")
        _need(self.pointnet_lr > 0, "model.pointnet_lr must be > 0")
        _need(self.pointnet_patience >= 0, "model.pointnet_patience must be >= 0")


@dataclass(frozen=True)
class EvaluationConfig:
    """Statistical reporting configuration."""

    confidence: float = 0.95
    equivalence_margin: float = 0.05
    flat_tolerance: float = 0.02
    alpha: float = 0.05
    threshold: str = "youden"  # youden | fixed
    fixed_threshold: float = 0.5

    def __post_init__(self) -> None:
        _need(0 < self.confidence < 1, "evaluation.confidence must be in (0, 1)")
        _need(0 < self.equivalence_margin < 1, "evaluation.equivalence_margin must be in (0, 1)")
        _need(self.flat_tolerance >= 0, "evaluation.flat_tolerance must be >= 0")
        _need(0 < self.alpha < 1, "evaluation.alpha must be in (0, 1)")
        _need(self.threshold in {"youden", "fixed"}, "evaluation.threshold must be 'youden' or 'fixed'")
        _need(0 <= self.fixed_threshold <= 1, "evaluation.fixed_threshold must be in [0, 1]")


@dataclass(frozen=True)
class ExperimentSettings:
    """Run-level settings shared by all experiment families."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    radii: tuple[float, ...] = (50.0, 20.0, 10.0, 5.0)
    reference_radius: float = 50.0
    amplitude_power: float = 1.0
    max_locality_ratio: float = 0.2
    experiments: tuple[str, ...] = ("exp1", "exp2", "exp3")
    output_dir: str = "results"
    cache_dir: str = "cache"
    cache: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))
        object.__setattr__(self, "radii", tuple(float(r) for r in self.radii))
        object.__setattr__(self, "experiments", tuple(str(e) for e in self.experiments))
        _need(len(self.seeds) > 0, "experiment.seeds must not be empty")
        _need(all(s >= 0 for s in self.seeds), "experiment.seeds must be non-negative")
        _need(len(set(self.seeds)) == len(self.seeds), "experiment.seeds contains duplicates")
        _need(len(self.radii) >= 2, "experiment.radii needs at least two radii")
        _need(all(r > 0 for r in self.radii), "experiment.radii must be positive")
        _need(len(set(self.radii)) == len(self.radii), "experiment.radii contains duplicates")
        _need(self.reference_radius > 0, "experiment.reference_radius must be > 0")
        _need(self.amplitude_power > 0, "experiment.amplitude_power must be > 0")
        _need(0 < self.max_locality_ratio < 1, "experiment.max_locality_ratio must be in (0, 1)")
        _need(len(self.experiments) > 0, "experiment.experiments must not be empty")
        _need(all(bool(e) for e in self.experiments), "experiment names must be non-empty strings")
        _need(self.log_level in {"DEBUG", "INFO", "WARNING", "ERROR"}, "experiment.log_level is invalid")


@dataclass(frozen=True)
class Config:
    """Top-level immutable configuration."""

    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        data = _mapping(data, "config")
        _check_keys(
            data,
            "config",
            ["geometry", "features", "anomaly", "splits", "model", "evaluation", "experiment"],
        )

        def section(name: str, klass: type) -> Any:
            raw = _mapping(data.get(name, {}), name)
            _check_keys(raw, name, [f.name for f in dataclasses.fields(klass)])
            return klass(**raw)

        return cls(
            geometry=section("geometry", GeometryConfig),
            features=section("features", FeatureConfig),
            anomaly=section("anomaly", AnomalyConfig),
            splits=section("splits", SplitConfig),
            model=section("model", ModelConfig),
            evaluation=section("evaluation", EvaluationConfig),
            experiment=section("experiment", ExperimentSettings),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        elif suffix == ".toml":
            if tomllib is None:
                raise ConfigError("TOML configs need Python >= 3.11 (tomllib)")
            with path.open("rb") as handle:
                data = tomllib.load(handle)
        else:
            raise ConfigError(f"unsupported config extension: {path.suffix}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
        return path

    def dataset_fingerprint(self) -> str:
        """Stable hash of everything that changes generated features/labels."""
        payload = {
            "geometry": dataclasses.asdict(self.geometry),
            "features": dataclasses.asdict(self.features),
            "anomaly": dataclasses.asdict(self.anomaly),
            "splits": dataclasses.asdict(self.splits),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def replace(self, **changes: Any) -> "Config":
        """Return a copy with top-level sections replaced (handy for CLI overrides)."""
        return dataclasses.replace(self, **changes)


def config_with_seeds(config: Config, seeds: Sequence[int]) -> Config:
    experiment = dataclasses.replace(config.experiment, seeds=tuple(int(s) for s in seeds))
    return dataclasses.replace(config, experiment=experiment)


def config_with_cache(config: Config, enabled: bool) -> Config:
    experiment = dataclasses.replace(config.experiment, cache=bool(enabled))
    return dataclasses.replace(config, experiment=experiment)


def config_with_experiments(config: Config, names: Sequence[str]) -> Config:
    experiment = dataclasses.replace(config.experiment, experiments=tuple(names))
    return dataclasses.replace(config, experiment=experiment)



