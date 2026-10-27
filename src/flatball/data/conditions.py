"""Experimental conditions (one radius / anomaly amplitude cell)."""

from __future__ import annotations

import zlib
from dataclasses import dataclass

from ..config import AnomalyConfig, ExperimentSettings


@dataclass(frozen=True)
class Condition:
    index: int
    radius: float
    r: float
    mu0: float
    sign: int
    amplitude_mode: str
    radius_mode: str

    def __post_init__(self) -> None:
        if self.radius <= 0 or self.r <= 0:
            raise ValueError("radius and r must be positive")
        if self.sign not in {-1, 1}:
            raise ValueError("sign must be -1 or 1")
        if self.mu0 == 0:
            raise ValueError("mu0 must be non-zero")
        if self.amplitude_mode not in {"absolute", "relative"}:
            raise ValueError("amplitude_mode must be 'absolute' or 'relative'")
        if self.radius_mode not in {"relative", "fixed"}:
            raise ValueError("radius_mode must be 'relative' or 'fixed'")

    @property
    def id(self) -> str:
        sign = "p" if self.mu0 > 0 else "m"
        return (
            f"R{self.radius:g}_r{self.r:g}_mu{sign}{abs(self.mu0):g}"
            f"_{self.radius_mode}-{self.amplitude_mode}"
        )

    @property
    def rng_key(self) -> int:
        return int(zlib.crc32(self.id.encode("utf-8")) & 0xFFFFFFFF)

    @property
    def locality_ratio(self) -> float:
        return float(self.r / self.radius)


def make_condition(
    index: int,
    radius: float,
    mu0_abs: float,
    sign: int,
    amplitude_mode: str,
    anomaly: AnomalyConfig,
    experiment: ExperimentSettings,
) -> Condition:
    """Resolve r and mu0 for one condition.

    ``radius_mode`` controls r:
      * relative -> r = r_ratio * R (default, self-similar geometry)
      * fixed    -> r = r_ratio * R_ref

    ``amplitude_mode`` controls mu0:
      * absolute -> mu0 = sign * mu0_abs
      * relative -> mu0 = sign * mu0_abs * (r / r_ref) ** amplitude_power
    """
    if mu0_abs <= 0:
        raise ValueError("mu0_abs must be positive")
    base_r = anomaly.r_ratio * experiment.reference_radius
    r = anomaly.r_ratio * radius if anomaly.radius_mode == "relative" else base_r
    if r / radius > experiment.max_locality_ratio + 1e-12:
        raise ValueError(
            f"locality constraint violated for R={radius:g}: r/R={r / radius:.3f} "
            f"> max_locality_ratio={experiment.max_locality_ratio:.3f}. "
            "Use anomaly.radius_mode='relative' or a larger R."
        )

    mu0 = float(sign) * float(mu0_abs)
    if amplitude_mode == "relative":
        mu0 *= (r / base_r) ** experiment.amplitude_power

    return Condition(
        index=int(index),
        radius=float(radius),
        r=float(r),
        mu0=float(mu0),
        sign=int(sign),
        amplitude_mode=amplitude_mode,
        radius_mode=anomaly.radius_mode,
    )


def conditions_for_transfer(
    experiment: ExperimentSettings,
    anomaly: AnomalyConfig,
    sign: int = 1,
    amplitude_mode: str = "absolute",
) -> list[Condition]:
    """One condition per radius for experiment 1 / 3."""
    return [
        make_condition(
            index=i,
            radius=radius,
            mu0_abs=anomaly.mu0,
            sign=sign,
            amplitude_mode=amplitude_mode,
            anomaly=anomaly,
            experiment=experiment,
        )
        for i, radius in enumerate(experiment.radii)
    ]


def conditions_for_sign_and_amplitude(
    experiment: ExperimentSettings,
    anomaly: AnomalyConfig,
) -> list[Condition]:
    """2 x 2 sign/amplitude conditions, each repeated over the radius gradient."""
    conditions: list[Condition] = []
    for sign in (1, -1):
        for mode in ("absolute", "relative"):
            for radius in experiment.radii:
                conditions.append(
                    make_condition(
                        index=len(conditions),
                        radius=radius,
                        mu0_abs=anomaly.mu0,
                        sign=sign,
                        amplitude_mode=mode,
                        anomaly=anomaly,
                        experiment=experiment,
                    )
                )
    return conditions
