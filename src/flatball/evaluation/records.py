"""Long-format metric records (one row per condition / model / domain / seed)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CSV_FIELDS = (
    "experiment",
    "seed",
    "condition_id",
    "model",
    "radius",
    "r",
    "mu0",
    "sign",
    "amplitude_mode",
    "radius_mode",
    "domain",
    "auc",
    "fnr",
    "fpr",
    "threshold",
    "anomaly_fraction",
    "mean_regions",
    "n_points",
)


@dataclass(frozen=True)
class MetricRecord:
    experiment: str
    seed: int
    condition_id: str
    model: str
    radius: float
    r: float
    mu0: float
    sign: int
    amplitude_mode: str
    radius_mode: str
    domain: str
    auc: float
    fnr: float
    fpr: float
    threshold: float
    anomaly_fraction: float
    mean_regions: float
    n_points: int

    def as_row(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in CSV_FIELDS}


def write_csv(records: Iterable[MetricRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_row())
    return path


def read_csv(path: str | Path) -> list[MetricRecord]:
    path = Path(path)
    records: list[MetricRecord] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                MetricRecord(
                    experiment=str(row["experiment"]),
                    seed=int(row["seed"]),
                    condition_id=str(row["condition_id"]),
                    model=str(row["model"]),
                    radius=float(row["radius"]),
                    r=float(row["r"]),
                    mu0=float(row["mu0"]),
                    sign=int(row["sign"]),
                    amplitude_mode=str(row["amplitude_mode"]),
                    radius_mode=str(row["radius_mode"]),
                    domain=str(row["domain"]),
                    auc=float(row["auc"]),
                    fnr=float(row["fnr"]),
                    fpr=float(row["fpr"]),
                    threshold=float(row["threshold"]),
                    anomaly_fraction=float(row["anomaly_fraction"]),
                    mean_regions=float(row["mean_regions"]),
                    n_points=int(row["n_points"]),
                )
            )
    return records
