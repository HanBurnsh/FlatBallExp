"""Aggregation, hypothesis flags, JSON/Markdown summaries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..config import Config
from .records import MetricRecord
from .stats import negative_spearman_test, noninferiority_p, spearman, t_interval, tost_p


# --------------------------------------------------------------------- utils
def _finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if np.isfinite(v)]


def _mean(values: Iterable[float]) -> float:
    values = _finite(values)
    return float(np.mean(values)) if values else float("nan")


def _records(records: Sequence[MetricRecord], experiment: str, model: str | None = None) -> list[MetricRecord]:
    selected = [r for r in records if r.experiment == experiment]
    if model is not None:
        selected = [r for r in selected if r.model == model]
    return selected


def _paired_diffs(
    records: Sequence[MetricRecord],
    radius: float,
    model: str,
) -> tuple[list[float], list[MetricRecord], list[MetricRecord]]:
    plane = {r.seed: r for r in records if r.model == model and r.radius == radius and r.domain == "plane"}
    sphere = {r.seed: r for r in records if r.model == model and r.radius == radius and r.domain == "sphere"}
    seeds = sorted(set(plane) & set(sphere))
    diffs = [sphere[s].auc - plane[s].auc for s in seeds]
    return diffs, [plane[s] for s in seeds], [sphere[s] for s in seeds]


def _radius_summary(records: Sequence[MetricRecord], radii: Sequence[float], cfg: Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    margin = cfg.evaluation.equivalence_margin
    for radius in radii:
        diffs, planes, spheres = _paired_diffs(records, radius, "gbdt")
        plane_auc = _mean(r.auc for r in planes)
        sphere_auc = _mean(r.auc for r in spheres)
        estimate = t_interval(diffs)
        drop_rate = (plane_auc - sphere_auc) / plane_auc if np.isfinite(plane_auc) and plane_auc > 0 else float("nan")
        rows.append(
            {
                "radius": float(radius),
                "curvature": 1.0 / float(radius) ** 2,
                "n_seeds": estimate["n"],
                "plane_auc": plane_auc,
                "sphere_auc": sphere_auc,
                "difference": estimate,
                "drop_rate": float(drop_rate),
                "noninferiority_p": noninferiority_p(diffs, margin),
                "tost_p": tost_p(diffs, margin),
                "h1_pass": bool(estimate["n"] >= 2 and estimate["ci_low"] > -margin),
            }
        )
    return rows


def _sphere_trend(records: Sequence[MetricRecord], radii: Sequence[float], model: str = "gbdt") -> dict[str, Any]:
    per_seed: dict[int, dict[float, float]] = {}
    for record in records:
        if record.model != model or record.domain != "sphere":
            continue
        per_seed.setdefault(record.seed, {})[record.radius] = record.auc
    rows: list[dict[str, Any]] = []
    for seed, mapping in sorted(per_seed.items()):
        available = [r for r in radii if r in mapping]
        if len(available) < 3:
            continue
        rho, p = spearman([1.0 / r**2 for r in available], [mapping[r] for r in available])
        rows.append({"seed": seed, "rho": rho, "p": p, "n_radii": len(available)})
    summary = negative_spearman_test([row["rho"] for row in rows])
    return {"per_seed": rows, **summary}


# ------------------------------------------------------------------ exp 1
def summarize_exp1(records: Sequence[MetricRecord], cfg: Config) -> dict[str, Any]:
    selected = _records(records, "exp1", "gbdt")
    radii = cfg.experiment.radii
    per_radius = _radius_summary(selected, radii, cfg)
    trend = _sphere_trend(selected, radii)
    margin = cfg.evaluation.equivalence_margin
    alpha = cfg.evaluation.alpha

    all_pass = bool(per_radius) and all(row["h1_pass"] for row in per_radius)
    h2_supported = bool(
        np.isfinite(trend["mean"]) and trend["mean"] < 0 and np.isfinite(trend["p"]) and trend["p"] < alpha
    )
    return {
        "per_radius": per_radius,
        "curvature_trend": trend,
        "verdicts": {
            "H1": {
                "supported": all_pass,
                "criterion": "for every R, paired 95% CI lower bound of (sphere - plane) AUROC > -margin",
                "margin": margin,
            },
            "H2": {
                "supported": h2_supported,
                "criterion": "mean per-seed Spearman(AUC, 1/R^2) < 0 with one-sided p < alpha",
                "alpha": alpha,
            },
        },
    }


# ------------------------------------------------------------------ exp 2
def summarize_exp2(records: Sequence[MetricRecord], cfg: Config) -> dict[str, Any]:
    selected = _records(records, "exp2", "gbdt")
    radii = cfg.experiment.radii
    tolerance = cfg.evaluation.flat_tolerance
    alpha = cfg.evaluation.alpha
    cells: list[dict[str, Any]] = []

    for sign in (1, -1):
        for mode in ("absolute", "relative"):
            cell_records = [r for r in selected if r.sign == sign and r.amplitude_mode == mode]
            rows = _radius_summary(cell_records, radii, cfg)
            trend = _sphere_trend(cell_records, radii)
            ordered = sorted(rows, key=lambda row: row["radius"])
            auc_by_radius = [row["sphere_auc"] for row in ordered if np.isfinite(row["sphere_auc"])]
            flat = bool(auc_by_radius) and (max(auc_by_radius) - min(auc_by_radius)) <= tolerance
            increasing_with_r = bool(len(auc_by_radius) >= 2) and all(
                np.diff(auc_by_radius) >= -1e-12
            )
            decreasing_with_curvature = bool(
                np.isfinite(trend["mean"]) and trend["mean"] < 0 and np.isfinite(trend["p"]) and trend["p"] < alpha
            )
            cells.append(
                {
                    "sign": sign,
                    "amplitude_mode": mode,
                    "per_radius": rows,
                    "curvature_trend": trend,
                    "range": float(max(auc_by_radius) - min(auc_by_radius)) if auc_by_radius else float("nan"),
                    "flat": flat,
                    "monotone_increasing_with_R": increasing_with_r,
                    "decreasing_with_curvature": decreasing_with_curvature,
                }
            )

    def find(sign: int, mode: str) -> dict[str, Any] | None:
        for cell in cells:
            if cell["sign"] == sign and cell["amplitude_mode"] == mode:
                return cell
        return None

    positive_absolute = find(1, "absolute")
    positive_relative = find(1, "relative")
    return {
        "cells": cells,
        "verdicts": {
            "H3a_criterion_met": bool(positive_absolute and positive_absolute["flat"]),
            "H3b_criterion_met": bool(
                positive_relative
                and not positive_relative["flat"]
                and positive_relative["decreasing_with_curvature"]
            ),
            "flat_tolerance": tolerance,
            "note": "Criterion flags are descriptive; see README for the mapping to H3a/H3b.",
        },
    }


# ------------------------------------------------------------------ exp 3
def summarize_exp3(records: Sequence[MetricRecord], cfg: Config) -> dict[str, Any]:
    selected = _records(records, "exp3")
    radii = cfg.experiment.radii
    per_radius: list[dict[str, Any]] = []
    per_seed_drop_diff: dict[int, list[float]] = {}
    per_seed_rate_diff: dict[int, list[float]] = {}

    for radius in radii:
        plane: dict[str, dict[int, MetricRecord]] = {}
        sphere: dict[str, dict[int, MetricRecord]] = {}
        for model in ("gbdt", "pointnet"):
            plane[model] = {
                r.seed: r
                for r in selected
                if r.model == model and r.radius == radius and r.domain == "plane"
            }
            sphere[model] = {
                r.seed: r
                for r in selected
                if r.model == model and r.radius == radius and r.domain == "sphere"
            }
        seeds = sorted(
            set(plane["gbdt"])
            & set(sphere["gbdt"])
            & set(plane["pointnet"])
            & set(sphere["pointnet"])
        )
        drops: dict[str, list[float]] = {"gbdt": [], "pointnet": []}
        rates: dict[str, list[float]] = {"gbdt": [], "pointnet": []}
        for seed in seeds:
            for model in ("gbdt", "pointnet"):
                base = plane[model][seed].auc
                drop = base - sphere[model][seed].auc
                drops[model].append(drop)
                rates[model].append(drop / base if base > 0 else float("nan"))
        drop_a, drop_b = drops["gbdt"], drops["pointnet"]
        rate_a, rate_b = rates["gbdt"], rates["pointnet"]
        drop_diff = [b - a for a, b in zip(drop_a, drop_b)]
        rate_diff = [b - a for a, b in zip(rate_a, rate_b)]
        for seed, value in zip(seeds, drop_diff):
            per_seed_drop_diff.setdefault(seed, []).append(value)
        for seed, value in zip(seeds, rate_diff):
            per_seed_rate_diff.setdefault(seed, []).append(value)
        per_radius.append(
            {
                "radius": float(radius),
                "n_seeds": len(seeds),
                "drop_gbdt": t_interval(drop_a),
                "drop_pointnet": t_interval(drop_b),
                "drop_difference": t_interval(drop_diff),
                "rate_gbdt": t_interval(rate_a),
                "rate_pointnet": t_interval(rate_b),
                "rate_difference": t_interval(rate_diff),
            }
        )

    # Pool across radii at the seed level: averaging radii within a seed keeps
    # the paired-seed unit of analysis instead of treating radii as independent
    # replicates.
    pooled = [float(np.mean(values)) for _, values in sorted(per_seed_drop_diff.items()) if values]
    pooled_rates = [float(np.mean(values)) for _, values in sorted(per_seed_rate_diff.items()) if values]
    difference = t_interval(pooled)
    alpha = cfg.evaluation.alpha
    if difference["n"] < 2:
        supported = None
        interpretation = "fewer than two paired seeds; no CI-based verdict"
    elif difference["ci_high"] < 0:
        supported = False
        interpretation = "pointnet drop significantly smaller than gbdt (original H4 direction)"
    else:
        supported = True
        interpretation = "pointnet drop is not significantly smaller than gbdt (revised H4 direction)"
    return {
        "per_radius": per_radius,
        "pooled_drop_difference": difference,
        "pooled_rate_difference": t_interval(pooled_rates),
        "verdicts": {
            "H4_revised_supported": supported,
            "criterion": "B-A absolute AUROC drop difference CI upper bound >= 0 (deep model not significantly better)",
            "alpha": alpha,
            "interpretation": interpretation,
        },
    }


# ---------------------------------------------------------------- top level
def summarize(records: Sequence[MetricRecord], cfg: Config) -> dict[str, Any]:
    by_experiment: dict[str, Sequence[MetricRecord]] = {}
    for record in records:
        by_experiment.setdefault(record.experiment, [])
        by_experiment[record.experiment].append(record)
    result: dict[str, Any] = {}
    for name in cfg.experiment.experiments:
        subset = by_experiment.get(name, [])
        if name == "exp1":
            result[name] = summarize_exp1(subset, cfg)
        elif name == "exp2":
            result[name] = summarize_exp2(subset, cfg)
        elif name == "exp3":
            result[name] = summarize_exp3(subset, cfg)
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_summary(summary: dict[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")


# ------------------------------------------------------------------ markdown
def _f(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _ci(estimate: dict[str, Any]) -> str:
    return f"[{_f(estimate.get('ci_low'))}, {_f(estimate.get('ci_high'))}]"


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = ["# FlatBall experiment summary", ""]
    run = summary.get("run", {})
    if run:
        lines.append(f"- run: `{run.get('run_id', 'n/a')}`")
        lines.append(f"- seeds: {run.get('seeds', [])}")
        lines.append(f"- radii: {run.get('radii', [])}")
        lines.append("")

    experiments = summary.get("experiments", {})

    exp1 = experiments.get("exp1")
    if exp1:
        lines.extend(["## Experiment 1 - flat to sphere transfer (H1/H2)", ""])
        lines.append("| R | curvature 1/R^2 | plane AUROC | sphere AUROC | sphere - plane [95% CI] | drop rate | NI p | TOST p |")
        lines.append("|---:|---:|---:|---:|---|---:|---:|---:|")
        for row in exp1.get("per_radius", []):
            lines.append(
                f"| {_f(row['radius'], 0)} | {_f(row['curvature'], 5)} | {_f(row['plane_auc'])} | "
                f"{_f(row['sphere_auc'])} | {_f(row['difference']['mean'])} {_ci(row['difference'])} | "
                f"{_f(row['drop_rate'])} | {_f(row['noninferiority_p'])} | {_f(row['tost_p'])} |"
            )
        trend = exp1.get("curvature_trend", {})
        lines.extend(
            [
                "",
                f"Curvature trend: mean Spearman rho = {_f(trend.get('mean'))}, "
                f"one-sided p = {_f(trend.get('p'))} (n={trend.get('n', 0)} seeds).",
                "",
                "Verdict flags: "
                f"H1={exp1['verdicts']['H1']['supported']}, "
                f"H2={exp1['verdicts']['H2']['supported']}.",
                "",
            ]
        )

    exp2 = experiments.get("exp2")
    if exp2:
        lines.extend(["## Experiment 2 - sign x amplitude (H3)", ""])
        lines.append("| sign | amplitude | R | plane AUROC | sphere AUROC | delta [95% CI] | flat cell | rho(curvature) |")
        lines.append("|:---:|:---:|---:|---:|---:|---|---:|")
        for cell in exp2.get("cells", []):
            for row in cell.get("per_radius", []):
                lines.append(
                    f"| {'+' if cell['sign'] > 0 else '-'} | {cell['amplitude_mode']} | {_f(row['radius'], 0)} | "
                    f"{_f(row['plane_auc'])} | {_f(row['sphere_auc'])} | "
                    f"{_f(row['difference']['mean'])} {_ci(row['difference'])} | "
                    f"{'yes' if cell['flat'] else 'no'} | {_f(cell['curvature_trend'].get('mean'))} |"
                )
        lines.extend(
            [
                "",
                f"H3a criterion met: {exp2['verdicts']['H3a_criterion_met']}; "
                f"H3b criterion met: {exp2['verdicts']['H3b_criterion_met']}.",
                "",
            ]
        )

    exp3 = experiments.get("exp3")
    if exp3:
        lines.extend(["## Experiment 3 - handcrafted vs PointNet (H4)", ""])
        lines.append("| R | gbdt drop | pointnet drop | drop difference B-A [95% CI] |")
        lines.append("|---:|---|---|---|")
        for row in exp3.get("per_radius", []):
            lines.append(
                f"| {_f(row['radius'], 0)} | {_f(row['drop_gbdt']['mean'])} | "
                f"{_f(row['drop_pointnet']['mean'])} | "
                f"{_f(row['drop_difference']['mean'])} {_ci(row['drop_difference'])} |"
            )
        pooled = exp3.get("pooled_drop_difference", {})
        lines.extend(
            [
                "",
                f"Pooled drop difference B-A: {_f(pooled.get('mean'))} {_ci(pooled)}.",
                f"Revised H4 supported: {exp3['verdicts']['H4_revised_supported']}.",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"
