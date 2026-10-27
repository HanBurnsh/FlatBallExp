from pathlib import Path

from flatball.config import Config, config_with_experiments
from flatball.evaluation.records import MetricRecord
from flatball.evaluation.reporting import render_markdown, summarize

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick.json"


def _record(experiment: str, seed: int, radius: float, domain: str, model: str, auc: float) -> MetricRecord:
    return MetricRecord(
        experiment=experiment,
        seed=seed,
        condition_id=f"R{radius:g}",
        model=model,
        radius=radius,
        r=radius * 0.1,
        mu0=0.1,
        sign=1,
        amplitude_mode="absolute",
        radius_mode="relative",
        domain=domain,
        auc=auc,
        fnr=0.1,
        fpr=0.1,
        threshold=0.5,
        anomaly_fraction=0.1,
        mean_regions=40.0,
        n_points=1024,
    )


def test_summaries_render_for_all_experiments():
    config = config_with_experiments(Config.load(CONFIG), ("exp1", "exp2", "exp3"))
    records = []
    for seed in (0, 1, 2):
        for radius in (50.0, 20.0, 10.0, 5.0):
            records.append(_record("exp1", seed, radius, "plane", "gbdt", 0.90))
            records.append(_record("exp1", seed, radius, "sphere", "gbdt", 0.89))
            for sign, mode in ((1, "absolute"), (1, "relative"), (-1, "absolute"), (-1, "relative")):
                records.append(
                    MetricRecord(
                        **{
                            **_record("exp2", seed, radius, "plane", "gbdt", 0.90).as_row(),
                            "sign": sign,
                            "amplitude_mode": mode,
                        }
                    )
                )
                records.append(
                    MetricRecord(
                        **{
                            **_record("exp2", seed, radius, "sphere", "gbdt", 0.88).as_row(),
                            "sign": sign,
                            "amplitude_mode": mode,
                        }
                    )
                )
            records.append(_record("exp3", seed, radius, "plane", "gbdt", 0.90))
            records.append(_record("exp3", seed, radius, "sphere", "gbdt", 0.89))
            records.append(_record("exp3", seed, radius, "plane", "pointnet", 0.90))
            records.append(_record("exp3", seed, radius, "sphere", "pointnet", 0.86))

    summary = summarize(records, config)
    assert "exp1" in summary
    assert "exp2" in summary
    assert "exp3" in summary
    assert summary["exp1"]["per_radius"]
    assert "H1" in summary["exp1"]["verdicts"]
    text = render_markdown({"run": {"run_id": "test"}, "experiments": summary})
    assert "Experiment 1" in text
    assert "Experiment 3" in text
