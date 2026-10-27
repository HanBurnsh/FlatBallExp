from pathlib import Path

import pytest

from flatball.config import AnomalyConfig, Config, ConfigError, config_with_experiments
from flatball.data.conditions import conditions_for_sign_and_amplitude, conditions_for_transfer, make_condition

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick.json"


def test_load_quick_config():
    config = Config.load(CONFIG)
    assert config.geometry.n_points == 1024
    assert config.anomaly.radius_mode == "relative"
    assert config.experiment.seeds == (0,)


def test_fingerprint_changes_with_protocol():
    config = Config.load(CONFIG)
    fingerprint = config.dataset_fingerprint()
    assert fingerprint == config.dataset_fingerprint()
    assert fingerprint != config.replace(geometry=config.geometry.__class__(n_points=2048)).dataset_fingerprint()


def test_unknown_key_and_bad_fraction_raise():
    with pytest.raises(ConfigError):
        Config.from_dict({"geometry": {"unknown": 1}})
    with pytest.raises(ConfigError):
        Config.from_dict({"anomaly": {"target_fraction": [0.5, 0.4]}})


def test_transfer_conditions_are_self_similar():
    config = Config.load(CONFIG)
    conditions = conditions_for_transfer(config.experiment, config.anomaly)
    assert len(conditions) == 4
    assert len({condition.id for condition in conditions}) == 4
    for condition in conditions:
        assert condition.r / condition.radius == pytest.approx(0.1)
        assert condition.mu0 > 0


def test_sign_amplitude_conditions():
    config = Config.load(CONFIG)
    conditions = conditions_for_sign_and_amplitude(config.experiment, config.anomaly)
    assert len(conditions) == 16
    assert {(condition.sign, condition.amplitude_mode) for condition in conditions} == {
        (1, "absolute"),
        (1, "relative"),
        (-1, "absolute"),
        (-1, "relative"),
    }


def test_fixed_radius_mode_respects_locality():
    config = Config.load(CONFIG)
    anomaly = AnomalyConfig(radius_mode="fixed")
    with pytest.raises(ValueError):
        make_condition(
            index=0,
            radius=5.0,
            mu0_abs=0.1,
            sign=1,
            amplitude_mode="absolute",
            anomaly=anomaly,
            experiment=config.experiment,
        )


def test_config_with_experiments():
    config = Config.load(CONFIG)
    changed = config_with_experiments(config, ("exp1", "exp3"))
    assert changed.experiment.experiments == ("exp1", "exp3")
