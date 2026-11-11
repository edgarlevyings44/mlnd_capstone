import json
from pathlib import Path

import numpy as np
import pytest

from mlnd import threshold
from mlnd.config import load_config

CFG = load_config()
Y = np.array([0, 0, 1, 1])
P = np.array([0.1, 0.4, 0.6, 0.9])


def test_threshold_module_never_references_the_test_split():
    src = Path(threshold.__file__).read_text(encoding="utf-8")
    assert "X_test" not in src and "y_test" not in src


def test_threshold_table_known_values():
    t = threshold.threshold_table(Y, P)
    at_half = t[t["threshold"] == 0.5].iloc[0]
    assert at_half["recall"] == 1.0 and at_half["precision"] == 1.0 and at_half["f1"] == pytest.approx(1.0)
    at_low = t[t["threshold"] == 0.05].iloc[0]
    assert at_low["recall"] == 1.0 and at_low["precision"] == pytest.approx(0.5)


def test_recall_target_picks_the_highest_threshold_that_meets_it():
    t = threshold.threshold_table(Y, P)
    assert threshold.choose_threshold(t, "recall_target", 1.0) == pytest.approx(0.6)
    assert threshold.choose_threshold(t, "recall_target", 0.5) == pytest.approx(0.9)


def test_f1_policy_lands_in_the_perfect_separation_band():
    t = threshold.threshold_table(Y, P)
    assert 0.4 < threshold.choose_threshold(t, "f1") <= 0.6


def test_unreachable_recall_target_raises():
    t = threshold.threshold_table(Y, P)
    with pytest.raises(ValueError):
        threshold.choose_threshold(t, "recall_target", 1.1)


def test_bootstrap_is_reproducible_with_the_same_seed():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 500), 0, 1)
    a = threshold.bootstrap_threshold(y, p, "f1", None, 0.5, n_boot=40, seed=3)
    b = threshold.bootstrap_threshold(y, p, "f1", None, 0.5, n_boot=40, seed=3)
    assert a == b


def test_run_writes_the_artifact_and_meets_the_recall_target(tmp_path):
    result = threshold.run(CFG, tmp_path / "artifacts", tmp_path / "reports", n_boot=30)
    saved = json.loads((tmp_path / "artifacts" / "threshold.json").read_text(encoding="utf-8"))
    assert saved["threshold"] == result["threshold"]
    assert 0.05 <= saved["threshold"] <= 0.95
    assert saved["oof_estimates"]["recall"] >= CFG["threshold"]["recall_target"]
    assert threshold.load_threshold(tmp_path / "artifacts" / "threshold.json") == result["threshold"]
    assert (tmp_path / "reports" / "threshold.md").exists()