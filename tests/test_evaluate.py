import numpy as np
import pytest

from mlnd import evaluate
from mlnd.config import load_config
from mlnd.data import load_raw
from mlnd.features import split_data

CFG = load_config()


def test_metrics_for_perfect_predictions():
    m = evaluate.metrics_at([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert all(m[k] == pytest.approx(1.0) for k in ("precision", "recall", "f1", "mcc", "gmean", "roc_auc", "pr_auc"))


def test_metrics_for_inverted_predictions():
    m = evaluate.metrics_at([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1], 0.5)
    assert m["recall"] == 0.0 and m["mcc"] == pytest.approx(-1.0)


def test_bootstrap_ci_is_reproducible_and_ordered():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 400), 0, 1)
    a = evaluate.bootstrap_ci(y, p, 0.5, n_boot=100, seed=2)
    b = evaluate.bootstrap_ci(y, p, 0.5, n_boot=100, seed=2)
    assert a == b
    assert a["pr_auc"][0] < a["pr_auc"][1]


def test_calibration_of_a_well_calibrated_model_is_small():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    table, ece = evaluate.calibration_table(y, p)
    assert ece < 0.05
    assert table["n"].sum() == 5000


def test_lock_refuses_a_second_scoring_of_the_test_split(tmp_path):
    lock = tmp_path / "final_evaluation.json"
    assert evaluate._locked(lock, force=False) is False
    lock.write_text("{}", encoding="utf-8")
    assert evaluate._locked(lock, force=False) is True
    assert evaluate._locked(lock, force=True) is False


def test_per_county_returns_one_row_per_county():
    rng = np.random.default_rng(3)
    counties = np.array(["A"] * 200 + ["B"] * 200)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 400), 0, 1)
    out = evaluate.per_county(counties, y, p, 0.5, n_boot=50, seed=0)
    assert list(out["county"]) == ["A", "B"]
    assert list(out["n"]) == [200, 200]


def test_leave_one_county_out_split_really_excludes_the_county():
    X_train, _, y_train, _ = split_data(load_raw(), CFG)
    for county in CFG["counties"]:
        X_tr, y_tr, X_ho, y_ho = evaluate.holdout_split(X_train, y_train, county, CFG["county"])
        assert county not in set(X_tr["County"])
        assert set(X_ho["County"]) == {county}
        assert len(X_tr) + len(X_ho) == len(X_train) and len(y_tr) == len(X_tr) and len(y_ho) == len(X_ho)