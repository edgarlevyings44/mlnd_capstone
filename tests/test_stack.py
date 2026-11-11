from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from mlnd import stack
from mlnd.config import load_config
from mlnd.data import load_raw
from mlnd.features import split_data
from mlnd.tune import load_best_params

CFG = load_config()


def test_stack_module_never_references_the_test_split():
    src = Path(stack.__file__).read_text(encoding="utf-8")
    assert "X_test" not in src and "y_test" not in src


def test_point_metrics_perfect_and_worst():
    y = np.array([0, 0, 1, 1])
    good = stack.point_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert good["pr_auc"] == pytest.approx(1.0)
    assert good["f1"] == pytest.approx(1.0) and good["recall"] == pytest.approx(1.0)
    assert stack.point_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))["recall"] == 0.0


def test_best_f1_is_one_for_perfect_separation():
    y = np.array([0, 1, 1, 0])
    assert stack.best_f1(y, np.array([0.1, 0.9, 0.8, 0.2])) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "lo, hi, mean, word",
    [(0.001, 0.01, 0.005, "WINS"), (-0.01, -0.001, -0.005, "LOSES"), (-0.004, 0.006, 0.001, "TIES")],
)
def test_verdict_states_win_lose_or_tie(lo, hi, mean, word):
    assert word in stack.verdict(lo, hi, mean, "xgboost")


def test_paired_bootstrap_identical_models_and_random_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 400), 0, 1)
    noise = rng.uniform(0, 1, 400)
    boots = stack.paired_bootstrap(y, {"a": good, "b": good.copy(), "c": noise}, n_boot=100, seed=1)
    assert np.allclose(boots["a"]["pr_auc"], boots["b"]["pr_auc"])
    diff = boots["a"]["pr_auc"] - boots["c"]["pr_auc"]
    assert np.percentile(diff, 2.5) > 0


def test_stack_has_the_configured_bases_and_a_logistic_meta_learner():
    st = CFG["stacking"]
    models = stack.build_models(CFG, load_best_params(), 1.8, st["bases"], st["singles"], inner_splits=5)
    assert [n for n, _ in models["stack"].estimators] == st["bases"]
    assert isinstance(models["stack"].final_estimator, LogisticRegression)
    assert set(st["singles"]) <= set(models)


def test_stack_fits_inside_outer_cv_on_a_small_sample():
    X_train, _, y_train, _ = split_data(load_raw(), CFG)
    X, y = X_train.head(400), y_train.head(400)
    models = stack.build_models(
        CFG, load_best_params(), 1.8, ["logistic_regression", "lightgbm"], ["logistic_regression"], inner_splits=3
    )
    outer = StratifiedKFold(3, shuffle=True, random_state=0)
    probs, per_fold = stack.out_of_fold(models, X, y, outer)
    assert set(probs) == {"logistic_regression", "stack"}
    assert all(len(p) == 400 and ((p >= 0) & (p <= 1)).all() for p in probs.values())
    assert len(per_fold) == 2 * 3