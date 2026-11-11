from pathlib import Path

import pytest

from mlnd import train
from mlnd.config import load_config
from mlnd.data import load_raw
from mlnd.features import split_data

CFG = load_config()
MODELS = CFG["baselines"]["models"]
STRATEGIES = CFG["baselines"]["strategies"]


def test_train_module_never_references_the_test_split():
    src = Path(train.__file__).read_text(encoding="utf-8")
    assert "X_test" not in src and "y_test" not in src


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_pipeline_has_expected_steps(model, strategy):
    pipe = train.build_pipeline(model, strategy, CFG, pos_weight=1.8)
    expected = ["engineer", "encode", "scale"]
    if strategy in ("smote", "adasyn"):
        expected.append("resample")
    expected.append("clf")
    assert [name for name, _ in pipe.steps] == expected


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        train.build_pipeline("xgboost", "bogus", CFG, pos_weight=1.8)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_combination_runs_end_to_end_on_a_small_training_sample(strategy):
    X_train, _, y_train, _ = split_data(load_raw(), CFG)
    row = train.evaluate_combination(
        "logistic_regression", strategy, X_train.head(600), y_train.head(600), CFG, n_splits=3
    )
    assert row["n_splits"] == 3
    assert 0 <= row["recall_mean"] <= 1
    assert row["pr_auc_mean"] > 0.3