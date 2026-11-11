import inspect
from pathlib import Path

import pytest

from mlnd import tune
from mlnd.config import load_config
from mlnd.data import load_raw
from mlnd.features import split_data
from mlnd.train import build_pipeline

CFG = load_config()
MODELS = list(tune.SEARCH_SPACES)


def test_tune_module_never_references_the_test_split():
    src = Path(tune.__file__).read_text(encoding="utf-8")
    assert "X_test" not in src and "y_test" not in src


@pytest.mark.parametrize("model", MODELS)
def test_search_space_keys_exist_on_the_pipeline(model):
    pipe = build_pipeline(model, CFG["tuning"]["strategy"], CFG, 1.8)
    clf = pipe.named_steps["clf"]
    assert all(k.startswith("clf__") for k in tune.SEARCH_SPACES[model])
    valid = set(clf.get_params()) | set(inspect.signature(type(clf).__init__).parameters)
    keys = {k.removeprefix("clf__") for k in tune.SEARCH_SPACES[model]}
    assert keys <= valid


def test_every_baseline_model_has_a_space_and_a_budget():
    for model in CFG["baselines"]["models"]:
        assert model in tune.SEARCH_SPACES
        assert model in CFG["tuning"]["n_iter"]


def test_same_seed_gives_same_candidates_and_best_params():
    X_train, _, y_train, _ = split_data(load_raw(), CFG)
    X, y = X_train.head(500), y_train.head(500)
    s1, r1 = tune.tune_model("lightgbm", X, y, CFG, n_iter=3, n_splits=3)
    s2, r2 = tune.tune_model("lightgbm", X, y, CFG, n_iter=3, n_splits=3)
    assert s1.cv_results_["params"] == s2.cv_results_["params"]
    assert r1["best_params"] == r2["best_params"]
    assert r1["pr_auc_mean"] == pytest.approx(r2["pr_auc_mean"], abs=1e-3)


def test_tuned_pipeline_applies_the_parameters():
    pipe = tune.tuned_pipeline("lightgbm", {"num_leaves": 15, "learning_rate": 0.05}, CFG, 1.8)
    clf = pipe.named_steps["clf"]
    assert clf.num_leaves == 15 and clf.learning_rate == 0.05