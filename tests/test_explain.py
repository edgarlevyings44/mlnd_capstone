import numpy as np
import pytest

from mlnd import explain
from mlnd.config import load_config
from mlnd.data import load_raw
from mlnd.features import split_data

CFG = load_config()


@pytest.fixture(scope="module")
def setup():
    X_train, _, _, _ = split_data(load_raw(), CFG)
    return explain.load_explainer(), X_train.head(150)


def test_shap_values_plus_base_reconstruct_the_model_output(setup):
    ex, X = setup
    values, _, _ = ex.shap_values(X)
    Z, _, _ = ex._features(X)
    raw_model = ex.clf.predict(Z, prediction_type="RawFormulaVal")
    assert np.allclose(ex.base_value + values.sum(axis=1), raw_model, atol=1e-3)
    p_pipeline = ex.pipe.predict_proba(X)[:, 1]
    assert np.allclose(explain.sigmoid(ex.base_value + values.sum(axis=1)), p_pipeline, atol=1e-3)


def test_county_columns_are_merged_into_one_driver(setup):
    ex, X = setup
    values, names, _ = ex.shap_values(X)
    agg, groups = explain.aggregate(values, names)
    assert "County" in groups
    assert not any(g.startswith("County_") for g in groups) and "Mean_Severity_By_County" not in groups
    assert np.allclose(agg.sum(axis=1), values.sum(axis=1))
    assert len(groups) == len(names) - 6  # six one-hot columns plus the county mean collapse into one


def test_explain_one_is_sorted_and_additive(setup):
    ex, X = setup
    out = ex.explain_one(X.iloc[[0]], k=5)
    contribs = [d["contribution"] for d in out["drivers"]]
    assert len(contribs) == 5
    assert contribs == sorted(contribs, key=abs, reverse=True)
    total = out["base_log_odds"] + sum(contribs) + out["other_contribution"]
    assert total == pytest.approx(out["raw_log_odds"], abs=1e-9)
    assert out["flagged"] == (out["risk_score"] >= out["threshold"])
    assert out["risk_score"] == pytest.approx(ex.pipe.predict_proba(X.iloc[[0]])[0, 1], abs=1e-3)


def test_directions_match_signs_and_every_driver_has_a_label(setup):
    ex, X = setup
    for i in range(5):
        for d in ex.explain_one(X.iloc[[i]], k=17)["drivers"]:
            assert d["feature"] in explain.FEATURE_LABELS
            assert (d["direction"] == "raises risk") == (d["contribution"] > 0)


def test_county_driver_shows_the_county_name(setup):
    ex, X = setup
    row = X.iloc[[3]]
    drivers = ex.explain_one(row, k=17)["drivers"]
    county = [d for d in drivers if d["feature"] == "County"]
    assert len(county) == 1 and county[0]["value"] == row["County"].iloc[0]


def test_global_importance_is_sorted_and_shares_sum_to_one(setup):
    ex, X = setup
    imp = explain.global_importance(ex, X)
    assert imp["share"].sum() == pytest.approx(1.0)
    assert list(imp["mean_abs_shap"]) == sorted(imp["mean_abs_shap"], reverse=True)