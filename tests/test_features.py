import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlnd.config import load_config
from mlnd.data import load_raw
from mlnd.features import COUNTY_MEAN, FeatureBuilder, build_feature_pipeline, split_data

CFG = load_config()


@pytest.fixture(scope="module")
def split():
    return split_data(load_raw(), CFG)


def _builder():
    return FeatureBuilder(
        county_col=CFG["county"],
        severity_weights=CFG["engineering"]["severity_weights"],
        ratio_eps=CFG["engineering"]["ratio_eps"],
    )


def test_split_sizes_stratified_and_disjoint(split):
    X_train, X_test, y_train, y_test = split
    assert len(X_train) == 6313 and len(X_test) == 1579
    assert abs(y_train.mean() - y_test.mean()) < 0.005
    assert not set(X_train.index) & set(X_test.index)
    assert CFG["label"] not in X_train.columns


def test_split_is_deterministic():
    a = split_data(load_raw(), CFG)[1].index
    b = split_data(load_raw(), CFG)[1].index
    assert a.equals(b)


def test_engineered_formulas():
    row = pd.DataFrame(
        {
            "County": ["Narok"],
            "Chlorosis_Intensity": [0.5],
            "Necrotic_Area": [0.25],
            "Infected_Distribution": [1.0],
            "Elongated_Streaks": [0.3],
            "Irregular_Lesions": [0.1],
            "Green_Yellow_Ratio": [0.35],
            "Texture_Roughness": [0.8],
            "Boundary_Sharpness": [0.9],
            "Transition_Smoothness": [0.4],
        }
    )
    out = _builder().fit(row).transform(row).iloc[0]
    assert out["Severity_Score"] == pytest.approx(0.4 * 0.5 + 0.4 * 0.25 + 0.2 * 1.0)
    assert out["Chlorotic_Necrotic_Index"] == pytest.approx(0.125)
    assert out["Lesion_Severity_Ratio"] == pytest.approx(0.4 / (0.35 + CFG["engineering"]["ratio_eps"]))
    assert out["Necrotic_Texture_Interaction"] == pytest.approx(0.2)
    assert out["Boundary_Transition_Diff"] == pytest.approx(0.5)


def test_county_mean_comes_from_training_rows_only(split):
    X_train, X_test, _, _ = split
    fb = _builder().fit(X_train)
    expected = (
        sum(w * X_train[c] for c, w in CFG["engineering"]["severity_weights"].items())
        .groupby(X_train["County"])
        .mean()
    )
    for county, value in fb.county_severity_.items():
        assert value == pytest.approx(expected[county])

    full = pd.concat([X_train, X_test])
    fb_full = _builder().fit(full)
    assert any(
        abs(fb.county_severity_[c] - fb_full.county_severity_[c]) > 1e-9 for c in fb.county_severity_
    )


def test_transforming_test_rows_does_not_change_fitted_state(split):
    X_train, X_test, _, _ = split
    pipe = build_feature_pipeline(CFG).fit(X_train)
    builder = pipe.named_steps["engineer"]
    before = (dict(builder.county_severity_), builder.global_severity_)
    pipe.transform(X_test)
    pipe.transform(X_test.assign(Necrotic_Area=1.0))  # wildly different test data
    assert (builder.county_severity_, builder.global_severity_) == before


def test_transform_is_rowwise_no_batch_statistics(split):
    X_train, X_test, _, _ = split
    pipe = build_feature_pipeline(CFG).fit(X_train)
    sample = X_test.head(40)
    whole = pipe.transform(sample)
    halves = pd.concat([pipe.transform(sample.iloc[:20]), pipe.transform(sample.iloc[20:])])
    single = pipe.transform(sample.iloc[[7]])
    pd.testing.assert_frame_equal(whole, halves)
    pd.testing.assert_frame_equal(whole.iloc[[7]], single)


def test_unseen_county_falls_back_to_global_mean(split):
    X_train, X_test, _, _ = split
    fb = _builder().fit(X_train)
    row = X_test.head(1).copy()
    row["County"] = "Nairobi"
    assert fb.transform(row)[COUNTY_MEAN].iloc[0] == pytest.approx(fb.global_severity_)


def test_output_columns_and_no_nans(split):
    X_train, X_test, _, _ = split
    pipe = build_feature_pipeline(CFG).fit(X_train)
    Z = pipe.transform(X_test)
    assert Z.shape == (1579, 23)
    assert "County" not in Z.columns and not Z.isna().any().any()
    onehot = Z[[c for c in Z.columns if c.startswith("County_")]]
    assert list(onehot.columns) == [f"County_{c}" for c in CFG["counties"]]
    assert (onehot.sum(axis=1) == 1).all()


def test_pipeline_refits_inside_each_cv_fold(split):
    X_train, _, y_train, _ = split
    X, y = X_train.head(800), y_train.head(800)
    model = make_pipeline(clone(build_feature_pipeline(CFG)), StandardScaler(), LogisticRegression(max_iter=2000))
    scores = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
    assert len(scores) == 5 and (scores > 0.6).all()