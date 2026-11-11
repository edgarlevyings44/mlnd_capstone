import numpy as np
import pandas as pd
import pytest

from mlnd.config import load_config
from mlnd.data import DataValidationError, audit, clipping_report, load_raw, validate

CFG = load_config()


@pytest.fixture
def good_df():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        rng.uniform(0.05, 0.95, size=(n, len(CFG["features"]))), columns=CFG["features"]
    )
    df.insert(0, CFG["county"], [CFG["counties"][i % 6] for i in range(n)])
    df.insert(0, CFG["label"], [i % 2 for i in range(n)])
    return df


def test_valid_data_passes(good_df):
    validate(good_df, CFG)


def test_real_data_passes():
    validate(load_raw(), CFG)


def test_missing_column_fails(good_df):
    with pytest.raises(DataValidationError, match="missing columns"):
        validate(good_df.drop(columns=["Necrotic_Area"]), CFG)


def test_out_of_range_fails(good_df):
    good_df.loc[3, "Patchiness"] = 1.5
    with pytest.raises(DataValidationError, match="outside"):
        validate(good_df, CFG)


def test_bad_label_fails(good_df):
    good_df.loc[0, CFG["label"]] = 2
    with pytest.raises(DataValidationError, match="not in"):
        validate(good_df, CFG)


def test_unknown_county_fails(good_df):
    good_df.loc[0, CFG["county"]] = "Nairobi"
    with pytest.raises(DataValidationError, match="unknown county"):
        validate(good_df, CFG)


def test_missing_values_fail(good_df):
    good_df.loc[5, "Texture_Roughness"] = np.nan
    with pytest.raises(DataValidationError, match="missing values"):
        validate(good_df, CFG)


def test_clipping_detected():
    rng = np.random.default_rng(1)
    x = np.clip(rng.normal(0.6, 0.3, 5000), 0, 1)
    df = pd.DataFrame({c: x for c in CFG["features"]})
    assert clipping_report(df, CFG)["clipped"].all()


def test_uniform_not_flagged():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({c: rng.uniform(0, 1, 20000) for c in CFG["features"]})
    assert not clipping_report(df, CFG)["clipped"].any()


def test_audit_on_real_data():
    r = audit(load_raw(), CFG)
    assert r["n_rows"] == 7892 and r["class_counts"] == {0: 5045, 1: 2847}