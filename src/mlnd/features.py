"""Leak-free splitting and feature engineering.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mlnd.config import load_config
from mlnd.data import load_raw, validate

SEVERITY = "Severity_Score"
COUNTY_MEAN = "Mean_Severity_By_County"
ENGINEERED = [
    SEVERITY,
    "Chlorotic_Necrotic_Index",
    "Lesion_Severity_Ratio",
    "Necrotic_Texture_Interaction",
    "Boundary_Transition_Diff",
]


def split_data(df: pd.DataFrame, cfg: dict | None = None):
    """Stratified train/test split on the label. Always the first step."""
    cfg = cfg or load_config()
    X = df.drop(columns=[cfg["label"]])
    y = df[cfg["label"]]
    return train_test_split(
        X,
        y,
        test_size=cfg["split"]["test_size"],
        stratify=y,
        random_state=cfg["seed"],
    )


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds engineered features. Only the county mean severity is learned (in fit)."""

    def __init__(
        self,
        county_col: str = "County",
        severity_weights: dict | None = None,
        ratio_eps: float = 0.05,
    ):
        self.county_col = county_col
        self.severity_weights = severity_weights
        self.ratio_eps = ratio_eps

    def _severity(self, X: pd.DataFrame) -> pd.Series:
        return sum(w * X[col] for col, w in self.severity_weights.items())

    def fit(self, X: pd.DataFrame, y=None):
        severity = self._severity(X)
        self.county_severity_ = severity.groupby(X[self.county_col]).mean().to_dict()
        self.global_severity_ = float(severity.mean())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out[SEVERITY] = self._severity(X)
        out["Chlorotic_Necrotic_Index"] = X["Chlorosis_Intensity"] * X["Necrotic_Area"]
        out["Lesion_Severity_Ratio"] = (X["Elongated_Streaks"] + X["Irregular_Lesions"]) / (
            X["Green_Yellow_Ratio"] + self.ratio_eps
        )
        out["Necrotic_Texture_Interaction"] = X["Necrotic_Area"] * X["Texture_Roughness"]
        out["Boundary_Transition_Diff"] = X["Boundary_Sharpness"] - X["Transition_Smoothness"]
        out[COUNTY_MEAN] = (
            X[self.county_col].map(self.county_severity_).fillna(self.global_severity_)
        )
        return out


def build_feature_pipeline(cfg: dict | None = None) -> Pipeline:
    """Raw frame (features + County) -> engineered + one-hot county, as a DataFrame."""
    cfg = cfg or load_config()
    county = cfg["county"]
    encode = ColumnTransformer(
        [
            (
                "county",
                OneHotEncoder(
                    categories=[cfg["counties"]], sparse_output=False, handle_unknown="error"
                ),
                [county],
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return Pipeline(
        [
            (
                "engineer",
                FeatureBuilder(
                    county_col=county,
                    severity_weights=cfg["engineering"]["severity_weights"],
                    ratio_eps=cfg["engineering"]["ratio_eps"],
                ),
            ),
            ("encode", encode),
        ]
    )


def main() -> int:
    cfg = load_config()
    df = load_raw()
    validate(df, cfg)
    X_train, X_test, y_train, y_test = split_data(df, cfg)
    print(f"train {X_train.shape}  test {X_test.shape}")
    print(f"infected share  train {y_train.mean():.4f}  test {y_test.mean():.4f}")

    pipe = build_feature_pipeline(cfg).fit(X_train)
    Z_train, Z_test = pipe.transform(X_train), pipe.transform(X_test)
    print(f"model input: {Z_train.shape[1]} columns")
    print(list(Z_train.columns))
    print("county severity learned from TRAIN only:")
    for k, v in pipe.named_steps["engineer"].county_severity_.items():
        print(f"  {k:12s} {v:.4f}")
    print(f"NaNs  train {int(Z_train.isna().sum().sum())}  test {int(Z_test.isna().sum().sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())