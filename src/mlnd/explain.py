"""
Contributions are in log-odds and add up exactly: base value + sum(contributions) = raw model score;
the risk score is the sigmoid of that. The one-hot county columns and the county-derived mean severity
are merged into a single "County" driver.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from mlnd.config import ROOT, load_config
from mlnd.data import _md_table, load_raw, validate
from mlnd.features import COUNTY_MEAN, split_data
from mlnd.threshold import load_threshold

FEATURE_LABELS = {
    "Chlorosis_Intensity": "Leaf yellowing (chlorosis intensity)",
    "Green_Yellow_Ratio": "Green-to-yellow balance",
    "Necrotic_Area": "Dead tissue (necrotic area)",
    "Texture_Roughness": "Leaf surface roughness",
    "Patchiness": "Patchy colour pattern",
    "Lesion_Distribution": "Spread of lesions",
    "Elongated_Streaks": "Streak-shaped marks",
    "Irregular_Lesions": "Irregularly shaped lesions",
    "Infected_Distribution": "Spread of the infected area",
    "Boundary_Sharpness": "Sharpness of lesion edges",
    "Transition_Smoothness": "Smoothness of the healthy-to-infected change",
    "Severity_Score": "Overall severity score",
    "Chlorotic_Necrotic_Index": "Yellowing x dead-tissue index",
    "Lesion_Severity_Ratio": "Streaks and irregular lesions relative to green balance",
    "Necrotic_Texture_Interaction": "Dead tissue x roughness",
    "Boundary_Transition_Diff": "Edge sharpness minus transition smoothness",
    "County": "County",
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def group_name(feature: str) -> str:
    return "County" if feature.startswith("County_") or feature == COUNTY_MEAN else feature


def aggregate(values: np.ndarray, names: list[str]):
    groups: list[str] = []
    for n in names:
        g = group_name(n)
        if g not in groups:
            groups.append(g)
    out = np.zeros((values.shape[0], len(groups)))
    for j, n in enumerate(names):
        out[:, groups.index(group_name(n))] += values[:, j]
    return out, groups


class MlndExplainer:
    def __init__(self, pipe, threshold: float):
        self.pipe = pipe
        self.clf = pipe.steps[-1][1]
        self.threshold = float(threshold)
        self.explainer = shap.TreeExplainer(self.clf)
        self.base_value = float(np.ravel(self.explainer.expected_value)[0])

    def _features(self, X: pd.DataFrame):
        """Run every step except the classifier. Returns (scaled matrix, names, unscaled frame)."""
        Z, names, shown = X, None, None
        for _, step in self.pipe.steps[:-1]:
            Z = step.transform(Z)
            if isinstance(Z, pd.DataFrame):
                names, shown = list(Z.columns), Z
        return np.asarray(Z, dtype=float), names, shown

    def shap_values(self, X: pd.DataFrame):
        Z, names, shown = self._features(X)
        values = self.explainer.shap_values(Z)
        if isinstance(values, list):
            values = values[-1]
        return np.asarray(values), names, shown

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        values, names, _ = self.shap_values(X)
        agg, groups = aggregate(values, names)
        return pd.DataFrame(agg, columns=groups, index=X.index)

    def explain_one(self, row: pd.DataFrame, k: int = 5) -> dict:
        values, names, shown = self.shap_values(row)
        agg, groups = aggregate(values, names)
        contrib = agg[0]
        raw = self.base_value + float(contrib.sum())
        risk = float(sigmoid(raw))
        order = np.argsort(-np.abs(contrib))
        drivers = []
        for j in order[:k]:
            g = groups[j]
            value = row["County"].iloc[0] if g == "County" else round(float(shown[g].iloc[0]), 3)
            drivers.append(
                {
                    "feature": g,
                    "label": FEATURE_LABELS.get(g, g),
                    "value": value,
                    "contribution": float(contrib[j]),
                    "direction": "raises risk" if contrib[j] > 0 else "lowers risk",
                }
            )
        return {
            "risk_score": risk,
            "threshold": self.threshold,
            "flagged": bool(risk >= self.threshold),
            "base_risk": float(sigmoid(self.base_value)),
            "base_log_odds": self.base_value,
            "raw_log_odds": raw,
            "drivers": drivers,
            "other_contribution": float(contrib[order[k:]].sum()),
        }


def load_explainer(art_dir=None) -> MlndExplainer:
    art = Path(art_dir) if art_dir else ROOT / "artifacts"
    return MlndExplainer(joblib.load(art / "model.joblib"), load_threshold(art / "threshold.json"))


def global_importance(explainer: MlndExplainer, X: pd.DataFrame) -> pd.DataFrame:
    imp = explainer.contributions(X).abs().mean().sort_values(ascending=False)
    return pd.DataFrame(
        {
            "feature": imp.index,
            "label": [FEATURE_LABELS.get(f, f) for f in imp.index],
            "mean_abs_shap": imp.values,
            "share": imp.values / imp.sum(),
        }
    )


def main() -> int:
    cfg = load_config()
    df = load_raw()
    validate(df, cfg)
    X_train, _, _, _ = split_data(df, cfg)
    sample = X_train.sample(n=min(1000, len(X_train)), random_state=cfg["seed"])

    ex = load_explainer()
    imp = global_importance(ex, sample)
    risk = ex.pipe.predict_proba(sample)[:, 1]
    flagged_row = sample[risk >= ex.threshold].iloc[[0]]
    clear_row = sample[risk < ex.threshold].iloc[[0]]
    examples = {"flagged_example": ex.explain_one(flagged_row), "not_flagged_example": ex.explain_one(clear_row)}
    for name, row in (("flagged_example", flagged_row), ("not_flagged_example", clear_row)):
        examples[name]["input"] = row.iloc[0].to_dict()

    out = ROOT / cfg["paths"]["reports_dir"]
    out.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out / "shap_global_importance.csv", index=False)
    (out / "shap_global_importance.md").write_text(
        "# Global SHAP importance (1000 training rows, final model)\n\n"
        "Mean absolute contribution in log-odds. County columns are merged into one driver. Overlapping features "
        "(e.g. the severity score and the symptoms it is built from) share credit, and SHAP describes the model, not "
        "field causes. The risk score is not a calibrated probability.\n\n"
        + _md_table(imp.round(4)) + "\n",
        encoding="utf-8",
    )
    (out / "shap_example_explanations.json").write_text(json.dumps(examples, indent=2, default=str), encoding="utf-8")

    print(imp.head(8).round(3).to_string(index=False))
    for name, e in examples.items():
        print(f"\n{name}: risk {e['risk_score']:.3f} (threshold {e['threshold']:.2f}, flagged={e['flagged']})")
        for d in e["drivers"]:
            print(f"   {d['label']:55s} value {d['value']!s:>8}  {d['contribution']:+.3f}  {d['direction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())