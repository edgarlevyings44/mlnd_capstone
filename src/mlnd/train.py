"""Baseline models x imbalance strategies, compared by cross-validation."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from contextlib import nullcontext

import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from mlnd.config import ROOT, load_config
from mlnd.data import _md_table, load_raw, validate
from mlnd.features import build_feature_pipeline, split_data

STRATEGIES = ("none", "class_weight", "smote", "adasyn")
SCORING = {
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "mcc": make_scorer(matthews_corrcoef),
}


def make_classifier(name: str, strategy: str, seed: int, pos_weight: float):
    weighted = strategy == "class_weight"
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000, class_weight="balanced" if weighted else None, random_state=seed
        )
    if name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced" if weighted else None, n_jobs=-1, random_state=seed
        )
    if name == "xgboost":
        return XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=pos_weight if weighted else 1.0,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            class_weight="balanced" if weighted else None,
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
    if name == "catboost":
        return CatBoostClassifier(
            auto_class_weights="Balanced" if weighted else None,
            random_seed=seed,
            verbose=0,
        )
    raise ValueError(f"Unknown model: {name}")


def make_resampler(strategy: str, seed: int):
    if strategy == "smote":
        return SMOTE(random_state=seed)
    if strategy == "adasyn":
        return ADASYN(random_state=seed)
    return None


def build_pipeline(model: str, strategy: str, cfg: dict, pos_weight: float) -> ImbPipeline:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    seed = cfg["seed"]
    steps = [*build_feature_pipeline(cfg).steps, ("scale", StandardScaler())]
    resampler = make_resampler(strategy, seed)
    if resampler is not None:
        steps.append(("resample", resampler))
    steps.append(("clf", make_classifier(model, strategy, seed, pos_weight)))
    return ImbPipeline(steps)


def evaluate_combination(model, strategy, X, y, cfg, n_splits) -> dict:
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    pipe = build_pipeline(model, strategy, cfg, pos_weight)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=1, error_score="raise")
    row = {
        "model": model,
        "strategy": strategy,
        "n_splits": n_splits,
        "pos_weight": round(pos_weight, 3),
    }
    for key in SCORING:
        scores = res[f"test_{key}"]
        row[f"{key}_mean"] = float(scores.mean())
        row[f"{key}_std"] = float(scores.std(ddof=1))
    row["fit_time_s"] = float(res["fit_time"].mean())
    return row


def run_grid(X, y, cfg, models, strategies, n_splits, on_result=None):
    rows, failures = [], []
    combos = list(itertools.product(models, strategies))
    for i, (model, strategy) in enumerate(combos, start=1):
        print(f"[{i}/{len(combos)}] {model} + {strategy} ...", flush=True)
        try:
            row = evaluate_combination(model, strategy, X, y, cfg, n_splits)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            failures.append((model, strategy, str(e)))
            continue
        print(
            f"    recall {row['recall_mean']:.3f}  f1 {row['f1_mean']:.3f}  "
            f"pr_auc {row['pr_auc_mean']:.3f}  ({row['fit_time_s']:.1f}s per fold)",
            flush=True,
        )
        rows.append(row)
        if on_result:
            on_result(row)
    return pd.DataFrame(rows), failures


def format_results(results: pd.DataFrame) -> pd.DataFrame:
    out = results[["model", "strategy"]].copy()
    for key in SCORING:
        out[key] = [
            f"{m:.3f} +/- {s:.3f}"
            for m, s in zip(results[f"{key}_mean"], results[f"{key}_std"])
        ]
    out["fit_s"] = results["fit_time_s"].round(1)
    return out


def _log_child(row: dict) -> None:
    with mlflow.start_run(run_name=f"{row['model']}+{row['strategy']}", nested=True):
        mlflow.log_params({k: row[k] for k in ("model", "strategy", "n_splits", "pos_weight")})
        mlflow.log_metrics(
            {k: v for k, v in row.items() if k.endswith(("_mean", "_std")) or k == "fit_time_s"}
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Baselines x imbalance strategies (CV only).")
    ap.add_argument("--quick", action="store_true", help="3 folds, fewer models and strategies")
    ap.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    args = ap.parse_args(argv)

    load_dotenv(ROOT / ".env")
    cfg = load_config()
    bl = cfg["baselines"]
    if args.quick:
        models, strategies, n_splits = bl["quick"]["models"], bl["quick"]["strategies"], bl["quick"]["n_splits"]
        tag = "quick"
    else:
        models, strategies, n_splits = bl["models"], bl["strategies"], cfg["cv"]["n_splits"]
        tag = "full"

    df = load_raw()
    validate(df, cfg)
    X_train, _, y_train, _ = split_data(df, cfg)
    print(f"Training rows: {len(X_train)}  |  {len(models)} models x {len(strategies)} strategies, {n_splits}-fold CV")

    use_mlflow = (not args.no_mlflow) and bool(os.getenv("MLFLOW_TRACKING_URI"))
    if use_mlflow:
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        parent = mlflow.start_run(run_name=f"baselines-{tag}")
        mlflow.log_params(
            {"models": ",".join(models), "strategies": ",".join(strategies),
             "n_splits": n_splits, "seed": cfg["seed"], "n_train": len(X_train)}
        )
    else:
        parent = nullcontext()
        print("MLflow logging is OFF (no tracking URI, or --no-mlflow).")

    with parent:
        t0 = time.time()
        results, failures = run_grid(
            X_train, y_train, cfg, models, strategies, n_splits,
            on_result=_log_child if use_mlflow else None,
        )
        results = results.sort_values(["f1_mean", "pr_auc_mean"], ascending=False).reset_index(drop=True)

        out_dir = ROOT / cfg["paths"]["reports_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"baselines_{tag}.csv"
        md_path = out_dir / f"baselines_{tag}.md"
        best_path = out_dir / f"baseline_best_strategy_{tag}.json"

        results.to_csv(csv_path, index=False)
        md_path.write_text(
            f"# Baselines x imbalance strategies ({tag}, {n_splits}-fold CV, training split only)\n\n"
            "Mean +/- SD across folds, infected class positive. recall/precision/F1/MCC use the 0.5 threshold.\n"
            "Sorted by F1 then PR-AUC.\n\n" + _md_table(format_results(results)) + "\n",
            encoding="utf-8",
        )
        best = results.groupby("model", sort=False).head(1)
        best_path.write_text(
            json.dumps(
                {
                    "best_strategy_by_model": dict(zip(best["model"], best["strategy"])),
                    "overall_best": {"model": results.loc[0, "model"], "strategy": results.loc[0, "strategy"]},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if use_mlflow:
            for p in (csv_path, md_path, best_path):
                mlflow.log_artifact(str(p))

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Wrote:\n  {csv_path}\n  {md_path}\n  {best_path}")
    print(results[["model", "strategy", "recall_mean", "f1_mean", "pr_auc_mean"]].round(3).to_string(index=False))
    if failures:
        print(f"\n{len(failures)} combination(s) FAILED:")
        for m, s, e in failures:
            print(f"  {m} + {s}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())