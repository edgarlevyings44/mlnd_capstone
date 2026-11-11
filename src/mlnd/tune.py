"""Hyperparameter tuning by randomised search inside CV (training split only)."""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from mlnd.config import ROOT, load_config
from mlnd.data import _md_table, load_raw, validate
from mlnd.features import split_data
from mlnd.train import SCORING, build_pipeline

SEARCH_SPACES = {
    "logistic_regression": {
        "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
    },
    "random_forest": {
        "clf__n_estimators": randint(100, 601),
        "clf__max_depth": [None, 5, 10, 15, 20],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", "log2", 0.5],
    },
    "xgboost": {
        "clf__n_estimators": randint(100, 601),
        "clf__max_depth": randint(3, 9),
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.6, 0.4),
    },
    "lightgbm": {
        "clf__n_estimators": randint(100, 601),
        "clf__max_depth": [-1, 4, 6, 8, 10],
        "clf__learning_rate": loguniform(0.01, 0.2),
        "clf__num_leaves": [15, 31, 50, 70, 100],
        "clf__subsample": uniform(0.6, 0.4),
        "clf__subsample_freq": [1],
        "clf__colsample_bytree": uniform(0.6, 0.4),
    },
    "catboost": {
        "clf__iterations": randint(200, 801),
        "clf__depth": [4, 6, 8],
        "clf__learning_rate": loguniform(0.01, 0.2),
        "clf__l2_leaf_reg": [1, 3, 5, 7, 9],
    },
}


def _py(v):
    return v.item() if isinstance(v, np.generic) else v


def tune_model(model, X, y, cfg, n_iter, n_splits):
    tn = cfg["tuning"]
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    pipe = build_pipeline(model, tn["strategy"], cfg, pos_weight)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
    search = RandomizedSearchCV(
        pipe,
        SEARCH_SPACES[model],
        n_iter=n_iter,
        scoring=SCORING,
        refit=tn["refit"],
        cv=cv,
        random_state=cfg["seed"],
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X, y)

    res, idx = search.cv_results_, search.best_index_
    row = {
        "model": model,
        "strategy": tn["strategy"],
        "n_iter": n_iter,
        "n_splits": n_splits,
        "refit_metric": tn["refit"],
    }
    for key in SCORING:
        scores = np.array([res[f"split{i}_test_{key}"][idx] for i in range(n_splits)])
        row[f"{key}_mean"] = float(scores.mean())
        row[f"{key}_std"] = float(scores.std(ddof=1))
    row["fit_time_s"] = float(res["mean_fit_time"][idx])
    row["best_params"] = {k.removeprefix("clf__"): _py(v) for k, v in search.best_params_.items()}
    return search, row


def tuned_pipeline(model, params, cfg, pos_weight):
    """Baseline pipeline for `model` with tuned classifier parameters applied."""
    pipe = build_pipeline(model, cfg["tuning"]["strategy"], cfg, pos_weight)
    return pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})


def load_best_params(path=None) -> dict:
    path = Path(path) if path else ROOT / load_config()["paths"]["reports_dir"] / "best_params_full.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_table(cfg):
    path = ROOT / cfg["paths"]["reports_dir"] / "baselines_full.csv"
    if not path.exists():
        return None
    base = pd.read_csv(path)
    base = base[base["strategy"] == cfg["tuning"]["strategy"]]
    return base[["model", "pr_auc_mean", "f1_mean"]].rename(
        columns={"pr_auc_mean": "pr_auc_baseline", "f1_mean": "f1_baseline"}
    )


def build_summary(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df[["model", "n_iter"]].copy()
    for key in ("pr_auc", "f1", "recall", "precision"):
        out[key] = [f"{m:.3f} +/- {s:.3f}" for m, s in zip(df[f"{key}_mean"], df[f"{key}_std"])]
    base = _baseline_table(cfg)
    if base is not None:
        merged = df[["model", "pr_auc_mean", "f1_mean"]].merge(base, on="model", how="left")
        out["pr_auc_baseline"] = merged["pr_auc_baseline"].round(3).values
        out["pr_auc_gain"] = (merged["pr_auc_mean"] - merged["pr_auc_baseline"]).round(3).values
        out["f1_baseline"] = merged["f1_baseline"].round(3).values
        out["f1_gain"] = (merged["f1_mean"] - merged["f1_baseline"]).round(3).values
    return out


def _merge_save_csv(new_df: pd.DataFrame, path) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["model"].isin(new_df["model"])]
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)
    return new_df


def _log_child(row: dict, cv_csv) -> None:
    with mlflow.start_run(run_name=f"tune-{row['model']}", nested=True):
        mlflow.log_params({k: row[k] for k in ("model", "strategy", "n_iter", "n_splits", "refit_metric")})
        mlflow.log_params({f"best_{k}": v for k, v in row["best_params"].items()})
        mlflow.log_metrics(
            {k: v for k, v in row.items() if k.endswith(("_mean", "_std")) or k == "fit_time_s"}
        )
        mlflow.log_artifact(str(cv_csv))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Randomised-search tuning (CV only).")
    ap.add_argument("--quick", action="store_true", help="3 folds, 2 candidates, 2 models")
    ap.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    ap.add_argument("--models", nargs="+", choices=list(SEARCH_SPACES), help="subset of models")
    args = ap.parse_args(argv)

    load_dotenv(ROOT / ".env")
    cfg = load_config()
    tn = cfg["tuning"]
    if args.quick:
        q = tn["quick"]
        models = args.models or q["models"]
        budgets = {m: q["n_iter"] for m in models}
        n_splits, tag = q["n_splits"], "quick"
    else:
        models = args.models or list(tn["n_iter"])
        budgets = {m: tn["n_iter"][m] for m in models}
        n_splits, tag = tn["n_splits"], "full"

    df = load_raw()
    validate(df, cfg)
    X_train, _, y_train, _ = split_data(df, cfg)
    print(f"Training rows: {len(X_train)}  |  models {models}  |  {n_splits}-fold CV  |  budgets {budgets}")

    out_dir = ROOT / cfg["paths"]["reports_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    use_mlflow = (not args.no_mlflow) and bool(os.getenv("MLFLOW_TRACKING_URI"))
    if use_mlflow:
        mlflow.set_experiment(tn["experiment"])
        parent = mlflow.start_run(run_name=f"tuning-{tag}")
        mlflow.log_params(
            {"models": ",".join(models), "strategy": tn["strategy"], "refit_metric": tn["refit"],
             "n_splits": n_splits, "seed": cfg["seed"], "n_train": len(X_train)}
        )
    else:
        parent = nullcontext()
        print("MLflow logging is OFF (no tracking URI, or --no-mlflow).")

    rows, failures = [], []
    t0 = time.time()
    with parent:
        for i, model in enumerate(models, start=1):
            print(f"[{i}/{len(models)}] tuning {model} ({budgets[model]} candidates) ...", flush=True)
            try:
                search, row = tune_model(model, X_train, y_train, cfg, budgets[model], n_splits)
            except Exception as e:
                print(f"    FAILED: {e}", flush=True)
                failures.append((model, str(e)))
                continue
            cv_csv = out_dir / f"cv_results_{model}_{tag}.csv"
            pd.DataFrame(search.cv_results_).drop(columns=["params"]).to_csv(cv_csv, index=False)
            print(
                f"    best CV pr_auc {row['pr_auc_mean']:.3f}  f1 {row['f1_mean']:.3f}  "
                f"recall {row['recall_mean']:.3f}  params {row['best_params']}",
                flush=True,
            )
            rows.append(row)
            if use_mlflow:
                _log_child(row, cv_csv)

        if rows:
            params_path = out_dir / f"best_params_{tag}.json"
            best = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
            best.update({r["model"]: r["best_params"] for r in rows})
            params_path.write_text(json.dumps(best, indent=2), encoding="utf-8")

            csv_path = out_dir / f"tuning_{tag}.csv"
            md_path = out_dir / f"tuning_{tag}.md"
            new_df = pd.DataFrame([{**r, "best_params": json.dumps(r["best_params"])} for r in rows])
            merged = _merge_save_csv(new_df, csv_path)
            summary = build_summary(merged, cfg)
            md_path.write_text(
                f"# Tuned models ({tag}, {n_splits}-fold CV, training split only)\n\n"
                f"Strategy: {tn['strategy']}. Search metric: {tn['refit']}. Mean +/- SD across folds at the best candidate.\n"
                "The best CV score of a search is optimistically biased; the unbiased estimate is the held-out test.\n"
                "`*_baseline` = baseline with the same strategy; `*_gain` = tuned minus baseline.\n\n"
                + _md_table(summary) + "\n",
                encoding="utf-8",
            )
            if use_mlflow:
                for p in (params_path, csv_path, md_path):
                    mlflow.log_artifact(str(p))
            print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Wrote:\n  {params_path}\n  {csv_path}\n  {md_path}")
            print(summary.to_string(index=False))

    if failures:
        print(f"\n{len(failures)} model(s) FAILED:")
        for m, e in failures:
            print(f"  {m}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())