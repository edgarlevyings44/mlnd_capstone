from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from mlnd.config import ROOT, load_config
from mlnd.data import _md_table, load_raw, validate
from mlnd.features import split_data
from mlnd.threshold import load_threshold
from mlnd.tune import load_best_params, tuned_pipeline

METRICS = ("precision", "recall", "specificity", "f1", "mcc", "gmean", "roc_auc", "pr_auc")


def metrics_at(y, p, t) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pred = p >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    denom = float(np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    two_classes = len(np.unique(y)) > 1
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "gmean": float(np.sqrt(recall * specificity)),
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
    }


def bootstrap_ci(y, p, t, n_boot, seed) -> dict:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    reps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        reps.append(metrics_at(y[idx], p[idx], t))
    df = pd.DataFrame(reps)
    return {c: [float(v) for v in np.nanpercentile(df[c], [2.5, 97.5])] for c in df.columns}


def per_county(counties, y, p, t, n_boot, seed) -> pd.DataFrame:
    counties, y, p = np.asarray(counties), np.asarray(y), np.asarray(p)
    rows = []
    for c in sorted(set(counties)):
        m = counties == c
        pt = metrics_at(y[m], p[m], t)
        ci = bootstrap_ci(y[m], p[m], t, n_boot, seed)
        rows.append(
            {
                "county": c, "n": int(m.sum()), "infected": int(y[m].sum()),
                **{k: pt[k] for k in ("pr_auc", "roc_auc", "recall", "precision", "f1", "mcc")},
                "pr_auc_lo": ci["pr_auc"][0], "pr_auc_hi": ci["pr_auc"][1],
                "recall_lo": ci["recall"][0], "recall_hi": ci["recall"][1],
            }
        )
    return pd.DataFrame(rows)


def calibration_table(y, p, n_bins=10):
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    g = (
        df.groupby("bin", observed=True)
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index(drop=True)
    )
    ece = float((g["n"] * (g["mean_predicted"] - g["observed_rate"]).abs()).sum() / g["n"].sum())
    return g, ece


def fit_final(cfg, X, y):
    name = cfg["threshold"]["model"]
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    pipe = tuned_pipeline(name, load_best_params()[name], cfg, pos_weight)
    return pipe.fit(X, y)


def holdout_split(X, y, county, county_col="County"):
    mask = (X[county_col] == county).to_numpy()
    return X[~mask], y[~mask], X[mask], y[mask]


def _locked(lock_path: Path, force: bool) -> bool:
    if lock_path.exists() and not force:
        print(
            f"REFUSED: {lock_path.name} exists, so the test split has already been scored once.\n"
            "Re-running after seeing test results is test-set tuning. Use --force only if nothing changed."
        )
        return True
    return False


def library_versions() -> dict:
    names = ("scikit-learn", "catboost", "pandas", "numpy", "imbalanced-learn", "mlflow")
    return {n: version(n) for n in names} | {"python": platform.python_version()}


def run_loco(cfg: dict) -> int:
    df = load_raw()
    validate(df, cfg)
    X_train, _, y_train, _ = split_data(df, cfg)
    t = load_threshold()
    n_boot = max(200, cfg["evaluation"]["n_boot"] // 2)
    rows = []
    for county in cfg["counties"]:
        X_tr, y_tr, X_ho, y_ho = holdout_split(X_train, y_train, county, cfg["county"])
        t0 = time.time()
        model = fit_final(cfg, X_tr, y_tr)
        p = model.predict_proba(X_ho)[:, 1]
        yy = y_ho.to_numpy()
        pt, ci = metrics_at(yy, p, t), bootstrap_ci(yy, p, t, n_boot, cfg["seed"])
        rows.append(
            {"held_out_county": county, "n": len(yy), "infected": int(yy.sum()),
             **{k: pt[k] for k in ("pr_auc", "roc_auc", "recall", "precision", "f1", "mcc")},
             "pr_auc_lo": ci["pr_auc"][0], "pr_auc_hi": ci["pr_auc"][1]}
        )
        print(f"  held out {county:12s} pr_auc {pt['pr_auc']:.3f}  recall {pt['recall']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    out = pd.DataFrame(rows)
    reports = ROOT / cfg["paths"]["reports_dir"]
    reports.mkdir(parents=True, exist_ok=True)
    out.to_csv(reports / "loco.csv", index=False)
    gap_pr = out["pr_auc"].max() - out["pr_auc"].min()
    gap_rec = out["recall"].max() - out["recall"].min()
    (reports / "loco.md").write_text(
        "# Leave-one-county-out (training split only)\n\n"
        f"Train on five counties, score the sixth, at threshold {t:.2f}. Best-to-worst gap: PR-AUC {gap_pr:.3f}, recall {gap_rec:.3f}.\n"
        "The settings and the threshold were chosen using all six counties, so these scores may be a little better than for a truly new county.\n\n"
        + _md_table(out.round(3)) + "\n",
        encoding="utf-8",
    )
    print(f"Gap best-to-worst county: PR-AUC {gap_pr:.3f}, recall {gap_rec:.3f}")
    print(out.round(3).to_string(index=False))
    return 0


def run_test(cfg: dict, force: bool) -> int:
    reports, art = ROOT / cfg["paths"]["reports_dir"], ROOT / "artifacts"
    lock = reports / "final_evaluation.json"
    if _locked(lock, force):
        return 1
    reports.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    validate(df, cfg)
    X_train, X_test, y_train, y_test = split_data(df, cfg)
    t = load_threshold()
    n_boot, seed = cfg["evaluation"]["n_boot"], cfg["seed"]

    print("Fitting the final model on the full training split ...", flush=True)
    model = fit_final(cfg, X_train, y_train)
    p, y = model.predict_proba(X_test)[:, 1], y_test.to_numpy()

    overall, ci = metrics_at(y, p, t), bootstrap_ci(y, p, t, n_boot, seed)
    at_default = metrics_at(y, p, 0.5)
    by_county = per_county(X_test[cfg["county"]], y, p, t, max(200, n_boot // 2), seed)
    calib, ece = calibration_table(y, p)
    prevalence = float(y.mean())
    calibration = {
        "brier": float(brier_score_loss(y, p)),
        "brier_of_constant_prevalence": prevalence * (1 - prevalence),
        "ece": ece,
        "mean_predicted": float(p.mean()),
        "prevalence": prevalence,
    }
    gap = {"pr_auc": float(by_county["pr_auc"].max() - by_county["pr_auc"].min()),
           "recall": float(by_county["recall"].max() - by_county["recall"].min())}

    joblib.dump(model, art / "model.joblib")
    name = cfg["threshold"]["model"]
    (art / "model_metadata.json").write_text(
        json.dumps(
            {
                "model": name,
                "hyperparameters": load_best_params()[name],
                "imbalance_strategy": cfg["tuning"]["strategy"],
                "threshold": t,
                "input_columns": list(X_train.columns),
                "counties": cfg["counties"],
                "label_meaning": {"0": "healthy", "1": "MLND infected"},
                "n_train": int(len(X_train)),
                "seed": seed,
                "library_versions": library_versions(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    by_county.to_csv(reports / "test_per_county.csv", index=False)
    calib.to_csv(reports / "test_calibration.csv", index=False)

    overall_df = pd.DataFrame(
        {"metric": METRICS, "estimate": [overall[m] for m in METRICS],
         "ci_lo": [ci[m][0] for m in METRICS], "ci_hi": [ci[m][1] for m in METRICS]}
    ).round(3)
    (reports / "test_evaluation.md").write_text(
        f"# Final evaluation on the held-out test split ({name}, threshold {t:.2f})\n\n"
        f"{len(y)} test leaves, {int(y.sum())} infected. Scored once; nothing was changed after seeing these numbers.\n\n"
        "## Overall (95% bootstrap CI)\n" + _md_table(overall_df) + "\n\n"
        f"At the default 0.5 threshold, for reference: recall {at_default['recall']:.3f}, precision {at_default['precision']:.3f}, F1 {at_default['f1']:.3f}.\n\n"
        "## Per county (test split; small samples, wide intervals)\n" + _md_table(by_county.round(3)) + "\n\n"
        f"Best-to-worst county gap: PR-AUC {gap['pr_auc']:.3f}, recall {gap['recall']:.3f}.\n\n"
        "## Calibration\n"
        f"Brier {calibration['brier']:.4f} (constant-prevalence baseline {calibration['brier_of_constant_prevalence']:.4f}), "
        f"ECE {ece:.3f}, mean predicted risk {calibration['mean_predicted']:.3f} vs true infection rate {prevalence:.3f}.\n\n"
        + _md_table(calib.round(3)) + "\n",
        encoding="utf-8",
    )
    lock.write_text(
        json.dumps(
            {
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": name, "threshold": t, "n_test": int(len(y)), "n_test_infected": int(y.sum()),
                "overall": overall, "overall_ci95": ci, "at_default_0.5": at_default,
                "calibration": calibration, "per_county_gap": gap,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Test split ({len(y)} leaves, {int(y.sum())} infected), threshold {t:.2f}:")
    print(overall_df.to_string(index=False))
    print(f"Calibration: mean predicted {p.mean():.3f} vs true rate {prevalence:.3f}, ECE {ece:.3f}, Brier {calibration['brier']:.4f}")
    print(f"Per-county gap: PR-AUC {gap['pr_auc']:.3f}, recall {gap['recall']:.3f}")
    print(f"Wrote {art / 'model.joblib'} and the reports.")
    return 0


def _log_and_register(model, run_id: str, name: str, art: Path) -> None:
    client = mlflow.MlflowClient()
    try:
        mlflow.sklearn.log_model(model, name="model", registered_model_name=name)
        how = "MLflow sklearn flavour"
    except Exception as e:
        print(f"log_model failed ({type(e).__name__}: {e}); falling back to a plain artifact.")
        mlflow.log_artifact(str(art / "model.joblib"), artifact_path="model_file")
        try:
            client.create_registered_model(name)
        except Exception:
            pass
        client.create_model_version(name=name, source=f"runs:/{run_id}/model_file", run_id=run_id)
        how = "plain joblib artifact"
    versions = client.search_model_versions(f"name='{name}'")
    latest = max(int(v.version) for v in versions)
    try:
        client.set_registered_model_alias(name, "champion", str(latest))
    except Exception as e:
        print(f"Could not set the 'champion' alias ({type(e).__name__}); the version still exists.")
    print(f"Registered '{name}' version {latest} ({how}).")


def run_register(cfg: dict) -> int:
    load_dotenv(ROOT / ".env")
    if not os.getenv("MLFLOW_TRACKING_URI"):
        print("MLFLOW_TRACKING_URI is not set; nothing to register.")
        return 1
    reports, art = ROOT / cfg["paths"]["reports_dir"], ROOT / "artifacts"
    res = json.loads((reports / "final_evaluation.json").read_text(encoding="utf-8"))
    meta = json.loads((art / "model_metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(art / "model.joblib")

    mlflow.set_experiment(cfg["evaluation"]["experiment"])
    with mlflow.start_run(run_name="final-model") as run:
        mlflow.log_params({"model": meta["model"], "threshold": meta["threshold"],
                           "imbalance_strategy": meta["imbalance_strategy"], "n_train": meta["n_train"],
                           **{f"hp_{k}": v for k, v in meta["hyperparameters"].items()}})
        for m, v in res["overall"].items():
            mlflow.log_metric(f"test_{m}", v)
            mlflow.log_metrics({f"test_{m}_ci_lo": res["overall_ci95"][m][0], f"test_{m}_ci_hi": res["overall_ci95"][m][1]})
        mlflow.log_metrics({f"calibration_{k}": v for k, v in res["calibration"].items()})
        for path in (reports / "test_evaluation.md", reports / "test_per_county.csv",
                     reports / "test_calibration.csv", reports / "loco.md",
                     art / "threshold.json", art / "model_metadata.json"):
            if path.exists():
                mlflow.log_artifact(str(path))
        _log_and_register(model, run.info.run_id, cfg["evaluation"]["registered_model"], art)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Final evaluation, robustness and registry.")
    ap.add_argument("mode", choices=["loco", "test", "register"])
    ap.add_argument("--force", action="store_true", help="allow a second scoring of the test split")
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.mode == "loco":
        return run_loco(cfg)
    if args.mode == "test":
        return run_test(cfg, args.force)
    return run_register(cfg)


if __name__ == "__main__":
    raise SystemExit(main())