"""Stacked ensemble vs the best single model (nested, paired comparison)."""
from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from mlnd.config import ROOT, load_config
from mlnd.data import _md_table, load_raw, validate
from mlnd.features import split_data
from mlnd.tune import load_best_params, tuned_pipeline


def build_models(cfg, params, pos_weight, bases, singles, inner_splits):
    stack = StackingClassifier(
        estimators=[(m, tuned_pipeline(m, params[m], cfg, pos_weight)) for m in bases],
        final_estimator=LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=cfg["seed"]
        ),
        cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=cfg["seed"]),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )
    models = {m: tuned_pipeline(m, params[m], cfg, pos_weight) for m in singles}
    models["stack"] = stack
    return models


def point_metrics(y, p, thr=0.5) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
    }


def best_f1(y, p) -> float:
    prec, rec, _ = precision_recall_curve(y, p)
    denom = prec + rec
    f = np.where(denom > 0, 2 * prec * rec / np.where(denom > 0, denom, 1.0), 0.0)
    return float(f.max())


def out_of_fold(models, X, y, outer):
    y_arr = np.asarray(y)
    folds = list(outer.split(X, y_arr))
    probs, per_fold = {}, []
    for name, model in models.items():
        t0 = time.time()
        p = cross_val_predict(model, X, y_arr, cv=folds, method="predict_proba", n_jobs=1)[:, 1]
        probs[name] = p
        for k, (_, val_idx) in enumerate(folds, start=1):
            per_fold.append({"model": name, "fold": k, **point_metrics(y_arr[val_idx], p[val_idx])})
        print(f"    {name:20s} pooled pr_auc {average_precision_score(y_arr, p):.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return probs, per_fold


def paired_bootstrap(y, probs, n_boot, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    out = {m: {"pr_auc": [], "best_f1": []} for m in probs}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0:
            continue
        for m, p in probs.items():
            pb = p[idx]
            out[m]["pr_auc"].append(average_precision_score(yb, pb))
            out[m]["best_f1"].append(best_f1(yb, pb))
    return {m: {k: np.array(v) for k, v in d.items()} for m, d in out.items()}


def verdict_word(lo, hi) -> str:
    if lo > 0:
        return "WINS"
    if hi < 0:
        return "LOSES"
    return "TIES"


def verdict(lo, hi, mean_diff, best_name, metric="PR-AUC") -> str:
    word = {"WINS": "WINS against", "LOSES": "LOSES to", "TIES": "TIES with"}[verdict_word(lo, hi)]
    return (
        f"Stacking {word} the best single model ({best_name}) on {metric}: "
        f"difference {mean_diff:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]."
    )


def summarize(y, probs, boots) -> pd.DataFrame:
    rows = []
    for m, p in probs.items():
        pm = point_metrics(y, p)
        pr_lo, pr_hi = np.percentile(boots[m]["pr_auc"], [2.5, 97.5])
        bf_lo, bf_hi = np.percentile(boots[m]["best_f1"], [2.5, 97.5])
        rows.append(
            {
                "model": m,
                "pr_auc": pm["pr_auc"], "pr_auc_lo": pr_lo, "pr_auc_hi": pr_hi,
                "best_f1": best_f1(y, p), "best_f1_lo": bf_lo, "best_f1_hi": bf_hi,
                "f1_at_0.5": pm["f1"], "recall_at_0.5": pm["recall"], "precision_at_0.5": pm["precision"],
            }
        )
    return pd.DataFrame(rows)


def diff_table(boots, reference="stack") -> pd.DataFrame:
    rows = []
    for m in boots:
        if m == reference:
            continue
        for metric in ("pr_auc", "best_f1"):
            d = boots[reference][metric] - boots[m][metric]
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append(
                {"stack_vs": m, "metric": metric, "mean_diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                 "verdict": verdict_word(lo, hi)}
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stacked ensemble vs best single model (nested CV).")
    ap.add_argument("--quick", action="store_true", help="3 folds, 2 base learners, 200 bootstraps")
    ap.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    args = ap.parse_args(argv)

    load_dotenv(ROOT / ".env")
    cfg = load_config()
    st = cfg["stacking"]
    if args.quick:
        q = st["quick"]
        bases, singles, outer_n, inner_n, n_boot, tag = (
            q["bases"], q["singles"], q["outer_splits"], q["inner_splits"], q["n_boot"], "quick")
    else:
        bases, singles, outer_n, inner_n, n_boot, tag = (
            st["bases"], st["singles"], st["outer_splits"], st["inner_splits"], st["n_boot"], "full")

    df = load_raw()
    validate(df, cfg)
    X_train, _, y_train, _ = split_data(df, cfg)
    y = np.asarray(y_train)
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    models = build_models(cfg, load_best_params(), pos_weight, bases, singles, inner_n)
    outer = StratifiedKFold(outer_n, shuffle=True, random_state=cfg["seed"] + st["seed_offset"])
    base_names = [n for n, _ in models["stack"].estimators]
    print(f"Training rows: {len(X_train)} | bases {base_names} | outer {outer_n}-fold, inner {inner_n}-fold | {n_boot} bootstraps")

    out_dir = ROOT / cfg["paths"]["reports_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    use_mlflow = (not args.no_mlflow) and bool(os.getenv("MLFLOW_TRACKING_URI"))
    if use_mlflow:
        mlflow.set_experiment(st["experiment"])
        parent = mlflow.start_run(run_name=f"stacking-{tag}")
    else:
        parent = nullcontext()
        print("MLflow logging is OFF (no tracking URI, or --no-mlflow).")

    t0 = time.time()
    with parent:
        print("Out-of-fold predictions:")
        probs, per_fold = out_of_fold(models, X_train, y, outer)
        print(f"Bootstrapping ({n_boot} resamples) ...", flush=True)
        boots = paired_bootstrap(y, probs, n_boot, cfg["seed"])

        summary = summarize(y, probs, boots)
        diffs = diff_table(boots)
        per_fold_df = pd.DataFrame(per_fold)

        singles_only = summary[summary["model"] != "stack"]
        best = singles_only.loc[singles_only["pr_auc"].idxmax(), "model"]
        d_pr = boots["stack"]["pr_auc"] - boots[best]["pr_auc"]
        d_f1 = boots["stack"]["best_f1"] - boots[best]["best_f1"]
        lo_pr, hi_pr = np.percentile(d_pr, [2.5, 97.5])
        lo_f1, hi_f1 = np.percentile(d_f1, [2.5, 97.5])
        v_pr = verdict(lo_pr, hi_pr, d_pr.mean(), best, "PR-AUC")
        v_f1 = verdict(lo_f1, hi_f1, d_f1.mean(), best, "best-threshold F1")

        pf_pr = per_fold_df.pivot(index="fold", columns="model", values="pr_auc")[list(probs)].round(3).reset_index()
        wins = int((pf_pr["stack"] > pf_pr[best]).sum())

        md_path = out_dir / f"stacking_{tag}.md"
        pf_path = out_dir / f"stacking_per_fold_{tag}.csv"
        oof_path = out_dir / f"oof_probabilities_{tag}.csv"
        per_fold_df.to_csv(pf_path, index=False)
        oof = pd.DataFrame(probs, index=X_train.index)
        oof["y"] = y
        oof.rename_axis("row_id").to_csv(oof_path)

        md_path.write_text(
            f"# Stacking vs single models ({tag})\n\n"
            f"Outer {outer_n}-fold CV on the training split, inner {inner_n}-fold for the meta-learner, "
            f"{n_boot} paired bootstrap resamples of the pooled out-of-fold predictions.\n"
            f"Bases: {base_names}. Meta-learner: logistic regression (balanced).\n\n"
            f"## Verdict\n- {v_pr}\n- {v_f1}\n- Per-fold PR-AUC: stack beat {best} in {wins} of {outer_n} folds.\n\n"
            "## Pooled out-of-fold performance (95% bootstrap CI)\n"
            + _md_table(summary.round(4)) + "\n\n"
            "`best_f1` = highest F1 over thresholds. `*_at_0.5` use the default threshold and depend on calibration; "
            "they are information only.\n\n"
            "## Per-fold PR-AUC\n" + _md_table(pf_pr) + "\n\n"
            "## Stack minus each single model (paired)\n" + _md_table(diffs.round(4)) + "\n\n"
            "Hyperparameters were tuned on these same rows, so scores are slightly optimistic for every model; "
            "the held-out test is the unbiased comparison.\n",
            encoding="utf-8",
        )

        if use_mlflow:
            mlflow.log_params(
                {"bases": ",".join(base_names), "singles": ",".join(singles),
                 "outer_splits": outer_n, "inner_splits": inner_n, "n_boot": n_boot,
                 "best_single": best, "meta_learner": "logistic_regression_balanced"}
            )
            for _, r in summary.iterrows():
                mlflow.log_metrics({f"{r['model']}_pr_auc": r["pr_auc"], f"{r['model']}_best_f1": r["best_f1"]})
            mlflow.log_metrics({"diff_pr_auc_mean": float(d_pr.mean()), "diff_pr_auc_lo": float(lo_pr),
                                "diff_pr_auc_hi": float(hi_pr), "diff_best_f1_mean": float(d_f1.mean()),
                                "diff_best_f1_lo": float(lo_f1), "diff_best_f1_hi": float(hi_f1)})
            mlflow.set_tag("verdict_pr_auc", v_pr)
            for p in (md_path, pf_path, oof_path):
                mlflow.log_artifact(str(p))

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Wrote:\n  {md_path}\n  {pf_path}\n  {oof_path}\n")
    print(summary[["model", "pr_auc", "pr_auc_lo", "pr_auc_hi", "best_f1"]].round(3).to_string(index=False))
    print(f"\n{v_pr}\n{v_f1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())