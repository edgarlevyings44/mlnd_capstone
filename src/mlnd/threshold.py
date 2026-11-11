"""Choose the decision threshold on out-of-fold predictions."""
from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from mlnd.config import ROOT, load_config
from mlnd.data import _md_table

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def threshold_table(y, p, grid=GRID) -> pd.DataFrame:
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=float)
    grid = np.asarray(grid, dtype=float)
    pred = p[:, None] >= grid[None, :]
    pos = y[:, None]
    tp = (pred & pos).sum(axis=0)
    fp = (pred & ~pos).sum(axis=0)
    fn = (~pred & pos).sum(axis=0)
    tn = (~pred & ~pos).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        specificity = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f2 = np.where(5 * tp + 4 * fn + fp > 0, 5 * tp / (5 * tp + 4 * fn + fp), 0.0)
    return pd.DataFrame(
        {
            "threshold": grid,
            "recall": recall,
            "precision": precision,
            "specificity": specificity,
            "f1": f1,
            "f2": f2,
            "gmean": np.sqrt(recall * specificity),
            "flagged_share": (tp + fp) / len(y),
        }
    )


def choose_threshold(table: pd.DataFrame, policy: str, recall_target: float | None = None) -> float:
    if policy == "f1":
        return float(table.loc[table["f1"].idxmax(), "threshold"])
    if policy == "f2":
        return float(table.loc[table["f2"].idxmax(), "threshold"])
    if policy == "recall_target":
        ok = table[table["recall"] >= recall_target]
        if ok.empty:
            raise ValueError(f"No threshold reaches recall >= {recall_target}")
        return float(ok["threshold"].max())
    raise ValueError(f"Unknown policy: {policy}")


def bootstrap_threshold(y, p, policy, recall_target, t, n_boot, seed) -> dict:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    thresholds, recalls, precisions = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        thresholds.append(choose_threshold(threshold_table(yb, pb), policy, recall_target))
        at_t = threshold_table(yb, pb, np.array([t])).iloc[0]
        recalls.append(at_t["recall"])
        precisions.append(at_t["precision"])

    def ci(values):
        return [float(v) for v in np.percentile(values, [2.5, 97.5])]

    return {
        "threshold_median": float(np.median(thresholds)),
        "threshold_ci95": ci(thresholds),
        "recall_ci95": ci(recalls),
        "precision_ci95": ci(precisions),
    }


def load_threshold(path=None) -> float:
    path = Path(path) if path else ROOT / "artifacts" / "threshold.json"
    return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])


def run(cfg: dict, artifacts_dir: Path, reports_dir: Path, n_boot: int | None = None) -> dict:
    th = cfg["threshold"]
    n_boot = n_boot or th["n_boot"]
    oof = pd.read_csv(ROOT / cfg["paths"]["reports_dir"] / th["oof_file"], index_col="row_id")
    y, p = oof["y"].to_numpy(), oof[th["model"]].to_numpy()

    table = threshold_table(y, p)
    target = th["recall_target"]
    t = choose_threshold(table, th["policy"], target)
    row = table.loc[table["threshold"] == t].iloc[0]

    points = {
        "default 0.5": 0.5,
        "max F1": choose_threshold(table, "f1"),
        "max F2": choose_threshold(table, "f2"),
        f"recall >= {target}": choose_threshold(table, "recall_target", target),
    }
    op = pd.DataFrame(
        [{"operating_point": k, **table.loc[table["threshold"] == v].iloc[0].to_dict()} for k, v in points.items()]
    )
    boot = bootstrap_threshold(y, p, th["policy"], target, t, n_boot, cfg["seed"])

    result = {
        "model": th["model"],
        "policy": th["policy"],
        "recall_target": target,
        "threshold": t,
        "oof_estimates": {k: float(row[k]) for k in ("recall", "precision", "specificity", "f1", "f2", "gmean", "flagged_share")},
        **boot,
        "n_boot": n_boot,
        "n_oof_rows": int(len(y)),
        "source": th["oof_file"],
        "selected_on": "out-of-fold predictions of the training split",
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "threshold.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    table.to_csv(reports_dir / "threshold_table.csv", index=False)

    every_05 = table[(table["threshold"] * 100).round().astype(int) % 5 == 0].round(3)
    e = result["oof_estimates"]
    (reports_dir / "threshold.md").write_text(
        f"# Threshold tuning ({th['model']}, policy: {th['policy']})\n\n"
        f"Chosen threshold: **{t:.2f}** (bootstrap median {boot['threshold_median']:.2f}, "
        f"95% interval [{boot['threshold_ci95'][0]:.2f}, {boot['threshold_ci95'][1]:.2f}]).\n\n"
        f"On out-of-fold predictions this flags {e['flagged_share']:.1%} of leaves, finds {e['recall']:.1%} of infected leaves "
        f"(95% CI [{boot['recall_ci95'][0]:.3f}, {boot['recall_ci95'][1]:.3f}]), and {e['precision']:.1%} of flagged leaves are truly infected "
        f"(95% CI [{boot['precision_ci95'][0]:.3f}, {boot['precision_ci95'][1]:.3f}]).\n\n"
        "## Operating points\n" + _md_table(op.round(3)) + "\n\n"
        "## Trade-off every 0.05\n" + _md_table(every_05) + "\n\n"
        "Notes: The threshold was chosen on out-of-fold predictions only, so the test split was not used. "
        f"Each of the five cross-validation {th['model']} models was trained on 80% of the training rows, while the deployed "
        f"{th['model']} model is trained on all of them, so real recall may differ slightly. Class weights make the scores run "
        "higher than the true infection rate, so they are not probabilities. We retune the threshold if the model changes.\n",
        encoding="utf-8",
    )
    result["_operating_points"] = op
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Choose the decision threshold on out-of-fold predictions.")
    ap.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    args = ap.parse_args(argv)

    load_dotenv(ROOT / ".env")
    cfg = load_config()
    th = cfg["threshold"]
    artifacts_dir, reports_dir = ROOT / "artifacts", ROOT / cfg["paths"]["reports_dir"]

    use_mlflow = (not args.no_mlflow) and bool(os.getenv("MLFLOW_TRACKING_URI"))
    if use_mlflow:
        mlflow.set_experiment(th["experiment"])
        parent = mlflow.start_run(run_name="threshold")
    else:
        parent = nullcontext()
        print("MLflow logging is OFF (no tracking URI, or --no-mlflow).")

    with parent:
        result = run(cfg, artifacts_dir, reports_dir)
        op = result.pop("_operating_points")
        if use_mlflow:
            mlflow.log_params({"model": th["model"], "policy": th["policy"],
                               "recall_target": th["recall_target"], "n_boot": result["n_boot"]})
            mlflow.log_metrics({"threshold": result["threshold"],
                                **{f"oof_{k}": v for k, v in result["oof_estimates"].items()},
                                "threshold_ci_lo": result["threshold_ci95"][0], "threshold_ci_hi": result["threshold_ci95"][1],
                                "recall_ci_lo": result["recall_ci95"][0], "recall_ci_hi": result["recall_ci95"][1]})
            for path in (artifacts_dir / "threshold.json", reports_dir / "threshold.md", reports_dir / "threshold_table.csv"):
                mlflow.log_artifact(str(path))

    e = result["oof_estimates"]
    print(f"Chosen threshold {result['threshold']:.2f} (policy {th['policy']}, model {th['model']})")
    print(f"  OOF recall {e['recall']:.3f}  precision {e['precision']:.3f}  f1 {e['f1']:.3f}  flagged {e['flagged_share']:.1%}")
    print(f"  threshold 95% interval {result['threshold_ci95']}  recall CI {result['recall_ci95']}")
    print(op[["operating_point", "threshold", "recall", "precision", "f1", "flagged_share"]].round(3).to_string(index=False))
    print(f"Wrote {artifacts_dir / 'threshold.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())