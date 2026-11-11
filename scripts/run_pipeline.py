from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from mlnd.config import ROOT, load_config


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    args: tuple = ()
    outputs: tuple = ()
    logs_to_mlflow: bool = False
    after: tuple = ()


def build_stages(reports: str) -> list[Stage]:
    def r(name: str) -> str:
        return f"{reports}/{name}"

    return [
        Stage("audit", "mlnd.data", outputs=(r("data_audit.json"), r("data_audit.md"))),
        Stage(
            "baselines", "mlnd.train", logs_to_mlflow=True,
            outputs=(r("baselines_full.csv"), r("baselines_full.md"), r("baseline_best_strategy_full.json")),
        ),
        Stage(
            "tuning", "mlnd.tune", logs_to_mlflow=True,
            outputs=(r("best_params_full.json"), r("tuning_full.csv"), r("tuning_full.md"), r("cv_results_*_full.csv")),
        ),
        Stage(
            "stacking", "mlnd.stack", logs_to_mlflow=True, after=("tuning",),
            outputs=(r("stacking_full.md"), r("stacking_per_fold_full.csv"), r("oof_probabilities_full.csv")),
        ),
        Stage(
            "threshold", "mlnd.threshold", logs_to_mlflow=True, after=("stacking",),
            outputs=("artifacts/threshold.json", r("threshold.md"), r("threshold_table.csv")),
        ),
        Stage("loco", "mlnd.evaluate", args=("loco",), after=("tuning", "threshold"), outputs=(r("loco.csv"), r("loco.md"))),
        Stage(
            "test", "mlnd.evaluate", args=("test",), after=("tuning", "threshold"),
            outputs=(
                r("final_evaluation.json"), r("test_evaluation.md"), r("test_per_county.csv"), r("test_calibration.csv"),
                "artifacts/model.joblib", "artifacts/model_metadata.json",
            ),
        ),
        Stage(
            "explain", "mlnd.explain", after=("test",),
            outputs=(r("shap_global_importance.csv"), r("shap_global_importance.md"), r("shap_example_explanations.json")),
        ),
    ]


def quick_stages(reports: str) -> list[Stage]:
    def r(name: str) -> str:
        return f"{reports}/{name}"

    return [
        build_stages(reports)[0],
        Stage(
            "baselines", "mlnd.train", args=("--quick",), logs_to_mlflow=True,
            outputs=(r("baselines_quick.csv"), r("baselines_quick.md"), r("baseline_best_strategy_quick.json")),
        ),
        Stage(
            "tuning", "mlnd.tune", args=("--quick",), logs_to_mlflow=True,
            outputs=(r("best_params_quick.json"), r("tuning_quick.csv"), r("tuning_quick.md"), r("cv_results_*_quick.csv")),
        ),
    ]


def is_done(stage: Stage, root: Path) -> bool:
    return all(any(root.glob(pattern)) for pattern in stage.outputs)


def outputs_of(stages, root: Path) -> list[Path]:
    return sorted({p for s in stages for pattern in s.outputs for p in root.glob(pattern)})


def command(stage: Stage, no_mlflow: bool) -> list[str]:
    cmd = [sys.executable, "-m", stage.module, *stage.args]
    if no_mlflow and stage.logs_to_mlflow:
        cmd.append("--no-mlflow")
    return cmd


def run_stage(stage: Stage, no_mlflow: bool) -> int:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(command(stage, no_mlflow), cwd=ROOT, env=env).returncode


def run(
    stages, selected, root: Path, *, no_mlflow=False, from_scratch=False, assume_yes=False,
    register=False, runner=run_stage, confirm=input,
) -> int:
    chosen = [s for s in stages if not selected or s.name in selected]
    by_name = {s.name: s for s in stages}

    if from_scratch:
        doomed = outputs_of(chosen, root)
        print(f"--from-scratch will delete {len(doomed)} generated file(s) and re-run: {[s.name for s in chosen]}")
        if any(p.name == "final_evaluation.json" for p in doomed):
            print(
                "This removes final_evaluation.json, so the held-out test split will be scored again. "
                "Done only for a reproducibility rerun in which nothing about the model has changed."
            )
        if not assume_yes and confirm("Continue? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return 1
        for path in doomed:
            path.unlink()

    results, failed = [], False
    for stage in chosen:
        blocked = [d for d in stage.after if not is_done(by_name[d], root)]
        if blocked:
            print(f"\n[{stage.name}] blocked: run {blocked} first")
            results.append((stage.name, "blocked", 0.0))
            failed = True
            break
        if is_done(stage, root):
            print(f"\n[{stage.name}] skipped: outputs exist")
            results.append((stage.name, "skipped", 0.0))
            continue
        print(f"\n[{stage.name}] running: {' '.join(command(stage, no_mlflow)[1:])}", flush=True)
        t0 = time.time()
        code = runner(stage, no_mlflow)
        elapsed = time.time() - t0
        if code != 0 or not is_done(stage, root):
            reason = f"exit code {code}" if code != 0 else "expected outputs are missing"
            print(f"[{stage.name}] FAILED ({reason})")
            results.append((stage.name, "FAILED", elapsed))
            failed = True
            break
        results.append((stage.name, "ran", elapsed))

    if register and not failed:
        if "test" not in by_name:
            print("\n--register needs the full pipeline (it registers the final model).")
            failed = True
        else:
            t0 = time.time()
            code = runner(Stage("register", "mlnd.evaluate", args=("register",)), no_mlflow)
            results.append(("register", "ran" if code == 0 else "FAILED", time.time() - t0))
            failed = code != 0

    print("\nSummary")
    for name, status, seconds in results:
        print(f"  {name:10s} {status:8s} {seconds / 60:5.1f} min")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the whole MLND pipeline.")
    ap.add_argument("--from-scratch", action="store_true", help="delete the generated outputs of the chosen stages first")
    ap.add_argument("--stages", nargs="+", help="run only these stages")
    ap.add_argument("--quick", action="store_true", help="audit plus quick baselines and tuning (smoke test)")
    ap.add_argument("--no-mlflow", action="store_true", help="do not log experiments to MLflow")
    ap.add_argument("--register", action="store_true", help="register the final model on MLflow afterwards")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation with --from-scratch")
    ap.add_argument("--list", action="store_true", help="show the stages and their status, then exit")
    args = ap.parse_args(argv)

    if args.register and args.no_mlflow:
        ap.error("--register cannot be combined with --no-mlflow")
    reports = load_config()["paths"]["reports_dir"]
    stages = quick_stages(reports) if args.quick else build_stages(reports)
    names = [s.name for s in stages]
    unknown = set(args.stages or []) - set(names)
    if unknown:
        ap.error(f"unknown stage(s) {sorted(unknown)}; choose from {names}")

    if args.list:
        for s in stages:
            print(f"{s.name:10s} {'done' if is_done(s, ROOT) else 'missing':8s} {', '.join(s.outputs)}")
        return 0
    return run(
        stages, args.stages, ROOT, no_mlflow=args.no_mlflow, from_scratch=args.from_scratch,
        assume_yes=args.yes, register=args.register,
    )


if __name__ == "__main__":
    raise SystemExit(main())