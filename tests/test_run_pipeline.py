import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"
spec = importlib.util.spec_from_file_location("run_pipeline", SCRIPT)
rp = importlib.util.module_from_spec(spec)
sys.modules["run_pipeline"] = rp
spec.loader.exec_module(rp)

STAGES = rp.build_stages("reports")
NAMES = ["audit", "baselines", "tuning", "stacking", "threshold", "loco", "test", "explain"]


def touch(root, stage):
    for pattern in stage.outputs:
        path = root / pattern.replace("*", "x")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")


def touch_all(root):
    for stage in STAGES:
        touch(root, stage)


def test_stages_are_in_pipeline_order():
    assert [s.name for s in STAGES] == NAMES


def test_a_stage_is_done_only_when_every_output_exists(tmp_path):
    audit = STAGES[0]
    assert not rp.is_done(audit, tmp_path)
    touch(tmp_path, audit)
    assert rp.is_done(audit, tmp_path)
    (tmp_path / audit.outputs[0]).unlink()
    assert not rp.is_done(audit, tmp_path)


def test_cleaning_removes_only_declared_outputs(tmp_path):
    touch_all(tmp_path)
    decoy = tmp_path / "reports" / "notes.txt"
    decoy.write_text("keep me")
    for path in rp.outputs_of(STAGES, tmp_path):
        path.unlink()
    assert decoy.exists()
    assert not any(rp.is_done(s, tmp_path) for s in STAGES)


def test_stages_with_existing_outputs_are_skipped(tmp_path):
    touch_all(tmp_path)
    calls = []
    code = rp.run(STAGES, [], tmp_path, runner=lambda s, n: calls.append(s.name) or 0)
    assert code == 0 and calls == []


def test_missing_stages_run_in_order_and_the_run_stops_at_the_first_failure(tmp_path):
    calls = []

    def runner(stage, no_mlflow):
        calls.append(stage.name)
        if stage.name == "tuning":
            return 1
        touch(tmp_path, stage)
        return 0

    assert rp.run(STAGES, [], tmp_path, runner=runner) == 1
    assert calls == ["audit", "baselines", "tuning"]


def test_a_stage_whose_upstream_outputs_are_missing_is_blocked(tmp_path):
    calls = []
    code = rp.run(STAGES, ["threshold"], tmp_path, runner=lambda s, n: calls.append(s.name) or 0)
    assert code == 1 and calls == []


def test_a_successful_run_with_missing_outputs_is_reported_as_failed(tmp_path):
    assert rp.run(STAGES, ["audit"], tmp_path, runner=lambda s, n: 0) == 1


def test_no_mlflow_flag_is_added_only_to_stages_that_log():
    by_name = {s.name: s for s in STAGES}
    assert "--no-mlflow" in rp.command(by_name["baselines"], True)
    assert "--no-mlflow" not in rp.command(by_name["audit"], True)
    assert "--no-mlflow" not in rp.command(by_name["baselines"], False)
    assert rp.command(by_name["test"], False)[1:] == ["-m", "mlnd.evaluate", "test"]


def test_quick_mode_stops_before_the_downstream_stages():
    quick = rp.quick_stages("reports")
    assert [s.name for s in quick] == ["audit", "baselines", "tuning"]
    assert "--quick" in quick[1].args and "--quick" in quick[2].args


def test_from_scratch_deletes_outputs_and_reruns_every_stage(tmp_path):
    touch_all(tmp_path)
    calls = []

    def runner(stage, no_mlflow):
        calls.append(stage.name)
        touch(tmp_path, stage)
        return 0

    assert rp.run(STAGES, [], tmp_path, from_scratch=True, assume_yes=True, runner=runner) == 0
    assert calls == NAMES


def test_from_scratch_asks_before_deleting(tmp_path):
    touch_all(tmp_path)
    calls = []
    code = rp.run(
        STAGES, [], tmp_path, from_scratch=True, confirm=lambda prompt: "n",
        runner=lambda s, n: calls.append(s.name) or 0,
    )
    assert code == 1 and calls == []
    assert all(rp.is_done(s, tmp_path) for s in STAGES)