import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from benchmarks import benchmark_stage5 as stage5


def flag(command, name):
    return command[command.index(name) + 1]


def test_modes_share_explicit_workload_and_have_unique_outputs(tmp_path):
    args = stage5.parse_args([
        "--output-dir", str(tmp_path), "--presets", "small", "medium",
        "--workers", "3", "--seed", "91", "--portfolio-seed", "42",
        "--years", "7", "--policy-chunk", "123", "--scenario-chunk", "19",
        "--repetitions", "2", "--warmups", "0",
    ])
    runs = stage5.build_manifest(args)["runs"]
    assert len(runs) == 8
    paths = []
    for preset in args.presets:
        group = [run for run in runs if run["workload"]["preset"] == preset]
        assert [run["mode"] for run in group] == list(stage5.MODES)
        for run in group:
            command = run["command"]
            assert command[:2] == [sys.executable, "-m"]
            assert run["workload"] == group[0]["workload"]
            for name, value in run["workload"].items():
                assert flag(command, f"--{name.replace('_', '-')}") == str(value)
            assert flag(command, "--policies") == str(stage5.PRESETS[preset][0])
            assert flag(command, "--scenarios") == str(stage5.PRESETS[preset][1])
            assert ("--adaptive" in command) == (run["mode"] == "gpu_adaptive")
            if run["mode"].startswith("cpu_"):
                assert command[2] == "benchmarks.benchmark_stage3_cpu"
                assert flag(command, "--workers") == ("1" if run["mode"] == "cpu_single" else "3")
                assert "--runs-output" in command
            else:
                assert command[2] == "benchmarks.benchmark_stage3_gpu"
                assert "--workers" not in command
                assert "--runs-output" not in command
            for name, output in run["outputs"].items():
                assert flag(command, f"--{name.replace('_', '-')}") == output
                assert Path(output).is_relative_to(tmp_path)
                paths.append(output)
    assert len(paths) == len(set(paths)) == 20


def test_workload_overrides_apply_to_every_mode(tmp_path):
    args = stage5.parse_args(["--policies", "9", "--scenarios", "11", "--output-dir", str(tmp_path)])
    for run in stage5.build_manifest(args)["runs"]:
        assert flag(run["command"], "--policies") == "9"
        assert flag(run["command"], "--scenarios") == "11"


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_dry_run_never_starts_subprocess(tmp_path, monkeypatch, capsys, extra):
    runner = Mock(side_effect=AssertionError("must not execute"))
    monkeypatch.setattr(stage5.subprocess, "run", runner)
    assert stage5.main(["--output-dir", str(tmp_path), *extra]) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest == json.loads(capsys.readouterr().out)
    assert manifest["status"] == "dry_run"
    assert all(run["status"] == "planned" for run in manifest["runs"])
    for run in manifest["runs"]:
        assert flag(run["command"], "--policy-chunk") == "4096"
        assert flag(run["command"], "--scenario-chunk") == "128"
    assert list(tmp_path.iterdir()) == [tmp_path / "manifest.json"]
    runner.assert_not_called()


def test_execution_records_progress_and_success(tmp_path, monkeypatch):
    observed = []

    def run_command(command, *, cwd, check):
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["status"] == "running"
        assert manifest["runs"][len(observed)]["status"] == "running"
        assert cwd == stage5.REPO_ROOT
        assert check is True
        observed.append(command)

    monkeypatch.setattr(stage5.subprocess, "run", run_command)
    assert stage5.main(["--execute", "--output-dir", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert observed == [run["command"] for run in manifest["runs"]]
    assert manifest["status"] == "succeeded"
    assert all(run["status"] == "succeeded" and run["returncode"] == 0 for run in manifest["runs"])


@pytest.mark.parametrize("error, code", [
    (subprocess.CalledProcessError(7, ["runner"]), 7),
    (FileNotFoundError("runner missing"), 1),
    (subprocess.CalledProcessError(-15, ["runner"]), 143),
    (KeyboardInterrupt(), 130),
])
def test_failure_stops_execution_and_preserves_manifest(tmp_path, monkeypatch, error, code):
    runner = Mock(side_effect=[None, error])
    monkeypatch.setattr(stage5.subprocess, "run", runner)
    assert stage5.main(["--execute", "--output-dir", str(tmp_path)]) == code
    assert runner.call_count == 2
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert [run["status"] for run in manifest["runs"]] == ["succeeded", "failed", "skipped", "skipped"]
    assert "error" in manifest["runs"][1]


@pytest.mark.parametrize("arguments", [
    ["--policies", "0"], ["--scenarios", "-1"], ["--years", "0"],
    ["--policy-chunk", "0"], ["--scenario-chunk", "-2"], ["--repetitions", "0"],
    ["--warmups", "-1"], ["--workers", "1"], ["--workers", "0"],
    ["--seed", "-1"], ["--portfolio-seed", "-1"], ["--years", "abc"],
    ["--gpu-memory-fraction", "nan"], ["--gpu-memory-fraction", "0"],
    ["--gpu-memory-fraction", "1.1"], ["--presets", "unknown"],
    ["--presets", "small", "small"], ["--execute", "--dry-run"],
])
def test_invalid_arguments_fail_before_execution(tmp_path, monkeypatch, arguments):
    runner = Mock()
    monkeypatch.setattr(stage5.subprocess, "run", runner)
    with pytest.raises(SystemExit) as error:
        stage5.main(["--output-dir", str(tmp_path / "unused"), *arguments])
    assert error.value.code == 2
    assert not (tmp_path / "unused").exists()
    runner.assert_not_called()


def test_output_directory_cannot_be_a_file(tmp_path):
    output = tmp_path / "file"
    output.write_text("existing")
    with pytest.raises(SystemExit):
        stage5.parse_args(["--output-dir", str(output)])


@pytest.mark.parametrize("execute", [[], ["--execute"]])
def test_existing_results_are_never_overwritten(tmp_path, monkeypatch, execute):
    output = tmp_path / "small" / "gpu_fixed" / "results.json"
    output.parent.mkdir(parents=True)
    output.write_text("existing results")
    runner = Mock()
    monkeypatch.setattr(stage5.subprocess, "run", runner)
    with pytest.raises(SystemExit) as error:
        stage5.main(["--output-dir", str(tmp_path), *execute])
    assert error.value.code == 2
    assert output.read_text() == "existing results"
    assert not (tmp_path / "manifest.json").exists()
    runner.assert_not_called()
