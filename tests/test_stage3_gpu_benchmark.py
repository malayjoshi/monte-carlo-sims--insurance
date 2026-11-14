import csv
import json
import sys
from functools import partial

import numpy as np
import pytest

from benchmarks import benchmark_stage3_gpu as benchmark
from monte_carlo_accelerator.actuarial_gpu_engine import AdaptiveGpuReport
from monte_carlo_accelerator.metrics import Measurement
from benchmarks.benchmark_stage3_gpu import (
    append_benchmark_csv,
    component_parity,
    gpu_memory_snapshot,
)
from monte_carlo_accelerator.models import ActuarialProjectionResult


def _result(premium, claim, expense):
    return ActuarialProjectionResult(
        np.asarray(premium, dtype=float),
        np.asarray(claim, dtype=float),
        np.asarray(expense, dtype=float),
    )


def test_component_parity_reports_each_cashflow_and_net():
    cpu = _result([10, 20], [3, 5], [1, 2])
    gpu = _result([10, 20 + 1e-10], [3, 5], [1, 2])

    parity = component_parity(cpu, gpu)

    assert parity["consistent"]
    assert set(parity["components"]) == {
        "premium_pv",
        "claim_pv",
        "expense_pv",
        "net_cashflow_pv",
    }
    assert parity["components"]["premium_pv"]["max_absolute_difference"] > 0


def test_component_parity_rejects_material_difference():
    cpu = _result([10], [3], [1])
    gpu = _result([10], [30], [1])

    parity = component_parity(cpu, gpu)

    assert not parity["consistent"]
    assert not parity["components"]["claim_pv"]["consistent"]
    assert not parity["components"]["net_cashflow_pv"]["consistent"]


def test_gpu_memory_snapshot_reports_runtime_and_pool_values():
    class Runtime:
        @staticmethod
        def memGetInfo():
            return 600, 1_000

    class Pool:
        @staticmethod
        def used_bytes():
            return 200

        @staticmethod
        def total_bytes():
            return 300

    class FakeCupy:
        class cuda:
            runtime = Runtime()

        @staticmethod
        def get_default_memory_pool():
            return Pool()

    assert gpu_memory_snapshot(FakeCupy()) == {
        "device_free_bytes": 600,
        "device_total_bytes": 1_000,
        "pool_used_bytes": 200,
        "pool_total_bytes": 300,
    }


def test_csv_expands_legacy_header_and_preserves_rows(tmp_path):
    path = tmp_path / "runs.csv"
    path.write_text("policy_chunk,scenario_chunk,legacy_metric\n100,20,42\n")
    row = {
        "policy_chunk": 200,
        "scenario_chunk": 40,
        "effective_policy_chunk": 100,
        "effective_scenario_chunk": 10,
    }

    append_benchmark_csv(path, row)
    append_benchmark_csv(path, row)

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == [
            "policy_chunk", "scenario_chunk", "legacy_metric",
            "effective_policy_chunk", "effective_scenario_chunk",
        ]
    assert len(rows) == 3
    assert rows[0] == {
        "policy_chunk": "100", "scenario_chunk": "20", "legacy_metric": "42",
        "effective_policy_chunk": "", "effective_scenario_chunk": "",
    }
    assert rows[1] == rows[2] == {**{key: str(value) for key, value in row.items()}, "legacy_metric": ""}


def test_csv_new_and_empty_files_support_repeated_appends(tmp_path):
    for name in ("new.csv", "empty.csv"):
        path = tmp_path / name
        if name == "empty.csv":
            path.touch()
        row = {"policy_chunk": 100, "effective_policy_chunk": 50}
        append_benchmark_csv(path, row)
        append_benchmark_csv(path, row)
        with path.open(newline="") as handle:
            assert list(csv.DictReader(handle)) == [
                {"policy_chunk": "100", "effective_policy_chunk": "50"},
                {"policy_chunk": "100", "effective_policy_chunk": "50"},
            ]


@pytest.fixture
def gpu_benchmark(monkeypatch, tmp_path):
    state = {"calls": [], "bad_run": None, "timing": False}
    baseline = _result([10, 20], [3, 5], [1, 2])

    def measure(function, *args, **kwargs):
        state["timing"] = True
        try:
            value = function(*args, **kwargs)
        finally:
            state["timing"] = False
        return Measurement(value, float(len(state["calls"]) + 1), 1.0)

    def projection(kind, adaptive, *args, **kwargs):
        state["calls"].append((kind, args, kwargs))
        number = len(state["calls"])
        result = _result([10, 20], [30, 5], [1, 2]) if number == state["bad_run"] else baseline
        if adaptive:
            return result, AdaptiveGpuReport(8, 4, number, 1, 1000, 700, 256, number)
        return result

    def parity(*args, **kwargs):
        assert not state["timing"]
        return component_parity(*args, **kwargs)

    def cpu_reference(*args):
        assert not state["timing"]
        return baseline

    monkeypatch.setattr(benchmark, "measure", measure)
    monkeypatch.setattr(benchmark, "component_parity", parity)
    monkeypatch.setattr(benchmark, "project_term_life", cpu_reference)
    monkeypatch.setattr(benchmark, "_load_cupy", lambda: object())
    monkeypatch.setattr(benchmark, "prepare_gpu_term_life_policies", lambda *args: object())
    monkeypatch.setattr(benchmark, "gpu_memory_snapshot", lambda *args: {"pool_used_bytes": 0})
    monkeypatch.setattr(benchmark, "device_metadata", lambda *args: {})
    monkeypatch.setattr(benchmark, "collect_environment", lambda: {})
    for name, kind, adaptive in (
        ("project_term_life_gpu_prepared", "fixed_prepared", False),
        ("project_term_life_gpu_adaptive_prepared", "adaptive_prepared", True),
        ("project_term_life_gpu", "fixed_host", False),
        ("project_term_life_gpu_adaptive", "adaptive_host", True),
    ):
        monkeypatch.setattr(benchmark, name, partial(projection, kind, adaptive))

    def run(adaptive):
        monkeypatch.setattr(sys, "argv", [
            "benchmark", "--policies", "2", "--scenarios", "2",
            "--policy-chunk", "8", "--scenario-chunk", "4",
            "--warmups", "1", "--repetitions", "2",
            "--output", str(tmp_path / "summary.csv"),
            "--json-output", str(tmp_path / "report.json"),
            *(["--adaptive"] if adaptive else []),
        ])
        benchmark.main()
        return json.loads((tmp_path / "report.json").read_text())

    return state, run


@pytest.mark.parametrize("adaptive", [False, True])
def test_gpu_repeats_full_api_and_archives_each_report(gpu_benchmark, adaptive):
    state, run = gpu_benchmark
    document = run(adaptive)
    prefix = "adaptive" if adaptive else "fixed"
    assert [call[0] for call in state["calls"]] == [f"{prefix}_prepared"] * 4 + [f"{prefix}_host"] * 2
    assert all(call[1][2].policy_chunk_size == 8 for call in state["calls"])
    runs = [document["cold_compute_run"], *document["warm_compute_runs"], *document["end_to_end_runs"]]
    assert all(item["parity"]["consistent"] for item in runs)
    assert document["end_to_end_run"] == document["end_to_end_runs"][0]
    assert document["end_to_end_summary"]["count"] == 2
    assert document["end_to_end_summary"]["median_seconds"] == 6.5
    assert "initialized CUDA context" in document["methodology"]["end_to_end"]
    if adaptive:
        assert [item["adaptive_gpu"]["attempts"] for item in runs] == [1, 3, 4, 5, 6]
        assert document["warmup_adaptive_reports"][0]["attempts"] == 2
        assert all(call[2]["memory_fraction"] == 0.7 for call in state["calls"])
    else:
        assert all(item["adaptive_gpu"] is None for item in runs)


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("bad_run", [1, 3, 4, 5, 6])
def test_gpu_rejects_any_measured_run_mismatch(gpu_benchmark, adaptive, bad_run, tmp_path):
    state, run = gpu_benchmark
    state["bad_run"] = bad_run
    with pytest.raises(RuntimeError, match="numerical acceptance"):
        run(adaptive)
    assert not (tmp_path / "summary.csv").exists()
    assert not (tmp_path / "report.json").exists()
