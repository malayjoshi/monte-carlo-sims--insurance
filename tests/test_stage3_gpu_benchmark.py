import csv

import numpy as np

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
