import numpy as np

from benchmarks.benchmark_stage3_gpu import component_parity, gpu_memory_snapshot
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
