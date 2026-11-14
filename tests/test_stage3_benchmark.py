import json
import sys

import numpy as np
import pytest

from benchmarks import benchmark_stage3_cpu as benchmark
from benchmarks.benchmark_stage3_cpu import cashflow_summary, results_equal
from monte_carlo_accelerator.metrics import Measurement
from monte_carlo_accelerator.models import ActuarialProjectionResult


def _result(premium=(10.0, 20.0), claim=(2.0, 4.0), expense=(1.0, 1.0)):
    return ActuarialProjectionResult(
        np.asarray(premium, dtype=float),
        np.asarray(claim, dtype=float),
        np.asarray(expense, dtype=float),
    )


def test_cashflow_summary_reports_component_and_net_statistics():
    summary = cashflow_summary(_result())

    assert summary == {
        "mean_premium_pv": 15.0,
        "mean_claim_pv": 3.0,
        "mean_expense_pv": 1.0,
        "mean_net_cashflow_pv": 11.0,
        "median_net_cashflow_pv": 11.0,
        "p95_net_cashflow_pv": 14.6,
    }


def test_results_equal_checks_every_cashflow_component():
    baseline = _result()

    assert results_equal(baseline, _result())
    assert not results_equal(baseline, _result(premium=(10.0, 21.0)))
    assert not results_equal(baseline, _result(claim=(2.0, 5.0)))
    assert not results_equal(baseline, _result(expense=(1.0, 2.0)))


@pytest.mark.parametrize("difference", [1e-10, 1.0])
@pytest.mark.parametrize("component", ["premium", "claim", "expense"])
def test_multicore_checks_single_cpu_components_untimed(monkeypatch, tmp_path, difference, component):
    timing = False
    references = []
    values = {"premium": [10.0, 20.0], "claim": [2.0, 4.0], "expense": [1.0, 1.0]}
    values[component][0] += difference
    values[component][1] -= difference

    def measure(function, *args, **kwargs):
        nonlocal timing
        timing = True
        try:
            result = function(*args, **kwargs)
        finally:
            timing = False
        return Measurement(result, 1.0, 1.0)

    def single_cpu(*args):
        assert not timing
        references.append(args)
        return _result()

    monkeypatch.setattr(benchmark, "measure", measure)
    monkeypatch.setattr(benchmark, "project_term_life", single_cpu)
    monkeypatch.setattr(benchmark, "project_term_life_multicore", lambda *args, **kwargs: _result(**values))
    monkeypatch.setattr(benchmark, "collect_environment", lambda: {})
    monkeypatch.setattr(sys, "argv", [
        "benchmark", "--policies", "2", "--scenarios", "2", "--workers", "2",
        "--warmups", "0", "--repetitions", "2",
        "--output", str(tmp_path / "summary.csv"),
        "--runs-output", str(tmp_path / "runs.csv"),
        "--json-output", str(tmp_path / "report.json"),
    ])
    if difference == 1.0:
        with pytest.raises(RuntimeError, match="single-CPU numerical acceptance"):
            benchmark.main()
        assert not (tmp_path / "summary.csv").exists()
    else:
        benchmark.main()
        parity = json.loads((tmp_path / "report.json").read_text())["parity"]
        assert parity["consistent"]
        assert parity["components"][f"{component}_pv"]["max_absolute_difference"] > 0
        assert len(parity["warm_runs"]) == 2
        for report in [parity["cold_run"], *parity["warm_runs"]]:
            assert report["rtol"] == 1e-12
            assert report["atol"] == 1e-6
            assert report["components"][f"{component}_pv"]["max_absolute_difference"] > 0
    assert len(references) == 1


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("bad_cold", [False, True])
def test_cpu_parity_includes_cold_and_preserves_environment(monkeypatch, tmp_path, workers, bad_cold):
    calls = 0
    timing = False
    original_parity = benchmark.component_parity

    def projection(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _result(claim=(20.0, 4.0)) if bad_cold and calls == 1 else _result()

    def measure(function, *args, **kwargs):
        nonlocal timing
        timing = True
        try:
            result = function(*args, **kwargs)
        finally:
            timing = False
        return Measurement(result, 1.0, 1.0)

    def parity(*args):
        assert not timing
        return original_parity(*args)

    environment = {"git": {"commit": "test-revision", "dirty": True}}
    monkeypatch.setattr(benchmark, "measure", measure)
    monkeypatch.setattr(benchmark, "component_parity", parity)
    monkeypatch.setattr(benchmark, "project_term_life", projection)
    monkeypatch.setattr(benchmark, "project_term_life_multicore", projection)
    monkeypatch.setattr(benchmark, "collect_environment", lambda: environment)
    monkeypatch.setattr(sys, "argv", [
        "benchmark", "--policies", "2", "--scenarios", "2", "--workers", str(workers),
        "--warmups", "0", "--repetitions", "2",
        "--output", str(tmp_path / "summary.csv"),
        "--runs-output", str(tmp_path / "runs.csv"),
        "--json-output", str(tmp_path / "report.json"),
    ])
    if bad_cold:
        with pytest.raises(RuntimeError, match="numerical acceptance"):
            benchmark.main()
        assert not (tmp_path / "summary.csv").exists()
    else:
        benchmark.main()
        document = json.loads((tmp_path / "report.json").read_text())
        assert document["parity"]["consistent"]
        assert len(document["parity"]["components"]) == 4
        assert document["summary"]["count"] == 2
        assert document["environment"] == environment
