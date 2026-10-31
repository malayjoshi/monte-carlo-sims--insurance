import numpy as np

from benchmarks.benchmark_stage3_cpu import cashflow_summary, results_equal
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
