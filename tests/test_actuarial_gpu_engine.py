import importlib.util

import numpy as np
import pytest

import monte_carlo_accelerator.actuarial_gpu_engine as gpu_module

from monte_carlo_accelerator.actuarial_engine import project_term_life
from monte_carlo_accelerator.actuarial_gpu_engine import (
    GpuTermLifePolicies,
    _project_term_life_gpu_backend,
    choose_adaptive_gpu_chunks,
    prepare_gpu_term_life_policies,
    project_term_life_gpu,
    project_term_life_gpu_adaptive,
    project_term_life_gpu_adaptive_prepared,
    project_term_life_gpu_prepared,
)
from monte_carlo_accelerator.gpu_engine import GPUUnavailableError
from monte_carlo_accelerator.models import ActuarialAssumptions
from monte_carlo_accelerator.policy_generator import generate_term_life_portfolio


def test_stage3_gpu_algorithm_matches_cpu_with_numpy_backend():
    policies = generate_term_life_portfolio(37, seed=9)
    assumptions = ActuarialAssumptions(
        projection_years=12,
        discount_rate=0.04,
        mortality_scale=2.5,
        annual_lapse_rate=0.13,
        expense_per_policy=31,
        expense_inflation=0.025,
        claim_inflation=0.015,
        seed=81,
        policy_chunk_size=11,
        scenario_chunk_size=4,
    )
    prepared = GpuTermLifePolicies(
        policies.policy_id,
        policies.age_at_entry,
        policies.sum_assured,
        policies.annual_premium,
        policies.term_years,
    )

    premium, claim, expense = _project_term_life_gpu_backend(prepared, 13, assumptions, np)
    cpu = project_term_life(policies, 13, assumptions)

    np.testing.assert_allclose(premium, cpu.premium_pv, rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(claim, cpu.claim_pv, rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(expense, cpu.expense_pv, rtol=1e-12, atol=1e-6)


def test_prepared_stage3_gpu_api_matches_cpu_with_numpy_backend():
    policies = generate_term_life_portfolio(15, seed=3)
    assumptions = ActuarialAssumptions(
        projection_years=8,
        seed=13,
        policy_chunk_size=6,
        scenario_chunk_size=3,
    )
    prepared = prepare_gpu_term_life_policies(policies, np)

    actual = project_term_life_gpu_prepared(prepared, 7, assumptions, np)
    expected = project_term_life(policies, 7, assumptions)

    np.testing.assert_allclose(actual.premium_pv, expected.premium_pv, rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(actual.claim_pv, expected.claim_pv, rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(actual.expense_pv, expected.expense_pv, rtol=1e-12, atol=1e-6)


def test_stage3_gpu_rejects_invalid_scenario_count_with_numpy_backend():
    policies = generate_term_life_portfolio(2)
    prepared = prepare_gpu_term_life_policies(policies, np)

    with pytest.raises(ValueError, match="scenarios"):
        project_term_life_gpu_prepared(prepared, 0, cupy=np)


def test_stage3_gpu_has_clear_error_without_cupy():
    if importlib.util.find_spec("cupy") is not None:
        pytest.skip("CuPy is installed in this environment")
    with pytest.raises(GPUUnavailableError, match="CuPy is not installed"):
        project_term_life_gpu(generate_term_life_portfolio(2), 2)


class _RuntimeMemory:
    free_bytes = 1_000_000

    @classmethod
    def memGetInfo(cls):
        return cls.free_bytes, 2_000_000


class _NumpyWithMemory:
    cuda = type("Cuda", (), {"runtime": _RuntimeMemory})

    def __getattr__(self, name):
        return getattr(np, name)


def test_adaptive_chunks_preserve_requested_sizes_when_memory_allows():
    policies = generate_term_life_portfolio(100)
    prepared = prepare_gpu_term_life_policies(policies, np)
    assumptions = ActuarialAssumptions(policy_chunk_size=40, scenario_chunk_size=20)
    backend = _NumpyWithMemory()

    effective, free_bytes, budget = choose_adaptive_gpu_chunks(
        prepared, 50, assumptions, backend
    )

    assert (effective.policy_chunk_size, effective.scenario_chunk_size) == (40, 20)
    assert free_bytes == 1_000_000
    assert budget == 700_000


def test_adaptive_chunks_shrink_scenarios_under_low_memory():
    policies = generate_term_life_portfolio(100)
    prepared = prepare_gpu_term_life_policies(policies, np)
    assumptions = ActuarialAssumptions(policy_chunk_size=100, scenario_chunk_size=50)
    backend = _NumpyWithMemory()
    _RuntimeMemory.free_bytes = 20_000
    try:
        effective, _, _ = choose_adaptive_gpu_chunks(
            prepared, 50, assumptions, backend, memory_fraction=0.5
        )
    finally:
        _RuntimeMemory.free_bytes = 1_000_000

    assert 1 <= effective.scenario_chunk_size < 50
    assert effective.policy_chunk_size * effective.scenario_chunk_size <= 34


def test_adaptive_projection_matches_cpu_with_numpy_backend():
    policies = generate_term_life_portfolio(19, seed=4)
    prepared = prepare_gpu_term_life_policies(policies, np)
    assumptions = ActuarialAssumptions(
        projection_years=6, policy_chunk_size=10, scenario_chunk_size=5
    )
    backend = _NumpyWithMemory()

    actual, report = project_term_life_gpu_adaptive_prepared(
        prepared, 7, assumptions, backend
    )
    expected = project_term_life(policies, 7, assumptions)

    np.testing.assert_allclose(actual.net_cashflow_pv, expected.net_cashflow_pv)
    assert report.attempts == 1


def test_adaptive_projection_retries_smaller_scenario_chunk(monkeypatch):
    policies = generate_term_life_portfolio(5, seed=4)
    prepared = prepare_gpu_term_life_policies(policies, np)
    assumptions = ActuarialAssumptions(
        projection_years=2, policy_chunk_size=5, scenario_chunk_size=4
    )
    backend = _NumpyWithMemory()
    calls = []
    expected = project_term_life(policies, 4, assumptions)

    def sometimes_out_of_memory(_policies, _scenarios, effective, _backend):
        calls.append(effective.scenario_chunk_size)
        if len(calls) == 1:
            raise MemoryError("simulated device allocation failure")
        return expected

    monkeypatch.setattr(
        gpu_module, "project_term_life_gpu_prepared", sometimes_out_of_memory
    )

    actual, report = project_term_life_gpu_adaptive_prepared(
        prepared, 4, assumptions, backend, max_retries=1
    )

    assert actual is expected
    assert calls == [4, 2]
    assert report.effective_scenario_chunk == 2
    assert report.attempts == 2


def test_adaptive_host_api_prepares_policies_and_forwards_options(monkeypatch):
    policies = generate_term_life_portfolio(3)
    expected = project_term_life(policies, 2)
    captured = {}

    monkeypatch.setattr(gpu_module, "_load_cupy", lambda: np)

    def adaptive(prepared, scenarios, assumptions, cupy, **options):
        captured.update(
            prepared_count=len(prepared),
            scenarios=scenarios,
            cupy=cupy,
            options=options,
        )
        return expected, "report"

    monkeypatch.setattr(
        gpu_module, "project_term_life_gpu_adaptive_prepared", adaptive
    )

    result, report = project_term_life_gpu_adaptive(
        policies, 2, memory_fraction=0.6, max_retries=3
    )

    assert result is expected
    assert report == "report"
    assert captured == {
        "prepared_count": 3,
        "scenarios": 2,
        "cupy": np,
        "options": {
            "memory_fraction": 0.6,
            "estimated_bytes_per_cell": 256,
            "max_retries": 3,
        },
    }
