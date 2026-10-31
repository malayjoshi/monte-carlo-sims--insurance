import importlib.util

import numpy as np
import pytest

from monte_carlo_accelerator.actuarial_engine import project_term_life
from monte_carlo_accelerator.actuarial_gpu_engine import (
    GpuTermLifePolicies,
    _project_term_life_gpu_backend,
    prepare_gpu_term_life_policies,
    project_term_life_gpu,
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
