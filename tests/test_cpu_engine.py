import numpy as np
import pytest

from monte_carlo_accelerator.cpu_engine import project_cpu, project_reference
from monte_carlo_accelerator.models import ProjectionConfig
from monte_carlo_accelerator.policy_generator import generate_policies


def test_vectorized_matches_reference():
    policies = generate_policies(24, seed=5)
    config = ProjectionConfig(years=12, seed=99, policy_chunk_size=7, scenario_chunk_size=3)
    actual = project_cpu(policies, 8, config).scenario_pv
    expected = project_reference(policies, 8, config).scenario_pv
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-8)


def test_chunk_sizes_do_not_change_results():
    policies = generate_policies(31)
    a = project_cpu(policies, 11, ProjectionConfig(policy_chunk_size=31, scenario_chunk_size=11))
    b = project_cpu(policies, 11, ProjectionConfig(policy_chunk_size=5, scenario_chunk_size=4))
    np.testing.assert_allclose(a.scenario_pv, b.scenario_pv, rtol=1e-13, atol=1e-8)


def test_generator_is_deterministic():
    a, b = generate_policies(10, 123), generate_policies(10, 123)
    np.testing.assert_array_equal(a.age, b.age)
    np.testing.assert_array_equal(a.benefit, b.benefit)


def test_invalid_counts_are_rejected():
    with pytest.raises(ValueError):
        generate_policies(0)
