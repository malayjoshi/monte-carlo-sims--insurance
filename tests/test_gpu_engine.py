import importlib.util

import numpy as np
import pytest

from src.cpu_engine import _uniform, project_cpu
from src.gpu_engine import (
    GPUUnavailableError,
    GpuPolicies,
    _project_gpu_backend,
    _uniform_backend,
    project_gpu,
)
from src.models import ProjectionConfig
from src.policy_generator import generate_policies


RNG_VECTORS = (
    (0, 0, 0, 2026, 0.1825856527684354),
    (1, 0, 0, 2026, 0.9043931387826301),
    (0, 1, 0, 2026, 0.5973645718550313),
    (42, 17, 9, 2026, 0.7746357995085181),
    (9999, 999, 29, 2026, 0.3542436783372036),
    (4_294_967_303, 1_048_579, 14, 123456, 0.09124866257569841),
)


@pytest.mark.parametrize("policy,scenario,year,seed,expected", RNG_VECTORS)
def test_frozen_rng_vectors(policy, scenario, year, seed, expected):
    assert float(_uniform(policy, scenario, year, seed)) == expected
    assert float(_uniform_backend(policy, scenario, year, seed, np)) == expected


def test_gpu_algorithm_matches_cpu_with_numpy_backend():
    policies = generate_policies(29, seed=7)
    config = ProjectionConfig(years=15, seed=123, policy_chunk_size=8, scenario_chunk_size=3)
    prepared = GpuPolicies(policies.policy_id, policies.age, policies.benefit)
    gpu_algorithm = _project_gpu_backend(prepared, 9, config, np)
    cpu_result = project_cpu(policies, 9, config).scenario_pv
    np.testing.assert_allclose(gpu_algorithm, cpu_result, rtol=1e-12, atol=1e-6)


def test_project_gpu_has_clear_error_without_cupy():
    if importlib.util.find_spec("cupy") is not None:
        pytest.skip("CuPy is installed in this environment")
    with pytest.raises(GPUUnavailableError, match="CuPy is not installed"):
        project_gpu(generate_policies(2), 2)
