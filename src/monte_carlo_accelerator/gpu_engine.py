"""Optional CuPy implementation of the Monte Carlo projection."""

from contextlib import nullcontext
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from .models import Policies, ProjectionConfig, ProjectionResult

_MASK = 0xFFFFFFFFFFFFFFFF
_C1 = 0x9E3779B97F4A7C15
_C2 = 0xBF58476D1CE4E5B9
_C3 = 0x94D049BB133111EB


class GPUUnavailableError(RuntimeError):
    """Raised when the optional CuPy/CUDA runtime is unavailable."""


@dataclass(frozen=True)
class GpuPolicies:
    policy_id: Any
    age: Any
    benefit: Any

    def __len__(self) -> int:
        return len(self.policy_id)


def _load_cupy():
    try:
        cupy = import_module("cupy")
    except ImportError as error:
        raise GPUUnavailableError(
            "CuPy is not installed. Install the CUDA-matched optional dependency on an NVIDIA host."
        ) from error
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except Exception as error:
        raise GPUUnavailableError("CuPy is installed, but the CUDA runtime is unavailable.") from error
    if device_count < 1:
        raise GPUUnavailableError("No CUDA-capable NVIDIA GPU was detected.")
    return cupy


def gpu_available() -> bool:
    try:
        _load_cupy()
    except GPUUnavailableError:
        return False
    return True


def prepare_gpu_policies(policies: Policies, cupy=None) -> GpuPolicies:
    """Transfer immutable policy inputs once for compute-only benchmarks."""
    xp = cupy or _load_cupy()
    return GpuPolicies(
        policy_id=xp.asarray(policies.policy_id, dtype=xp.uint64),
        age=xp.asarray(policies.age, dtype=xp.int16),
        benefit=xp.asarray(policies.benefit, dtype=xp.float64),
    )


def _uniform_backend(policy_id, scenario_id, year: int, seed: int, xp):
    uint64 = xp.uint64
    mask, c1, c2, c3 = map(uint64, (_MASK, _C1, _C2, _C3))
    # NumPy warns for the intentional uint64 wraparound used by SplitMix64.
    # CuPy performs the same wraparound on-device but does not expose errstate.
    overflow_context = xp.errstate(over="ignore") if hasattr(xp, "errstate") else nullcontext()
    with overflow_context:
        x = (xp.asarray(policy_id, dtype=uint64) * c1) & mask
        x = x ^ ((xp.asarray(scenario_id, dtype=uint64) * c2) & mask)
        x = x ^ ((uint64(year + 1) * c3) & mask)
        x = x ^ uint64(seed)
        x = (x + c1) & mask
        x = ((x ^ (x >> uint64(30))) * c2) & mask
        x = ((x ^ (x >> uint64(27))) * c3) & mask
        x ^= x >> uint64(31)
    return (x >> uint64(11)).astype(xp.float64) * (1.0 / (1 << 53))


def _mortality_rate_backend(age, xp):
    return xp.minimum(0.00035 * xp.exp(0.085 * (xp.asarray(age) - 30)), 0.35)


def _project_gpu_backend(policies: GpuPolicies, scenarios: int, config: ProjectionConfig, xp):
    if scenarios <= 0 or config.years <= 0:
        raise ValueError("scenarios and years must be positive")
    totals = xp.zeros(scenarios, dtype=xp.float64)
    for p0 in range(0, len(policies), config.policy_chunk_size):
        p1 = min(p0 + config.policy_chunk_size, len(policies))
        ids = policies.policy_id[p0:p1, None]
        ages = policies.age[p0:p1, None]
        benefits = policies.benefit[p0:p1, None]
        for s0 in range(0, scenarios, config.scenario_chunk_size):
            s1 = min(s0 + config.scenario_chunk_size, scenarios)
            scenario_ids = xp.arange(s0, s1, dtype=xp.uint64)[None, :]
            alive = xp.ones((p1 - p0, s1 - s0), dtype=bool)
            block_pv = xp.zeros(s1 - s0, dtype=xp.float64)
            for year in range(config.years):
                deaths = alive & (
                    _uniform_backend(ids, scenario_ids, year, config.seed, xp)
                    < _mortality_rate_backend(ages + year, xp)
                )
                factor = (1.0 + config.claim_inflation) ** year / (1.0 + config.discount_rate) ** (year + 1)
                block_pv += xp.sum(deaths * benefits, axis=0) * factor
                alive &= ~deaths
            totals[s0:s1] += block_pv
    return totals


def project_gpu_prepared(
    policies: GpuPolicies, scenarios: int, config: ProjectionConfig = ProjectionConfig(), cupy=None
) -> ProjectionResult:
    """Project policies already resident on the GPU and return host scenario totals."""
    xp = cupy or _load_cupy()
    totals = _project_gpu_backend(policies, scenarios, config, xp)
    return ProjectionResult(np.asarray(xp.asnumpy(totals), dtype=np.float64))


def project_gpu(
    policies: Policies, scenarios: int, config: ProjectionConfig = ProjectionConfig()
) -> ProjectionResult:
    """CPU-compatible convenience API; includes host-to-device policy transfer."""
    cupy = _load_cupy()
    prepared = prepare_gpu_policies(policies, cupy)
    return project_gpu_prepared(prepared, scenarios, config, cupy)
