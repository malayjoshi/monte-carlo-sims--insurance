"""Optional CuPy implementation of the Stage 3 term-life projection."""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .actuarial_engine import _LAPSE_SEED_OFFSET
from .gpu_engine import _load_cupy, _mortality_rate_backend, _uniform_backend
from .models import ActuarialAssumptions, ActuarialProjectionResult, TermLifePolicies


@dataclass(frozen=True)
class GpuTermLifePolicies:
    policy_id: Any
    age_at_entry: Any
    sum_assured: Any
    annual_premium: Any
    term_years: Any

    def __len__(self) -> int:
        return len(self.policy_id)


@dataclass(frozen=True)
class AdaptiveGpuReport:
    """Memory decision and retry history for one adaptive GPU projection."""

    requested_policy_chunk: int
    requested_scenario_chunk: int
    effective_policy_chunk: int
    effective_scenario_chunk: int
    free_device_bytes: int
    memory_budget_bytes: int
    estimated_bytes_per_cell: int
    attempts: int


def choose_adaptive_gpu_chunks(
    policies: GpuTermLifePolicies,
    scenarios: int,
    assumptions: ActuarialAssumptions,
    cupy,
    *,
    memory_fraction: float = 0.70,
    estimated_bytes_per_cell: int = 256,
) -> tuple[ActuarialAssumptions, int, int]:
    """Fit requested chunks into a conservative share of free device memory."""
    if scenarios <= 0:
        raise ValueError("scenarios must be positive")
    if not 0 < memory_fraction <= 1:
        raise ValueError("memory_fraction must be between 0 and 1")
    if estimated_bytes_per_cell <= 0:
        raise ValueError("estimated_bytes_per_cell must be positive")

    free_bytes, _ = cupy.cuda.runtime.memGetInfo()
    budget = int(free_bytes * memory_fraction)
    output_bytes = scenarios * 3 * np.dtype(np.float64).itemsize
    cell_budget = max(1, (budget - output_bytes) // estimated_bytes_per_cell)

    policy_chunk = min(assumptions.policy_chunk_size, len(policies))
    scenario_chunk = min(assumptions.scenario_chunk_size, scenarios)
    if policy_chunk * scenario_chunk > cell_budget:
        scenario_chunk = max(1, min(scenario_chunk, cell_budget // max(policy_chunk, 1)))
    if policy_chunk * scenario_chunk > cell_budget:
        policy_chunk = max(1, min(policy_chunk, cell_budget // scenario_chunk))

    return (
        replace(
            assumptions,
            policy_chunk_size=policy_chunk,
            scenario_chunk_size=scenario_chunk,
        ),
        int(free_bytes),
        budget,
    )


def _is_gpu_out_of_memory(error: BaseException, cupy) -> bool:
    if isinstance(error, MemoryError):
        return True
    memory_module = getattr(getattr(cupy, "cuda", None), "memory", None)
    oom_type = getattr(memory_module, "OutOfMemoryError", None)
    return oom_type is not None and isinstance(error, oom_type)


def _release_unused_gpu_memory(cupy) -> None:
    pool = getattr(cupy, "get_default_memory_pool", lambda: None)()
    release = getattr(pool, "free_all_blocks", None)
    if release is not None:
        release()


def prepare_gpu_term_life_policies(
    policies: TermLifePolicies, cupy=None
) -> GpuTermLifePolicies:
    """Transfer immutable Stage 3 policy inputs to one array backend."""
    xp = cupy or _load_cupy()
    return GpuTermLifePolicies(
        policy_id=xp.asarray(policies.policy_id, dtype=xp.uint64),
        age_at_entry=xp.asarray(policies.age_at_entry, dtype=xp.int16),
        sum_assured=xp.asarray(policies.sum_assured, dtype=xp.float64),
        annual_premium=xp.asarray(policies.annual_premium, dtype=xp.float64),
        term_years=xp.asarray(policies.term_years, dtype=xp.int16),
    )


def _project_term_life_gpu_backend(
    policies: GpuTermLifePolicies,
    scenarios: int,
    assumptions: ActuarialAssumptions,
    xp,
):
    """Run the Stage 3 algorithm using NumPy-compatible GPU operations."""
    if scenarios <= 0:
        raise ValueError("scenarios must be positive")

    premium_pv = xp.zeros(scenarios, dtype=xp.float64)
    claim_pv = xp.zeros(scenarios, dtype=xp.float64)
    expense_pv = xp.zeros(scenarios, dtype=xp.float64)
    lapse_seed = assumptions.seed ^ _LAPSE_SEED_OFFSET

    for p0 in range(0, len(policies), assumptions.policy_chunk_size):
        p1 = min(p0 + assumptions.policy_chunk_size, len(policies))
        ids = policies.policy_id[p0:p1, None]
        ages = policies.age_at_entry[p0:p1, None]
        benefits = policies.sum_assured[p0:p1, None]
        premiums = policies.annual_premium[p0:p1, None]
        terms = policies.term_years[p0:p1, None]

        for s0 in range(0, scenarios, assumptions.scenario_chunk_size):
            s1 = min(s0 + assumptions.scenario_chunk_size, scenarios)
            scenario_ids = xp.arange(s0, s1, dtype=xp.uint64)[None, :]
            in_force = xp.ones((p1 - p0, s1 - s0), dtype=bool)
            block_premium = xp.zeros(s1 - s0, dtype=xp.float64)
            block_claim = xp.zeros(s1 - s0, dtype=xp.float64)
            block_expense = xp.zeros(s1 - s0, dtype=xp.float64)

            for year in range(assumptions.projection_years):
                active = in_force & (year < terms)
                beginning_discount = (1.0 + assumptions.discount_rate) ** year
                end_discount = beginning_discount * (1.0 + assumptions.discount_rate)
                expense = (
                    assumptions.expense_per_policy
                    * (1.0 + assumptions.expense_inflation) ** year
                )
                benefit = benefits * (1.0 + assumptions.claim_inflation) ** year

                block_premium += xp.sum(active * premiums, axis=0) / beginning_discount
                block_expense += xp.sum(active, axis=0) * expense / beginning_discount

                mortality = xp.minimum(
                    _mortality_rate_backend(ages + year, xp) * assumptions.mortality_scale,
                    1.0,
                )
                deaths = active & (
                    _uniform_backend(ids, scenario_ids, year, assumptions.seed, xp)
                    < mortality
                )
                block_claim += xp.sum(deaths * benefit, axis=0) / end_discount

                survivors = active & ~deaths
                lapses = survivors & (
                    _uniform_backend(ids, scenario_ids, year, lapse_seed, xp)
                    < assumptions.annual_lapse_rate
                )
                in_force = survivors & ~lapses & (year + 1 < terms)

            premium_pv[s0:s1] += block_premium
            claim_pv[s0:s1] += block_claim
            expense_pv[s0:s1] += block_expense

    return premium_pv, claim_pv, expense_pv


def _as_numpy(value, xp) -> np.ndarray:
    transferred = xp.asnumpy(value) if hasattr(xp, "asnumpy") else value
    return np.asarray(transferred, dtype=np.float64)


def project_term_life_gpu_prepared(
    policies: GpuTermLifePolicies,
    scenarios: int,
    assumptions: ActuarialAssumptions = ActuarialAssumptions(),
    cupy=None,
) -> ActuarialProjectionResult:
    """Project GPU-resident policies and copy scenario cash flows to the host."""
    xp = cupy or _load_cupy()
    premium, claim, expense = _project_term_life_gpu_backend(
        policies, scenarios, assumptions, xp
    )
    return ActuarialProjectionResult(
        _as_numpy(premium, xp),
        _as_numpy(claim, xp),
        _as_numpy(expense, xp),
    )


def project_term_life_gpu_adaptive_prepared(
    policies: GpuTermLifePolicies,
    scenarios: int,
    assumptions: ActuarialAssumptions = ActuarialAssumptions(),
    cupy=None,
    *,
    memory_fraction: float = 0.70,
    estimated_bytes_per_cell: int = 256,
    max_retries: int = 8,
) -> tuple[ActuarialProjectionResult, AdaptiveGpuReport]:
    """Project with memory-aware chunks and retry smaller blocks after GPU OOM."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    xp = cupy or _load_cupy()
    effective, free_bytes, budget = choose_adaptive_gpu_chunks(
        policies,
        scenarios,
        assumptions,
        xp,
        memory_fraction=memory_fraction,
        estimated_bytes_per_cell=estimated_bytes_per_cell,
    )

    attempts = 0
    while True:
        attempts += 1
        try:
            result = project_term_life_gpu_prepared(
                policies, scenarios, effective, xp
            )
            return result, AdaptiveGpuReport(
                requested_policy_chunk=assumptions.policy_chunk_size,
                requested_scenario_chunk=assumptions.scenario_chunk_size,
                effective_policy_chunk=effective.policy_chunk_size,
                effective_scenario_chunk=effective.scenario_chunk_size,
                free_device_bytes=free_bytes,
                memory_budget_bytes=budget,
                estimated_bytes_per_cell=estimated_bytes_per_cell,
                attempts=attempts,
            )
        except Exception as error:
            if not _is_gpu_out_of_memory(error, xp) or attempts > max_retries:
                raise
            if effective.scenario_chunk_size > 1:
                effective = replace(
                    effective,
                    scenario_chunk_size=max(1, effective.scenario_chunk_size // 2),
                )
            elif effective.policy_chunk_size > 1:
                effective = replace(
                    effective,
                    policy_chunk_size=max(1, effective.policy_chunk_size // 2),
                )
            else:
                raise
            _release_unused_gpu_memory(xp)


def project_term_life_gpu(
    policies: TermLifePolicies,
    scenarios: int,
    assumptions: ActuarialAssumptions = ActuarialAssumptions(),
) -> ActuarialProjectionResult:
    """Project Stage 3 cash flows with CuPy, including policy transfer."""
    cupy = _load_cupy()
    prepared = prepare_gpu_term_life_policies(policies, cupy)
    return project_term_life_gpu_prepared(prepared, scenarios, assumptions, cupy)


def project_term_life_gpu_adaptive(
    policies: TermLifePolicies,
    scenarios: int,
    assumptions: ActuarialAssumptions = ActuarialAssumptions(),
    *,
    memory_fraction: float = 0.70,
    estimated_bytes_per_cell: int = 256,
    max_retries: int = 8,
) -> tuple[ActuarialProjectionResult, AdaptiveGpuReport]:
    """Transfer policies, then run the memory-aware Stage 4 GPU projection."""
    cupy = _load_cupy()
    prepared = prepare_gpu_term_life_policies(policies, cupy)
    return project_term_life_gpu_adaptive_prepared(
        prepared,
        scenarios,
        assumptions,
        cupy,
        memory_fraction=memory_fraction,
        estimated_bytes_per_cell=estimated_bytes_per_cell,
        max_retries=max_retries,
    )
