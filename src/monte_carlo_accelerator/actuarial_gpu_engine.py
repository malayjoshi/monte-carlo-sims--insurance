"""Optional CuPy implementation of the Stage 3 term-life projection."""

from dataclasses import dataclass
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


def project_term_life_gpu(
    policies: TermLifePolicies,
    scenarios: int,
    assumptions: ActuarialAssumptions = ActuarialAssumptions(),
) -> ActuarialProjectionResult:
    """Project Stage 3 cash flows with CuPy, including policy transfer."""
    cupy = _load_cupy()
    prepared = prepare_gpu_term_life_policies(policies, cupy)
    return project_term_life_gpu_prepared(prepared, scenarios, assumptions, cupy)
