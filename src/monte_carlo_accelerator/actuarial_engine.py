"""Stage 3 cash-flow projection for level-premium term-life business."""

import numpy as np

from .cpu_engine import _mortality_rate, _uniform
from .models import ActuarialAssumptions, ActuarialProjectionResult, TermLifePolicies

_LAPSE_SEED_OFFSET = 0xD1B54A32D192ED03


def project_term_life(
    policies: TermLifePolicies,
    scenarios: int,
    assumptions: ActuarialAssumptions = ActuarialAssumptions(),
) -> ActuarialProjectionResult:
    """Project premium, death-claim, and expense present values by scenario.

    Premiums and maintenance expenses occur at the beginning of each policy
    year. Death claims occur at year end. Lapses occur after mortality and affect
    the next policy year. Policies leave the projection at death, lapse, or term.
    """
    if scenarios <= 0:
        raise ValueError("scenarios must be positive")

    premium_pv = np.zeros(scenarios, dtype=np.float64)
    claim_pv = np.zeros(scenarios, dtype=np.float64)
    expense_pv = np.zeros(scenarios, dtype=np.float64)
    years = assumptions.projection_years

    for p0 in range(0, len(policies), assumptions.policy_chunk_size):
        p1 = min(p0 + assumptions.policy_chunk_size, len(policies))
        ids = policies.policy_id[p0:p1, None]
        ages = policies.age_at_entry[p0:p1, None]
        benefits = policies.sum_assured[p0:p1, None]
        premiums = policies.annual_premium[p0:p1, None]
        terms = policies.term_years[p0:p1, None]

        for s0 in range(0, scenarios, assumptions.scenario_chunk_size):
            s1 = min(s0 + assumptions.scenario_chunk_size, scenarios)
            scenario_ids = np.arange(s0, s1, dtype=np.uint64)[None, :]
            in_force = np.ones((p1 - p0, s1 - s0), dtype=bool)
            block_premium = np.zeros(s1 - s0, dtype=np.float64)
            block_claim = np.zeros(s1 - s0, dtype=np.float64)
            block_expense = np.zeros(s1 - s0, dtype=np.float64)

            for year in range(years):
                active = in_force & (year < terms)
                if not np.any(active):
                    break

                beginning_discount = (1.0 + assumptions.discount_rate) ** year
                end_discount = beginning_discount * (1.0 + assumptions.discount_rate)
                expense = assumptions.expense_per_policy * (1.0 + assumptions.expense_inflation) ** year
                benefit = benefits * (1.0 + assumptions.claim_inflation) ** year

                block_premium += np.sum(active * premiums, axis=0) / beginning_discount
                block_expense += np.sum(active, axis=0) * expense / beginning_discount

                mortality = np.minimum(
                    _mortality_rate(ages + year) * assumptions.mortality_scale,
                    1.0,
                )
                deaths = active & (
                    _uniform(ids, scenario_ids, year, assumptions.seed) < mortality
                )
                block_claim += np.sum(deaths * benefit, axis=0) / end_discount

                survivors = active & ~deaths
                lapse_seed = assumptions.seed ^ _LAPSE_SEED_OFFSET
                lapses = survivors & (
                    _uniform(ids, scenario_ids, year, lapse_seed)
                    < assumptions.annual_lapse_rate
                )
                in_force = survivors & ~lapses & (year + 1 < terms)

            premium_pv[s0:s1] += block_premium
            claim_pv[s0:s1] += block_claim
            expense_pv[s0:s1] += block_expense

    return ActuarialProjectionResult(premium_pv, claim_pv, expense_pv)
