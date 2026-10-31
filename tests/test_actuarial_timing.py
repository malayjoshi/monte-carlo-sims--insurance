"""Acceptance tests for the Stage 3 policy-year timing convention."""

import numpy as np

import monte_carlo_accelerator.actuarial_engine as actuarial_engine
from monte_carlo_accelerator.cpu_engine import _mortality_rate, _uniform
from monte_carlo_accelerator.models import ActuarialAssumptions, TermLifePolicies

_LAPSE_SEED_OFFSET = 0xD1B54A32D192ED03


def _policies(
    *,
    ages=(40,),
    benefits=(100_000.0,),
    premiums=(1_000.0,),
    terms=(5,),
) -> TermLifePolicies:
    count = len(ages)
    return TermLifePolicies(
        policy_id=np.arange(1, count + 1, dtype=np.uint64),
        age_at_entry=np.asarray(ages, dtype=np.int16),
        sum_assured=np.asarray(benefits, dtype=np.float64),
        annual_premium=np.asarray(premiums, dtype=np.float64),
        term_years=np.asarray(terms, dtype=np.int16),
    )


def _scalar_oracle(
    policies: TermLifePolicies,
    scenarios: int,
    assumptions: ActuarialAssumptions,
):
    """Deliberately scalar statement of the documented timing convention."""
    premium_pv = np.zeros(scenarios)
    claim_pv = np.zeros(scenarios)
    expense_pv = np.zeros(scenarios)

    for policy in range(len(policies)):
        for scenario in range(scenarios):
            for year in range(min(assumptions.projection_years, int(policies.term_years[policy]))):
                beginning_discount = (1.0 + assumptions.discount_rate) ** year
                end_discount = beginning_discount * (1.0 + assumptions.discount_rate)
                premium_pv[scenario] += policies.annual_premium[policy] / beginning_discount
                expense_pv[scenario] += (
                    assumptions.expense_per_policy
                    * (1.0 + assumptions.expense_inflation) ** year
                    / beginning_discount
                )

                mortality = min(
                    float(_mortality_rate(int(policies.age_at_entry[policy]) + year))
                    * assumptions.mortality_scale,
                    1.0,
                )
                death_draw = float(
                    _uniform(
                        policies.policy_id[policy],
                        np.uint64(scenario),
                        year,
                        assumptions.seed,
                    )
                )
                if death_draw < mortality:
                    claim_pv[scenario] += (
                        policies.sum_assured[policy]
                        * (1.0 + assumptions.claim_inflation) ** year
                        / end_discount
                    )
                    break

                lapse_draw = float(
                    _uniform(
                        policies.policy_id[policy],
                        np.uint64(scenario),
                        year,
                        assumptions.seed ^ _LAPSE_SEED_OFFSET,
                    )
                )
                if lapse_draw < assumptions.annual_lapse_rate:
                    break

    return premium_pv, claim_pv, expense_pv


def test_vectorized_projection_matches_scalar_timing_oracle():
    policies = _policies(
        ages=(35, 58, 72),
        benefits=(80_000.0, 150_000.0, 60_000.0),
        premiums=(400.0, 1_200.0, 1_600.0),
        terms=(2, 5, 4),
    )
    assumptions = ActuarialAssumptions(
        projection_years=5,
        discount_rate=0.04,
        mortality_scale=8.0,
        annual_lapse_rate=0.35,
        expense_per_policy=30.0,
        expense_inflation=0.03,
        claim_inflation=0.02,
        seed=77,
        policy_chunk_size=2,
        scenario_chunk_size=3,
    )

    actual = actuarial_engine.project_term_life(policies, 7, assumptions)
    expected = _scalar_oracle(policies, 7, assumptions)

    np.testing.assert_allclose(actual.premium_pv, expected[0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(actual.claim_pv, expected[1], rtol=0, atol=1e-10)
    np.testing.assert_allclose(actual.expense_pv, expected[2], rtol=0, atol=1e-10)


def test_term_one_collects_one_beginning_premium_and_expense():
    assumptions = ActuarialAssumptions(
        projection_years=10,
        discount_rate=0,
        mortality_scale=0,
        annual_lapse_rate=0,
        expense_per_policy=25,
    )

    result = actuarial_engine.project_term_life(_policies(terms=(1,)), 2, assumptions)

    np.testing.assert_array_equal(result.premium_pv, [1_000.0, 1_000.0])
    np.testing.assert_array_equal(result.expense_pv, [25.0, 25.0])
    np.testing.assert_array_equal(result.claim_pv, [0.0, 0.0])


def test_earliest_year_death_keeps_beginning_premium(monkeypatch):
    monkeypatch.setattr(actuarial_engine, "_mortality_rate", lambda age: np.ones_like(age, dtype=float))
    assumptions = ActuarialAssumptions(
        projection_years=5,
        discount_rate=0.25,
        annual_lapse_rate=1,
        expense_per_policy=20,
    )

    result = actuarial_engine.project_term_life(_policies(), 1, assumptions)

    np.testing.assert_array_equal(result.premium_pv, [1_000.0])
    np.testing.assert_array_equal(result.expense_pv, [20.0])
    np.testing.assert_array_equal(result.claim_pv, [80_000.0])


def test_first_year_lapse_has_no_claim_and_no_later_cashflows():
    assumptions = ActuarialAssumptions(
        projection_years=5,
        discount_rate=0,
        mortality_scale=0,
        annual_lapse_rate=1,
        expense_per_policy=20,
    )

    result = actuarial_engine.project_term_life(_policies(), 1, assumptions)

    np.testing.assert_array_equal(result.premium_pv, [1_000.0])
    np.testing.assert_array_equal(result.expense_pv, [20.0])
    np.testing.assert_array_equal(result.claim_pv, [0.0])


def test_short_and_long_terms_stop_at_their_own_boundaries():
    policies = _policies(premiums=(100.0, 200.0), terms=(1, 3), ages=(30, 30), benefits=(1, 1))
    assumptions = ActuarialAssumptions(
        projection_years=5,
        discount_rate=0,
        mortality_scale=0,
        annual_lapse_rate=0,
        expense_per_policy=10,
        expense_inflation=0,
    )

    result = actuarial_engine.project_term_life(policies, 1, assumptions)

    np.testing.assert_array_equal(result.premium_pv, [700.0])
    np.testing.assert_array_equal(result.expense_pv, [40.0])
    np.testing.assert_array_equal(result.claim_pv, [0.0])
