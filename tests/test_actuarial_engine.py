import numpy as np
import pytest

from monte_carlo_accelerator.actuarial_engine import project_term_life
from monte_carlo_accelerator.models import ActuarialAssumptions, TermLifePolicies


def _portfolio() -> TermLifePolicies:
    return TermLifePolicies(
        policy_id=np.array([10, 20, 30], dtype=np.uint64),
        age_at_entry=np.array([30, 45, 60], dtype=np.int16),
        sum_assured=np.array([100_000, 150_000, 80_000], dtype=np.float64),
        annual_premium=np.array([500, 900, 1_100], dtype=np.float64),
        term_years=np.array([5, 8, 3], dtype=np.int16),
    )


def test_component_cashflows_reconcile_to_net_cashflow():
    result = project_term_life(_portfolio(), 7, ActuarialAssumptions(projection_years=6))

    np.testing.assert_array_equal(
        result.net_cashflow_pv,
        result.premium_pv - result.claim_pv - result.expense_pv,
    )
    assert result.mean_net_cashflow_pv == pytest.approx(np.mean(result.net_cashflow_pv))


def test_zero_decrements_produce_deterministic_premiums_and_expenses():
    assumptions = ActuarialAssumptions(
        projection_years=2,
        discount_rate=0,
        mortality_scale=0,
        annual_lapse_rate=0,
        expense_per_policy=10,
        expense_inflation=0,
    )
    result = project_term_life(_portfolio(), 4, assumptions)

    np.testing.assert_array_equal(result.premium_pv, np.full(4, 5_000.0))
    np.testing.assert_array_equal(result.expense_pv, np.full(4, 60.0))
    np.testing.assert_array_equal(result.claim_pv, np.zeros(4))


def test_chunk_dimensions_do_not_change_actuarial_results():
    common = dict(projection_years=7, seed=41, annual_lapse_rate=0.12)
    first = project_term_life(
        _portfolio(), 9, ActuarialAssumptions(**common, policy_chunk_size=3, scenario_chunk_size=9)
    )
    second = project_term_life(
        _portfolio(), 9, ActuarialAssumptions(**common, policy_chunk_size=1, scenario_chunk_size=2)
    )

    np.testing.assert_allclose(first.premium_pv, second.premium_pv, rtol=0, atol=1e-10)
    np.testing.assert_allclose(first.claim_pv, second.claim_pv, rtol=0, atol=1e-10)
    np.testing.assert_allclose(first.expense_pv, second.expense_pv, rtol=0, atol=1e-10)


def test_invalid_actuarial_inputs_are_rejected():
    with pytest.raises(ValueError, match="scenarios"):
        project_term_life(_portfolio(), 0)
    with pytest.raises(ValueError, match="lapse"):
        ActuarialAssumptions(annual_lapse_rate=1.1)
