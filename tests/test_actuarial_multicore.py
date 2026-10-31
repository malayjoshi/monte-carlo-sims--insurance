import numpy as np
import pytest

from monte_carlo_accelerator.actuarial_engine import project_term_life
from monte_carlo_accelerator.actuarial_multicore import (
    _partition_ranges,
    project_term_life_multicore,
)
from monte_carlo_accelerator.models import ActuarialAssumptions
from monte_carlo_accelerator.policy_generator import generate_term_life_portfolio


def test_partition_ranges_are_balanced_and_cover_every_policy():
    assert _partition_ranges(10, 3) == [(0, 4), (4, 7), (7, 10)]
    assert _partition_ranges(3, 8) == [(0, 1), (1, 2), (2, 3)]


def test_multicore_projection_matches_single_worker_components():
    policies = generate_term_life_portfolio(37, seed=9)
    assumptions = ActuarialAssumptions(
        projection_years=12,
        mortality_scale=2.5,
        annual_lapse_rate=0.13,
        seed=81,
        policy_chunk_size=11,
        scenario_chunk_size=4,
    )

    expected = project_term_life(policies, 13, assumptions)
    actual = project_term_life_multicore(policies, 13, assumptions, workers=3)

    np.testing.assert_allclose(actual.premium_pv, expected.premium_pv, rtol=1e-12)
    np.testing.assert_allclose(actual.claim_pv, expected.claim_pv, rtol=1e-12)
    np.testing.assert_allclose(actual.expense_pv, expected.expense_pv, rtol=1e-12)


def test_multicore_projection_is_exactly_repeatable():
    policies = generate_term_life_portfolio(19, seed=5)

    first = project_term_life_multicore(policies, 7, workers=4)
    second = project_term_life_multicore(policies, 7, workers=4)

    assert np.array_equal(first.premium_pv, second.premium_pv)
    assert np.array_equal(first.claim_pv, second.claim_pv)
    assert np.array_equal(first.expense_pv, second.expense_pv)


def test_one_worker_uses_single_worker_engine(monkeypatch):
    policies = generate_term_life_portfolio(2)
    expected = object()
    monkeypatch.setattr(
        "monte_carlo_accelerator.actuarial_multicore.project_term_life",
        lambda *args: expected,
    )

    assert project_term_life_multicore(policies, 2, workers=1) is expected


def test_multicore_rejects_invalid_worker_count():
    policies = generate_term_life_portfolio(2)
    with pytest.raises(ValueError, match="workers"):
        project_term_life_multicore(policies, 2, workers=0)
