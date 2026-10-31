import numpy as np
import pytest

from monte_carlo_accelerator.policy_generator import generate_term_life_portfolio
from monte_carlo_accelerator.portfolio_io import (
    load_term_life_portfolio,
    write_term_life_portfolio_csv,
)


def test_stage3_generator_is_deterministic():
    first = generate_term_life_portfolio(100, seed=91)
    second = generate_term_life_portfolio(100, seed=91)

    np.testing.assert_array_equal(first.policy_id, second.policy_id)
    np.testing.assert_array_equal(first.age_at_entry, second.age_at_entry)
    np.testing.assert_array_equal(first.sum_assured, second.sum_assured)
    np.testing.assert_array_equal(first.annual_premium, second.annual_premium)
    np.testing.assert_array_equal(first.term_years, second.term_years)


def test_stage3_generator_produces_valid_bounded_portfolio():
    portfolio = generate_term_life_portfolio(10_000, seed=8)

    assert len(portfolio) == 10_000
    np.testing.assert_array_equal(portfolio.policy_id, np.arange(10_000, dtype=np.uint64))
    assert portfolio.age_at_entry.dtype == np.int16
    assert portfolio.term_years.dtype == np.int16
    assert portfolio.sum_assured.dtype == np.float64
    assert portfolio.annual_premium.dtype == np.float64
    assert np.all((portfolio.age_at_entry >= 20) & (portfolio.age_at_entry <= 65))
    assert np.all((portfolio.term_years >= 5) & (portfolio.term_years <= 30))
    assert np.all(portfolio.age_at_entry + portfolio.term_years <= 80)
    assert np.all((portfolio.sum_assured >= 25_000) & (portfolio.sum_assured <= 2_000_000))
    assert np.all(portfolio.annual_premium > 0)


def test_age_sensitive_premium_relationship_is_present_at_scale():
    portfolio = generate_term_life_portfolio(20_000, seed=42)
    premium_rate = portfolio.annual_premium / portfolio.sum_assured
    young = premium_rate[portfolio.age_at_entry < 30]
    older = premium_rate[portfolio.age_at_entry >= 55]

    assert np.median(older) > np.median(young)


def test_generated_portfolio_round_trips_through_csv(tmp_path):
    expected = generate_term_life_portfolio(25, seed=4)
    destination = tmp_path / "generated" / "portfolio.csv"

    write_term_life_portfolio_csv(expected, destination)
    actual = load_term_life_portfolio(destination)

    np.testing.assert_array_equal(actual.policy_id, expected.policy_id)
    np.testing.assert_array_equal(actual.age_at_entry, expected.age_at_entry)
    np.testing.assert_array_equal(actual.sum_assured, expected.sum_assured)
    np.testing.assert_array_equal(actual.annual_premium, expected.annual_premium)
    np.testing.assert_array_equal(actual.term_years, expected.term_years)


def test_stage3_generator_rejects_non_positive_count():
    with pytest.raises(ValueError, match="positive"):
        generate_term_life_portfolio(0)
