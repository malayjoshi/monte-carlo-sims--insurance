import numpy as np

from .models import Policies, TermLifePolicies


def generate_policies(count: int, seed: int = 17) -> Policies:
    """Generate a reproducible synthetic term-life portfolio."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    return Policies(
        policy_id=np.arange(count, dtype=np.uint64),
        age=rng.integers(20, 71, count, dtype=np.int16),
        benefit=np.round(rng.lognormal(np.log(100_000), 0.55, count), 2),
    )


def generate_term_life_portfolio(count: int, seed: int = 17) -> TermLifePolicies:
    """Generate a deterministic, illustrative term-life portfolio.

    The generator is intended for correctness and performance testing. Premiums
    increase with entry-age mortality, term length, and sum assured, but they are
    not calibrated to a commercial pricing basis.
    """
    if count <= 0:
        raise ValueError("count must be positive")

    rng = np.random.default_rng(seed)
    age = rng.integers(20, 66, count, dtype=np.int16)
    maximum_term = np.minimum(30, 80 - age).astype(np.int16)
    term = rng.integers(5, maximum_term + 1, dtype=np.int16)
    sum_assured = np.round(
        np.clip(rng.lognormal(np.log(250_000), 0.65, count), 25_000, 2_000_000),
        -2,
    ).astype(np.float64)

    entry_mortality = np.minimum(0.00035 * np.exp(0.085 * (age.astype(float) - 30)), 0.35)
    term_loading = 1.0 + 0.01 * np.maximum(term.astype(float) - 10, 0)
    annual_premium = np.round(
        50.0 + sum_assured * (0.0025 + 1.6 * entry_mortality) * term_loading,
        2,
    )

    return TermLifePolicies(
        policy_id=np.arange(count, dtype=np.uint64),
        age_at_entry=age,
        sum_assured=sum_assured,
        annual_premium=annual_premium,
        term_years=term,
    )
