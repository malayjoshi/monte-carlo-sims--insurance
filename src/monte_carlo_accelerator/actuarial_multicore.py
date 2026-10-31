"""Threaded multicore execution for Stage 3 term-life projections."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .actuarial_engine import project_term_life
from .models import ActuarialAssumptions, ActuarialProjectionResult, TermLifePolicies


def _policy_slice(
    policies: TermLifePolicies, start: int, stop: int
) -> TermLifePolicies:
    return TermLifePolicies(
        policy_id=policies.policy_id[start:stop],
        age_at_entry=policies.age_at_entry[start:stop],
        sum_assured=policies.sum_assured[start:stop],
        annual_premium=policies.annual_premium[start:stop],
        term_years=policies.term_years[start:stop],
    )


def _partition_ranges(count: int, workers: int) -> list[tuple[int, int]]:
    """Split a portfolio into ordered, near-equal contiguous ranges."""
    worker_count = min(workers, count)
    base, remainder = divmod(count, worker_count)
    ranges = []
    start = 0
    for index in range(worker_count):
        stop = start + base + (index < remainder)
        ranges.append((start, stop))
        start = stop
    return ranges


def project_term_life_multicore(
    policies: TermLifePolicies,
    scenarios: int,
    assumptions: ActuarialAssumptions = ActuarialAssumptions(),
    *,
    workers: int = 2,
) -> ActuarialProjectionResult:
    """Project Stage 3 cash flows across configurable CPU worker threads.

    Policy partitions retain the existing policy and scenario identifiers, so
    the stateless mortality and lapse draws match the single-worker engine.
    NumPy performs the array-heavy work outside Python's interpreter lock.
    """
    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers == 1 or len(policies) == 0:
        return project_term_life(policies, scenarios, assumptions)

    ranges = _partition_ranges(len(policies), workers)
    partitions = [_policy_slice(policies, start, stop) for start, stop in ranges]
    with ThreadPoolExecutor(max_workers=len(partitions)) as executor:
        partial_results = list(
            executor.map(
                project_term_life,
                partitions,
                [scenarios] * len(partitions),
                [assumptions] * len(partitions),
            )
        )

    return ActuarialProjectionResult(
        premium_pv=np.sum([item.premium_pv for item in partial_results], axis=0),
        claim_pv=np.sum([item.claim_pv for item in partial_results], axis=0),
        expense_pv=np.sum([item.expense_pv for item in partial_results], axis=0),
    )
