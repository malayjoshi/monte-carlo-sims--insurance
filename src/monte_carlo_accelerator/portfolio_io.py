"""CSV input and validation for Stage 3 term-life portfolios."""

import csv
from pathlib import Path

import numpy as np

from .models import TermLifePolicies

REQUIRED_COLUMNS = (
    "policy_id",
    "age_at_entry",
    "sum_assured",
    "annual_premium",
    "term_years",
)


def load_term_life_portfolio(path: str | Path) -> TermLifePolicies:
    """Load a term-life portfolio from a CSV file with strict validation."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"portfolio is missing required columns: {', '.join(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("portfolio must contain at least one policy")

    try:
        policy_id = np.asarray([int(row["policy_id"]) for row in rows], dtype=np.uint64)
        age = np.asarray([int(row["age_at_entry"]) for row in rows], dtype=np.int16)
        sum_assured = np.asarray([float(row["sum_assured"]) for row in rows], dtype=np.float64)
        premium = np.asarray([float(row["annual_premium"]) for row in rows], dtype=np.float64)
        term = np.asarray([int(row["term_years"]) for row in rows], dtype=np.int16)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"portfolio contains an invalid value: {error}") from error

    if len(np.unique(policy_id)) != len(policy_id):
        raise ValueError("policy_id values must be unique")
    if np.any((age < 0) | (age > 120)):
        raise ValueError("age_at_entry must be between 0 and 120")
    if np.any(sum_assured < 0) or np.any(premium < 0):
        raise ValueError("sum_assured and annual_premium must be non-negative")
    if np.any(term <= 0):
        raise ValueError("term_years must be positive")

    return TermLifePolicies(policy_id, age, sum_assured, premium, term)


def write_term_life_portfolio_csv(policies: TermLifePolicies, path: str | Path) -> None:
    """Write a term-life portfolio using the canonical input schema."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(REQUIRED_COLUMNS)
        writer.writerows(
            zip(
                policies.policy_id,
                policies.age_at_entry,
                policies.sum_assured,
                policies.annual_premium,
                policies.term_years,
                strict=True,
            )
        )
