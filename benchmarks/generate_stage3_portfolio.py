#!/usr/bin/env python3
"""Generate a deterministic Stage 3 portfolio CSV on demand."""

import argparse
from pathlib import Path

from monte_carlo_accelerator import generate_term_life_portfolio, write_term_life_portfolio_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic term-life portfolio")
    parser.add_argument("--count", type=int, required=True, help="number of policies")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    portfolio = generate_term_life_portfolio(args.count, args.seed)
    write_term_life_portfolio_csv(portfolio, args.output)
    print(f"wrote {len(portfolio):,} policies to {args.output}")


if __name__ == "__main__":
    main()
