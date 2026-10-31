# Stage 3 actuarial workflow

Stage 3 adds a recognizable level-premium term-life workflow without changing
the Stage 1 claim-only CPU baseline or the Stage 2 CPU/GPU benchmarks.

## Portfolio input

Load a CSV containing these required columns:

```text
policy_id,age_at_entry,sum_assured,annual_premium,term_years
101,35,250000,1250,20
102,51,100000,900,10
```

Policy identifiers must be unique. Ages must be between 0 and 120; monetary
amounts must be non-negative; terms must be positive.

## Projection

```python
from monte_carlo_accelerator import (
    ActuarialAssumptions,
    load_term_life_portfolio,
    project_term_life,
)

portfolio = load_term_life_portfolio("portfolio.csv")
assumptions = ActuarialAssumptions(
    projection_years=30,
    discount_rate=0.03,
    mortality_scale=1.0,
    annual_lapse_rate=0.05,
    expense_per_policy=25.0,
)
result = project_term_life(portfolio, scenarios=10_000, assumptions=assumptions)

print(result.mean_net_cashflow_pv)
```

The projection returns scenario-level present values for premiums, death claims,
expenses, and net cash flow. Premiums and maintenance expenses occur at the
beginning of each policy year. Death claims occur at year end. Lapses occur after
mortality and affect the following year. A policy exits at death, lapse, or the
end of its term.

Mortality currently uses the repository's illustrative smooth mortality curve
with a configurable scale. It is not a published mortality table and the output
is not suitable for pricing, reserving, or financial reporting.

## Scope boundary

This first Stage 3 slice is CPU-only. The established claim-only Stage 2 engine
still supports automatic CPU/GPU dispatch. GPU acceleration for the richer
multi-cash-flow model should follow only after the actuarial definitions and
acceptance examples are agreed and frozen.
