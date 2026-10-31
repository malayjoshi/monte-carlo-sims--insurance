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

### Synthetic portfolios

For correctness and performance testing, generate deterministic portfolios in
memory without storing large datasets in Git:

```python
from monte_carlo_accelerator import generate_term_life_portfolio

portfolio = generate_term_life_portfolio(100_000, seed=17)
```

To exercise the CSV ingestion path, create a file on demand:

```bash
python benchmarks/generate_stage3_portfolio.py \
  --count 100000 \
  --seed 17 \
  --output results/stage3_portfolio_100k.csv
```

The synthetic generator produces entry ages from 20 to 65, terms of 5 to 30
years ending no later than age 80, skewed sum-assured values between 25,000 and
2,000,000, and positive age-sensitive premiums. These distributions are
illustrative and are not calibrated to observed business or a pricing basis.

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

## CPU/GPU execution

The original `project_term_life` API remains the explicit CPU path. The reviewed
timing convention is also implemented with optional CuPy:

```python
from monte_carlo_accelerator import project_term_life_gpu

result = project_term_life_gpu(portfolio, scenarios=10_000, assumptions=assumptions)
```

For repeated runs, transfer a portfolio once and reuse it:

```python
from monte_carlo_accelerator import (
    prepare_gpu_term_life_policies,
    project_term_life_gpu_prepared,
)

gpu_portfolio = prepare_gpu_term_life_policies(portfolio)
result = project_term_life_gpu_prepared(gpu_portfolio, 10_000, assumptions)
```

CuPy remains optional and the CPU environment does not initialize CUDA. Local
tests run the GPU algorithm through its NumPy-compatible backend and require
component-level parity for premium, claim, and expense present values. Real CUDA
correctness, memory use, and performance must be verified on NVIDIA hardware.

For automatic backend selection, use `project_term_life_auto`:

```python
from monte_carlo_accelerator import project_term_life_auto

result = project_term_life_auto(portfolio, scenarios=10_000, assumptions=assumptions)
```

The default threshold is 10 million policy-scenario combinations. Smaller work
uses CPU without probing or initializing CUDA. At or above the threshold, the
dispatcher tries GPU and falls back to CPU only when CuPy or CUDA is unavailable.
Other GPU errors remain visible. Both the threshold and backend can be set per
call with `gpu_threshold=...` and `backend="auto"`, `"cpu"`, or `"gpu"`.

The default is a conservative operating point based on matched CPU/GPU runs on
an AWS `g4dn.xlarge` with an NVIDIA Tesla T4:

| Workload | CPU | Warm GPU | Speedup |
| ---: | ---: | ---: | ---: |
| 100,000 | 0.169 s | 0.048 s | 3.55x |
| 10 million | 15.322 s | 1.208 s | 12.68x |
| 100 million | 140.877 s | 10.354 s | 13.61x |
| 500 million | 688.394 s | 51.490 s | 13.37x |

This is a hardware-specific benchmark policy, not a universal crossover. Adjust
it after measuring the target CPU, GPU, CUDA startup model, and workload mix.

## CPU benchmark

Run a repeatable Stage 3 CPU benchmark with a generated portfolio:

```bash
python benchmarks/benchmark_stage3_cpu.py --preset small
```

The presets are `small` (100,000 combinations), `medium` (10 million), `large`
(100 million), and `stress` (500 million). Custom policy and scenario counts can
be supplied with `--policies` and `--scenarios`.

To benchmark projection using a portfolio loaded from CSV:

```bash
python benchmarks/benchmark_stage3_cpu.py \
  --portfolio-csv results/stage3_portfolio_100k.csv \
  --scenarios 100
```

Portfolio generation or CSV loading is measured and reported separately. It is
excluded from projection throughput. The benchmark records the first projection
as cold, performs configurable warmups, measures repeated warm projections,
checks exact repeatability across premium, claim, and expense arrays, and writes
summary CSV, run-level CSV, and detailed JSON outputs under `results/`.

Throughput counts policy-scenario combinations, matching the earlier benchmark
convention. `policy_scenario_years` is also recorded in JSON because the Stage 3
engine performs multiple cash-flow and decrement operations per projected year.

## GPU benchmark

On an NVIDIA CUDA host with the optional CuPy dependency installed, start with
the small workload:

```bash
python benchmarks/benchmark_stage3_gpu.py --preset small
```

Then run medium before deciding whether available memory and elapsed time permit
large and stress workloads. The runner records CUDA startup, policy preparation,
cold prepared compute, repeated warm prepared compute, warm end-to-end execution,
CuPy memory-pool snapshots, device metadata, and environment metadata.

Every GPU benchmark also runs an untimed CPU reference using identical inputs,
assumptions, seed, and chunk sizes. Premium, claim, expense, and derived net cash
flows must each satisfy `rtol=1e-12, atol=1e-6`; the benchmark fails instead of
writing a successful result when any component exceeds that tolerance.

CPU performance remains the responsibility of `benchmark_stage3_cpu.py`. This
prevents a single untuned CPU correctness run inside the GPU runner from being
reported as a robust speedup baseline. Compare repeated CPU and GPU medians only
when they use matching hardware, workload, assumptions, and declared methodology.
