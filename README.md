# Monte Carlo Accelerator — CPU Baseline

A deterministic, chunked NumPy baseline for a 30-year life-insurance Monte Carlo projection. It simulates mortality, claim payments, claim inflation, and discounting without allocating a policy × scenario × year array.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python benchmarks/benchmark_cpu.py --preset small
```

Benchmark rows are appended to `results/cpu_benchmarks.csv`. Available presets are `small` (100K combinations), `medium` (10M), `large` (100M), and `stress` (500M). Override either dimension with `--policies` and `--scenarios`.

## Reproducibility and scaling

- Policies and simulation draws are deterministic for fixed seeds.
- Random draws are indexed by policy, scenario, and year, so changing chunk sizes does not change results.
- Peak working memory is approximately proportional to `policy_chunk × scenario_chunk`, not the total workload or projection horizon.
- `project_reference` is intentionally slow and is only a correctness oracle for small test cases.

The model is illustrative rather than an actuarial pricing basis. The mortality curve and synthetic portfolio should be replaced or calibrated before using results for business decisions.
