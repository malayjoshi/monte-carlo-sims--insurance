# Monte Carlo Accelerator — CPU Baseline

A deterministic, chunked NumPy baseline for a 30-year synthetic life-insurance Monte Carlo projection. It simulates mortality, claim payments, claim inflation, and discounting without allocating a policy × scenario × year array.

This is an educational performance-engineering project. The model and data are synthetic and are not an actuarial pricing basis.

## Quick start

```fish
source venv/bin/activate.fish
python -m pip install -e '.[dev]'
python -m pytest
python benchmarks/benchmark_cpu.py --preset small
```

Available presets are `small` (100K combinations), `medium` (10M), `large` (100M), and `stress` (500M). `--policies` and `--scenarios` override either dimension.

## Numerical model and reproducibility

- Policies are deterministically generated from a fixed seed.
- SplitMix64-derived draws are indexed by policy, scenario, year, and seed.
- Random inputs therefore do not depend on chunk dimensions or processing order.
- Projection totals use `float64`; different policy chunk boundaries can change reduction order and the final few bits.
- Working memory is proportional to `policy_chunk × scenario_chunk`, not the total workload or projection horizon.
- `project_reference` is a deliberately slow scalar oracle used only for small correctness tests.

## Benchmark methodology

The enhanced runner performs these operations in order:

1. Import modules, construct configuration, and generate policies outside the timed region.
2. Measure the first `project_cpu` call as a **cold compute invocation**.
3. Execute configurable unmeasured warmups.
4. Measure multiple warmed `project_cpu` repetitions.
5. Report mean, median, sample standard deviation, minimum, and maximum runtime and throughput.

“Cold” here means the first projection call in an already-started Python process; it does not include interpreter startup, imports, policy generation, metadata collection, or result serialization. It is not an end-to-end latency measurement.

Linux resident memory is sampled from `/proc/self/status` every 5 ms during each measured call. This captures native NumPy allocations better than Python-only allocation tracing, but it is sampled rather than an exact allocator high-water mark.

```fish
python benchmarks/benchmark_cpu.py \
  --preset medium \
  --warmups 1 \
  --repetitions 5
```

Outputs:

- `results/cpu_benchmarks.csv`: append-only legacy-compatible summary rows; enhanced rows use the warm median.
- `results/cpu_benchmark_runs.csv`: cold and warmed run-level observations.
- `results/cpu_benchmark_latest.json`: full methodology, statistics, workload, and environment metadata.
- `results/environment.json`: standalone host metadata.

Capture host metadata independently with `python benchmarks/capture_environment.py`.

## Validated local environment

- CPU: AMD Ryzen 7 5825U with Radeon Graphics
- CPU topology: 8 physical cores, 16 logical CPUs
- RAM: 20,365,152 kB reported by Linux (about 19.4 GiB)
- Kernel: 6.18.52-1-cachyos-lts
- Python: 3.14.7
- NumPy: 2.5.4
- Thread-count environment variables: unset during the recorded run

The complete Linux distribution, platform, BLAS configuration, Git SHA, timestamp, and threading variables are stored in `results/environment.json` and each enhanced benchmark JSON document.

## Repeated medium-workload result

The validated workload contains 10,000 policies × 1,000 scenarios × 30 years. Imports and policy generation are excluded.

| Configuration | Repetitions | Warm median | Std. dev. | Median throughput | Peak sampled RSS |
|---|---:|---:|---:|---:|---:|
| Original `2048 × 256` | 5 | 3.984 s | 0.100 s | 2.510M combinations/s | 58.0 MB |
| Candidate `4096 × 128` | 5 | 3.502 s | 0.055 s | 2.855M combinations/s | 58.1 MB |

The candidate delivered about 13.8% higher median throughput in these runs. This is a single-host result, not a universal optimum; rerun the sweep on every target system.

The original configuration's first compute invocation took 5.155 s, compared with its 3.984 s warm median. Cold/warm behavior must therefore be reported separately.

## Profiling

Run `python benchmarks/profile_cpu.py` to create a reusable `.prof` capture and text report.

The medium-workload cProfile report records 4.068 s total under profiling. `_uniform` accounts for 3.060 s cumulative (about 75%); visible reduction calls account for 0.088 s, and `_mortality_rate` accounts for 0.024 s. The remaining time attributed to `project_cpu` includes array expressions, allocation, boolean operations, and compiled NumPy work.

cProfile observes Python call boundaries and aggregate native calls. It does **not** identify time inside individual compiled NumPy kernels, so the report must not be interpreted as native-kernel-level attribution. Reports are stored under `results/profiling/`.

## Chunk-size sensitivity

Run `python benchmarks/benchmark_chunks.py` to compare identical inputs across block shapes. The sweep warms each configuration once and records three measured repetitions.

On the recorded host, `4096 × 128` was the best of the tested configurations. Larger scenario blocks were not consistently faster: `2048 × 512` was slower and used more memory. Full results are in `results/chunk_sensitivity.csv`.

## CPU/GPU numerical acceptance criteria

Future GPU work should use these primary criteria:

- Use `float64` for policy benefits, mortality calculations, discount factors, and projection totals.
- Reproduce the indexed uint64 SplitMix64 input stream exactly wherever practical.
- Validate representative raw uint64 states and converted uniforms before validating projections.
- Compare per-scenario present values with `rtol=1e-12` and `atol=1e-6`.
- Do not require bitwise-identical totals across different reduction trees.

The local chunk sweep observed a maximum absolute difference of `1.19209289551e-7` and maximum relative difference of `5.239e-16`, comfortably within these criteria. Statistical equivalence alone is not sufficient for the primary CPU/GPU parity test. It may be used only for a separately identified alternative RNG experiment with predeclared distributional tests and sample sizes.

## Limitations

- Results come from one laptop-class CPU and one software environment.
- CPU frequency, thermal state, scheduler activity, and power mode were not pinned.
- RSS is sampled and can miss extremely short-lived allocation peaks.
- cProfile adds overhead and cannot decompose compiled NumPy kernels.
- The validated repeated result covers the medium workload; earlier large and stress results were single runs.
- The combinations/s metric counts policy-scenario pairs, each projected across 30 annual steps.
- No NVIDIA GPU or AWS resources are used in this phase.

## Part 2: optional NVIDIA GPU engine

Part 2 development lives in `src/monte_carlo_accelerator/gpu_engine.py` and keeps CuPy optional so the CPU environment remains usable. See [`docs/PART2_GPU.md`](docs/PART2_GPU.md) for CUDA installation, parity checks, and benchmark methodology.

Use `monte_carlo_accelerator.project(...)` for safe automatic dispatch. It keeps workloads below 10 million policy-scenario combinations on CPU, attempts GPU at or above that configurable threshold, and falls back to CPU when CuPy/CUDA is unavailable. The threshold comes from one completed `g4dn.xlarge`/Tesla T4 benchmark and should be recalibrated for other production hardware and workload patterns.

For a service that handles repeated GPU jobs, create one warm worker when the
process starts and pass it to each projection:

```python
from monte_carlo_accelerator import WarmGPUWorker, project

gpu = WarmGPUWorker()  # initializes CUDA and runs one tiny warm-up projection
result = project(policies, scenarios, gpu_worker=gpu)
```

The worker keeps the CUDA context and compiled CuPy kernels alive and serializes
access to its GPU. Create it only in processes intended to use CUDA; constructing
it eagerly initializes the GPU even if a later automatic-dispatch call is below
the threshold.
