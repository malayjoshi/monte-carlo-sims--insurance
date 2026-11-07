# Stage 4: Memory-Aware GPU Execution

Stage 4 makes the term-life GPU path safer for workloads whose requested chunk
size may exceed current device memory. It does not change actuarial assumptions,
random numbers, cash-flow timing, or `float64` precision.

## Behavior

`project_term_life_gpu_adaptive` and
`project_term_life_gpu_adaptive_prepared`:

- read current free CUDA device memory;
- reserve 30% by default for the runtime and other processes;
- include the three scenario output arrays in the memory budget;
- reduce scenario chunks first, then policy chunks;
- retry smaller chunks after a CuPy allocation failure;
- return an `AdaptiveGpuReport` with the decision and attempt count.

The `estimated_bytes_per_cell` default is deliberately conservative. It covers
simultaneously live boolean, integer, and floating-point intermediates. The OOM
retry remains the final safety mechanism because allocator fragmentation and
CuPy kernel behavior vary by device and runtime.

## Application use

```python
from monte_carlo_accelerator import project_term_life_gpu_adaptive

result, report = project_term_life_gpu_adaptive(
    policies,
    scenarios=1_000,
    assumptions=assumptions,
    memory_fraction=0.70,
)
print(report)
```

## Local validation

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

The local tests use NumPy as the compatible array backend. They verify chunk
selection, numerical parity, retry behavior, and public API wiring without
requiring CUDA.

## AWS Tesla T4 validation

Run the existing fixed-chunk benchmark first, then the adaptive benchmark on
the same warm instance:

```bash
python benchmarks/benchmark_stage3_gpu.py \
  --preset medium \
  --warmups 1 \
  --repetitions 5 \
  --json-output results/stage4_fixed_t4.json

python benchmarks/benchmark_stage3_gpu.py \
  --preset medium \
  --adaptive \
  --gpu-memory-fraction 0.70 \
  --warmups 1 \
  --repetitions 5 \
  --json-output results/stage4_adaptive_t4.json
```

Repeat the adaptive run for `large` and `stress` only after medium passes CPU/GPU
parity. Compare warm median runtime, throughput, effective chunks, memory-pool
use, and retry count. Do not claim a speed improvement until those matched T4
runs complete.

## Completed AWS validation

On 2026-11-07, commit `e815f92863450698345c0bad0ea49195ab64a2af` completed
validation on a `g4dn.xlarge` / Tesla T4 in `ap-south-1`. All runs used 30-year
projections, one warmup, and five timed repetitions. CPU reference projections
used identical inputs, assumptions, seed, and chunks.

| Mode | Combinations | Warm compute median | Throughput (combinations/s) |
|---|---:|---:|---:|
| Fixed medium | 10,000,000 | 1.197322 s | 8,351,974 |
| Adaptive medium | 10,000,000 | 1.183437 s | 8,449,966 |
| Adaptive large | 100,000,000 | 10.366368 s | 9,646,580 |
| Adaptive stress | 500,000,000 | 51.566261 s | 9,696,262 |

Every run passed CPU/GPU parity at `rtol=1e-12, atol=1e-6`. Adaptive runs
retained the requested `4096 x 128` chunks. The small medium timing difference
does not establish a general speed improvement.

Real CUDA checks also passed for default execution, a restricted memory budget,
allocator OOM recovery, and the host-facing API. The restricted-budget case
reduced `512 x 128` chunks to `512 x 11`; a 1 MiB allocator limit triggered
three attempts and succeeded at `512 x 32`, preserving numerical parity.
The test suite reported 72 passed and 2 skipped.

The largest validated workload is 500 million policy-scenario combinations
(15 billion policy-scenario-years). This is not a measured maximum capacity.
The run exited with status 0 at 17:40:58 UTC (23:10:58 IST), and instance
`i-01768984ed0af0a3d` subsequently terminated.

Raw benchmark JSON, CSV, environment details, CUDA validation script and results,
and test output are archived in
[`results/aws-stage4-t4-20261107`](../results/aws-stage4-t4-20261107).
The archive was recovered from the completed console response captured before
termination; subsequent EC2 console requests returned an older snapshot.

## Completion boundary

Stage 4 memory-aware execution and the planned Tesla T4 validation are complete.
Further capacity-limit discovery or device-specific tuning is separate work.
