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

## Completion boundary

The code and CPU-backed tests establish Stage 4 memory behavior. Final Tesla T4
performance and maximum safe workload remain AWS validation tasks.
