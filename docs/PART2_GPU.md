# Part 2 — NVIDIA GPU acceleration

Part 2 ports the validated NumPy projection to CuPy while preserving the indexed random stream and numerical model. No AWS resources are created by the repository.

## Current milestone

- `project_gpu` matches the CPU-facing API and includes host-to-device policy transfer.
- `prepare_gpu_policies` plus `project_gpu_prepared` separates policy transfer for compute-only measurements.
- CuPy is imported lazily, so CPU-only development and tests do not require CUDA.
- Frozen RNG vectors cover scalar, boundary, and large identifier inputs.
- The GPU algorithm is exercised with a NumPy backend locally; real CUDA validation must run on the AWS host.
- `src.engine.project` provides explicit `cpu`/`gpu` selection and automatic dispatch. The automatic default attempts GPU execution at 10 million policy-scenario combinations and otherwise stays on CPU. If CUDA is unavailable it falls back to CPU; execution failures after a GPU is selected are not hidden.

```python
from src import project

result = project(policies, scenarios)  # backend="auto"
```

`gpu_threshold` is configurable. The 10 million default is a conservative policy based on the recorded `g4dn.xlarge`/Tesla T4 run, not a universal crossover point. Automatic calls below the threshold do not probe CUDA, avoiding the observed first-call initialization penalty for small work.

## Persistent warm worker

Long-running services can initialize CUDA once and reuse the same context and
compiled kernels:

```python
from src import WarmGPUWorker, project

gpu = WarmGPUWorker()

def handle_projection(policies, scenarios, config):
    return project(
        policies,
        scenarios,
        config,
        backend="auto",
        gpu_worker=gpu,
    )
```

`WarmGPUWorker` performs a tiny projection during construction and serializes
calls through one GPU. GPU failures remain visible. Automatic dispatch may still
choose CPU below the threshold, but constructing the worker itself intentionally
initializes CUDA, so create it only in a GPU-serving process.

When the same policies serve several scenario runs, avoid repeating the host-to-
device transfer:

```python
prepared = gpu.prepare(policies)
first = gpu.project_prepared(prepared, 1_000, config)
second = gpu.project_prepared(prepared, 2_000, config)
```

Prepared policies belong to the worker/device that created them. Keep the worker
alive for the lifetime of those arrays and do not share one worker concurrently
across multiple processes.

### AWS warm-worker validation

The 2026-10-24 `g4dn.xlarge`/Tesla T4 validation completed successfully. Worker
construction and its one-time warm-up took 6.015 seconds. After that initialization,
the small workload had a 0.022330-second median and the medium workload had a
0.554947-second median across five repetitions. These correspond to 4.48 million
and 18.02 million combinations per second. Warm end-to-end calls, including policy
transfer, took 0.022489 and 0.552275 seconds respectively.

Both workloads matched the CPU result within the declared tolerance. Maximum
relative differences were `3.47e-16` for small and `4.01e-16` for medium. The run
confirms that a process-lifetime worker moves the roughly six-second CUDA startup
cost out of request latency; it does not remove that cost from process startup.

The validation instance initiated termination immediately after success.

## NVIDIA host setup

CuPy publishes distinct wheels for CUDA 12 and CUDA 13. Install exactly one CuPy package. For the first AWS run, use CUDA 12:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e '.[gpu-cuda12]'
python -c 'import cupy; cupy.show_config()'
nvidia-smi
python -m pytest
```

If the AMI supplies only the NVIDIA driver rather than the CUDA runtime components, use the official CuPy toolkit extra instead:

```bash
python -m pip install 'cupy-cuda12x[ctk]>=14,<15'
```

## Benchmark

Start with correctness and memory safety before scaling:

```bash
python benchmarks/benchmark_gpu.py --preset small
python benchmarks/benchmark_gpu.py --preset medium
```

The runner records two scopes:

- **Compute-only:** policies are already resident on the GPU; projection and the small scenario-total copy back to the host are timed.
- **End-to-end:** policy transfer, projection, and result transfer are timed.

Every run computes the CPU result using the same inputs and requires `rtol=1e-12, atol=1e-6`.

## Completed AWS benchmark

The 2026-10-24 run completed within the three-hour/USD 5 constraint, and all benchmark instances were terminated. No new AWS run is required for automatic dispatch.

Warm GPU compute medians on a Tesla T4 were 0.022 s, 0.557 s, 4.893 s, and 24.226 s for 100K, 10M, 100M, and 500M combinations. Against the recorded EC2 NumPy medians, these correspond to observed implementation speedups of 3.68x, 14.95x, 16.45x, and 16.22x. Per-scenario CPU/GPU comparisons passed the declared tolerance; maximum relative differences were approximately `3.5e-16` to `5.0e-16`.

These are performance observations, not production cost claims. CPU and GPU measurements used the same instance type, AMI, region, workload, and seed, but came from separate instances/physical hosts. The CPU used `2048 x 256` chunks while the GPU used `4096 x 128`, and the baseline is the current NumPy implementation rather than an optimized multicore CPU implementation. Pricing, utilization, queueing, startup, idle time, and operational overhead were not measured as a production system.

The first small GPU compute call took 5.55 s due to CUDA initialization; later calls took about 0.022 s. `WarmGPUWorker` now moves that initialization into process startup. Chunk tuning remains a separate controlled experiment requiring identical workloads, repeated measurements, numerical checks, and memory telemetry; the existing evidence does not establish a better T4 chunk configuration.
