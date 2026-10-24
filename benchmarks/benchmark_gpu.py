#!/usr/bin/env python3
"""Run equivalent CuPy and NumPy projections on an NVIDIA host."""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_cpu import PRESETS
from src.cpu_engine import project_cpu
from src.environment import collect_environment
from src.gpu_engine import _load_cupy, prepare_gpu_policies, project_gpu, project_gpu_prepared
from src.metrics import measure, summarize
from src.models import ProjectionConfig
from src.policy_generator import generate_policies


def _device_metadata(cupy) -> dict:
    device_id = cupy.cuda.Device().id
    properties = cupy.cuda.runtime.getDeviceProperties(device_id)
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return {
        "device_id": device_id,
        "name": name,
        "compute_capability": f"{properties.get('major')}.{properties.get('minor')}",
        "total_global_memory_bytes": properties.get("totalGlobalMem"),
        "multiprocessor_count": properties.get("multiProcessorCount"),
        "cupy_version": cupy.__version__,
        "cuda_runtime_version": cupy.cuda.runtime.runtimeGetVersion(),
        "cuda_driver_version": cupy.cuda.runtime.driverGetVersion(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the CuPy projection against NumPy")
    parser.add_argument("--preset", choices=PRESETS, default="small")
    parser.add_argument("--policies", type=int)
    parser.add_argument("--scenarios", type=int)
    parser.add_argument("--policy-chunk", type=int, default=4_096)
    parser.add_argument("--scenario-chunk", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/gpu_benchmarks.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("results/gpu_benchmark_latest.json"))
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("warmups must be >= 0 and repetitions must be >= 1")

    cupy = _load_cupy()
    policy_count, scenario_count = PRESETS[args.preset]
    policy_count = args.policies or policy_count
    scenario_count = args.scenarios or scenario_count
    combinations = policy_count * scenario_count
    config = ProjectionConfig(policy_chunk_size=args.policy_chunk, scenario_chunk_size=args.scenario_chunk)
    policies = generate_policies(policy_count)
    prepared = prepare_gpu_policies(policies, cupy)

    cold = measure(project_gpu_prepared, prepared, scenario_count, config, cupy)
    for _ in range(args.warmups):
        project_gpu_prepared(prepared, scenario_count, config, cupy)
    warm = [measure(project_gpu_prepared, prepared, scenario_count, config, cupy) for _ in range(args.repetitions)]
    warm_summary = summarize(warm, combinations)
    end_to_end = measure(project_gpu, policies, scenario_count, config)

    cpu_result = project_cpu(policies, scenario_count, config)
    gpu_result = warm[0].value
    difference = np.abs(cpu_result.scenario_pv - gpu_result.scenario_pv)
    max_abs = float(np.max(difference))
    max_rel = float(np.max(difference / np.maximum(np.abs(cpu_result.scenario_pv), 1.0)))
    consistent = bool(np.allclose(cpu_result.scenario_pv, gpu_result.scenario_pv, rtol=1e-12, atol=1e-6))
    if not consistent:
        raise RuntimeError("GPU result exceeds CPU/GPU numerical acceptance criteria")

    timestamp = datetime.now(timezone.utc).isoformat()
    row = {
        "timestamp_utc": timestamp,
        "preset": args.preset,
        "policies": policy_count,
        "scenarios": scenario_count,
        "years": config.years,
        "combinations": combinations,
        "policy_chunk": config.policy_chunk_size,
        "scenario_chunk": config.scenario_chunk_size,
        "repetitions": args.repetitions,
        "compute_median_seconds": f"{warm_summary.median_seconds:.6f}",
        "compute_median_throughput": f"{warm_summary.median_throughput:.3f}",
        "end_to_end_seconds": f"{end_to_end.runtime_seconds:.6f}",
        "max_absolute_difference": f"{max_abs:.12g}",
        "max_relative_difference": f"{max_rel:.12g}",
        "numerically_consistent": consistent,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    exists = args.output.exists()
    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    document = {
        "schema_version": 1,
        "timestamp_utc": timestamp,
        "methodology": {
            "compute_only": "policies resident on GPU; synchronized scenario totals copied to host",
            "end_to_end": "host policies copied to GPU, projection executed, scenario totals copied to host",
            "warmups": args.warmups,
            "repetitions": args.repetitions,
        },
        "workload": row,
        "cold_compute_run": {
            "runtime_seconds": cold.runtime_seconds,
            "throughput": combinations / cold.runtime_seconds,
        },
        "warm_compute_runs": [
            {"runtime_seconds": item.runtime_seconds, "throughput": combinations / item.runtime_seconds}
            for item in warm
        ],
        "warm_compute_summary": asdict(warm_summary),
        "end_to_end_run": {"runtime_seconds": end_to_end.runtime_seconds},
        "gpu": _device_metadata(cupy),
        "environment": collect_environment(),
    }
    args.json_output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
