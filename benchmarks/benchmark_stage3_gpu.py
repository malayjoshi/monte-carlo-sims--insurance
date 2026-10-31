#!/usr/bin/env python3
"""Benchmark the Stage 3 CuPy projection against its CPU reference."""

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from monte_carlo_accelerator.actuarial_engine import project_term_life
from monte_carlo_accelerator.actuarial_gpu_engine import (
    prepare_gpu_term_life_policies,
    project_term_life_gpu,
    project_term_life_gpu_prepared,
)
from monte_carlo_accelerator.environment import collect_environment
from monte_carlo_accelerator.gpu_engine import _load_cupy
from monte_carlo_accelerator.metrics import measure, summarize
from monte_carlo_accelerator.models import ActuarialAssumptions, ActuarialProjectionResult
from monte_carlo_accelerator.policy_generator import generate_term_life_portfolio

PRESETS = {
    "small": (1_000, 100),
    "medium": (10_000, 1_000),
    "large": (100_000, 1_000),
    "stress": (250_000, 2_000),
}


def device_metadata(cupy) -> dict:
    """Return stable identifiers for the active CUDA device and runtime."""
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


def cashflow_summary(result: ActuarialProjectionResult) -> dict[str, float]:
    """Summarize scenario cash flows outside every timed region."""
    net = result.net_cashflow_pv
    return {
        "mean_premium_pv": float(np.mean(result.premium_pv)),
        "mean_claim_pv": float(np.mean(result.claim_pv)),
        "mean_expense_pv": float(np.mean(result.expense_pv)),
        "mean_net_cashflow_pv": float(np.mean(net)),
        "median_net_cashflow_pv": float(np.median(net)),
        "p95_net_cashflow_pv": float(np.percentile(net, 95)),
    }


def component_parity(
    cpu: ActuarialProjectionResult,
    gpu: ActuarialProjectionResult,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-6,
) -> dict:
    """Return component-level numerical differences and acceptance status."""
    components = {}
    for name, cpu_values, gpu_values in (
        ("premium_pv", cpu.premium_pv, gpu.premium_pv),
        ("claim_pv", cpu.claim_pv, gpu.claim_pv),
        ("expense_pv", cpu.expense_pv, gpu.expense_pv),
        ("net_cashflow_pv", cpu.net_cashflow_pv, gpu.net_cashflow_pv),
    ):
        difference = np.abs(cpu_values - gpu_values)
        components[name] = {
            "max_absolute_difference": float(np.max(difference)),
            "max_relative_difference": float(
                np.max(difference / np.maximum(np.abs(cpu_values), 1.0))
            ),
            "consistent": bool(np.allclose(cpu_values, gpu_values, rtol=rtol, atol=atol)),
        }
    return {
        "rtol": rtol,
        "atol": atol,
        "consistent": all(item["consistent"] for item in components.values()),
        "components": components,
    }


def gpu_memory_snapshot(cupy) -> dict[str, int]:
    """Capture device and CuPy pool memory without resetting allocator state."""
    free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    pool = cupy.get_default_memory_pool()
    return {
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
        "pool_used_bytes": int(pool.used_bytes()),
        "pool_total_bytes": int(pool.total_bytes()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Stage 3 CuPy against the CPU reference")
    parser.add_argument("--preset", choices=PRESETS, default="small")
    parser.add_argument("--policies", type=int)
    parser.add_argument("--scenarios", type=int)
    parser.add_argument("--years", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--portfolio-seed", type=int, default=17)
    parser.add_argument("--policy-chunk", type=int, default=4_096)
    parser.add_argument("--scenario-chunk", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/stage3_gpu_benchmarks.csv"))
    parser.add_argument(
        "--json-output", type=Path, default=Path("results/stage3_gpu_benchmark_latest.json")
    )
    args = parser.parse_args()

    if args.warmups < 0 or args.repetitions < 1:
        parser.error("warmups must be >= 0 and repetitions must be >= 1")
    if args.years <= 0 or args.policy_chunk <= 0 or args.scenario_chunk <= 0:
        parser.error("years and chunk sizes must be positive")

    policy_count, scenario_count = PRESETS[args.preset]
    policy_count = args.policies or policy_count
    scenario_count = args.scenarios or scenario_count
    if policy_count <= 0 or scenario_count <= 0:
        parser.error("policies and scenarios must be positive")

    combinations = policy_count * scenario_count
    assumptions = ActuarialAssumptions(
        projection_years=args.years,
        seed=args.seed,
        policy_chunk_size=args.policy_chunk,
        scenario_chunk_size=args.scenario_chunk,
    )
    policies = generate_term_life_portfolio(policy_count, args.portfolio_seed)

    cuda_startup = measure(_load_cupy)
    cupy = cuda_startup.value
    memory_before_prepare = gpu_memory_snapshot(cupy)
    policy_prepare = measure(prepare_gpu_term_life_policies, policies, cupy)
    prepared = policy_prepare.value
    memory_after_prepare = gpu_memory_snapshot(cupy)

    cold = measure(project_term_life_gpu_prepared, prepared, scenario_count, assumptions, cupy)
    for _ in range(args.warmups):
        project_term_life_gpu_prepared(prepared, scenario_count, assumptions, cupy)
    warm = [
        measure(project_term_life_gpu_prepared, prepared, scenario_count, assumptions, cupy)
        for _ in range(args.repetitions)
    ]
    warm_summary = summarize(warm, combinations)
    end_to_end = measure(project_term_life_gpu, policies, scenario_count, assumptions)
    memory_after_benchmark = gpu_memory_snapshot(cupy)

    cpu_result = project_term_life(policies, scenario_count, assumptions)
    parity = component_parity(cpu_result, warm[0].value)
    if not parity["consistent"]:
        raise RuntimeError("Stage 3 GPU result exceeds CPU/GPU numerical acceptance criteria")

    timestamp = datetime.now(timezone.utc).isoformat()
    row = {
        "timestamp_utc": timestamp,
        "preset": args.preset,
        "policies": policy_count,
        "scenarios": scenario_count,
        "years": assumptions.projection_years,
        "combinations": combinations,
        "policy_chunk": assumptions.policy_chunk_size,
        "scenario_chunk": assumptions.scenario_chunk_size,
        "repetitions": args.repetitions,
        "cuda_startup_seconds": f"{cuda_startup.runtime_seconds:.6f}",
        "policy_prepare_seconds": f"{policy_prepare.runtime_seconds:.6f}",
        "compute_median_seconds": f"{warm_summary.median_seconds:.6f}",
        "compute_median_throughput": f"{warm_summary.median_throughput:.3f}",
        "end_to_end_seconds": f"{end_to_end.runtime_seconds:.6f}",
        "gpu_pool_used_bytes": memory_after_benchmark["pool_used_bytes"],
        "numerically_consistent": parity["consistent"],
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
            "compute_only": "policies resident on GPU; three synchronized scenario arrays copied to host",
            "end_to_end": "host policies transferred, projection executed, and three scenario arrays copied to host",
            "cpu_reference": "untimed CPU projection with identical inputs, assumptions, seed, and chunks",
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "throughput_unit": "policy-scenario combinations per second",
        },
        "workload": {
            **row,
            "policy_scenario_years": combinations * assumptions.projection_years,
            "assumptions": asdict(assumptions),
        },
        "cuda_startup": {"runtime_seconds": cuda_startup.runtime_seconds},
        "policy_prepare": {"runtime_seconds": policy_prepare.runtime_seconds},
        "cold_compute_run": {
            "runtime_seconds": cold.runtime_seconds,
            "throughput": combinations / cold.runtime_seconds,
        },
        "warm_compute_runs": [
            {
                "runtime_seconds": item.runtime_seconds,
                "throughput": combinations / item.runtime_seconds,
            }
            for item in warm
        ],
        "warm_compute_summary": asdict(warm_summary),
        "end_to_end_run": {"runtime_seconds": end_to_end.runtime_seconds},
        "cashflows": cashflow_summary(warm[0].value),
        "parity": parity,
        "gpu_memory": {
            "before_prepare": memory_before_prepare,
            "after_prepare": memory_after_prepare,
            "after_benchmark": memory_after_benchmark,
        },
        "gpu": device_metadata(cupy),
        "environment": collect_environment(),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
