#!/usr/bin/env python3
"""Validate persistent warm-worker latency and numerical consistency."""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cpu_engine import project_cpu
from src.environment import collect_environment
from src.gpu_worker import WarmGPUWorker
from src.metrics import measure, summarize
from src.models import ProjectionConfig
from src.policy_generator import generate_policies

PRESETS = {
    "small": (1_000, 100),
    "medium": (10_000, 1_000),
    "large": (100_000, 1_000),
    "stress": (250_000, 2_000),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one persistent warm GPU worker")
    parser.add_argument("--presets", nargs="+", choices=PRESETS, default=["small", "medium"])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/gpu_worker_benchmarks.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("results/gpu_worker_benchmark.json"))
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be >= 1")

    started = perf_counter()
    worker = WarmGPUWorker()
    startup_seconds = perf_counter() - started
    rows = []
    details = []

    for preset in args.presets:
        policy_count, scenario_count = PRESETS[preset]
        combinations = policy_count * scenario_count
        config = ProjectionConfig(policy_chunk_size=4_096, scenario_chunk_size=128)
        policies = generate_policies(policy_count)

        prepare_started = perf_counter()
        prepared = worker.prepare(policies)
        prepare_seconds = perf_counter() - prepare_started
        runs = [
            measure(worker.project_prepared, prepared, scenario_count, config)
            for _ in range(args.repetitions)
        ]
        summary = summarize(runs, combinations)
        end_to_end = measure(worker.project, policies, scenario_count, config)

        cpu_result = project_cpu(policies, scenario_count, config)
        difference = np.abs(cpu_result.scenario_pv - runs[0].value.scenario_pv)
        max_abs = float(np.max(difference))
        max_rel = float(np.max(difference / np.maximum(np.abs(cpu_result.scenario_pv), 1.0)))
        consistent = bool(
            np.allclose(cpu_result.scenario_pv, runs[0].value.scenario_pv, rtol=1e-12, atol=1e-6)
        )
        if not consistent:
            raise RuntimeError(f"{preset} warm-worker result exceeds numerical acceptance criteria")

        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "preset": preset,
            "policies": policy_count,
            "scenarios": scenario_count,
            "combinations": combinations,
            "worker_startup_seconds": f"{startup_seconds:.6f}",
            "policy_prepare_seconds": f"{prepare_seconds:.6f}",
            "warm_median_seconds": f"{summary.median_seconds:.6f}",
            "warm_median_throughput": f"{summary.median_throughput:.3f}",
            "warm_end_to_end_seconds": f"{end_to_end.runtime_seconds:.6f}",
            "max_absolute_difference": f"{max_abs:.12g}",
            "max_relative_difference": f"{max_rel:.12g}",
            "numerically_consistent": consistent,
        }
        rows.append(row)
        details.append(
            {
                "workload": row,
                "warm_runs": [item.runtime_seconds for item in runs],
                "warm_summary": summary.__dict__,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    document = {
        "schema_version": 1,
        "worker_startup_seconds": startup_seconds,
        "repetitions": args.repetitions,
        "presets": details,
        "environment": collect_environment(),
    }
    args.json_output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
