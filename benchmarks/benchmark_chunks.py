#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from monte_carlo_accelerator.cpu_engine import project_cpu
from monte_carlo_accelerator.metrics import measure, summarize
from monte_carlo_accelerator.models import ProjectionConfig
from monte_carlo_accelerator.policy_generator import generate_policies

DEFAULT_CHUNKS = "512x128,1024x128,1024x256,2048x128,2048x256,2048x512,4096x128,4096x256"


def parse_chunks(value: str) -> list[tuple[int, int]]:
    chunks = []
    try:
        for pair in value.split(","):
            policy, scenario = pair.lower().split("x", 1)
            dimensions = (int(policy), int(scenario))
            if min(dimensions) <= 0:
                raise ValueError
            chunks.append(dimensions)
    except ValueError as error:
        raise argparse.ArgumentTypeError("chunks must look like 2048x256,4096x128") from error
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunk sizes on an identical workload")
    parser.add_argument("--policies", type=int, default=10_000)
    parser.add_argument("--scenarios", type=int, default=1_000)
    parser.add_argument("--chunks", type=parse_chunks, default=parse_chunks(DEFAULT_CHUNKS))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/chunk_sensitivity.csv"))
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("warmups must be >= 0 and repetitions must be >= 1")

    policies = generate_policies(args.policies)
    combinations = args.policies * args.scenarios
    expected = None
    rows = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for policy_chunk, scenario_chunk in args.chunks:
        config = ProjectionConfig(policy_chunk_size=policy_chunk, scenario_chunk_size=scenario_chunk)
        for _ in range(args.warmups):
            project_cpu(policies, args.scenarios, config)
        measurements = [measure(project_cpu, policies, args.scenarios, config) for _ in range(args.repetitions)]
        result = measurements[0].value.scenario_pv
        if expected is None:
            expected = result.copy()
        bitwise_consistent = bool(np.array_equal(result, expected))
        difference = np.abs(result - expected)
        max_absolute_difference = float(np.max(difference))
        relative_difference = difference / np.maximum(np.abs(expected), 1.0)
        max_relative_difference = float(np.max(relative_difference))
        numerically_consistent = bool(np.allclose(result, expected, rtol=1e-12, atol=1e-6))
        if not numerically_consistent:
            raise RuntimeError("chunk configuration exceeds numerical acceptance criteria")
        summary = summarize(measurements, combinations)
        rows.append({
            "timestamp_utc": timestamp,
            "policies": args.policies,
            "scenarios": args.scenarios,
            "combinations": combinations,
            "policy_chunk": policy_chunk,
            "scenario_chunk": scenario_chunk,
            "repetitions": args.repetitions,
            "median_seconds": f"{summary.median_seconds:.6f}",
            "mean_seconds": f"{summary.mean_seconds:.6f}",
            "stdev_seconds": f"{summary.stdev_seconds:.6f}",
            "median_throughput": f"{summary.median_throughput:.3f}",
            "peak_memory_mb": f"{max(item.peak_memory_mb for item in measurements):.3f}",
            "bitwise_consistent": bitwise_consistent,
            "max_absolute_difference": f"{max_absolute_difference:.12g}",
            "max_relative_difference": f"{max_relative_difference:.12g}",
            "numerically_consistent": numerically_consistent,
        })
        print(rows[-1])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
