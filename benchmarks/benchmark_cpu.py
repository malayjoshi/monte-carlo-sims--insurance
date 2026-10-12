#!/usr/bin/env python3
import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cpu_engine import project_cpu
from src.metrics import measure
from src.models import ProjectionConfig
from src.policy_generator import generate_policies

PRESETS = {
    "small": (1_000, 100),
    "medium": (10_000, 1_000),
    "large": (100_000, 1_000),
    "stress": (250_000, 2_000),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the chunked NumPy projection")
    parser.add_argument("--preset", choices=PRESETS, default="small")
    parser.add_argument("--policies", type=int)
    parser.add_argument("--scenarios", type=int)
    parser.add_argument("--policy-chunk", type=int, default=2_048)
    parser.add_argument("--scenario-chunk", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("results/cpu_benchmarks.csv"))
    args = parser.parse_args()
    policy_count, scenario_count = PRESETS[args.preset]
    policy_count = args.policies or policy_count
    scenario_count = args.scenarios or scenario_count
    config = ProjectionConfig(policy_chunk_size=args.policy_chunk, scenario_chunk_size=args.scenario_chunk)
    policies = generate_policies(policy_count)
    measured = measure(project_cpu, policies, scenario_count, config)
    combinations = policy_count * scenario_count
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "policies": policy_count,
        "scenarios": scenario_count,
        "years": config.years,
        "combinations": combinations,
        "runtime_seconds": f"{measured.runtime_seconds:.6f}",
        "peak_memory_mb": f"{measured.peak_memory_mb:.3f}",
        "combinations_per_second": f"{combinations / measured.runtime_seconds:.3f}",
        "mean_pv": f"{measured.value.mean_pv:.6f}",
        "seed": config.seed,
        "policy_chunk": config.policy_chunk_size,
        "scenario_chunk": config.scenario_chunk_size,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    exists = args.output.exists()
    with args.output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
