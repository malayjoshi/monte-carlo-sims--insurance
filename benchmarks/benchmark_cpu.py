#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cpu_engine import project_cpu
from src.environment import collect_environment
from src.metrics import measure, summarize
from src.models import ProjectionConfig
from src.policy_generator import generate_policies

PRESETS = {
    "small": (1_000, 100),
    "medium": (10_000, 1_000),
    "large": (100_000, 1_000),
    "stress": (250_000, 2_000),
}
LEGACY_FIELDS = (
    "timestamp_utc", "preset", "policies", "scenarios", "years", "combinations",
    "runtime_seconds", "peak_memory_mb", "combinations_per_second", "mean_pv",
    "seed", "policy_chunk", "scenario_chunk",
)


def append_csv(path: Path, row: dict, fieldnames=LEGACY_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({name: row[name] for name in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeatable benchmark for the chunked NumPy projection")
    parser.add_argument("--preset", choices=PRESETS, default="small")
    parser.add_argument("--policies", type=int)
    parser.add_argument("--scenarios", type=int)
    parser.add_argument("--policy-chunk", type=int, default=2_048)
    parser.add_argument("--scenario-chunk", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/cpu_benchmarks.csv"))
    parser.add_argument("--runs-output", type=Path, default=Path("results/cpu_benchmark_runs.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("results/cpu_benchmark_latest.json"))
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("warmups must be >= 0 and repetitions must be >= 1")
    policy_count, scenario_count = PRESETS[args.preset]
    policy_count = args.policies or policy_count
    scenario_count = args.scenarios or scenario_count
    config = ProjectionConfig(policy_chunk_size=args.policy_chunk, scenario_chunk_size=args.scenario_chunk)

    # Imports, configuration, and policy generation are outside all timed regions.
    policies = generate_policies(policy_count)
    combinations = policy_count * scenario_count
    timestamp = datetime.now(timezone.utc).isoformat()
    cold = measure(project_cpu, policies, scenario_count, config)
    for _ in range(args.warmups):
        project_cpu(policies, scenario_count, config)
    measured = [measure(project_cpu, policies, scenario_count, config) for _ in range(args.repetitions)]
    summary = summarize(measured, combinations)
    reference = measured[0].value.scenario_pv
    if any(not (item.value.scenario_pv == reference).all() for item in measured[1:]):
        raise RuntimeError("deterministic repetitions produced different results")

    peak_memory = max(item.peak_memory_mb for item in measured)
    legacy_row = {
        "timestamp_utc": timestamp,
        "preset": args.preset,
        "policies": policy_count,
        "scenarios": scenario_count,
        "years": config.years,
        "combinations": combinations,
        "runtime_seconds": f"{summary.median_seconds:.6f}",
        "peak_memory_mb": f"{peak_memory:.3f}",
        "combinations_per_second": f"{summary.median_throughput:.3f}",
        "mean_pv": f"{measured[0].value.mean_pv:.6f}",
        "seed": config.seed,
        "policy_chunk": config.policy_chunk_size,
        "scenario_chunk": config.scenario_chunk_size,
    }
    append_csv(args.output, legacy_row)

    run_fields = (*LEGACY_FIELDS, "run_type", "run_number")
    runs = [("cold", 0, cold)] + [("warm", index, value) for index, value in enumerate(measured, 1)]
    for run_type, run_number, item in runs:
        row = dict(legacy_row)
        row.update(
            runtime_seconds=f"{item.runtime_seconds:.6f}",
            peak_memory_mb=f"{item.peak_memory_mb:.3f}",
            combinations_per_second=f"{combinations / item.runtime_seconds:.3f}",
            run_type=run_type,
            run_number=run_number,
        )
        append_csv(args.runs_output, row, run_fields)

    document = {
        "schema_version": 1,
        "timestamp_utc": timestamp,
        "methodology": {
            "timed_scope": "project_cpu only; imports, policy generation, metadata, and serialization excluded",
            "cold_definition": "first project_cpu invocation in this process after policy generation",
            "warmup_runs_after_cold": args.warmups,
            "measured_warm_runs": args.repetitions,
            "memory_method": "peak sampled Linux /proc/self/status VmRSS at 5 ms intervals",
        },
        "workload": {
            "preset": args.preset, "policies": policy_count, "scenarios": scenario_count,
            "years": config.years, "combinations": combinations, "seed": config.seed,
            "policy_chunk": config.policy_chunk_size, "scenario_chunk": config.scenario_chunk_size,
        },
        "cold_run": {
            "runtime_seconds": cold.runtime_seconds,
            "throughput": combinations / cold.runtime_seconds,
            "peak_memory_mb": cold.peak_memory_mb,
        },
        "warm_runs": [
            {
                "runtime_seconds": item.runtime_seconds,
                "throughput": combinations / item.runtime_seconds,
                "peak_memory_mb": item.peak_memory_mb,
            }
            for item in measured
        ],
        "summary": asdict(summary),
        "mean_pv": measured[0].value.mean_pv,
        "environment": collect_environment(),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cold_run": document["cold_run"], "warm_summary": document["summary"]}, indent=2))


if __name__ == "__main__":
    main()
