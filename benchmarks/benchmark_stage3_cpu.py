#!/usr/bin/env python3
"""Benchmark the Stage 3 term-life projection on CPU."""

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from monte_carlo_accelerator.actuarial_engine import project_term_life
from monte_carlo_accelerator.environment import collect_environment
from monte_carlo_accelerator.metrics import Measurement, measure, summarize
from monte_carlo_accelerator.models import ActuarialAssumptions, ActuarialProjectionResult
from monte_carlo_accelerator.policy_generator import generate_term_life_portfolio
from monte_carlo_accelerator.portfolio_io import load_term_life_portfolio

PRESETS = {
    "small": (1_000, 100),
    "medium": (10_000, 1_000),
    "large": (100_000, 1_000),
    "stress": (250_000, 2_000),
}

SUMMARY_FIELDS = (
    "timestamp_utc",
    "preset",
    "portfolio_source",
    "policies",
    "scenarios",
    "years",
    "combinations",
    "runtime_seconds",
    "peak_memory_mb",
    "combinations_per_second",
    "mean_premium_pv",
    "mean_claim_pv",
    "mean_expense_pv",
    "mean_net_cashflow_pv",
    "median_net_cashflow_pv",
    "p95_net_cashflow_pv",
    "seed",
    "policy_chunk",
    "scenario_chunk",
)


def append_csv(path: Path, row: dict, fieldnames=SUMMARY_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({name: row[name] for name in fieldnames})


def cashflow_summary(result: ActuarialProjectionResult) -> dict[str, float]:
    """Return scenario-level cash-flow statistics outside the timed region."""
    net = result.net_cashflow_pv
    return {
        "mean_premium_pv": float(np.mean(result.premium_pv)),
        "mean_claim_pv": float(np.mean(result.claim_pv)),
        "mean_expense_pv": float(np.mean(result.expense_pv)),
        "mean_net_cashflow_pv": float(np.mean(net)),
        "median_net_cashflow_pv": float(np.median(net)),
        "p95_net_cashflow_pv": float(np.percentile(net, 95)),
    }


def results_equal(first: ActuarialProjectionResult, second: ActuarialProjectionResult) -> bool:
    """Require exact repeatability for identical inputs and chunking."""
    return all(
        np.array_equal(left, right)
        for left, right in (
            (first.premium_pv, second.premium_pv),
            (first.claim_pv, second.claim_pv),
            (first.expense_pv, second.expense_pv),
        )
    )


def _measurement_document(item: Measurement, combinations: int) -> dict[str, float]:
    return {
        "runtime_seconds": item.runtime_seconds,
        "throughput": combinations / item.runtime_seconds,
        "peak_memory_mb": item.peak_memory_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the Stage 3 CPU actuarial projection")
    parser.add_argument("--preset", choices=PRESETS, default="small")
    parser.add_argument("--policies", type=int)
    parser.add_argument("--scenarios", type=int)
    parser.add_argument("--portfolio-csv", type=Path)
    parser.add_argument("--years", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--portfolio-seed", type=int, default=17)
    parser.add_argument("--policy-chunk", type=int, default=2_048)
    parser.add_argument("--scenario-chunk", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/stage3_cpu_benchmarks.csv"))
    parser.add_argument(
        "--runs-output", type=Path, default=Path("results/stage3_cpu_benchmark_runs.csv")
    )
    parser.add_argument(
        "--json-output", type=Path, default=Path("results/stage3_cpu_benchmark_latest.json")
    )
    args = parser.parse_args()

    if args.warmups < 0 or args.repetitions < 1:
        parser.error("warmups must be >= 0 and repetitions must be >= 1")
    if args.years <= 0 or args.policy_chunk <= 0 or args.scenario_chunk <= 0:
        parser.error("years and chunk sizes must be positive")
    if args.portfolio_csv is not None and args.policies is not None:
        parser.error("--policies cannot be combined with --portfolio-csv")

    preset_policies, preset_scenarios = PRESETS[args.preset]
    scenario_count = args.scenarios or preset_scenarios
    if scenario_count <= 0:
        parser.error("scenarios must be positive")

    if args.portfolio_csv is None:
        policy_count = args.policies or preset_policies
        if policy_count <= 0:
            parser.error("policies must be positive")
        setup = measure(generate_term_life_portfolio, policy_count, args.portfolio_seed)
        portfolio_source = "generated"
    else:
        setup = measure(load_term_life_portfolio, args.portfolio_csv)
        policy_count = len(setup.value)
        portfolio_source = "csv"

    assumptions = ActuarialAssumptions(
        projection_years=args.years,
        seed=args.seed,
        policy_chunk_size=args.policy_chunk,
        scenario_chunk_size=args.scenario_chunk,
    )
    policies = setup.value
    combinations = policy_count * scenario_count
    timestamp = datetime.now(timezone.utc).isoformat()

    cold = measure(project_term_life, policies, scenario_count, assumptions)
    for _ in range(args.warmups):
        project_term_life(policies, scenario_count, assumptions)
    measured = [
        measure(project_term_life, policies, scenario_count, assumptions)
        for _ in range(args.repetitions)
    ]
    reference = measured[0].value
    if any(not results_equal(reference, item.value) for item in measured[1:]):
        raise RuntimeError("deterministic repetitions produced different cash-flow results")

    summary = summarize(measured, combinations)
    cashflows = cashflow_summary(reference)
    peak_memory = max(item.peak_memory_mb for item in measured)
    summary_row = {
        "timestamp_utc": timestamp,
        "preset": args.preset,
        "portfolio_source": portfolio_source,
        "policies": policy_count,
        "scenarios": scenario_count,
        "years": assumptions.projection_years,
        "combinations": combinations,
        "runtime_seconds": f"{summary.median_seconds:.6f}",
        "peak_memory_mb": f"{peak_memory:.3f}",
        "combinations_per_second": f"{summary.median_throughput:.3f}",
        **{name: f"{value:.6f}" for name, value in cashflows.items()},
        "seed": assumptions.seed,
        "policy_chunk": assumptions.policy_chunk_size,
        "scenario_chunk": assumptions.scenario_chunk_size,
    }
    append_csv(args.output, summary_row)

    run_fields = (*SUMMARY_FIELDS, "run_type", "run_number")
    runs = [("cold", 0, cold)] + [
        ("warm", number, item) for number, item in enumerate(measured, 1)
    ]
    for run_type, run_number, item in runs:
        row = dict(summary_row)
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
            "timed_scope": "project_term_life only; setup, summaries, metadata, and serialization excluded",
            "cold_definition": "first projection in this process after portfolio setup",
            "warmup_runs_after_cold": args.warmups,
            "measured_warm_runs": args.repetitions,
            "memory_method": "peak sampled Linux /proc/self/status VmRSS at 5 ms intervals",
            "throughput_unit": "policy-scenario combinations per second",
        },
        "portfolio_setup": {
            "source": portfolio_source,
            "path": str(args.portfolio_csv) if args.portfolio_csv is not None else None,
            "runtime_seconds": setup.runtime_seconds,
            "peak_memory_mb": setup.peak_memory_mb,
            "generator_seed": args.portfolio_seed if args.portfolio_csv is None else None,
        },
        "workload": {
            "preset": args.preset,
            "policies": policy_count,
            "scenarios": scenario_count,
            "combinations": combinations,
            "policy_scenario_years": combinations * assumptions.projection_years,
            "assumptions": asdict(assumptions),
        },
        "cold_run": _measurement_document(cold, combinations),
        "warm_runs": [_measurement_document(item, combinations) for item in measured],
        "summary": asdict(summary),
        "cashflows": cashflows,
        "environment": collect_environment(),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "portfolio_setup": document["portfolio_setup"],
                "cold_run": document["cold_run"],
                "warm_summary": document["summary"],
                "cashflows": document["cashflows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
