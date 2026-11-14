"""Validate a Stage 5 manifest and write a local Markdown report and JSON summary.

Pricing is optional: all five price flags are required together. --price-verified
attests that the supplied dated source was checked by the caller; this offline
tool cannot verify a provider's price. Rates cover the same shared GPU host.
"""

import argparse
import json
import math
import re
import statistics
from datetime import date
from pathlib import Path

MODES = ("cpu_single", "cpu_multicore", "gpu_fixed", "gpu_adaptive")
COMPONENTS = ("premium_pv", "claim_pv", "expense_pv", "net_cashflow_pv")
ENVIRONMENT_FIELDS = (
    "cpu_model", "physical_cpu_cores", "logical_cpu_count", "total_ram",
    "linux_distribution", "kernel", "platform", "python_version",
    "numpy_version", "blas_configuration", "threading_environment", "git_commit_sha",
)
LIMITATIONS = [
    "Synthetic generated portfolios only; these observations do not establish production performance.",
    "CPU RSS is sampled process memory; GPU memory values are snapshots, not peaks. They are not comparable peak-memory measures.",
    "Costs, when supplied, are shared-host compute costs: CPU and GPU use the same priced GPU host. Storage, idle time, startup, and other charges are excluded.",
    "Prepared GPU projection excludes host policy transfer; host-input GPU timings include transfer and projection with CUDA already initialized. CPU timings exclude portfolio generation. Neither scope is process end-to-end latency.",
    "Crossover brackets describe adjacent observed workloads only, without interpolation or a universal crossover claim.",
    "A validated report alone does not establish a complete release.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"{label} must be finite and positive")
    return value


def check_parity(parity):
    require(isinstance(parity, dict) and parity.get("consistent") is True,
            "missing or failed parity")
    components = parity.get("components", {})
    require(all(components.get(name, {}).get("consistent") is True for name in COMPONENTS),
            "missing or failed component parity")
    require(parity.get("rtol") == 1e-12 and parity.get("atol") == 1e-6,
            "unexpected parity tolerances")
    for component in components.values():
        for key in ("max_absolute_difference", "max_relative_difference"):
            value = component[key]
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    "invalid parity difference")


def timing(runs, summary, repetitions):
    require(len(runs) == repetitions and summary["count"] == repetitions,
            "repetition count mismatch")
    seconds = [positive(run["runtime_seconds"], "runtime") for run in runs]
    median = statistics.median(seconds)
    require(math.isclose(positive(summary["median_seconds"], "median"), median,
                         rel_tol=1e-9), "summary median does not match per-run results")
    return {"count": len(seconds), "runs": runs, "median_seconds": median,
            "min_seconds": min(seconds), "max_seconds": max(seconds)}


def validate_price(price):
    if price is None:
        return
    positive(price["hourly_usd"], "hourly USD price")
    require(price.get("verified") is True, "price must be explicitly verified by caller")
    require(bool(price["source"].strip()) and bool(price["region"].strip()),
            "price source and region are required")
    require(date.fromisoformat(price["date"]) <= date.today(), "price date is in the future")


def build_report(manifest_path, price=None):
    """Read every results.json; reject incomplete or incomparable evidence."""
    validate_price(price)
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["schema_version"] == 1 and manifest["execute"] is True
            and manifest["status"] == "succeeded", "manifest must be a successful execution")
    require(bool(manifest["runs"]), "manifest contains no runs")
    groups = {}
    environment = gpu = None
    seen_paths = set()
    for entry in manifest["runs"]:
        mode, expected = entry["mode"], entry["workload"]
        require(mode in MODES and entry["status"] == "succeeded" and entry["returncode"] == 0,
                "every manifest run must succeed and use a recognized mode")
        path = Path(entry["outputs"]["json_output"])
        if not path.is_absolute():
            path = Path(manifest["cwd"]) / path
        if not path.exists():
            preset = expected["preset"]
            require(preset in ("small", "medium", "large", "stress") and
                    path.parts[-3:] == (preset, mode, "results.json"),
                    "missing result does not match portable preset/mode/results.json layout")
            path = manifest_path.parent / preset / mode / "results.json"
            require(path.resolve().is_relative_to(manifest_path.parent),
                    "portable result escapes manifest directory")
        path = path.resolve()
        require(path not in seen_paths, "duplicate results path")
        seen_paths.add(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        require(result["schema_version"] == 3, "expected Stage 3 schema version 3")
        actual = result["workload"]
        assumptions = actual["assumptions"]
        for key in ("policies", "scenarios", "years", "policy_chunk", "scenario_chunk", "repetitions"):
            require(type(expected[key]) is int and expected[key] > 0, f"invalid {key}")
        for key in ("seed", "portfolio_seed", "warmups"):
            require(type(expected[key]) is int and expected[key] >= 0, f"invalid {key}")
        for key in ("preset", "policies", "scenarios"):
            require(actual[key] == expected[key], f"workload {key} mismatch")
        for key, target in (("years", "projection_years"), ("seed", "seed"),
                            ("policy_chunk", "policy_chunk_size"),
                            ("scenario_chunk", "scenario_chunk_size")):
            require(assumptions[target] == expected[key], f"workload {key} mismatch")
        combinations = expected["policies"] * expected["scenarios"]
        require(actual["combinations"] == combinations and
                actual["policy_scenario_years"] == combinations * expected["years"],
                "workload combinations mismatch")
        source = result.get("portfolio_setup", {})
        source_kind = source.get("source", actual.get("portfolio_source"))
        if source_kind is None and mode.startswith("gpu_"):
            require(entry["command"][1:3] == ["-m", "benchmarks.benchmark_stage3_gpu"]
                    and "--portfolio-csv" not in entry["command"], "unknown GPU portfolio source")
            source_kind = "generated"
        require(source_kind == "generated"
                and source.get("generator_seed", actual.get("portfolio_seed")) == expected["portfolio_seed"],
                "generated portfolio source/seed missing or mismatched")
        current_environment = result["environment"]
        require(all(key in current_environment for key in ENVIRONMENT_FIELDS),
                "incomplete environment metadata")
        current_environment = {key: value for key, value in current_environment.items()
                               if key != "captured_at_utc"}
        require(re.fullmatch(r"[0-9a-fA-F]{40}", current_environment["git_commit_sha"] or "") is not None,
                "missing or invalid git SHA")
        if environment is None:
            environment = current_environment
        require(current_environment == environment, "source git SHA/environment mismatch")
        group = groups.setdefault(expected["preset"], {
            "workload": expected, "assumptions": assumptions, "modes": {},
        })
        require(group["workload"] == expected and group["assumptions"] == assumptions,
                "mismatched workload/assumptions within preset")
        require(mode not in group["modes"], "duplicate mode within preset")
        repetitions = expected["repetitions"]
        methodology = result["methodology"]
        scopes = {}
        if mode.startswith("cpu_"):
            workers = methodology["workers"]
            require(type(workers) is int and (workers == 1 if mode == "cpu_single" else workers >= 2),
                    "CPU worker mode mismatch")
            require(methodology["measured_warm_runs"] == repetitions and
                    methodology["warmup_runs_after_cold"] == expected["warmups"],
                    "CPU methodology repetition/warmup mismatch")
            positive(result["cold_run"]["runtime_seconds"], "cold runtime")
            scopes["warm_host_input"] = timing(result["warm_runs"], result["summary"], repetitions)
            if mode == "cpu_multicore" or result.get("parity") is not None:
                parity = result["parity"]
                check_parity(parity)
                require(len(parity["warm_runs"]) == repetitions, "CPU parity repetition mismatch")
                for item in [parity["cold_run"], *parity["warm_runs"]]:
                    check_parity(item)
        else:
            require(methodology["repetitions"] == repetitions and
                    methodology["warmups"] == expected["warmups"],
                    "GPU methodology repetition/warmup mismatch")
            require(bool(result.get("gpu")), "missing GPU metadata")
            if gpu is None:
                gpu = result["gpu"]
            require(result["gpu"] == gpu, "GPU environment mismatch")
            check_parity(result["parity"])
            all_runs = [result["cold_compute_run"], *result["warm_compute_runs"], *result["end_to_end_runs"]]
            for item in all_runs:
                positive(item["runtime_seconds"], "GPU runtime")
                check_parity(item["parity"])
                require((item.get("adaptive_gpu") is not None) == (mode == "gpu_adaptive"),
                        "adaptive mode/per-run report mismatch")
                if mode == "gpu_adaptive":
                    adaptive = item["adaptive_gpu"]
                    for dimension in ("policy", "scenario"):
                        requested = expected[f"{dimension}_chunk"]
                        effective = adaptive[f"effective_{dimension}_chunk"]
                        require(adaptive[f"requested_{dimension}_chunk"] == requested and
                                type(effective) is int and 0 < effective <= requested,
                                "adaptive requested/effective chunks mismatch")
            scopes["warm_prepared_projection"] = timing(
                result["warm_compute_runs"], result["warm_compute_summary"], repetitions)
            scopes["warm_host_input"] = timing(
                result["end_to_end_runs"], result["end_to_end_summary"], repetitions)
        group["modes"][mode] = {"result_path": str(path), "scopes": scopes,
                                "methodology": methodology, "parity": result.get("parity")}
    for group in groups.values():
        require(set(group["modes"]) == set(MODES), "each preset requires all four modes")
        combinations = group["workload"]["policies"] * group["workload"]["scenarios"]
        baseline = group["modes"]["cpu_single"]["scopes"]["warm_host_input"]["median_seconds"]
        for mode in group["modes"].values():
            for scope in mode["scopes"].values():
                seconds = scope["median_seconds"]
                scope["speedup_vs_cpu_single"] = baseline / seconds
                scope["combinations_per_second"] = combinations / seconds
                scope["shared_host_usd_per_billion_combinations"] = (
                    price["hourly_usd"] * seconds / 3600 * 1e9 / combinations if price else None)
    brackets = []
    ordered = sorted(groups.values(), key=lambda group: group["workload"]["policies"] * group["workload"]["scenarios"])
    for lower, upper in zip(ordered, ordered[1:]):
        left, right = lower["workload"], upper["workload"]
        same = all(left[key] == right[key] for key in left if key not in ("preset", "policies", "scenarios"))
        same = same and lower["assumptions"] == upper["assumptions"]
        if not same or left["policies"] * left["scenarios"] >= right["policies"] * right["scenarios"]:
            continue
        for mode in MODES[1:]:
            for scope in lower["modes"][mode]["scopes"]:
                before = lower["modes"][mode]["scopes"][scope]["speedup_vs_cpu_single"]
                after = upper["modes"][mode]["scopes"][scope]["speedup_vs_cpu_single"]
                if (before - 1) * (after - 1) < 0:
                    brackets.append({"mode": mode, "scope": scope, "lower_workload": left,
                                     "upper_workload": right, "lower_speedup": before, "upper_speedup": after})
    return {"schema_version": 1, "manifest_path": str(manifest_path), "environment": environment,
            "gpu": gpu, "price": price, "presets": groups, "observed_crossover_brackets": brackets,
            "limitations": LIMITATIONS}


def markdown(report):
    lines = ["# Stage 5 benchmark report", "", f"Source git SHA: `{report['environment']['git_commit_sha']}`", "",
             "Speedups use the same workload's warmed cpu_single projection. Prepared GPU ratios compare different input-residency scopes.", "",
             "| Preset | Mode | Scope | Runs | Median seconds | CPU speedup | Shared-host USD / billion combinations |",
             "|---|---|---|---:|---:|---:|---:|"]
    for preset, group in report["presets"].items():
        for mode, result in group["modes"].items():
            for name, scope in result["scopes"].items():
                cost = scope["shared_host_usd_per_billion_combinations"]
                cost = "not supplied" if cost is None else f"{cost:.6g}"
                lines.append(f"| {preset} | {mode} | {name} | {scope['count']} | {scope['median_seconds']:.6g} | {scope['speedup_vs_cpu_single']:.4g} | {cost} |")
    if report["price"]:
        price = report["price"]
        lines.extend(["", f"Caller-verified shared GPU host price: USD {price['hourly_usd']}/hour; date {price['date']}; region {price['region']}; source: {price['source']}."])
    lines.extend(["", "## Observed crossover brackets", ""])
    for bracket in report["observed_crossover_brackets"]:
        lower, upper = bracket["lower_workload"], bracket["upper_workload"]
        lines.append(f"- {bracket['mode']} / {bracket['scope']}: {lower['preset']} ({lower['policies']} policies × {lower['scenarios']} scenarios, {bracket['lower_speedup']:.4g}×) to {upper['preset']} ({upper['policies']} policies × {upper['scenarios']} scenarios, {bracket['upper_speedup']:.4g}×).")
    if not report["observed_crossover_brackets"]:
        lines.append("No crossover bracket observed in comparable sampled workloads.")
    lines.extend(["", "## Limitations", "", *[f"- {item}" for item in report["limitations"]]])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Markdown output")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--hourly-price", type=float)
    parser.add_argument("--price-date")
    parser.add_argument("--price-source")
    parser.add_argument("--price-region")
    parser.add_argument("--price-verified", action="store_true")
    args = parser.parse_args(argv)
    values = (args.hourly_price, args.price_date, args.price_source, args.price_region)
    price = None
    if any(value is not None for value in values) or args.price_verified:
        if not all(value is not None for value in values) or not args.price_verified:
            parser.error("pricing requires hourly-price, price-date, price-source, price-region, and price-verified")
        price = dict(zip(("hourly_usd", "date", "source", "region"), values), verified=True)
    try:
        report = build_report(args.manifest, price)
        require(args.output.resolve() != args.json_output.resolve(), "outputs must differ")
        require(not args.output.exists() and not args.json_output.exists(), "outputs must not already exist")
        rendered = markdown(report)
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        for path, content in ((args.output, rendered), (args.json_output, serialized)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"invalid benchmark evidence: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
