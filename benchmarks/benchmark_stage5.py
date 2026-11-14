#!/usr/bin/env python3
"""Plan matched Stage 5 benchmarks; run locally only with --execute.

GPU per-run measurements are included in its JSON output. Adaptive runs may
reduce effective chunks; every mode receives the same requested chunks.
Adaptive timings include chunk selection and retries on every call.
End-to-end timings use an initialized CUDA context. Use a fresh output directory.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PRESETS = {
    "small": (1_000, 100),
    "medium": (10_000, 1_000),
    "large": (100_000, 1_000),
    "stress": (250_000, 2_000),
}
MODES = ("cpu_single", "cpu_multicore", "gpu_fixed", "gpu_adaptive")
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--execute", action="store_true", help="execute local runners")
    execution.add_argument("--dry-run", action="store_true", help="plan only (the default)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage5"))
    parser.add_argument("--presets", "--preset", nargs="+", choices=PRESETS, default=["small"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--policies", type=int)
    parser.add_argument("--scenarios", type=int)
    parser.add_argument("--years", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--portfolio-seed", type=int, default=17)
    parser.add_argument("--policy-chunk", type=int, default=4_096)
    parser.add_argument("--scenario-chunk", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.70)
    args = parser.parse_args(argv)
    for name in ("policies", "scenarios", "years", "policy_chunk", "scenario_chunk", "repetitions"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.workers < 2:
        parser.error("workers must be >= 2 for the multicore mode")
    if args.warmups < 0 or args.seed < 0 or args.portfolio_seed < 0:
        parser.error("warmups and seeds must be nonnegative")
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error("gpu-memory-fraction must be > 0 and <= 1")
    if len(set(args.presets)) != len(args.presets):
        parser.error("presets must be unique")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error("output-dir must be a directory")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty; existing results will not be overwritten")
    return args


def build_manifest(args):
    runs = []
    for preset in args.presets:
        policies, scenarios = PRESETS[preset]
        workload = {
            "preset": preset,
            "policies": args.policies if args.policies is not None else policies,
            "scenarios": args.scenarios if args.scenarios is not None else scenarios,
            **{name: getattr(args, name) for name in (
                "years", "seed", "portfolio_seed", "policy_chunk", "scenario_chunk",
                "repetitions", "warmups",
            )},
        }
        common = []
        for name, value in workload.items():
            common.extend([f"--{name.replace('_', '-')}", str(value)])
        for mode in MODES:
            is_cpu = mode.startswith("cpu_")
            runner = "benchmark_stage3_cpu" if is_cpu else "benchmark_stage3_gpu"
            command = [sys.executable, "-m", f"benchmarks.{runner}", *common]
            directory = args.output_dir / preset / mode
            outputs = {
                "output": str(directory / "summary.csv"),
                "json_output": str(directory / "results.json"),
            }
            if is_cpu:
                command.extend(["--workers", str(1 if mode == "cpu_single" else args.workers)])
                outputs["runs_output"] = str(directory / "runs.csv")
            if mode == "gpu_adaptive":
                command.extend(["--adaptive", "--gpu-memory-fraction", str(args.gpu_memory_fraction)])
            for name, path in outputs.items():
                command.extend([f"--{name.replace('_', '-')}", path])
            runs.append({
                "mode": mode,
                "workload": dict(workload),
                "command": command,
                "outputs": outputs,
                "status": "pending" if args.execute else "planned",
                "returncode": None,
            })
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execute": args.execute,
        "cwd": str(REPO_ROOT),
        "status": "pending" if args.execute else "dry_run",
        "runs": runs,
    }


def save_manifest(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    manifest = build_manifest(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "manifest.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    if not args.execute:
        print(json.dumps(manifest, indent=2))
        return 0

    manifest["status"] = "running"
    for run in manifest["runs"]:
        run["status"] = "running"
        save_manifest(path, manifest)
        try:
            subprocess.run(run["command"], cwd=REPO_ROOT, check=True)
        except (subprocess.CalledProcessError, OSError, KeyboardInterrupt) as error:
            returncode = getattr(error, "returncode", 130 if isinstance(error, KeyboardInterrupt) else 1)
            run.update(status="failed", returncode=returncode, error=str(error))
            manifest["status"] = "failed"
            for remaining in manifest["runs"]:
                if remaining["status"] == "pending":
                    remaining["status"] = "skipped"
            save_manifest(path, manifest)
            print(f"Benchmark failed: {run['mode']}; see {path}", file=sys.stderr)
            return returncode if returncode > 0 else 128 - returncode
        run.update(status="succeeded", returncode=0)
        save_manifest(path, manifest)
    manifest["status"] = "succeeded"
    save_manifest(path, manifest)
    print(f"Benchmark manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
