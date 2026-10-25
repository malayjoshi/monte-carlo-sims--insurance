#!/usr/bin/env python3
import argparse
import cProfile
import io
import pstats
from pathlib import Path

from monte_carlo_accelerator.cpu_engine import project_cpu
from monte_carlo_accelerator.models import ProjectionConfig
from monte_carlo_accelerator.policy_generator import generate_policies


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Python-level CPU projection calls")
    parser.add_argument("--policies", type=int, default=10_000)
    parser.add_argument("--scenarios", type=int, default=1_000)
    parser.add_argument("--policy-chunk", type=int, default=2_048)
    parser.add_argument("--scenario-chunk", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("results/profiling"))
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    policies = generate_policies(args.policies)
    config = ProjectionConfig(policy_chunk_size=args.policy_chunk, scenario_chunk_size=args.scenario_chunk)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    binary_path = args.output_dir / "cpu_profile.prof"
    text_path = args.output_dir / "cpu_profile.txt"

    profiler = cProfile.Profile()
    profiler.enable()
    project_cpu(policies, args.scenarios, config)
    profiler.disable()
    profiler.dump_stats(binary_path)

    stream = io.StringIO()
    stream.write(
        "Python-level cProfile report. NumPy native kernels appear as aggregate built-in calls; "
        "this report does not attribute time within compiled kernels.\n\n"
    )
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(args.limit)
    text_path.write_text(stream.getvalue(), encoding="utf-8")
    print(f"wrote {binary_path} and {text_path}")


if __name__ == "__main__":
    main()
