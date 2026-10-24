#!/usr/bin/env python3
import argparse
from pathlib import Path

from monte_carlo_accelerator.environment import write_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture benchmark host metadata")
    parser.add_argument("--output", type=Path, default=Path("results/environment.json"))
    args = parser.parse_args()
    metadata = write_environment(args.output)
    print(f"wrote {args.output} for {metadata['cpu_model']}")


if __name__ == "__main__":
    main()
