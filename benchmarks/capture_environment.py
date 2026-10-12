#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment import write_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture benchmark host metadata")
    parser.add_argument("--output", type=Path, default=Path("results/environment.json"))
    args = parser.parse_args()
    metadata = write_environment(args.output)
    print(f"wrote {args.output} for {metadata['cpu_model']}")


if __name__ == "__main__":
    main()
