"""Run routine formatting and numerical checks; build docs/distributions separately."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Include actual CUDA integration tests")
    parser.add_argument("paths", nargs="*", default=["tests/python"], help="Targeted test paths")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    for command in (
        [sys.executable, "-m", "ruff", "check", "python", "tests/python", "tools"],
        [sys.executable, "-m", "ruff", "format", "--check", "python", "tests/python", "tools"],
        [sys.executable, "-m", "pytest", *args.paths, "-q", *(["--run-gpu"] if args.cuda else [])],
    ):
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
