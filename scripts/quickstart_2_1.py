#!/usr/bin/env python3
"""Smoke-test the bundled RA-SAC 2-1 actor on 10 fixed demand paths."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "source_snapshot/native"


def main() -> None:
    command = [
        sys.executable,
        "evaluate_manifest_three_seed_project_checkpoint_common1000_20260923.py",
        "--specs", "examples/quick_specs.json",
        "--output-dir", "examples/output",
        "--job", "quick_2_1",
        "--episodes", "10", "--steps", "1000", "--warmup", "500",
    ]
    subprocess.run(command, cwd=NATIVE, check=True)


if __name__ == "__main__":
    main()
