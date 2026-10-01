#!/usr/bin/env python3
"""Verify SHA-256 hashes of every packaged file (except the hash list)."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    lines = (ROOT / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines()
    for line in lines:
        digest, name = line.split("  ", 1)
        target = ROOT / name
        if not target.is_file():
            raise FileNotFoundError(target)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            raise AssertionError(f"Checksum mismatch: {name}")
    print(f"Verified {len(lines)} file hashes")


if __name__ == "__main__":
    main()
