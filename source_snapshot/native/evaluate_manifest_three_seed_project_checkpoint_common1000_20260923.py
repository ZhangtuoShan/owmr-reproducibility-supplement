#!/usr/bin/env python3
"""Evaluate one validation-selected project actor on a fixed common demand bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluate_manifest_kaynov_small import (
    build_project_env,
    evaluate_project_actor,
    load_demand_bank,
)
from evaluate_recent_sac_candidates_common_1000ep import restore_bound_key_types
from evaluate_table15_four_methods_common1000 import build_env as build_table_env


ROOT = Path(__file__).resolve().parent


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=500)
    args = parser.parse_args()

    specs_path = args.specs if args.specs.is_absolute() else ROOT / args.specs
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    spec = next(item for item in specs if item["job"] == args.job)
    if spec["kind"] != "project_actor":
        raise ValueError(f"Not a project actor job: {args.job}")
    part_path = output_dir / "parts" / f"{args.job}.json"
    if part_path.is_file() and args.episodes == 1000:
        print(part_path.read_text(encoding="utf-8"), end="")
        return 0

    torch.set_num_threads(1)
    protocol = restore_bound_key_types(spec["protocol"])
    bank_path = ROOT / spec["demand_bank"]
    bank = load_demand_bank(bank_path)
    if spec["family"] == "kaynov":
        env, _, input_args = build_project_env(spec["case"], protocol, bank)
    else:
        env, _, input_args = build_table_env(spec["case"], protocol, bank, 20260923, args.steps)
    costs = evaluate_project_actor(
        spec["benchmark"], ROOT / spec["checkpoint"], env, input_args, protocol,
        args.episodes, args.steps, args.warmup,
    )
    episode_means = costs.mean(axis=1)
    if args.episodes != 1000:
        print(json.dumps({"job": args.job, "mean_cost": float(episode_means.mean())}))
        return 0

    episode_path = output_dir / "episode_means" / f"{args.job}.npz"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(episode_path, episode_means=episode_means)
    result = {
        **spec,
        "mean_cost": float(episode_means.mean()),
        "se_cost": float(episode_means.std(ddof=1) / math.sqrt(args.episodes)),
        "median_cost": float(np.median(episode_means)),
        "episodes": args.episodes,
        "episode_steps": args.steps,
        "warmup_steps": args.warmup,
        "episode_values": str(episode_path.relative_to(ROOT)),
        "demand_bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
    }
    atomic_json(part_path, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
