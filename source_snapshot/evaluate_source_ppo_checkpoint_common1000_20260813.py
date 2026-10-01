#!/usr/bin/env python3
"""Evaluate one Kaynov-source PPO checkpoint on a fixed Table15 demand bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "scm_rl-master"
EPISODE_STEPS = 1000
WARMUP_STEPS = 500


def load_bank(path: Path) -> list[np.ndarray]:
    with np.load(path) as data:
        return [
            np.asarray(data[key], dtype=np.int64)
            for key in sorted(data.files, key=lambda item: int(item.split("_")[-1]))
        ]


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    specs_path = args.specs if args.specs.is_absolute() else ROOT / args.specs
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    spec = next(item for item in specs if item["job"] == args.job)
    if spec["kind"] != "source_ppo":
        raise ValueError(f"Not a source-PPO job: {args.job}")

    part_path = output_dir / "parts" / f"{args.job}.json"
    if part_path.exists() and args.episodes == 1000:
        print(part_path.read_text(encoding="utf-8"), end="")
        return 0

    import ray

    sys.path.insert(0, str(SOURCE_ROOT))
    from scsim.experiment import load_rllib_agent

    trial_dir = ROOT / spec["trial_dir"]
    checkpoint = ROOT / spec["checkpoint"]
    bank_path = ROOT / spec["demand_bank"]
    demand_bank = load_bank(bank_path)
    if not demand_bank or demand_bank[0].shape[0] < args.episodes:
        raise RuntimeError(f"Demand bank is too small: {bank_path}")

    # Ray creates Unix-domain sockets below this path; keep it short enough for
    # the macOS sockaddr_un limit even when the project path is deeply nested.
    ray_temp = Path("/tmp") / f"owmr_ppo_eval_{os.getpid()}"
    ray_temp.mkdir(parents=True, exist_ok=True)
    ray.init(
        num_cpus=1,
        include_dashboard=False,
        ignore_reinit_error=True,
        _temp_dir=str(ray_temp),
    )
    agent = None
    episode_means = np.empty(args.episodes, dtype=np.float64)
    try:
        agent, env = load_rllib_agent(
            trial_dir, checkpoint_name=checkpoint.name, num_workers=0, explore=False
        )
        env.reward_scale = 1.0
        env.episode_length = EPISODE_STEPS
        for episode in range(args.episodes):
            np.random.seed(20260802 + episode)
            state = env.reset()
            env.demand = [series[episode].copy() for series in demand_bank]
            tail: list[float] = []
            for period in range(EPISODE_STEPS):
                action = agent.compute_action(state, explore=False)
                state, reward, done, _ = env.step(action)
                if period >= WARMUP_STEPS:
                    tail.append(-float(reward))
                if done:
                    break
            if len(tail) != EPISODE_STEPS - WARMUP_STEPS:
                raise RuntimeError(
                    f"Episode {episode} produced {len(tail)} evaluated periods"
                )
            episode_means[episode] = float(np.mean(tail))
            if (episode + 1) % 100 == 0:
                print(
                    f"[{spec['case']} rank={spec['training_rank']}] "
                    f"{episode + 1}/{args.episodes}",
                    flush=True,
                )
    finally:
        if agent is not None:
            agent.stop()
        ray.shutdown()

    if args.episodes != 1000:
        print(
            json.dumps(
                {
                    "job": args.job,
                    "episodes": args.episodes,
                    "mean_cost": float(episode_means.mean()),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    episode_path = output_dir / "episode_means" / f"{args.job}.npz"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(episode_path, episode_means=episode_means)
    result = {
        **spec,
        "benchmark": "PPO (Kaynov source)",
        "mean_cost": float(episode_means.mean()),
        "se_cost": float(episode_means.std(ddof=1) / math.sqrt(args.episodes)),
        "median_cost": float(np.median(episode_means)),
        "full_mean_cost": None,
        "full_se_cost": None,
        "full_median_cost": None,
        "episodes": args.episodes,
        "episode_steps": EPISODE_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "episode_values": str(episode_path.relative_to(ROOT)),
        "demand_bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
    }
    atomic_json(part_path, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
