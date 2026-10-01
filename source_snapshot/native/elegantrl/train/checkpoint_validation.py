"""Two-tier checkpoint promotion for long reinforcement-learning runs."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from evaluation_statistics import mean_se_median
from elegantrl.train.evaluator import get_cumulative_rewards_and_steps
from my_log import logger


def add_checkpoint_validation_arguments(parser) -> None:
    parser.add_argument("--checkpoint-validation", action="store_true")
    parser.add_argument("--checkpoint-validation-interval", type=int, default=200000)
    parser.add_argument("--checkpoint-validation-initial-step", type=int, default=10000)
    parser.add_argument("--checkpoint-validation-top-k", type=int, default=3)
    parser.add_argument("--checkpoint-validation-times", type=int, default=100)
    parser.add_argument("--checkpoint-validation-episode-steps", type=int, default=500)
    parser.add_argument("--checkpoint-validation-burn-in-steps", type=int, default=250)
    parser.add_argument("--checkpoint-validation-seed", type=int, default=700001)


def configure_checkpoint_validation(args, cli) -> None:
    args.checkpoint_validation_enabled = bool(cli.checkpoint_validation)
    for name in (
        "checkpoint_validation_interval",
        "checkpoint_validation_initial_step",
        "checkpoint_validation_top_k",
        "checkpoint_validation_times",
        "checkpoint_validation_episode_steps",
        "checkpoint_validation_burn_in_steps",
        "checkpoint_validation_seed",
    ):
        setattr(args, name, getattr(cli, name))


def _reset_random_streams(env, seed: int) -> None:
    """Reset all known stochastic streams so every promotion sees one bank."""
    if hasattr(env, "demand_rng"):
        env.demand_rng = np.random.default_rng(seed)
    if hasattr(env, "backorder_rng"):
        env.backorder_rng = np.random.default_rng(seed + 1)
    if hasattr(env, "emergency_rng"):
        env.emergency_rng = np.random.default_rng(seed + 2)
    if hasattr(env, "allocation_rng"):
        env.allocation_rng = random.Random(seed + 1_000_003)
    if hasattr(env, "episode"):
        env.episode = 0


class CheckpointValidationManager:
    """Promote short-listed actors on a reproducible medium validation bank."""

    def __init__(self, cwd: str, env, args):
        self.cwd = Path(cwd)
        self.env = env
        self.interval = int(args.checkpoint_validation_interval)
        self.initial_step = int(args.checkpoint_validation_initial_step)
        self.top_k = int(args.checkpoint_validation_top_k)
        self.eval_times = int(args.checkpoint_validation_times)
        self.episode_steps = int(args.checkpoint_validation_episode_steps)
        self.burn_in_steps = int(args.checkpoint_validation_burn_in_steps)
        self.seed = int(args.checkpoint_validation_seed)
        if self.interval <= 0 or self.initial_step < 0 or self.top_k <= 0:
            raise ValueError("Invalid checkpoint-validation schedule")
        if self.eval_times <= 0 or self.episode_steps <= 0:
            raise ValueError("Invalid checkpoint-validation evaluation size")
        if not 0 <= self.burn_in_steps < self.episode_steps:
            raise ValueError("Invalid checkpoint-validation burn-in")
        self.next_step = self.initial_step
        self.short_top: list[tuple[float, int]] = []
        self.records: list[dict] = []
        self.best_avg_r = -np.inf
        self.best_step: int | None = None
        self.last_short_step = -1
        self.validated_steps: set[int] = set()

    def _candidate_actor_path(self, step: int) -> Path:
        return self.cwd / f"checkpoint_candidate_actor__{step:012}.pt"

    def _write_candidates(self) -> None:
        payload = [
            {
                "short_rank": rank,
                "step": step,
                "short_avg_r": score,
                "actor": self._candidate_actor_path(step).name,
                "validated": step in self.validated_steps,
            }
            for rank, (score, step) in enumerate(self.short_top, start=1)
        ]
        (self.cwd / "checkpoint_candidates.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _record_short_candidate(self, actor, avg_r: float, step: int) -> tuple[bool, list[int]]:
        old_steps = {candidate_step for _, candidate_step in self.short_top}
        by_step = {candidate_step: score for score, candidate_step in self.short_top}
        by_step[int(step)] = float(avg_r)
        self.short_top = sorted(
            ((score, candidate_step) for candidate_step, score in by_step.items()),
            reverse=True,
        )[:self.top_k]
        new_steps = {candidate_step for _, candidate_step in self.short_top}
        added = step in new_steps and step not in old_steps
        evicted = sorted(old_steps - new_steps)
        if added:
            torch.save(actor, self._candidate_actor_path(step))
        self._write_candidates()
        return added, evicted

    def _evaluate(self, actor, reallocation_agent=None) -> tuple[float, float, float]:
        _reset_random_streams(self.env, self.seed)
        episode_means = []
        with torch.no_grad():
            for _ in range(self.eval_times):
                _, _, rewards = get_cumulative_rewards_and_steps(
                    self.env,
                    actor,
                    reallocation_agent=reallocation_agent,
                    episode_steps=self.episode_steps,
                    burn_in_steps=self.burn_in_steps,
                )
                episode_means.append(float(np.mean(rewards)))
        return mean_se_median(np.asarray(episode_means, dtype=np.float64))

    def _write_records(self) -> None:
        json_path = self.cwd / "checkpoint_validation_recorder.json"
        csv_path = self.cwd / "checkpoint_validation_recorder.csv"
        json_path.write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    "step", "short_avg_r", "validation_avg_r",
                    "validation_se_r", "validation_median_r", "short_rank",
                    "improved", "actor",
                ),
            )
            writer.writeheader()
            writer.writerows(self.records)

    def _validate_pending(self, reallocation_agent=None) -> list[dict]:
        results = []
        rank_by_step = {
            step: (rank, score)
            for rank, (score, step) in enumerate(self.short_top, start=1)
        }
        for _, step in self.short_top:
            if step in self.validated_steps:
                continue
            actor_path = self._candidate_actor_path(step)
            if not actor_path.exists():
                raise FileNotFoundError(f"Missing queued checkpoint actor: {actor_path}")
            actor = torch.load(
                actor_path,
                map_location=next(reallocation_agent.act.parameters()).device
                if reallocation_agent is not None else "cpu",
                weights_only=False,
            )
            avg_r, se_r, median_r = self._evaluate(actor, reallocation_agent)
            improved = avg_r > self.best_avg_r
            if improved:
                self.best_avg_r = avg_r
                self.best_step = step
            rank, short_avg_r = rank_by_step[step]
            validation_actor = self.cwd / f"validation_actor__{step:012}_{avg_r:09.3f}.pt"
            torch.save(actor, validation_actor)
            record = {
                "step": step,
                "short_avg_r": float(short_avg_r),
                "validation_avg_r": float(avg_r),
                "validation_se_r": float(se_r),
                "validation_median_r": float(median_r),
                "short_rank": int(rank),
                "improved": bool(improved),
                "actor": validation_actor.name,
            }
            self.records.append(record)
            self.validated_steps.add(step)
            results.append(record)
            logger.info(
                f"| Medium checkpoint validation: step={step}, rank={rank}, "
                f"short={short_avg_r:.6f}, avgR={avg_r:.6f}, "
                f"SE={se_r:.6f}, improved={improved}"
            )
        if results:
            self._write_records()
            self._write_candidates()
        return results

    def consider(self, actor, short_metrics: dict | None, reallocation_agent=None) -> dict | None:
        if not short_metrics:
            return None
        step = int(short_metrics["step"])
        if step == self.last_short_step:
            return None
        self.last_short_step = step
        added, evicted = self._record_short_candidate(
            actor, float(short_metrics["avg_r"]), step
        )
        validations = []
        if step >= self.next_step:
            while self.next_step <= step:
                self.next_step += self.interval
            validations = self._validate_pending(reallocation_agent)
        return {
            "candidate_added": added,
            "candidate_step": step if added else None,
            "evicted_steps": evicted,
            "validations": validations,
            "promoted_step": self.best_step,
        }

    def finalize(self, reallocation_agent=None) -> dict:
        """Validate any Top-K candidates that arrived after the last interval."""
        return {
            "candidate_added": False,
            "candidate_step": None,
            "evicted_steps": [],
            "validations": self._validate_pending(reallocation_agent),
            "promoted_step": self.best_step,
        }
