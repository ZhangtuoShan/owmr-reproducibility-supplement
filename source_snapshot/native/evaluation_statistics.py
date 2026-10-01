"""Shared statistics for simulation-based policy evaluation."""

from __future__ import annotations

import math

import numpy as np


def mean_se_median(values) -> tuple[float, float, float]:
    """Summarize independent replications with the standard error of their mean."""
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    if samples.size == 0:
        raise ValueError("At least one evaluation replication is required.")

    mean = float(samples.mean())
    median = float(np.median(samples))
    if samples.size == 1:
        standard_error = 0.0
    else:
        standard_error = float(samples.std(ddof=1) / math.sqrt(samples.size))
    return mean, standard_error, median


def episode_average_statistics(values_by_episode) -> tuple[float, float, float]:
    """Return mean, SE, and median of per-episode average observations."""
    values = np.asarray(values_by_episode, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            "values_by_episode must have shape (episodes, evaluated_periods); "
            f"got {values.shape}."
        )
    if values.shape[1] == 0:
        raise ValueError("Each evaluation episode must contain at least one evaluated period.")
    return mean_se_median(values.mean(axis=1))
