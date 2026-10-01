import math
from numbers import Number

import numpy as np
from scipy import stats


MAX_DEMAND = 500
QUANTILE_SERIES_LEVELS = tuple(round(0.1 * i, 1) for i in range(1, 10))


def poisson_tail_lump_cap(lam, tail_prob=1e-10):
    """Return the smallest practical cap for lumping a Poisson upper tail."""
    lam = float(lam)
    tail_prob = float(tail_prob)
    if lam < 0:
        raise ValueError(f"Poisson rate must be nonnegative, got {lam}.")
    if not 0.0 < tail_prob < 1.0:
        raise ValueError(f"tail_prob must be in (0, 1), got {tail_prob}.")
    if lam == 0:
        return 0
    quantile = stats.poisson.ppf(1.0 - tail_prob, lam)
    if not np.isfinite(quantile):
        raise ValueError(
            f"Could not determine a finite Poisson cap for lam={lam}, tail_prob={tail_prob}."
        )
    return int(max(0, math.ceil(float(quantile))))


def _is_quantile_series_spec(spec):
    return (
        isinstance(spec, dict)
        and str(spec.get("type", "")).lower() in {
            "quantile_series",
            "nonstationary_quantile",
            "nonstationary_quantile_series",
        }
    )


def _parse_demand_spec(spec):
    if _is_quantile_series_spec(spec):
        return "quantile_series", (spec,)
    if isinstance(spec, Number):
        return "poisson", (float(spec),)
    if not isinstance(spec, tuple) or not spec:
        raise ValueError(f"Unsupported demand specification: {spec!r}")

    dist_name = str(spec[0]).lower()
    if dist_name == "poisson":
        if len(spec) != 2:
            raise ValueError(f"Poisson demand expects ('Poisson', lambda), got {spec!r}")
        return "poisson", (float(spec[1]),)
    if dist_name in {"truncated_poisson", "poisson_truncated", "poisson_tail_lump"}:
        if len(spec) != 3:
            raise ValueError(
                "Truncated Poisson demand expects "
                "('truncated_poisson', lambda, max_demand), got "
                f"{spec!r}"
            )
        lam = float(spec[1])
        demand_max = int(spec[2])
        if demand_max < 0:
            raise ValueError(f"Truncated Poisson max demand must be nonnegative, got {spec!r}")
        return "truncated_poisson", (lam, demand_max)
    if dist_name == "uniform":
        if len(spec) != 3:
            raise ValueError(f"Uniform demand expects ('uniform', low, high), got {spec!r}")
        low, high = float(spec[1]), float(spec[2])
        if high < low:
            raise ValueError(f"Uniform demand upper bound must be >= lower bound, got {spec!r}")
        return "uniform", (low, high)
    if dist_name == "normal":
        if len(spec) != 3:
            raise ValueError(f"Normal demand expects ('normal', mean, variance), got {spec!r}")
        mean, variance = float(spec[1]), float(spec[2])
        if variance < 0:
            raise ValueError(f"Normal demand variance must be nonnegative, got {spec!r}")
        if variance == 0:
            return "deterministic", (max(0.0, mean),)
        return "normal", (mean, variance)
    raise ValueError(f"Unsupported demand distribution: {spec!r}")


def demand_mean(spec):
    dist_name, params = _parse_demand_spec(spec)
    if dist_name == "poisson":
        return params[0]
    if dist_name == "quantile_series":
        return quantile_series_mean(params[0])

    support, pmf = _single_period_integer_pmf(spec)
    return float(np.dot(support, pmf))


def demand_std(spec):
    dist_name, params = _parse_demand_spec(spec)
    if dist_name == "poisson":
        return math.sqrt(max(params[0], 0.0))
    if dist_name == "quantile_series":
        support, pmf = quantile_series_lead_time_pmf(params[0], 0, 1)
        mean = float(np.dot(support, pmf))
        variance = float(np.dot((support - mean) ** 2, pmf))
        return math.sqrt(max(variance, 0.0))

    support, pmf = _single_period_integer_pmf(spec)
    mean = float(np.dot(support, pmf))
    variance = float(np.dot((support - mean) ** 2, pmf))
    return math.sqrt(max(variance, 0.0))


def demand_quantile(spec, quantile):
    q = float(np.clip(quantile, 0.0, 1.0))
    if _is_quantile_series_spec(spec):
        return quantile_series_quantile(spec, 0, q)
    support, pmf = _single_period_integer_pmf(spec)
    idx = int(np.searchsorted(np.cumsum(pmf), q, side="left"))
    idx = min(max(idx, 0), len(support) - 1)
    return float(support[idx])


def _normalize_pmf(pmf):
    pmf = np.asarray(pmf, dtype=float)
    pmf = np.maximum(pmf, 0.0)
    total = pmf.sum()
    if total <= 0:
        raise ValueError("Demand PMF has zero mass.")
    return pmf / total


def _single_period_integer_pmf(spec, tail_prob=1e-10):
    dist_name, params = _parse_demand_spec(spec)
    if dist_name == "quantile_series":
        return quantile_series_period_pmf(params[0], 0)

    if dist_name == "poisson":
        lam = params[0]
        if lam <= 0:
            return np.array([0], dtype=float), np.array([1.0], dtype=float)
        max_k = int(max(0, math.ceil(stats.poisson.ppf(1.0 - tail_prob, lam))))
        if max_k >= MAX_DEMAND:
            support = np.arange(MAX_DEMAND + 1, dtype=float)
            pmf = stats.poisson.pmf(np.arange(MAX_DEMAND + 1), lam)
            pmf[-1] = stats.poisson.sf(MAX_DEMAND - 1, lam)
        else:
            support = np.arange(max_k + 1, dtype=float)
            pmf = stats.poisson.pmf(support, lam)
        return support, _normalize_pmf(pmf)

    if dist_name == "truncated_poisson":
        lam, demand_max = params
        support = np.arange(demand_max + 1, dtype=float)
        if lam <= 0:
            pmf = np.zeros(demand_max + 1, dtype=float)
            pmf[0] = 1.0
            return support, pmf
        pmf = stats.poisson.pmf(support, lam)
        if demand_max > 0:
            pmf[-1] = stats.poisson.sf(demand_max - 1, lam)
        else:
            pmf[0] = 1.0
        return support, _normalize_pmf(pmf)

    if dist_name == "deterministic":
        value = int(np.clip(math.floor(params[0]), 0, MAX_DEMAND))
        return np.array([value], dtype=float), np.array([1.0], dtype=float)

    if dist_name == "uniform":
        low, high = params
        if high == low:
            value = int(np.clip(math.floor(low), 0, MAX_DEMAND))
            return np.array([value], dtype=float), np.array([1.0], dtype=float)
        support = np.arange(MAX_DEMAND + 1, dtype=float)
        pmf = []
        width = high - low
        for k in support:
            if k == 0:
                lower, upper = low, min(high, 1.0)
            elif k == MAX_DEMAND:
                lower, upper = max(low, float(MAX_DEMAND)), high
            else:
                lower, upper = max(low, float(k)), min(high, float(k + 1))
            pmf.append(max(0.0, upper - lower) / width)
        return support, _normalize_pmf(pmf)

    # Kaynov-compatible normal demand: draw an ordinary Gaussian, floor it, and
    # clip to [0, 500]. This is different from sampling a conditional truncated
    # normal distribution and different from rounding to the nearest integer.
    mean, variance = params
    sigma = math.sqrt(variance)
    support = np.arange(MAX_DEMAND + 1, dtype=float)
    pmf = []
    for k in support:
        if k == 0:
            prob = stats.norm.cdf(1.0, loc=mean, scale=sigma)
        elif k == MAX_DEMAND:
            prob = stats.norm.sf(float(MAX_DEMAND), loc=mean, scale=sigma)
        else:
            lower = float(k)
            upper = float(k + 1)
            prob = stats.norm.cdf(upper, loc=mean, scale=sigma)
            prob -= stats.norm.cdf(lower, loc=mean, scale=sigma)
        pmf.append(prob)
    return support, _normalize_pmf(pmf)


def lead_time_demand_pmf(spec, periods, tail_prob=1e-10):
    if _is_quantile_series_spec(spec):
        return quantile_series_lead_time_pmf(spec, 0, periods)

    periods = int(periods)
    if periods <= 0:
        return np.array([0], dtype=float), np.array([1.0], dtype=float)

    support, pmf = _single_period_integer_pmf(spec, tail_prob=tail_prob)
    base_pmf = pmf
    total_pmf = np.array([1.0], dtype=float)
    for _ in range(periods):
        total_pmf = np.convolve(total_pmf, base_pmf)
    total_pmf = _normalize_pmf(total_pmf)
    total_support = np.arange(len(total_pmf), dtype=float) + support[0] * periods
    return total_support, total_pmf


def lead_time_demand_quantile(spec, quantile, periods, tail_prob=1e-10):
    if _is_quantile_series_spec(spec):
        support, pmf = quantile_series_lead_time_pmf(spec, 0, periods)
        q = float(np.clip(quantile, 0.0, 1.0))
        idx = int(np.searchsorted(np.cumsum(pmf), q, side="left"))
        idx = min(max(idx, 0), len(support) - 1)
        return float(support[idx])

    support, pmf = lead_time_demand_pmf(spec, periods, tail_prob=tail_prob)
    q = float(np.clip(quantile, 0.0, 1.0))
    idx = int(np.searchsorted(np.cumsum(pmf), q, side="left"))
    idx = min(max(idx, 0), len(support) - 1)
    return float(support[idx])


def smooth_lead_time_quantile_table(
        spec,
        periods,
        grid_size=1025,
        tau=0.25,
        prob_eps=1e-5,
        tail_prob=1e-10,
):
    support, pmf = lead_time_demand_pmf(spec, periods, tail_prob=tail_prob)
    grid_size = int(grid_size)
    if grid_size < 8:
        raise ValueError("grid_size must be at least 8.")
    tau = float(tau)
    if tau <= 0:
        raise ValueError("tau must be positive.")

    interior_size = grid_size - 2
    p_grid = np.linspace(prob_eps, 1.0 - prob_eps, interior_size, dtype=float)
    low = np.full_like(p_grid, -0.5)
    high = np.full_like(p_grid, support[-1] + 0.5 + 20.0 * tau)

    support_row = support.reshape(1, -1)
    pmf_row = pmf.reshape(1, -1)
    for _ in range(64):
        mid = 0.5 * (low + high)
        z = np.clip((mid.reshape(-1, 1) + 0.5 - support_row) / tau, -60.0, 60.0)
        smooth_cdf = (1.0 / (1.0 + np.exp(-z)) * pmf_row).sum(axis=1)
        low = np.where(smooth_cdf < p_grid, mid, low)
        high = np.where(smooth_cdf < p_grid, high, mid)

    q_grid = np.maximum(0.0, 0.5 * (low + high))
    p_grid = np.concatenate(([0.0], p_grid, [1.0]))
    q_grid = np.concatenate(([0.0], q_grid, [support[-1]]))
    return p_grid.astype(np.float32), q_grid.astype(np.float32)


def sample_demand(spec, size=None, random_state=None):
    dist_name, params = _parse_demand_spec(spec)
    if dist_name == "quantile_series":
        return sample_quantile_series_demand(params[0], 0, size=size, random_state=random_state)
    if dist_name == "poisson":
        samples = stats.poisson.rvs(params[0], size=size, random_state=random_state)
    elif dist_name == "truncated_poisson":
        support, pmf = _single_period_integer_pmf(spec)
        support = support.astype(int)
        rng = random_state
        if rng is None:
            samples = np.random.choice(support, size=size, p=pmf)
        elif isinstance(rng, (int, np.integer)):
            samples = np.random.default_rng(int(rng)).choice(support, size=size, p=pmf)
        elif hasattr(rng, "choice"):
            samples = rng.choice(support, size=size, p=pmf)
        else:
            samples = np.random.default_rng(rng).choice(support, size=size, p=pmf)
    elif dist_name == "deterministic":
        value = int(np.clip(math.floor(params[0]), 0, MAX_DEMAND))
        samples = np.full(size if size is not None else (), value, dtype=int)
    elif dist_name == "uniform":
        low, high = params
        samples = stats.uniform.rvs(loc=low, scale=high - low, size=size, random_state=random_state)
    else:
        mean, variance = params
        sigma = math.sqrt(variance)
        samples = stats.norm.rvs(loc=mean, scale=sigma, size=size, random_state=random_state)

    # An explicit truncated-Poisson specification owns its support. Applying
    # the legacy global cap here would silently turn, for example,
    # truncated-Poisson(1000, 1208) back into a point mass near 500.
    demand_cap = int(params[1]) if dist_name == "truncated_poisson" else MAX_DEMAND
    samples = np.clip(np.floor(samples), 0, demand_cap).astype(int)
    if size is None:
        return int(samples)
    return samples


def _quantile_series_horizon(spec):
    quantiles = spec.get("quantiles", spec)
    first_key = str(QUANTILE_SERIES_LEVELS[0])
    if first_key not in quantiles:
        first_key = QUANTILE_SERIES_LEVELS[0]
    return len(quantiles[first_key])


def _quantile_series_values(spec, period):
    quantiles = spec.get("quantiles", spec)
    horizon = _quantile_series_horizon(spec)
    if horizon <= 0:
        raise ValueError("Quantile-series demand has empty horizon.")
    t = int(np.clip(period, 0, horizon - 1))
    values = []
    for q in QUANTILE_SERIES_LEVELS:
        key = str(q)
        if key not in quantiles:
            key = q
        values.append(max(float(quantiles[key][t]), 0.0))
    return np.maximum.accumulate(np.asarray(values, dtype=float))


def _round_half_up_nonnegative(values):
    return np.floor(np.asarray(values, dtype=float) + 0.5).astype(int)


def quantile_series_period_pmf(spec, period, max_demand=None):
    """Build an integer PMF from 0.1-0.9 quantile forecasts for one period.

    The inverse CDF is piecewise linear from (0, 0) to (0.1, q0.1), then
    through q0.2..q0.9. The remaining upper-tail probability [0.9, 1] is
    assigned to q0.9. Demand is integerized by round-half-up.
    """
    if max_demand is None:
        max_demand = int(spec.get("max_demand", MAX_DEMAND))
    max_demand = max(0, int(max_demand))
    q_values = np.clip(_quantile_series_values(spec, period), 0.0, max_demand)
    probs = np.asarray((0.0,) + QUANTILE_SERIES_LEVELS, dtype=float)
    values = np.concatenate(([0.0], q_values))

    pmf = np.zeros(max_demand + 1, dtype=float)
    for i in range(len(probs) - 1):
        p0, p1 = probs[i], probs[i + 1]
        x0, x1 = values[i], values[i + 1]
        mass = p1 - p0
        if mass <= 0:
            continue
        if x1 <= x0 + 1e-12:
            k = int(np.clip(math.floor(x0 + 0.5), 0, max_demand))
            pmf[k] += mass
            continue
        lo = max(0, int(math.floor(x0 + 0.5)) - 1)
        hi = min(max_demand, int(math.floor(x1 + 0.5)) + 1)
        for k in range(lo, hi + 1):
            left = max(x0, k - 0.5)
            right = min(x1, k + 0.5)
            if right > left:
                pmf[k] += mass * (right - left) / (x1 - x0)

    upper_tail_k = int(np.clip(math.floor(q_values[-1] + 0.5), 0, max_demand))
    pmf[upper_tail_k] += 0.1
    support = np.flatnonzero(pmf > 0).astype(float)
    return support, _normalize_pmf(pmf[support.astype(int)])


def quantile_series_lead_time_pmf(spec, start_period, periods):
    periods = int(periods)
    if periods <= 0:
        return np.array([0], dtype=float), np.array([1.0], dtype=float)
    horizon = _quantile_series_horizon(spec)
    total_pmf = np.array([1.0], dtype=float)
    total_offset = 0
    for offset in range(periods):
        period = int(start_period) + offset
        if period >= horizon:
            break
        support, pmf = quantile_series_period_pmf(spec, period)
        base_offset = int(support[0])
        dense = np.zeros(int(support[-1] - support[0]) + 1, dtype=float)
        dense[(support - support[0]).astype(int)] = pmf
        total_pmf = np.convolve(total_pmf, dense)
        total_offset += base_offset
    support = np.arange(len(total_pmf), dtype=float) + total_offset
    return support, _normalize_pmf(total_pmf)


def quantile_series_sum_window_pmf(windows):
    """Convolve integer demand PMFs from aligned nonstationary windows.

    ``windows`` contains ``(spec, start_period, periods)`` tuples. This is used
    for system-level CDC protection, where demand from all RDCs must be summed
    for the same calendar window rather than by summing unrelated per-RDC peak
    quantiles.
    """
    total_pmf = np.array([1.0], dtype=float)
    total_offset = 0
    for spec, start_period, periods in windows:
        support, pmf = quantile_series_lead_time_pmf(spec, start_period, periods)
        if len(support) == 0:
            continue
        support_int = np.rint(support).astype(int)
        lo = int(support_int[0])
        hi = int(support_int[-1])
        dense = np.zeros(hi - lo + 1, dtype=float)
        dense[support_int - lo] = pmf
        total_pmf = np.convolve(total_pmf, dense)
        total_offset += lo
    support = np.arange(len(total_pmf), dtype=float) + total_offset
    return support, _normalize_pmf(total_pmf)


def quantile_series_quantile(spec, period, quantile):
    support, pmf = quantile_series_period_pmf(spec, period)
    q = float(np.clip(quantile, 0.0, 1.0))
    idx = int(np.searchsorted(np.cumsum(pmf), q, side="left"))
    idx = min(max(idx, 0), len(support) - 1)
    return float(support[idx])


def quantile_series_mean(spec, period=None):
    horizon = _quantile_series_horizon(spec)
    periods = range(horizon) if period is None else [int(period)]
    means = []
    for t in periods:
        support, pmf = quantile_series_period_pmf(spec, t)
        means.append(float(np.dot(support, pmf)))
    return float(np.mean(means)) if means else 0.0


def sample_quantile_series_demand(spec, period, size=None, random_state=None):
    support, pmf = quantile_series_period_pmf(spec, period)
    support = support.astype(int)
    rng = random_state
    if rng is None:
        samples = np.random.choice(support, size=size, p=pmf)
    elif isinstance(rng, (int, np.integer)):
        samples = np.random.default_rng(int(rng)).choice(support, size=size, p=pmf)
    elif hasattr(rng, "choice"):
        samples = rng.choice(support, size=size, p=pmf)
    else:
        samples = np.random.default_rng(rng).choice(support, size=size, p=pmf)
    if size is None:
        return int(samples)
    return np.asarray(samples, dtype=int)
