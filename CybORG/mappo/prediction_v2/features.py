"""Leakage-aware feature engineering from defender-visible zone observations."""
from __future__ import annotations
import numpy as np

ZONE_ORDER = ("restricted_a", "operational_a", "restricted_b", "operational_b", "hq")


def observation_vector(observation, expected_dim=256):
    arr = np.asarray(observation, dtype=np.float32).reshape(-1)
    if arr.size != expected_dim:
        raise ValueError(f"Expected {expected_dim} observation values, received {arr.size}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def zone_features(observation, zone, step, episode_steps, history=None, expected_dim=256):
    """Feature vector using only that zone's blue observation and public clock.

    `history` must contain prior *observation-derived* vectors only; never pass
    true-state labels or simulator-only foothold flags to this function.
    """
    base = observation_vector(observation, expected_dim)
    zone_idx = ZONE_ORDER.index(zone)
    phase = min(2, int(3 * step / max(1, episode_steps)))
    hist = list(history or [])[-5:]
    if hist:
        hist_arr = np.stack([observation_vector(x, expected_dim) for x in hist])
        hist_mean = hist_arr.mean(axis=0)
        hist_delta = base - hist_arr[-1]
        hist_std = hist_arr.std(axis=0)
    else:
        hist_mean = np.zeros_like(base)
        hist_delta = np.zeros_like(base)
        hist_std = np.zeros_like(base)
    # Keep a compact and stable feature vector: current 256 + historical summary 3*256 + metadata.
    metadata = np.asarray([zone_idx / 4.0, phase / 2.0, step / max(1, episode_steps - 1), len(hist) / 5.0], dtype=np.float32)
    return np.concatenate([base, hist_mean, hist_delta, hist_std, metadata]).astype(np.float32)


def feature_names(expected_dim=256):
    return ([f"obs_{i}" for i in range(expected_dim)]
            + [f"history_mean_{i}" for i in range(expected_dim)]
            + [f"history_delta_{i}" for i in range(expected_dim)]
            + [f"history_std_{i}" for i in range(expected_dim)]
            + ["zone_index", "mission_phase", "episode_progress", "history_count"])
