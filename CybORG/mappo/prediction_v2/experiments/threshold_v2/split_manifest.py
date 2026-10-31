"""Frozen, reproducible threshold_v2 episode split validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .policy_adapter import EXPECTED_CHECKPOINT_SHA256

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_MANIFEST = EXPERIMENT_DIR / "split_manifest_20261101.json"
MASTER_SEED = 20261101
EPISODE_COUNT = 300
PARTITION_RANGES = {
    "training": range(0, 135),
    "calibration": range(135, 180),
    "threshold_selection": range(180, 240),
    "final_testing": range(240, 300),
}
PARTITION_COUNTS = {
    "training": 135,
    "calibration": 45,
    "threshold_selection": 60,
    "final_testing": 60,
}
PARTITION_BOUNDARIES = {
    "training": [0, 134],
    "calibration": [135, 179],
    "threshold_selection": [180, 239],
    "final_testing": [240, 299],
}
COLLECTION_RUN_ID = "full_300_master_20261101"
KNOWN_V2_PREFLIGHT_SEEDS = frozenset({3088784133, 541812428, 1519898546})
EXPECTED_FIXED_METADATA = {
    "experiment": "threshold_v2",
    "manifest_version": 1,
    "run_id": COLLECTION_RUN_ID,
    "master_seed": MASTER_SEED,
    "episode_count": EPISODE_COUNT,
    "steps_per_episode": 100,
    "seed_generation_method": (
        "numpy.random.SeedSequence(master_seed).spawn(300); "
        "child.generate_state(1, dtype=numpy.uint32)[0]"
    ),
    "spawn_key_meaning": "SeedSequence child spawn key; one unique child and one episode per spawn index",
    "partition_assignment": "episode_id equals child spawn index; fixed inclusive boundaries below",
    "partition_boundaries": PARTITION_BOUNDARIES,
    "partition_counts": PARTITION_COUNTS,
    "related_episode_grouping": (
        "No shared-seed variants are planned. Any replay of a seed must retain its original "
        "episode_id, spawn_key, and partition."
    ),
    "known_threshold_v2_preflight_seeds_excluded": sorted(KNOWN_V2_PREFLIGHT_SEEDS),
    "threshold_v1_seed_independence": (
        "unverified: threshold_v1 has no per-episode CybORG seed ledger"
    ),
    "checkpoint_path": "CybORG/mappo/mappo_model.pt",
    "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    "output_directory": (
        "CybORG/mappo/prediction_v2/experiments/threshold_v2/data/"
        f"{COLLECTION_RUN_ID}"
    ),
    "output_safety": "Collector refuses to run when the output directory already exists.",
}


def derive_episode_seeds(master_seed: int = MASTER_SEED, count: int = EPISODE_COUNT) -> list[dict]:
    children = np.random.SeedSequence(master_seed).spawn(count)
    return [
        {
            "episode_id": episode_id,
            "child_seed": int(child.generate_state(1, dtype=np.uint32)[0]),
            "spawn_key": [int(part) for part in child.spawn_key],
            "partition": partition_for_episode(episode_id),
            "master_seed": master_seed,
        }
        for episode_id, child in enumerate(children)
    ]


def partition_for_episode(episode_id: int) -> str:
    for partition, episode_ids in PARTITION_RANGES.items():
        if episode_id in episode_ids:
            return partition
    raise ValueError(f"Episode ID is outside the frozen 0-{EPISODE_COUNT - 1} range: {episode_id}")


def validate_manifest_data(manifest: dict, checkpoint_sha256: str | None = None) -> list[dict]:
    if not isinstance(manifest, dict) or set(manifest) != {"metadata", "episodes"}:
        raise ValueError("Split manifest must contain exactly metadata and episodes fields.")
    metadata = manifest.get("metadata")
    episodes = manifest.get("episodes")
    if not isinstance(metadata, dict) or not isinstance(episodes, list):
        raise ValueError("Split manifest must contain metadata and episodes fields.")
    if set(metadata) != set(EXPECTED_FIXED_METADATA):
        missing = sorted(set(EXPECTED_FIXED_METADATA) - set(metadata))
        extra = sorted(set(metadata) - set(EXPECTED_FIXED_METADATA))
        raise ValueError(f"Split manifest metadata schema mismatch; missing={missing}, extra={extra}.")
    for key, expected_value in EXPECTED_FIXED_METADATA.items():
        if metadata[key] != expected_value:
            raise ValueError(
                f"Split manifest metadata field {key!r} mismatch: "
                f"expected {expected_value!r}, got {metadata[key]!r}."
            )
    if len(episodes) != EPISODE_COUNT:
        raise ValueError(f"Split manifest must contain exactly {EPISODE_COUNT} episodes.")
    if checkpoint_sha256 is not None and checkpoint_sha256.lower() != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Current MAPPO checkpoint hash does not match the pinned MAPPO hash.")

    expected = derive_episode_seeds()
    if episodes != expected:
        raise ValueError(
            "Split manifest episode records differ from the frozen SeedSequence derivation "
            "or partition assignment."
        )
    episode_ids = [int(record["episode_id"]) for record in episodes]
    seeds = [int(record["child_seed"]) for record in episodes]
    if episode_ids != list(range(EPISODE_COUNT)) or len(set(episode_ids)) != EPISODE_COUNT:
        raise ValueError("Split manifest episode IDs must be unique and exactly 0-299.")
    if len(set(seeds)) != EPISODE_COUNT:
        raise ValueError("Split manifest child seeds are not unique.")
    if KNOWN_V2_PREFLIGHT_SEEDS.intersection(seeds):
        raise ValueError("Split manifest reuses a known threshold_v2 preflight seed.")

    counts = {
        partition: sum(record["partition"] == partition for record in episodes)
        for partition in PARTITION_RANGES
    }
    if counts != PARTITION_COUNTS:
        raise ValueError(f"Unexpected partition counts: {counts}")
    return episodes


def load_split_manifest(path: Path = DEFAULT_SPLIT_MANIFEST) -> tuple[dict, list[dict]]:
    path = Path(path).resolve()
    experiment_root = EXPERIMENT_DIR.resolve()
    if path != experiment_root and experiment_root not in path.parents:
        raise ValueError(f"Split manifest must remain inside threshold_v2: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Frozen threshold_v2 split manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest, validate_manifest_data(manifest)