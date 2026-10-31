"""Validate the frozen 300-episode split and pinned checkpoint hash."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .policy_adapter import CHECKPOINT_PATH, checkpoint_sha256
from .split_manifest import DEFAULT_SPLIT_MANIFEST, validate_manifest_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    current_hash = checkpoint_sha256(CHECKPOINT_PATH)
    episodes = validate_manifest_data(manifest, checkpoint_sha256=current_hash)
    counts = {
        partition: sum(row["partition"] == partition for row in episodes)
        for partition in ("training", "calibration", "threshold_selection", "final_testing")
    }
    print(json.dumps({
        "status": "passed",
        "manifest": str(args.manifest.resolve()),
        "episode_count": len(episodes),
        "unique_episode_ids": len({row["episode_id"] for row in episodes}),
        "unique_child_seeds": len({row["child_seed"] for row in episodes}),
        "partition_counts": counts,
        "known_preflight_seed_overlap": [],
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_sha256": current_hash,
    }, indent=2))


if __name__ == "__main__":
    main()