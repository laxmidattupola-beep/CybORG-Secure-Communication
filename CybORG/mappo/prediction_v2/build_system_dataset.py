"""Build a system-level attack-risk dataset from the five zone observations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ZONE_TO_AGENT
from .models import feature_columns


ZONE_ORDER = (
    "restricted_a",
    "operational_a",
    "restricted_b",
    "operational_b",
    "hq",
)

HORIZONS = (3, 5, 10)


def build_system_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    print(f"Reading zone dataset: {input_path}")

    frame = pd.read_csv(input_path)

    required = {
        "episode_id",
        "step",
        "zone",
        *[f"attack_risk_{h}" for h in HORIZONS],
        *[f"label_valid_{h}" for h in HORIZONS],
    }

    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    actual_zones = set(frame["zone"].unique())
    expected_zones = set(ZONE_ORDER)

    if actual_zones != expected_zones:
        raise ValueError(
            f"Zone mismatch. Expected {sorted(expected_zones)}, "
            f"received {sorted(actual_zones)}"
        )

    group_columns = ["episode_id", "step"]

    # Every episode-step must contain exactly one row for each zone.
    zone_counts = frame.groupby(group_columns)["zone"].nunique()

    if not (zone_counts == len(ZONE_ORDER)).all():
        raise ValueError("Some episode-step groups do not contain all five zones.")

    if frame.duplicated(group_columns + ["zone"]).any():
        raise ValueError("Duplicate episode-step-zone rows detected.")

    # Use the same feature-selection policy as zone-level training.
    features = feature_columns(frame)

    if not features:
        raise ValueError("No input features were found.")

    print(f"Input features per zone: {len(features)}")

    # Verify horizon validity is identical across all five zones.
    validity_columns = [f"label_valid_{h}" for h in HORIZONS]

    validity_counts = (
        frame.groupby(group_columns)[validity_columns].nunique()
    )

    if (validity_counts > 1).any().any():
        raise ValueError("Horizon validity differs between zones.")

    # Construct one system-level row for each episode-step.
    records = []

    for (episode_id, step), group in frame.groupby(
        group_columns, sort=True
    ):
        zone_lookup = group.set_index("zone")

        row = {
            "episode_id": episode_id,
            "step": int(step),
        }

        for zone in ZONE_ORDER:
            zone_row = zone_lookup.loc[zone]

            for feature in features:
                row[f"{zone}__{feature}"] = zone_row[feature]

        # System risk is positive if ANY monitored zone is positive.
        for horizon in HORIZONS:
            target = f"attack_risk_{horizon}"
            valid = f"label_valid_{horizon}"

            row[f"system_attack_risk_{horizon}"] = int(
                group[target].astype(int).any()
            )

            row[valid] = int(group[valid].iloc[0])

        records.append(row)

    system_frame = pd.DataFrame(records)

    # Ensure system features are numeric and finite.
    system_features = [
        column
        for column in system_frame.columns
        if "__" in column
    ]

    system_frame[system_features] = (
        system_frame[system_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    system_frame.to_csv(output_path, index=False)

    print("\nSYSTEM DATASET CREATED")
    print("----------------------")
    print(f"Rows: {len(system_frame)}")
    print(f"Episodes: {system_frame['episode_id'].nunique()}")
    print(f"Features per zone: {len(features)}")
    print(f"Total system features: {len(system_features)}")
    print(f"Zones combined: {len(ZONE_ORDER)}")

    print("\nSystem-level label counts:")
    for horizon in HORIZONS:
        target = f"system_attack_risk_{horizon}"
        valid = f"label_valid_{horizon}"

        valid_rows = system_frame[system_frame[valid] == 1]
        positives = int(valid_rows[target].sum())

        print(
            f"Horizon {horizon}: "
            f"{positives} positive / "
            f"{len(valid_rows) - positives} negative / "
            f"{len(valid_rows)} valid"
        )

    print(f"\nSaved to: {output_path.resolve()}")

    return system_frame


def main():
    parser = argparse.ArgumentParser(
        description="Build system-level future attack-risk dataset."
    )

    package_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--input",
        type=Path,
        default=package_dir / "data_corrected" / "zone_risk_dataset.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=package_dir / "data_corrected" / "system_risk_dataset.csv",
    )

    args = parser.parse_args()

    build_system_dataset(args.input, args.output)


if __name__ == "__main__":
    main()