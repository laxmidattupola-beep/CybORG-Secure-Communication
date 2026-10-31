
"""Train separate system-level future attack-risk models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV

from .models import (
    build_models,
    metrics_for,
    split_episode_ids,
    _safe_probability,
)


HORIZONS = (3, 5, 10)

NON_FEATURE_COLUMNS = {
    "episode_id",
    "step",
    "system_attack_risk_3",
    "system_attack_risk_5",
    "system_attack_risk_10",
    "label_valid_3",
    "label_valid_5",
    "label_valid_10",
}


def main():
    parser = argparse.ArgumentParser(
        description="Train system-level attack-risk models."
    )

    package_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--dataset",
        type=Path,
        default=package_dir / "data_corrected" / "system_risk_dataset.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=package_dir / "models_system",
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    frame = pd.read_csv(args.dataset)

    required = {
        "episode_id",
        "step",
        *[f"system_attack_risk_{h}" for h in HORIZONS],
        *[f"label_valid_{h}" for h in HORIZONS],
    }

    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    feature_columns = [
        column for column in frame.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No system-level features found.")

    X_all = (
        frame[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    split = split_episode_ids(frame, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    (args.output_dir / "episode_split.json").write_text(
        json.dumps(split, indent=2),
        encoding="utf-8",
    )

    ids = {
        name: set(values)
        for name, values in split.items()
    }

    records = []

    print(f"Episodes: {frame.episode_id.nunique()}")
    print(f"System features: {len(feature_columns)}")
    print(f"Episode split: {split}")

    for horizon in HORIZONS:
        target = f"system_attack_risk_{horizon}"
        valid_column = f"label_valid_{horizon}"

        y_all = frame[target].astype(int)
        valid_mask = frame[valid_column].astype(bool)

        train_mask = frame.episode_id.isin(ids["train"]) & valid_mask
        val_mask = frame.episode_id.isin(ids["validation"]) & valid_mask
        test_mask = frame.episode_id.isin(ids["test"]) & valid_mask

        if y_all[train_mask].nunique() < 2:
            raise ValueError(f"{target}: training data contains one class.")

        if y_all[test_mask].nunique() < 2:
            raise ValueError(f"{target}: test data contains one class.")

        print(f"\nHorizon {horizon}")
        print(
            f"Train: {train_mask.sum()} | "
            f"Validation: {val_mask.sum()} | "
            f"Test: {test_mask.sum()}"
        )

        for model_name, model in build_models(args.seed).items():

            model.fit(X_all.loc[train_mask], y_all.loc[train_mask])

            if (
                model_name != "base_rate"
                and y_all[val_mask].nunique() == 2
            ):
                try:
                    calibrated = CalibratedClassifierCV(
                        model,
                        method="sigmoid",
                        cv="prefit",
                    )
                    calibrated.fit(
                        X_all.loc[val_mask],
                        y_all.loc[val_mask],
                    )
                    model = calibrated
                except (ValueError, TypeError) as exc:
                    print(f"Calibration skipped: {exc}")

            probabilities = _safe_probability(
                model,
                X_all.loc[test_mask],
            )

            metrics = metrics_for(
                y_all.loc[test_mask],
                probabilities,
            )

            records.append({
                "horizon": horizon,
                "target": target,
                "model": model_name,
                **metrics,
            })

            package = {
                "model": model,
                "feature_columns": feature_columns,
                "target": target,
                "window": horizon,
                "seed": args.seed,
                "split_episode_ids": split,
                "feature_policy": (
                    "Five-zone concatenated defender observations; "
                    "no simulator true-state features or labels."
                ),
                "label_validity_column": valid_column,
            }

            filename = (
                f"{model_name}_system_attack_risk_{horizon}.joblib"
            )

            joblib.dump(
                package,
                args.output_dir / filename,
            )

            print(
                f"{model_name}: "
                f"F1={metrics['f1']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"FPR={metrics['false_positive_rate']:.4f}, "
                f"AP={metrics['average_precision']}"
            )

    report = pd.DataFrame(records)

    report.to_csv(
        args.output_dir / "system_heldout_metrics.csv",
        index=False,
    )

    print("\nSYSTEM MODEL TRAINING COMPLETE")
    print(report.to_string(index=False))
    print(f"\nModels saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
