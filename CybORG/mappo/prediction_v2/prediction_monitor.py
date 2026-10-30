"""Monitoring-only integration for the three trained zone-risk models."""

import csv
from pathlib import Path

from .config import ZONE_TO_AGENT
from .risk_api import ZoneRiskPredictor


HORIZONS = (3, 5, 10)


class MAPPOPredictionMonitor:
    """Runs independent risk predictors without influencing MAPPO actions."""

    def __init__(self):
        self.predictors = {}

        # Each zone and horizon gets its own predictor and history.
        for zone, agent_id in ZONE_TO_AGENT.items():
            self.predictors[zone] = {}

            for horizon in HORIZONS:
                model_files = {
                    3: "gradient_boosting_attack_risk_3.joblib",
                    5: "gradient_boosting_attack_risk_5.joblib",
                    10: "logistic_regression_attack_risk_10.joblib",
                }

                corrected_model_dir = (
                    Path(__file__).resolve().parent / "models_corrected"
                )
                model_path = corrected_model_dir / model_files[horizon]

                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Prediction model not found: {model_path}"
                    )

                self.predictors[zone][horizon] = ZoneRiskPredictor(
                    model_path
                )

        print(
            "Prediction monitor initialized: "
            f"{len(ZONE_TO_AGENT)} zones x {len(HORIZONS)} horizons"
        )

    def reset_episode(self):
        """Clear observation history at the beginning of each episode."""
        for zone_predictors in self.predictors.values():
            for predictor in zone_predictors.values():
                predictor.history.clear()
                predictor.zone = None

    def predict_all(self, observations, step, episode_steps=100):
        """Predict risk using only the corresponding blue agent observation."""
        results = []

        for zone, agent_id in ZONE_TO_AGENT.items():
            if agent_id not in observations:
                raise KeyError(
                    f"Observation missing for {agent_id} ({zone})"
                )

            observation = observations[agent_id]

            for horizon in HORIZONS:
                probability = self.predictors[zone][horizon].predict(
                    observation=observation,
                    zone=zone,
                    step=step,
                    episode_steps=episode_steps,
                )

                results.append({
                    "step": step,
                    "zone": zone,
                    "agent_id": agent_id,
                    "horizon": horizon,
                    "attack_risk_probability": probability,
                })

        return results


def save_prediction_results(rows, output_path):
    """Save prediction results separately from XAI results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "episode",
        "step",
        "zone",
        "agent_id",
        "horizon",
        "attack_risk_probability",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path
