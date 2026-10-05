import numpy as np
import torch


class MAPPOExplainer:
    """
    Perturbation/ablation-based XAI for the MAPPO actor.

    The trained MAPPO model is NOT modified.
    The explainer only observes the actor's output and measures
    how the selected-action probability changes when an input
    feature is replaced by a baseline value.
    """

    def __init__(
        self,
        mappo,
        action_names=None,
        top_k=10,
        max_features=128,
        baseline=0.0,
        device=None,
    ):
        self.mappo = mappo
        self.action_names = action_names or [
            "AllowTrafficZone",
            "BlockTrafficZone",
            "Monitor",
            "Analyse",
            "Restore",
            "Remove",
            "DeployDecoy",
            "Sleep",
        ]

        self.top_k = int(top_k)
        self.max_features = int(max_features)
        self.baseline = float(baseline)

        if device is None:
            self.device = next(self.mappo.actors[0].parameters()).device
        else:
            self.device = torch.device(device)

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _actor(self, agent_index):
        return self.mappo.actors[agent_index]

    def _action_name(self, action):
        action = int(action)
        if 0 <= action < len(self.action_names):
            return self.action_names[action]
        return f"Action_{action}"

    def _prepare_mask(self, action_mask):
        if action_mask is None:
            return None

        mask = np.asarray(action_mask, dtype=np.float32).reshape(-1)

        if len(mask) != len(self.action_names):
            return None

        return torch.tensor(
            mask,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def _get_probabilities(self, actor, observation, action_mask):
        obs_tensor = torch.tensor(
            np.asarray(observation, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)

        mask_tensor = self._prepare_mask(action_mask)

        with torch.no_grad():
            distribution = actor.get_distribution(
                obs_tensor,
                mask_tensor,
            )
            probabilities = distribution.probs.squeeze(0).detach().cpu().numpy()

        return probabilities

    # ------------------------------------------------------------
    # Main explanation
    # ------------------------------------------------------------

    def explain(
        self,
        agent_index,
        observation,
        action,
        action_mask=None,
        feature_names=None,
        agent_id=None,
    ):
        observation = np.asarray(
            observation,
            dtype=np.float32,
        ).reshape(-1)

        action = int(action)

        if feature_names is None:
            feature_names = [
                f"Feature {i}"
                for i in range(len(observation))
            ]
        else:
            feature_names = list(feature_names)

            if len(feature_names) < len(observation):
                feature_names.extend(
                    f"Feature {i}"
                    for i in range(
                        len(feature_names),
                        len(observation),
                    )
                )

            feature_names = feature_names[:len(observation)]

        actor = self._actor(agent_index)

        # --------------------------------------------------------
        # Original policy output
        # --------------------------------------------------------

        original_probabilities = self._get_probabilities(
            actor,
            observation,
            action_mask,
        )

        selected_probability = float(
            original_probabilities[action]
        )

        sorted_probs = np.sort(
            original_probabilities
        )[::-1]

        if len(sorted_probs) >= 2:
            decision_margin = float(
                sorted_probs[0] - sorted_probs[1]
            )
        else:
            decision_margin = float(sorted_probs[0])

        # --------------------------------------------------------
        # Identify usable features
        # --------------------------------------------------------

        finite_mask = np.isfinite(observation)

        nonzero_indices = np.where(
            finite_mask & (np.abs(observation) > 1e-12)
        )[0]

        # Limit expensive perturbations.
        if len(nonzero_indices) > self.max_features:
            # Use largest absolute-value features as candidates.
            ranked = nonzero_indices[
                np.argsort(
                    np.abs(observation[nonzero_indices])
                )[::-1]
            ]

            candidate_indices = ranked[:self.max_features]
        else:
            candidate_indices = nonzero_indices

        # --------------------------------------------------------
        # No observable features
        # --------------------------------------------------------

        if len(candidate_indices) == 0:
            return {
                "agent_id": agent_id,
                "agent_index": agent_index,
                "selected_action": action,
                "selected_action_name": self._action_name(action),
                "selected_probability": selected_probability,
                "decision_margin": decision_margin,
                "action_probabilities": original_probabilities,
                "features": [],
                "has_observable_features": False,
                "interpretation": (
                    "No non-zero environmental features were "
                    "available for ablation. The current observation "
                    "does not provide a feature-level explanation."
                ),
            }

        # --------------------------------------------------------
        # Batched perturbations
        # --------------------------------------------------------

        perturbed = np.repeat(
            observation.reshape(1, -1),
            len(candidate_indices),
            axis=0,
        )

        for row, feature_index in enumerate(candidate_indices):
            perturbed[
                row,
                feature_index
            ] = self.baseline

        obs_tensor = torch.tensor(
            perturbed,
            dtype=torch.float32,
            device=self.device,
        )

        mask_tensor = None

        if action_mask is not None:
            mask = np.asarray(
                action_mask,
                dtype=np.float32,
            ).reshape(-1)

            if len(mask) == len(self.action_names):
                mask_tensor = torch.tensor(
                    np.repeat(
                        mask.reshape(1, -1),
                        len(candidate_indices),
                        axis=0,
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )

        with torch.no_grad():
            distribution = actor.get_distribution(
                obs_tensor,
                mask_tensor,
            )

            perturbed_probabilities = (
                distribution.probs[:, action]
                .detach()
                .cpu()
                .numpy()
            )

        # --------------------------------------------------------
        # Calculate effects
        # --------------------------------------------------------

        effects = (
            selected_probability
            - perturbed_probabilities
        )

        importances = np.abs(effects)

        features = []

        for feature_index, effect, importance in zip(
            candidate_indices,
            effects,
            importances,
        ):
            effect = float(effect)
            importance = float(importance)

            if effect > 0:
                direction = "SUPPORTS"
            elif effect < 0:
                direction = "SUPPRESSES"
            else:
                direction = "NEUTRAL"

            features.append(
                {
                    "index": int(feature_index),
                    "name": feature_names[feature_index],
                    "value": float(observation[feature_index]),
                    "importance": importance,
                    "effect": effect,
                    "direction": direction,
                    "perturbed_probability": float(
                        perturbed_probabilities[
                            list(candidate_indices).index(feature_index)
                        ]
                    ),
                }
            )

        features.sort(
            key=lambda item: item["importance"],
            reverse=True,
        )

        top_features = features[:self.top_k]

        selected_name = self._action_name(action)

        if top_features:
            strongest = top_features[0]

            interpretation = (
                f"The agent selected '{selected_name}'. "
                f"The strongest observed feature was "
                f"'{strongest['name']}' "
                f"(value={strongest['value']:.4f}), "
                f"which {strongest['direction'].lower()} the decision "
                f"with an ablation effect of "
                f"{strongest['effect']:+.4f}."
            )
        else:
            interpretation = (
                f"The agent selected '{selected_name}', but no "
                f"meaningful feature-level effect was detected."
            )

        return {
            "agent_id": agent_id,
            "agent_index": agent_index,
            "selected_action": action,
            "selected_action_name": selected_name,
            "selected_probability": selected_probability,
            "decision_margin": decision_margin,
            "action_probabilities": original_probabilities,
            "features": top_features,
            "has_observable_features": True,
            "interpretation": interpretation,
        }

    # ------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------

    def print_explanation(self, explanation):
        print()
        print("=" * 72)
        print("                    XAI DECISION EXPLANATION")
        print("=" * 72)

        agent_id = explanation.get("agent_id")

        if agent_id is None:
            agent_id = (
                f"Agent {explanation.get('agent_index', '?')}"
            )

        print(f"Agent              : {agent_id}")
        print(
            f"Selected Action    : "
            f"{explanation['selected_action_name']}"
        )
        print(
            f"Action Probability : "
            f"{explanation['selected_probability']:.4f}"
        )
        print(
            f"Decision Margin    : "
            f"{explanation['decision_margin']:.4f}"
        )

        print()
        print("Action Probabilities:")
        print("-" * 68)

        probabilities = explanation["action_probabilities"]

        for index, probability in enumerate(probabilities):
            print(
                f"{self._action_name(index):20s} : "
                f"{probability:.4f}"
            )

        print()
        print("Top Influential Features:")
        print("-" * 68)

        features = explanation.get("features", [])

        if not explanation.get("has_observable_features", False):
            print("No observable environmental features were available.")
            print()
            print("XAI Status:")
            print(
                "The current 256-D observation contains no non-zero "
                "features that can be ablated."
            )
        elif not features:
            print(
                "Observable features were present, but no measurable "
                "feature effect was detected."
            )
        else:
            for rank, feature in enumerate(features, start=1):
                print(
                    f"{rank}. {feature['name']}"
                )
                print(
                    f"   Value      : "
                    f"{feature['value']:.6f}"
                )
                print(
                    f"   Importance : "
                    f"{feature['importance']:.6f}"
                )
                print(
                    f"   Effect     : "
                    f"{feature['effect']:+.6f}"
                )
                print(
                    f"   Direction  : "
                    f"{feature['direction']}"
                )
                print()

        print()
        print("Interpretation:")
        print("-" * 68)
        print(explanation["interpretation"])
        print("=" * 72)

    # ------------------------------------------------------------
    # Convenience method
    # ------------------------------------------------------------

    def explain_and_print(
        self,
        agent_index,
        observation,
        action,
        action_mask=None,
        feature_names=None,
        agent_id=None,
    ):
        explanation = self.explain(
            agent_index=agent_index,
            observation=observation,
            action=action,
            action_mask=action_mask,
            feature_names=feature_names,
            agent_id=agent_id,
        )

        self.print_explanation(explanation)

        return explanation
