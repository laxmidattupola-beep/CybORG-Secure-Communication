import re
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

    @staticmethod
    def _is_identifier_or_unhelpful_feature(name, value):
        """Exclude identifiers, raw ports, padding, and unnamed features."""
        import re

        path = str(name).strip().lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", path).strip()
        tokens = set(normalized.split())

        excluded_names = {
            "pid", "ppid", "gid", "uid",
            "session", "sessionid", "groupid",
            "processid", "parentprocessid",
            "localport", "remoteport", "port",
            "padding", "feature"
        }

        if tokens.intersection(excluded_names):
            return True

        if re.fullmatch(r"feature\s*\d+", normalized):
            return True

        return False

    def explain(
        self,
        agent_index,
        observation,
        action,
        action_mask=None,
        feature_names=None,
        agent_id=None,
    ):
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        action = int(action)

        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(observation))]
        else:
            feature_names = list(feature_names)
            if len(feature_names) < len(observation):
                feature_names.extend(
                    f"Feature {i}" for i in range(len(feature_names), len(observation))
                )
            feature_names = feature_names[:len(observation)]

        actor = self._actor(agent_index)
        original_probabilities = self._get_probabilities(actor, observation, action_mask)
        if action < 0 or action >= len(original_probabilities):
            raise ValueError(f"Action index {action} is outside the policy output.")

        selected_probability = float(original_probabilities[action])
        other_probabilities = np.delete(original_probabilities, action)
        decision_margin = (
            selected_probability - float(np.max(other_probabilities))
            if len(other_probabilities) else selected_probability
        )

        base_result = {
            "agent_id": agent_id,
            "agent_index": agent_index,
            "selected_action": action,
            "selected_action_name": self._action_name(action),
            "selected_probability": selected_probability,
            "decision_margin": float(decision_margin),
            "action_probabilities": original_probabilities,
            "features": [],
            "has_observable_features": False,
            "status": "NO_OBSERVABLE_FEATURES",
        }

        # If only one action is permitted, the action was effectively forced by
        # the environment mask. Do not attribute it to observation features.
        if action_mask is not None:
            try:
                mask = np.asarray(action_mask, dtype=np.float32).reshape(-1)
                if len(mask) == len(self.action_names) and np.count_nonzero(mask > 0) == 1:
                    base_result.update({
                        "status": "FORCED_ACTION_MASK",
                        "interpretation": (
                            f"'{self._action_name(action)}' was the only action permitted "
                            "by the action mask. This is not a feature-based policy explanation."
                        ),
                    })
                    return base_result
            except (TypeError, ValueError):
                pass

        finite_mask = np.isfinite(observation)
        nonzero_indices = np.where(finite_mask & (np.abs(observation) > 1e-12))[0]

        if len(nonzero_indices) == 0:
            only_padding = all(str(name).lower().startswith("padding[") for name in feature_names)
            base_result.update({
                "status": "EMPTY_OBSERVATION" if only_padding else "ZERO_VALUED_OBSERVATION",
                "interpretation": (
                    "The observation contains no non-zero numeric values. No feature-level "
                    "attribution can be calculated without inventing or carrying forward data."
                ),
            })
            return base_result

        eligible_indices = [
            int(i) for i in nonzero_indices
            if not self._is_identifier_or_unhelpful_feature(feature_names[int(i)], observation[int(i)])
        ]

        # Only use interpretable features. Never reintroduce excluded
        # identifiers or raw ports as a fallback explanation.
        # A missing checkpoint encoder version means names are intentionally
        # generic. Report numeric sensitivity only, never semantic attribution.
        generic_dimension_names = all(
            re.fullmatch(r"input dimension\s+\d+", str(name).strip().lower())
            for name in feature_names
        )
        nonsemantic_fallback = generic_dimension_names

        if not eligible_indices:
            base_result.update({
                "status": "NO_SEMANTIC_FEATURES",
                "interpretation": (
                    "No eligible semantic features were found in this observation. "
                    "Identifiers, raw port numbers, padding, and unnamed features "
                    "are excluded, so no semantic feature attribution is available."
                ),
            })
            return base_result

        # Select candidates by observation order, not absolute magnitude. Ranking
        # by magnitude would systematically prioritize raw numeric scales.
        candidate_indices = np.asarray(eligible_indices[:self.max_features], dtype=int)

        perturbed = np.repeat(observation.reshape(1, -1), len(candidate_indices), axis=0)
        for row, feature_index in enumerate(candidate_indices):
            perturbed[row, feature_index] = self.baseline

        obs_tensor = torch.tensor(perturbed, dtype=torch.float32, device=self.device)
        mask_tensor = None
        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=np.float32).reshape(-1)
            if len(mask) == len(self.action_names):
                mask_tensor = torch.tensor(
                    np.repeat(mask.reshape(1, -1), len(candidate_indices), axis=0),
                    dtype=torch.float32,
                    device=self.device,
                )

        with torch.no_grad():
            distribution = actor.get_distribution(obs_tensor, mask_tensor)
            perturbed_probabilities = (
                distribution.probs[:, action].detach().cpu().numpy()
            )

        effects = selected_probability - perturbed_probabilities
        features = []
        for position, feature_index in enumerate(candidate_indices):
            effect = float(effects[position])
            features.append({
                "index": int(feature_index),
                "name": feature_names[int(feature_index)],
                "value": float(observation[int(feature_index)]),
                "importance": abs(effect),
                "effect": effect,
                "probability_effect": effect,
                "direction": "SUPPORTS" if effect > 0 else "SUPPRESSES" if effect < 0 else "NEUTRAL",
                "perturbed_probability": float(perturbed_probabilities[position]),
            })

        features.sort(key=lambda item: item["importance"], reverse=True)
        top_features = features[:self.top_k]
        selected_name = self._action_name(action)
        strongest = top_features[0] if top_features else None

        if strongest and strongest["importance"] > 1e-8:
            if nonsemantic_fallback:
                interpretation = (
                    f"The policy selected '{selected_name}'. Ablating the numeric input "
                    f"'{strongest['name']}' changed its probability from "
                    f"{selected_probability:.4f} to {strongest['perturbed_probability']:.4f} "
                    f"(effect {strongest['effect']:+.4f}). The tested inputs were identifiers "
                    "or other non-semantic numeric fields, so this is only a model-sensitivity "
                    "diagnostic—not a semantic explanation and not proof of an attack."
                )
                status = "NON_SEMANTIC_INPUT_SENSITIVITY"
            else:
                interpretation = (
                    f"The policy selected '{selected_name}'. Ablating the observable feature "
                    f"'{strongest['name']}' changed its probability from "
                    f"{selected_probability:.4f} to {strongest['perturbed_probability']:.4f} "
                    f"(effect {strongest['effect']:+.4f}). This is a model-sensitivity result, "
                    "not proof that the feature represents an attack."
                )
                status = "FEATURE_ATTRIBUTIONS"
        else:
            if nonsemantic_fallback:
                interpretation = (
                    f"The policy selected '{selected_name}', but ablating the available "
                    "non-semantic numeric inputs did not measurably change its probability. "
                    "This observation does not support a semantic feature-level explanation."
                )
                status = "NO_NON_SEMANTIC_INPUT_EFFECT"
            else:
                interpretation = (
                    f"The policy selected '{selected_name}', but the tested semantic features "
                    "did not produce a measurable change in its probability."
                )
                status = "NO_FEATURE_EFFECT"

        base_result.update({
            "features": top_features,
            "has_observable_features": True,
            "status": status,
            "attribution_type": "non_semantic_numeric_sensitivity" if nonsemantic_fallback else "semantic_feature_sensitivity",
            "interpretation": interpretation,
        })
        return base_result

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
            print("No feature-level attribution was generated.")
            print(f"XAI Status: {explanation.get('status', 'NO_OBSERVABLE_FEATURES')}")
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
