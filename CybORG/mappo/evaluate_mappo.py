
import os
import csv
import json
import numpy as np
import torch

from CybORG.mappo.cyborg_env import CybORGMAPPOEnv
from CybORG.mappo.mappo import MAPPO
from CybORG.mappo.XAI.explainer import MAPPOExplainer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mappo_model.pt")

# Keep these small while testing XAI.
# Increase later after everything is working.
NUM_EPISODES = 2
STEPS_PER_EPISODE = 10

XAI_TOP_K = 10
XAI_MAX_FEATURES = 128
XAI_BASELINE = 0.0
XAI_FREQUENCY = 1


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_current_action_masks(env):
    """Get the current action mask for every agent."""

    action_masks = {}

    for agent_id in env.agent_ids:
        action_masks[agent_id] = env.get_action_mask(
            agent_id
        )

    return action_masks


def extract_reset_data(reset_result, env):
    """Normalize the environment reset output."""

    if isinstance(reset_result, dict):
        observations = reset_result

    elif isinstance(reset_result, tuple):
        observations = None

        for item in reset_result:
            if isinstance(item, dict):
                if any(
                    agent_id in item
                    for agent_id in env.agent_ids
                ):
                    observations = item
                    break

        if observations is None:
            raise RuntimeError(
                "Could not identify observations "
                "from env.reset() output."
            )

    else:
        raise RuntimeError(
            "Unexpected env.reset() return type: "
            f"{type(reset_result)}"
        )

    action_masks = get_current_action_masks(env)

    return observations, action_masks


def extract_step_data(step_result, env):
    """Normalize the environment step output."""

    if not isinstance(step_result, tuple):
        raise RuntimeError(
            "Unexpected env.step() return type: "
            f"{type(step_result)}"
        )

    if len(step_result) == 4:
        observations = step_result[0]
        rewards = step_result[1]
        terminated = step_result[2]
        infos = step_result[3]

    elif len(step_result) == 3:
        observations = step_result[0]
        rewards = step_result[1]
        terminated = step_result[2]
        infos = {}

    else:
        raise RuntimeError(
            "Unexpected env.step() output length: "
            f"{len(step_result)}"
        )

    action_masks = get_current_action_masks(env)

    return (
        observations,
        rewards,
        terminated,
        infos,
        action_masks,
    )


def get_feature_names(env, agent_id):
    """
    Get feature names from the environment.

    Falls back to generic names if the environment does not
    expose get_feature_names().
    """

    if hasattr(env, "get_feature_names"):
        try:
            names = env.get_feature_names(agent_id)

            if names is not None:
                return list(names)

        except Exception as error:
            print(
                f"Warning: Could not get feature names for "
                f"{agent_id}: {error}"
            )

    return [
        f"Feature {i}"
        for i in range(env.observation_dim)
    ]


def load_mappo_checkpoint(mappo, model_path, device):
    """Load the existing trained MAPPO checkpoint."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Trained MAPPO model was not found.\n"
            f"Expected path: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unsupported checkpoint format: "
            f"{type(checkpoint)}"
        )

    encoder_version = checkpoint.get("observation_encoder_version")
    if encoder_version is not None and encoder_version != "semantic_enum_aggregate_v1":
        raise RuntimeError(
            "Checkpoint encoder version does not match this environment: "
            f"{encoder_version!r}. Refusing to generate misleading explanations."
        )

    encoder_verified = encoder_version == "semantic_enum_aggregate_v1"
    if not encoder_verified:
        print("WARNING: checkpoint has no observation-encoder metadata.")
        print("Semantic feature names will be disabled; XAI will report only")
        print("numeric input-dimension sensitivity, not semantic explanations.")

    expected_dims = {
        "observation_dim": mappo.actors[0].network[0].in_features
            if hasattr(mappo.actors[0], "network") and len(mappo.actors[0].network) > 0
            and hasattr(mappo.actors[0].network[0], "in_features") else None,
        "action_dim": len(mappo.actors[0].network[-1].weight)
            if hasattr(mappo.actors[0], "network") and len(mappo.actors[0].network) > 0
            and hasattr(mappo.actors[0].network[-1], "weight") else None,
        "num_agents": len(mappo.actors),
    }
    for key, expected in expected_dims.items():
        saved = checkpoint.get(key)
        if saved is not None and expected is not None and int(saved) != int(expected):
            raise RuntimeError(
                f"Checkpoint {key}={saved} does not match model value {expected}."
            )

    # --------------------------------------------------------
    # Actors
    # --------------------------------------------------------

    actor_states = None

    if "actors" in checkpoint:
        actor_states = checkpoint["actors"]

    elif "actor_states" in checkpoint:
        actor_states = checkpoint["actor_states"]

    elif "actors_state_dict" in checkpoint:
        actor_states = checkpoint["actors_state_dict"]

    if actor_states is None:
        raise RuntimeError(
            "Could not find actor weights in checkpoint."
        )

    if len(actor_states) != len(mappo.actors):
        raise RuntimeError(
            "Checkpoint actor count does not match the model. "
            f"Checkpoint: {len(actor_states)}, "
            f"model: {len(mappo.actors)}."
        )

    for i, actor_state in enumerate(actor_states):
        mappo.actors[i].load_state_dict(
            actor_state
        )

    # --------------------------------------------------------
    # Critic
    # --------------------------------------------------------

    if "critic" in checkpoint:
        mappo.critic.load_state_dict(
            checkpoint["critic"]
        )

    elif "critic_state_dict" in checkpoint:
        mappo.critic.load_state_dict(
            checkpoint["critic_state_dict"]
        )

    else:
        raise RuntimeError(
            "Could not find critic weights in checkpoint."
        )

    print("Trained MAPPO model loaded successfully.")
    print(f"Checkpoint path: {os.path.abspath(model_path)}")
    return encoder_verified


def calculate_step_reward(rewards, agent_ids):
    """Calculate average team reward for the current step."""

    if isinstance(rewards, dict):
        agent_rewards = []

        for agent_id in agent_ids:
            if agent_id not in rewards:
                continue

            try:
                agent_rewards.append(
                    float(rewards[agent_id])
                )
            except (TypeError, ValueError):
                continue

        if not agent_rewards:
            return 0.0

        return float(
            sum(agent_rewards) / len(agent_rewards)
        )

    try:
        return float(rewards)
    except (TypeError, ValueError):
        return 0.0


def is_terminated(terminated, agent_ids):
    """Check whether the episode has terminated."""

    if isinstance(terminated, dict):
        return all(
            bool(
                terminated.get(
                    agent_id,
                    False
                )
            )
            for agent_id in agent_ids
        )

    return bool(terminated)


def sanitize_xai_context(context):
    """Keep only descriptive, non-identifying context for the CSV."""
    if not isinstance(context, dict):
        return {}

    allowed_key_parts = (
        "success", "ostype", "os_type", "osdistribution",
        "os_distribution", "osversion", "os_version", "architecture",
    )
    blocked_key_parts = (
        "hostname", "host_name", "ip", "address", "subnet",
        "interface", "username", "user_name", "session", "process",
        "pid", "ppid", "uid", "gid", "port", "identifier",
    )

    safe = {}
    for key, value in context.items():
        key_text = str(key).lower()
        compact_key = "".join(ch for ch in key_text if ch.isalnum())
        if any(part.replace("_", "") in compact_key for part in blocked_key_parts):
            continue
        if not any(part.replace("_", "") in compact_key for part in allowed_key_parts):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
        elif hasattr(value, "name"):
            safe[str(key)] = str(value.name)
    return safe


def make_csv_row(common_row, feature=None):
    """Create a CSV row with a consistent set of fields."""

    row = dict(common_row)

    if feature is None:
        row["record_type"] = "DECISION_ONLY"
        row.update({
            "feature_index": "",
            "feature_name": "",
            "feature_value": "",
            "perturbed_probability": "",
            "probability_effect": "",
            "effect": "",
            "importance": "",
            "direction": "",
        })

    else:
        row["record_type"] = "FEATURE_ATTRIBUTION"
        row.update({
            "feature_index": feature.get("index", ""),
            "feature_name": feature.get("name", ""),
            "feature_value": feature.get("value", ""),
            "perturbed_probability": feature.get(
                "perturbed_probability", ""
            ),
            "probability_effect": feature.get("effect", ""),
            "effect": feature.get("effect", ""),
            "importance": feature.get("importance", ""),
            "direction": feature.get("direction", ""),
        })

    return row


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_mappo():

    print("=" * 70)
    print("                 MAPPO + XAI EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device              : {device}")

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = CybORGMAPPOEnv(
        steps=STEPS_PER_EPISODE
    )

    agent_ids = list(env.agent_ids)
    num_agents = len(agent_ids)
    observation_dim = env.observation_dim
    action_dim = env.action_dim

    print(f"Agents              : {agent_ids}")
    print(f"Number of agents    : {num_agents}")
    print(f"Observation dim     : {observation_dim}")
    print(f"Action dim          : {action_dim}")
    print(f"XAI max features    : {XAI_MAX_FEATURES}")
    print(f"XAI frequency       : every {XAI_FREQUENCY} step(s)")

    # --------------------------------------------------------
    # MAPPO
    # --------------------------------------------------------

    mappo = MAPPO(
        observation_dim=observation_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        hidden_dim=128,
        device=device,
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print()
    print("Loading trained MAPPO model...")

    try:
        encoder_verified = load_mappo_checkpoint(
            mappo,
            MODEL_PATH,
            device
        )

    except Exception as error:
        print()
        print("ERROR while loading MAPPO model:")
        print(error)

        env.close()
        return

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    for actor in mappo.actors:
        actor.eval()

    mappo.critic.eval()

    # --------------------------------------------------------
    # XAI
    # --------------------------------------------------------

    explainer = MAPPOExplainer(
        mappo=mappo,
        top_k=XAI_TOP_K,
        max_features=XAI_MAX_FEATURES,
        baseline=XAI_BASELINE,
    )

    print()
    print("Perturbation-based XAI initialized.")

    # --------------------------------------------------------
    # Action names
    # --------------------------------------------------------

    action_names = getattr(
        env,
        "ACTION_NAMES",
        [
            "AllowTrafficZone",
            "BlockTrafficZone",
            "Monitor",
            "Analyse",
            "Restore",
            "Remove",
            "DeployDecoy",
            "Sleep",
        ]
    )

    print()
    print("Action mapping:")
    print("-" * 70)

    for index, name in enumerate(action_names):
        print(f"{index}: {name}")

    # --------------------------------------------------------
    # Evaluation data
    # --------------------------------------------------------

    episode_rewards = []
    explanation_count = 0
    xai_rows = []

    # --------------------------------------------------------
    # Episodes
    # --------------------------------------------------------

    for episode in range(NUM_EPISODES):

        print()
        print("=" * 70)
        print(f"Episode {episode + 1}/{NUM_EPISODES}")
        print("=" * 70)

        reset_result = env.reset()

        observations, action_masks = extract_reset_data(
            reset_result,
            env
        )

        total_reward = 0.0
        steps_taken = 0

        # ----------------------------------------------------
        # Steps
        # ----------------------------------------------------

        for step in range(STEPS_PER_EPISODE):

            print()
            print(f"--- Step {step + 1} ---")

            # ------------------------------------------------
            # Select MAPPO actions
            # ------------------------------------------------

            with torch.no_grad():
                actions = mappo.select_actions(
                    observations,
                    action_masks,
                    deterministic=True
                )

            print()
            print("MAPPO Actions:")
            print("-" * 60)

            for agent_id in agent_ids:
                action_index = int(actions[agent_id])

                if 0 <= action_index < len(action_names):
                    action_name = action_names[action_index]
                else:
                    action_name = f"Action {action_index}"

                print(
                    f"{agent_id:<20} "
                    f"Action: {action_name}"
                )

            # ------------------------------------------------
            # XAI explanation
            # ------------------------------------------------

            if (step + 1) % XAI_FREQUENCY == 0:

                print()
                print("=" * 70)
                print("                    XAI ANALYSIS")
                print("=" * 70)

                for agent_index, agent_id in enumerate(agent_ids):

                    if encoder_verified:
                        feature_names = get_feature_names(env, agent_id)
                    else:
                        feature_names = [
                            f"Input dimension {i}"
                            for i in range(env.observation_dim)
                        ]

                    try:
                        explanation = explainer.explain(
                            agent_index=agent_index,
                            observation=observations[agent_id],
                            action=actions[agent_id],
                            action_mask=action_masks[agent_id],
                            feature_names=feature_names,
                            agent_id=agent_id,
                        )

                        explainer.print_explanation(
                            explanation
                        )

                        # ------------------------------------
                        # Common information for every row
                        # ------------------------------------

                        common_row = {
                            "episode": episode + 1,
                            "step": step + 1,
                            "agent_id": agent_id,

                            "selected_action": explanation.get(
                                "selected_action_name",
                                ""
                            ),

                            "action_probability": explanation.get(
                                "selected_probability",
                                ""
                            ),

                            "decision_margin": explanation.get(
                                "decision_margin",
                                ""
                            ),

                            "explanation_status": explanation.get(
                                "status",
                                "STATUS_NOT_PROVIDED"
                            ),

                            "attribution_type": explanation.get(
                                "attribution_type",
                                "TYPE_NOT_PROVIDED"
                            ),

                            "interpretation": explanation.get(
                                "interpretation",
                                ""
                            ),

                            "xai_context": json.dumps(
                                sanitize_xai_context(
                                    env.get_xai_context(agent_id)
                                    if hasattr(env, "get_xai_context")
                                    else explanation.get("xai_context", {})
                                ),
                                ensure_ascii=False,
                                default=str
                            ),
                        }

                        # ------------------------------------
                        # Save feature-level explanations
                        # ------------------------------------

                        features = explanation.get(
                            "features",
                            []
                        )

                        if features:
                            for feature in features:
                                xai_rows.append(
                                    make_csv_row(
                                        common_row,
                                        feature
                                    )
                                )

                        else:
                            # Keep the decision record even when
                            # feature-level attribution is unavailable.
                            xai_rows.append(
                                make_csv_row(common_row)
                            )

                        explanation_count += 1

                    except Exception as error:
                        print()
                        print(
                            f"XAI error for {agent_id}: {error}"
                        )

            # ------------------------------------------------
            # Environment step
            # ------------------------------------------------

            step_result = env.step(actions)

            (
                next_observations,
                rewards,
                terminated,
                infos,
                next_action_masks,
            ) = extract_step_data(
                step_result,
                env
            )

            # ------------------------------------------------
            # Reward
            # ------------------------------------------------

            step_reward = calculate_step_reward(
                rewards,
                agent_ids
            )

            total_reward += step_reward
            steps_taken += 1

            print()
            print(f"Step reward: {step_reward:.4f}")

            # ------------------------------------------------
            # Update state
            # ------------------------------------------------

            observations = next_observations
            action_masks = next_action_masks

            # ------------------------------------------------
            # Termination
            # ------------------------------------------------

            if is_terminated(terminated, agent_ids):
                print()
                print("Episode terminated.")
                break

        episode_rewards.append(total_reward)

        print()
        print(
            f"Episode {episode + 1} completed | "
            f"Reward: {total_reward:.4f} | "
            f"Steps: {steps_taken}"
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    episode_rewards = np.asarray(
        episode_rewards,
        dtype=np.float64
    )

    mean_reward = (
        float(np.mean(episode_rewards))
        if len(episode_rewards) > 0
        else 0.0
    )

    std_reward = (
        float(np.std(episode_rewards))
        if len(episode_rewards) > 0
        else 0.0
    )

    best_reward = (
        float(np.max(episode_rewards))
        if len(episode_rewards) > 0
        else 0.0
    )

    worst_reward = (
        float(np.min(episode_rewards))
        if len(episode_rewards) > 0
        else 0.0
    )

    # ========================================================
    # SAVE XAI RESULTS
    # ========================================================

    xai_csv_path = os.path.join(
        os.path.dirname(__file__),
        "xai_results.csv"
    )

    xai_fieldnames = [
        "episode",
        "step",
        "agent_id",
        "selected_action",
        "action_probability",
        "decision_margin",
        "explanation_status",
        "attribution_type",
        "record_type",
        "interpretation",
        "xai_context",
        "feature_index",
        "feature_name",
        "feature_value",
        "perturbed_probability",
        "probability_effect",
        "effect",
        "importance",
        "direction",
    ]

    try:
        with open(
            xai_csv_path,
            mode="w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=xai_fieldnames
            )

            writer.writeheader()
            
            # Final XAI privacy guard. Match whole tokens, not substrings:
            # e.g. do not hide valid names containing "session" or "port".
            blocked_tokens = {
                "pid", "ppid", "uid", "gid", "session", "sessions",
                "sessionid", "processid", "parentprocessid", "groupid",
                "localport", "remoteport", "port", "padding",
                "hostname", "username", "ip", "address", "subnet",
                "interface",
            }

            for row in xai_rows:
                name = str(row.get("feature_name", "") or "").strip()
                normalized = "".join(
                    ch.lower() if ch.isalnum() else " " for ch in name
                )
                tokens = set(normalized.split())
                compact = "".join(ch.lower() for ch in name if ch.isalnum())
                identifier_like = (
                    bool(tokens.intersection(blocked_tokens))
                    or any(token in compact for token in (
                        "sessionid", "processid", "parentprocessid",
                        "localport", "remoteport", "hostname", "username",
                    ))
                    or name.lower().startswith("padding[")
                )

                if identifier_like:
                    row["record_type"] = "EXCLUDED_FEATURE"
                    row["explanation_status"] = "EXCLUDED_FEATURE"
                    row["attribution_type"] = "excluded_identifier_like"
                    for field in (
                        "feature_index", "feature_name", "feature_value",
                        "perturbed_probability", "probability_effect",
                        "effect", "importance", "direction",
                    ):
                        row[field] = ""
                    row["interpretation"] = (
                        "Excluded identifier-like feature; not a semantic explanation."
                    )

            writer.writerows(xai_rows)

        print()
        print(f"XAI CSV saved to   : {xai_csv_path}")
        print(f"XAI CSV data rows  : {len(xai_rows)}")

    except OSError as error:
        print()
        print(f"ERROR saving XAI CSV: {error}")

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("                 MAPPO + XAI RESULTS")
    print("=" * 70)

    print(f"Number of episodes : {NUM_EPISODES}")
    print(f"Mean reward        : {mean_reward:.4f}")
    print(f"Std reward         : {std_reward:.4f}")
    print(f"Best reward        : {best_reward:.4f}")
    print(f"Worst reward       : {worst_reward:.4f}")
    print(f"XAI explanations   : {explanation_count}")
    print(f"XAI feature rows   : {len(xai_rows)}")
    print(f"XAI results CSV    : {xai_csv_path}")

    print("=" * 70)

    env.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    evaluate_mappo()

