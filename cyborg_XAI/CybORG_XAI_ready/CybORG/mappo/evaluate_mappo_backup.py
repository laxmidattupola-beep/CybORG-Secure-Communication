import os
import csv
import numpy as np
import torch

from CybORG.mappo.cyborg_env import CybORGMAPPOEnv
from CybORG.mappo.mappo import MAPPO
from CybORG.mappo.XAI.explainer import MAPPOExplainer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "mappo_model.pt"
)

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
    """
    Normalize the environment reset output.

    The current CybORG MAPPO environment returns a
    dictionary of processed observations.
    """

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
    """
    Normalize the environment step output.

    Supported:
        (observations, rewards, terminated, infos)
        (observations, rewards, terminated)
    """

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
    Get feature names from the modified CybORG environment.

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

    for i, actor_state in enumerate(actor_states):

        if i < len(mappo.actors):

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

    print("Trained MAPPO model loaded successfully.")


def calculate_step_reward(rewards, agent_ids):
    """Calculate average team reward for the current step."""

    step_reward = 0.0

    if isinstance(rewards, dict):

        for agent_id in agent_ids:

            if agent_id in rewards:

                try:
                    step_reward += float(
                        rewards[agent_id]
                    )
                except (
                    TypeError,
                    ValueError
                ):
                    pass

    else:

        try:
            step_reward = float(rewards)
        except (
            TypeError,
            ValueError
        ):
            step_reward = 0.0

    if len(agent_ids) > 0:
        step_reward /= len(agent_ids)

    return step_reward


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

    agent_ids = list(
        env.agent_ids
    )

    num_agents = len(
        agent_ids
    )

    observation_dim = (
        env.observation_dim
    )

    action_dim = (
        env.action_dim
    )

    print(
        f"Agents              : {agent_ids}"
    )

    print(
        f"Number of agents    : {num_agents}"
    )

    print(
        f"Observation dim     : {observation_dim}"
    )

    print(
        f"Action dim          : {action_dim}"
    )

    print(
        f"XAI max features    : {XAI_MAX_FEATURES}"
    )

    print(
        f"XAI frequency       : every {XAI_FREQUENCY} step(s)"
    )

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

        load_mappo_checkpoint(
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
        action_names=list(getattr(env, "ACTION_NAMES", [])) or None,
        top_k=XAI_TOP_K,
        max_features=XAI_MAX_FEATURES,
        baseline=XAI_BASELINE,
    )

    print()
    print(
        "Perturbation-based XAI initialized."
    )

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
        print(
            f"{index}: {name}"
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    episode_rewards = []
    explanation_count = 0
    # One CSV row per explained feature, for reproducible analysis.
    xai_csv_rows = []

    for episode in range(
        NUM_EPISODES
    ):

        print()
        print("=" * 70)
        print(
            f"Episode {episode + 1}/{NUM_EPISODES}"
        )
        print("=" * 70)

        reset_result = env.reset()

        (
            observations,
            action_masks
        ) = extract_reset_data(
            reset_result,
            env
        )

        total_reward = 0.0
        steps_taken = 0

        for step in range(
            STEPS_PER_EPISODE
        ):

            print()
            print(
                f"--- Step {step + 1} ---"
            )

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

                action_index = int(
                    actions[agent_id]
                )

                if (
                    0 <= action_index
                    < len(action_names)
                ):

                    action_name = (
                        action_names[action_index]
                    )

                else:

                    action_name = (
                        f"Action {action_index}"
                    )

                print(
                    f"{agent_id:<20} "
                    f"Action: {action_name}"
                )

            # ------------------------------------------------
            # XAI explanation
            # ------------------------------------------------

            if (
                (step + 1)
                % XAI_FREQUENCY
                == 0
            ):

                print()
                print("=" * 70)
                print(
                    "                    XAI ANALYSIS"
                )
                print("=" * 70)

                for agent_index, agent_id in enumerate(
                    agent_ids
                ):
                    # CybORG emits an IN_PROGRESS observation while a multi-tick
                    # action is running. The controller ignores replacement actions
                    # during this period, so do not present a fresh MAPPO output as
                    # an explanation for an action that was not actually executed.
                    if (
                        hasattr(env, "is_action_in_progress")
                        and env.is_action_in_progress(agent_id)
                    ):
                        active_action = env.get_in_progress_action_name(agent_id)
                        print(
                            f"{agent_id}: XAI skipped — {active_action} "
                            "is still in progress; no new action is being executed."
                        )
                        xai_csv_rows.append({
                            "explanation_status": "ACTION_IN_PROGRESS",
                            "episode": episode + 1,
                            "step": step + 1,
                            "agent_id": agent_id,
                            "selected_action": active_action,
                            "action_probability": "",
                            "decision_margin": "",
                            "feature_index": "",
                            "feature_name": "",
                            "feature_value": "",
                            "perturbed_probability": "",
                            "effect": "",
                            "importance": "",
                            "direction": "No new decision is executed while the prior action is in progress.",
                            "xai_context": str(xai_context),
                        })
                        continue
                    xai_context = env.extract_xai_context(
                            observations[agent_id]
                    )
                    if episode == 0 and step == 0:
                        obs = observations[agent_id]
                        print("\nOBSERVATION DEBUG")
                        print("Agent:", agent_id)
                        print("Type:", type(obs))
                        print("Value:", repr(obs)[:3000])
                        print("Context:", xai_context)
                    feature_names = (
                        get_feature_names(
                            env,
                            agent_id
                        )
                    )

                    try:

                        explanation = (
                            explainer.explain(
                                agent_index=agent_index,
                                observation=(
                                    observations[
                                        agent_id
                                    ]
                                ),
                                action=(
                                    actions[
                                        agent_id
                                    ]
                                ),
                                action_mask=(
                                    action_masks[
                                        agent_id
                                    ]
                                ),
                                feature_names=(
                                    feature_names
                                ),
                                agent_id=agent_id,
                            )
                        )

                        explainer.print_explanation(
                            explanation
                        )

                        explanation_count += 1

                        # Persist the explanation so the demo produces
                        # evidence, not only terminal output.
                        top_features = explanation.get("features", [])
                        if top_features:
                            for feature in top_features:
                                xai_csv_rows.append({
                                    "explanation_status": "FEATURE_ATTRIBUTIONS",
                                    "episode": episode + 1,
                                    "step": step + 1,
                                    "agent_id": agent_id,
                                    "selected_action": explanation.get("selected_action_name", ""),
                                    "action_probability": explanation.get("selected_probability", ""),
                                    "decision_margin": explanation.get("decision_margin", ""),
                                    "feature_index": feature.get("index", ""),
                                    "feature_name": feature.get("name", ""),
                                    "feature_value": feature.get("value", ""),
                                    "perturbed_probability": feature.get("perturbed_probability", ""),
                                    "effect": feature.get("effect", ""),
                                    "importance": feature.get("importance", ""),
                                    "direction": feature.get("direction", ""),
                                    "xai_context": str(xai_context),
                                })
                        else:
                            has_observable_features = explanation.get("has_observable_features")
                            if has_observable_features is False:
                                explanation_status = explanation.get("status", "NO_OBSERVABLE_FEATURES")
                                feature_name = explanation_status
                            else:
                                explanation_status = explanation.get("status", "NO_FEATURE_EFFECT")
                                feature_name = explanation_status

                            xai_csv_rows.append({
                                "explanation_status": explanation_status,
                                "episode": episode + 1,
                                "step": step + 1,
                                "agent_id": agent_id,
                                "selected_action": explanation.get("selected_action_name", ""),
                                "action_probability": explanation.get("selected_probability", ""),
                                "decision_margin": explanation.get("decision_margin", ""),
                                "feature_index": "",
                                "feature_name": feature_name,
                                "feature_value": "",
                                "perturbed_probability": "",
                                "effect": "",
                                "importance": "",
                                "direction": explanation.get("interpretation", ""),
                                "xai_context": str(xai_context),
                            })

                    except Exception as error:

                        print()
                        print(
                            f"XAI error for "
                            f"{agent_id}: {error}"
                        )

            # ------------------------------------------------
            # Environment step
            # ------------------------------------------------

            step_result = env.step(
                actions
            )

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

            step_reward = (
                calculate_step_reward(
                    rewards,
                    agent_ids
                )
            )

            total_reward += step_reward
            steps_taken += 1

            print()
            print(
                f"Step reward: {step_reward:.4f}"
            )

            # ------------------------------------------------
            # Update state
            # ------------------------------------------------

            observations = (
                next_observations
            )

            action_masks = (
                next_action_masks
            )

            # ------------------------------------------------
            # Termination
            # ------------------------------------------------

            if is_terminated(
                terminated,
                agent_ids
            ):

                print()
                print(
                    "Episode terminated."
                )

                break

        episode_rewards.append(
            total_reward
        )

        print()
        print(
            f"Episode {episode + 1} "
            f"completed | "
            f"Reward: {total_reward:.4f} | "
            f"Steps: {steps_taken}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "                 MAPPO + XAI RESULTS"
    )
    print("=" * 70)

    print(
        f"Number of episodes : {NUM_EPISODES}"
    )

    print(
        f"Mean reward        : {mean_reward:.4f}"
    )

    print(
        f"Std reward         : {std_reward:.4f}"
    )

    print(
        f"Best reward        : {best_reward:.4f}"
    )

    print(
        f"Worst reward       : {worst_reward:.4f}"
    )

    print(
        f"XAI explanations   : {explanation_count}"
    )

    print("=" * 70)

    # Write the XAI results beside this script, independent of the
    # directory from which Python was launched.
    xai_csv_path = os.path.join(os.path.dirname(__file__), "xai_results.csv")
    csv_columns = [
        "explanation_status", "episode", "step", "agent_id", "selected_action",
        "action_probability", "decision_margin", "feature_index",
        "feature_name", "feature_value", "perturbed_probability",
        "effect", "importance", "direction","xai_context",
    ]
    with open(xai_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(xai_csv_rows)
    print(f"XAI results CSV     : {xai_csv_path}")
    print(f"XAI feature rows    : {len(xai_csv_rows)}")

    env.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    evaluate_mappo()
