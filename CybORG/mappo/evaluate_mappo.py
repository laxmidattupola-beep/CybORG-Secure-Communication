import os
import numpy as np
import torch

from CybORG.mappo.cyborg_env import CybORGMAPPOEnv
from CybORG.mappo.mappo import MAPPO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "mappo_model.pt"
)

NUM_EPISODES = 50
STEPS_PER_EPISODE = 50


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_current_action_masks(env):
    """
    Get the current action mask for every agent.
    """

    action_masks = {}

    for agent_id in env.agent_ids:

        action_masks[agent_id] = env.get_action_mask(
            agent_id
        )

    return action_masks


def extract_reset_data(reset_result, env):
    """
    Normalize the current environment reset output.
    """

    if isinstance(reset_result, tuple):

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

    elif isinstance(reset_result, dict):

        observations = reset_result

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


# ============================================================
# EVALUATION
# ============================================================

def evaluate_mappo():

    print("=" * 60)
    print("             MAPPO EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = CybORGMAPPOEnv(
        steps=STEPS_PER_EPISODE
    )

    agent_ids = env.agent_ids

    num_agents = len(agent_ids)

    observation_dim = env.observation_dim
    action_dim = env.action_dim

    print(f"Agents: {agent_ids}")
    print(f"Number of agents: {num_agents}")
    print(f"Observation dimension: {observation_dim}")
    print(f"Action dimension: {action_dim}")

    # --------------------------------------------------------
    # Create MAPPO
    # --------------------------------------------------------

    mappo = MAPPO(
        observation_dim=observation_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        hidden_dim=128,
    )

    # Move actors and critic to device
    for actor in mappo.actors:
        actor.to(device)

    mappo.critic.to(device)

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        print()
        print("ERROR: Trained MAPPO model was not found.")
        print(f"Expected path: {MODEL_PATH}")

        env.close()
        return

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        # ----------------------------------------------------
        # Actors
        # ----------------------------------------------------

        if "actors" in checkpoint:

            for i, actor_state in enumerate(
                checkpoint["actors"]
            ):

                if i < len(mappo.actors):

                    mappo.actors[i].load_state_dict(
                        actor_state
                    )

        elif "actor_states" in checkpoint:

            for i, actor_state in enumerate(
                checkpoint["actor_states"]
            ):

                if i < len(mappo.actors):

                    mappo.actors[i].load_state_dict(
                        actor_state
                    )

        elif "actors_state_dict" in checkpoint:

            for i, actor_state in enumerate(
                checkpoint["actors_state_dict"]
            ):

                if i < len(mappo.actors):

                    mappo.actors[i].load_state_dict(
                        actor_state
                    )

        # ----------------------------------------------------
        # Critic
        # ----------------------------------------------------

        if "critic" in checkpoint:

            mappo.critic.load_state_dict(
                checkpoint["critic"]
            )

        elif "critic_state_dict" in checkpoint:

            mappo.critic.load_state_dict(
                checkpoint["critic_state_dict"]
            )

    else:

        print()
        print("ERROR: Unsupported checkpoint format.")
        print(f"Checkpoint type: {type(checkpoint)}")

        env.close()
        return

    print()
    print("Trained MAPPO model loaded successfully.")
    print()

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    for actor in mappo.actors:
        actor.eval()

    mappo.critic.eval()

    # --------------------------------------------------------
    # Evaluation episodes
    # --------------------------------------------------------

    episode_rewards = []

    for episode in range(NUM_EPISODES):

        # ----------------------------------------------------
        # Reset environment
        # ----------------------------------------------------

        reset_result = env.reset()

        observations, action_masks = extract_reset_data(
            reset_result,
            env
        )

        total_reward = 0.0
        steps_taken = 0

        # ----------------------------------------------------
        # Episode loop
        # ----------------------------------------------------

        for step in range(STEPS_PER_EPISODE):

            # ------------------------------------------------
            # Select actions
            #
            # IMPORTANT:
            # select_actions() returns:
            #
            #     actions, log_probs
            #
            # Evaluation only needs the actions.
            # ------------------------------------------------

            with torch.no_grad():

                action_result = mappo.select_actions(
                    observations,
                    action_masks
                )

            # Handle tuple return
            if isinstance(action_result, tuple):

                actions = action_result[0]

            else:

                actions = action_result

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
            # Calculate average reward
            # ------------------------------------------------

            step_reward = 0.0

            for agent_id in agent_ids:

                if agent_id in rewards:

                    step_reward += float(
                        rewards[agent_id]
                    )

            if num_agents > 0:

                step_reward /= num_agents

            total_reward += step_reward

            steps_taken += 1

            # ------------------------------------------------
            # Update state
            # ------------------------------------------------

            observations = next_observations

            action_masks = next_action_masks

            # ------------------------------------------------
            # Check termination
            # ------------------------------------------------

            if isinstance(terminated, dict):

                if all(
                    bool(
                        terminated.get(
                            agent_id,
                            False
                        )
                    )
                    for agent_id in agent_ids
                ):

                    break

            elif bool(terminated):

                break

        # ----------------------------------------------------
        # Store episode reward
        # ----------------------------------------------------

        episode_rewards.append(
            total_reward
        )

        print(
            f"Episode {episode + 1:02d}/{NUM_EPISODES} "
            f"| Reward: {total_reward:.4f} "
            f"| Steps: {steps_taken}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    episode_rewards = np.asarray(
        episode_rewards,
        dtype=np.float64
    )

    mean_reward = np.mean(
        episode_rewards
    )

    std_reward = np.std(
        episode_rewards
    )

    best_reward = np.max(
        episode_rewards
    )

    worst_reward = np.min(
        episode_rewards
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("             MAPPO EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"Number of episodes : {NUM_EPISODES}"
    )

    print(
        f"Mean reward       : {mean_reward:.4f}"
    )

    print(
        f"Std reward        : {std_reward:.4f}"
    )

    print(
        f"Best reward       : {best_reward:.4f}"
    )

    print(
        f"Worst reward      : {worst_reward:.4f}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    env.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate_mappo()