import numpy as np
import torch

from CybORG.mappo.cyborg_env import CybORGMAPPOEnv
from CybORG.mappo.mappo import MAPPO


NUM_EPISODES = 10
STEPS_PER_EPISODE = 50

MODEL_PATH = "CybORG/mappo/mappo_model.pt"


def evaluate_mappo():

    print("=" * 60)
    print("             MAPPO EVALUATION")
    print("=" * 60)

    # Create environment
    env = CybORGMAPPOEnv(
        steps=STEPS_PER_EPISODE
    )

    agent_ids = env.get_agent_ids()
    num_agents = env.get_num_agents()
    observation_dim = env.get_observation_dim()
    action_dim = env.get_action_dim()

    print("Agents:", agent_ids)
    print("Number of agents:", num_agents)
    print("Observation dimension:", observation_dim)
    print("Action dimension:", action_dim)

    # Create MAPPO model
    model = MAPPO(
        observation_dim=observation_dim,
        action_dim=action_dim,
        num_agents=num_agents
    )

    # Load trained model
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    for i, actor in enumerate(model.actors):
        actor.load_state_dict(
            checkpoint["actors"][i]
        )
        actor.eval()

    model.critic.load_state_dict(
        checkpoint["critic"]
    )
    model.critic.eval()

    print("\nLoaded trained MAPPO model.")

    episode_rewards = []

    # Evaluation
    for episode in range(1, NUM_EPISODES + 1):

        observations = env.reset()

        total_team_reward = 0.0

        done = False

        for step in range(STEPS_PER_EPISODE):

            actions = {}

            # Deterministic action selection
            for agent_index, agent_id in enumerate(agent_ids):

                observation = torch.tensor(
                    observations[agent_id],
                    dtype=torch.float32
                ).unsqueeze(0)

                with torch.no_grad():

                    logits = model.actors[
                        agent_index
                    ](observation)

                    action = torch.argmax(
                        logits,
                        dim=-1
                    ).item()

                actions[agent_id] = action

            # Execute actions
            (
                next_observations,
                rewards,
                done,
                info
            ) = env.step(actions)

            # SAME team reward calculation
            # used during MAPPO training
            team_reward = np.mean(
                [
                    rewards[agent_id]
                    for agent_id in agent_ids
                ]
            )

            total_team_reward += team_reward

            observations = next_observations

            if done:
                break

        episode_rewards.append(
            total_team_reward
        )

        print(
            f"Episode {episode:02d}: "
            f"Steps = {step + 1}, "
            f"Team reward = {total_team_reward:.3f}"
        )

    env.close()

    # Statistics
    mean_reward = np.mean(
        episode_rewards
    )

    std_reward = np.std(
        episode_rewards
    )

    print("\n" + "=" * 60)
    print("              MAPPO RESULTS")
    print("=" * 60)

    print(
        f"Mean reward: {mean_reward:.3f}"
    )

    print(
        f"Std reward: {std_reward:.3f}"
    )

    print(
        f"Best episode: "
        f"{np.max(episode_rewards):.3f}"
    )

    print(
        f"Worst episode: "
        f"{np.min(episode_rewards):.3f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    evaluate_mappo()