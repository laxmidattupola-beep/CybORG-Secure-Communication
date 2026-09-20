import numpy as np

from CybORG.mappo.cyborg_env import CybORGMAPPOEnv


NUM_EPISODES = 50
STEPS_PER_EPISODE = 50


def evaluate_random():

    env = CybORGMAPPOEnv(
        steps=STEPS_PER_EPISODE
    )

    agent_ids = env.agent_ids
    action_dim = env.action_dim

    episode_rewards = []

    print("===== RANDOM POLICY EVALUATION =====")
    print(f"Agents: {agent_ids}")
    print(f"Action dimension: {action_dim}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Steps per episode: {STEPS_PER_EPISODE}")
    print()

    for episode in range(NUM_EPISODES):

        observations = env.reset()

        total_reward = 0.0

        for step in range(STEPS_PER_EPISODE):

            # Generate a random action for each agent
            actions = {
                agent_id: np.random.randint(0, action_dim)
                for agent_id in agent_ids
            }

            # CybORG MAPPO environment returns 4 values
            observations, rewards, done, info = env.step(actions)

            # Calculate average reward of all 5 agents
            team_reward = np.mean(
                [
                    rewards[agent_id]
                    for agent_id in agent_ids
                ]
            )

            total_reward += team_reward

            if done:
                break

        episode_rewards.append(total_reward)

        print(
            f"Episode {episode + 1:02d} | "
            f"Team Reward: {total_reward:.3f}"
        )

    env.close()

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    best_reward = np.max(episode_rewards)
    worst_reward = np.min(episode_rewards)

    print()
    print("===== RANDOM POLICY RESULTS =====")
    print(f"Mean Reward : {mean_reward:.3f}")
    print(f"Std Reward  : {std_reward:.3f}")
    print(f"Best Reward : {best_reward:.3f}")
    print(f"Worst Reward: {worst_reward:.3f}")
    print("=================================")


if __name__ == "__main__":
    evaluate_random()