import numpy as np
import torch

from CybORG.mappo.cyborg_env import CybORGMAPPOEnv
from CybORG.mappo.mappo import MAPPO


def test_rollout():

    print("=" * 60)
    print("===== MAPPO ROLLOUT TEST =====")
    print("=" * 60)

    # ---------------------------------------------------------
    # Create environment
    # ---------------------------------------------------------

    env = CybORGMAPPOEnv(
        steps=50
    )

    agent_ids = env.get_agent_ids()
    observation_dim = env.get_observation_dim()
    action_dim = env.get_action_dim()
    num_agents = env.get_num_agents()

    print("Agents:", agent_ids)
    print("Observation dimension:", observation_dim)
    print("Action dimension:", action_dim)

    # ---------------------------------------------------------
    # Create MAPPO model
    # ---------------------------------------------------------

    model = MAPPO(
        observation_dim=observation_dim,
        action_dim=action_dim,
        num_agents=num_agents
    )

    print("MAPPO model created")

    # ---------------------------------------------------------
    # Reset environment
    # ---------------------------------------------------------

    observations = env.reset()

    total_rewards = {
        agent_id: 0.0
        for agent_id in agent_ids
    }

    # ---------------------------------------------------------
    # Rollout
    # ---------------------------------------------------------

    for step in range(1, 51):

        print(f"\nStep {step}")

        actions = {}

        # -----------------------------------------------------
        # Select action for every blue agent
        # -----------------------------------------------------

        for agent_index, agent_id in enumerate(agent_ids):

            observation = torch.tensor(
                observations[agent_id],
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():

                action, log_probability = (
                    model.actors[agent_index].get_action(
                        observation
                    )
                )

            actions[agent_id] = action.item()

        print(
            "Actions:",
            actions
        )

        # -----------------------------------------------------
        # Execute actions
        # -----------------------------------------------------

        (
            next_observations,
            rewards,
            terminated,
            info
        ) = env.step(actions)

        print(
            "Rewards:",
            rewards
        )

        # -----------------------------------------------------
        # Update total rewards
        # -----------------------------------------------------

        for agent_id in agent_ids:

            total_rewards[agent_id] += (
                rewards.get(
                    agent_id,
                    0.0
                )
            )

        print(
            "Total reward:",
            total_rewards
        )

        # -----------------------------------------------------
        # Move to next observations
        # -----------------------------------------------------

        observations = next_observations

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # cyborg_env.step() returns a single boolean:
        #
        # terminated = True / False
        #
        # So we DO NOT use terminated.values()
        # -----------------------------------------------------

        if terminated:

            print(
                "\nEnvironment terminated."
            )

            break

    # ---------------------------------------------------------
    # Final results
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("             ROLLOUT RESULTS")
    print("=" * 60)

    print(
        "Total rewards:"
    )

    for agent_id in agent_ids:

        print(
            f"{agent_id}: "
            f"{total_rewards[agent_id]:.3f}"
        )

    # Team reward

    team_reward = np.mean(
        list(
            total_rewards.values()
        )
    )

    print(
        f"\nTeam average reward: "
        f"{team_reward:.3f}"
    )

    print("=" * 60)
    print("===== ROLLOUT TEST PASSED =====")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    test_rollout()