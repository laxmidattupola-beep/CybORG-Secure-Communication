import torch
import torch.nn as nn
from torch.distributions import Categorical


# ============================================================
# Actor Network
# ============================================================

class Actor(nn.Module):

    def __init__(
        self,
        observation_dim,
        action_dim=8,
        hidden_dim=128
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                observation_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                action_dim
            )
        )

    def forward(
        self,
        observation
    ):

        return self.network(
            observation
        )

    def get_distribution(
        self,
        observation
    ):

        logits = self.forward(
            observation
        )

        return Categorical(
            logits=logits
        )

    def get_action(
        self,
        observation
    ):

        distribution = self.get_distribution(
            observation
        )

        action = distribution.sample()

        log_probability = (
            distribution.log_prob(
                action
            )
        )

        return (
            action,
            log_probability
        )


# ============================================================
# Centralized Critic
# ============================================================

class CentralizedCritic(nn.Module):

    def __init__(
        self,
        observation_dim,
        num_agents=5,
        hidden_dim=128
    ):

        super().__init__()

        input_dim = (
            observation_dim
            * num_agents
        )

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(
        self,
        observations
    ):

        batch_size = (
            observations.shape[0]
        )

        observations = observations.reshape(
            batch_size,
            -1
        )

        value = self.network(
            observations
        )

        return value.squeeze(-1)


# ============================================================
# MAPPO
# ============================================================

class MAPPO:

    def __init__(
        self,
        observation_dim,
        action_dim=8,
        num_agents=5,
        hidden_dim=128
    ):

        self.observation_dim = (
            observation_dim
        )

        self.action_dim = (
            action_dim
        )

        self.num_agents = (
            num_agents
        )

        # One independent actor per agent
        self.actors = [

            Actor(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim
            )

            for _ in range(num_agents)
        ]

        # One centralized critic
        self.critic = CentralizedCritic(

            observation_dim=observation_dim,

            num_agents=num_agents,

            hidden_dim=hidden_dim
        )


# ============================================================
# Model Test
# ============================================================

if __name__ == "__main__":

    print(
        "===== MAPPO MODEL TEST ====="
    )

    observation_dim = 256

    action_dim = 8

    num_agents = 5

    model = MAPPO(

        observation_dim=observation_dim,

        action_dim=action_dim,

        num_agents=num_agents
    )

    print(
        "Number of agents:",
        model.num_agents
    )

    print(
        "Observation dimension:",
        model.observation_dim
    )

    print(
        "Action dimension:",
        model.action_dim
    )

    observations = torch.randn(
        num_agents,
        observation_dim
    )

    print(
        "\nTesting actors..."
    )

    for i, actor in enumerate(
        model.actors
    ):

        observation = (
            observations[i]
            .unsqueeze(0)
        )

        action, log_probability = (
            actor.get_action(
                observation
            )
        )

        print(
            f"Agent {i}: "
            f"action={action.item()}, "
            f"log_prob="
            f"{log_probability.item():.4f}"
        )

    print(
        "\nTesting centralized critic..."
    )

    centralized_observations = (
        observations.unsqueeze(0)
    )

    value = model.critic(
        centralized_observations
    )

    print(
        "Global state value:",
        value.item()
    )

    print(
        "\n===== MODEL TEST PASSED ====="
    )