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
        observation,
        action_mask=None
    ):

        logits = self.forward(
            observation
        )

        # --------------------------------------------------------
        # Apply action mask
        # --------------------------------------------------------

        if action_mask is not None:

            # 0 = unavailable
            # 1 = available
            invalid_actions = (
                action_mask <= 0
            )

            logits = logits.masked_fill(
                invalid_actions,
                -1e9
            )

        return Categorical(
            logits=logits
        )

    def get_action(
        self,
        observation,
        action_mask=None,
        deterministic=False
    ):

        distribution = self.get_distribution(
            observation,
            action_mask
        )

        if deterministic:

            action = torch.argmax(
                distribution.logits,
                dim=-1
            )

        else:

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
        hidden_dim=128,
        device=None
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

        # --------------------------------------------------------
        # Device
        # --------------------------------------------------------

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )

        # --------------------------------------------------------
        # One independent actor per agent
        # --------------------------------------------------------

        self.actors = [

            Actor(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim
            ).to(self.device)

            for _ in range(num_agents)
        ]

        # --------------------------------------------------------
        # Centralized critic
        # --------------------------------------------------------

        self.critic = CentralizedCritic(

            observation_dim=observation_dim,

            num_agents=num_agents,

            hidden_dim=hidden_dim

        ).to(self.device)

    # ============================================================
    # Action selection
    # ============================================================

    @torch.no_grad()
    def select_actions(
        self,
        observations,
        action_masks=None,
        deterministic=True
    ):
        """
        Select one action for every agent.

        Parameters
        ----------
        observations:
            Dictionary:

                {
                    "blue_agent_0": observation,
                    "blue_agent_1": observation,
                    ...
                }

            OR a tensor of shape:

                [num_agents, observation_dim]

        action_masks:
            Dictionary:

                {
                    "blue_agent_0": mask,
                    ...
                }

            where 1 = valid and 0 = invalid.

        deterministic:
            If True, select the highest-probability action.
            If False, sample from the policy.

        Returns
        -------
        actions:
            Dictionary mapping agent IDs to integer actions.
        """

        # --------------------------------------------------------
        # Dictionary input
        # --------------------------------------------------------

        if isinstance(
            observations,
            dict
        ):

            agent_ids = list(
                observations.keys()
            )

            observation_list = []

            for agent_id in agent_ids:

                observation_list.append(
                    observations[agent_id]
                )

        # --------------------------------------------------------
        # Tensor input
        # --------------------------------------------------------

        else:

            agent_ids = [
                f"blue_agent_{i}"
                for i in range(
                    self.num_agents
                )
            ]

            observation_list = observations

        # --------------------------------------------------------
        # Convert observations to tensor
        # --------------------------------------------------------

        if isinstance(
            observation_list,
            torch.Tensor
        ):

            observation_tensor = (
                observation_list.to(
                    self.device,
                    dtype=torch.float32
                )
            )

        else:

            observation_tensor = torch.tensor(
                observation_list,
                dtype=torch.float32,
                device=self.device
            )

        # Make sure shape is:
        #
        # [num_agents, observation_dim]

        if observation_tensor.dim() == 1:

            observation_tensor = (
                observation_tensor.unsqueeze(0)
            )

        # --------------------------------------------------------
        # Select actions
        # --------------------------------------------------------

        actions = {}

        for i, agent_id in enumerate(
            agent_ids
        ):

            observation = (
                observation_tensor[i]
                .unsqueeze(0)
            )

            # ----------------------------------------------------
            # Get action mask
            # ----------------------------------------------------

            action_mask = None

            if action_masks is not None:

                if isinstance(
                    action_masks,
                    dict
                ):

                    action_mask = (
                        action_masks.get(
                            agent_id
                        )
                    )

                elif isinstance(
                    action_masks,
                    torch.Tensor
                ):

                    action_mask = (
                        action_masks[i]
                    )

            # ----------------------------------------------------
            # Convert mask to tensor
            # ----------------------------------------------------

            if action_mask is not None:

                if not isinstance(
                    action_mask,
                    torch.Tensor
                ):

                    action_mask = torch.tensor(
                        action_mask,
                        dtype=torch.float32,
                        device=self.device
                    )

                else:

                    action_mask = (
                        action_mask.to(
                            self.device,
                            dtype=torch.float32
                        )
                    )

                if action_mask.dim() == 1:

                    action_mask = (
                        action_mask.unsqueeze(0)
                    )

            # ----------------------------------------------------
            # Get action
            # ----------------------------------------------------

            action, _ = self.actors[i].get_action(

                observation,

                action_mask=action_mask,

                deterministic=deterministic

            )

            actions[agent_id] = (
                int(action.item())
            )

        return actions

    # ============================================================
    # Evaluate actions
    # ============================================================

    @torch.no_grad()
    def evaluate_actions(
        self,
        observations,
        actions,
        action_masks=None
    ):
        """
        Evaluate log probabilities and entropy
        for selected actions.
        """

        if isinstance(
            observations,
            dict
        ):

            agent_ids = list(
                observations.keys()
            )

            observation_list = [
                observations[agent_id]
                for agent_id in agent_ids
            ]

        else:

            agent_ids = [
                f"blue_agent_{i}"
                for i in range(
                    self.num_agents
                )
            ]

            observation_list = observations

        observation_tensor = torch.tensor(
            observation_list,
            dtype=torch.float32,
            device=self.device
        )

        if observation_tensor.dim() == 1:

            observation_tensor = (
                observation_tensor.unsqueeze(0)
            )

        log_probabilities = []
        entropies = []

        for i, agent_id in enumerate(
            agent_ids
        ):

            observation = (
                observation_tensor[i]
                .unsqueeze(0)
            )

            action_mask = None

            if action_masks is not None:

                if isinstance(
                    action_masks,
                    dict
                ):

                    action_mask = (
                        action_masks.get(
                            agent_id
                        )
                    )

                else:

                    action_mask = (
                        action_masks[i]
                    )

                if action_mask is not None:

                    action_mask = torch.tensor(
                        action_mask,
                        dtype=torch.float32,
                        device=self.device
                    )

                    if action_mask.dim() == 1:

                        action_mask = (
                            action_mask.unsqueeze(0)
                        )

            distribution = (
                self.actors[i].get_distribution(
                    observation,
                    action_mask
                )
            )

            if isinstance(
                actions,
                dict
            ):

                action = actions[
                    agent_id
                ]

            else:

                action = actions[i]

            if not isinstance(
                action,
                torch.Tensor
            ):

                action = torch.tensor(
                    [action],
                    dtype=torch.long,
                    device=self.device
                )

            else:

                action = action.to(
                    self.device,
                    dtype=torch.long
                )

                if action.dim() == 0:

                    action = action.unsqueeze(0)

            log_probabilities.append(
                distribution.log_prob(
                    action
                )
            )

            entropies.append(
                distribution.entropy()
            )

        return (
            torch.cat(
                log_probabilities
            ),
            torch.cat(
                entropies
            )
        )

    # ============================================================
    # Value estimation
    # ============================================================

    @torch.no_grad()
    def get_value(
        self,
        observations
    ):

        if isinstance(
            observations,
            dict
        ):

            observation_list = [
                observations[agent_id]
                for agent_id in
                observations.keys()
            ]

        else:

            observation_list = observations

        observations_tensor = torch.tensor(
            observation_list,
            dtype=torch.float32,
            device=self.device
        )

        if observations_tensor.dim() == 2:

            observations_tensor = (
                observations_tensor.unsqueeze(0)
            )

        return self.critic(
            observations_tensor
        )

    # ============================================================
    # Save model
    # ============================================================

    def save(
        self,
        path
    ):

        checkpoint = {

            "observation_dim":
                self.observation_dim,

            "action_dim":
                self.action_dim,

            "num_agents":
                self.num_agents,

            "actors": [
                actor.state_dict()
                for actor in self.actors
            ],

            "critic":
                self.critic.state_dict()
        }

        torch.save(
            checkpoint,
            path
        )

    # ============================================================
    # Load model
    # ============================================================

    def load(
        self,
        path,
        map_location=None
    ):

        if map_location is None:

            map_location = self.device

        checkpoint = torch.load(
            path,
            map_location=map_location
        )

        # --------------------------------------------------------
        # Load actors
        # --------------------------------------------------------

        if "actors" in checkpoint:

            for i, state_dict in enumerate(
                checkpoint["actors"]
            ):

                if i < len(
                    self.actors
                ):

                    self.actors[i].load_state_dict(
                        state_dict
                    )

        # --------------------------------------------------------
        # Load critic
        # --------------------------------------------------------

        if "critic" in checkpoint:

            self.critic.load_state_dict(
                checkpoint["critic"]
            )

        # --------------------------------------------------------
        # Evaluation mode
        # --------------------------------------------------------

        for actor in self.actors:

            actor.eval()

        self.critic.eval()

        return self


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

    print(
        "Device:",
        model.device
    )

    observations = torch.randn(
        num_agents,
        observation_dim
    )

    # ---------------------------------------------------------
    # Test individual actors
    # ---------------------------------------------------------

    print(
        "\nTesting actors..."
    )

    for i, actor in enumerate(
        model.actors
    ):

        observation = (
            observations[i]
            .unsqueeze(0)
            .to(model.device)
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

    # ---------------------------------------------------------
    # Test centralized critic
    # ---------------------------------------------------------

    print(
        "\nTesting centralized critic..."
    )

    centralized_observations = (
        observations
        .unsqueeze(0)
        .to(model.device)
    )

    value = model.critic(
        centralized_observations
    )

    print(
        "Global state value:",
        value.item()
    )

    # ---------------------------------------------------------
    # Test select_actions
    # ---------------------------------------------------------

    print(
        "\nTesting select_actions..."
    )

    observation_dict = {

        f"blue_agent_{i}":
            observations[i].numpy()

        for i in range(
            num_agents
        )
    }

    action_masks = {

        f"blue_agent_{i}":
            [1.0] * action_dim

        for i in range(
            num_agents
        )
    }

    actions = model.select_actions(

        observation_dict,

        action_masks=action_masks,

        deterministic=True
    )

    print(
        "Selected actions:",
        actions
    )

    print(
        "\n===== MODEL TEST PASSED ====="
    )