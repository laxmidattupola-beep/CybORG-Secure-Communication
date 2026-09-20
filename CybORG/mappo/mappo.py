import torch
import torch.nn as nn
from torch.distributions import Categorical


# ================================================================
# ACTOR
# ================================================================

class Actor(nn.Module):

    def __init__(
        self,
        observation_dim,
        action_dim=352,
        hidden_dim=128,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                observation_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                action_dim,
            ),
        )

    # ============================================================
    # FORWARD
    # ============================================================

    def forward(
        self,
        observation,
    ):

        return self.network(
            observation
        )

    # ============================================================
    # APPLY ACTION MASK
    # ============================================================

    def apply_action_mask(
        self,
        logits,
        action_mask,
    ):
        """
        Prevent invalid CybORG actions from
        being selected.

        action_mask:

            1 -> valid action
            0 -> invalid action

        Invalid actions receive a very large
        negative logit so their probability
        becomes approximately zero.
        """

        if action_mask is None:

            return logits

        # Make sure mask is a tensor

        if not torch.is_tensor(
            action_mask
        ):

            action_mask = torch.tensor(
                action_mask,
                dtype=torch.float32,
                device=logits.device,
            )

        else:

            action_mask = action_mask.to(
                device=logits.device,
                dtype=torch.float32,
            )

        # --------------------------------------------------------
        # Handle single observation
        # --------------------------------------------------------

        if logits.dim() == 1:

            if action_mask.dim() > 1:

                action_mask = (
                    action_mask.squeeze(0)
                )

        # --------------------------------------------------------
        # Handle batch
        # --------------------------------------------------------

        else:

            if action_mask.dim() == 1:

                action_mask = (
                    action_mask.unsqueeze(0)
                )

        # --------------------------------------------------------
        # Invalid actions -> huge negative value
        # --------------------------------------------------------

        masked_logits = logits.masked_fill(
            action_mask <= 0,
            -1e9,
        )

        return masked_logits

    # ============================================================
    # GET DISTRIBUTION
    # ============================================================

    def get_distribution(
        self,
        observation,
        action_mask=None,
    ):

        logits = self.forward(
            observation
        )

        logits = self.apply_action_mask(
            logits,
            action_mask,
        )

        return Categorical(
            logits=logits
        )

    # ============================================================
    # SAMPLE ACTION
    # ============================================================

    def get_action(
        self,
        observation,
        action_mask=None,
    ):
        """
        Sample an action only from valid
        CybORG actions.
        """

        distribution = (
            self.get_distribution(
                observation,
                action_mask,
            )
        )

        action = (
            distribution.sample()
        )

        log_probability = (
            distribution.log_prob(
                action
            )
        )

        return (
            action,
            log_probability,
        )

    # ============================================================
    # LOG PROBABILITY
    # ============================================================

    def get_log_probability(
        self,
        observation,
        action,
        action_mask=None,
    ):
        """
        Calculate log probability of an
        already-selected action.

        IMPORTANT:

        The same action mask used during
        action selection must also be used
        during PPO updates.
        """

        distribution = (
            self.get_distribution(
                observation,
                action_mask,
            )
        )

        return (
            distribution.log_prob(
                action
            )
        )

    # ============================================================
    # ENTROPY
    # ============================================================

    def get_entropy(
        self,
        observation,
        action_mask=None,
    ):

        distribution = (
            self.get_distribution(
                observation,
                action_mask,
            )
        )

        return distribution.entropy()


# ================================================================
# CENTRALIZED CRITIC
# ================================================================

class CentralizedCritic(nn.Module):

    def __init__(
        self,
        observation_dim,
        num_agents=5,
        hidden_dim=128,
    ):

        super().__init__()

        input_dim = (
            observation_dim
            * num_agents
        )

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    # ============================================================
    # FORWARD
    # ============================================================

    def forward(
        self,
        observations,
    ):

        batch_size = (
            observations.shape[0]
        )

        observations = (
            observations.reshape(
                batch_size,
                -1,
            )
        )

        value = self.network(
            observations
        )

        return value.squeeze(
            -1
        )


# ================================================================
# MAPPO
# ================================================================

class MAPPO:

    def __init__(
        self,
        observation_dim,
        action_dim=352,
        num_agents=5,
        hidden_dim=128,
        learning_rate=3e-4,
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
        # ACTORS
        # --------------------------------------------------------

        self.actors = [

            Actor(
                observation_dim=(
                    observation_dim
                ),

                action_dim=(
                    action_dim
                ),

                hidden_dim=(
                    hidden_dim
                ),
            )

            for _ in range(
                num_agents
            )
        ]

        # --------------------------------------------------------
        # CENTRALIZED CRITIC
        # --------------------------------------------------------

        self.critic = (
            CentralizedCritic(
                observation_dim=(
                    observation_dim
                ),

                num_agents=(
                    num_agents
                ),

                hidden_dim=(
                    hidden_dim
                ),
            )
        )

        # --------------------------------------------------------
        # ACTOR OPTIMIZERS
        # --------------------------------------------------------

        self.actor_optimizers = [

            torch.optim.Adam(
                actor.parameters(),
                lr=learning_rate,
            )

            for actor in self.actors
        ]

        # --------------------------------------------------------
        # CRITIC OPTIMIZER
        # --------------------------------------------------------

        self.critic_optimizer = (
            torch.optim.Adam(
                self.critic.parameters(),
                lr=learning_rate,
            )
        )

    # ============================================================
    # SELECT ACTIONS
    # ============================================================

    def select_actions(
        self,
        observations,
        action_masks,
    ):
        """
        Select one action for every
        Blue agent.

        observations:
            {
                "blue_agent_0": np.array,
                ...
            }

        action_masks:
            {
                "blue_agent_0": np.array,
                ...
            }

        Returns:

            actions
            log_probabilities
        """

        actions = {}

        log_probabilities = {}

        # --------------------------------------------------------
        # AGENT ORDER
        # --------------------------------------------------------

        agent_ids = list(
            observations.keys()
        )

        # --------------------------------------------------------
        # SELECT ACTION FOR EACH AGENT
        # --------------------------------------------------------

        for index, agent_id in enumerate(
            agent_ids
        ):

            observation = (
                observations[
                    agent_id
                ]
            )

            action_mask = (
                action_masks[
                    agent_id
                ]
            )

            # ----------------------------------------------------
            # Convert observation
            # ----------------------------------------------------

            if not torch.is_tensor(
                observation
            ):

                observation = torch.tensor(
                    observation,
                    dtype=torch.float32,
                )

            else:

                observation = observation.float()

            # ----------------------------------------------------
            # Convert mask
            # ----------------------------------------------------

            if not torch.is_tensor(
                action_mask
            ):

                action_mask = torch.tensor(
                    action_mask,
                    dtype=torch.float32,
                )

            # ----------------------------------------------------
            # Actor
            # ----------------------------------------------------

            actor = self.actors[
                index
            ]

            # ----------------------------------------------------
            # Sample action
            # ----------------------------------------------------

            action, log_probability = (
                actor.get_action(
                    observation,
                    action_mask,
                )
            )

            actions[
                agent_id
            ] = int(
                action.item()
            )

            log_probabilities[
                agent_id
            ] = float(
                log_probability.item()
            )

        return (
            actions,
            log_probabilities,
        )