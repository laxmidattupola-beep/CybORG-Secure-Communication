import os
import csv
import numpy as np
import torch
import torch.nn.functional as F

from CybORG.mappo.cyborg_env import CybORGMAPPOEnv
from CybORG.mappo.mappo import MAPPO


# ================================================================
# CONFIGURATION
# ================================================================

NUM_EPISODES = 50

STEPS_PER_EPISODE = 50

LEARNING_RATE = 3e-4

GAMMA = 0.99

GAE_LAMBDA = 0.95

CLIP_EPSILON = 0.20

PPO_EPOCHS = 4

ENTROPY_COEFFICIENT = 0.01

VALUE_COEFFICIENT = 0.5

HIDDEN_DIM = 128

MODEL_PATH = (
    "CybORG/mappo/mappo_model.pt"
)

HISTORY_PATH = (
    "CybORG/mappo/training_history.csv"
)


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# GAE
# ================================================================

def calculate_gae(
    rewards,
    values,
    next_value,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
):
    """
    Generalized Advantage Estimation.

    rewards:
        List of rewards.

    values:
        Value estimates for each state.

    next_value:
        Value estimate of final next state.
    """

    advantages = []

    gae = 0.0

    values_extended = (
        values
        + [next_value]
    )

    for step in reversed(
        range(
            len(rewards)
        )
    ):

        delta = (
            rewards[step]
            + gamma
            * values_extended[
                step + 1
            ]
            - values_extended[
                step
            ]
        )

        gae = (
            delta
            + gamma
            * gae_lambda
            * gae
        )

        advantages.insert(
            0,
            gae
        )

    return advantages


# ================================================================
# TRAINING
# ================================================================

def train():

    print()
    print(
        "========================================"
    )

    print(
        "        MAPPO TRAINING STARTED"
    )

    print(
        "========================================"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Episodes: {NUM_EPISODES}"
    )

    print(
        f"Steps per episode: "
        f"{STEPS_PER_EPISODE}"
    )

    print()

    # ------------------------------------------------------------
    # ENVIRONMENT
    # ------------------------------------------------------------

    env = CybORGMAPPOEnv(
        steps=STEPS_PER_EPISODE
    )

    print(
        f"Agents: {env.agent_ids}"
    )

    print(
        f"Observation dimension: "
        f"{env.observation_dim}"
    )

    print(
        f"Action dimension: "
        f"{env.action_dim}"
    )

    print()

    # ------------------------------------------------------------
    # MAPPO
    # ------------------------------------------------------------

    mappo = MAPPO(
        observation_dim=(
            env.observation_dim
        ),

        action_dim=(
            env.action_dim
        ),

        num_agents=(
            len(
                env.agent_ids
            )
        ),

        hidden_dim=(
            HIDDEN_DIM
        ),

        learning_rate=(
            LEARNING_RATE
        ),
    )

    # ------------------------------------------------------------
    # MOVE NETWORKS TO DEVICE
    # ------------------------------------------------------------

    for actor in mappo.actors:

        actor.to(
            DEVICE
        )

    mappo.critic.to(
        DEVICE
    )

    # ------------------------------------------------------------
    # TRAINING HISTORY
    # ------------------------------------------------------------

    history = []

    # ============================================================
    # EPISODES
    # ============================================================

    for episode in range(
        1,
        NUM_EPISODES + 1,
    ):

        observations = (
            env.reset()
        )

        # --------------------------------------------------------
        # EPISODE STORAGE
        # --------------------------------------------------------

        episode_observations = {
            agent_id: []
            for agent_id
            in env.agent_ids
        }

        episode_actions = {
            agent_id: []
            for agent_id
            in env.agent_ids
        }

        episode_log_probs = {
            agent_id: []
            for agent_id
            in env.agent_ids
        }

        episode_masks = {
            agent_id: []
            for agent_id
            in env.agent_ids
        }

        episode_rewards = {
            agent_id: []
            for agent_id
            in env.agent_ids
        }

        episode_values = []

        # --------------------------------------------------------
        # EPISODE REWARD
        # --------------------------------------------------------

        total_team_reward = 0.0

        # ========================================================
        # ROLLOUT
        # ========================================================

        for step in range(
            STEPS_PER_EPISODE
        ):

            # ----------------------------------------------------
            # CURRENT ACTION MASKS
            # ----------------------------------------------------

            action_masks = {
                agent_id:
                    env.get_action_mask(
                        agent_id
                    )
                for agent_id
                in env.agent_ids
            }

            # ----------------------------------------------------
            # STORE OBSERVATIONS
            # ----------------------------------------------------

            for agent_id in (
                env.agent_ids
            ):

                episode_observations[
                    agent_id
                ].append(
                    observations[
                        agent_id
                    ].copy()
                )

                episode_masks[
                    agent_id
                ].append(
                    action_masks[
                        agent_id
                    ].copy()
                )

            # ----------------------------------------------------
            # SELECT ACTIONS
            # ----------------------------------------------------

            actions, log_probs = (
                mappo.select_actions(
                    observations,
                    action_masks,
                )
            )

            # ----------------------------------------------------
            # STORE ACTIONS
            # ----------------------------------------------------

            for agent_id in (
                env.agent_ids
            ):

                episode_actions[
                    agent_id
                ].append(
                    actions[
                        agent_id
                    ]
                )

                episode_log_probs[
                    agent_id
                ].append(
                    log_probs[
                        agent_id
                    ]
                )

            # ----------------------------------------------------
            # CENTRALIZED CRITIC INPUT
            # ----------------------------------------------------

            joint_observation = np.concatenate(
                [
                    observations[
                        agent_id
                    ]
                    for agent_id
                    in env.agent_ids
                ]
            )

            joint_observation_tensor = (
                torch.tensor(
                    joint_observation,
                    dtype=torch.float32,
                    device=DEVICE,
                )
                .unsqueeze(0)
            )

            with torch.no_grad():

                value = (
                    mappo.critic(
                        joint_observation_tensor
                    )
                )

            episode_values.append(
                float(
                    value.item()
                )
            )

            # ----------------------------------------------------
            # ENVIRONMENT STEP
            # ----------------------------------------------------

            next_observations, rewards, done, infos = (
                env.step(
                    actions
                )
            )

            # ----------------------------------------------------
            # STORE REWARDS
            # ----------------------------------------------------

            for agent_id in (
                env.agent_ids
            ):

                reward = float(
                    rewards.get(
                        agent_id,
                        0.0,
                    )
                )

                episode_rewards[
                    agent_id
                ].append(
                    reward
                )

            # ----------------------------------------------------
            # TEAM REWARD
            # ----------------------------------------------------

            team_reward = np.mean(
                [
                    rewards.get(
                        agent_id,
                        0.0,
                    )

                    for agent_id
                    in env.agent_ids
                ]
            )

            total_team_reward += (
                team_reward
            )

            # ----------------------------------------------------
            # NEXT STATE
            # ----------------------------------------------------

            observations = (
                next_observations
            )

            if done:

                break

        # ========================================================
        # FINAL VALUE
        # ========================================================

        joint_observation = np.concatenate(
            [
                observations[
                    agent_id
                ]
                for agent_id
                in env.agent_ids
            ]
        )

        joint_observation_tensor = (
            torch.tensor(
                joint_observation,
                dtype=torch.float32,
                device=DEVICE,
            )
            .unsqueeze(0)
        )

        with torch.no_grad():

            next_value = (
                mappo.critic(
                    joint_observation_tensor
                )
            )

            next_value = float(
                next_value.item()
            )

        # ========================================================
        # PPO UPDATE
        # ========================================================

        actor_losses = []

        critic_losses = []

        entropy_values = []

        clip_fractions = []

        # --------------------------------------------------------
        # UPDATE EACH ACTOR
        # --------------------------------------------------------

        for agent_index, agent_id in enumerate(
            env.agent_ids
        ):

            observations_tensor = torch.tensor(
                np.asarray(
                    episode_observations[
                        agent_id
                    ],
                    dtype=np.float32,
                ),
                dtype=torch.float32,
                device=DEVICE,
            )

            actions_tensor = torch.tensor(
                episode_actions[
                    agent_id
                ],
                dtype=torch.long,
                device=DEVICE,
            )

            old_log_probs_tensor = torch.tensor(
                episode_log_probs[
                    agent_id
                ],
                dtype=torch.float32,
                device=DEVICE,
            )

            masks_tensor = torch.tensor(
                np.asarray(
                    episode_masks[
                        agent_id
                    ],
                    dtype=np.float32,
                ),
                dtype=torch.float32,
                device=DEVICE,
            )

            rewards_tensor = torch.tensor(
                episode_rewards[
                    agent_id
                ],
                dtype=torch.float32,
                device=DEVICE,
            )

            # ----------------------------------------------------
            # VALUE ESTIMATES
            # ----------------------------------------------------

            agent_values = []

            for timestep in range(
                len(
                    episode_rewards[
                        agent_id
                    ]
                )
            ):

                value = (
                    episode_values[
                        min(
                            timestep,
                            len(
                                episode_values
                            ) - 1,
                        )
                    ]
                )

                agent_values.append(
                    value
                )

            # ----------------------------------------------------
            # GAE
            # ----------------------------------------------------

            advantages = calculate_gae(
                rewards=[
                    float(
                        value
                    )

                    for value
                    in rewards_tensor.detach()
                    .cpu()
                    .numpy()
                ],

                values=[
                    float(
                        value
                    )

                    for value
                    in agent_values
                ],

                next_value=(
                    next_value
                ),
            )

            advantages_tensor = torch.tensor(
                advantages,
                dtype=torch.float32,
                device=DEVICE,
            )

            # ----------------------------------------------------
            # RETURNS
            # ----------------------------------------------------

            returns_tensor = (
                advantages_tensor
                + torch.tensor(
                    agent_values,
                    dtype=torch.float32,
                    device=DEVICE,
                )
            )

            # ----------------------------------------------------
            # NORMALIZE ADVANTAGES
            # ----------------------------------------------------

            if len(
                advantages_tensor
            ) > 1:

                advantages_tensor = (
                    (
                        advantages_tensor
                        - advantages_tensor.mean()
                    )
                    / (
                        advantages_tensor.std()
                        + 1e-8
                    )
                )

            actor = (
                mappo.actors[
                    agent_index
                ]
            )

            optimizer = (
                mappo.actor_optimizers[
                    agent_index
                ]
            )

            # ====================================================
            # PPO EPOCHS
            # ====================================================

            for _ in range(
                PPO_EPOCHS
            ):

                # ------------------------------------------------
                # NEW LOG PROBABILITIES
                # ------------------------------------------------

                new_log_probs = (
                    actor.get_log_probability(
                        observations_tensor,
                        actions_tensor,
                        masks_tensor,
                    )
                )

                # ------------------------------------------------
                # RATIO
                # ------------------------------------------------

                ratios = torch.exp(
                    new_log_probs
                    - old_log_probs_tensor
                )

                # ------------------------------------------------
                # SURROGATE 1
                # ------------------------------------------------

                surrogate_1 = (
                    ratios
                    * advantages_tensor
                )

                # ------------------------------------------------
                # SURROGATE 2
                # ------------------------------------------------

                clipped_ratios = torch.clamp(
                    ratios,
                    1.0 - CLIP_EPSILON,
                    1.0 + CLIP_EPSILON,
                )

                surrogate_2 = (
                    clipped_ratios
                    * advantages_tensor
                )

                # ------------------------------------------------
                # ACTOR LOSS
                # ------------------------------------------------

                actor_loss = -torch.min(
                    surrogate_1,
                    surrogate_2,
                ).mean()

                # ------------------------------------------------
                # ENTROPY
                # ------------------------------------------------

                entropy = (
                    actor.get_entropy(
                        observations_tensor,
                        masks_tensor,
                    ).mean()
                )

                # ------------------------------------------------
                # TOTAL ACTOR LOSS
                # ------------------------------------------------

                total_actor_loss = (
                    actor_loss
                    - ENTROPY_COEFFICIENT
                    * entropy
                )

                # ------------------------------------------------
                # OPTIMIZER
                # ------------------------------------------------

                optimizer.zero_grad()

                total_actor_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(),
                    max_norm=0.5,
                )

                optimizer.step()

                # ------------------------------------------------
                # METRICS
                # ------------------------------------------------

                actor_losses.append(
                    float(
                        actor_loss.item()
                    )
                )

                entropy_values.append(
                    float(
                        entropy.item()
                    )
                )

                clip_fraction = (
                    (
                        torch.abs(
                            ratios - 1.0
                        )
                        > CLIP_EPSILON
                    )
                    .float()
                    .mean()
                )

                clip_fractions.append(
                    float(
                        clip_fraction.item()
                    )
                )

            # ====================================================
            # CRITIC UPDATE
            # ====================================================

            joint_observations = []

            for timestep in range(
                len(
                    episode_rewards[
                        agent_id
                    ]
                )
            ):

                timestep_observation = (
                    np.concatenate(
                        [
                            episode_observations[
                                other_agent
                            ][timestep]
                            for other_agent
                            in env.agent_ids
                        ]
                    )
                )

                joint_observations.append(
                    timestep_observation
                )

            joint_observations_tensor = (
                torch.tensor(
                    np.asarray(
                        joint_observations,
                        dtype=np.float32,
                    ),
                    dtype=torch.float32,
                    device=DEVICE,
                )
            )

            predicted_values = (
                mappo.critic(
                    joint_observations_tensor
                )
            )

            critic_loss = F.mse_loss(
                predicted_values,
                returns_tensor,
            )

            mappo.critic_optimizer.zero_grad()

            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                mappo.critic.parameters(),
                max_norm=0.5,
            )

            mappo.critic_optimizer.step()

            critic_losses.append(
                float(
                    critic_loss.item()
                )
            )

        # ========================================================
        # EPISODE METRICS
        # ========================================================

        average_reward = (
            total_team_reward
            / max(
                len(
                    episode_rewards[
                        env.agent_ids[0]
                    ]
                ),
                1,
            )
        )

        average_actor_loss = (
            np.mean(
                actor_losses
            )
            if actor_losses
            else 0.0
        )

        average_critic_loss = (
            np.mean(
                critic_losses
            )
            if critic_losses
            else 0.0
        )

        average_entropy = (
            np.mean(
                entropy_values
            )
            if entropy_values
            else 0.0
        )

        average_clip_fraction = (
            np.mean(
                clip_fractions
            )
            if clip_fractions
            else 0.0
        )

        # --------------------------------------------------------
        # MOVING AVERAGE
        # --------------------------------------------------------

        history.append(
            {
                "episode": episode,
                "average_reward": (
                    average_reward
                ),
                "actor_loss": (
                    average_actor_loss
                ),
                "critic_loss": (
                    average_critic_loss
                ),
                "entropy": (
                    average_entropy
                ),
                "clip_fraction": (
                    average_clip_fraction
                ),
            }
        )

        recent_rewards = [
            item[
                "average_reward"
            ]

            for item
            in history[
                max(
                    0,
                    len(history) - 10,
                ):
            ]
        ]

        moving_average = (
            np.mean(
                recent_rewards
            )
        )

        # ========================================================
        # PRINT
        # ========================================================

        print(
            f"Episode "
            f"{episode:03d}/{NUM_EPISODES} | "
            f"Reward: "
            f"{average_reward:.4f} | "
            f"Moving Avg: "
            f"{moving_average:.4f} | "
            f"Actor Loss: "
            f"{average_actor_loss:.4f} | "
            f"Critic Loss: "
            f"{average_critic_loss:.4f} | "
            f"Entropy: "
            f"{average_entropy:.4f} | "
            f"Clip: "
            f"{average_clip_fraction:.4f}"
        )

    # ============================================================
    # SAVE MODEL
    # ============================================================

    torch.save(
        {
            "actors": [
                actor.state_dict()
                for actor
                in mappo.actors
            ],

            "critic": (
                mappo.critic.state_dict()
            ),

            "observation_dim": (
                env.observation_dim
            ),

            "action_dim": (
                env.action_dim
            ),

            "num_agents": (
                len(
                    env.agent_ids
                )
            ),
        },

        MODEL_PATH,
    )

    print()
    print(
        f"Model saved to:"
    )

    print(
        MODEL_PATH
    )

    # ============================================================
    # SAVE CSV
    # ============================================================

    os.makedirs(
        os.path.dirname(
            HISTORY_PATH
        ),
        exist_ok=True,
    )

    with open(
        HISTORY_PATH,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "average_reward",
                "actor_loss",
                "critic_loss",
                "entropy",
                "clip_fraction",
            ],
        )

        writer.writeheader()

        writer.writerows(
            history
        )

    print()
    print(
        f"Training history saved to:"
    )

    print(
        HISTORY_PATH
    )

    # ============================================================
    # CLOSE ENV
    # ============================================================

    env.close()

    print()
    print(
        "========================================"
    )

    print(
        "        MAPPO TRAINING FINISHED"
    )

    print(
        "========================================"
    )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    train()