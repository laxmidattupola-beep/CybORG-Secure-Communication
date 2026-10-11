import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from CybORG.mappo.cyborg_env import CybORGMAPPOEnv
from CybORG.mappo.mappo import MAPPO


# ============================================================
# Configuration
# ============================================================

NUM_EPISODES = 10

STEPS_PER_EPISODE = 10

LEARNING_RATE = 3e-4

GAMMA = 0.99

GAE_LAMBDA = 0.95

CLIP_EPSILON = 0.2

PPO_EPOCHS = 4

ENTROPY_COEFFICIENT = 0.01

VALUE_COEFFICIENT = 0.5

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# GAE
# ============================================================

def calculate_gae(
    rewards,
    values,
    next_value,
    dones
):

    rewards = np.asarray(
        rewards,
        dtype=np.float32
    )

    values = np.asarray(
        values,
        dtype=np.float32
    )

    dones = np.asarray(
        dones,
        dtype=np.float32
    )

    advantages = np.zeros_like(
        rewards
    )

    gae = 0.0

    for t in reversed(
        range(len(rewards))
    ):

        if t == len(rewards) - 1:

            next_val = next_value

        else:

            next_val = values[t + 1]

        if dones[t] > 0:

            next_val = 0.0

        delta = (
            rewards[t]
            + GAMMA * next_val
            - values[t]
        )

        gae = (
            delta
            + GAMMA
            * GAE_LAMBDA
            * (1.0 - dones[t])
            * gae
        )

        advantages[t] = gae

    returns = (
        advantages
        + values
    )

    return (
        advantages,
        returns
    )


# ============================================================
# PPO Actor Update
# ============================================================

def update_actor(
    actor,
    optimizer,
    observations,
    actions,
    old_log_probs,
    advantages
):

    observations = torch.from_numpy(
        np.asarray(
            observations,
            dtype=np.float32
        )
    ).to(DEVICE)

    actions = torch.tensor(
        actions,
        dtype=torch.long
    ).to(DEVICE)

    old_log_probs = torch.tensor(
        old_log_probs,
        dtype=torch.float32
    ).to(DEVICE)

    advantages = torch.tensor(
        advantages,
        dtype=torch.float32
    ).to(DEVICE)

    # Normalize advantages
    advantages = (
        advantages
        - advantages.mean()
    ) / (
        advantages.std()
        + 1e-8
    )

    total_loss = 0.0

    total_entropy = 0.0

    total_clip_fraction = 0.0

    for _ in range(
        PPO_EPOCHS
    ):

        logits = actor(
            observations
        )

        distribution = Categorical(
            logits=logits
        )

        new_log_probs = (
            distribution.log_prob(
                actions
            )
        )

        entropy = (
            distribution.entropy().mean()
        )

        ratio = torch.exp(
            new_log_probs
            - old_log_probs
        )

        clipped_ratio = torch.clamp(
            ratio,
            1.0 - CLIP_EPSILON,
            1.0 + CLIP_EPSILON
        )

        objective_1 = (
            ratio
            * advantages
        )

        objective_2 = (
            clipped_ratio
            * advantages
        )

        actor_loss = -torch.min(
            objective_1,
            objective_2
        ).mean()

        loss = (
            actor_loss
            - ENTROPY_COEFFICIENT
            * entropy
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            actor.parameters(),
            max_norm=0.5
        )

        optimizer.step()

        clip_fraction = (
            (
                torch.abs(
                    ratio - 1.0
                )
                > CLIP_EPSILON
            )
            .float()
            .mean()
            .item()
        )

        total_loss += loss.item()

        total_entropy += (
            entropy.item()
        )

        total_clip_fraction += (
            clip_fraction
        )

    return (

        total_loss
        / PPO_EPOCHS,

        total_entropy
        / PPO_EPOCHS,

        total_clip_fraction
        / PPO_EPOCHS
    )


# ============================================================
# Critic Update
# ============================================================

def update_critic(
    critic,
    optimizer,
    states,
    returns
):

    states = torch.from_numpy(
        np.asarray(
            states,
            dtype=np.float32
        )
    ).to(DEVICE)

    returns = torch.tensor(
        returns,
        dtype=torch.float32
    ).to(DEVICE)

    total_loss = 0.0

    for _ in range(
        PPO_EPOCHS
    ):

        predicted_values = critic(
            states
        )

        value_loss = nn.functional.mse_loss(
            predicted_values,
            returns
        )

        loss = (
            VALUE_COEFFICIENT
            * value_loss
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            critic.parameters(),
            max_norm=0.5
        )

        optimizer.step()

        total_loss += (
            loss.item()
        )

    return (
        total_loss
        / PPO_EPOCHS
    )


# ============================================================
# Moving Average
# ============================================================

def moving_average(
    rewards,
    window=10
):

    if len(rewards) == 0:

        return 0.0

    return float(
        np.mean(
            rewards[
                -window:
            ]
        )
    )


# ============================================================
# Save history
# ============================================================

def save_history(
    history
):

    os.makedirs(
        "CybORG/mappo",
        exist_ok=True
    )

    path = (
        "CybORG/mappo/"
        "training_history.csv"
    )

    with open(
        path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Episode",
            "Steps",
            "AverageReward",
            "MovingAverage",
            "ActorLoss",
            "CriticLoss",
            "Entropy",
            "ClipFraction"
        ])

        for i in range(
            len(
                history["episode"]
            )
        ):

            writer.writerow([

                history["episode"][i],

                history["steps"][i],

                history["reward"][i],

                history["moving_average"][i],

                history["actor_loss"][i],

                history["critic_loss"][i],

                history["entropy"][i],

                history["clip_fraction"][i]
            ])

    print(
        "\nTraining history saved to:"
    )

    print(
        path
    )


# ============================================================
# Plot
# ============================================================

def plot_results(
    history
):

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "Matplotlib not installed."
        )

        return

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        history["episode"],
        history["reward"],
        label="Episode Reward"
    )

    plt.plot(
        history["episode"],
        history["moving_average"],
        label="Moving Average"
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Average Reward"
    )

    plt.title(
        "MAPPO Training Reward"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    path = (
        "CybORG/mappo/"
        "training_reward.png"
    )

    plt.savefig(
        path
    )

    plt.close()

    print(
        "Reward graph saved to:"
    )

    print(
        path
    )


# ============================================================
# Training
# ============================================================

def train():

    print("=" * 60)

    print(
        "              MAPPO TRAINING"
    )

    print("=" * 60)

    print(
        "Device:",
        DEVICE
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = CybORGMAPPOEnv(
        steps=STEPS_PER_EPISODE
    )

    agent_ids = (
        env.get_agent_ids()
    )

    num_agents = (
        env.get_num_agents()
    )

    observation_dim = (
        env.get_observation_dim()
    )

    action_dim = (
        env.get_action_dim()
    )

    print(
        "Agents:",
        agent_ids
    )

    print(
        "Number of agents:",
        num_agents
    )

    print(
        "Observation dimension:",
        observation_dim
    )

    print(
        "Action dimension:",
        action_dim
    )

    # --------------------------------------------------------
    # MAPPO
    # --------------------------------------------------------

    model = MAPPO(

        observation_dim=observation_dim,

        action_dim=action_dim,

        num_agents=num_agents
    )

    for actor in model.actors:

        actor.to(
            DEVICE
        )

    model.critic.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------

    actor_optimizers = [

        torch.optim.Adam(

            actor.parameters(),

            lr=LEARNING_RATE

        )

        for actor in model.actors
    ]

    critic_optimizer = (
        torch.optim.Adam(

            model.critic.parameters(),

            lr=LEARNING_RATE

        )
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = {

        "episode": [],

        "steps": [],

        "reward": [],

        "moving_average": [],

        "actor_loss": [],

        "critic_loss": [],

        "entropy": [],

        "clip_fraction": []
    }

    # ========================================================
    # Episodes
    # ========================================================

    for episode in range(
        1,
        NUM_EPISODES + 1
    ):

        observations = (
            env.reset()
        )

        # ----------------------------------------------------
        # Rollout buffers
        # ----------------------------------------------------

        global_states = []

        team_rewards = []

        global_values = []

        dones = []

        agent_buffers = {

            agent_id: {

                "observations": [],

                "actions": [],

                "log_probs": []

            }

            for agent_id in agent_ids
        }

        done = False

        # ----------------------------------------------------
        # Collect rollout
        # ----------------------------------------------------

        for step in range(
            STEPS_PER_EPISODE
        ):

            observation_matrix = np.asarray(

                [

                    observations[
                        agent_id
                    ]

                    for agent_id in agent_ids

                ],

                dtype=np.float32
            )

            global_states.append(
                observation_matrix
            )

            # ------------------------------------------------
            # Centralized critic
            # ------------------------------------------------

            state_tensor = (
                torch.from_numpy(
                    observation_matrix
                )
                .float()
                .unsqueeze(0)
                .to(DEVICE)
            )

            with torch.no_grad():

                value = model.critic(
                    state_tensor
                ).item()

            global_values.append(
                value
            )

            # ------------------------------------------------
            # Actions
            # ------------------------------------------------

            actions = {}

            for agent_index, agent_id in enumerate(
                agent_ids
            ):

                observation_tensor = (
                    torch.from_numpy(
                        observations[
                            agent_id
                        ]
                    )
                    .float()
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                with torch.no_grad():

                    action, log_prob = (
                        model.actors[
                            agent_index
                        ].get_action(
                            observation_tensor
                        )
                    )

                actions[
                    agent_id
                ] = action.item()

                agent_buffers[
                    agent_id
                ]["observations"].append(
                    observations[
                        agent_id
                    ]
                )

                agent_buffers[
                    agent_id
                ]["actions"].append(
                    action.item()
                )

                agent_buffers[
                    agent_id
                ]["log_probs"].append(
                    log_prob.item()
                )

            # ------------------------------------------------
            # Environment step
            # ------------------------------------------------

            (
                next_observations,
                rewards,
                done,
                info
            ) = env.step(
                actions
            )

            # ------------------------------------------------
            # Team reward
            # ------------------------------------------------

            team_reward = np.mean(

                [

                    rewards[
                        agent_id
                    ]

                    for agent_id in agent_ids

                ]
            )

            team_rewards.append(
                float(team_reward)
            )

            dones.append(
                bool(done)
            )

            observations = (
                next_observations
            )

            if done:

                break

        # ----------------------------------------------------
        # Final state
        # ----------------------------------------------------

        final_matrix = np.asarray(

            [

                observations[
                    agent_id
                ]

                for agent_id in agent_ids

            ],

            dtype=np.float32
        )

        final_state = (
            torch.from_numpy(
                final_matrix
            )
            .float()
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            next_value = (
                model.critic(
                    final_state
                ).item()
            )

        # ----------------------------------------------------
        # GAE
        # ----------------------------------------------------

        advantages, returns = calculate_gae(

            rewards=team_rewards,

            values=global_values,

            next_value=next_value,

            dones=dones
        )

        # ----------------------------------------------------
        # Actor updates
        # ----------------------------------------------------

        actor_losses = []

        entropies = []

        clip_fractions = []

        for agent_index, agent_id in enumerate(
            agent_ids
        ):

            buffer = (
                agent_buffers[
                    agent_id
                ]
            )

            (
                actor_loss,
                entropy,
                clip_fraction

            ) = update_actor(

                actor=model.actors[
                    agent_index
                ],

                optimizer=actor_optimizers[
                    agent_index
                ],

                observations=buffer[
                    "observations"
                ],

                actions=buffer[
                    "actions"
                ],

                old_log_probs=buffer[
                    "log_probs"
                ],

                advantages=advantages
            )

            actor_losses.append(
                actor_loss
            )

            entropies.append(
                entropy
            )

            clip_fractions.append(
                clip_fraction
            )

        # ----------------------------------------------------
        # Critic update
        # ----------------------------------------------------

        critic_loss = update_critic(

            critic=model.critic,

            optimizer=critic_optimizer,

            states=np.asarray(
                global_states,
                dtype=np.float32
            ),

            returns=returns
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        average_reward = float(
            np.mean(
                team_rewards
            )
        )

        average_actor_loss = float(
            np.mean(
                actor_losses
            )
        )

        average_entropy = float(
            np.mean(
                entropies
            )
        )

        average_clip_fraction = float(
            np.mean(
                clip_fractions
            )
        )

        history[
            "episode"
        ].append(
            episode
        )

        history[
            "steps"
        ].append(
            len(team_rewards)
        )

        history[
            "reward"
        ].append(
            average_reward
        )

        history[
            "moving_average"
        ].append(
            moving_average(
                history["reward"]
            )
        )

        history[
            "actor_loss"
        ].append(
            average_actor_loss
        )

        history[
            "critic_loss"
        ].append(
            critic_loss
        )

        history[
            "entropy"
        ].append(
            average_entropy
        )

        history[
            "clip_fraction"
        ].append(
            average_clip_fraction
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"\nEpisode "
            f"{episode:03d}/"
            f"{NUM_EPISODES}"
        )

        print(
            f"Steps: "
            f"{len(team_rewards)}"
        )

        print(
            f"Average reward: "
            f"{average_reward:.3f}"
        )

        print(
            f"Moving average: "
            f"{history['moving_average'][-1]:.3f}"
        )

        print(
            f"Actor loss: "
            f"{average_actor_loss:.5f}"
        )

        print(
            f"Critic loss: "
            f"{critic_loss:.5f}"
        )

        print(
            f"Entropy: "
            f"{average_entropy:.5f}"
        )

        print(
            f"Clip fraction: "
            f"{average_clip_fraction:.5f}"
        )

    # ========================================================
    # Save model
    # ========================================================

    checkpoint = {

        "actors": [

            actor.state_dict()

            for actor in model.actors

        ],

        "critic":
            model.critic.state_dict(),

        "observation_dim":
            observation_dim,

        "action_dim":
            action_dim,

        "num_agents":
            num_agents
    }

    model_path = (
        "CybORG/mappo/"
        "mappo_model.pt"
    )

    torch.save(
        checkpoint,
        model_path
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    save_history(
        history
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_results(
        history
    )

    # ========================================================
    # Complete
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "             TRAINING COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Total episodes: "
        f"{NUM_EPISODES}"
    )

    print(
        f"Final average reward: "
        f"{history['reward'][-1]:.3f}"
    )

    print(
        f"Final moving average: "
        f"{history['moving_average'][-1]:.3f}"
    )

    print(
        f"Model saved to:"
    )

    print(
        model_path
    )

    env.close()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    train()