import os
import torch
import numpy as np

from CybORG.mappo.cyborg_env import CybORGMAPPOEnv
from CybORG.mappo.mappo import MAPPO


# ================================================================
# CONFIGURATION
# ================================================================

MODEL_PATH = "CybORG/mappo/mappo_model.pt"

STEPS = 1

TOP_K = 5


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# LOAD MODEL
# ================================================================

def load_model(env):

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    mappo = MAPPO(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        num_agents=len(env.agent_ids),
        hidden_dim=128,
        learning_rate=3e-4,
    )

    # ------------------------------------------------------------
    # LOAD ACTORS
    # ------------------------------------------------------------

    for index, actor in enumerate(
        mappo.actors
    ):

        actor.load_state_dict(
            checkpoint["actors"][index]
        )

        actor.to(DEVICE)

        actor.eval()

    # ------------------------------------------------------------
    # LOAD CRITIC
    # ------------------------------------------------------------

    mappo.critic.load_state_dict(
        checkpoint["critic"]
    )

    mappo.critic.to(DEVICE)

    mappo.critic.eval()

    return mappo


# ================================================================
# INSPECT POLICY
# ================================================================

def inspect_policy():

    print()
    print(
        "============================================================"
    )

    print(
        "              MAPPO POLICY INSPECTION"
    )

    print(
        "============================================================"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    # ------------------------------------------------------------
    # CHECK MODEL
    # ------------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        print()
        print(
            "ERROR: Trained model not found."
        )

        print(
            f"Expected: {MODEL_PATH}"
        )

        return

    # ------------------------------------------------------------
    # ENVIRONMENT
    # ------------------------------------------------------------

    env = CybORGMAPPOEnv(
        steps=STEPS
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

    # ------------------------------------------------------------
    # LOAD TRAINED MODEL
    # ------------------------------------------------------------

    mappo = load_model(env)

    print()
    print(
        "Trained MAPPO model loaded successfully."
    )

    # ------------------------------------------------------------
    # RESET ENVIRONMENT
    # ------------------------------------------------------------

    observations = env.reset()

    # ============================================================
    # INSPECT EACH AGENT
    # ============================================================

    for agent_index, agent_id in enumerate(
        env.agent_ids
    ):

        print()
        print(
            "------------------------------------------------------------"
        )

        print(
            f"Agent: {agent_id}"
        )

        print(
            "------------------------------------------------------------"
        )

        # --------------------------------------------------------
        # OBSERVATION
        # --------------------------------------------------------

        observation = observations[
            agent_id
        ]

        observation_tensor = torch.tensor(
            observation,
            dtype=torch.float32,
            device=DEVICE,
        )

        # --------------------------------------------------------
        # ACTION MASK
        # --------------------------------------------------------

        action_mask = env.get_action_mask(
            agent_id
        )

        action_mask_tensor = torch.tensor(
            action_mask,
            dtype=torch.float32,
            device=DEVICE,
        )

        # --------------------------------------------------------
        # VALID ACTIONS
        # --------------------------------------------------------

        valid_action_indices = (
            np.where(
                np.asarray(
                    action_mask
                ) > 0
            )[0]
        )

        number_valid_actions = len(
            valid_action_indices
        )

        print(
            f"Valid actions: "
            f"{number_valid_actions} / "
            f"{env.action_dim}"
        )

        # --------------------------------------------------------
        # ACTOR
        # --------------------------------------------------------

        actor = mappo.actors[
            agent_index
        ]

        # --------------------------------------------------------
        # GET DISTRIBUTION
        # --------------------------------------------------------

        with torch.no_grad():

            distribution = (
                actor.get_distribution(
                    observation_tensor,
                    action_mask_tensor,
                )
            )

            probabilities = (
                distribution.probs
                .detach()
                .cpu()
                .numpy()
            )

            entropy = (
                distribution.entropy()
                .item()
            )

        # --------------------------------------------------------
        # TOP ACTIONS
        # --------------------------------------------------------

        sorted_indices = np.argsort(
            probabilities
        )[::-1]

        print(
            f"Policy entropy: "
            f"{entropy:.4f}"
        )

        print()

        print(
            f"Top {TOP_K} actions:"
        )

        print()

        for rank, action_index in enumerate(
            sorted_indices[
                :TOP_K
            ],
            start=1,
        ):

            probability = probabilities[
                action_index
            ]

            print(
                f"{rank}. "
                f"Global Action {action_index:03d} "
                f"-> "
                f"Probability: "
                f"{probability:.6f}"
            )

        # --------------------------------------------------------
        # PROBABILITY STATISTICS
        # --------------------------------------------------------

        valid_probabilities = probabilities[
            valid_action_indices
        ]

        if len(
            valid_probabilities
        ) > 0:

            max_probability = float(
                np.max(
                    valid_probabilities
                )
            )

            min_probability = float(
                np.min(
                    valid_probabilities
                )
            )

            mean_probability = float(
                np.mean(
                    valid_probabilities
                )
            )

            print()

            print(
                "Valid-action probability statistics:"
            )

            print(
                f"Maximum probability: "
                f"{max_probability:.6f}"
            )

            print(
                f"Minimum probability: "
                f"{min_probability:.6f}"
            )

            print(
                f"Mean probability: "
                f"{mean_probability:.6f}"
            )

            # ----------------------------------------------------
            # APPROXIMATE UNIFORM PROBABILITY
            # ----------------------------------------------------

            uniform_probability = (
                1.0
                / number_valid_actions
            )

            print(
                f"Uniform probability would be: "
                f"{uniform_probability:.6f}"
            )

        # --------------------------------------------------------
        # MOST LIKELY ACTION
        # --------------------------------------------------------

        best_action = int(
            sorted_indices[0]
        )

        best_probability = float(
            probabilities[
                best_action
            ]
        )

        print()

        print(
            f"Most likely action: "
            f"{best_action}"
        )

        print(
            f"Probability: "
            f"{best_probability:.6f}"
        )

    # ============================================================
    # FINISH
    # ============================================================

    env.close()

    print()
    print(
        "============================================================"
    )

    print(
        "             POLICY INSPECTION COMPLETE"
    )

    print(
        "============================================================"
    )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    inspect_policy()