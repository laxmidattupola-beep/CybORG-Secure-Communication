"""Check pinned-checkpoint compatibility and one masked, deterministic action pass."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from CybORG.mappo.cyborg_env import CybORGMAPPOEnv

from .policy_adapter import (
    ACTION_DIM,
    AGENT_IDS,
    EXPECTED_CHECKPOINT_SHA256,
    OBSERVATION_DIM,
    PinnedMAPPOPolicy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    policy = PinnedMAPPOPolicy.load(args.device)
    environment = CybORGMAPPOEnv(steps=100)
    try:
        if environment.observation_dim != OBSERVATION_DIM:
            raise ValueError(
                f"Environment observation dimension mismatch: expected {OBSERVATION_DIM}, "
                f"got {environment.observation_dim}."
            )
        if environment.action_dim != ACTION_DIM:
            raise ValueError(
                f"Environment action dimension mismatch: expected {ACTION_DIM}, "
                f"got {environment.action_dim}."
            )
        if tuple(environment.agent_ids) != AGENT_IDS:
            raise ValueError(
                f"Environment agent IDs mismatch: expected {AGENT_IDS}, "
                f"got {tuple(environment.agent_ids)}."
            )

        environment.env.set_seed(20261031)
        observations = environment.reset()
        masks = {
            agent_id: environment.get_action_mask(agent_id)
            for agent_id in environment.agent_ids
        }
        for agent_id, observation in observations.items():
            if np.asarray(observation).reshape(-1).size != OBSERVATION_DIM:
                raise ValueError(f"Observation for {agent_id} is not {OBSERVATION_DIM}-dimensional.")

        with torch.no_grad():
            actions = policy.select_actions(observations, masks)
        print(
            "Policy validation passed: "
            f"checkpoint_sha256={EXPECTED_CHECKPOINT_SHA256}, "
            f"agents={len(actions)}, observations={OBSERVATION_DIM}, actions={ACTION_DIM}, "
            "deterministic masked action selection succeeded."
        )
    finally:
        environment.close()


if __name__ == "__main__":
    main()