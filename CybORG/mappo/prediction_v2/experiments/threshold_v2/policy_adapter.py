"""Load the pinned MAPPO policy and reproduce evaluator action selection."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from CybORG.mappo.mappo import MAPPO

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CHECKPOINT_PATH = PROJECT_ROOT / "CybORG" / "mappo" / "mappo_model.pt"
EXPECTED_CHECKPOINT_SHA256 = "6fab6f9d6cf79c20d2cce677077dd92eca352d39e7a013e34696656b6eb01170"
OBSERVATION_DIM = 256
ACTION_DIM = 8
NUM_AGENTS = 5
HIDDEN_DIM = 128
AGENT_IDS = tuple(f"blue_agent_{index}" for index in range(NUM_AGENTS))


def checkpoint_sha256(path: Path = CHECKPOINT_PATH) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_shape(state_dict, key: str, expected: tuple[int, ...], label: str) -> None:
    value = state_dict.get(key)
    actual = tuple(value.shape) if isinstance(value, torch.Tensor) else None
    if actual != expected:
        raise ValueError(
            f"Checkpoint {label} tensor {key!r} has shape {actual}; "
            f"expected {expected}."
        )


def _validate_actor_state(state_dict, actor_index: int) -> None:
    label = f"actor {actor_index}"
    _require_shape(state_dict, "network.0.weight", (HIDDEN_DIM, OBSERVATION_DIM), label)
    _require_shape(state_dict, "network.0.bias", (HIDDEN_DIM,), label)
    _require_shape(state_dict, "network.2.weight", (HIDDEN_DIM, HIDDEN_DIM), label)
    _require_shape(state_dict, "network.2.bias", (HIDDEN_DIM,), label)
    _require_shape(state_dict, "network.4.weight", (ACTION_DIM, HIDDEN_DIM), label)
    _require_shape(state_dict, "network.4.bias", (ACTION_DIM,), label)


def _validate_critic_state(state_dict) -> None:
    label = "critic"
    _require_shape(state_dict, "network.0.weight", (HIDDEN_DIM, OBSERVATION_DIM * NUM_AGENTS), label)
    _require_shape(state_dict, "network.0.bias", (HIDDEN_DIM,), label)
    _require_shape(state_dict, "network.2.weight", (HIDDEN_DIM, HIDDEN_DIM), label)
    _require_shape(state_dict, "network.2.bias", (HIDDEN_DIM,), label)
    _require_shape(state_dict, "network.4.weight", (1, HIDDEN_DIM), label)
    _require_shape(state_dict, "network.4.bias", (1,), label)


class PinnedMAPPOPolicy:
    """Strict adapter for mappo_model.pt; no alternate checkpoint is accepted."""

    def __init__(self, model: MAPPO, checkpoint_hash: str, device: torch.device):
        self.model = model
        self.checkpoint_hash = checkpoint_hash
        self.device = device
        self.agent_ids = AGENT_IDS

    @classmethod
    def load(cls, device: str | torch.device = "cpu") -> "PinnedMAPPOPolicy":
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(
                f"Pinned MAPPO checkpoint is missing: {CHECKPOINT_PATH}"
            )

        actual_hash = checkpoint_sha256()
        if actual_hash.lower() != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError(
                "Pinned MAPPO checkpoint SHA-256 mismatch: "
                f"expected {EXPECTED_CHECKPOINT_SHA256}, got {actual_hash}. "
                "Refusing to load another or changed checkpoint."
            )

        resolved_device = torch.device(device)
        try:
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location=resolved_device,
                weights_only=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load the verified pinned checkpoint {CHECKPOINT_PATH}: {error}"
            ) from error

        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"Pinned checkpoint must be a dictionary; got {type(checkpoint).__name__}."
            )

        expected_dimensions = {
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "num_agents": NUM_AGENTS,
        }
        for key, expected in expected_dimensions.items():
            saved = checkpoint.get(key)
            if saved is None or int(saved) != expected:
                raise ValueError(
                    f"Pinned checkpoint {key} mismatch: expected {expected}, got {saved!r}."
                )

        actor_states = checkpoint.get("actors")
        critic_state = checkpoint.get("critic")
        if not isinstance(actor_states, (list, tuple)) or len(actor_states) != NUM_AGENTS:
            count = len(actor_states) if isinstance(actor_states, (list, tuple)) else "missing"
            raise ValueError(
                f"Pinned checkpoint actor count mismatch: expected {NUM_AGENTS}, got {count}."
            )
        if not isinstance(critic_state, dict):
            raise ValueError("Pinned checkpoint is missing its centralized critic state.")

        for index, actor_state in enumerate(actor_states):
            if not isinstance(actor_state, dict):
                raise TypeError(f"Pinned checkpoint actor {index} state is not a state dictionary.")
            _validate_actor_state(actor_state, index)
        _validate_critic_state(critic_state)

        model = MAPPO(
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            num_agents=NUM_AGENTS,
            hidden_dim=HIDDEN_DIM,
            device=resolved_device,
        )
        for actor, actor_state in zip(model.actors, actor_states):
            actor.load_state_dict(actor_state, strict=True)
            actor.eval()
        model.critic.load_state_dict(critic_state, strict=True)
        model.critic.eval()
        return cls(model, actual_hash, resolved_device)

    @torch.no_grad()
    def select_actions(self, observations: dict, action_masks: dict) -> dict[str, int]:
        missing_observations = set(self.agent_ids) - set(observations)
        missing_masks = set(self.agent_ids) - set(action_masks)
        if missing_observations:
            raise KeyError(f"Observations missing agents: {sorted(missing_observations)}")
        if missing_masks:
            raise KeyError(f"Action masks missing agents: {sorted(missing_masks)}")

        ordered_observations = {agent_id: observations[agent_id] for agent_id in self.agent_ids}
        ordered_masks = {agent_id: action_masks[agent_id] for agent_id in self.agent_ids}
        actions = self.model.select_actions(
            ordered_observations,
            ordered_masks,
            deterministic=True,
        )
        normalized = {agent_id: int(actions[agent_id]) for agent_id in self.agent_ids}
        for agent_id, action in normalized.items():
            if not 0 <= action < ACTION_DIM:
                raise ValueError(f"MAPPO selected invalid action {action} for {agent_id}.")
            if float(ordered_masks[agent_id][action]) <= 0:
                raise ValueError(
                    f"MAPPO selected masked action {action} for {agent_id}; refusing to step."
                )
        return normalized