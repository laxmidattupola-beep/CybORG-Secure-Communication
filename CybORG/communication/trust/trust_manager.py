"""
Trust management module for CybORG.

This module maintains trust values between distributed blue agents.

Trust is based on three factors:

1. Communication reliability
2. Authentication validity
3. Observed behaviour

The factors are combined using:

    trust =
        0.3 * communication +
        0.3 * authentication +
        0.4 * behaviour

A history of trust observations is maintained for each agent.
The final trust value is the average of the historical weighted scores.

Trust values are always in [0.0, 1.0].

0.0 -> completely untrusted
0.5 -> neutral
1.0 -> completely trusted
"""

from __future__ import annotations

from typing import Dict, List


class TrustManager:
    """
    Maintains and updates trust values for communicating agents.
    """

    def __init__(
        self,
        decay_rate: float = 0.01,
        communication_weight: float = 0.3,
        authentication_weight: float = 0.3,
        behavior_weight: float = 0.4,
    ):
        """
        Initialise the trust manager.

        Parameters
        ----------
        decay_rate:
            Amount by which trust moves towards neutral trust during
            decay.

        communication_weight:
            Weight assigned to communication reliability.

        authentication_weight:
            Weight assigned to authentication validity.

        behavior_weight:
            Weight assigned to observed behaviour.
        """

        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError("decay_rate must be between 0 and 1.")

        total_weight = (
            communication_weight
            + authentication_weight
            + behavior_weight
        )

        if total_weight <= 0:
            raise ValueError("Trust weights must sum to a positive value.")

        self.decay_rate = float(decay_rate)

        # Normalise weights so that they always sum to 1.
        self.trust_factors = {
            "communication": communication_weight / total_weight,
            "authentication": authentication_weight / total_weight,
            "behavior": behavior_weight / total_weight,
        }

        # Current trust value for every known agent.
        self.trust_scores: Dict[str, float] = {}

        # Historical trust evidence for every agent.
        self.trust_history: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(
        self,
        agent_name: str,
        initial_trust: float = 0.5,
    ) -> float:
        """
        Register an agent.

        Existing agents are not reset.
        """

        if not 0.0 <= initial_trust <= 1.0:
            raise ValueError(
                "initial_trust must be between 0 and 1."
            )

        if agent_name not in self.trust_scores:
            self.trust_scores[agent_name] = float(initial_trust)
            self.trust_history[agent_name] = [float(initial_trust)]

        return self.trust_scores[agent_name]

    def register_agents(
        self,
        agent_names,
        initial_trust: float = 0.5,
    ) -> None:
        """
        Register multiple agents.
        """

        for agent_name in agent_names:
            self.register_agent(
                agent_name,
                initial_trust=initial_trust,
            )

    # ------------------------------------------------------------------
    # Trust retrieval
    # ------------------------------------------------------------------

    def get_trust(self, agent_name: str) -> float:
        """
        Return the current trust value of an agent.

        Unknown agents are automatically registered with neutral trust.
        """

        if agent_name not in self.trust_scores:
            self.register_agent(agent_name)

        return self.trust_scores[agent_name]

    # ------------------------------------------------------------------
    # Trust update
    # ------------------------------------------------------------------

    def update_trust(
        self,
        agent_name: str,
        successful_communication,
        authentication,
        behavior,
    ) -> float:
        """
        Update trust using communication, authentication and behaviour.

        Each factor can be supplied as:

            True / False
            1 / 0
            any floating point value between 0 and 1

        The weighted score is:

            communication * 0.3
            + authentication * 0.3
            + behavior * 0.4

        The weighted score is stored in the agent's history.

        The current trust is the average of the historical scores.

        Returns
        -------
        float
            Updated trust value.
        """

        self.register_agent(agent_name)

        communication_score = self._normalise_factor(
            successful_communication
        )

        authentication_score = self._normalise_factor(
            authentication
        )

        behavior_score = self._normalise_factor(
            behavior
        )

        weighted_score = (
            self.trust_factors["communication"]
            * communication_score
            + self.trust_factors["authentication"]
            * authentication_score
            + self.trust_factors["behavior"]
            * behavior_score
        )

        self.trust_history[agent_name].append(
            weighted_score
        )

        self.trust_scores[agent_name] = self._calculate_average(
            self.trust_history[agent_name]
        )

        return self.trust_scores[agent_name]

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(self, agent_name: str) -> List[float]:
        """
        Return the trust history of an agent.

        A copy is returned so that callers cannot accidentally modify
        internal trust state.
        """

        if agent_name not in self.trust_scores:
            self.register_agent(agent_name)

        return list(self.trust_history[agent_name])

    # ------------------------------------------------------------------
    # Trust decay
    # ------------------------------------------------------------------

    def decay_trust(self, agent_name: str) -> float:
        """
        Apply trust decay towards neutral trust (0.5).

        This prevents an old trust decision from permanently dominating
        the current trust value.

        Formula:

            new_trust =
                current_trust
                + decay_rate * (0.5 - current_trust)
        """

        self.register_agent(agent_name)

        current = self.trust_scores[agent_name]

        new_trust = current + (
            self.decay_rate * (0.5 - current)
        )

        self.trust_scores[agent_name] = self._clamp(
            new_trust
        )

        return self.trust_scores[agent_name]

    def decay_all(self) -> None:
        """
        Apply trust decay to all registered agents.
        """

        for agent_name in list(self.trust_scores.keys()):
            self.decay_trust(agent_name)

    # ------------------------------------------------------------------
    # Trust decision
    # ------------------------------------------------------------------

    def is_trusted(
        self,
        agent_name: str,
        threshold: float = 0.5,
    ) -> bool:
        """
        Determine whether an agent is trusted.

        An agent is trusted when:

            trust >= threshold
        """

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        return self.get_trust(agent_name) >= threshold

    # ------------------------------------------------------------------
    # Trust classification
    # ------------------------------------------------------------------

    def get_trust_level(
        self,
        agent_name: str,
    ) -> str:
        """
        Return a human-readable trust classification.
        """

        trust = self.get_trust(agent_name)

        if trust < 0.25:
            return "untrusted"

        if trust < 0.5:
            return "low"

        if trust < 0.75:
            return "moderate"

        return "high"

    # ------------------------------------------------------------------
    # Message weighting
    # ------------------------------------------------------------------

    def get_message_weight(
        self,
        agent_name: str,
    ) -> float:
        """
        Return the trust value to use when weighting information
        received from another agent.

        This allows an agent to reduce the influence of information
        from an agent with low historical reliability rather than
        immediately discarding it.
        """

        return self.get_trust(agent_name)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(
        self,
        agent_name: str,
    ) -> dict:
        """
        Return trust statistics for an agent.
        """

        trust = self.get_trust(agent_name)
        history = self.get_history(agent_name)

        return {
            "agent": agent_name,
            "trust": trust,
            "trust_level": self.get_trust_level(agent_name),
            "history_length": len(history),
            "communication_weight": self.trust_factors[
                "communication"
            ],
            "authentication_weight": self.trust_factors[
                "authentication"
            ],
            "behavior_weight": self.trust_factors[
                "behavior"
            ],
        }

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """
        Return all trust state.

        Useful for saving trust information between episodes.
        """

        return {
            "decay_rate": self.decay_rate,
            "trust_factors": dict(self.trust_factors),
            "trust_scores": dict(self.trust_scores),
            "trust_history": {
                agent: list(history)
                for agent, history in self.trust_history.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restore trust state.
        """

        self.decay_rate = float(
            state.get("decay_rate", 0.01)
        )

        if "trust_factors" in state:
            self.trust_factors = dict(
                state["trust_factors"]
            )

        self.trust_scores = {
            agent: float(score)
            for agent, score
            in state.get("trust_scores", {}).items()
        }

        self.trust_history = {
            agent: list(history)
            for agent, history
            in state.get("trust_history", {}).items()
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Remove all trust information.
        """

        self.trust_scores.clear()
        self.trust_history.clear()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_factor(value) -> float:
        """
        Convert a trust factor to a value in [0, 1].
        """

        if isinstance(value, bool):
            return 1.0 if value else 0.0

        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                "Trust factors must be boolean or numeric."
            )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "Trust factors must be between 0 and 1."
            )

        return value

    @staticmethod
    def _calculate_average(
        values: List[float],
    ) -> float:
        """
        Calculate the historical average trust score.
        """

        if not values:
            return 0.5

        return sum(values) / len(values)

    @staticmethod
    def _clamp(value: float) -> float:
        """
        Clamp trust to [0, 1].
        """

        return max(
            0.0,
            min(1.0, float(value)),
        )

    def __contains__(self, agent_name: str) -> bool:
        return agent_name in self.trust_scores

    def __len__(self) -> int:
        return len(self.trust_scores)

    def __repr__(self) -> str:
        return (
            f"TrustManager("
            f"agents={len(self)}, "
            f"decay_rate={self.decay_rate}, "
            f"trust_factors={self.trust_factors}"
            f")"
        )