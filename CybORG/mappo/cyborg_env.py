import inspect
import numpy as np

from CybORG import CybORG
from CybORG.Agents import (
    SleepAgent,
    EnterpriseGreenAgent,
    FiniteStateRedAgent
)
from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import (
    EnterpriseScenarioGenerator
)


class CybORGMAPPOEnv:
    """
    Adapter between CybORG and our custom MAPPO implementation.

    - Controls the 5 blue agents
    - Converts CybORG observations into fixed-size vectors
    - Converts discrete MAPPO actions into real CybORG actions
    - Collects rewards
    """

    # CybORG high-level actions
    ACTION_NAMES = [
        "AllowTrafficZone",
        "BlockTrafficZone",
        "Monitor",
        "Analyse",
        "Restore",
        "Remove",
        "DeployDecoy",
        "Sleep"
    ]

    def __init__(self, steps=100):
        self.steps = steps

        # ---------------------------------------------------------
        # Create CybORG scenario
        # ---------------------------------------------------------
        scenario_generator = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=steps
        )

        self.env = CybORG(
            scenario_generator=scenario_generator
        )

        # ---------------------------------------------------------
        # Blue agents
        # ---------------------------------------------------------
        self.agent_ids = [
            agent_id
            for agent_id in self.env.environment_controller.agent_interfaces
            if agent_id.startswith("blue_agent")
        ]

        self.agent_ids.sort()

        self.num_agents = len(self.agent_ids)

        # ---------------------------------------------------------
        # MAPPO settings
        # ---------------------------------------------------------

        # FIXED observation size
        self.observation_dim = 256

        # 8 high-level CybORG actions
        self.action_dim = 8

        # ---------------------------------------------------------
        # Connect secure communication layer
        # ---------------------------------------------------------
        self._connect_agents_to_communication()

    # =============================================================
    # Communication
    # =============================================================

    def _connect_agents_to_communication(self):
        """
        Connect all CybORG agents to the secure communication
        network already implemented in the project.
        """

        try:
            from CybORG.communication.communication_layer import (
                MultiAgentCommunicationNetwork
            )

            agent_ids = list(
                self.env.environment_controller.agent_interfaces.keys()
            )

            self.communication_layer = (
                MultiAgentCommunicationNetwork(agent_ids)
            )

            for agent_interface in (
                self.env.environment_controller.agent_interfaces.values()
            ):
                agent = agent_interface.agent

                if hasattr(agent, "set_communication_layer"):
                    agent.set_communication_layer(
                        self.communication_layer
                    )

        except Exception as error:
            print(
                "Warning: Secure communication layer could not "
                f"be connected: {error}"
            )

            self.communication_layer = None

    # =============================================================
    # Observation handling
    # =============================================================

    def _flatten_observation(self, observation):
        """
        Recursively convert a CybORG observation dictionary into
        numerical values.
        """

        values = []

        def flatten(value):

            # Dictionary
            if isinstance(value, dict):

                for item in value.values():
                    flatten(item)

            # List / tuple
            elif isinstance(value, (list, tuple)):

                for item in value:
                    flatten(item)

            # Boolean
            elif isinstance(value, bool):

                values.append(float(value))

            # Numeric values
            elif isinstance(value, (int, float, np.integer, np.floating)):

                values.append(float(value))

            else:

                # Try converting other simple values to float
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    pass

        flatten(observation)

        return np.asarray(
            values,
            dtype=np.float32
        )

    def _process_observation(self, observation):
        """
        Convert a raw CybORG observation into a fixed-size
        256-dimensional vector.

        If the raw observation is smaller than 256:
            zero padding is added.

        If larger than 256:
            extra values are truncated.
        """

        vector = self._flatten_observation(
            observation
        )

        # ---------------------------------------------------------
        # Padding
        # ---------------------------------------------------------

        if len(vector) < self.observation_dim:

            vector = np.pad(
                vector,
                (
                    0,
                    self.observation_dim - len(vector)
                ),
                mode="constant"
            )

        # ---------------------------------------------------------
        # Truncation
        # ---------------------------------------------------------

        elif len(vector) > self.observation_dim:

            vector = vector[
                :self.observation_dim
            ]

        return vector.astype(
            np.float32
        )

    # =============================================================
    # Reset
    # =============================================================
    def reset(self):
        """
        Reset CybORG and return processed observations.
        """

        results = self.env.reset()

        # Reconnect communication after reset
        self._connect_agents_to_communication()

        observations = {}

        for agent_id in self.agent_ids:

            try:
                # CybORG reset() returns a Results object.
                # The actual observations are stored here.
                raw_observation = results.observation[agent_id]

                observations[agent_id] = (
                    self._process_observation(
                        raw_observation
                    )
                )

            except Exception:
                # If an observation is unavailable,
                # use a zero vector.
                observations[agent_id] = np.zeros(
                    self.observation_dim,
                    dtype=np.float32
                )

        return observations
    # =============================================================
    # Agent information
    # =============================================================

    def get_agent_ids(self):
        return self.agent_ids

    def get_num_agents(self):
        return self.num_agents

    def get_observation_dim(self):
        return self.observation_dim

    def get_action_dim(self):
        return self.action_dim

    # =============================================================
    # Action class
    # =============================================================

    def _find_action_class(
        self,
        agent_id,
        action_name
    ):
        """
        Find the actual CybORG action class.

        CybORG stores action classes as dictionary keys.
        """

        try:

            action_space = (
                self.env.get_action_space(
                    agent_id
                )
            )

            action_options = action_space.get(
                "action",
                {}
            )

            for action_class in action_options.keys():

                if action_class.__name__ == action_name:

                    return action_class

        except Exception:
            pass

        return None

    # =============================================================
    # Parameter selection
    # =============================================================

    def _get_parameter_value(
        self,
        parameter_values
    ):
        """
        Select the first currently valid parameter.
        """

        if not isinstance(
            parameter_values,
            dict
        ):
            return None

        for value, enabled in parameter_values.items():

            if enabled:
                return value

        return None

    def _build_action_parameters(
        self,
        agent_id,
        action_class
    ):
        """
        Build valid parameters required by a CybORG action.
        """

        parameters = {}

        try:
            action_space = self.env.get_action_space(
                agent_id
            )
        except Exception:
            return parameters

        try:
            signature = inspect.signature(
                action_class
            )
        except (TypeError, ValueError):
            return parameters

        # ---------------------------------------------------------
        # Helper: get first valid value
        # ---------------------------------------------------------

        def first_valid(space):

            if not isinstance(space, dict):
                return None

            for value, enabled in space.items():

                if enabled:
                    return value

            return None

        # ---------------------------------------------------------
        # Get available values
        # ---------------------------------------------------------

        session = first_valid(
            action_space.get("session", {})
        )

        agent = first_valid(
            action_space.get("agent", {})
        )

        subnet = first_valid(
            action_space.get("subnet", {})
        )

        hostname = first_valid(
            action_space.get("hostname", {})
        )

        # ---------------------------------------------------------
        # For traffic actions
        #
        # CybORG requires:
        #
        # AllowTrafficZone(
        #     session,
        #     agent,
        #     from_subnet,
        #     to_subnet
        # )
        #
        # BlockTrafficZone(
        #     session,
        #     agent,
        #     from_subnet,
        #     to_subnet
        # )
        # ---------------------------------------------------------

        if action_class.__name__ in (
            "AllowTrafficZone",
            "BlockTrafficZone"
        ):

            parameters["session"] = (
                session
                if session is not None
                else 0
            )

            parameters["agent"] = (
                agent
                if agent is not None
                else agent_id
            )

            # Current available subnet
            if subnet is not None:

                parameters["from_subnet"] = subnet

                parameters["to_subnet"] = subnet

            return parameters

        # ---------------------------------------------------------
        # Normal actions
        # ---------------------------------------------------------

        for parameter_name, parameter in (
            signature.parameters.items()
        ):

            if parameter_name == "self":
                continue

            # Parameter has a default value
            if (
                parameter.default
                is not inspect.Parameter.empty
            ):
                continue

            if parameter_name == "session":

                parameters[
                    "session"
                ] = (
                    session
                    if session is not None
                    else 0
                )

            elif parameter_name == "agent":

                parameters[
                    "agent"
                ] = (
                    agent
                    if agent is not None
                    else agent_id
                )

            elif parameter_name == "hostname":

                if hostname is not None:

                    parameters[
                        "hostname"
                    ] = hostname

        return parameters
    # =============================================================
    # Create CybORG action
    # =============================================================

    def _create_cyborg_action(
        self,
        agent_id,
        action_index
    ):
        """
        Convert a discrete MAPPO action index
        into a real CybORG action object.
        """

        # Make sure action is valid
        action_index = int(action_index)

        if action_index < 0:
            action_index = 0

        if action_index >= self.action_dim:
            action_index = self.action_dim - 1

        action_name = (
            self.ACTION_NAMES[
                action_index
            ]
        )

        # ---------------------------------------------------------
        # Find actual CybORG action class
        # ---------------------------------------------------------

        action_class = self._find_action_class(
            agent_id,
            action_name
        )

        # ---------------------------------------------------------
        # If unavailable -> Sleep
        # ---------------------------------------------------------

        if action_class is None:

            sleep_class = self._find_action_class(
                agent_id,
                "Sleep"
            )

            if sleep_class is not None:

                try:
                    return sleep_class()
                except Exception:
                    pass

            return None

        # ---------------------------------------------------------
        # Build parameters
        # ---------------------------------------------------------

        action_parameters = (
            self._build_action_parameters(
                agent_id,
                action_class
            )
        )

        # ---------------------------------------------------------
        # Create action
        # ---------------------------------------------------------

        try:

            return action_class(
                **action_parameters
            )

        except Exception:

            # If parameter construction fails,
            # use Sleep instead.
            sleep_class = self._find_action_class(
                agent_id,
                "Sleep"
            )

            if sleep_class is not None:

                try:
                    return sleep_class()
                except Exception:
                    pass

            return None

    # =============================================================
    # Reward processing
    # =============================================================

    def _process_reward(self, reward):
        """
        Convert CybORG reward output into a single float.
        """

        # Direct number
        if isinstance(
            reward,
            (int, float, np.integer, np.floating)
        ):

            return float(reward)

        # Dictionary of reward components
        if isinstance(
            reward,
            dict
        ):

            total = 0.0

            for value in reward.values():

                if isinstance(
                    value,
                    (int, float, np.integer, np.floating)
                ):

                    total += float(value)

            return total

        return 0.0

    # =============================================================
    # Step
    # =============================================================

    def step(self, actions):
        """
        Execute one multi-agent step.

        Input:
            actions = {
                "blue_agent_0": 3,
                "blue_agent_1": 7,
                ...
            }

        Output:
            observations
            rewards
            terminated
            info
        """

        # ---------------------------------------------------------
        # Convert MAPPO actions into CybORG actions
        # ---------------------------------------------------------

        cyborg_actions = {}

        for agent_id in self.agent_ids:

            action_index = actions.get(
                agent_id,
                7
            )

            action = self._create_cyborg_action(
                agent_id,
                action_index
            )

            if action is not None:

                cyborg_actions[
                    agent_id
                ] = action

        # ---------------------------------------------------------
        # Execute CybORG step
        # ---------------------------------------------------------

        result = self.env.parallel_step(
            actions=cyborg_actions
        )

        # ---------------------------------------------------------
        # Handle CybORG result
        # ---------------------------------------------------------

        if isinstance(result, tuple):

            if len(result) == 4:

                raw_observations, raw_rewards, dones, infos = result

            elif len(result) == 3:

                raw_observations, raw_rewards, dones = result
                infos = {}

            else:

                raw_observations = result[0]
                raw_rewards = result[1]
                dones = result[2]
                infos = {}

        else:

            raw_observations = {}
            raw_rewards = {}
            dones = False
            infos = {}

        # ---------------------------------------------------------
        # Process observations
        # ---------------------------------------------------------

        observations = {}

        for agent_id in self.agent_ids:

            if agent_id in raw_observations:

                observations[agent_id] = (
                    self._process_observation(
                        raw_observations[
                            agent_id
                        ]
                    )
                )

            else:

                observations[agent_id] = np.zeros(
                    self.observation_dim,
                    dtype=np.float32
                )

        # ---------------------------------------------------------
        # Process rewards
        # ---------------------------------------------------------

        rewards = {}

        for agent_id in self.agent_ids:

            if isinstance(
                raw_rewards,
                dict
            ):

                rewards[agent_id] = (
                    self._process_reward(
                        raw_rewards.get(
                            agent_id,
                            0.0
                        )
                    )
                )

            else:

                rewards[agent_id] = (
                    self._process_reward(
                        raw_rewards
                    )
                )

        # ---------------------------------------------------------
        # Process termination
        # ---------------------------------------------------------

        if isinstance(
            dones,
            dict
        ):

            terminated = any(
                bool(value)
                for value in dones.values()
            )

        else:

            terminated = bool(
                dones
            )

        return (
            observations,
            rewards,
            terminated,
            infos
        )

    # =============================================================
    # Close
    # =============================================================

    def close(self, **kwargs):
        """
        CybORG CC4 environment does not require GUI cleanup.
        """

        return


# =============================================================
# Direct test
# =============================================================

if __name__ == "__main__":

    print(
        "===== CybORG MAPPO ENV TEST ====="
    )

    env = CybORGMAPPOEnv(
        steps=10
    )

    print(
        "Agents:",
        env.get_agent_ids()
    )

    print(
        "Number of agents:",
        env.get_num_agents()
    )

    print(
        "Observation dimension:",
        env.get_observation_dim()
    )

    print(
        "Action dimension:",
        env.get_action_dim()
    )

    observations = env.reset()

    print(
        "\nObservation shapes:"
    )

    for agent_id in env.get_agent_ids():

        print(
            agent_id,
            observations[
                agent_id
            ].shape
        )

    print(
        "\n===== ENV TEST PASSED ====="
    )