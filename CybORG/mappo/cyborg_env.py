import numpy as np

from CybORG import CybORG
from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import (
    EnterpriseScenarioGenerator,
)

from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import (
    EnterpriseGreenAgent,
)
from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import (
    FiniteStateRedAgent,
)

from CybORG.communication.communication_layer import (
    MultiAgentCommunicationNetwork,
)


class CybORGMAPPOEnv:
    """
    Custom CybORG environment for MAPPO.

    Features:
        - 5 Blue agents controlled by MAPPO
        - Fixed observation dimension
        - Global target-aware action space
        - Agent-specific action masks
        - Secure communication network
        - Positive rewards
        - Compatible with CybORG tuple-based parallel_step()
    """

    def __init__(self, steps=50):

        self.steps = steps

        # =========================================================
        # CREATE CYBORG SCENARIO
        # =========================================================

        scenario_generator = EnterpriseScenarioGenerator(
            red_agent_class=FiniteStateRedAgent,
            green_agent_class=EnterpriseGreenAgent,
            steps=steps,
        )

        self.env = CybORG(
            scenario_generator=scenario_generator,
            seed=0,
        )

        # =========================================================
        # BLUE AGENTS
        # =========================================================

        self.agent_ids = sorted(
            [
                agent_id
                for agent_id in self.env.agents
                if agent_id.startswith("blue_agent")
            ]
        )

        # =========================================================
        # OBSERVATION
        # =========================================================

        self.observation_dim = 256

        # =========================================================
        # SECURE COMMUNICATION
        # =========================================================

        self.communication_network = None

        self._connect_agents_to_communication()

        # =========================================================
        # ACTION MAPPINGS
        # =========================================================

        self.global_action_mapping = []

        self.action_mappings = {}

        self.action_masks = {}

        self._build_global_action_mappings()

        self.action_dim = len(
            self.global_action_mapping
        )

    # =============================================================
    # COMMUNICATION
    # =============================================================

    def _connect_agents_to_communication(self):

        try:

            self.communication_network = (
                MultiAgentCommunicationNetwork(
                    self.agent_ids
                )
            )

        except TypeError:

            try:

                self.communication_network = (
                    MultiAgentCommunicationNetwork(
                        agent_ids=self.agent_ids
                    )
                )

            except Exception:

                self.communication_network = None

        except Exception:

            self.communication_network = None

    # =============================================================
    # OBSERVATION PROCESSING
    # =============================================================

    def _flatten_observation(
        self,
        observation,
    ):

        values = []

        def recursive_flatten(value):

            if value is None:
                return

            # -----------------------------------------------------
            # Dictionary
            # -----------------------------------------------------

            if isinstance(
                value,
                dict,
            ):

                for key in sorted(
                    value.keys(),
                    key=str,
                ):

                    recursive_flatten(
                        value[key]
                    )

            # -----------------------------------------------------
            # List / Tuple
            # -----------------------------------------------------

            elif isinstance(
                value,
                (list, tuple),
            ):

                for item in value:

                    recursive_flatten(
                        item
                    )

            # -----------------------------------------------------
            # NumPy Array
            # -----------------------------------------------------

            elif isinstance(
                value,
                np.ndarray,
            ):

                for item in value.flatten():

                    if np.isscalar(item):

                        try:

                            values.append(
                                float(item)
                            )

                        except Exception:

                            pass

            # -----------------------------------------------------
            # Numeric value
            # -----------------------------------------------------

            elif isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                    bool,
                ),
            ):

                values.append(
                    float(value)
                )

        recursive_flatten(
            observation
        )

        return np.asarray(
            values,
            dtype=np.float32,
        )

    def _process_observation(
        self,
        observation,
    ):

        flattened = (
            self._flatten_observation(
                observation
            )
        )

        result = np.zeros(
            self.observation_dim,
            dtype=np.float32,
        )

        size = min(
            len(flattened),
            self.observation_dim,
        )

        if size > 0:

            result[:size] = (
                flattened[:size]
            )

        return result

    # =============================================================
    # RESET
    # =============================================================

    def reset(self):

        results = self.env.reset()

        # ---------------------------------------------------------
        # Handle CybORG reset result
        # ---------------------------------------------------------

        if isinstance(
            results,
            tuple,
        ):

            raw_observations = (
                results[0]
            )

        else:

            raw_observations = getattr(
                results,
                "observation",
                results,
            )

        observations = {}

        for agent_id in self.agent_ids:

            if (
                isinstance(
                    raw_observations,
                    dict,
                )
                and agent_id
                in raw_observations
            ):

                observations[
                    agent_id
                ] = self._process_observation(
                    raw_observations[
                        agent_id
                    ]
                )

            else:

                observations[
                    agent_id
                ] = np.zeros(
                    self.observation_dim,
                    dtype=np.float32,
                )

        return observations

    # =============================================================
    # FIND ACTION CLASS
    # =============================================================

    def _find_action_class(
        self,
        action_name,
    ):

        import CybORG.Simulator.Actions as actions_module

        return getattr(
            actions_module,
            action_name,
            None,
        )

    # =============================================================
    # GET VALID VALUES
    # =============================================================

    def _get_all_valid_values(
        self,
        space,
        key,
    ):
        """
        CybORG action spaces use dictionaries like:

            {
                value_1: True,
                value_2: False,
                value_3: True
            }

        True means the value is currently valid.

        Therefore we return only the KEYS whose value is True.
        """

        try:

            if key not in space:

                return []

            values = space[key]

            if values is None:

                return []

            # -----------------------------------------------------
            # CybORG dictionary format
            # -----------------------------------------------------

            if isinstance(
                values,
                dict,
            ):

                valid_values = []

                for actual_value, is_valid in (
                    values.items()
                ):

                    if is_valid is True:

                        valid_values.append(
                            actual_value
                        )

                return valid_values

            # -----------------------------------------------------
            # Other iterable formats
            # -----------------------------------------------------

            if isinstance(
                values,
                (list, tuple, set),
            ):

                return list(
                    values
                )

            # -----------------------------------------------------
            # Single value
            # -----------------------------------------------------

            return [
                values
            ]

        except Exception:

            return []

    # =============================================================
    # ACTION KEY
    # =============================================================

    def _action_key(
        self,
        action,
    ):

        action_name = action[
            "action_name"
        ]

        # ---------------------------------------------------------
        # SLEEP
        # ---------------------------------------------------------

        if action_name == "Sleep":

            return (
                "Sleep",
            )

        # ---------------------------------------------------------
        # MONITOR
        # ---------------------------------------------------------

        if action_name == "Monitor":

            return (
                "Monitor",
                action.get(
                    "session"
                ),
                action.get(
                    "agent"
                ),
            )

        # ---------------------------------------------------------
        # HOST ACTIONS
        # ---------------------------------------------------------

        if action_name in [
            "Analyse",
            "Restore",
            "Remove",
            "DeployDecoy",
        ]:

            return (
                action_name,
                action.get(
                    "session"
                ),
                action.get(
                    "agent"
                ),
                action.get(
                    "hostname"
                ),
            )

        # ---------------------------------------------------------
        # TRAFFIC ACTIONS
        # ---------------------------------------------------------

        if action_name in [
            "AllowTrafficZone",
            "BlockTrafficZone",
        ]:

            return (
                action_name,
                action.get(
                    "session"
                ),
                action.get(
                    "agent"
                ),
                str(
                    action.get(
                        "from_subnet"
                    )
                ),
                str(
                    action.get(
                        "to_subnet"
                    )
                ),
            )

        return tuple(
            sorted(
                (
                    key,
                    str(value),
                )
                for key, value in action.items()
                if key != "action_name"
            )
        )

    # =============================================================
    # BUILD AGENT ACTION CANDIDATES
    # =============================================================

    def _build_agent_action_candidates(
        self,
        agent_id,
    ):

        candidates = []

        # ---------------------------------------------------------
        # GET CYBORG ACTION SPACE
        # ---------------------------------------------------------

        try:

            action_space = (
                self.env.get_action_space(
                    agent_id
                )
            )

        except Exception:

            action_space = {}

        # ---------------------------------------------------------
        # SLEEP
        # ---------------------------------------------------------

        candidates.append(
            {
                "action_name": "Sleep"
            }
        )

        # ---------------------------------------------------------
        # SESSIONS
        # ---------------------------------------------------------

        sessions = (
            self._get_all_valid_values(
                action_space,
                "session",
            )
        )

        # ---------------------------------------------------------
        # MONITOR
        # ---------------------------------------------------------

        for session in sessions:

            candidates.append(
                {
                    "action_name": "Monitor",
                    "session": session,
                    "agent": agent_id,
                }
            )

        # ---------------------------------------------------------
        # HOSTNAMES
        # ---------------------------------------------------------

        hostnames = (
            self._get_all_valid_values(
                action_space,
                "hostname",
            )
        )

        # ---------------------------------------------------------
        # HOST ACTIONS
        # ---------------------------------------------------------

        for session in sessions:

            for hostname in hostnames:

                for action_name in [
                    "Analyse",
                    "Restore",
                    "Remove",
                    "DeployDecoy",
                ]:

                    candidates.append(
                        {
                            "action_name":
                                action_name,

                            "session":
                                session,

                            "agent":
                                agent_id,

                            "hostname":
                                hostname,
                        }
                    )

        # ---------------------------------------------------------
        # SUBNETS
        # ---------------------------------------------------------

        subnets = (
            self._get_all_valid_values(
                action_space,
                "subnet",
            )
        )

        # ---------------------------------------------------------
        # TRAFFIC ACTIONS
        # ---------------------------------------------------------

        for session in sessions:

            for subnet in subnets:

                candidates.append(
                    {
                        "action_name":
                            "AllowTrafficZone",

                        "session":
                            session,

                        "agent":
                            agent_id,

                        "from_subnet":
                            subnet,

                        "to_subnet":
                            subnet,
                    }
                )

                candidates.append(
                    {
                        "action_name":
                            "BlockTrafficZone",

                        "session":
                            session,

                        "agent":
                            agent_id,

                        "from_subnet":
                            subnet,

                        "to_subnet":
                            str(subnet),
                    }
                )

        return candidates

    # =============================================================
    # BUILD GLOBAL ACTION MAPPING
    # =============================================================

    def _build_global_action_mappings(self):

        global_actions = []

        global_keys = {}

        # ---------------------------------------------------------
        # COLLECT ACTIONS
        # ---------------------------------------------------------

        for agent_id in self.agent_ids:

            candidates = (
                self._build_agent_action_candidates(
                    agent_id
                )
            )

            for action in candidates:

                key = (
                    self._action_key(
                        action
                    )
                )

                if key not in global_keys:

                    global_keys[
                        key
                    ] = len(
                        global_actions
                    )

                    global_actions.append(
                        action.copy()
                    )

        # ---------------------------------------------------------
        # ENSURE ONLY ONE SLEEP
        # ---------------------------------------------------------

        sleep_found = False

        cleaned_actions = []

        for action in global_actions:

            if action[
                "action_name"
            ] == "Sleep":

                if sleep_found:

                    continue

                sleep_found = True

            cleaned_actions.append(
                action
            )

        if not sleep_found:

            cleaned_actions.insert(
                0,
                {
                    "action_name":
                        "Sleep"
                },
            )

        self.global_action_mapping = (
            cleaned_actions
        )

        # ---------------------------------------------------------
        # BUILD MASKS
        # ---------------------------------------------------------

        for agent_id in self.agent_ids:

            candidates = (
                self._build_agent_action_candidates(
                    agent_id
                )
            )

            valid_keys = {
                self._action_key(
                    action
                )
                for action in candidates
            }

            mask = np.zeros(
                len(
                    self.global_action_mapping
                ),
                dtype=np.float32,
            )

            for index, action in enumerate(
                self.global_action_mapping
            ):

                key = (
                    self._action_key(
                        action
                    )
                )

                if key in valid_keys:

                    mask[index] = 1.0

            # -----------------------------------------------------
            # Sleep always valid
            # -----------------------------------------------------

            for index, action in enumerate(
                self.global_action_mapping
            ):

                if (
                    action[
                        "action_name"
                    ]
                    == "Sleep"
                ):

                    mask[index] = 1.0

            self.action_masks[
                agent_id
            ] = mask

            self.action_mappings[
                agent_id
            ] = (
                self.global_action_mapping
            )

    # =============================================================
    # GET ACTION MASK
    # =============================================================

    def get_action_mask(
        self,
        agent_id,
    ):

        return self.action_masks[
            agent_id
        ].copy()

    # =============================================================
    # PRINT ACTION MAPPINGS
    # =============================================================

    def print_action_mappings(
        self,
    ):

        print()
        print(
            "===== GLOBAL ACTION MAPPING ====="
        )

        for index, action in enumerate(
            self.global_action_mapping
        ):

            print(
                f"{index:03d} -> "
                f"{action}"
            )

        print()

        print(
            "Total global actions:",
            len(
                self.global_action_mapping
            ),
        )

        print()

        print(
            "===== AGENT ACTION MASKS ====="
        )

        for agent_id in self.agent_ids:

            mask = (
                self.action_masks[
                    agent_id
                ]
            )

            valid_count = int(
                np.sum(mask)
            )

            invalid_count = int(
                np.sum(
                    mask == 0
                )
            )

            print(
                f"{agent_id}: "
                f"{valid_count}/"
                f"{self.action_dim} "
                f"valid actions "
                f"({invalid_count} invalid)"
            )

    # =============================================================
    # CREATE CYBORG ACTION
    # =============================================================

    def _create_cyborg_action(
        self,
        agent_id,
        action_index,
    ):

        # ---------------------------------------------------------
        # VALIDATE INDEX
        # ---------------------------------------------------------

        if action_index < 0:

            action_index = 0

        if (
            action_index
            >= self.action_dim
        ):

            action_index = 0

        # ---------------------------------------------------------
        # VALIDATE MASK
        # ---------------------------------------------------------

        mask = (
            self.action_masks.get(
                agent_id
            )
        )

        if mask is None:

            action_index = 0

        elif (
            mask[action_index]
            == 0
        ):

            action_index = 0

        # ---------------------------------------------------------
        # GET MAPPING
        # ---------------------------------------------------------

        mapping = (
            self.global_action_mapping[
                action_index
            ]
        )

        action_name = (
            mapping[
                "action_name"
            ]
        )

        # ---------------------------------------------------------
        # SLEEP
        # ---------------------------------------------------------

        if action_name == "Sleep":

            action_class = (
                self._find_action_class(
                    "Sleep"
                )
            )

            return action_class()

        # ---------------------------------------------------------
        # PARAMETERS
        # ---------------------------------------------------------

        parameters = {
            key: value
            for key, value in mapping.items()
            if key != "action_name"
        }

        # ---------------------------------------------------------
        # FIND ACTION CLASS
        # ---------------------------------------------------------

        action_class = (
            self._find_action_class(
                action_name
            )
        )

        if action_class is None:

            sleep_class = (
                self._find_action_class(
                    "Sleep"
                )
            )

            return sleep_class()

        # ---------------------------------------------------------
        # CREATE ACTION
        # ---------------------------------------------------------

        try:

            return action_class(
                **parameters
            )

        except Exception:

            sleep_class = (
                self._find_action_class(
                    "Sleep"
                )
            )

            return sleep_class()

    # =============================================================
    # POSITIVE REWARD
    # =============================================================

    def _process_reward(
        self,
        reward,
    ):

        raw_reward = 0.0

        # ---------------------------------------------------------
        # NUMERIC
        # ---------------------------------------------------------

        if isinstance(
            reward,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):

            raw_reward = float(
                reward
            )

        # ---------------------------------------------------------
        # DICTIONARY
        # ---------------------------------------------------------

        elif isinstance(
            reward,
            dict,
        ):

            for value in (
                reward.values()
            ):

                if isinstance(
                    value,
                    (
                        int,
                        float,
                        np.integer,
                        np.floating,
                    ),
                ):

                    raw_reward += float(
                        value
                    )

        # ---------------------------------------------------------
        # SCALE
        # ---------------------------------------------------------

        scaled_reward = (
            raw_reward / 5.0
        )

        # ---------------------------------------------------------
        # SOFTPLUS
        # ---------------------------------------------------------

        if scaled_reward > 20:

            positive_reward = (
                scaled_reward
            )

        else:

            positive_reward = (
                np.log1p(
                    np.exp(
                        scaled_reward
                    )
                )
            )

        # ---------------------------------------------------------
        # GUARANTEE POSITIVE
        # ---------------------------------------------------------

        positive_reward += 0.01

        return float(
            positive_reward
        )

    # =============================================================
    # STEP
    # =============================================================

    def step(
        self,
        actions,
    ):
        """
        Execute one CybORG environment step.

        Supports both possible CybORG return formats:

        1. Result object:

            results.observation
            results.reward
            results.done
            results.info

        2. Tuple:

            (
                observations,
                rewards,
                done,
                info
            )
        """

        # =========================================================
        # CONVERT MAPPO ACTIONS
        # =========================================================

        cyborg_actions = {}

        for agent_id in self.agent_ids:

            action_index = actions.get(
                agent_id,
                0,
            )

            cyborg_actions[
                agent_id
            ] = self._create_cyborg_action(
                agent_id,
                int(
                    action_index
                ),
            )

        # =========================================================
        # EXECUTE CYBORG
        # =========================================================

        results = self.env.parallel_step(
            actions=cyborg_actions
        )

        # =========================================================
        # NORMALIZE RESULT
        # =========================================================

        if isinstance(
            results,
            tuple,
        ):

            # -----------------------------------------------------
            # TUPLE RETURN
            # -----------------------------------------------------

            if len(results) == 4:

                raw_observations = (
                    results[0]
                )

                raw_rewards = (
                    results[1]
                )

                raw_done = (
                    results[2]
                )

                raw_infos = (
                    results[3]
                )

            elif len(results) == 3:

                raw_observations = (
                    results[0]
                )

                raw_rewards = (
                    results[1]
                )

                raw_done = (
                    results[2]
                )

                raw_infos = {}

            else:

                raise RuntimeError(
                    "Unexpected CybORG "
                    "parallel_step() result. "
                    f"Expected 3 or 4 values, "
                    f"got {len(results)}."
                )

        else:

            # -----------------------------------------------------
            # RESULT OBJECT
            # -----------------------------------------------------

            raw_observations = getattr(
                results,
                "observation",
                {},
            )

            raw_rewards = getattr(
                results,
                "reward",
                {},
            )

            raw_done = getattr(
                results,
                "done",
                False,
            )

            raw_infos = getattr(
                results,
                "info",
                {},
            )

        # =========================================================
        # OBSERVATIONS
        # =========================================================

        observations = {}

        for agent_id in self.agent_ids:

            if (
                isinstance(
                    raw_observations,
                    dict,
                )
                and agent_id
                in raw_observations
            ):

                observations[
                    agent_id
                ] = self._process_observation(
                    raw_observations[
                        agent_id
                    ]
                )

            else:

                observations[
                    agent_id
                ] = np.zeros(
                    self.observation_dim,
                    dtype=np.float32,
                )

        # =========================================================
        # REWARDS
        # =========================================================

        rewards = {}

        for agent_id in self.agent_ids:

            raw_reward = 0.0

            if (
                isinstance(
                    raw_rewards,
                    dict,
                )
                and agent_id
                in raw_rewards
            ):

                raw_reward = (
                    raw_rewards[
                        agent_id
                    ]
                )

            rewards[
                agent_id
            ] = self._process_reward(
                raw_reward
            )

        # =========================================================
        # TERMINATION
        # =========================================================

        terminated = False

        if isinstance(
            raw_done,
            dict,
        ):

            terminated = any(
                bool(value)
                for value
                in raw_done.values()
            )

        else:

            terminated = bool(
                raw_done
            )

        # =========================================================
        # INFO
        # =========================================================

        infos = {}

        for agent_id in self.agent_ids:

            original_info = {}

            if (
                isinstance(
                    raw_infos,
                    dict,
                )
                and agent_id
                in raw_infos
            ):

                if isinstance(
                    raw_infos[
                        agent_id
                    ],
                    dict,
                ):

                    original_info = (
                        raw_infos[
                            agent_id
                        ].copy()
                    )

            # -----------------------------------------------------
            # Add action mask
            # -----------------------------------------------------

            original_info[
                "action_mask"
            ] = self.action_masks[
                agent_id
            ].copy()

            infos[
                agent_id
            ] = original_info

        # =========================================================
        # RETURN
        # =========================================================

        return (
            observations,
            rewards,
            terminated,
            infos,
        )

    # =============================================================
    # CLOSE
    # =============================================================

    def close(self):

        try:

            self.env.close()

        except Exception:

            pass


# =================================================================
# TEST ENVIRONMENT
# =================================================================

if __name__ == "__main__":

    print()
    print(
        "===== CybORG MAPPO ENV TEST ====="
    )

    env = CybORGMAPPOEnv(
        steps=10
    )

    # -------------------------------------------------------------
    # BASIC INFORMATION
    # -------------------------------------------------------------

    print(
        f"Agents: "
        f"{env.agent_ids}"
    )

    print(
        f"Number of agents: "
        f"{len(env.agent_ids)}"
    )

    print(
        f"Observation dimension: "
        f"{env.observation_dim}"
    )

    print(
        f"Global action dimension: "
        f"{env.action_dim}"
    )

    # -------------------------------------------------------------
    # ACTION MAPPING
    # -------------------------------------------------------------

    env.print_action_mappings()

    # -------------------------------------------------------------
    # ACTION MASK TEST
    # -------------------------------------------------------------

    print()
    print(
        "===== ACTION MASK TEST ====="
    )

    for agent_id in env.agent_ids:

        mask = (
            env.get_action_mask(
                agent_id
            )
        )

        valid_count = int(
            np.sum(mask)
        )

        invalid_count = int(
            np.sum(
                mask == 0
            )
        )

        print(
            f"{agent_id}: "
            f"shape={mask.shape}, "
            f"valid={valid_count}, "
            f"invalid={invalid_count}"
        )

    # -------------------------------------------------------------
    # RESET
    # -------------------------------------------------------------

    observations = (
        env.reset()
    )

    print()
    print(
        "Observation shapes:"
    )

    for agent_id in env.agent_ids:

        print(
            agent_id,
            observations[
                agent_id
            ].shape,
        )

    # -------------------------------------------------------------
    # BASIC ASSERTIONS
    # -------------------------------------------------------------

    assert (
        env.action_dim
        == len(
            env.global_action_mapping
        )
    )

    assert (
        env.action_dim > 0
    )

    for agent_id in env.agent_ids:

        assert (
            agent_id
            in env.action_masks
        )

        assert (
            len(
                env.action_masks[
                    agent_id
                ]
            )
            == env.action_dim
        )

        assert (
            np.sum(
                env.action_masks[
                    agent_id
                ]
            )
            >= 1
        )

    # -------------------------------------------------------------
    # EXACTLY ONE SLEEP
    # -------------------------------------------------------------

    sleep_indices = [
        index
        for index, action
        in enumerate(
            env.global_action_mapping
        )
        if (
            action[
                "action_name"
            ]
            == "Sleep"
        )
    ]

    assert (
        len(
            sleep_indices
        )
        == 1
    )

    sleep_index = (
        sleep_indices[0]
    )

    # -------------------------------------------------------------
    # SLEEP MUST BE VALID
    # -------------------------------------------------------------

    for agent_id in env.agent_ids:

        assert (
            env.action_masks[
                agent_id
            ][sleep_index]
            == 1.0
        )

    print()
    print(
        f"Unique Sleep action index: "
        f"{sleep_index}"
    )

    print()
    print(
        "===== ENV TEST PASSED ====="
    )

    env.close()