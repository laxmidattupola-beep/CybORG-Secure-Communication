import inspect
import enum
import numpy as np
import CybORG.Shared.Enums as CyEnums

from CybORG import CybORG
from CybORG.Agents import (
    SleepAgent,
    EnterpriseGreenAgent,
    FiniteStateRedAgent,
)
from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import (
    EnterpriseScenarioGenerator,
)


def _build_semantic_feature_names():
    """Build the stable names used by the 256-dimensional observation encoder."""
    enum_classes = []
    for candidate in vars(CyEnums).values():
        if isinstance(candidate, type):
            try:
                if (
                    issubclass(candidate, enum.Enum)
                    and candidate.__module__ == CyEnums.__name__
                ):
                    enum_classes.append(candidate)
            except TypeError:
                pass

    enum_names = []
    for enum_class in sorted(enum_classes, key=lambda item: item.__name__):
        for member in enum_class:
            enum_names.append(f"enum.{enum_class.__name__}.{member.name}")

    structure_metrics = [
        "structure.host_count",
        "structure.process_count",
        "structure.connection_count",
        "structure.interface_count",
        "structure.session_count",
        "structure.file_count",
        "structure.user_count",
        "structure.vulnerability_count",
        "structure.system_info_count",
        "structure.network_connection_count",
        "structure.blocked_ip_count",
        "structure.processes_with_process_name_count",
        "structure.processes_with_known_process_count",
        "structure.processes_with_program_name_count",
        "structure.processes_with_service_name_count",
        "structure.processes_with_username_count",
        "structure.processes_with_known_path_count",
        "structure.processes_with_process_type_count",
        "structure.processes_with_process_version_count",
        "structure.processes_with_status_count",
        "structure.processes_with_transport_protocol_count",
        "structure.processes_with_application_protocol_count",
        "structure.processes_with_vulnerability_count",
        "structure.hosts_with_system_info_count",
        "structure.hosts_with_interface_count",
        "structure.hosts_with_processes_count",
        "structure.hosts_with_sessions_count",
        "structure.hosts_with_files_count",
        "structure.hosts_with_users_count",
        "structure.hosts_with_vulnerabilities_count",
    ]

    structural_keys = [
        "success", "processes", "connections", "interface", "sessions",
        "files", "users", "vulnerability", "systeminfo", "knownprocess",
        "programname", "processname", "servicename", "username",
        "knownpath", "processtype", "processversion", "status",
        "transportprotocol", "applicationprotocol", "ostype",
        "osdistribution", "osversion", "oskernelversion", "architecture",
        "ospatches", "subnet", "interfacename", "networkconnections",
        "blockedips", "filetype", "vendor", "version", "signed",
        "userpermissions", "grouppermissions", "defaultpermissions",
        "sessiontype", "sessionstatus", "sourcehost", "desthost",
    ]
    structural_names = [
        f"structure.field_count.{key}" for key in structural_keys
    ]

    names = (enum_names + structure_metrics + structural_names)[:256]
    while len(names) < 256:
        names.append(f"padding[{len(names)}]")
    return names


SEMANTIC_FEATURE_NAMES = _build_semantic_feature_names()
OBSERVATION_ENCODER_VERSION = "semantic_enum_aggregate_v1"


class CybORGMAPPOEnv:
    """Adapter between CybORG and the custom MAPPO implementation."""

    ACTION_NAMES = [
        "AllowTrafficZone",
        "BlockTrafficZone",
        "Monitor",
        "Analyse",
        "Restore",
        "Remove",
        "DeployDecoy",
        "Sleep",
    ]

    def __init__(self, steps=100):
        self.steps = steps
        scenario_generator = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=steps,
        )
        self.env = CybORG(scenario_generator=scenario_generator)

        self.agent_ids = sorted(
            agent_id
            for agent_id in self.env.environment_controller.agent_interfaces
            if agent_id.startswith("blue_agent")
        )
        self.num_agents = len(self.agent_ids)
        self.observation_dim = 256
        self.action_dim = 8
        self.feature_names_by_agent = {}
        self.xai_context_by_agent = {}
        self.communication_layer = None
        self._connect_agents_to_communication()

    # -------------------------------------------------------------
    # Communication
    # -------------------------------------------------------------

    def _connect_agents_to_communication(self):
        """Connect agents to the project's secure communication layer, if available."""
        try:
            from CybORG.communication.communication_layer import (
                MultiAgentCommunicationNetwork,
            )

            agent_ids = list(
                self.env.environment_controller.agent_interfaces.keys()
            )
            self.communication_layer = MultiAgentCommunicationNetwork(agent_ids)

            for agent_interface in (
                self.env.environment_controller.agent_interfaces.values()
            ):
                agent = agent_interface.agent
                if hasattr(agent, "set_communication_layer"):
                    agent.set_communication_layer(self.communication_layer)

        except Exception as error:
            print(
                "Warning: Secure communication layer could not be connected: "
                f"{error}"
            )
            self.communication_layer = None

    # -------------------------------------------------------------
    # Observation encoding
    # -------------------------------------------------------------

    def _flatten_observation_with_names(self, observation):
        """Convert a raw CybORG observation into a stable 256-value vector."""
        names = list(SEMANTIC_FEATURE_NAMES[:self.observation_dim])
        vector = np.zeros(self.observation_dim, dtype=np.float32)
        index_by_name = {name: i for i, name in enumerate(names)}

        field_counts = {}
        item_counts = {
            "processes": 0,
            "connections": 0,
            "interface": 0,
            "sessions": 0,
            "files": 0,
            "users": 0,
            "vulnerability": 0,
            "systeminfo": 0,
            "networkconnections": 0,
            "blockedips": 0,
        }
        process_records = []
        host_records = []

        def normalize_key(key):
            return "".join(ch.lower() for ch in str(key) if ch.isalnum())

        def walk(value):
            if isinstance(value, enum.Enum):
                feature_name = f"enum.{type(value).__name__}.{value.name}"
                feature_index = index_by_name.get(feature_name)
                if feature_index is not None:
                    vector[feature_index] = 1.0
                return

            if isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(key, enum.Enum):
                        walk(key)

                    key_text = key.name if isinstance(key, enum.Enum) else key
                    normalized_key = normalize_key(key_text)
                    field_counts[normalized_key] = (
                        field_counts.get(normalized_key, 0) + 1
                    )

                    if normalized_key in item_counts:
                        if isinstance(child, (list, tuple)):
                            item_counts[normalized_key] += len(child)
                        elif child is not None:
                            item_counts[normalized_key] += 1

                    if (
                        normalized_key == "processes"
                        and isinstance(child, (list, tuple))
                    ):
                        process_records.extend(
                            item for item in child if isinstance(item, dict)
                        )

                    if isinstance(child, dict):
                        walk(child)
                    elif isinstance(child, (list, tuple)):
                        for item in child:
                            walk(item)
                    elif isinstance(child, enum.Enum):
                        walk(child)
                return

            if isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)

        if isinstance(observation, dict):
            metadata_keys = {"success", "message", "msg"}
            for key, value in observation.items():
                if (
                    isinstance(value, dict)
                    and normalize_key(key) not in metadata_keys
                ):
                    host_records.append(value)

        walk(observation)

        def record_has(record, wanted_key):
            return wanted_key in [normalize_key(key) for key in record]

        metric_values = {
            "structure.host_count": len(host_records),
            "structure.process_count": item_counts["processes"],
            "structure.connection_count": item_counts["connections"],
            "structure.interface_count": item_counts["interface"],
            "structure.session_count": item_counts["sessions"],
            "structure.file_count": item_counts["files"],
            "structure.user_count": item_counts["users"],
            "structure.vulnerability_count": item_counts["vulnerability"],
            "structure.system_info_count": item_counts["systeminfo"],
            "structure.network_connection_count": item_counts["networkconnections"],
            "structure.blocked_ip_count": item_counts["blockedips"],
            "structure.processes_with_process_name_count": sum(
                record_has(record, "processname") for record in process_records
            ),
            "structure.processes_with_known_process_count": sum(
                record_has(record, "knownprocess") for record in process_records
            ),
            "structure.processes_with_program_name_count": sum(
                record_has(record, "programname") for record in process_records
            ),
            "structure.processes_with_service_name_count": sum(
                record_has(record, "servicename") for record in process_records
            ),
            "structure.processes_with_username_count": sum(
                record_has(record, "username") for record in process_records
            ),
            "structure.processes_with_known_path_count": sum(
                record_has(record, "knownpath") for record in process_records
            ),
            "structure.processes_with_process_type_count": sum(
                record_has(record, "processtype") for record in process_records
            ),
            "structure.processes_with_process_version_count": sum(
                record_has(record, "processversion") for record in process_records
            ),
            "structure.processes_with_status_count": sum(
                record_has(record, "status") for record in process_records
            ),
            "structure.processes_with_transport_protocol_count": sum(
                record_has(record, "transportprotocol") for record in process_records
            ),
            "structure.processes_with_application_protocol_count": sum(
                record_has(record, "applicationprotocol") for record in process_records
            ),
            "structure.processes_with_vulnerability_count": sum(
                record_has(record, "vulnerability") for record in process_records
            ),
            "structure.hosts_with_system_info_count": sum(
                record_has(host, "systeminfo") for host in host_records
            ),
            "structure.hosts_with_interface_count": sum(
                record_has(host, "interface") for host in host_records
            ),
            "structure.hosts_with_processes_count": sum(
                record_has(host, "processes") for host in host_records
            ),
            "structure.hosts_with_sessions_count": sum(
                record_has(host, "sessions") for host in host_records
            ),
            "structure.hosts_with_files_count": sum(
                record_has(host, "files") for host in host_records
            ),
            "structure.hosts_with_users_count": sum(
                record_has(host, "users") for host in host_records
            ),
            "structure.hosts_with_vulnerabilities_count": sum(
                record_has(host, "vulnerability") for host in host_records
            ),
        }

        for feature_name, count in metric_values.items():
            feature_index = index_by_name.get(feature_name)
            if feature_index is not None:
                vector[feature_index] = min(float(count), 10.0) / 10.0

        prefix = "structure.field_count."
        for feature_name in names:
            if feature_name.startswith(prefix):
                key_name = feature_name[len(prefix):]
                vector[index_by_name[feature_name]] = min(
                    float(field_counts.get(key_name, 0)), 10.0
                ) / 10.0

        return vector, names

    def extract_xai_context(self, observation):
        """Extract descriptive context without changing MAPPO model inputs."""
        context = {}
        keywords = (
            "success", "hostname", "ostype", "osdistribution",
            "osversion", "architecture", "interface", "ip_address",
            "subnet", "username", "type",
        )
        visited = set()

        def walk(value, path="observation", depth=0):
            if depth > 12 or value is None:
                return

            if isinstance(value, (str, int, float, bool)):
                if any(word in path.lower() for word in keywords):
                    context[path] = str(value)
                return

            if hasattr(value, "name") and hasattr(value, "value"):
                if any(word in path.lower() for word in keywords):
                    context[path] = str(value.name)
                return

            if isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    walk(item, f"{path}[{index}]", depth + 1)
                return

            if isinstance(value, dict):
                object_id = id(value)
                if object_id in visited:
                    return
                visited.add(object_id)
                for key, item in value.items():
                    walk(item, f"{path}.{key}", depth + 1)
                return

            attributes = getattr(value, "__dict__", None)
            if isinstance(attributes, dict):
                object_id = id(value)
                if object_id in visited:
                    return
                visited.add(object_id)
                for key, item in attributes.items():
                    if not key.startswith("_"):
                        walk(item, f"{path}.{key}", depth + 1)

        walk(observation)
        return context

    def get_xai_context(self, agent_id):
        """Return context captured from the original CybORG observation."""
        return dict(self.xai_context_by_agent.get(agent_id, {}))

    def _flatten_observation(self, observation):
        """Backward-compatible helper that returns only the numeric vector."""
        values, _ = self._flatten_observation_with_names(observation)
        return values

    def _process_observation(self, observation, agent_id=None):
        """Encode an observation and store feature names/context for XAI."""
        if agent_id is not None:
            self.xai_context_by_agent[agent_id] = self.extract_xai_context(
                observation
            )

        vector, names = self._flatten_observation_with_names(observation)

        if agent_id == "blue_agent_0":
            print(
                "[SEMANTIC FEATURE AUDIT] active features: "
                f"{int(np.count_nonzero(vector))}; "
                f"model input size: {self.observation_dim}"
            )

        if len(vector) < self.observation_dim:
            padding_count = self.observation_dim - len(vector)
            vector = np.pad(vector, (0, padding_count), mode="constant")
            names.extend(
                f"padding[{i}]"
                for i in range(padding_count)
            )
        elif len(vector) > self.observation_dim:
            vector = vector[:self.observation_dim]
            names = names[:self.observation_dim]

        self.feature_names_by_agent[agent_id] = names
        return vector.astype(np.float32)

    def get_feature_names(self, agent_id):
        """Return the 256 feature names for an agent."""
        names = list(self.feature_names_by_agent.get(agent_id, []))
        if len(names) < self.observation_dim:
            names.extend(
                f"padding[{i}]"
                for i in range(len(names), self.observation_dim)
            )
        return names[:self.observation_dim]

    # -------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------

    def reset(self):
        """Reset CybORG and return processed observations."""
        results = self.env.reset()
        self._connect_agents_to_communication()
        observations = {}

        for agent_id in self.agent_ids:
            try:
                raw_observation = self.env.get_observation(agent_id)
                observations[agent_id] = self._process_observation(
                    raw_observation, agent_id
                )
            except Exception as error:
                print(
                    f"[OBSERVATION ERROR] {agent_id}: "
                    f"{type(error).__name__}: {error}"
                )
                result_observation = getattr(results, "observation", None)
                print(
                    "[OBSERVATION DEBUG] "
                    f"results_type={type(results).__name__}, "
                    f"observation_type={type(result_observation).__name__}, "
                    "observation_keys="
                    f"{list(result_observation.keys())[:30] if isinstance(result_observation, dict) else 'not a dictionary'}"
                )
                observations[agent_id] = np.zeros(
                    self.observation_dim, dtype=np.float32
                )
                self.xai_context_by_agent[agent_id] = {}
                self.feature_names_by_agent[agent_id] = [
                    f"missing_observation[{i}]"
                    for i in range(self.observation_dim)
                ]

        return observations

    # -------------------------------------------------------------
    # Agent information
    # -------------------------------------------------------------

    def get_agent_ids(self):
        return self.agent_ids

    def get_num_agents(self):
        return self.num_agents

    def get_observation_dim(self):
        return self.observation_dim

    def get_action_dim(self):
        return self.action_dim

    # -------------------------------------------------------------
    # Action progress and masks
    # -------------------------------------------------------------

    def is_action_in_progress(self, agent_id):
        """Return whether CybORG has an action in progress for this agent."""
        try:
            progress = (
                self.env.environment_controller.actions_in_progress.get(agent_id)
            )
            return progress is not None
        except (AttributeError, TypeError):
            return False

    def get_in_progress_action_name(self, agent_id):
        """Return the class name of an action currently in progress."""
        try:
            progress = (
                self.env.environment_controller.actions_in_progress.get(agent_id)
            )
            if progress is None:
                return ""
            action = progress.get("action") if isinstance(progress, dict) else None
            return type(action).__name__ if action is not None else "ACTION_IN_PROGRESS"
        except (AttributeError, TypeError):
            return ""

    def get_action_mask(self, agent_id):
        """Return a mask of currently available high-level actions."""
        mask = np.zeros(self.action_dim, dtype=np.float32)
        try:
            action_space = self.env.get_action_space(agent_id)
            available_actions = action_space.get("action", {})
            for index, action_name in enumerate(self.ACTION_NAMES):
                for action_class in available_actions.keys():
                    if getattr(action_class, "__name__", "") == action_name:
                        mask[index] = 1.0
                        break
        except Exception as error:
            print(
                f"Warning: Could not generate action mask for {agent_id}: {error}"
            )

        # Keep Sleep as a safe fallback action.
        mask[7] = 1.0
        return mask

    def _find_action_class(self, agent_id, action_name):
        """Find a CybORG action class in the agent's current action space."""
        try:
            action_space = self.env.get_action_space(agent_id)
            for action_class in action_space.get("action", {}).keys():
                if getattr(action_class, "__name__", "") == action_name:
                    return action_class
        except Exception:
            pass
        return None

    # -------------------------------------------------------------
    # Action parameter selection
    # -------------------------------------------------------------

    @staticmethod
    def _get_parameter_value(parameter_values):
        """Select the first enabled parameter value."""
        if not isinstance(parameter_values, dict):
            return None
        for value, enabled in parameter_values.items():
            if enabled:
                return value
        return None

    def _build_action_parameters(self, agent_id, action_class):
        """Build required parameters using the current CybORG action space."""
        parameters = {}

        try:
            action_space = self.env.get_action_space(agent_id)
        except Exception:
            return parameters

        try:
            signature = inspect.signature(action_class)
        except (TypeError, ValueError):
            return parameters

        def first_valid(space):
            if not isinstance(space, dict):
                return None
            for value, enabled in space.items():
                if enabled:
                    return value
            return None

        session = first_valid(action_space.get("session", {}))
        agent = first_valid(action_space.get("agent", {}))
        subnet = first_valid(action_space.get("subnet", {}))
        hostname = first_valid(action_space.get("hostname", {}))

        if action_class.__name__ in ("AllowTrafficZone", "BlockTrafficZone"):
            parameters["session"] = session if session is not None else 0
            parameters["agent"] = agent if agent is not None else agent_id
            if subnet is not None:
                parameters["from_subnet"] = subnet
                parameters["to_subnet"] = subnet
            return parameters

        for parameter_name, parameter in signature.parameters.items():
            if parameter_name == "self":
                continue
            if parameter.default is not inspect.Parameter.empty:
                continue

            if parameter_name == "session":
                parameters["session"] = session if session is not None else 0
            elif parameter_name == "agent":
                parameters["agent"] = agent if agent is not None else agent_id
            elif parameter_name == "hostname" and hostname is not None:
                parameters["hostname"] = hostname

        return parameters

    def _create_cyborg_action(self, agent_id, action_index):
        """Convert a discrete MAPPO action index into a CybORG action object."""
        action_index = int(action_index)
        action_index = max(0, min(action_index, self.action_dim - 1))
        action_name = self.ACTION_NAMES[action_index]
        action_class = self._find_action_class(agent_id, action_name)

        if action_class is None:
            print(
                f"[ACTION FALLBACK] {agent_id}: {action_name} unavailable; "
                "using Sleep"
            )
            sleep_class = self._find_action_class(agent_id, "Sleep")
            if sleep_class is not None:
                try:
                    return sleep_class()
                except Exception:
                    pass
            return None

        action_parameters = self._build_action_parameters(
            agent_id, action_class
        )
        try:
            return action_class(**action_parameters)
        except Exception as error:
            print(
                f"[ACTION FALLBACK] {agent_id}: could not construct "
                f"{action_name} ({error}); using Sleep"
            )
            sleep_class = self._find_action_class(agent_id, "Sleep")
            if sleep_class is not None:
                try:
                    return sleep_class()
                except Exception:
                    pass
            return None

    # -------------------------------------------------------------
    # Reward processing
    # -------------------------------------------------------------

    def _process_reward(self, reward):
        """Convert a CybORG reward into one float."""
        if isinstance(reward, (int, float, np.integer, np.floating)):
            return float(reward)

        if isinstance(reward, dict):
            total = 0.0
            for value in reward.values():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    total += float(value)
            return total

        return 0.0

    # -------------------------------------------------------------
    # Step
    # -------------------------------------------------------------

    def step(self, actions):
        """
        Execute one multi-agent step.

        Returns:
            observations, rewards, terminated, infos
        """
        cyborg_actions = {}
        for agent_id in self.agent_ids:
            action_index = actions.get(agent_id, 7)
            action = self._create_cyborg_action(agent_id, action_index)
            if action is not None:
                cyborg_actions[agent_id] = action

        result = self.env.parallel_step(actions=cyborg_actions)

        # Parse the documented tuple shapes without using uninitialized locals.
        if not isinstance(result, tuple):
            raise TypeError(
                "Expected CybORG parallel_step() to return a tuple, "
                f"received {type(result).__name__}"
            )

        if len(result) == 4:
            raw_observations, raw_rewards, dones, infos = result
        elif len(result) == 3:
            raw_observations, raw_rewards, dones = result
            infos = {}
        else:
            raise ValueError(
                f"Unexpected CybORG parallel_step() tuple length: {len(result)}"
            )

        if not isinstance(raw_observations, dict):
            raise TypeError(
                "CybORG observations should be a dictionary, received "
                f"{type(raw_observations).__name__}"
            )

        # Diagnostic: show what parallel_step actually returned.
        # Remove this block after the observation investigation is complete.
        if getattr(self, "debug_observations", True):
            print("\n========== PARALLEL_STEP OBSERVATION CHECK ==========")
            for agent_id in self.agent_ids:
                raw_obs = raw_observations.get(agent_id, {})
                if not isinstance(raw_obs, dict):
                    print(
                        f"\nAgent: {agent_id}; unexpected observation type: "
                        f"{type(raw_obs).__name__}"
                    )
                    continue

                excluded_keys = {"success", "action", "message", "msg"}
                host_keys = [
                    key for key, value in raw_obs.items()
                    if key not in excluded_keys and isinstance(value, dict)
                ]
                print(f"\nAgent: {agent_id}")
                print("Raw observation keys:", list(raw_obs.keys()))
                print("Host record count:", len(host_keys))
                print("Host names:", host_keys[:5])
                print(
                    "Action in progress:",
                    self.is_action_in_progress(agent_id)
                )
            print("=====================================================\n")

        observations = {}
        for agent_id in self.agent_ids:
            if agent_id in raw_observations:
                observations[agent_id] = self._process_observation(
                    raw_observations[agent_id], agent_id
                )
            else:
                observations[agent_id] = np.zeros(
                    self.observation_dim, dtype=np.float32
                )
                self.xai_context_by_agent[agent_id] = {}
                self.feature_names_by_agent[agent_id] = [
                    f"missing_observation[{i}]"
                    for i in range(self.observation_dim)
                ]

        rewards = {}
        for agent_id in self.agent_ids:
            if isinstance(raw_rewards, dict):
                rewards[agent_id] = self._process_reward(
                    raw_rewards.get(agent_id, 0.0)
                )
            else:
                rewards[agent_id] = self._process_reward(raw_rewards)

        if isinstance(dones, dict):
            terminated = any(bool(value) for value in dones.values())
        else:
            terminated = bool(dones)

        return observations, rewards, terminated, infos

    # -------------------------------------------------------------
    # Close
    # -------------------------------------------------------------

    def close(self, **kwargs):
        """Close hook; this CybORG adapter has no GUI resources to release."""
        return None


if __name__ == "__main__":
    print("===== CybORG MAPPO ENV TEST =====")
    env = CybORGMAPPOEnv(steps=10)

    print("Agents:", env.get_agent_ids())
    print("Number of agents:", env.get_num_agents())
    print("Observation dimension:", env.get_observation_dim())
    print("Action dimension:", env.get_action_dim())

    print("\nAction masks:")
    for agent_id in env.get_agent_ids():
        print(agent_id, env.get_action_mask(agent_id))

    observations = env.reset()
    print("\nObservation shapes:")
    for agent_id in env.get_agent_ids():
        print(agent_id, observations[agent_id].shape)

    print("\n===== ENV TEST PASSED =====")
