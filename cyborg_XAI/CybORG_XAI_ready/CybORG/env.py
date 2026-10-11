# Copyright DST Group. Licensed under the MIT license.
import queue
from threading import Thread, Event
import warnings
from typing import Any, Tuple, Union

import gym
import numpy as np
import pygame
from gym.utils import seeding

from CybORG.Simulator.SimulationController import SimulationController
from CybORG.communication.communication_layer import MultiAgentCommunicationNetwork
from CybORG.Shared import Observation, Results, CybORGLogger
from CybORG.Shared.Enums import DecoyType
from CybORG.Shared.Scenarios.ScenarioGenerator import ScenarioGenerator
from CybORG.Simulator.Actions import DiscoverNetworkServices, DiscoverRemoteSystems, ExploitRemoteService, \
    InvalidAction, \
    Sleep, PrivilegeEscalate, Impact, Remove, Restore, RemoveOtherSessions
from CybORG.Simulator.Actions.ConcreteActions.ActivateTrojan import ActivateTrojan
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import BlockTraffic, AllowTraffic
from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.ExploitAction import ExploitAction
from CybORG.Tests.utils import CustomGenerator


class CybORG(CybORGLogger):
    """The main interface for the Cyber Operations Research Gym.

    The primary purpose of this class is to provide a unified interface for the Cyber Operations Research Gym.

    Attributes
    ----------
    scenario_generator: ScenarioGenerator
        ScenarioGenerator object that creates scenarios.
    environment: str
        The environment to use.
    env_config: dict
        Configuration keyword arguments for environment controller.
    agents: dict
        Defines the agent that selects the default action.
    """

    supported_envs = ['sim']

    def _connect_agents_to_communication(self):
        """
        Connect all CybORG agents to the secure communication network.
        """

        for agent_id, agent_interface in (
            self.environment_controller.agent_interfaces.items()
        ):

            communication_layer = (
                self.communication_layer.get_agent(agent_id)
            )

            if communication_layer is not None:
                agent_interface.agent.set_communication_layer(
                    communication_layer
                )

    def __init__(
        self,
        scenario_generator: ScenarioGenerator,
        agents: dict = None,
        seed: Union[int, CustomGenerator] = None
    ):
        """Instantiates the CybORG class."""

        assert issubclass(
            type(scenario_generator),
            ScenarioGenerator
        ), (
            f'Scenario generator object of type '
            f'{type(scenario_generator)} must be a subclass '
            f'of ScenarioGenerator'
        )

        self.scenario_generator = scenario_generator

        self._log_info(
            f"Using scenario generator "
            f"{str(scenario_generator)}"
        )

        if seed is None or isinstance(seed, int):
            self.np_random, seed = seeding.np_random(seed)
        else:
            self.np_random = seed

        self.environment_controller = SimulationController(
            self.scenario_generator,
            agents,
            self.np_random
        )

        # ---------------------------------------------------------
        # SECURE MULTI-AGENT COMMUNICATION NETWORK
        # ---------------------------------------------------------

        agent_ids = list(
            self.environment_controller.agent_interfaces.keys()
        )

        self.communication_layer = (
            MultiAgentCommunicationNetwork(agent_ids)
        )

        self._connect_agents_to_communication()

        # # CC4: GUI not implemented for CC4, disable by default
        # self._disable_gui = True

        # self.renderer: Renderer = None
        # self.gui: Thread = None
        # self.gui_actions_queue: queue.Queue = None

        # used to signal termination to the threads
        # self.stop_event: Event = None if self._disable_gui else Event()

    def parallel_step(
        self,
        actions: dict = None,
        messages: dict = None,
        skip_valid_action_check: bool = False
    ) -> Tuple[dict, dict, dict, dict]:
        """Performs a step in CybORG accepting multiple external agent inputs."""

        if actions is not None and len(actions) > 0:
            agents = list(actions.keys())
        else:
            agents = []

        if actions is self.environment_controller.action:
            warnings.warn(
                "Reuse of the actions input. This variable is altered "
                "inside the simulation and may contain actions from "
                "previous steps"
            )

        self.environment_controller.step(
            actions,
            skip_valid_action_check
        )

        if messages is not None:

            for sender, message in messages.items():

                receivers = (
                    self.environment_controller
                    .get_connected_agents(sender)
                )

                for receiver in receivers:

                    self.communication_layer.send_message(
                        sender=sender,
                        receiver=receiver,
                        message_type="INFO",
                        payload=message
                    )

        agents = set(
            agents + self.active_agents
        )

        return (
            {
                agent: self.get_observation(agent)
                for agent in agents
            },
            {
                agent: self.environment_controller.get_reward(agent)
                for agent in agents
            },
            {
                agent: self.environment_controller.done
                for agent in agents
            },
            {}
        )

    def step(
        self,
        agent: str = None,
        action=None,
        messages: dict = None,
        skip_valid_action_check: bool = False
    ) -> Results:
        """Performs a step in CybORG for the given agent."""

        if action is None or agent is None:
            action = {}
        else:
            action = {
                agent: action
            }

        self.environment_controller.step(
            action,
            skip_valid_action_check
        )

        self.environment_controller.send_messages(
            messages
        )

        if agent is None:

            result = Results(
                observation=Observation().data
            )

        else:

            reward = round(
                sum(
                    self.environment_controller
                    .get_reward(agent)
                    .values()
                ),
                1
            )

            action_space = (
                self.environment_controller
                .agent_interfaces[agent]
                .action_space
                .get_action_space()
            )

            result = Results(
                observation=self.get_observation(agent),
                done=self.environment_controller.done,
                reward=reward,
                action_space=action_space,
                action=self.environment_controller.action[agent]
            )

        return result

    def start(
        self,
        steps: int,
        log_file=None,
        verbose=False
    ) -> bool:
        """Start CybORG and run for a specified number of steps."""

        return self.environment_controller.start(
            steps,
            log_file,
            verbose
        )

    def get_true_state(self, info: dict) -> dict:
        """Creates a dictionary containing requested information."""

        return self.environment_controller.get_true_state(
            info
        ).data

    def get_agent_state(self, agent_name) -> dict:
        """Get the initial observation of the specified agent."""

        return self.environment_controller.get_agent_state(
            agent_name
        ).data

    def reset(
        self,
        agent: str = None,
        seed: int = None
    ) -> Results:
        """Resets CybORG and gets initial observation."""

        if seed is not None:
            self.np_random, seed = seeding.np_random(seed)

        self.environment_controller.reset(
            np_random=self.np_random
        )

        # Reconnect agents after reset
        self._connect_agents_to_communication()

        if agent is None:

            return Results(
                observation=self.environment_controller.init_state
            )

        obs = (
            self.environment_controller
            .observation[agent]
            .get_combined_observation()
            .data
        )

        action_space = (
            self.environment_controller
            .agent_interfaces[agent]
            .action_space
            .get_action_space()
        )

        return Results(
            observation=obs,
            action_space=action_space
        )

    def get_observation(self, agent: str) -> dict:
        """Get the last observation for an agent."""

        return (
            self.environment_controller
            .get_last_observation(agent)
            .data
        )

    def get_action_space(self, agent: str):
        """Returns the most recent action space."""

        return self.environment_controller.get_action_space(
            agent
        )

    def get_observation_space(self, agent: str):
        """Returns the most recent observation space."""

        return self.environment_controller.get_observation_space(
            agent
        )

    def get_last_action(self, agent: str):
        """Returns the last executed action."""

        return self.environment_controller.get_last_action(
            agent
        )

    def set_seed(self, seed: int):
        """Creates an np_random object to seed CybORG."""

        self.np_random, seed = seeding.np_random(
            seed
        )

        self.environment_controller.set_np_random(
            self.np_random
        )

    def get_ip_map(self):
        """Returns mapping of hostnames to IP addresses."""

        return self.environment_controller.hostname_ip_map

    def get_cidr_map(self):
        """Returns mapping of hostnames to subnet CIDRs."""

        return self.environment_controller.subnet_cidr_map

    def get_rewards(self):
        """Returns rewards for each agent."""

        return self.environment_controller.reward

    def get_reward_breakdown(self, agent: str):
        """Returns reward breakdown."""

        return self.environment_controller.get_reward_breakdown(
            agent
        )

    def get_attr(self, attribute: str) -> Any:
        """Returns the specified attribute if present."""

        if hasattr(self, attribute):
            return self.__getattribute__(
                attribute
            )

        return None

    @property
    def agents(self) -> list:
        """Returns all external-facing agents."""

        return [
            agent_name
            for agent_name, agent_info
            in self.environment_controller.agent_interfaces.items()
            if not agent_info.internal_only
        ]

    @property
    def active_agents(self) -> list:
        """Returns all active agents."""

        return (
            self.environment_controller
            .get_active_agents()
        )

    def get_agent_ids(self):
        """Returns IDs for the agents."""

        return list(
            self.environment_controller
            .agent_interfaces
            .keys()
        )

    def close(self, **kwargs):
        """Shuts down CybORG."""

        if not self._disable_gui:
            self.gui_actions_queue.put(
                'shutdown'
            )

    @property
    def unwrapped(self):
        """Returns CybORG without wrappers."""

        return self

    def get_message_space(
        self,
        agent: str
    ) -> gym.Space:
        """Returns Gym space containing possible messages."""

        return (
            self.environment_controller
            .get_message_space(agent)
        )

    # ---------------------------------------------------------
    # GUI functions removed in CC4 trim
    # ---------------------------------------------------------

    def get_renderer(self):
        pass

    def gui_thread(self):
        pass

    def stop_gui(self):
        pass

    def render(self, mode='human'):
        pass

    def add_red_actions(self, data):
        pass

    def add_blue_actions(self, data, red_action):
        pass

    def get_render_data(self):
        pass

    def add_rewards(self, data):
        pass

    def update_symbols(
        self,
        data,
        red_hosts,
        red_low_hosts,
        blue_hosts,
        blue_protected_hosts
    ):
        pass