
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

# Secure multi-agent communication
from CybORG.communication.communication_layer import (
    MultiAgentCommunicationNetwork
)

from CybORG.Shared import Observation, Results, CybORGLogger
from CybORG.Shared.Enums import DecoyType
from CybORG.Shared.Scenarios.ScenarioGenerator import ScenarioGenerator

from CybORG.Simulator.Actions import (
    DiscoverNetworkServices,
    DiscoverRemoteSystems,
    ExploitRemoteService,
    InvalidAction,
    Sleep,
    PrivilegeEscalate,
    Impact,
    Remove,
    Restore,
    RemoveOtherSessions
)

from CybORG.Simulator.Actions.ConcreteActions.ActivateTrojan import (
    ActivateTrojan
)

from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import (
    BlockTraffic,
    AllowTraffic
)

from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.ExploitAction import (
    ExploitAction
)

from CybORG.Tests.utils import CustomGenerator


class CybORG(CybORGLogger):

    supported_envs = ['sim']

    # ========================================================
    # CONNECT AGENTS TO COMMUNICATION
    # ========================================================

    def _connect_agents_to_communication(self):
        """
        Connect all CybORG agents to the secure
        multi-agent communication network.
        """

        # Get the current agent IDs after reset
        agent_ids = list(
            self.environment_controller.agent_interfaces.keys()
        )

        # Create a fresh communication network
        self.communication_layer = MultiAgentCommunicationNetwork(
            agent_ids
        )

        # Connect every CybORG agent to the network
        for agent_interface in (
            self.environment_controller.agent_interfaces.values()
        ):
            agent_interface.agent.set_communication_layer(
                self.communication_layer
            )

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        scenario_generator: ScenarioGenerator,
        agents: dict = None,
        seed: Union[int, CustomGenerator] = None
    ):
        """
        Instantiates the CybORG class.
        """

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

        # ----------------------------------------------------
        # RANDOM SEED
        # ----------------------------------------------------

        if seed is None or isinstance(seed, int):
            self.np_random, seed = seeding.np_random(seed)
        else:
            self.np_random = seed

        # ----------------------------------------------------
        # ENVIRONMENT CONTROLLER
        # ----------------------------------------------------

        self.environment_controller = SimulationController(
            self.scenario_generator,
            agents,
            self.np_random
        )

        # ====================================================
        # SECURE MULTI-AGENT COMMUNICATION
        # ====================================================

        # Get agent IDs created by CybORG
        agent_ids = list(
            self.environment_controller.agent_interfaces.keys()
        )

        # Create secure communication network
        self.communication_layer = MultiAgentCommunicationNetwork(
            agent_ids
        )

        # Connect all agents to communication network
        for agent_interface in (
            self.environment_controller.agent_interfaces.values()
        ):
            agent_interface.agent.set_communication_layer(
                self.communication_layer
            )

        # ====================================================
        # GUI
        # ====================================================

        # CC4 GUI is disabled.
        # No GUI attributes are required.

    # ========================================================
    # PARALLEL STEP
    # ========================================================

    def parallel_step(
        self,
        actions: dict = None,
        messages: dict = None,
        skip_valid_action_check: bool = False
    ) -> Tuple[dict, dict, dict, dict]:
        """
        Performs a step in CybORG accepting multiple
        external agent inputs.
        """

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

        # Execute environment actions
        self.environment_controller.step(
            actions,
            skip_valid_action_check
        )

        # ----------------------------------------------------
        # Secure communication
        # ----------------------------------------------------

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
                agent:
                self.environment_controller.get_reward(agent)
                for agent in agents
            },

            {
                agent:
                self.environment_controller.done
                for agent in agents
            },

            {}
        )

    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        agent: str = None,
        action=None,
        messages: dict = None,
        skip_valid_action_check: bool = False
    ) -> Results:
        """
        Performs a step in CybORG for the given agent.
        """

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

    # ========================================================
    # START
    # ========================================================

    def start(
        self,
        steps: int,
        log_file=None,
        verbose=False
    ) -> bool:

        return self.environment_controller.start(
            steps,
            log_file,
            verbose
        )

    # ========================================================
    # TRUE STATE
    # ========================================================

    def get_true_state(
        self,
        info: dict
    ) -> dict:

        return self.environment_controller.get_true_state(
            info
        ).data

    # ========================================================
    # AGENT STATE
    # ========================================================

    def get_agent_state(
        self,
        agent_name
    ) -> dict:

        return self.environment_controller.get_agent_state(
            agent_name
        ).data

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        agent: str = None,
        seed: int = None
    ) -> Results:
        """
        Resets CybORG and reconnects all agents to
        the secure communication network.
        """

        # ----------------------------------------------------
        # Seed
        # ----------------------------------------------------

        if seed is not None:

            self.np_random, seed = seeding.np_random(
                seed
            )

        # ----------------------------------------------------
        # Reset simulation
        # ----------------------------------------------------

        self.environment_controller.reset(
            np_random=self.np_random
        )

        # ----------------------------------------------------
        # Reconnect secure communication
        # ----------------------------------------------------

        self._connect_agents_to_communication()

        # ----------------------------------------------------
        # No specific agent
        # ----------------------------------------------------

        if agent is None:

            return Results(
                observation=self.environment_controller.init_state
            )

        # ----------------------------------------------------
        # Get observation
        # ----------------------------------------------------

        obs = (
            self.environment_controller
            .observation[agent]
            .get_combined_observation()
            .data
        )

        # ----------------------------------------------------
        # Get action space
        # ----------------------------------------------------

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

    # ========================================================
    # OBSERVATION
    # ========================================================

    def get_observation(
        self,
        agent: str
    ) -> dict:

        return (
            self.environment_controller
            .get_last_observation(agent)
            .data
        )

    # ========================================================
    # ACTION SPACE
    # ========================================================

    def get_action_space(
        self,
        agent: str
    ):

        return self.environment_controller.get_action_space(
            agent
        )

    # ========================================================
    # OBSERVATION SPACE
    # ========================================================

    def get_observation_space(
        self,
        agent: str
    ):

        return self.environment_controller.get_observation_space(
            agent
        )

    # ========================================================
    # LAST ACTION
    # ========================================================

    def get_last_action(
        self,
        agent: str
    ):

        return self.environment_controller.get_last_action(
            agent
        )

    # ========================================================
    # SEED
    # ========================================================

    def set_seed(
        self,
        seed: int
    ):

        self.np_random, seed = seeding.np_random(
            seed
        )

        self.environment_controller.set_np_random(
            self.np_random
        )

    # ========================================================
    # IP MAP
    # ========================================================

    def get_ip_map(self):

        return self.environment_controller.hostname_ip_map

    # ========================================================
    # CIDR MAP
    # ========================================================

    def get_cidr_map(self):

        return self.environment_controller.subnet_cidr_map

    # ========================================================
    # REWARDS
    # ========================================================

    def get_rewards(self):

        return self.environment_controller.reward

    # ========================================================
    # REWARD BREAKDOWN
    # ========================================================

    def get_reward_breakdown(
        self,
        agent: str
    ):

        return self.environment_controller.get_reward_breakdown(
            agent
        )

    # ========================================================
    # GET ATTRIBUTE
    # ========================================================

    def get_attr(
        self,
        attribute: str
    ) -> Any:

        if hasattr(self, attribute):
            return self.__getattribute__(
                attribute
            )

        return None

    # ========================================================
    # AGENTS
    # ========================================================

    @property
    def agents(self) -> list:

        return [
            agent_name
            for agent_name, agent_info
            in self.environment_controller
            .agent_interfaces.items()
            if not agent_info.internal_only
        ]

    # ========================================================
    # ACTIVE AGENTS
    # ========================================================

    @property
    def active_agents(self) -> list:

        return self.environment_controller.get_active_agents()

    # ========================================================
    # AGENT IDS
    # ========================================================

    def get_agent_ids(self):

        return list(
            self.environment_controller
            .agent_interfaces.keys()
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self, **kwargs):
        """
        Shuts down CybORG.

        GUI is disabled for CC4, so there are no GUI
        resources to close.
        """

        return

    # ========================================================
    # UNWRAPPED
    # ========================================================

    @property
    def unwrapped(self):

        return self

    # ========================================================
    # MESSAGE SPACE
    # ========================================================

    def get_message_space(
        self,
        agent: str
    ) -> gym.Space:

        return self.environment_controller.get_message_space(
            agent
        )

    # ========================================================
    # GUI FUNCTIONS
    # ========================================================

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

