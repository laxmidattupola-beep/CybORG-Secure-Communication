from CybORG import CybORG
from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import (
    EnterpriseScenarioGenerator
)
from CybORG.Agents.SimpleAgents.RandomAgent import cc4BlueRandomAgent


def create_cyborg():
    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=cc4BlueRandomAgent,
        steps=10
    )

    return CybORG(scenario)


def test_cyborg_agents_have_communication():
    cyborg = create_cyborg()

    blue_agents = [
        agent for agent in cyborg.agents
        if agent.startswith("blue")
    ]

    assert len(blue_agents) >= 2

    agent_0 = cyborg.environment_controller.agent_interfaces[
        blue_agents[0]
    ].agent

    agent_1 = cyborg.environment_controller.agent_interfaces[
        blue_agents[1]
    ].agent

    assert agent_0.communication_layer is cyborg.communication_layer
    assert agent_1.communication_layer is cyborg.communication_layer


def test_cyborg_agents_can_communicate():
    cyborg = create_cyborg()

    blue_agents = [
        agent for agent in cyborg.agents
        if agent.startswith("blue")
    ]

    assert len(blue_agents) >= 2

    sender_name = blue_agents[0]
    receiver_name = blue_agents[1]

    sender = cyborg.environment_controller.agent_interfaces[
        sender_name
    ].agent

    receiver = cyborg.environment_controller.agent_interfaces[
        receiver_name
    ].agent

    sender.send_message(
        receiver=receiver_name,
        message_type="ALERT",
        payload="Host compromised"
    )

    messages = receiver.receive_messages()

    assert len(messages) == 1

    message = messages[0]

    assert message.sender == sender_name
    assert message.receiver == receiver_name
    assert message.message_type == "ALERT"
    assert message.payload == "Host compromised"


def test_messages_are_cleared_on_reset():
    cyborg = create_cyborg()

    blue_agents = [
        agent for agent in cyborg.agents
        if agent.startswith("blue")
    ]

    assert len(blue_agents) >= 2

    sender_name = blue_agents[0]
    receiver_name = blue_agents[1]

    sender = cyborg.environment_controller.agent_interfaces[
        sender_name
    ].agent

    receiver = cyborg.environment_controller.agent_interfaces[
        receiver_name
    ].agent

    sender.send_message(
        receiver=receiver_name,
        message_type="INFO",
        payload="Test message"
    )

    assert len(receiver.peek_messages()) == 1

    cyborg.reset()

    receiver = cyborg.environment_controller.agent_interfaces[
        receiver_name
    ].agent

    assert receiver.receive_messages() == []

def test_agents_remain_connected_after_reset():
    cyborg = create_cyborg()

    blue_agents = [
        agent for agent in cyborg.agents
        if agent.startswith("blue")
    ]

    assert len(blue_agents) >= 2

    agent_name = blue_agents[0]

    agent_before_reset = cyborg.environment_controller.agent_interfaces[
        agent_name
    ].agent

    assert agent_before_reset.communication_layer is cyborg.communication_layer

    cyborg.reset()

    agent_after_reset = cyborg.environment_controller.agent_interfaces[
        agent_name
    ].agent

    assert agent_after_reset.communication_layer is cyborg.communication_layer