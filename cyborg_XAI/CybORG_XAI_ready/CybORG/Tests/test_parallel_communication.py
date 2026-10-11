from CybORG import CybORG
from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import (
    EnterpriseScenarioGenerator
)


def test_parallel_step_communication():

    cyborg = CybORG(
        EnterpriseScenarioGenerator(steps=10)
    )

    cyborg.reset()

    sender = "blue_agent_0"
    receiver = "blue_agent_1"

    message = [1, 0, 1, 0, 0, 1, 0, 1]

    cyborg.parallel_step(
        actions={},
        messages={
            sender: message
        }
    )

    received = cyborg.communication_layer.receive_messages(
        receiver
    )

    assert len(received) == 1
    assert received[0].sender == sender
    assert received[0].receiver == receiver
    assert received[0].message_type == "INFO"
    assert received[0].payload == message


def test_parallel_message_is_authenticated():

    cyborg = CybORG(
        EnterpriseScenarioGenerator(steps=10)
    )

    cyborg.reset()

    sender = "blue_agent_0"
    receiver = "blue_agent_1"

    cyborg.parallel_step(
        actions={},
        messages={
            sender: "Secure message"
        }
    )

    queued = cyborg.communication_layer.peek_messages(receiver)

    assert len(queued) == 1

    message = queued[0]

    assert message.authentication_tag != ""
    assert message.verify(
        cyborg.communication_layer.authentication_key
    )

    received = cyborg.communication_layer.receive_messages(receiver)

    assert received[0].payload == "Secure message"


def test_parallel_message_is_encrypted():

    cyborg = CybORG(
        EnterpriseScenarioGenerator(steps=10)
    )

    cyborg.reset()

    sender = "blue_agent_0"
    receiver = "blue_agent_1"
    message = "Secret communication"

    cyborg.parallel_step(
        actions={},
        messages={
            sender: message
        }
    )

    queued = cyborg.communication_layer.peek_messages(receiver)

    assert len(queued) == 1
    assert queued[0].payload != message

    received = cyborg.communication_layer.receive_messages(receiver)

    assert received[0].payload == message