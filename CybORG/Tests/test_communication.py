from CybORG.communication.communication_layer import CommunicationLayer
from CybORG.communication.message import Message


def test_structured_message():
    communication = CommunicationLayer()

    communication.register_agents([
        "blue_agent_0",
        "blue_agent_1"
    ])

    message = communication.send_message(
        sender="blue_agent_0",
        receiver="blue_agent_1",
        message_type="ALERT",
        payload="Host compromised"
    )

    assert isinstance(message, Message)

    assert message.sender == "blue_agent_0"
    assert message.receiver == "blue_agent_1"
    assert message.message_type == "ALERT"
    assert message.payload != "Host compromised"
    received = communication.receive_messages("blue_agent_1")

    assert received[0].payload == "Host compromised"

    assert message.message_id is not None
    assert message.timestamp is not None


def test_message_delivery():
    communication = CommunicationLayer()

    communication.register_agents([
        "blue_agent_0",
        "blue_agent_1"
    ])

    communication.send_message(
        sender="blue_agent_0",
        receiver="blue_agent_1",
        message_type="INFO",
        payload="Scanning host"
    )

    messages = communication.receive_messages("blue_agent_1")

    assert len(messages) == 1

    message = messages[0]

    assert message.sender == "blue_agent_0"
    assert message.receiver == "blue_agent_1"
    assert message.message_type == "INFO"
    assert message.payload == "Scanning host"


def test_message_ids_are_unique():
    communication = CommunicationLayer()

    communication.register_agents([
        "blue_agent_0",
        "blue_agent_1"
    ])

    message1 = communication.send_message(
        sender="blue_agent_0",
        receiver="blue_agent_1",
        message_type="INFO",
        payload="Message 1"
    )

    message2 = communication.send_message(
        sender="blue_agent_0",
        receiver="blue_agent_1",
        message_type="INFO",
        payload="Message 2"
    )

    assert message1.message_id != message2.message_id