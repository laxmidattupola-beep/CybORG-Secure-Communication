from CybORG.communication.communication_layer import CommunicationLayer


def test_tampered_message_is_rejected():

    communication = CommunicationLayer()

    communication.register_agents([
        "agent_a",
        "agent_b"
    ])

    message = communication.send_message(
        sender="agent_a",
        receiver="agent_b",
        message_type="ALERT",
        payload="Host compromised"
    )

    # Attacker modifies the encrypted message
    message.payload = "Tampered message"

    communication.message_queues["agent_b"].append(message)

    received = communication.receive_messages("agent_b")

    assert received == []