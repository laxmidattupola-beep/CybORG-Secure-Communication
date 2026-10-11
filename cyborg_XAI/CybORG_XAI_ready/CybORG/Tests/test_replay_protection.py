from CybORG.communication.communication_layer import CommunicationLayer


def test_replay_attack_is_rejected():

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

    # First delivery should succeed
    received = communication.receive_messages("agent_b")

    assert len(received) == 1
    assert received[0].payload == "Host compromised"

    # Replay the exact same message
    communication.message_queues["agent_b"].append(message)

    received_again = communication.receive_messages("agent_b")

    # Replay must be rejected
    assert received_again == []