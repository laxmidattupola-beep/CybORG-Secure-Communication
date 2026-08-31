from CybORG.communication.message import Message


def test_message_authentication():

    key = b"secret-key"

    message = Message(
        sender="agent_a",
        receiver="agent_b",
        message_type="ALERT",
        payload="Host compromised"
    )

    message.authenticate(key)

    assert message.authentication_tag != ""
    assert message.verify(key)


def test_tampered_message_is_detected():

    key = b"secret-key"

    message = Message(
        sender="agent_a",
        receiver="agent_b",
        message_type="ALERT",
        payload="Host compromised"
    )

    message.authenticate(key)

    message.payload = "Tampered message"

    assert not message.verify(key)


def test_wrong_key_is_rejected():

    message = Message(
        sender="agent_a",
        receiver="agent_b",
        message_type="ALERT",
        payload="Host compromised"
    )

    message.authenticate(b"correct-key")

    assert not message.verify(b"wrong-key")