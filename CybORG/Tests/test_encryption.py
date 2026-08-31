from cryptography.fernet import Fernet, InvalidToken
from CybORG.communication.message import Message


def test_message_encryption():

    key = Fernet.generate_key()

    message = Message(
        sender="agent_a",
        receiver="agent_b",
        message_type="ALERT",
        payload="Host compromised"
    )

    original = message.payload

    message.encrypt(key)

    assert message.payload != original

    message.decrypt(key)

    assert message.payload == original


def test_wrong_key_cannot_decrypt():

    correct_key = Fernet.generate_key()
    wrong_key = Fernet.generate_key()

    message = Message(
        sender="agent_a",
        receiver="agent_b",
        message_type="ALERT",
        payload="Secret data"
    )

    message.encrypt(correct_key)

    try:
        message.decrypt(wrong_key)
        assert False
    except InvalidToken:
        assert True