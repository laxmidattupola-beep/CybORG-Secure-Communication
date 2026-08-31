from CybORG.communication.communication_layer import CommunicationLayer


def main():

    print("=" * 60)
    print("       CybORG SECURE COMMUNICATION DEMONSTRATION")
    print("=" * 60)

    communication = CommunicationLayer()

    communication.register_agents([
        "blue_agent_0",
        "blue_agent_1"
    ])

    print("\n[1] AGENTS")
    print("-" * 60)
    print("Sender  :", "blue_agent_0")
    print("Receiver:", "blue_agent_1")

    # ---------------------------------------------------------
    # Send secure message
    # ---------------------------------------------------------

    print("\n[2] SENDING MESSAGE")
    print("-" * 60)

    original_payload = "Host compromised"

    message = communication.send_message(
        sender="blue_agent_0",
        receiver="blue_agent_1",
        message_type="ALERT",
        payload=original_payload
    )

    print("Message ID       :", message.message_id)
    print("Message Type     :", message.message_type)
    print("Original Payload :", original_payload)
    print("Encrypted Payload:", message.payload)
    print("Authentication   :", message.authentication_tag)

    # ---------------------------------------------------------
    # Authentication
    # ---------------------------------------------------------

    print("\n[3] AUTHENTICATION")
    print("-" * 60)

    authenticated = message.verify(
        communication.authentication_key
    )

    print("HMAC verification:", authenticated)

    # ---------------------------------------------------------
    # Receive and decrypt
    # ---------------------------------------------------------

    print("\n[4] RECEIVING MESSAGE")
    print("-" * 60)

    received = communication.receive_messages(
        "blue_agent_1"
    )

    print("Messages received:", len(received))

    if received:
        print("Sender           :", received[0].sender)
        print("Receiver         :", received[0].receiver)
        print("Message Type     :", received[0].message_type)
        print("Decrypted Payload:", received[0].payload)

    # ---------------------------------------------------------
    # Tampering attack
    # ---------------------------------------------------------

    print("\n[5] TAMPERING ATTACK")
    print("-" * 60)

    tampered = communication.send_message(
        sender="blue_agent_0",
        receiver="blue_agent_1",
        message_type="ALERT",
        payload="Original message"
    )

    tampered.payload = "ATTACKER MODIFIED THIS"

    valid = tampered.verify(
        communication.authentication_key
    )

    print("Message modified by attacker.")
    print("Authentication valid:", valid)

    if not valid:
        print("RESULT: TAMPERING DETECTED")

    # ---------------------------------------------------------
    # Replay attack
    # ---------------------------------------------------------

    print("\n[6] REPLAY ATTACK")
    print("-" * 60)

    replay = communication.send_message(
        sender="blue_agent_0",
        receiver="blue_agent_1",
        message_type="INFO",
        payload="One-time message"
    )

    first_delivery = communication.receive_messages(
        "blue_agent_1"
    )

    communication.message_queues["blue_agent_1"].append(
        replay
    )

    second_delivery = communication.receive_messages(
        "blue_agent_1"
    )

    print("First delivery :", len(first_delivery), "message")
    print("Replay delivery:", len(second_delivery), "messages")

    if len(second_delivery) == 0:
        print("RESULT: REPLAY ATTACK BLOCKED")

    print("\n" + "=" * 60)
    print("       SECURE COMMUNICATION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()