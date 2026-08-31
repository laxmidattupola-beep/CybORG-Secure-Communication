"""
CybORG Communication + Trust Demonstration

Demonstrates:
1. Two blue agents
2. Structured communication
3. 8-bit message encoding/decoding
4. Message authentication
5. Trust evaluation
6. Trust increase after reliable communication
7. Trust decrease after unreliable behaviour
8. Trust-based message weighting
"""

from communication.communication_layer import CommunicationLayer
from trust.trust_manager import TrustManager


def separator():
    print("\n" + "=" * 70)


def main():

    separator()
    print("CYBORG COMMUNICATION + TRUST DEMONSTRATION")
    separator()

    # ==============================================================
    # 1. CREATE TRUST MANAGER
    # ==============================================================

    trust_manager = TrustManager()

    print("\n[1] TRUST MANAGER")
    print("-" * 70)

    print("Default trust:",
          trust_manager.get_trust("BlueAgent1"))

    # ==============================================================
    # 2. CREATE TWO BLUE AGENTS
    # ==============================================================

    agent1 = CommunicationLayer(
        agent_id="BlueAgent1",
        agent_ids=[
            "BlueAgent1",
            "BlueAgent2"
        ],
        trust_manager=trust_manager
    )

    agent2 = CommunicationLayer(
        agent_id="BlueAgent2",
        agent_ids=[
            "BlueAgent1",
            "BlueAgent2"
        ],
        trust_manager=trust_manager
    )

    print("\n[2] BLUE AGENTS")
    print("-" * 70)

    print("Agent 1:", agent1.agent_id)
    print("Agent 2:", agent2.agent_id)

    print(
        "Initial trust of BlueAgent1:",
        agent2.get_sender_trust("BlueAgent1")
    )

    # ==============================================================
    # 3. 8-BIT COMMUNICATION
    # ==============================================================

    separator()
    print("[3] 8-BIT COMMUNICATION")
    print("-" * 70)

    value = 37

    encoded = CommunicationLayer.encode_8bit(value)
    decoded = CommunicationLayer.decode_8bit(encoded)

    print("Original value :", value)
    print("Encoded 8-bit  :", encoded)
    print("Decoded value  :", decoded)

    # ==============================================================
    # 4. SEND STRUCTURED MESSAGE
    # ==============================================================

    separator()
    print("[4] BLUE AGENT 1 → BLUE AGENT 2")
    print("-" * 70)

    message = agent1.send_message(
        receiver="BlueAgent2",
        message_type="THREAT_ALERT",
        payload={
            "host": "Server_1",
            "threat": "suspicious_activity",
            "severity": 3,
            "zone": "Operational_A"
        }
    )

    print("Message ID   :", message.message_id)
    print("Sender       :", message.sender)
    print("Receiver     :", message.receiver)
    print("Type         :", message.message_type)
    print("Payload      :", message.payload)
    print("Sequence     :", message.sequence_number)
    print("Signature    :", message.signature)

    # ==============================================================
    # 5. RECEIVE MESSAGE
    # ==============================================================

    separator()
    print("[5] BLUE AGENT 2 RECEIVES MESSAGE")
    print("-" * 70)

    result = agent2.receive_message(message)

    if result is None:

        print("MESSAGE REJECTED")

    else:

        print("MESSAGE ACCEPTED")

        print("Sender       :", result["sender"])
        print("Message type :", result["message_type"])
        print("Payload      :", result["payload"])
        print("Trust        :", result["trust"])
        print("Trusted      :", result["trusted"])

    # ==============================================================
    # 6. TRUST AFTER VALID MESSAGE
    # ==============================================================

    separator()
    print("[6] TRUST AFTER VALID COMMUNICATION")
    print("-" * 70)

    trust = agent2.get_sender_trust(
        "BlueAgent1"
    )

    print("BlueAgent1 trust :", round(trust, 4))

    print(
        "Trust level      :",
        trust_manager.get_trust_level(
            "BlueAgent1"
        )
    )

    print(
        "Is trusted       :",
        trust_manager.is_trusted(
            "BlueAgent1"
        )
    )

    # ==============================================================
    # 7. TRUST-BASED MESSAGE WEIGHT
    # ==============================================================

    separator()
    print("[7] TRUST-BASED MESSAGE WEIGHT")
    print("-" * 70)

    weight = agent2.get_sender_trust(
        "BlueAgent1"
    )

    print(
        "Message influence weight:",
        round(weight, 4)
    )

    print(
        "\nThe trust value can be used as the influence "
        "of received information."
    )

    # ==============================================================
    # 8. SIMULATE RELIABLE BEHAVIOUR
    # ==============================================================

    separator()
    print("[8] RELIABLE COMMUNICATION")
    print("-" * 70)

    for i in range(3):

        trust = trust_manager.update_trust(
            "BlueAgent1",
            successful_communication=True,
            authentication=True,
            behavior=True
        )

        print(
            f"Reliable interaction {i + 1}: "
            f"trust = {trust:.4f}"
        )

    # ==============================================================
    # 9. SIMULATE UNRELIABLE BEHAVIOUR
    # ==============================================================

    separator()
    print("[9] UNRELIABLE BEHAVIOUR")
    print("-" * 70)

    for i in range(3):

        trust = trust_manager.update_trust(
            "BlueAgent1",
            successful_communication=False,
            authentication=True,
            behavior=False
        )

        print(
            f"Unreliable interaction {i + 1}: "
            f"trust = {trust:.4f}"
        )

    # ==============================================================
    # 10. FINAL TRUST
    # ==============================================================

    separator()
    print("[10] FINAL TRUST")
    print("-" * 70)

    final_trust = trust_manager.get_trust(
        "BlueAgent1"
    )

    print(
        "Final trust      :",
        round(final_trust, 4)
    )

    print(
        "Trust level      :",
        trust_manager.get_trust_level(
            "BlueAgent1"
        )
    )

    print(
        "Is trusted       :",
        trust_manager.is_trusted(
            "BlueAgent1"
        )
    )

    # ==============================================================
    # 11. TRUST HISTORY
    # ==============================================================

    separator()
    print("[11] TRUST HISTORY")
    print("-" * 70)

    history = trust_manager.get_history(
        "BlueAgent1"
    )

    for i, value in enumerate(history):

        print(
            f"Interaction {i}: "
            f"{value:.4f}"
        )

    # ==============================================================
    # 12. TRUST STATISTICS
    # ==============================================================

    separator()
    print("[12] TRUST STATISTICS")
    print("-" * 70)

    statistics = trust_manager.get_statistics(
        "BlueAgent1"
    )

    for key, value in statistics.items():

        print(
            f"{key}: {value}"
        )

    separator()
    print("DEMONSTRATION COMPLETE")
    separator()


if __name__ == "__main__":
    main()