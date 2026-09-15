import json
import socket

from communication_layer import (
    MultiAgentCommunicationNetwork,
    MessageEncoder,
    Message
)


HOST = "127.0.0.1"
PORT = 5000


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_line():
    print("-" * 70)


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# MENU
# ============================================================

def get_choice(prompt, options):

    while True:

        try:
            choice = int(input(prompt))

            if 0 <= choice < len(options):
                return options[choice]

            print(
                f"Please enter a value between "
                f"0 and {len(options) - 1}."
            )

        except ValueError:
            print("Please enter a number.")


# ============================================================
# TARGET ID
# ============================================================

def get_target_id(target_type):

    if target_type == "NONE":
        return 0

    if target_type == "HOST":
        minimum = 0
        maximum = 93
    else:
        minimum = 0
        maximum = 8

    while True:

        try:

            target_id = int(
                input("Enter Target ID: ")
            )

            if minimum <= target_id <= maximum:
                return target_id

            print(
                f"Please enter a value between "
                f"{minimum} and {maximum}."
            )

        except ValueError:
            print("Please enter a number.")


# ============================================================
# TARGET NAME
# ============================================================

def get_target_name(target_type, target_id):

    if target_type == "NONE":
        return "None"

    if target_type == "HOST":
        return f"Host_id={target_id}"

    if target_type == "SUBNET":
        return f"Subnet_id={target_id}"

    return "Unknown"


# ============================================================
# MESSAGE → DICTIONARY
# ============================================================

def message_to_dict(message):

    return {
        "message_id": message.message_id,
        "sender": message.sender,
        "receiver": message.receiver,
        "message_type": message.message_type,
        "encoded_message": message.encoded_message,
        "target_type": message.target_type,
        "target_id": message.target_id,
        "target_name": message.target_name,
        "timestamp": message.timestamp,
        "sequence_number": message.sequence_number,
        "signature": message.signature
    }


# ============================================================
# DICTIONARY → MESSAGE
# ============================================================

def dict_to_message(data):

    return Message(
        message_id=data["message_id"],
        sender=data["sender"],
        receiver=data["receiver"],
        message_type=data["message_type"],
        encoded_message=data["encoded_message"],
        target_type=data["target_type"],
        target_id=data["target_id"],
        target_name=data["target_name"],
        timestamp=data["timestamp"],
        sequence_number=data["sequence_number"],
        signature=data["signature"]
    )


# ============================================================
# SEND ALL BROADCAST MESSAGES TO ATTACKER
# ============================================================

def communicate_with_attacker(messages):

    message_data = [
        message_to_dict(message)
        for message in messages
    ]

    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    ) as server_socket:

        server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )

        server_socket.bind(
            (HOST, PORT)
        )

        server_socket.listen(1)

        connection, address = (
            server_socket.accept()
        )

        with connection:

            # Send all receiver-specific messages
            data = json.dumps(
                message_data
            ).encode("utf-8")

            connection.sendall(data)

            # Receive attacker's selected message
            response = connection.recv(8192)

            if not response:
                return None

            return json.loads(
                response.decode("utf-8")
            )


# ============================================================
# DISPLAY RECEIVER RESULT
# ============================================================

def display_receiver_result(
    network,
    received_message
):

    result = network.deliver_message(
        received_message
    )

    print(
        f"\n{received_message.receiver}"
    )

    print_line()

    print(
        f"Received From       : "
        f"{received_message.sender}"
    )

    print(
        f"Received Message    : "
        f"{received_message.encoded_message}"
    )

    print(
        f"Target ID           : "
        f"{received_message.target_id}"
    )

    print(
        f"Target              : "
        f"{received_message.target_name}"
    )

    print()

    authentication = result.get(
        "authentication",
        False
    )

    replay = result.get(
        "replay",
        False
    )

    print(
        f"HMAC Verification   : "
        f"{authentication}"
    )

    print(
        f"Replay Check        : "
        f"{'PASSED' if replay else 'BLOCKED'}"
    )

    if result.get("accepted", False):

        print(
            "Authentication      : VALID"
        )

        print(
            f"Decoded Message     : "
            f"{result.get('decoded_message')}"
        )

        print(
            f"Trust Score         : "
            f"{result.get('trust', 0.0):.3f}"
        )

        print(
            f"Trusted Sender      : "
            f"{'YES' if result.get('trusted') else 'NO'}"
        )

        print(
            "Message Accepted    : YES"
        )

    else:

        reason = result.get(
            "reason",
            "MESSAGE_REJECTED"
        )

        if reason == "TAMPERING_DETECTED":

            print(
                "Authentication      : INVALID"
            )

            print(
                "RESULT              : "
                "TAMPERING DETECTED"
            )

        elif reason == "REPLAY_ATTACK_BLOCKED":

            print(
                "Authentication      : VALID"
            )

            print(
                "RESULT              : "
                "REPLAY ATTACK BLOCKED"
            )

        else:

            print(
                "Authentication      : INVALID"
            )

            print(
                f"RESULT              : {reason}"
            )

        print(
            "Message Accepted    : NO"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # AGENTS
    # --------------------------------------------------------

    agents = [
        "Agent_1",
        "Agent_2",
        "Agent_3",
        "Agent_4",
        "Agent_5"
    ]

    network = MultiAgentCommunicationNetwork(
        agents
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print_header(
        "CYBORG SECURE BROADCAST COMMUNICATION"
    )

    # --------------------------------------------------------
    # MANUAL INPUT
    # --------------------------------------------------------

    print("\nMANUAL INPUT")
    print_line()

    # --------------------------------------------------------
    # EVENT
    # --------------------------------------------------------

    events = [
        "NORMAL",
        "DISCOVERY",
        "SCAN",
        "SUSPICIOUS_ACTIVITY",
        "MALWARE_DETECTED",
        "HOST_COMPROMISED",
        "LATERAL_MOVEMENT",
        "PRIVILEGE_ESCALATION",
        "DATA_EXFILTRATION",
        "NETWORK_INTRUSION",
        "UNAUTHORIZED_ACCESS",
        "PORT_SCAN_DETECTED",
        "CREDENTIAL_ATTACK",
        "HIGH_RISK_ATTACK"
    ]

    print("\nEvent Type")
    print_line()

    for i, event in enumerate(events):
        print(f"  {i}. {event}")

    event = get_choice(
        "\nSelect number: ",
        events
    )

    # --------------------------------------------------------
    # TARGET TYPE
    # --------------------------------------------------------

    target_types = [
        "NONE",
        "HOST",
        "SUBNET"
    ]

    print("\nTarget Type")
    print_line()

    for i, target in enumerate(target_types):
        print(f"  {i}. {target}")

    target_type = get_choice(
        "\nSelect number: ",
        target_types
    )

    # --------------------------------------------------------
    # TARGET ID
    # --------------------------------------------------------

    if target_type != "NONE":

        print("\nTarget ID")
        print_line()

        if target_type == "HOST":
            print("Valid range: 0 - 93")
        else:
            print("Valid range: 0 - 8")

    target_id = get_target_id(
        target_type
    )

    target_name = get_target_name(
        target_type,
        target_id
    )

    # --------------------------------------------------------
    # THREAT LEVEL
    # --------------------------------------------------------

    threat_levels = [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]

    print("\nThreat Level")
    print_line()

    for i, level in enumerate(threat_levels):
        print(f"  {i}. {level}")

    threat_level = get_choice(
        "\nSelect number: ",
        threat_levels
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    statuses = [
        "NORMAL",
        "SUSPICIOUS",
        "COMPROMISED",
        "CONTAINED"
    ]

    print("\nStatus")
    print_line()

    for i, status in enumerate(statuses):
        print(f"  {i}. {status}")

    status = get_choice(
        "\nSelect number: ",
        statuses
    )

    # --------------------------------------------------------
    # SENDER
    # --------------------------------------------------------

    print("\nSender")
    print_line()

    for i, agent in enumerate(agents):
        print(f"  {i}. {agent}")

    sender = get_choice(
        "\nSelect number: ",
        agents
    )

    receivers = [
        agent
        for agent in agents
        if agent != sender
    ]

    # --------------------------------------------------------
    # SELECTED EVENT
    # --------------------------------------------------------

    print_header(
        "SELECTED SECURITY EVENT"
    )

    print(
        f"Sender              : {sender}"
    )

    print(
        f"Receivers           : "
        f"{', '.join(receivers)}"
    )

    print(
        f"Event               : {event}"
    )

    print(
        f"Target Type         : {target_type}"
    )

    print(
        f"Target ID           : {target_id}"
    )

    print(
        f"Target              : {target_name}"
    )

    print(
        f"Threat Level        : {threat_level}"
    )

    print(
        f"Status              : {status}"
    )

    # --------------------------------------------------------
    # BROADCAST
    # --------------------------------------------------------

    messages = network.broadcast(
        sender=sender,
        message_type=event,
        target_type=target_type,
        target_id=target_id,
        target_name=target_name
    )

    encoded_message = messages[0].encoded_message

    # --------------------------------------------------------
    # ENCODING
    # --------------------------------------------------------

    print_header(
        "ENCODED 8-BIT MESSAGE"
    )

    print(
        f"Semantic Message    : {event}"
    )

    print(
        f"Message Code        : "
        f"{MessageEncoder.MESSAGE_CODES[event]}"
    )

    print(
        f"Encoded Message     : "
        f"{encoded_message}"
    )

    print(
        "Message Size        : 8 bits"
    )

    print()

    print(
        f"Target ID           : {target_id}"
    )

    print(
        f"Target ID Encoding  : "
        f"{format(target_id, '08b')}"
    )

    print(
        "Target Size         : 8 bits"
    )

    # --------------------------------------------------------
    # BROADCAST DISPLAY
    # --------------------------------------------------------

    print_header(
        "BROADCAST"
    )

    for message in messages:

        print(
            f"{message.sender:<12} --> "
            f"{message.receiver}"
        )

    print()

    print(
        f"Broadcast Message   : "
        f"{encoded_message}"
    )

    print(
        f"Affected Target ID  : "
        f"{target_id}"
    )

    print(
        f"Affected Target     : "
        f"{target_name}"
    )

    print(
        "Broadcast Status    : SUCCESSFUL"
    )

    # --------------------------------------------------------
    # COMMUNICATE WITH ATTACKER
    # --------------------------------------------------------

    attacker_response = (
        communicate_with_attacker(messages)
    )

    if attacker_response is None:

        print(
            "No response received from attacker."
        )

        return

    # --------------------------------------------------------
    # RECEIVE ATTACKER RESULT
    # --------------------------------------------------------

    # The attacker returns a LIST of messages
    # so that the normal receiver messages
    # can also be displayed.

    received_messages = [
        dict_to_message(data)
        for data in attacker_response
    ]

    # --------------------------------------------------------
    # RECEIVER SECURITY CHECK
    # --------------------------------------------------------

    print_header(
        "RECEIVER SECURITY CHECK"
    )

    for received_message in received_messages:

        display_receiver_result(
            network,
            received_message
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "       SECURE BROADCAST COMMUNICATION TEST COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()