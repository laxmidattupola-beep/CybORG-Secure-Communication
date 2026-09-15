import json
import socket
import copy


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
# MAIN
# ============================================================

def main():

    print_header(
        "MANUAL ATTACK TERMINAL"
    )

    print(
        "Waiting for a broadcast message..."
    )

    # --------------------------------------------------------
    # CONNECT TO TERMINAL 1
    # --------------------------------------------------------

    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    ) as attacker_socket:

        attacker_socket.connect(
            (HOST, PORT)
        )

        # ----------------------------------------------------
        # RECEIVE ALL BROADCAST MESSAGES
        # ----------------------------------------------------

        data = attacker_socket.recv(
            16384
        )

        if not data:

            print(
                "No broadcast message received."
            )

            return

        messages = json.loads(
            data.decode("utf-8")
        )

        # ----------------------------------------------------
        # DISPLAY CAPTURED BROADCAST
        # ----------------------------------------------------

        print_header(
            "CAPTURED BROADCAST MESSAGE"
        )

        sender = messages[0]["sender"]

        print(
            f"Sender              : {sender}"
        )

        print(
            f"Message             : "
            f"{messages[0]['encoded_message']}"
        )

        print(
            f"Target Type         : "
            f"{messages[0]['target_type']}"
        )

        print(
            f"Target ID           : "
            f"{messages[0]['target_id']}"
        )

        print(
            f"Target              : "
            f"{messages[0]['target_name']}"
        )

        # ----------------------------------------------------
        # RECEIVER SELECTION
        # ----------------------------------------------------

        print_header(
            "SELECT RECEIVER TO ATTACK"
        )

        for i, message in enumerate(messages, 1):

            print(
                f"{i}. {message['receiver']}"
            )

        print()

        while True:

            try:

                receiver_choice = int(
                    input(
                        "Select receiver: "
                    )
                )

                if 1 <= receiver_choice <= len(messages):
                    break

                print(
                    f"Please enter a value between "
                    f"1 and {len(messages)}."
                )

            except ValueError:

                print(
                    "Please enter a number."
                )

        selected_index = (
            receiver_choice - 1
        )

        selected_message = messages[
            selected_index
        ]

        # ----------------------------------------------------
        # DISPLAY SELECTED RECEIVER
        # ----------------------------------------------------

        print_header(
            "SELECTED COMMUNICATION"
        )

        print(
            f"Sender              : "
            f"{selected_message['sender']}"
        )

        print(
            f"Receiver            : "
            f"{selected_message['receiver']}"
        )

        print(
            f"Message             : "
            f"{selected_message['encoded_message']}"
        )

        # ----------------------------------------------------
        # ATTACK SELECTION
        # ----------------------------------------------------

        print_header(
            "SELECT ATTACK"
        )

        print(
            "1. Tampering Attack"
        )

        print(
            "2. Replay Attack"
        )

        print(
            "3. No Attack"
        )

        while True:

            try:

                attack_choice = int(
                    input("\nSelect attack: ")
                )

                if attack_choice in [1, 2, 3]:
                    break

                print(
                    "Please enter 1, 2 or 3."
                )

            except ValueError:

                print(
                    "Please enter a number."
                )

        # ====================================================
        # CREATE RESULT MESSAGES
        # ====================================================

        result_messages = []

        # ====================================================
        # NO ATTACK
        # ====================================================

        if attack_choice == 3:

            print_header(
                "NO ATTACK"
            )

            print(
                "Original broadcast message "
                "will be forwarded."
            )

            # Forward every message unchanged
            result_messages = copy.deepcopy(
                messages
            )

        # ====================================================
        # TAMPERING ATTACK
        # ====================================================

        elif attack_choice == 1:

            print_header(
                "TAMPERING ATTACK"
            )

            original_message = (
                selected_message["encoded_message"]
            )

            # Create a copy so only the
            # selected receiver is affected.
            tampered_messages = copy.deepcopy(
                messages
            )

            selected_tampered = (
                tampered_messages[selected_index]
            )

            original_value = int(
                original_message,
                2
            )

            # Change one semantic code.
            if original_value == 255:
                modified_value = 0
            else:
                modified_value = (
                    original_value + 1
                )

            selected_tampered[
                "encoded_message"
            ] = format(
                modified_value,
                "08b"
            )

            print(
                f"Sender              : "
                f"{selected_message['sender']}"
            )

            print(
                f"Target Receiver     : "
                f"{selected_message['receiver']}"
            )

            print()

            print(
                f"Original Message    : "
                f"{original_message}"
            )

            print(
                f"Modified Message    : "
                f"{selected_tampered['encoded_message']}"
            )

            print()

            print(
                "Original HMAC retained."
            )

            print(
                "Only the selected receiver's "
                "message was modified."
            )

            # Return all messages
            # selected one is tampered.
            result_messages = tampered_messages

        # ====================================================
        # REPLAY ATTACK
        # ====================================================

        elif attack_choice == 2:

            print_header(
                "REPLAY ATTACK"
            )

            print(
                f"Sender              : "
                f"{selected_message['sender']}"
            )

            print(
                f"Target Receiver     : "
                f"{selected_message['receiver']}"
            )

            print()

            print(
                f"Replayed Message    : "
                f"{selected_message['encoded_message']}"
            )

            print()

            print(
                "Original message is "
                "sent again unchanged."
            )

            print(
                "Original HMAC retained."
            )

            # For replay demonstration:
            #
            # First send all messages normally.
            # Then send the selected message again.
            #
            # The receiver has already processed
            # the first copy, so the second copy
            # should be blocked.

            result_messages = copy.deepcopy(
                messages
            )

            # We append the exact same message
            # again for the selected receiver.
            replayed_message = copy.deepcopy(
                selected_message
            )

            result_messages.append(
                replayed_message
            )

        # ----------------------------------------------------
        # SEND RESULT BACK TO TERMINAL 1
        # ----------------------------------------------------

        response = json.dumps(
            result_messages
        ).encode("utf-8")

        attacker_socket.sendall(
            response
        )

        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        print("\n" + "=" * 70)

        print(
            "          ATTACK TRANSMISSION COMPLETE"
        )

        print("=" * 70)


if __name__ == "__main__":
    main()