from .message import Message
from cryptography.fernet import Fernet


class CommunicationLayer:

    def __init__(
        self,
        authentication_key: bytes = b"cyborg-secret-key",
        encryption_key: bytes = None
    ):
        self.message_queues = {}
        self.authentication_key = authentication_key
        self.encryption_key = encryption_key or Fernet.generate_key()

        # Store message IDs that have already been delivered
        self.processed_message_ids = set()

    def register_agent(self, agent_name):
        if agent_name not in self.message_queues:
            self.message_queues[agent_name] = []

    def register_agents(self, agent_names):
        for agent_name in agent_names:
            self.register_agent(agent_name)

    def send_message(self, sender, receiver, message_type, payload):

        if sender not in self.message_queues:
            raise ValueError(f"Unknown sender: {sender}")

        if receiver not in self.message_queues:
            raise ValueError(f"Unknown receiver: {receiver}")

        message = Message(
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            payload=payload
        )

        message.encrypt(self.encryption_key)
        message.authenticate(self.authentication_key)

        self.message_queues[receiver].append(message)

        return message

    def receive_messages(self, agent_name):

        if agent_name not in self.message_queues:
            raise ValueError(f"Unknown agent: {agent_name}")

        messages = self.message_queues[agent_name].copy()
        self.message_queues[agent_name].clear()

        valid_messages = []

        for message in messages:

            # Reject replayed messages
            if message.message_id in self.processed_message_ids:
                continue

            # Verify authentication
            if not message.verify(self.authentication_key):
                continue

            # Decrypt valid message
            message.decrypt(self.encryption_key)

            # Mark message as processed
            self.processed_message_ids.add(message.message_id)

            valid_messages.append(message)

        return valid_messages

    def peek_messages(self, agent_name):

        if agent_name not in self.message_queues:
            raise ValueError(f"Unknown agent: {agent_name}")

        return self.message_queues[agent_name].copy()

    def clear(self):

        for agent_name in self.message_queues:
            self.message_queues[agent_name].clear()

        self.processed_message_ids.clear()