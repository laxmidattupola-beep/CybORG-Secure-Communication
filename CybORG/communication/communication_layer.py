from .message import Message


class CommunicationLayer:
    """
    Basic agent-to-agent communication layer.

    Stage 2:
    - Uses structured Message objects.
    - Allows any CybORG agent to send a message to another agent.
    - Stores messages in per-agent receive queues.

    Security features such as authentication, encryption,
    integrity checking, and replay protection will be added later.
    """

    def __init__(self):
        self.message_queues = {}

    def register_agent(self, agent_name):
        """
        Register an agent with the communication layer.
        """

        if agent_name not in self.message_queues:
            self.message_queues[agent_name] = []

    def register_agents(self, agent_names):
        """
        Register multiple agents.
        """

        for agent_name in agent_names:
            self.register_agent(agent_name)

    def send_message(self, sender, receiver, message_type, payload):
        """
        Create and send a structured message from one agent to another.
        """

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

        self.message_queues[receiver].append(message)

        return message

    def receive_messages(self, agent_name):
        """
        Return all messages received by an agent.

        Messages are removed from the queue after being received.
        """

        if agent_name not in self.message_queues:
            raise ValueError(f"Unknown agent: {agent_name}")

        messages = self.message_queues[agent_name].copy()

        self.message_queues[agent_name].clear()

        return messages

    def peek_messages(self, agent_name):
        """
        View messages without removing them.
        """

        if agent_name not in self.message_queues:
            raise ValueError(f"Unknown agent: {agent_name}")

        return self.message_queues[agent_name].copy()

    def clear(self):
        """
        Clear all communication queues.
        """

        for agent_name in self.message_queues:
            self.message_queues[agent_name].clear()