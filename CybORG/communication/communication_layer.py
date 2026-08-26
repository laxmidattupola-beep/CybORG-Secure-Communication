class CommunicationLayer:
    """
    Basic agent-to-agent communication layer.

    Stage 1:
    - Allows any CybORG agent to send a message to another agent.
    - Stores messages in per-agent receive queues.
    - No encryption, authentication, integrity checking,
      replay protection, or attack simulation yet.
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

    def send_message(self, sender, receiver, message):
        """
        Send a message from one agent to another.
        """

        if sender not in self.message_queues:
            raise ValueError(f"Unknown sender: {sender}")

        if receiver not in self.message_queues:
            raise ValueError(f"Unknown receiver: {receiver}")

        self.message_queues[receiver].append({
            "sender": sender,
            "receiver": receiver,
            "message": message
        })

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