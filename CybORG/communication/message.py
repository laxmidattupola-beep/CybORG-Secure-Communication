from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass
class Message:
    """
    Structured message used by the communication layer.

    Security features such as authentication, encryption,
    and replay protection will be added in later stages.
    """

    sender: str
    receiver: str
    message_type: str
    payload: Any

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self):
        """
        Convert the structured message into a dictionary.
        """

        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type,
            "timestamp": self.timestamp,
            "payload": self.payload
        }