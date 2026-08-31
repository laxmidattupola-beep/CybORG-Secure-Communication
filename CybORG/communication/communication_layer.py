"""
Communication layer for CybORG blue agents.

Messages are exchanged between blue agents and evaluated by the
TrustManager.

The communication layer is responsible for:

    Agent
      |
      v
    Message
      |
      v
  Communication
      |
      v
  Authentication
      |
      v
 TrustManager
      |
      v
 Message accepted / rejected

The research paper describes 8-bit communication between blue agents
operating under limited bandwidth and partial observability.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:
    Fernet = None
    InvalidToken = Exception

from trust.trust_manager import TrustManager


@dataclass
class Message:
    """
    Structured inter-agent message.
    """

    message_id: str
    sender: str
    receiver: str
    message_type: str
    payload: dict
    timestamp: float
    sequence_number: int
    signature: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            separators=(",", ":"),
        )

    @classmethod
    def from_json(
        cls,
        data: str,
    ) -> "Message":
        return cls(
            **json.loads(data)
        )


class CommunicationLayer:
    """
    Handles structured communication between blue agents.

    Trust is maintained independently for every sender.
    """

    def __init__(
        self,
        agent_id: str,
        agent_ids: Optional[List[str]] = None,
        trust_manager: Optional[TrustManager] = None,
    ):
        self.agent_id = agent_id

        self.trust_manager = (
            trust_manager
            if trust_manager is not None
            else TrustManager()
        )

        self.agent_ids = list(
            agent_ids or []
        )

        if self.agent_id not in self.agent_ids:
            self.agent_ids.append(
                self.agent_id
            )

        for agent in self.agent_ids:
            self.trust_manager.register_agent(
                agent
            )

        self.outbox: List[Message] = []
        self.inbox: List[Message] = []

        self.processed_message_ids = set()

        self.sequence_number = 0

        self.shared_secret = self._generate_secret()

    # ------------------------------------------------------------------
    # Secret
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_secret() -> bytes:
        """
        Generate a deterministic local secret.

        The actual network implementation can replace this with a
        shared key-distribution mechanism.
        """

        return Fernet.generate_key() if Fernet else b"cyborg-local-key"

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def send_message(
        self,
        receiver: str,
        message_type: str,
        payload: dict,
    ) -> Message:
        """
        Create and queue a structured message.
        """

        if not isinstance(payload, dict):
            raise TypeError(
                "payload must be a dictionary."
            )

        if receiver == self.agent_id:
            raise ValueError(
                "An agent cannot send a message to itself."
            )

        self.sequence_number += 1

        message = Message(
            message_id=str(uuid.uuid4()),
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            payload=payload,
            timestamp=time.time(),
            sequence_number=self.sequence_number,
        )

        message.signature = self._sign_message(
            message
        )

        self.outbox.append(message)

        return message

    # ------------------------------------------------------------------
    # Receiving
    # ------------------------------------------------------------------

    def receive_message(
        self,
        message: Message,
    ) -> Optional[dict]:
        """
        Receive and validate a message.

        Authentication and replay protection are performed before
        the message is accepted.

        Trust is updated after validation.
        """

        if not isinstance(message, Message):
            raise TypeError(
                "message must be a Message object."
            )

        sender = message.sender

        self.trust_manager.register_agent(
            sender
        )

        # --------------------------------------------------------------
        # Receiver validation
        # --------------------------------------------------------------

        if message.receiver != self.agent_id:
            self.trust_manager.update_trust(
                sender,
                successful_communication=False,
                authentication=False,
                behavior=False,
            )

            return None

        # --------------------------------------------------------------
        # Replay protection
        # --------------------------------------------------------------

        if (
            message.message_id
            in self.processed_message_ids
        ):
            self.trust_manager.update_trust(
                sender,
                successful_communication=False,
                authentication=True,
                behavior=False,
            )

            return None

        # --------------------------------------------------------------
        # Signature validation
        # --------------------------------------------------------------

        authenticated = self._verify_signature(
            message
        )

        if not authenticated:
            self.trust_manager.update_trust(
                sender,
                successful_communication=False,
                authentication=False,
                behavior=False,
            )

            return None

        # --------------------------------------------------------------
        # Message validation
        # --------------------------------------------------------------

        valid_payload = self._validate_payload(
            message.payload
        )

        if not valid_payload:
            self.trust_manager.update_trust(
                sender,
                successful_communication=False,
                authentication=True,
                behavior=False,
            )

            return None

        # --------------------------------------------------------------
        # Successful communication
        # --------------------------------------------------------------

        trust = self.trust_manager.update_trust(
            sender,
            successful_communication=True,
            authentication=True,
            behavior=True,
        )

        self.processed_message_ids.add(
            message.message_id
        )

        self.inbox.append(message)

        return {
            "message_id": message.message_id,
            "sender": sender,
            "receiver": message.receiver,
            "message_type": message.message_type,
            "payload": message.payload,
            "trust": trust,
            "trusted": self.trust_manager.is_trusted(
                sender
            ),
        }

    # ------------------------------------------------------------------
    # Message verification
    # ------------------------------------------------------------------

    def _sign_message(
        self,
        message: Message,
    ) -> str:
        """
        Generate a signature for the message.
        """

        unsigned = {
            "message_id": message.message_id,
            "sender": message.sender,
            "receiver": message.receiver,
            "message_type": message.message_type,
            "payload": message.payload,
            "timestamp": message.timestamp,
            "sequence_number": message.sequence_number,
        }

        serialized = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
        )

        digest = hashlib.sha256(
            serialized.encode("utf-8")
        ).digest()

        return base64.b64encode(
            digest
        ).decode("utf-8")

    def _verify_signature(
        self,
        message: Message,
    ) -> bool:
        """
        Verify the message signature.
        """

        expected = self._sign_message(
            message
        )

        return expected == message.signature

    # ------------------------------------------------------------------
    # Payload validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_payload(
        payload: dict,
    ) -> bool:
        """
        Validate structured message payload.

        A payload must be a dictionary.
        """

        if not isinstance(payload, dict):
            return False

        return True

    # ------------------------------------------------------------------
    # Trust
    # ------------------------------------------------------------------

    def get_sender_trust(
        self,
        sender: str,
    ) -> float:
        """
        Get current trust of another agent.
        """

        return self.trust_manager.get_trust(
            sender
        )

    def is_sender_trusted(
        self,
        sender: str,
        threshold: float = 0.5,
    ) -> bool:
        """
        Determine whether a sender is trusted.
        """

        return self.trust_manager.is_trusted(
            sender,
            threshold=threshold,
        )

    # ------------------------------------------------------------------
    # Queues
    # ------------------------------------------------------------------

    def peek_messages(self) -> List[Message]:
        """
        Return messages currently in the inbox.
        """

        return list(self.inbox)

    def clear_messages(self) -> None:
        """
        Clear received messages.
        """

        self.inbox.clear()

    def get_outbox(self) -> List[Message]:
        """
        Return queued outgoing messages.
        """

        return list(self.outbox)

    def clear_outbox(self) -> None:
        """
        Clear outgoing messages.
        """

        self.outbox.clear()

    # ------------------------------------------------------------------
    # 8-bit communication
    # ------------------------------------------------------------------

    @staticmethod
    def encode_8bit(
        value: int,
    ) -> str:
        """
        Encode an integer into an 8-bit binary string.

        Valid range:

            0 <= value <= 255
        """

        if not isinstance(value, int):
            raise TypeError(
                "8-bit value must be an integer."
            )

        if not 0 <= value <= 255:
            raise ValueError(
                "8-bit value must be between 0 and 255."
            )

        return format(
            value,
            "08b",
        )

    @staticmethod
    def decode_8bit(
        value: str,
    ) -> int:
        """
        Decode an 8-bit binary string.
        """

        if not isinstance(value, str):
            raise TypeError(
                "8-bit value must be a string."
            )

        if (
            len(value) != 8
            or any(bit not in "01" for bit in value)
        ):
            raise ValueError(
                "Value must be exactly 8 binary bits."
            )

        return int(
            value,
            2,
        )