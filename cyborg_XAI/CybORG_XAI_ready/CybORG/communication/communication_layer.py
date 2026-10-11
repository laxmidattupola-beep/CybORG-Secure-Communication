import hashlib
import hmac
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List

from CybORG.communication.trust.trust_manager import TrustManager

# ============================================================
# MESSAGE ENCODER
# ============================================================

class MessageEncoder:

    MESSAGE_CODES = {
        "NORMAL": 0,
        "SUSPICIOUS_ACTIVITY": 1,
        "MALWARE_DETECTED": 2,
        "HOST_COMPROMISED": 3,
        "LATERAL_MOVEMENT": 4,
        "HIGH_RISK_ATTACK": 5,
        "NETWORK_INTRUSION": 6,
        "PORT_SCAN_DETECTED": 7,
        "UNAUTHORIZED_ACCESS": 8,
        "NETWORK_ANOMALY": 9,
        "PRIVILEGE_ESCALATION": 10,
        "DATA_EXFILTRATION": 11,
        "CREDENTIAL_ATTACK": 12,

        "FIREWALL_ENABLED": 20,
        "FIREWALL_DISABLED": 21,
        "HOST_ISOLATED": 22,
        "THREAT_BLOCKED": 23,
        "INCIDENT_CONTAINED": 24,

        "HOST_SECURE": 30,
        "HOST_AT_RISK": 31,
        "NETWORK_SECURE": 32,
        "NETWORK_AT_RISK": 33,

        "REQUEST_ASSISTANCE": 40,
        "REQUEST_ISOLATION": 41,
        "REQUEST_SCAN": 42,
        "REQUEST_RESPONSE": 43,

        "THREAT_LEVEL_LOW": 50,
        "THREAT_LEVEL_MEDIUM": 51,
        "THREAT_LEVEL_HIGH": 52,
        "THREAT_LEVEL_CRITICAL": 53,
    }

    @classmethod
    def encode(cls, message_type: str) -> str:

        if message_type not in cls.MESSAGE_CODES:
            raise ValueError(
                f"Unknown message type: {message_type}"
            )

        return format(
            cls.MESSAGE_CODES[message_type],
            "08b"
        )

    @classmethod
    def decode(cls, encoded_message: str) -> str:

        if (
            len(encoded_message) != 8
            or any(bit not in "01"
                   for bit in encoded_message)
        ):
            raise ValueError(
                "Message must contain exactly 8 binary bits."
            )

        code = int(encoded_message, 2)

        for message_type, message_code in cls.MESSAGE_CODES.items():

            if message_code == code:
                return message_type

        return "UNKNOWN"


# ============================================================
# MESSAGE
# ============================================================

@dataclass
class Message:

    message_id: str
    sender: str
    receiver: str

    message_type: str
    encoded_message: str

    target_type: str
    target_id: int
    target_name: str

    timestamp: float
    sequence_number: int

    signature: str = ""


# ============================================================
# COMMUNICATION LAYER
# ============================================================

class CommunicationLayer:

    def __init__(
        self,
        agent_id: str,
        agent_ids: List[str],
        trust_manager: TrustManager,
        shared_secret: bytes
    ):

        self.agent_id = agent_id
        self.agent_ids = agent_ids
        self.trust_manager = trust_manager
        self.shared_secret = shared_secret

        self.inbox: List[Message] = []
        self.outbox: List[Message] = []

        self.sequence_number = 0

        # Store processed message IDs
        self.processed_message_ids = set()

    # ========================================================
    # CREATE HMAC
    # ========================================================

    def _create_hmac(self, message: Message) -> str:

        data = (
            f"{message.message_id}|"
            f"{message.sender}|"
            f"{message.receiver}|"
            f"{message.message_type}|"
            f"{message.encoded_message}|"
            f"{message.target_type}|"
            f"{message.target_id}|"
            f"{message.target_name}|"
            f"{message.timestamp}|"
            f"{message.sequence_number}"
        )

        return hmac.new(
            self.shared_secret,
            data.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    # ========================================================
    # CREATE MESSAGE
    # ========================================================

    def _create_message(
        self,
        receiver: str,
        message_type: str,
        target_type: str,
        target_id: int,
        target_name: str
    ) -> Message:

        if not 0 <= target_id <= 255:
            raise ValueError(
                "Target ID must be between 0 and 255."
            )

        encoded_message = MessageEncoder.encode(
            message_type
        )

        self.sequence_number += 1

        message = Message(
            message_id=str(uuid.uuid4()),
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            encoded_message=encoded_message,
            target_type=target_type,
            target_id=target_id,
            target_name=target_name,
            timestamp=time.time(),
            sequence_number=self.sequence_number
        )

        message.signature = self._create_hmac(message)

        self.outbox.append(message)

        return message

    # ========================================================
    # BROADCAST
    # ========================================================

    def broadcast_message(
        self,
        message_type: str,
        target_type: str,
        target_id: int,
        target_name: str
    ) -> List[Message]:

        messages = []

        for receiver in self.agent_ids:

            if receiver == self.agent_id:
                continue

            message = self._create_message(
                receiver=receiver,
                message_type=message_type,
                target_type=target_type,
                target_id=target_id,
                target_name=target_name
            )

            messages.append(message)

        return messages

    # ========================================================
    # VERIFY HMAC
    # ========================================================

    def verify_hmac(self, message: Message) -> bool:

        expected_signature = self._create_hmac(
            message
        )

        return hmac.compare_digest(
            expected_signature,
            message.signature
        )

    # ========================================================
    # RECEIVE MESSAGE
    # ========================================================

    def receive_message(
        self,
        message: Message,
        update_trust: bool = True
    ) -> Dict:

        sender = message.sender

        # Make sure sender exists in TrustManager
        self.trust_manager.register_agent(sender)

        # ----------------------------------------------------
        # Receiver validation
        # ----------------------------------------------------

        if message.receiver != self.agent_id:

            if update_trust:
                self.trust_manager.update_trust(
                    sender,
                    successful_communication=False,
                    authentication=False,
                    behavior=False
                )

            return {
                "accepted": False,
                "reason": "WRONG_RECEIVER",
                "authentication": False,
                "replay": False
            }

        # ----------------------------------------------------
        # HMAC authentication
        # ----------------------------------------------------

        authentication_valid = self.verify_hmac(
            message
        )

        if not authentication_valid:

            if update_trust:
                self.trust_manager.update_trust(
                    sender,
                    successful_communication=False,
                    authentication=False,
                    behavior=False
                )

            return {
                "accepted": False,
                "reason": "TAMPERING_DETECTED",
                "authentication": False,
                "replay": False
            }

        # ----------------------------------------------------
        # Replay protection
        # ----------------------------------------------------

        if message.message_id in self.processed_message_ids:

            if update_trust:
                self.trust_manager.update_trust(
                    sender,
                    successful_communication=False,
                    authentication=True,
                    behavior=False
                )

            return {
                "accepted": False,
                "reason": "REPLAY_ATTACK_BLOCKED",
                "authentication": True,
                "replay": False
            }

        # ----------------------------------------------------
        # Validate 8-bit message
        # ----------------------------------------------------

        if (
            len(message.encoded_message) != 8
            or any(
                bit not in "01"
                for bit in message.encoded_message
            )
        ):

            return {
                "accepted": False,
                "reason": "INVALID_8_BIT_MESSAGE",
                "authentication": True,
                "replay": True
            }

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        decoded_message = MessageEncoder.decode(
            message.encoded_message
        )

        # ----------------------------------------------------
        # Accept message
        # ----------------------------------------------------

        self.processed_message_ids.add(
            message.message_id
        )

        self.inbox.append(message)

        # ----------------------------------------------------
        # Update trust
        # ----------------------------------------------------

        if update_trust:

            trust = self.trust_manager.update_trust(
                sender,
                successful_communication=True,
                authentication=True,
                behavior=True
            )

        else:

            trust = self.trust_manager.get_trust(
                sender
            )

        return {
            "accepted": True,
            "reason": "MESSAGE_ACCEPTED",

            "authentication": True,
            "replay": True,

            "decoded_message": decoded_message,

            "target_type": message.target_type,
            "target_id": message.target_id,
            "target_name": message.target_name,

            "trust": trust,

            "trusted": self.trust_manager.is_trusted(
                sender
            )
        }


# ============================================================
# MULTI-AGENT COMMUNICATION NETWORK
# ============================================================

class MultiAgentCommunicationNetwork:

    def __init__(self, agent_ids: List[str]):

        self.agent_ids = agent_ids

        # ----------------------------------------------------
        # ONE TRUST MANAGER
        # ----------------------------------------------------

        self.trust_manager = TrustManager()

        # Register all agents using the existing
        # TrustManager interface.
        self.trust_manager.register_agents(
            agent_ids
        )

        # ----------------------------------------------------
        # ONE SHARED SECRET
        # ----------------------------------------------------

        self.shared_secret = (
            b"CYBORG_SECURE_SHARED_SECRET_2026"
        )

        # ----------------------------------------------------
        # Create communication layer for each agent
        # ----------------------------------------------------

        self.agents = {}

        for agent_id in agent_ids:

            self.agents[agent_id] = CommunicationLayer(
                agent_id=agent_id,
                agent_ids=agent_ids,
                trust_manager=self.trust_manager,
                shared_secret=self.shared_secret
            )

    # ========================================================
    # BROADCAST
    # ========================================================

    def broadcast(
        self,
        sender: str,
        message_type: str,
        target_type: str,
        target_id: int,
        target_name: str
    ) -> List[Message]:

        if sender not in self.agents:

            raise ValueError(
                f"Unknown sender: {sender}"
            )

        return self.agents[sender].broadcast_message(
            message_type=message_type,
            target_type=target_type,
            target_id=target_id,
            target_name=target_name
        )

    # ========================================================
    # DELIVER
    # ========================================================

    def deliver_message(
        self,
        message: Message,
        update_trust: bool = True
    ) -> Dict:

        receiver = self.agents.get(
            message.receiver
        )

        if receiver is None:

            return {
                "accepted": False,
                "reason": "UNKNOWN_RECEIVER"
            }

        return receiver.receive_message(
            message,
            update_trust=update_trust
        )

    # ========================================================
    # GET AGENT
    # ========================================================

    def get_agent(
        self,
        agent_id: str
    ):

        return self.agents.get(agent_id)