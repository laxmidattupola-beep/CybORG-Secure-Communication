from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from cryptography.fernet import Fernet
import uuid
import hashlib
import hmac
import json


@dataclass
class Message:
    sender: str
    receiver: str
    message_type: str
    payload: Any

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    authentication_tag: str = ""

    def encrypt(self, key: bytes):
        f = Fernet(key)

        self.payload = f.encrypt(
        json.dumps(self.payload).encode()
    ).decode()


    def decrypt(self, key: bytes):
        f = Fernet(key)

        self.payload = json.loads(
        f.decrypt(self.payload.encode()).decode()
    )

    def _data_to_sign(self):
        return json.dumps({
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type,
            "timestamp": self.timestamp,
            "payload": self.payload
        }, sort_keys=True, default=str)

    def authenticate(self, key: bytes):
        data = self._data_to_sign().encode()

        self.authentication_tag = hmac.new(
            key,
            data,
            hashlib.sha256
        ).hexdigest()

    def verify(self, key: bytes) -> bool:
        if not self.authentication_tag:
            return False

        data = self._data_to_sign().encode()

        expected_tag = hmac.new(
            key,
            data,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(
            self.authentication_tag,
            expected_tag
        )

    def to_dict(self):
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "authentication_tag": self.authentication_tag
        }