"""Central, explicit configuration for the zone-risk prediction pipeline."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
MODEL_DIR = PACKAGE_DIR / "models"
REPORT_DIR = PACKAGE_DIR / "reports"

ZONE_TO_AGENT = {
    "restricted_a": "blue_agent_0",
    "operational_a": "blue_agent_1",
    "restricted_b": "blue_agent_2",
    "operational_b": "blue_agent_3",
    "hq": "blue_agent_4",
}
ZONE_HOST_TOKENS = {
    "restricted_a": ("restricted_zone_a",),
    "operational_a": ("operational_zone_a",),
    "restricted_b": ("restricted_zone_b",),
    "operational_b": ("operational_zone_b",),
    "hq": ("public_access_zone", "admin_network", "office_network"),
}
# Contractor hosts are deliberately not silently assigned to HQ: that network is
# outside the five blue-zone ownership mapping above and must be handled explicitly.
UNMAPPED_HOST_TOKENS = ("contractor_network", "internet_subnet")

ATTACK_ACTIONS = {"ExploitRemoteService", "PrivilegeEscalate", "DegradeServices", "Impact"}
SCAN_ACTIONS = {"DiscoverRemoteSystems", "DiscoverNetworkServices", "AggressiveServiceDiscovery", "StealthServiceDiscovery", "DiscoverDeception"}
DAMAGE_ACTIONS = {"DegradeServices", "Impact"}

@dataclass(frozen=True)
class PredictionConfig:
    episode_steps: int = 100
    windows: tuple = (3, 5, 10)
    event_merge_gap: int = 5
    random_seed: int = 42
    min_episodes: int = 10
    test_size: float = 0.2
    validation_size: float = 0.2
    history_steps: int = 5
    zone_local_runtime_risk: bool = True

    def validate(self):
        if self.episode_steps < max(self.windows) + 1:
            raise ValueError("episode_steps must exceed the largest prediction window")
        if self.event_merge_gap < 0:
            raise ValueError("event_merge_gap must be >= 0")
        if not self.windows or any(int(w) < 1 for w in self.windows):
            raise ValueError("windows must contain positive integers")
        if not 0 < self.test_size < 1 or not 0 < self.validation_size < 1:
            raise ValueError("test_size and validation_size must be between 0 and 1")
        return self

    def save(self, path=PACKAGE_DIR / "config.json"):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def load_config(path=None):
    path = Path(path) if path else PACKAGE_DIR / "config.json"
    if not path.exists():
        return PredictionConfig().validate()
    data = json.loads(path.read_text(encoding="utf-8"))
    if "windows" in data:
        data["windows"] = tuple(data["windows"])
    return PredictionConfig(**data).validate()
