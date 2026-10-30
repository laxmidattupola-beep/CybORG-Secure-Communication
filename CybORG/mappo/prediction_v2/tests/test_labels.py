import unittest
import pandas as pd
from CybORG.mappo.prediction_v2.models import feature_columns
from types import SimpleNamespace
from CybORG.mappo.prediction_v2.labels import RedEvent, zone_for_host, resolve_zone_for_target, set_subnet_inventory, merge_event_sequences, future_window_labels

class LabelTests(unittest.TestCase):
    def test_zone_mapping(self):
        self.assertEqual(zone_for_host('restricted_zone_a_subnet_server_host_0'), 'restricted_a')
        self.assertEqual(zone_for_host('office_network_subnet_user_host_1'), 'hq')
        self.assertIsNone(zone_for_host('contractor_network_subnet_user_host_1'))

    def test_live_ip_and_cidr_zone_resolution(self):
        names = {
            "10.0.249.0/24": "restricted_zone_a_subnet",
            "10.0.116.0/24": "operational_zone_a_subnet",
            "10.0.64.0/24": "restricted_zone_b_subnet",
            "10.0.88.0/24": "operational_zone_b_subnet",
            "10.0.105.0/24": "public_access_zone_subnet",
            "10.0.143.0/24": "admin_network_subnet",
            "10.0.154.0/24": "office_network_subnet",
            "10.0.212.0/24": "contractor_network_subnet",
            "10.0.103.0/24": "internet_subnet",
        }
        inventory = {
            cidr: SimpleNamespace(name=name, cidr=cidr)
            for cidr, name in names.items()
        }
        set_subnet_inventory(inventory)

        expected = {
            "10.0.249.17": "restricted_a",
            "10.0.249.0/24": "restricted_a",
            "10.0.116.8": "operational_a",
            "10.0.64.12": "restricted_b",
            "10.0.88.23": "operational_b",
            "10.0.105.7": "hq",
            "10.0.143.9": "hq",
            "10.0.154.11": "hq",
        }
        for target, zone in expected.items():
            with self.subTest(target=target):
                self.assertEqual(resolve_zone_for_target(target), zone)

        self.assertIsNone(resolve_zone_for_target("10.0.212.5"))
        self.assertIsNone(resolve_zone_for_target("10.0.212.0/24"))
        self.assertIsNone(resolve_zone_for_target("10.0.103.4"))
        self.assertIsNone(resolve_zone_for_target("10.0.103.0/24"))
        self.assertEqual(
            resolve_zone_for_target("restricted_zone_a_subnet_server_host_0"),
            "restricted_a",
        )

    def test_action_event_resolves_live_ip_target(self):
        class PrivilegeEscalate:
            hostname = "10.0.88.23"
            success = True

        set_subnet_inventory({
            "10.0.88.0/24": SimpleNamespace(
                name="operational_zone_b_subnet", cidr="10.0.88.0/24"
            )
        })
        from CybORG.mappo.prediction_v2.labels import classify_action_event
        event = classify_action_event(4, PrivilegeEscalate())
        self.assertIsNotNone(event)
        self.assertEqual(event.zone, "operational_b")
        self.assertEqual(event.category, "attack_attempt")

    def test_merge_same_zone_category_with_gap(self):
        events = [RedEvent(2,'restricted_a','attack_attempt','Exploit'),
                  RedEvent(7,'restricted_a','attack_attempt','Exploit'),
                  RedEvent(13,'restricted_a','attack_attempt','Exploit')]
        merged = merge_event_sequences(events, gap=5)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]['count'], 2)

    def test_future_windows_are_strictly_future(self):
        events = [RedEvent(5,'restricted_a','attack_attempt','Exploit')]
        rows = future_window_labels(events, 12, windows=(3,5), zones=['restricted_a'])
        by_step = {r['step']: r for r in rows}
        self.assertEqual(by_step[5]['attack_risk_3'], 0)
        self.assertEqual(by_step[2]['attack_risk_3'], 1)
        self.assertEqual(by_step[1]['attack_risk_3'], 0)
        self.assertEqual(by_step[0]['attack_risk_5'], 1)
        self.assertEqual(by_step[0]['compromise_risk_5'], 0)

    def test_future_window_validity_at_episode_boundary(self):
        events = [RedEvent(14, "restricted_a", "attack_attempt", "Exploit")]
        rows = future_window_labels(
            events, 14, windows=(3, 5, 10), zones=["restricted_a"]
        )
        by_step = {row["step"]: row for row in rows}

        self.assertEqual(by_step[10]["label_valid_3"], 1)
        self.assertEqual(by_step[8]["label_valid_5"], 1)
        self.assertEqual(by_step[3]["label_valid_10"], 1)

        self.assertEqual(by_step[11]["label_valid_3"], 0)
        self.assertEqual(by_step[9]["label_valid_5"], 0)
        self.assertEqual(by_step[4]["label_valid_10"], 0)
        self.assertEqual(by_step[13]["label_valid_3"], 0)

    def test_feature_columns_excludes_targets_and_validity_flags(self):
        frame = pd.DataFrame(columns=[
            "episode_id", "step", "zone", "feature_a",
            "attack_risk_3", "attack_risk_5", "attack_risk_10",
            "label_valid_3", "label_valid_5", "label_valid_10",
            "evaluation_stratum_compromised",
        ])
        self.assertEqual(feature_columns(frame), ["feature_a"])

if __name__ == '__main__': unittest.main()
