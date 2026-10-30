import unittest
import numpy as np
from CybORG.mappo.prediction_v2.features import zone_features, feature_names

class FeatureTests(unittest.TestCase):
    def test_shape_and_finiteness(self):
        x = np.zeros(256, dtype=np.float32); x[0] = np.nan
        y = zone_features(x, 'hq', 20, 100, history=[np.ones(256)])
        self.assertEqual(len(y), len(feature_names()))
        self.assertTrue(np.isfinite(y).all())

    def test_zone_features_differ_by_zone(self):
        x = np.zeros(256, dtype=np.float32)
        a = zone_features(x, 'restricted_a', 0, 100)
        b = zone_features(x, 'hq', 0, 100)
        self.assertNotEqual(float(a[-4]), float(b[-4]))

if __name__ == '__main__': unittest.main()
