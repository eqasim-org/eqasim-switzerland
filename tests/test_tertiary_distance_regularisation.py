"""A shared soft preference must preserve support and normalised origin rows."""
import unittest
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from data.od.tertiary_education_smoothing import regularise_distances


class TertiaryDistanceRegularisationTest(unittest.TestCase):
    def test_soft_penalty_preserves_support_and_can_be_disabled(self):
        municipalities = gpd.GeoDataFrame({'municipality_id': [1, 2]},
            geometry=[Point(2600000, 1200000), Point(2700000, 1200000)], crs=2056)
        quarters = gpd.GeoDataFrame({'quarter_id': []}, geometry=[], crs=2056)
        zones = pd.DataFrame({'zone_id': [10, 20], 'zone_level_id': [1, 2], 'zone_level': 'municipality'})
        profiles = pd.DataFrame([[.5, .5], [0., 1.]], index=[10, 20], columns=[10, 20])
        adjusted = regularise_distances(profiles, zones, municipalities, quarters, .005)
        self.assertAlmostEqual(adjusted.loc[10, 10], 1 / (1 + np.exp(-.5)))
        np.testing.assert_allclose(adjusted.sum(axis=1), 1)
        np.testing.assert_array_equal(adjusted.to_numpy() > 0, profiles.to_numpy() > 0)
        pd.testing.assert_frame_equal(regularise_distances(profiles, zones, municipalities, quarters, 0), profiles)
        pd.testing.assert_frame_equal(regularise_distances(profiles, zones, municipalities, quarters, .005, [1., 1.]), profiles)
        with self.assertRaises(ValueError):
            regularise_distances(profiles, zones, municipalities, quarters, -.005)


if __name__ == '__main__':
    unittest.main()
