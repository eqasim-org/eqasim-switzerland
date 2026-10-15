"""Secondary assignment compatibility, long commutes and fallback invariants."""

import unittest

import numpy as np
import pandas as pd

from synthesis.population.spatial.primary.education.secondary import (
    SCHOOL_CODES, assign_secondary_locations, choose_schools)


class SecondarySchoolSelectionTest(unittest.TestCase):
    def test_long_distance_school_survives_many_nearby_sites(self):
        schools = pd.DataFrame({"enterprise_id": [str(i) for i in range(32)],
            "x": [100.] * 30 + [10000., 20000.], "y": 0., "number_employees": 10})
        people = pd.DataFrame({"home_x": [0.] * 100, "home_y": 0., "commute_home_distance": 10000.})
        selected, valid = choose_schools(people, schools, np.random.RandomState(0))
        self.assertTrue(valid.all())
        self.assertTrue((selected == "30").all())

    def test_far_from_every_school_keeps_distance_preference(self):
        schools = pd.DataFrame({"enterprise_id": ["near", "far"], "x": [10000., 20000.],
            "y": 0., "number_employees": [10, 100]})
        people = pd.DataFrame({"home_x": [0.], "home_y": [0.], "commute_home_distance": [0.]})
        # The closer match remains preferred even where raw Gaussian weights
        # underflow; an extreme example checks stable log-space normalisation.
        selected, valid = choose_schools(people, schools, np.random.RandomState(0))
        self.assertEqual(len(selected), 1)
        self.assertTrue(valid[0])
        self.assertEqual(selected[0], "near")

    def test_missing_distance_uses_nearest_compatible_site(self):
        schools = pd.DataFrame({"enterprise_id": ["near", "far"], "x": [100., 10000.],
            "y": 0., "number_employees": [0, 100]})
        people = pd.DataFrame({"home_x": [0., 0.], "home_y": 0., "commute_home_distance": [np.nan, -8.]})
        selected, valid = choose_schools(people, schools, np.random.RandomState(0))
        self.assertFalse(valid.any())
        self.assertTrue((selected == "near").all())

    def test_programme_pools_and_unaffected_people(self):
        people = pd.DataFrame({"person_id": range(1, 8), "age": [16, 16, 16, 14, 16, 16, 20],
            "is_student": [1, 1, 1, 1, 1, 0, 1], "current_education": [6, 7, 5, 3, 6, 6, 11],
            "education_destination_status": ["domestic"] * 4 + ["abroad", "domestic", "domestic"],
            "home_x": 0., "home_y": 0., "commute_home_distance": 1000.})
        schools = pd.DataFrame({"enterprise_id": ["voc", "gym", "fms", "lower"],
            "noga": ["853200", "853102", "853103", "853101"], "x": 1000., "y": 0., "number_employees": 10})
        locations = pd.DataFrame({"person_id": range(1, 8), "destination_id": "old", "x": 1., "y": 2.})
        result = assign_secondary_locations(people, locations, schools, 0).set_index("person_id")
        self.assertEqual(result.loc[1, "destination_id"], "voc")
        self.assertEqual(result.loc[2, "destination_id"], "gym")
        self.assertIn(result.loc[3, "destination_id"], ["fms", "gym"])
        self.assertTrue((result.loc[[4, 5, 6, 7], "destination_id"] == "old").all())
        self.assertTrue((result.loc[[4, 5, 6, 7], "x"] == 1).all())
        self.assertEqual(set(SCHOOL_CODES[8]), {"853200", "853102", "853103"})
        with self.assertRaisesRegex(ValueError, "No schools for secondary study code"):
            assign_secondary_locations(people, locations, schools[schools.noga.ne("853200")], 0)


if __name__ == "__main__":
    unittest.main()
