"""Protect private household membership and source IDs when recovering residents."""
import unittest

import numpy as np
import pandas as pd

from data.statpop.households import attach_persons


class MissingHouseholdTest(unittest.TestCase):
    def setUp(self):
        self.persons = pd.DataFrame({
            "person_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "household_id": [10, 10, -9, -9, 999, 999, 999, 999],
            "home_x": [2530000.] * 7 + [np.nan],
            "home_y": [1150000.] * 8,
            "statpop_geocoding_indicator": [1, 1, 1, 1, 0, 2, 1, 1],
            "collective_housing_resident": [False] * 6 + [True, False],
        })
        self.households = pd.DataFrame({"household_id": [10, 2000]})

    def test_recovery_is_unique_and_preserves_private_membership(self):
        result = attach_persons(self.persons, self.households).set_index("person_id")
        self.assertEqual(result.index.tolist(), [1, 2, 3, 4])
        self.assertEqual(result.loc[[1, 2], "household_id"].tolist(), [10, 10])
        self.assertEqual(result.loc[[3, 4], "household_id"].tolist(), [2001, 2002])
        self.assertEqual(result.loc[[3, 4], "statpop_source_household_id"].tolist(), [-9, -9])
        self.assertEqual(result["is_synthetic_household"].tolist(), [False, False, True, True])
        reordered = attach_persons(self.persons.iloc[::-1], self.households).set_index("person_id")
        pd.testing.assert_series_equal(result.household_id.sort_index(), reordered.household_id.sort_index())

    def test_disabled_matches_old_inner_join(self):
        result = attach_persons(self.persons, self.households, False)
        expected = self.persons.merge(self.households, on="household_id", validate="many_to_one")
        pd.testing.assert_frame_equal(result[self.persons.columns].reset_index(drop=True), expected)
        self.assertFalse(result.is_synthetic_household.any())

    def test_invalid_coordinates_are_not_recovered(self):
        self.persons.loc[2, "home_x"] = -9
        self.persons.loc[3, "home_y"] = np.inf
        result = attach_persons(self.persons, self.households)
        self.assertEqual(result.person_id.tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
