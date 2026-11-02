"""Home selection follows diary order instead of selecting the nearest home."""
import unittest
import pandas as pd
from data.microcensus.commute import use_diary_home


class MicrocensusEducationHomeTest(unittest.TestCase):
    def test_preceding_following_and_household_fallback(self):
        persons = pd.DataFrame({'person_id': [1, 2, 3], 'home_x': 0., 'home_y': 0.})
        commutes = pd.DataFrame({'person_id': [1, 2, 3], 'commute_trip_id': [2, 1, 1],
            'commute_home_distance': 10000., 'commute_x': 10000., 'commute_y': 0.})
        trips = pd.DataFrame([
            [1, 1, 'home_secondary', 'work', 6000., 7000.],
            [1, 2, 'work', 'education', 7000., 10000.],
            [1, 3, 'education', 'home', 10000., 9000.],
            [2, 1, 'education', 'home_secondary', 10000., 9000.],
            [3, 1, 'other', 'education', 8000., 10000.]],
            columns=['person_id','trip_id','origin_purpose','purpose','origin_x','destination_x'])
        trips['origin_y'] = 0.; trips['destination_y'] = 0.
        result = use_diary_home(commutes, trips, persons)
        self.assertEqual(result.commute_home_distance.tolist(), [4000., 1000., 10000.])
        self.assertEqual(result.commute_home_source.tolist(), ['diary_preceding_home','diary_following_home','household_fallback'])
        self.assertTrue(result.commute_household_home_distance.eq(10000.).all())
        self.assertTrue(persons.home_x.eq(0).all())


if __name__ == '__main__':
    unittest.main()
