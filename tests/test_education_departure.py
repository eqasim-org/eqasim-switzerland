"""Departure origins must remain distinct from registered residence."""

import unittest

import numpy as np
import pandas as pd

from data.structural_survey.structural_survey import prepare_education_departure
from data.od.education_matrix import prepare_observations


class EducationDepartureTest(unittest.TestCase):
    def test_departure_mapping_and_domestic_od_exclusions(self):
        survey = pd.DataFrame({"home_municipality_id": [1.] * 4, "home_quarter_id": [np.nan] * 4,
            "home_zone_id": [10] * 4, "home_zone_level": ["municipality"] * 4,
            "education_departure_municipality": [2, 2, -8, 2],
            "education_departure_quarter": [20, -8, -8, 20],
            "education_departure_country_code": [8100, -8, -8, 8207],
            "education_municipality_id": [2.] * 4, "education_quarter_id": [20.] * 4,
            "education_zone_id": [30] * 4, "education_zone_level": ["quarter"] * 4,
            "mode": ["pt"] * 4, "weight": [1.] * 4, "education_country_code": [-8] * 4,
            "current_education": [11] * 4, "survey_year": [2023] * 4, "is_student":[1]*4, "age":[22]*4})
        mapping = pd.DataFrame({"deprecated_municipality_id": [1, 2], "municipality_id": [1, 2]})
        quarters = pd.DataFrame({"quarter_id": [20]})
        zones = pd.DataFrame({"zone_id": [10, 20, 30], "zone_level_id": [1, 2, 20],
                              "zone_level": ["municipality", "municipality", "quarter"]})
        result = prepare_education_departure(survey, mapping, quarters, zones)
        self.assertTrue(result.home_municipality_id.eq(1).all())
        self.assertEqual(result.education_departure_zone_id.iloc[0], 30)
        self.assertEqual(result.education_departure_zone_id.iloc[1], 20)
        self.assertEqual(result.education_departure_status.tolist(), ["domestic", "domestic", "unresolved", "abroad"])
        self.assertTrue(result.education_departure_differs_from_residence.iloc[:2].all())
        od = prepare_observations(result, tertiary_only=True)
        self.assertEqual(len(od), 3)
        self.assertEqual(od.home_municipality_id.tolist(), [2, 2, 1])
        self.assertEqual(od.home_zone_id.tolist(), [30, 20, 10])
        self.assertEqual(od.education_origin_source.tolist(),
                         ['school_departure', 'school_departure', 'residence_fallback'])
        # Preparing education flows cannot alter the work/home source geography.
        self.assertTrue(result.home_zone_id.eq(10).all())


if __name__ == "__main__":
    unittest.main()
