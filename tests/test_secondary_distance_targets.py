"""Check weighted targets, sparse-cell pooling and programme isolation."""

import unittest

import numpy as np
import pandas as pd

from synthesis.population.spatial.primary.education.secondary_distances import fit_distance_model, assign_distance_targets


def survey_fixture():
    return pd.DataFrame({"age": [16, 16, 20, 20, 20], "current_education": [6, 6, 6, 6, 6],
        "is_student": 1, "weight": [1., 9., 1., 100., 100.],
        "crowfly_distance_to_school": [1., 10., 20., -8., 1000.],
        "education_country_code": [-8, -8, -8, -8, 8207],
        "education_departure_status": "domestic", "education_zone_id": 1,
        "education_zone_level": "municipality"})


class SecondaryDistanceTargetsTest(unittest.TestCase):
    def test_weighted_quantiles_and_donor_rank(self):
        model = fit_distance_model(survey_fixture(), minimum_effective_n=1)
        people = pd.DataFrame({"person_id": range(10), "age": 16, "current_education": 6,
                               "commute_home_distance": np.arange(10) * 100000.})
        original = people.copy()
        result = assign_distance_targets(people, model, 0)
        np.testing.assert_array_equal(result.education_secondary_target_distance_m, [1000.] + [10000.] * 9)
        self.assertTrue(result.education_secondary_distance_source.eq("structural_survey_age_programme").all())
        pd.testing.assert_frame_equal(people, original)
        pd.testing.assert_frame_equal(result, assign_distance_targets(people, model, 0))

    def test_effective_sample_size_pools_sparse_age_cell(self):
        model = fit_distance_model(survey_fixture(), minimum_effective_n=2)
        people = pd.DataFrame({"person_id": [1], "age": [20], "current_education": [6],
                               "commute_home_distance": [150000.]})
        result = assign_distance_targets(people, model, 0)
        self.assertEqual(result.education_secondary_distance_source.iloc[0], "structural_survey_programme_low_support")
        # Pooled weights 1,9,1 select 10 km at the midpoint, rather than the
        # lone 20-year-old observation's 20 km or its large expansion weight.
        self.assertEqual(result.education_secondary_target_distance_m.iloc[0], 10000.)
        self.assertLess(result.education_secondary_distance_effective_n.iloc[0], 2)

    def test_no_cross_programme_fallback_and_missing_donors(self):
        model = fit_distance_model(survey_fixture(), minimum_effective_n=1)
        people = pd.DataFrame({"person_id": [1, 2, 3], "age": 16, "current_education": [6, 6, 7],
                               "commute_home_distance": [np.nan, -8., 500000.]})
        result = assign_distance_targets(people, model, 0)
        self.assertTrue(result.education_secondary_target_distance_m.iloc[:2].notna().all())
        self.assertTrue(result.education_secondary_target_distance_m.iloc[:2].gt(0).all())
        self.assertTrue(np.isnan(result.education_secondary_target_distance_m.iloc[2]))
        self.assertEqual(result.education_secondary_distance_source.iloc[2], "unsupported_programme")


if __name__ == "__main__":
    unittest.main()
