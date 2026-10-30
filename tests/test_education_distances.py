"""Check weighting, geography and activity scope of the distance comparison."""

import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box, Point

from analysis.education_distances import build_observations, weighted_distribution


class EducationDistanceTest(unittest.TestCase):
    def test_weighted_cdf_excludes_missing_codes(self):
        x, cdf = weighted_distribution(np.array([-8., 1., 10., 10.]), np.array([100., 1., 1., 2.]))
        np.testing.assert_array_equal(x, [1., 10.])
        np.testing.assert_allclose(cdf, [.25, 1.])

    def test_sources_have_one_person_distance_and_common_geography(self):
        municipalities = gpd.GeoDataFrame({"municipality_id": [1, 2]},
            geometry=[box(0, 0, 1000, 1000), box(2000, 0, 3000, 1000)], crs="EPSG:2056")
        mz = pd.DataFrame({"person_id": [1], "age": [20], "workday": [True], "weekend": [False],
            "day": ["Monday"], "home_x": [500.], "home_y": [500.], "person_weight": [2.],
            "employment_status": [2], "date": pd.to_datetime(["2015-03-02"])})
        commute = pd.DataFrame({"person_id": [1], "commute_x": [2500.], "commute_y": [500.],
                                "commute_home_distance": [2000.]})
        survey = pd.DataFrame({"age": [20, 20], "is_student": [1, 1], "weight": [3., 4.],
            "education_country_code": [-8, 8207], "home_municipality_id": [1, 1],
            "education_departure_municipality_id": [1, 1], "education_departure_status": ["domestic", "domestic"],
            "education_municipality_id": [2, 2], "crowfly_distance_to_school": [2000., 2000.],
            "survey_year": [2023, 2023]})
        persons = pd.DataFrame({"person_id": [100], "age": [20], "is_student": [1],
                                "home_x": [500.], "home_y": [500.]})
        locations = gpd.GeoDataFrame({"person_id": [100]}, geometry=[Point(2500, 500)], crs="EPSG:2056")
        # Two education activities must still contribute only one person.
        activities = pd.DataFrame({"person_id": [100, 100], "purpose": ["education", "education"]})
        result, coverage = build_observations(mz, commute, survey, persons, locations, activities,
                                              municipalities, ss_unit="m")
        self.assertEqual(len(result), 3)
        np.testing.assert_allclose(result.distance_native_km, [2., 2., 2.])
        np.testing.assert_allclose(result.distance_municipality_km.astype(float), [2., 2., 2.])
        self.assertEqual(result.source.nunique(), 3)
        self.assertEqual(coverage.loc[coverage.source.eq("Structural survey"), "domestic_endpoints"].iloc[0], 1)
        self.assertTrue(result.age_group.eq("18–23").all())
        # A term-time origin in the school municipality must not inherit the
        # registered-residence municipality's long centroid commute.
        survey.loc[0, "education_departure_municipality_id"] = 2
        revised, _ = build_observations(mz, commute, survey, persons, locations, activities,
                                        municipalities, ss_unit="m")
        ss = revised.loc[revised.source.eq("Structural survey")].iloc[0]
        self.assertEqual(ss.distance_municipality_km, 0.)
        self.assertEqual(ss.distance_registered_residence_municipality_km, 2.)
        commute["commute_home_x"], commute["commute_home_y"] = 2500., 500.
        commute["commute_home_distance"] = 0.
        revised, _ = build_observations(mz, commute, survey, persons, locations, activities,
                                        municipalities, ss_unit="m")
        observed = revised.loc[revised.source.eq("MZ")].iloc[0]
        self.assertEqual(observed.distance_native_km, 0.)
        self.assertEqual(observed.distance_municipality_km, 0.)
        self.assertEqual(observed.distance_registered_residence_municipality_km, 2.)


if __name__ == "__main__":
    unittest.main()
