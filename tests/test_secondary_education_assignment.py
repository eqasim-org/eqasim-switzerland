"""Check that the audit detects inaccessible donor distances without duplicating activities."""

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from analysis.secondary_education_assignment import audit


class SecondaryAssignmentAuditTest(unittest.TestCase):
    def test_nearest_seven_restriction_and_repeated_activities(self):
        persons = pd.DataFrame({"person_id": [1], "age": [16], "is_student": [1],
            "mz_person_id": [101], "current_education": [6], "home_x": [2600000.], "home_y": [1200000.],
            "education_destination_status": ["domestic"]})
        activities = pd.DataFrame({"person_id": [1, 1], "purpose": ["education", "education"]})
        locations = gpd.GeoDataFrame({"person_id": [1], "destination_id": ["school0"],
            "education_assignment_group": ["classified_secondary"], "education_assignment_status": ["classified_domestic"]},
            geometry=[Point(2601000., 1200000.)], crs=2056)
        commute = pd.DataFrame({"person_id": [101], "commute_home_distance": [10000.]})
        schools = pd.DataFrame({"enterprise_id": ["school%d" % i for i in range(100)],
            "x": [2600000. + (i + 1) * 1000 for i in range(100)], "y": 1200000.,
            "education_type": "secondary", "noga": "853200", "number_employees": 10})
        with tempfile.TemporaryDirectory() as directory:
            summary = audit(persons, activities, locations, commute, schools, directory)
            result = summary.loc[summary["group"].eq("secondary_students")].iloc[0]
            self.assertEqual(result["persons"], 1)
            self.assertEqual(result["donor_mean_km"], 10)
            self.assertEqual(result["assigned_mean_km"], 1)
            self.assertEqual(result["donor_beyond_seventh_school"], 1)
            self.assertEqual(result["nearest_seven_mean_best_error_km"], 3)
            self.assertEqual(result["nearest_100_mean_best_error_km"], 0)
            self.assertEqual(result["all_seven_distance_weights_at_floor"], 1)
            self.assertEqual(len(pd.read_csv(Path(directory) / "person_diagnostics.csv")), 1)


if __name__ == "__main__":
    unittest.main()
