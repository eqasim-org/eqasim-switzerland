"""Nonstudent visits retain their diaries and become variable secondary stops."""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from shapely.geometry import Point

from data.microcensus.secondary_trips import recode_nonstudent_education, execute as reference_trips
from synthesis.population.trips import execute as synthetic_trips, BORDER_MATCH_COLUMNS
from synthesis.population.activities import execute as synthetic_activities
from synthesis.population.spatial.secondary.problems import find_assignment_problems
from synthesis.population.spatial.secondary.distance_distributions import execute as distance_reference


class Context:
    def __init__(self, stages):
        self.stages = stages

    def stage(self, name):
        return self.stages[name]

    def config(self, name):
        return {"random_seed": 0}[name]


class NonstudentEducationRecodingTest(unittest.TestCase):
    def test_both_endpoints_variants_and_input_preservation(self):
        source = pd.DataFrame({"person_id": [1, 1, 2, 3], "trip_id": [1, 2, 1, 1],
            "purpose": pd.Categorical(["education", "home", "education", "education_secondary"]),
            "origin_purpose": ["home", "education_main", "home", "education_secondary"],
            "mode": ["walk"] * 4, "departure_time": [100., 200., 300., 400.]})
        before = source.copy(deep=True)
        result = recode_nonstudent_education(source, [1, 3])
        self.assertEqual(result.purpose.tolist(), ["other", "home", "education", "other"])
        self.assertEqual(result.origin_purpose.tolist(), ["home", "other", "home", "other"])
        self.assertEqual(result.original_purpose.tolist(), before.purpose.tolist())
        self.assertEqual(result.original_origin_purpose.tolist(), before.origin_purpose.tolist())
        pd.testing.assert_frame_equal(source, before)
        pd.testing.assert_frame_equal(result[source.columns.difference(["purpose", "origin_purpose"])],
                                      before[source.columns.difference(["purpose", "origin_purpose"])])
        pd.testing.assert_frame_equal(result, recode_nonstudent_education(result, [1, 3]))

    def test_mz_reference_retains_students_employed_students_and_unknown(self):
        persons = pd.DataFrame({"person_id": [1, 2, 3, 4, 5],
                                "employment_status": pd.Series([0, 1, 2, 3, pd.NA], dtype="Int64")})
        source = pd.DataFrame({"person_id": [1, 2, 3, 4, 5], "purpose": ["education"] * 5,
                                "origin_purpose": ["home"] * 5})
        result = reference_trips(Context({"data.microcensus.persons": persons,
                                         "data.microcensus.trips": (source, set(), [], [])}))
        self.assertEqual(result.purpose.tolist(), ["other", "other", "education", "education", "education"])
        self.assertTrue(source.purpose.eq("education").all())

    def make_trip_context(self, student_flags=(0, 1), border=False):
        people = pd.DataFrame({"person_id": [1, 2], "mz_person_id": [101, 101], "age": [25, 25],
            "is_student": list(student_flags), "is_truck_driver": False, "is_outside_of_switzerland": False,
            "is_crossing_the_border": [False, border], "canton_id": 1})
        for column in BORDER_MATCH_COLUMNS:
            people[column] = None
        if border:
            people.loc[1, "cross_border_person_id"] = 901
            people.loc[1, "border_crossing_trip_mode"] = "car"
        trips = pd.DataFrame({"person_id": [101, 101], "trip_id": [1, 2],
            "departure_time": [3600., 7200.], "arrival_time": [4200., 7800.], "mode": ["walk", "walk"],
            "origin_purpose": ["home", "education"], "purpose": ["education", "home"]})
        return Context({"data.constants": SimpleNamespace(census="statpop", MZ_AGE_THRESHOLD=6),
            "synthesis.population.enriched": people, "data.microcensus.trips": (trips, set(), [], []),
            "synthesis.population.spatial.primary.work.work_remotly": pd.DataFrame({"person_id": []})})

    def test_recipient_status_drives_recode_without_changing_diary_times(self):
        context = self.make_trip_context()
        before = context.stage("data.microcensus.trips")[0].copy()
        trips = synthetic_trips(context)
        baseline = synthetic_trips(self.make_trip_context(student_flags=(1, 1)))
        fixed = ["person_id", "mz_person_id", "trip_id", "trip_index", "departure_time", "arrival_time", "trip_duration", "mode"]
        pd.testing.assert_frame_equal(trips[fixed], baseline[fixed])
        pd.testing.assert_frame_equal(context.stage("data.microcensus.trips")[0], before)
        self.assertEqual(trips.loc[trips.person_id.eq(1), "following_purpose"].tolist(), ["other", "home"])
        self.assertEqual(trips.loc[trips.person_id.eq(2), "following_purpose"].tolist(), ["education", "home"])
        context.stages["synthesis.population.trips"] = trips
        activities = synthetic_activities(context)
        stop = activities.loc[activities.person_id.eq(1) & activities.activity_index.eq(1)].iloc[0]
        self.assertEqual(stop.purpose, "other")
        self.assertEqual(stop.original_purpose, "education")
        self.assertTrue(stop.nonstudent_education_recoded)
        self.assertEqual(len(activities), 6)
        self.assertFalse(activities.loc[activities.person_id.eq(2), "nonstudent_education_recoded"].any())
        # The recoded stop has home anchors; no education location is needed.
        primary = pd.DataFrame({"person_id": [1], "home": [Point(0, 0)], "work": [None], "education": [None]})
        # RDA derives travel_time from the preserved diary times before splitting.
        secondary_trips = trips[trips.person_id.eq(1)].assign(travel_time=lambda f: f.trip_duration)
        problems = list(find_assignment_problems(secondary_trips, primary))
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0]["purposes"], ["other"])
        np.testing.assert_array_equal(problems[0]["origin"], [[0., 0.]])
        np.testing.assert_array_equal(problems[0]["destination"], [[0., 0.]])

    def test_initial_and_final_reported_education_are_preserved(self):
        context = self.make_trip_context()
        raw = context.stage("data.microcensus.trips")[0]
        raw.loc[0, "origin_purpose"] = "education"
        raw.loc[1, "purpose"] = "education"
        context.stages["synthesis.population.trips"] = synthetic_trips(context)
        activities = synthetic_activities(context)
        nonstudent = activities[activities.person_id.eq(1)]
        self.assertEqual(nonstudent.purpose.tolist(), ["other", "other", "other"])
        self.assertTrue(nonstudent.original_purpose.eq("education").all())
        self.assertTrue(nonstudent.nonstudent_education_recoded.all())

    def test_new_border_trips_do_not_inherit_education_provenance(self):
        context = self.make_trip_context(border=True)
        trips = synthetic_trips(context)
        border = trips[trips.person_id.eq(2)]
        self.assertEqual(len(border), 1)
        self.assertEqual(border.original_preceding_purpose.iloc[0], border.preceding_purpose.iloc[0])
        self.assertEqual(border.original_following_purpose.iloc[0], border.following_purpose.iloc[0])

    def test_rda_reference_includes_recoded_education_trips(self):
        persons = pd.DataFrame({"person_id": [1, 2], "employment_status": [0, 2],
                                "person_weight": [1., 1.], "weekend": [False, False]})
        raw = pd.DataFrame({"person_id": [1, 1, 2, 2], "trip_id": [1, 2, 1, 2],
            "purpose": ["education", "home", "education", "home"],
            "origin_purpose": ["home", "education", "home", "education"],
            "mode": ["car"] * 4, "departure_time": [0., 1000., 0., 1000.],
            "arrival_time": [600., 1600., 600., 1600.], "crowfly_distance": [1000., 2000., 9000., 9000.]})
        context = Context({"data.microcensus.persons": persons, "data.microcensus.trips": (raw, set(), [], [])})
        context.stages["data.microcensus.secondary_trips"] = reference_trips(context)
        result = distance_reference(context)
        np.testing.assert_array_equal(result["car"]["distributions"][0]["values"], [1000., 2000.])
        self.assertEqual(len(raw), len(context.stage("data.microcensus.secondary_trips")))


if __name__ == "__main__":
    unittest.main()
