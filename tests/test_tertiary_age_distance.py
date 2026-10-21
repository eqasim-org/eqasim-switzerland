"""Age preferences must improve supported choices without losing campus counts."""

import unittest
from unittest.mock import patch, sentinel

import numpy as np
import pandas as pd
from synpp.pipeline import ConfigurationContext, ExecuteContext

from data.od.tertiary_education_age import adjust_probabilities, fit_decay
from synthesis.population.spatial.primary.education import balancing
from synthesis.population.spatial.primary.education.balancing import build_balancing
from synthesis.population.spatial.primary.education.tertiary import assign_od_locations


class TertiaryAgeDistanceTest(unittest.TestCase):
    def test_stage_entry_point_uses_synpp_configuration_contract(self):
        # Exercise real Synpp config contexts: algorithm-only replay contexts
        # can accidentally accept defaults that ExecuteContext does not support.
        for settings, enabled in [({}, True), ({"education_tertiary_age_distance": False}, False),
                                  ({"education_tertiary_balance": False}, False)]:
            with self.subTest(settings=settings):
                configured = ConfigurationContext({"data_path": "/unused", **settings})
                balancing.configure(configured)
                names = [stage["descriptor"] for stage in configured.required_stages]
                self.assertEqual("data.od.tertiary_education_age" in names, enabled)
                context = ExecuteContext(configured.required_config, configured.required_stages,
                    {}, None, {}, None, {}, None, {}, {})
                stages = {name: object() for name in names}
                stages["data.od.tertiary_education_matrix"] = (sentinel.pdf, None)
                if enabled:
                    stages["data.od.tertiary_education_age"] = sentinel.age_model
                with patch.object(context, "stage", side_effect=stages.__getitem__) as stage, \
                        patch.object(balancing.pd, "read_csv", return_value=pd.DataFrame()), \
                        patch.object(balancing, "pooled_compatibility", return_value=sentinel.compatibility), \
                        patch.object(balancing, "build_balancing", return_value=sentinel.result) as build:
                    result = balancing.execute(context)
                    if settings.get("education_tertiary_balance") is False:
                        self.assertEqual(result["models"], {})
                        stage.assert_not_called()
                        build.assert_not_called()
                    else:
                        self.assertIs(result, sentinel.result)
                        self.assertIs(build.call_args.args[-1], sentinel.age_model if enabled else None)

    def test_fit_uses_departure_evidence_and_preserves_support(self):
        profiles = pd.DataFrame([[.5, .5, 0.]], index=[0], columns=[0, 1, 2])
        distances = np.array([[0., 100., 200.]])
        observations = pd.DataFrame({"age": [20] * 100, "home_zone_id": 0,
            "education_zone_id": [0] * 80 + [1] * 20, "weight": 1.,
            "education_origin_source": "school_departure"})
        fitted = fit_decay(profiles, distances, observations)
        # For this binary example the likelihood optimum has a closed form.
        self.assertAlmostEqual(fitted["decay_per_km"], np.log(4) / 100, places=6)
        fallback = observations.assign(education_origin_source="residence_fallback", education_zone_id=1)
        older = observations.assign(age=40, education_zone_id=1)
        self.assertAlmostEqual(fit_decay(profiles, distances, pd.concat([observations, fallback, older]))[
            "decay_per_km"], fitted["decay_per_km"])
        adjusted = adjust_probabilities(profiles.to_numpy(), distances, fitted["decay_per_km"])
        np.testing.assert_allclose(adjusted, [[.8, .2, 0.]], atol=1e-6)
        np.testing.assert_array_equal(adjusted > 0, profiles.to_numpy() > 0)
        np.testing.assert_allclose(adjust_probabilities(profiles.to_numpy(), distances, 0.), profiles)
        self.assertEqual(fit_decay(profiles, distances, fallback)["decay_per_km"], 0.)
        with self.assertRaises(ValueError):
            adjust_probabilities([0., 0.], [0., 100.], .01)

    def test_age_rows_survive_balancing_and_actual_assignment(self):
        persons = pd.DataFrame({"person_id": range(8), "age": [20, 20, 30, 30] * 2,
            "home_zone_id": [0] * 4 + [1] * 4, "is_student": 1, "study_level": "tertiary",
            "education_destination_status": "domestic", "home_x": [0.] * 4 + [100000.] * 4,
            "home_y": 0., "commute_home_distance": 0.})
        sites = pd.DataFrame({"enterprise_id": ["a", "b"], "zone_id": [0, 1],
            "zone_municipality_id": [0, 1], "x": [0., 100000.], "y": 0.,
            "number_employees_fte": 1., "number_employees": 1.})
        registry = {"establishments": pd.DataFrame({"enterprise_id": ["a", "b"],
            "campus_id": ["A", "B"], "institution_type": "UH"})}
        compatibility = registry["establishments"].assign(study_level="tertiary")
        zones = pd.DataFrame({"zone_id": [0, 1], "zone_level": "municipality"})
        targets = {"institution_type_targets": pd.DataFrame({"study_level": ["tertiary"],
            "institution_type": ["UH"], "enrolment_share": [1.]}),
            "campus_targets": pd.DataFrame({"study_level": "tertiary", "campus_id": ["A", "B"],
                "institution_type": "UH", "institution_enrolled_students": 50.,
                "bfs_level_enrolment": 100., "campus_share": 1.})}
        age_model = {"age_limit": 24, "decay_per_km": .02,
                     "zone_centres": {0: (0., 0.), 1: (100000., 0.)}}
        pdf = np.full((2, 2), .5)
        old = build_balancing(persons, None, sites, registry, compatibility, zones, pdf, targets)
        new = build_balancing(persons, None, sites, registry, compatibility, zones, pdf, targets, age_model)
        a, b = old["models"]["tertiary"], new["models"]["tertiary"]
        self.assertNotIn("age_limit", a)
        self.assertEqual(set(b["origins"]), {(0, 0), (0, 1), (1, 0), (1, 1)})
        np.testing.assert_array_equal(b["integer_flows"].sum(axis=1), [2, 2, 2, 2])
        np.testing.assert_array_equal(b["integer_flows"].sum(axis=0), a["integer_flows"].sum(axis=0))
        near = [i for i, (zone, _) in enumerate(b["alternatives"]) if zone == 0]
        self.assertGreater(b["probabilities"][b["origins"][(0, 0)], near].sum(), .5)
        locations = persons[["person_id"]].assign(x=0., y=0., destination_id="a")
        assigned = assign_od_locations(persons, locations, sites, compatibility, zones, pdf,
                                       None, 42, new)
        self.assertEqual(len(assigned), len(persons))
        self.assertTrue(assigned.person_id.is_unique)
        self.assertTrue(assigned.education_od_status.eq("balanced_tertiary_od").all())
        counts = assigned.groupby("education_od_calibration_group").size().reindex(b["groups"], fill_value=0)
        np.testing.assert_array_equal(counts, b["integer_flows"].sum(axis=0))


if __name__ == "__main__":
    unittest.main()
