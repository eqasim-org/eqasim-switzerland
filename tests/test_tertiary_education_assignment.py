"""Assignment invariants that matter when campus calibration becomes integral."""

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from data.statent.education_enrolment import validate as validate_enrolment

from synthesis.population.spatial.primary.education.balancing import fit_flows, integerise_flows
from synthesis.population.spatial.primary.education.tertiary import (
    choose_campus_establishments, sample_balanced_alternatives)


class CampusAssignmentTest(unittest.TestCase):
    def test_link_cache_token_survives_json_and_detects_content_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            mapping = path / "links.csv"
            mapping.write_text("first mapping")
            (path / "snapshot.json").write_text("{}")
            settings = {"education_enrolment_path":str(path), "education_institution_links":str(mapping)}
            class Context:
                def config(self, name):
                    return settings[name]
            token = validate_enrolment(Context())
            self.assertEqual(token, json.loads(json.dumps(token)))
            self.assertEqual(token, validate_enrolment(Context()))
            mapping.write_text("changed mapping at the same path")
            self.assertNotEqual(token, validate_enrolment(Context()))

    def test_rounding_preserves_origins_and_structural_zeros(self):
        flows = np.array([[.4, .6, 0.], [0., .2, .8], [0., 0., 2.]])
        rounded = integerise_flows(flows, [1, 1, 2])
        np.testing.assert_array_equal(rounded.sum(axis=1), [1, 1, 2])
        self.assertFalse(((flows == 0) & (rounded > 0)).any())
        self.assertTrue((np.abs(rounded.sum(axis=0) - flows.sum(axis=0)) < 1).all())

    def test_unreachable_target_does_not_drop_an_origin(self):
        # The second origin can only reach B, even when B's requested quota is 0.
        flows, diagnostics = fit_flows(np.eye(2), [1, 1], [2, 0], max_iterations=5)
        self.assertFalse(diagnostics["converged"])
        np.testing.assert_array_equal(integerise_flows(flows, [1, 1]).sum(axis=1), [1, 1])

    def test_alternative_sampling_obeys_integer_campus_counts(self):
        model = {"origins": {12: 0}, "integer_flows": np.array([[3, 1]]),
                 "groups": ["A", "B"], "alternatives": [(20, "A"), (21, "A"), (22, "B")],
                 "probabilities": np.array([[.2, .4, .4]])}
        selected = sample_balanced_alternatives(model, 12, 4, np.random.RandomState(0))
        self.assertEqual(sum(model["alternatives"][i][1] == "A" for i in selected), 3)
        self.assertEqual(sum(model["alternatives"][i][1] == "B" for i in selected), 1)
        with self.assertRaises(ValueError):
            sample_balanced_alternatives(model, 12, 5, np.random.RandomState(0))

    def test_many_nearby_buildings_cannot_exclude_another_campus(self):
        # Campus B lies beyond 30 closer A buildings. Aggregate FTE is equal, so
        # B must remain selectable before the within-campus nearest-20 search.
        candidates = pd.DataFrame({
            "enterprise_id": [str(i) for i in range(31)],
            "campus_id": ["A"] * 30 + ["B"], "x": [0.] * 30 + [10000.],
            "y": [0.] * 31, "number_employees": [1.] * 30 + [30.],
            "number_employees_fte": [1.] * 30 + [30.]}).set_index("enterprise_id", drop=False)
        people = pd.DataFrame({"home_x": np.zeros(100), "home_y": np.zeros(100),
                               "commute_home_distance": np.ones(100) * 1000})
        picked = choose_campus_establishments(people, candidates, np.random.RandomState(0))
        self.assertIn("30", picked)
        self.assertEqual(len(picked), len(people))
        only_b = choose_campus_establishments(people, candidates[candidates.campus_id.eq("B")],
                                             np.random.RandomState(0))
        self.assertTrue((only_b == "30").all())


if __name__ == "__main__":
    unittest.main()
