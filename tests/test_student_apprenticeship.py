"""Exercise apprentice conditioning through the actual student stage boundary."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from synthesis.population.models import students


class StudentApprenticeshipTest(unittest.TestCase):
    def run_stage(self, use_apprentice_probability):
        source = pd.DataFrame({
            "person_id": range(12), "age": [20, 20, 14, 80] * 3,
            "sex": 1, "home_municipality_id": 1, "district_id": 1,
            "canton_id": np.repeat([1, 22, 25], 4), "municipality_type": 1,
            "employed": [1, 1, 3, 1] * 3, "job_position": [50, 43, 70, 50] * 3,
            "non_private_home_only_households": [False, False, False, True] * 3,
        })
        survey = source.loc[source.age.ge(15)].copy()
        survey["is_student"] = survey.job_position.eq(50).astype(int)
        survey["survey_year"] = 2023
        survey["weight"] = np.arange(1, len(survey) + 1, dtype=float)
        original = source.copy(deep=True)
        original_survey = survey.copy(deep=True)
        seen = {}

        class Classifier:
            classes_ = np.array([0, 1])

            def fit(self, features, labels, sample_weight):
                seen["training"] = features.copy()
                seen["labels"] = labels.copy()
                seen["weights"] = sample_weight.copy()

            def predict_proba(self, features):
                seen["prediction"] = features.copy()
                probability = features.is_apprentice.to_numpy() if use_apprentice_probability else np.zeros(len(features))
                return np.column_stack([1 - probability, probability])

        with tempfile.TemporaryDirectory() as directory:
            class Context:
                def stage(self, name):
                    return survey if name == "data.structural_survey.student_training" else source

                def path(self):
                    return directory

            with patch.object(students, "CatBoostClassifier", return_value=Classifier()):
                result = students.execute(Context())
            self.assertTrue((Path(directory) / "national_apprentice_stock.csv").exists())

        pd.testing.assert_frame_equal(source, original)
        pd.testing.assert_frame_equal(survey, original_survey)
        pd.testing.assert_frame_equal(result[["person_id", "age", "employed", "job_position"]],
                                      original[["person_id", "age", "employed", "job_position"]])
        np.testing.assert_array_equal(seen["training"].is_apprentice, survey.job_position.eq(50))
        np.testing.assert_array_equal(seen["weights"], survey.weight)
        np.testing.assert_array_equal(seen["labels"], survey.is_student)
        active = source.loc[~source.non_private_home_only_households]
        np.testing.assert_array_equal(seen["prediction"].is_apprentice, active.job_position.eq(50))
        self.assertNotIn("is_apprentice", result)
        self.assertTrue(result.loc[source.age.eq(80), "is_student"].eq(0).all())
        self.assertTrue(result.loc[source.age.eq(80), "student_probability"].eq(0).all())
        self.assertTrue(result.loc[source.age.lt(15), "is_student"].eq(1).all())
        return result

    def test_feature_conditions_participation_and_preserves_bypass(self):
        result = self.run_stage(True)
        workers = result.loc[result.age.eq(20)]
        self.assertTrue(workers.loc[workers.job_position.eq(50), "is_student"].eq(1).all())
        self.assertTrue(workers.loc[workers.job_position.eq(43), "is_student"].eq(0).all())

    def test_model_prediction_is_not_replaced_with_a_hard_apprentice_rule(self):
        result = self.run_stage(False)
        self.assertTrue(result.loc[result.age.ge(15), "is_student"].eq(0).all())


if __name__ == "__main__":
    unittest.main()
