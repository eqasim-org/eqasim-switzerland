"""Preserve apprenticeship information through programme imputation and fallback."""

import unittest

import numpy as np
import pandas as pd

from synthesis.population.models.education import (
    OUTCOMES, TERTIARY_CODE, assign_education, fit_model, predict_probabilities,
    prepare_features,
)


class EducationApprenticeshipTest(unittest.TestCase):
    def survey(self):
        # Same age/employment/geography: only apprenticeship distinguishes these
        # programme distributions. Unequal weights also exercise survey weighting.
        return pd.DataFrame({
            'age': [22] * 8, 'employed': 1, 'sex': 0, 'canton_id': 1,
            'job_position': [50] * 4 + [43] * 4, 'is_student': 1,
            'weight': [1., 2., 1., 2.] * 2,
            'current_education': [6] * 4 + [11, 12, 10, 13],
            'education_country_code': 8100, 'education_municipality': 1,
        })

    def people(self):
        return pd.DataFrame({
            'age': [22, 22], 'employed': 1, 'sex': 0, 'canton_id': 1,
            'job_position': [50, 43], 'is_student': 1,
        })

    def test_employed_apprentices_and_other_students_have_different_programmes(self):
        survey = self.survey()
        people = self.people()
        original_survey, original_people = survey.copy(deep=True), people.copy(deep=True)
        result = assign_education(people, survey, seed=0, prior_strength=0.)
        self.assertEqual(result.study_level.tolist(), ['vocational_secondary', 'tertiary'])
        self.assertEqual(result.loc[0, 'current_education'], 6)
        self.assertTrue(pd.isna(result.loc[1, 'current_education']))
        pd.testing.assert_frame_equal(result[people.columns], people)
        pd.testing.assert_frame_equal(survey, original_survey)
        pd.testing.assert_frame_equal(people, original_people)

    def test_default_smoothing_keeps_probabilities_instead_of_a_hard_rule(self):
        probabilities, _, _ = predict_probabilities(fit_model(self.survey()), self.people())
        vocational = OUTCOMES.index((6, 'domestic'))
        tertiary = OUTCOMES.index((TERTIARY_CODE, 'domestic'))
        self.assertGreater(probabilities[0, vocational], probabilities[1, vocational])
        self.assertGreater(probabilities[1, tertiary], probabilities[0, tertiary])
        self.assertGreater(probabilities[0, tertiary], 0.)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.)

    def test_unseen_canton_retains_apprentice_profile_but_unknown_job_backs_off(self):
        people = pd.concat([self.people().iloc[[0]]] * 2, ignore_index=True)
        people['canton_id'] = 25  # No Geneva observations in this fixture.
        people['job_position'] = pd.Series(['50', pd.NA], dtype='string')
        probabilities, methods, _ = predict_probabilities(
            fit_model(self.survey(), prior_strength=0.), people)
        vocational = OUTCOMES.index((6, 'domestic'))
        self.assertEqual(probabilities[0, vocational], 1.)
        self.assertAlmostEqual(probabilities[1, vocational], .5)
        self.assertEqual(methods.tolist(), [
            'age_band_employed_is_apprentice_sex_smoothed', 'age_band_employed_smoothed'])
        self.assertEqual(prepare_features(people).is_apprentice.tolist(), [1, -1])

    def test_assignment_preserves_bypasses_and_is_independent_of_chunk_size(self):
        people = pd.concat([self.people()] * 20, ignore_index=True)
        people.loc[0, 'age'] = 12
        people.loc[1, 'is_student'] = 0
        people.loc[1, 'age'] = 80
        survey = self.survey()
        small = assign_education(people, survey, seed=37, chunk_size=1)
        large = assign_education(people, survey, seed=37, chunk_size=100)
        pd.testing.assert_frame_equal(small, large)
        pd.testing.assert_frame_equal(small[people.columns], people)
        self.assertEqual(small.loc[0, 'study_level'], 'below_survey_age_coverage')
        self.assertEqual(small.loc[1, 'study_level'], 'none')
        self.assertNotIn('is_apprentice', small.columns)


if __name__ == '__main__':
    unittest.main()
