"""Protect source labels and national stock accounting in the partial SS repair."""

import unittest

import numpy as np
import pandas as pd

from data.structural_survey.apprenticeship import (
    apprentice_summary, model_apprentice_diagnostics, repair_vocational_apprentices,
)


class ApprenticeRepairTest(unittest.TestCase):
    def survey(self):
        # Include older vocational students and a large weight. The repair is
        # national across ages, and its quota concerns weighted people, not rows.
        pool = pd.DataFrame({
            "age": [16, 20, 28, 40] * 8,
            "survey_year": [2021] * 16 + [2023] * 16,
            "weight": [1., 3., 20., 100.] * 8,
            "is_student": 1, "current_education": 6,
            "employed": [2, 3] * 16, "job_position": [60, 70] * 16,
        })
        protected = pd.DataFrame({
            "age": [18, 21, 22, 23, 14], "survey_year": 2023,
            "weight": 7., "is_student": [1, 1, 0, 1, 1],
            "current_education": [6, 6, 6, 11, 6],
            "employed": [1, 1, 3, 3, 3], "job_position": [50, 43, 70, 70, 70],
        })
        # Pooled SS frames may have duplicate source indices.
        return pd.concat([pool, protected])

    def test_only_eligible_labels_change_and_student_stock_is_preserved(self):
        source = self.survey()
        before = source.copy(deep=True)
        result, audit = repair_vocational_apprentices(source, .65, seed=17)
        pd.testing.assert_frame_equal(source, before)
        pd.testing.assert_frame_equal(result.iloc[32:], source.iloc[32:])
        pd.testing.assert_frame_equal(result.drop(columns=["employed", "job_position"]),
                                      source.drop(columns=["employed", "job_position"]))
        changed = result.job_position.to_numpy() != source.job_position.to_numpy()
        self.assertTrue(changed.any())
        self.assertTrue(result.iloc[np.flatnonzero(changed)].employed.eq(1).all())
        self.assertTrue(result.iloc[np.flatnonzero(changed)].job_position.eq(50).all())
        self.assertTrue(result.index.equals(source.index))
        for row in audit.itertuples():
            maximum_weight = source.loc[source.survey_year.eq(row.survey_year), "weight"].max()
            self.assertLessEqual(abs(row.repaired_weight - row.requested_repair_weight), maximum_weight/2 + 1e-9)

    def test_zero_and_full_repair_keep_other_jobs(self):
        source = self.survey()
        zero, _ = repair_vocational_apprentices(source, 0)
        pd.testing.assert_frame_equal(zero, source)
        full, audit = repair_vocational_apprentices(source, 1)
        self.assertTrue(full.iloc[:32].employed.eq(1).all())
        self.assertTrue(full.iloc[:32].job_position.eq(50).all())
        pd.testing.assert_frame_equal(full.iloc[32:], source.iloc[32:])
        self.assertTrue(np.allclose(audit.repaired_weight, audit.eligible_weight))

    def test_reproducibility_and_independence_between_years(self):
        source = self.survey()
        first, audit = repair_vocational_apprentices(source, .65, seed=27)
        again, audit_again = repair_vocational_apprentices(source, .65, seed=27)
        pd.testing.assert_frame_equal(first, again)
        pd.testing.assert_frame_equal(audit, audit_again)
        last_year = source.loc[source.survey_year.eq(2023)]
        alone, _ = repair_vocational_apprentices(last_year, .65, seed=27)
        pd.testing.assert_frame_equal(first.loc[first.survey_year.eq(2023)], alone)

    def test_invalid_fraction_or_weight_is_rejected(self):
        for fraction in [-.1, 1.1, np.nan, np.inf]:
            with self.subTest(fraction=fraction), self.assertRaisesRegex(ValueError, "between 0 and 1"):
                repair_vocational_apprentices(self.survey(), fraction)
        for weight in [0., -1., np.nan, np.inf]:
            source = self.survey()
            source.iloc[0, source.columns.get_loc("weight")] = weight
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "positive survey weights"):
                repair_vocational_apprentices(source)

    def test_no_eligible_people_preserves_labels(self):
        source = self.survey().iloc[32:]
        result, audit = repair_vocational_apprentices(source)
        pd.testing.assert_frame_equal(result, source)
        self.assertTrue(audit.repaired_observations.eq(0).all())

    def test_diagnostics_separate_years_samples_and_nonstudent_apprentices(self):
        population = pd.DataFrame({
            "age": [14, 20, 30, 40], "employed": [3, 1, 1, 3],
            "job_position": [70, 50, 50, 70], "is_student": [1, 1, 0, 0],
            "is_synthetic_household": [False, False, True, True],
        })
        diagnostic = model_apprentice_diagnostics(self.survey(), population)
        ss = diagnostic.loc[diagnostic.source.eq("structural_survey")]
        self.assertEqual(ss.survey_year.tolist(), [2021, 2023])
        pop = diagnostic.loc[diagnostic.population_group.eq("all_modelled")].iloc[0]
        self.assertEqual(pop.count_unit, "sample_persons")
        self.assertEqual(pop.persons, 3.)
        self.assertEqual(pop.apprentices, 2.)
        self.assertEqual(pop.apprentice_students, 1.)
        self.assertEqual(pop.apprentice_nonstudents, 1.)
        groups = diagnostic.loc[diagnostic.population_group.isin(["private_households", "recovered"])]
        self.assertEqual(groups.apprentices.sum(), pop.apprentices)
        before_student = apprentice_summary(population.drop(columns="is_student"), include_students=False)
        self.assertTrue(np.isnan(before_student["students"]))
        self.assertEqual(before_student["apprentices"], pop.apprentices)


if __name__ == "__main__":
    unittest.main()
