"""Keep source counts intact while aligning diploma campus/type references."""

import unittest

import numpy as np
import pandas as pd

from data.statent.education_enrolment import link_institutions, model_enrolment_reference, pooled_enrolment_reference
from synthesis.population.spatial.primary.education.targets import build_target_preview


def raw_reference():
    rows = []
    for kind, code, levels in [
        ("UH", "U", [("bachelor", 100.), ("master", 40.), ("legacy_degree", 10.),
                     ("doctorate_habilitation", 20.), ("continuing_education_and_other", 30.)]),
        ("FH_PH", "F", [("bachelor", 80.), ("legacy_degree", 20.), ("master", 10.)])]:
        for level, count in levels:
            rows.append(dict(academic_year="2023/24", institution_type=kind, bfs_institution_code=code,
                bfs_institution_label=code, aggregate_provider=False, study_level=level, students=count,
                bfs_table_id=kind, source_url="source", source_cells=1, missing_source_cells=0,
                bfs_study_level_code=level, bfs_study_level_label=level,
                model_study_level=level in ["bachelor", "master", "doctorate_habilitation"]))
    return pd.DataFrame(rows)


class EnrolmentReferenceTest(unittest.TestCase):
    def test_bridge_aggregates_once_and_preserves_raw_source(self):
        raw = raw_reference()
        original = raw.copy(deep=True)
        result = model_enrolment_reference(raw)
        pd.testing.assert_frame_equal(raw, original)
        totals = result.groupby("study_level").students.sum()
        self.assertEqual(totals.to_dict(), {"bachelor": 200., "master": 60., "doctorate_habilitation": 20.})
        self.assertEqual(result.students.sum(), raw.loc[raw.study_level.ne("continuing_education_and_other"), "students"].sum())
        self.assertFalse(result.duplicated(["academic_year", "institution_type", "bfs_institution_code", "study_level"]).any())
        uh_master = result.loc[result.institution_type.eq("UH") & result.study_level.eq("master")].iloc[0]
        self.assertEqual(uh_master.strict_degree_students, 40.)
        self.assertEqual(uh_master.legacy_degree_students, 10.)
        self.assertIn("legacy_degree", uh_master.bfs_source_study_level_labels)

    def test_unknown_legacy_propagates_and_duplicate_raw_totals_are_rejected(self):
        raw = raw_reference()
        mask = raw.institution_type.eq("FH_PH") & raw.study_level.eq("legacy_degree")
        raw.loc[mask, "students"] = np.nan
        raw.loc[mask, "missing_source_cells"] = 1
        result = model_enrolment_reference(raw)
        bachelor = result.loc[result.institution_type.eq("FH_PH") & result.study_level.eq("bachelor")].iloc[0]
        self.assertTrue(pd.isna(bachelor.students))
        self.assertEqual(bachelor.missing_source_cells, 1)
        with self.assertRaises(ValueError):
            model_enrolment_reference(pd.concat([raw, raw.iloc[:1]]))

    def test_linked_and_national_targets_use_same_aligned_totals(self):
        raw = raw_reference()
        institutions = pd.DataFrame({"statent_enterprise_id": ["SU", "SF"],
                                     "institution_types": ["UH", "FH_PH"], "institution_id": ["U", "F"]})
        links = pd.DataFrame({"institution_type": ["UH", "FH_PH"], "bfs_institution_code": ["U", "F"],
            "statent_enterprise_id": ["SU", "SF"], "match_status": ["proposed", "proposed"],
            "institution_name": ["U", "F"], "evidence_url": ["source"] * 2, "evidence_note": ["review"] * 2})
        linked = link_institutions(raw, institutions, links, "2023/24")
        pd.testing.assert_frame_equal(raw, linked["enrolment"])
        self.assertTrue(linked["linked_model_enrolment"].verified_enrolled_students.isna().all())
        inst = linked["institution_totals"].set_index("bfs_institution_code")
        self.assertEqual(inst.loc["U", "enrolled_students"], 200.)
        self.assertEqual(inst.loc["U", "model_level_enrolled_students"], 170.)
        self.assertEqual(inst.loc["U", "strict_degree_enrolled_students"], 160.)
        persons = pd.DataFrame({"person_id": range(10), "age": [20] * 10, "is_student": [1] * 10,
                               "current_education": [11] * 10, "education_destination_status": ["domestic"] * 10})
        higher = pd.DataFrame({"academic_year":["2023/24"], "study_level":["HF"],
                               "students":[100.], "source_url":["source"]})
        reference = pooled_enrolment_reference(raw, higher)
        campuses = linked["linked_enrolment"].groupby(
            ["institution_id","institution_type","bfs_institution_code","academic_year"], as_index=False).students.sum()
        campuses["campus_id"] = campuses.institution_id
        campuses["institution_enrolled_students"] = campuses.students
        campuses["estimated_enrolled_students"] = campuses.students
        campuses["study_level"] = "tertiary"
        campuses["campus_share"] = 1.
        targets = build_target_preview(persons, reference, campuses, "2023/24")
        types = targets["institution_type_targets"].set_index("institution_type")
        self.assertAlmostEqual(types.loc['UH','preview_cohort_students'], 10 * 200 / 410)
        self.assertAlmostEqual(types.loc['FH_PH','preview_cohort_students'], 10 * 110 / 410)
        self.assertAlmostEqual(types.loc['higher_vocational','preview_cohort_students'], 10 * 100 / 410)
        self.assertAlmostEqual(types.preview_cohort_students.sum(), 10.)
        self.assertAlmostEqual(targets["campus_targets"].preview_cohort_students.sum(), 10 * 310 / 410)
        self.assertTrue(targets['campus_targets'].study_level.eq('tertiary').all())


if __name__ == "__main__":
    unittest.main()
