"""Doctoral programme scopes and occupational evidence must stay distinct."""
import unittest
from itertools import product

import pandas as pd

from analysis.education_doctoral_scope import doctoral_field_totals,doctoral_evidence


class DoctoralScopeTest(unittest.TestCase):
    def cube(self, order):
        labels={'Jahr':{'a':'2023/24'},'Hochschule':{'u':'UH'},'Studienstufe':{'d':'Doktorat'},
                'Fachbereich':{'med':'5.1 Humanmedizin','pharm':'5.4 Pharmazie','science':'4.2 Naturwissenschaften'},
                'Bildungsherkunft':{'ch':'Schweizer und Bildungsinländer','foreign':'Bildungsausländer'}}
        categories={c:list(labels[c]) for c in order}
        values=[]
        for combination in product(*(categories[c] for c in order)):
            row=dict(zip(order,combination));values.append({'med':10,'pharm':2,'science':3}[row['Fachbereich']])
        return {'id':order,'size':[len(categories[c]) for c in order],'value':values,
                'dimension':{c:{'category':{'index':{k:i for i,k in enumerate(categories[c])},'label':labels[c]}} for c in order}}

    def test_pharmacy_retained_and_educational_origins_summed_once(self):
        order=['Jahr','Fachbereich','Studienstufe','Bildungsherkunft','Hochschule']
        result=doctoral_field_totals(self.cube(order)).set_index('field_scope')
        self.assertEqual(result.loc['medical_fields_excluded_in_employment_publication','students'],20)
        self.assertEqual(result.loc['other_fields_including_pharmacy','students'],10)
        reordered=doctoral_field_totals(self.cube(list(reversed(order))))
        pd.testing.assert_frame_equal(result.reset_index().sort_index(axis=1),reordered.set_index('field_scope').reset_index().sort_index(axis=1))

    def test_missing_bfs_cell_does_not_become_a_complete_total(self):
        cube=self.cube(['Jahr','Fachbereich','Studienstufe','Bildungsherkunft','Hochschule'])
        cube['value'][0]=None
        result=doctoral_field_totals(cube).set_index('field_scope')
        self.assertTrue(pd.isna(result.loc['medical_fields_excluded_in_employment_publication','students']))
        self.assertEqual(result.loc['medical_fields_excluded_in_employment_publication','missing_cells'],1)

    def test_healthcare_evidence_never_becomes_a_study_field_or_student_override(self):
        people=pd.DataFrame({'current_education':[13]*3,'is_student':[1,1,0],'in_education':[1,1,0],
            'age':[30]*3,'weight':[1.]*3,'survey_year':[2023]*3,'employed':[1,3,1],
            'isco_code':[2212,-8,2131],'work_noga':['861001','-8','854201'],
            'work_enterprise_noga':['861001','-8','854201'],'education_noga':['854201','-9','-9'],
            'education_enterprise_noga':['854201','-9','-9'],'highest_completed_education':[12,12,13],
            'education_country_code':[8100,-9,-9],'education_municipality':[1,-9,-9]})
        result=doctoral_evidence(people)
        self.assertEqual(result.is_student.tolist(),[1,1,0])
        self.assertEqual(result.evidence_group.tolist(),['clinical_occupation_evidence','no_specific_medical_evidence','no_specific_medical_evidence'])
        self.assertTrue(result.doctoral_study_field.eq('unavailable_in_survey_extract').all())
        self.assertEqual(result.education_destination_status.tolist(),['domestic','unknown','unknown'])
        self.assertNotIn('evidence_group',people)


if __name__=='__main__':unittest.main()
